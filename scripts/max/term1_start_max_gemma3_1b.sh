#!/usr/bin/env bash
set -Eeuo pipefail

# MonitorMe helper for starting the known-good Node1 MAX + Gemma 3 1B server.
# Keep this terminal running while another terminal validates MonitorMe.
#
# Important v0.2.2 behavior:
# - This script does NOT mutate or upgrade the pixi environment by default.
# - The earlier helper used `pixi add modular huggingface_hub openai` on every
#   start. On a rolling MAX/nightly channel this can upgrade a previously
#   working environment and break Mojo/MAX startup.
# - Set MONITORME_MAX_PIXI_SYNC=1 only when you intentionally want to run pixi
#   add/sync before starting MAX.
#
# Defaults are based on the validated Node1 RTX 5060 Ti path:
# - google/gemma-3-1b-it
# - GPU serving
# - Optional --sample-on-host workaround for older MAX nightly GPU fused top-k sampler crashes
#
# Override examples:
#   MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart" \
#   MONITORME_LLM_MODEL_ID="google/gemma-3-1b-it" \
#   ./scripts/max/term1_start_max_gemma3_1b.sh

PROJECT_DIR="${MONITORME_MAX_PROJECT_DIR:-$HOME/dev/modular/project/quickstart}"
MODEL_ID="${MONITORME_LLM_MODEL_ID:-google/gemma-3-1b-it}"
LOG_FILE="${MONITORME_MAX_LOG_FILE:-max_gemma3_1b.log}"
API_PORT="${MONITORME_MAX_API_PORT:-8000}"
METRICS_PORT="${MONITORME_MAX_METRICS_PORT:-8001}"
DEVICE_MEMORY_UTILIZATION="${MONITORME_MAX_DEVICE_MEMORY_UTILIZATION:-0.90}"
MAX_PIXI_SYNC="${MONITORME_MAX_PIXI_SYNC:-0}"
MAX_CLEAR_MOJO_CACHE="${MONITORME_MAX_CLEAR_MOJO_CACHE:-0}"
MAX_OVERRIDE_MOJO_IMPORT_PATH="${MONITORME_MAX_OVERRIDE_MOJO_IMPORT_PATH:-1}"
MAX_SKIP_HELP_PREFLIGHT="${MONITORME_MAX_SKIP_HELP_PREFLIGHT:-1}"
MAX_SAMPLE_ON_HOST="${MONITORME_MAX_SAMPLE_ON_HOST:-0}"
MAX_PREBUILD_MOJO_CACHES="${MONITORME_MAX_PREBUILD_MOJO_CACHES:-1}"
MAX_PATCH_PIXI_ACTIVATION_ENV="${MONITORME_MAX_PATCH_PIXI_ACTIVATION_ENV:-0}"
MAX_RUNTIME_MOJO_IMPORT_MODE="${MONITORME_MAX_RUNTIME_MOJO_IMPORT_MODE:-single-lib-mojo}"

cd "$PROJECT_DIR"

err() { echo "ERROR: $*" >&2; }
info() { echo "$*"; }

# Run a command inside pixi while explicitly preserving the Mojo package roots.
# Node1 logs showed that exporting MODULAR_MOJO_MAX_IMPORT_PATH in the parent
# shell is not always enough because pixi activation/dependency scripts can
# override outside environment variables. `pixi run env VAR=... command` applies
# these values after pixi activation and before Python/MAX starts.
pixi_env_run() {
  if [ -n "${MODULAR_MOJO_MAX_IMPORT_PATH:-}" ]; then
    pixi run env \
      "MODULAR_MOJO_MAX_IMPORT_PATH=$MODULAR_MOJO_MAX_IMPORT_PATH" \
      "MOJO_PACKAGE_PATH=${MOJO_PACKAGE_PATH:-$MODULAR_MOJO_MAX_IMPORT_PATH}" \
      "$@"
  else
    pixi run "$@"
  fi
}

discover_and_export_mojo_import_path() {
  local prefix="$PROJECT_DIR/.pixi/envs/default"
  local discovered=""
  local lib_mojo="$prefix/lib/mojo"

  if [ -d "$prefix" ]; then
    # Prefer real compiled Mojo package roots first. Node1 logs showed that
    # broad `find -name std -o -name nn` discovery can accidentally add
    # non-Mojo directories such as share/locale or share/tabset before lib/mojo.
    # Those paths can shadow the real std.mojoc/nn.mojoc package root and break
    # serving-time modules such as max/_distributed_ops/distributed_ops.mojo.
    if [ -d "$lib_mojo" ] && { [ -f "$lib_mojo/std.mojoc" ] || [ -f "$lib_mojo/nn.mojoc" ]; }; then
      discovered="$lib_mojo"
    fi

    while IFS= read -r dir; do
      [ -n "$dir" ] || continue
      case "$dir" in
        */share/locale|*/share/locale/*|*/share/tabset|*/share/tabset/*)
          continue
          ;;
      esac
      case ":$discovered:" in
        *":$dir:"*) ;;
        *) discovered="${discovered:+$discovered:}$dir" ;;
      esac
    done < <(
      {
        # Add directories containing compiled Mojo packages.
        find "$prefix" -type f \( -name 'std.mojoc' -o -name 'nn.mojoc' -o -name 'comm.mojoc' -o -name 'layout.mojoc' -o -name 'stdlib.mojopkg' -o -name 'std.mojopkg' -o -name 'nn.mojopkg' \) -printf '%h\n' 2>/dev/null || true
        # Add MAX package root as a fallback for MAX-owned .mojo package trees.
        find "$prefix" -type d -path '*/site-packages/max' -printf '%p\n' 2>/dev/null || true
      } | awk 'NF && !seen[$0]++'
    )
  fi

  if [ -n "$discovered" ]; then
    # v0.2.12: keep the runtime MAX/Mojo import path conservative.
    # Node1 logs proved that a broad colon-separated runtime value
    #   lib/mojo:site-packages/max
    # allowed Python cache prebuilds to pass, but the MAX graph worker later
    # failed kernel package registration with:
    #   failed to resolve built-in kernel package paths
    #   MAXG_addKernelPackage: failed to import kernels from ''
    # The built-in .mojoc packages used by the graph compiler live in lib/mojo,
    # while explicit cache prebuild still passes -I lib/mojo and -I max_site.
    local runtime_import_path="$discovered"
    if [ "$MAX_RUNTIME_MOJO_IMPORT_MODE" = "single-lib-mojo" ] && [ -d "$lib_mojo" ]; then
      runtime_import_path="$lib_mojo"
    fi

    if [ "$MAX_OVERRIDE_MOJO_IMPORT_PATH" = "1" ]; then
      export MODULAR_MOJO_MAX_IMPORT_PATH="$runtime_import_path"
      export MOJO_PACKAGE_PATH="$runtime_import_path"
    else
      export MODULAR_MOJO_MAX_IMPORT_PATH="$runtime_import_path${MODULAR_MOJO_MAX_IMPORT_PATH:+:$MODULAR_MOJO_MAX_IMPORT_PATH}"
      export MOJO_PACKAGE_PATH="$runtime_import_path${MOJO_PACKAGE_PATH:+:$MOJO_PACKAGE_PATH}"
    fi
    echo "Mojo import roots discovered: $discovered"
    echo "Runtime Mojo import path: $runtime_import_path"
    echo "Runtime Mojo import mode: $MAX_RUNTIME_MOJO_IMPORT_MODE"
    echo "Override Mojo import path: $MAX_OVERRIDE_MOJO_IMPORT_PATH"
  else
    echo "WARNING: no Mojo std/nn import roots discovered under $prefix" >&2
  fi
}

clear_mojo_compile_cache() {
  if [ "$MAX_CLEAR_MOJO_CACHE" = "1" ]; then
    find "$PROJECT_DIR/.pixi/envs/default" -type d -name '__mojocache__' -prune -exec rm -rf {} + 2>/dev/null || true
  fi
}

prebuild_mojo_cache_for_module() {
  local module="$1"
  local label="$2"
  local prefix="$PROJECT_DIR/.pixi/envs/default"
  local max_site="$prefix/lib/python3.12/site-packages/max"
  local build_log
  build_log="$(mktemp "/tmp/monitorme_${label}_import_XXXXXX.log")"

  if pixi_env_run python - <<PYMOD >"$build_log" 2>&1
import ${module}
print("${module} import ok")
PYMOD
  then
    echo "${module}: import ok"
    rm -f "$build_log"
    return 0
  fi

  echo "${module}: import failed; attempting explicit Mojo cache prebuild"
  cat "$build_log" >&2 || true

  local src out
  src="$(grep -oE 'Error compiling Mojo at .*[.]mojo' "$build_log" | sed 's/^Error compiling Mojo at //' | tail -1 || true)"
  out="$(grep -oE -- '-o [^ ]+[.]so' "$build_log" | sed 's/^-o //' | tail -1 || true)"

  if [ -z "$src" ] || [ -z "$out" ]; then
    err "Could not infer Mojo source/cache target for ${module}. See $build_log"
    return 1
  fi

  mkdir -p "$(dirname "$out")"
  pixi_env_run mojo build "$src" \
    -I "$prefix/lib/mojo" \
    -I "$max_site" \
    --emit shared-lib \
    -o "$out"

  pixi_env_run python - <<PYMOD
import ${module}
print("${module} import ok after prebuild")
PYMOD
  rm -f "$build_log"
}

prebuild_known_mojo_caches() {
  if [ "$MAX_PREBUILD_MOJO_CACHES" != "1" ]; then
    echo "Skipping Mojo cache prebuild. Set MONITORME_MAX_PREBUILD_MOJO_CACHES=1 to enable."
    return 0
  fi

  echo "=== Prebuilding MAX Mojo caches that Python mojo.importer cannot compile with -I ==="
  prebuild_mojo_cache_for_module "max._kv_cache_ops" "kv_cache_ops"
  prebuild_mojo_cache_for_module "max._distributed_ops" "distributed_ops"
}

patch_pixi_activation_env_if_requested() {
  if [ "$MAX_PATCH_PIXI_ACTIVATION_ENV" != "1" ]; then
    if ! grep -q '^MODULAR_MOJO_MAX_IMPORT_PATH *= *"' "$PROJECT_DIR/pixi.toml" 2>/dev/null; then
      cat <<'WARN'
WARNING: pixi.toml does not contain MODULAR_MOJO_MAX_IMPORT_PATH in [activation.env].
MAX may still start but the model worker can later crash with:
  failed to resolve built-in kernel package paths
  MAXG_addKernelPackage: failed to import kernels from ''
Run once before TERM1 if this happens:
  ./scripts/max/patch_max_pixi_activation_env.sh
Or set MONITORME_MAX_PATCH_PIXI_ACTIVATION_ENV=1 for this script to patch it.
WARN
    fi
    return 0
  fi

  local patcher
  patcher="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_max_pixi_activation_env.sh"
  MONITORME_MAX_PROJECT_DIR="$PROJECT_DIR" "$patcher"
}

print_mojo_std_help() {
  cat >&2 <<'HELP'

Detected a broken MAX/Mojo pixi environment: Mojo cannot locate module 'std'.
This is usually caused by a mismatched or partially upgraded MAX/Mojo package set.
For MonitorMe, the common cause is running `pixi add modular ...` on a rolling
MAX nightly channel, which can upgrade a previously working quickstart env.

Recommended recovery on Node1:
  1) Do not keep retrying this startup script with automatic pixi upgrades.
  2) Run diagnostics:
       ./scripts/max/diagnose_max_gemma3_1b_env.sh
  3) If the quickstart project is under git and has a known-good pixi.lock:
       cd "$MONITORME_MAX_PROJECT_DIR"
       git checkout -- pixi.toml pixi.lock
       pixi install --locked
  4) If no known-good lock exists, recreate the MAX quickstart environment from
     the exact recipe/version you previously validated, then keep it pinned.
  5) If the error appears only during max serve, check the discovered Mojo import
     roots printed by this script. MonitorMe now exports MODULAR_MOJO_MAX_IMPORT_PATH
     and MOJO_PACKAGE_PATH from the pixi environment when possible.
  6) Start MAX again using this script. This script will not mutate pixi unless
     MONITORME_MAX_PIXI_SYNC=1 is explicitly set.

HELP
}

echo "=== TERM1: MAX Gemma 3 1B server startup for MonitorMe ==="
echo "Project:      $PROJECT_DIR"
echo "Model:        $MODEL_ID"
echo "API port:     $API_PORT"
echo "Metrics port: $METRICS_PORT"
echo "Log:          $PROJECT_DIR/$LOG_FILE"
echo "Pixi sync:    $MAX_PIXI_SYNC"
echo "Clear Mojo cache: $MAX_CLEAR_MOJO_CACHE"
echo "Override Mojo import path: $MAX_OVERRIDE_MOJO_IMPORT_PATH"
echo "Skip max --help preflight: $MAX_SKIP_HELP_PREFLIGHT"
echo "Sample on host: $MAX_SAMPLE_ON_HOST"
echo "Prebuild Mojo caches: $MAX_PREBUILD_MOJO_CACHES"
echo "Patch pixi activation env: $MAX_PATCH_PIXI_ACTIVATION_ENV"
echo "Runtime Mojo import mode: $MAX_RUNTIME_MOJO_IMPORT_MODE"
echo

echo "=== Checking pixi project ==="
if [ ! -f pixi.toml ]; then
  err "pixi.toml not found in $PROJECT_DIR"
  cat >&2 <<EOF2
Create the MAX quickstart project first:
  pixi init quickstart -c https://conda.modular.com/max-nightly/ -c conda-forge
  cd quickstart
  pixi add modular huggingface_hub openai
EOF2
  exit 1
fi

if [ "$MAX_PIXI_SYNC" = "1" ]; then
  echo "=== MONITORME_MAX_PIXI_SYNC=1: intentionally syncing pixi packages ==="
  pixi add modular huggingface_hub openai
else
  echo "=== Checking required pixi commands without modifying environment ==="
  if ! pixi run python -c 'import openai, huggingface_hub' >/dev/null 2>&1; then
    err "openai and/or huggingface_hub are missing from the MAX pixi environment."
    echo "Install once intentionally with:" >&2
    echo "  cd $PROJECT_DIR" >&2
    echo "  pixi add huggingface_hub openai" >&2
    echo "or rerun this script with MONITORME_MAX_PIXI_SYNC=1 if you accept package changes." >&2
    exit 1
  fi
fi

echo "=== MAX/Mojo import path setup ==="
discover_and_export_mojo_import_path
patch_pixi_activation_env_if_requested
clear_mojo_compile_cache

echo "=== MAX/Mojo preflight ==="
if [ "$MAX_SKIP_HELP_PREFLIGHT" = "1" ]; then
  echo "Skipping pixi run max --help preflight to avoid eager compilation of optional MAX serving extensions."
  echo "Set MONITORME_MAX_SKIP_HELP_PREFLIGHT=0 to force this check."
else
  MAX_HELP_LOG="$(mktemp /tmp/monitorme_max_help_XXXXXX.log)"
  if ! pixi_env_run max --help >"$MAX_HELP_LOG" 2>&1; then
    cat "$MAX_HELP_LOG" >&2 || true
    if grep -q "unable to locate module 'std'" "$MAX_HELP_LOG"; then
      print_mojo_std_help
    else
      err "pixi run max --help failed. Inspect $MAX_HELP_LOG or run scripts/max/diagnose_max_gemma3_1b_env.sh."
    fi
    exit 1
  fi
  rm -f "$MAX_HELP_LOG"
fi

# Keep compiled Mojo caches by default. The Node1 full logs showed probe_mojo_std_nn_roots.sh
# could PASS max._kv_cache_ops using an existing cache, while TERM1 then failed after
# clearing __mojocache__ and forcing recompilation with missing std/nn. Clear caches only
# when intentionally debugging a stale compiled artifact:
#   MONITORME_MAX_CLEAR_MOJO_CACHE=1 ./scripts/max/term1_start_max_gemma3_1b.sh
#
# `max --help` can eagerly compile optional serving extensions, so it is skipped by default.
# This direct import checks the core bridge and kv cache path without clearing caches.
# Import the MAX Mojo bridge directly so the stdlib mismatch is caught before
# model startup and before the error scrolls through a long serve traceback.
MAX_CORE_LOG="$(mktemp /tmp/monitorme_max_core_mojo_XXXXXX.log)"
if ! pixi_env_run python - <<'PY' >"$MAX_CORE_LOG" 2>&1
import max._core_mojo
print("max._core_mojo import ok")
# Import a serving-time Mojo extension too. max --help and max._core_mojo can
# pass while max serve later fails compiling _kv_cache_ops without std/nn.
try:
    import max._kv_cache_ops
    print("max._kv_cache_ops import ok")
except Exception as exc:
    print(f"max._kv_cache_ops import failed: {exc!r}")
    raise
PY
then
  cat "$MAX_CORE_LOG" >&2 || true
  if grep -q "unable to locate module 'std'" "$MAX_CORE_LOG"; then
    print_mojo_std_help
  else
    err "MAX core Mojo import failed. Inspect $MAX_CORE_LOG or run scripts/max/diagnose_max_gemma3_1b_env.sh."
  fi
  exit 1
fi
rm -f "$MAX_CORE_LOG"

prebuild_known_mojo_caches

echo "=== Clearing unsafe/old inline HF_TOKEN variable ==="
unset HF_TOKEN || true

echo "=== Checking Hugging Face login ==="
if ! pixi_env_run hf auth whoami; then
  echo
  err "Hugging Face login missing."
  echo "Run once in $PROJECT_DIR:" >&2
  echo "  pixi run hf auth login" >&2
  echo
  echo "Then ensure the Hugging Face account has accepted Gemma access terms:" >&2
  echo "  https://huggingface.co/google/gemma-3-1b-it" >&2
  exit 1
fi

echo
echo "=== Checking Gemma gated-model access ==="
if ! pixi_env_run hf download "$MODEL_ID" config.json >/dev/null; then
  echo
  err "Cannot access $MODEL_ID."
  echo "Open Hugging Face in a browser with the same account and accept/request access:" >&2
  echo "  https://huggingface.co/google/gemma-3-1b-it" >&2
  exit 1
fi

echo
echo "=== Stopping any old MAX server ==="
pkill -f "max serve" || true
sleep 2

echo
echo "=== GPU state before MAX start ==="
nvidia-smi || true

echo
echo "=== Starting known-good MAX server ==="
echo "Keep this terminal running. Do not press Ctrl+C while MonitorMe validates."
if [ "$MAX_SAMPLE_ON_HOST" = "1" ]; then
  echo "Optional MAX sampler workaround enabled: --sample-on-host"
else
  echo "Optional MAX sampler workaround disabled. Set MONITORME_MAX_SAMPLE_ON_HOST=1 only if this MAX build supports it."
fi
echo

# MAX currently uses the default API/metrics ports for this path.
# Keep API_PORT/METRICS_PORT in the script output so future MAX flag changes can be made in one place.
MAX_SERVE_ARGS=(
  serve
  --model "$MODEL_ID"
  --devices=gpu
  --max-length 1024
  --max-batch-size 1
  --max-batch-input-tokens 1024
  --max-batch-total-tokens 1024
  --device-memory-utilization "$DEVICE_MEMORY_UTILIZATION"
  --temperature 0.0
)

if [ "$MAX_SAMPLE_ON_HOST" = "1" ]; then
  MAX_SERVE_ARGS+=(--sample-on-host)
fi

MODULAR_DEBUG=device-sync-mode pixi_env_run max "${MAX_SERVE_ARGS[@]}"   2>&1 | tee "$LOG_FILE"
