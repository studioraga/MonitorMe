#!/usr/bin/env bash
set -Eeuo pipefail

# Diagnose the external pixi MAX/Gemma environment used by MonitorMe.
# This script is read-only: it does not run pixi add/remove/install.

PROJECT_DIR="${MONITORME_MAX_PROJECT_DIR:-$HOME/dev/modular/project/quickstart}"
RESULTS_DIR="${MONITORME_MAX_DIAG_RESULTS_DIR:-$PROJECT_DIR/results/max_env_diag_$(date +%Y%m%d_%H%M%S)}"
MODEL_ID="${MONITORME_LLM_MODEL_ID:-google/gemma-3-1b-it}"

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_DIR"



discover_and_export_mojo_import_path() {
  local prefix="$PROJECT_DIR/.pixi/envs/default"
  local discovered=""
  if [ -d "$prefix" ]; then
    while IFS= read -r dir; do
      [ -n "$dir" ] || continue
      case ":$discovered:" in
        *":$dir:"*) ;;
        *) discovered="${discovered:+$discovered:}$dir" ;;
      esac
    done < <(
      {
        find "$prefix" -type d -name std -printf '%h\n' 2>/dev/null || true
        find "$prefix" -type d -name nn -printf '%h\n' 2>/dev/null || true
        find "$prefix" -type f \( -name 'stdlib.mojopkg' -o -name 'std.mojopkg' -o -name 'nn.mojopkg' \) -printf '%h\n' 2>/dev/null || true
        find "$prefix" -type d -path '*/site-packages/max' -printf '%p\n' 2>/dev/null || true
        find "$prefix" -type d -path '*/lib/mojo' -printf '%p\n' 2>/dev/null || true
      } | awk 'NF && !seen[$0]++'
    )
  fi
  echo "$discovered" >"$RESULTS_DIR/discovered_mojo_import_path.txt"
  if [ -n "$discovered" ]; then
    export MODULAR_MOJO_MAX_IMPORT_PATH="$discovered${MODULAR_MOJO_MAX_IMPORT_PATH:+:$MODULAR_MOJO_MAX_IMPORT_PATH}"
    export MOJO_PACKAGE_PATH="$discovered${MOJO_PACKAGE_PATH:+:$MOJO_PACKAGE_PATH}"
  fi
}

run_capture() {
  local name="$1"
  shift
  echo
  echo "=== $name ==="
  set +e
  "$@" >"$RESULTS_DIR/${name}.stdout" 2>"$RESULTS_DIR/${name}.stderr"
  local rc=$?
  set -e
  echo "returncode=$rc" | tee "$RESULTS_DIR/${name}.rc"
  if [ -s "$RESULTS_DIR/${name}.stdout" ]; then
    cat "$RESULTS_DIR/${name}.stdout" | head -120
  fi
  if [ -s "$RESULTS_DIR/${name}.stderr" ]; then
    cat "$RESULTS_DIR/${name}.stderr" | head -160 >&2
  fi
  return 0
}

echo "=== MonitorMe MAX/Gemma pixi environment diagnosis ==="
echo "Project: $PROJECT_DIR"
echo "Model:   $MODEL_ID"
echo "Results: $RESULTS_DIR"
echo

if [ ! -f pixi.toml ]; then
  echo "ERROR: pixi.toml not found in $PROJECT_DIR" >&2
  exit 1
fi

cp pixi.toml "$RESULTS_DIR/pixi.toml" || true
[ -f pixi.lock ] && cp pixi.lock "$RESULTS_DIR/pixi.lock" || true
discover_and_export_mojo_import_path
echo "MODULAR_MOJO_MAX_IMPORT_PATH=${MODULAR_MOJO_MAX_IMPORT_PATH:-}" >"$RESULTS_DIR/mojo_import_env.txt"
echo "MOJO_PACKAGE_PATH=${MOJO_PACKAGE_PATH:-}" >>"$RESULTS_DIR/mojo_import_env.txt"

run_capture pixi_info pixi info
run_capture pixi_list pixi list
run_capture python_version pixi run python --version
run_capture python_imports pixi run python -c 'import sys; print(sys.version); import openai, huggingface_hub; print("python imports ok")'
run_capture max_core_mojo_import pixi run python -c 'import max._core_mojo; print("max._core_mojo import ok")'
run_capture max_kv_cache_ops_import pixi run python -c 'import max._kv_cache_ops; print("max._kv_cache_ops import ok")'
run_capture hf_whoami pixi run hf auth whoami
run_capture mojo_version pixi run mojo --version
run_capture max_help pixi run max --help
run_capture max_serve_help pixi run max serve --help
run_capture hf_model_config pixi run hf download "$MODEL_ID" config.json

echo
PYTHON_VERSION_TEXT="$(cat "$RESULTS_DIR/python_version.stdout" 2>/dev/null || true)"
HAS_LOCK=0
[ -f pixi.lock ] && HAS_LOCK=1
IN_GIT=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IN_GIT=1
fi

if grep -R "unable to locate module 'std'" "$RESULTS_DIR" >/dev/null 2>&1; then
  cat <<'HELP'
=== DIAGNOSIS: BROKEN MAX/MOJO STDLIB ===
The pixi environment failed with: unable to locate module 'std'.
This means the external MAX/Mojo environment is broken or internally mismatched.
It is not caused by MonitorMe's SQLite/event/assistant code.

Strong signals to check in this diagnostic:
- python_version: if this is Python 3.14, recreate a Python 3.12 MAX env first.
- pixi.lock: if missing, `pixi install --locked` cannot restore an older good solve.
- git repo: if this quickstart is not under git, `git checkout -- pixi.toml pixi.lock` cannot work.

Recommended recovery:
1) Keep this diagnostic artifact directory.
2) Create a clean Python 3.12 MAX workspace:
     ./scripts/max/create_max_gemma3_1b_py312_env.sh
3) Use that clean workspace for MonitorMe:
     export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312"
     ./scripts/max/term1_start_max_gemma3_1b.sh
4) In a second terminal, run:
     ./scripts/max/term2_validate_max_gemma3_1b.sh
     ./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
HELP
else
  echo "No Mojo stdlib error string found in diagnostic outputs. Review $RESULTS_DIR for details."
fi

echo
echo "=== quick environment interpretation ==="
echo "python_version=$PYTHON_VERSION_TEXT"
echo "discovered_mojo_import_path=$(cat "$RESULTS_DIR/discovered_mojo_import_path.txt" 2>/dev/null || true)"
echo "pixi_lock_present=$HAS_LOCK"
echo "project_is_git_repo=$IN_GIT"
if echo "$PYTHON_VERSION_TEXT" | grep -q '3.14'; then
  echo "warning=Python 3.14 was selected by pixi; for this MAX/Gemma path, prefer a clean Python 3.12 pixi environment."
fi
if [ "$HAS_LOCK" != "1" ]; then
  echo "warning=pixi.lock is missing; pixi install --locked cannot restore a previously validated solve."
fi
if [ "$IN_GIT" != "1" ]; then
  echo "warning=quickstart is not a git repo; git checkout -- pixi.toml pixi.lock cannot restore files here."
fi

echo
echo "Artifacts saved under: $RESULTS_DIR"
