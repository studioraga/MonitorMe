#!/usr/bin/env bash
set -Eeuo pipefail

# Create a clean, isolated MAX + Gemma 3 1B pixi workspace pinned to Python 3.12.
# Use this when the existing quickstart env was accidentally upgraded and fails
# with Mojo stdlib errors such as: unable to locate module 'std'.
#
# This script intentionally creates a new project directory by default instead of
# rewriting the existing quickstart env.

PROJECT_DIR="${MONITORME_MAX_NEW_PROJECT_DIR:-$HOME/dev/modular/project/quickstart_py312}"
MODEL_ID="${MONITORME_LLM_MODEL_ID:-google/gemma-3-1b-it}"
FORCE_RECREATE="${MONITORME_MAX_FORCE_RECREATE:-0}"
CHANNEL1="${MONITORME_MAX_CHANNEL:-https://conda.modular.com/max-nightly/}"
CHANNEL2="${MONITORME_CONDA_FORGE_CHANNEL:-conda-forge}"
ADD_EXPLICIT_MOJO="${MONITORME_MAX_ADD_EXPLICIT_MOJO:-1}"

PARENT_DIR="$(dirname "$PROJECT_DIR")"
PROJECT_NAME="$(basename "$PROJECT_DIR")"



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
  if [ -n "$discovered" ]; then
    export MODULAR_MOJO_MAX_IMPORT_PATH="$discovered${MODULAR_MOJO_MAX_IMPORT_PATH:+:$MODULAR_MOJO_MAX_IMPORT_PATH}"
    export MOJO_PACKAGE_PATH="$discovered${MOJO_PACKAGE_PATH:+:$MOJO_PACKAGE_PATH}"
    echo "Mojo import roots discovered: $discovered"
  else
    echo "WARNING: no Mojo std/nn import roots discovered under $prefix" >&2
  fi
}

mkdir -p "$PARENT_DIR"

if [ -e "$PROJECT_DIR" ] && [ "$FORCE_RECREATE" != "1" ]; then
  echo "ERROR: $PROJECT_DIR already exists." >&2
  echo "Set MONITORME_MAX_FORCE_RECREATE=1 to move it aside and recreate." >&2
  exit 1
fi

if [ -e "$PROJECT_DIR" ] && [ "$FORCE_RECREATE" = "1" ]; then
  BACKUP="${PROJECT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
  echo "Moving existing project to: $BACKUP"
  mv "$PROJECT_DIR" "$BACKUP"
fi

echo "=== Creating clean MAX/Gemma pixi workspace ==="
echo "Project: $PROJECT_DIR"
echo "Model:   $MODEL_ID"
echo "Python:  python=3.12"
echo "Channel: $CHANNEL1"
echo "Channel: $CHANNEL2"
echo "Explicit mojo package: $ADD_EXPLICIT_MOJO"

cd "$PARENT_DIR"
pixi init "$PROJECT_NAME" -c "$CHANNEL1" -c "$CHANNEL2"
cd "$PROJECT_DIR"

# Pin Python first so conda-forge does not solve the MAX workspace to Python 3.14.
if [ "$ADD_EXPLICIT_MOJO" = "1" ]; then
  # Add the standalone mojo package explicitly. Some modular-only solves expose
  # the mojo executable but fail serving-time compilation of max._kv_cache_ops
  # because Mojo cannot locate std/nn package roots.
  pixi add "python=3.12" modular mojo huggingface_hub openai
else
  pixi add "python=3.12" modular huggingface_hub openai
fi

echo
echo "=== Environment versions ==="
discover_and_export_mojo_import_path
pixi run python --version
pixi run mojo --version || true
echo
echo "=== Modular/Mojo package inventory ==="
pixi list | grep -E '^(modular|mojo)([[:space:]]|$)' || true
pixi run max --help >/dev/null

echo
echo "=== MAX Mojo import preflight ==="
pixi run python - <<'PY'
import importlib
for name in ("max._core_mojo", "max._kv_cache_ops"):
    importlib.import_module(name)
    print(f"{name} import ok")
PY

echo
echo "=== Hugging Face login/access check ==="
if ! pixi run hf auth whoami; then
  echo
  echo "Hugging Face login is missing. Run this once:" >&2
  echo "  cd $PROJECT_DIR" >&2
  echo "  pixi run hf auth login" >&2
  echo "Then rerun this script or the MonitorMe TERM1 script." >&2
  exit 1
fi

pixi run hf download "$MODEL_ID" config.json >/dev/null

echo
echo "=== Clean MAX/Gemma env created ==="
echo "Use this from MonitorMe TERM1:"
echo "  export MONITORME_MAX_PROJECT_DIR=\"$PROJECT_DIR\""
echo "  ./scripts/max/term1_start_max_gemma3_1b.sh"
echo
echo "Then validate from TERM2:"
echo "  ./scripts/max/term2_validate_max_gemma3_1b.sh"
echo "  ./scripts/max/term2_validate_monitorme_gemma_v02_live.sh"
