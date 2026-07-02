#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${MONITORME_MAX_PROJECT_DIR:-$HOME/dev/modular/project/quickstart_py312_stable_mojo}"
cd "$PROJECT_DIR"
PREFIX="$PROJECT_DIR/.pixi/envs/default"

export MONITORME_MAX_OVERRIDE_MOJO_IMPORT_PATH="${MONITORME_MAX_OVERRIDE_MOJO_IMPORT_PATH:-1}"
export MONITORME_MAX_RUNTIME_MOJO_IMPORT_MODE="${MONITORME_MAX_RUNTIME_MOJO_IMPORT_MODE:-single-lib-mojo}"

# Build a clean Mojo import path: prefer lib/mojo, exclude false-positive
# directories such as share/locale/nn and share/tabset/std.
DISCOVERED=""
if [ -d "$PREFIX/lib/mojo" ]; then
  DISCOVERED="$PREFIX/lib/mojo"
fi
while IFS= read -r dir; do
  [ -n "$dir" ] || continue
  case "$dir" in
    */share/locale|*/share/locale/*|*/share/tabset|*/share/tabset/*) continue ;;
  esac
  case ":$DISCOVERED:" in
    *":$dir:"*) ;;
    *) DISCOVERED="${DISCOVERED:+$DISCOVERED:}$dir" ;;
  esac
done < <(
  {
    find "$PREFIX" -type f \( -name 'std.mojoc' -o -name 'nn.mojoc' -o -name 'comm.mojoc' -o -name 'layout.mojoc' -o -name 'stdlib.mojopkg' -o -name 'std.mojopkg' -o -name 'nn.mojopkg' \) -printf '%h\n' 2>/dev/null || true
    find "$PREFIX" -type d -path '*/site-packages/max' -printf '%p\n' 2>/dev/null || true
  } | awk 'NF && !seen[$0]++'
)
if [ -n "$DISCOVERED" ]; then
  RUNTIME_DISCOVERED="$DISCOVERED"
  if [ "$MONITORME_MAX_RUNTIME_MOJO_IMPORT_MODE" = "single-lib-mojo" ] && [ -d "$PREFIX/lib/mojo" ]; then
    RUNTIME_DISCOVERED="$PREFIX/lib/mojo"
  fi
  export MODULAR_MOJO_MAX_IMPORT_PATH="$RUNTIME_DISCOVERED"
  export MOJO_PACKAGE_PATH="$RUNTIME_DISCOVERED"
fi

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

echo "=== Mojo std/nn root probe ==="
echo "Project: $PROJECT_DIR"
echo "Prefix:  $PREFIX"
echo

echo "=== package inventory ==="
pixi list | grep -E '^(modular|mojo)([[:space:]]|$)' || true

echo
echo "=== mojo executable ==="
pixi_env_run which mojo || true
pixi_env_run mojo --version || true

echo
echo "=== clean Mojo import roots used ==="
echo "Pixi env wrapper: 1"
echo "Runtime Mojo import mode: ${MONITORME_MAX_RUNTIME_MOJO_IMPORT_MODE}"
echo "Discovered compile roots:"
printf '%s\n' "$DISCOVERED" | tr ':' '\n'
echo "Runtime exported roots:"
printf '%s\n' "${MODULAR_MOJO_MAX_IMPORT_PATH:-}" | tr ':' '\n'
echo
echo "=== candidate std/nn directories, excluding locale/tabset false positives ==="
find "$PREFIX" -type d \( -name std -o -name nn \) -print 2>/dev/null | grep -Ev '/share/(locale|tabset)(/|$)' | head -80 || true

echo
echo "=== candidate mojopkg files ==="
find "$PREFIX" -type f \( -name 'stdlib.mojopkg' -o -name 'std.mojopkg' -o -name 'nn.mojopkg' \) -print 2>/dev/null | head -80 || true

echo
echo "=== max internal mojo modules ==="
find "$PREFIX" -path '*/site-packages/max/*' -name '*.mojo' -print 2>/dev/null | head -40 || true

echo
echo "=== import preflight ==="
pixi_env_run python - <<'PY'
import importlib
for name in ("max._core_mojo", "max._kv_cache_ops"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: FAIL: {type(exc).__name__}: {exc}")
    else:
        print(f"{name}: PASS")
PY
