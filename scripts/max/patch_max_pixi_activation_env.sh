#!/usr/bin/env bash
set -Eeuo pipefail

# Patch the external MAX/Gemma pixi workspace so Mojo package roots are part of
# Pixi activation itself, not only parent-shell environment. Node1 logs showed
# that `max serve` can spawn a worker that reaches MAX graph/kernel resolution
# and then crashes with:
#   failed to resolve built-in kernel package paths
#   MAXG_addKernelPackage: failed to import kernels from ''
# A Modular forum report for this exact error recommends setting
# MODULAR_MOJO_MAX_IMPORT_PATH through [activation.env] in pixi.toml.
# v0.2.12 intentionally writes only the lib/mojo root, not a broad colon list,
# because Node1 logs showed lib/mojo:site-packages/max still led to an empty
# kernel package path during MAXG_addKernelPackage.

PROJECT_DIR="${MONITORME_MAX_PROJECT_DIR:-$HOME/dev/modular/project/quickstart_py312_stable_mojo}"
PREFIX="$PROJECT_DIR/.pixi/envs/default"
MAX_SITE="$PREFIX/lib/python3.12/site-packages/max"
LIB_MOJO="$PREFIX/lib/mojo"
PIXI_TOML="$PROJECT_DIR/pixi.toml"
FORCE="${MONITORME_MAX_PATCH_FORCE:-0}"

cd "$PROJECT_DIR"

if [ ! -f "$PIXI_TOML" ]; then
  echo "ERROR: pixi.toml not found: $PIXI_TOML" >&2
  exit 1
fi
if [ ! -d "$LIB_MOJO" ]; then
  echo "ERROR: missing Mojo package root: $LIB_MOJO" >&2
  exit 1
fi
if [ ! -d "$MAX_SITE" ]; then
  echo "ERROR: missing MAX site-package root: $MAX_SITE" >&2
  exit 1
fi

# Keep Pixi activation env focused on the real compiled Mojo package root.
# The MAX graph compiler resolves built-in .mojoc kernel packages from lib/mojo.
# Explicit cache prebuilds can still use -I "$MAX_SITE" separately.
IMPORT_ROOTS="$LIB_MOJO"

if grep -q '^MODULAR_MOJO_MAX_IMPORT_PATH *= *"' "$PIXI_TOML" && grep -q '^MOJO_PACKAGE_PATH *= *"' "$PIXI_TOML" && [ "$FORCE" != "1" ]; then
  echo "pixi.toml already contains Mojo activation env entries. Use MONITORME_MAX_PATCH_FORCE=1 to rewrite."
else
  backup="$PIXI_TOML.backup.$(date +%Y%m%d_%H%M%S)"
  cp -a "$PIXI_TOML" "$backup"
  echo "Backup: $backup"

  PIXI_TOML="$PIXI_TOML" IMPORT_ROOTS="$IMPORT_ROOTS" python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

path = Path(os.environ["PIXI_TOML"])
roots = os.environ["IMPORT_ROOTS"]
text = path.read_text()
lines = text.splitlines()

out: list[str] = []
in_activation_env = False
inserted = False
saw_activation_env = False

keys = ("MODULAR_MOJO_MAX_IMPORT_PATH", "MOJO_PACKAGE_PATH")

def activation_lines() -> list[str]:
    return [
        f'MODULAR_MOJO_MAX_IMPORT_PATH = "{roots}"',
        f'MOJO_PACKAGE_PATH = "{roots}"',
    ]

for line in lines:
    stripped = line.strip()
    is_section = stripped.startswith("[") and stripped.endswith("]")

    if stripped == "[activation.env]":
        saw_activation_env = True
        in_activation_env = True
        out.append(line)
        out.extend(activation_lines())
        inserted = True
        continue

    if is_section and in_activation_env:
        in_activation_env = False

    if in_activation_env and any(stripped.startswith(k + " ") or stripped.startswith(k + "=") for k in keys):
        continue

    out.append(line)

if not saw_activation_env:
    if out and out[-1].strip():
        out.append("")
    out.append("[activation.env]")
    out.extend(activation_lines())
    inserted = True

if not inserted:
    raise SystemExit("failed to insert activation env")

path.write_text("\n".join(out) + "\n")
PY
fi

echo "=== Patched Mojo activation env ==="
grep -A3 '^\[activation.env\]' "$PIXI_TOML" || true

echo
printf 'Expected import roots:\n%s\n' "$IMPORT_ROOTS"

echo
printf 'Pixi activation check:\n'
pixi run env | grep -E '^(MODULAR_MOJO_MAX_IMPORT_PATH|MOJO_PACKAGE_PATH)=' || true
