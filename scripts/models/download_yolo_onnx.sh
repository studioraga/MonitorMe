#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/models/download_yolo_onnx.sh [--force] [--model-path PATH] [--url URL] [--sha256 SHA256] [--env-file FILE]

Downloads the default YOLO ONNX model into the MonitorMe repo-local model
folder and persists MonitorMe detector settings in .env.

Run this before installing/running detector support:
  ./scripts/models/download_yolo_onnx.sh
  python -m pip install -e '.[api,camera,detector,test]'

Defaults:
  MONITORME_MODEL_DIR=models/object_detection
  MONITORME_DETECTOR_MODEL_ID=yolo11n-coco-onnx
  MONITORME_DETECTOR_MODEL_PATH=models/object_detection/yolo11n.onnx
  MONITORME_DETECTOR_MODEL_URL=https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx
  MONITORME_ENV_FILE=.env

Options:
  --force              re-download even when the target ONNX file already exists
  --model-path PATH    override repo-relative or absolute ONNX target path
  --url URL            override YOLO ONNX download URL
  --sha256 SHA256      verify the downloaded ONNX file against this SHA256
  --env-file FILE      write MonitorMe detector variables to this env file
  -h, --help           show this help

Notes:
  - This script never uploads private CCTV frames or events.
  - It only downloads a public model file into the local MonitorMe repo.
  - Existing non-empty model files are reused unless --force is supplied.
USAGE
}

FORCE=0
CLI_MODEL_PATH=""
CLI_MODEL_URL=""
CLI_MODEL_SHA256=""
CLI_ENV_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      ;;
    --model-path)
      [[ $# -ge 2 ]] || { echo "ERROR: --model-path requires a value" >&2; exit 2; }
      CLI_MODEL_PATH="$2"
      shift
      ;;
    --url)
      [[ $# -ge 2 ]] || { echo "ERROR: --url requires a value" >&2; exit 2; }
      CLI_MODEL_URL="$2"
      shift
      ;;
    --sha256)
      [[ $# -ge 2 ]] || { echo "ERROR: --sha256 requires a value" >&2; exit 2; }
      CLI_MODEL_SHA256="$2"
      shift
      ;;
    --env-file)
      [[ $# -ge 2 ]] || { echo "ERROR: --env-file requires a value" >&2; exit 2; }
      CLI_ENV_FILE="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_DIR_REL="${MONITORME_MODEL_DIR:-models/object_detection}"
MODEL_ID="${MONITORME_DETECTOR_MODEL_ID:-yolo11n-coco-onnx}"
MODEL_PATH_REL="${CLI_MODEL_PATH:-${MONITORME_DETECTOR_MODEL_PATH:-${MODEL_DIR_REL}/yolo11n.onnx}}"
MODEL_URL="${CLI_MODEL_URL:-${MONITORME_DETECTOR_MODEL_URL:-https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx}}"
MODEL_SHA256="${CLI_MODEL_SHA256:-${MONITORME_DETECTOR_MODEL_SHA256:-}}"
ENV_FILE="${CLI_ENV_FILE:-${MONITORME_ENV_FILE:-$REPO_ROOT/.env}}"

abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$path"
  fi
}

rel_or_abs_for_env() {
  local path="$1"
  if [[ "$path" = "$REPO_ROOT"/* ]]; then
    printf '%s\n' "${path#$REPO_ROOT/}"
  else
    printf '%s\n' "$path"
  fi
}

MODEL_PATH_ABS="$(abs_path "$MODEL_PATH_REL")"
MODEL_DIR_ABS="$(dirname "$MODEL_PATH_ABS")"
MODEL_PATH_ENV="$(rel_or_abs_for_env "$MODEL_PATH_ABS")"
TMP_PATH="${MODEL_PATH_ABS}.tmp"
LOG_DIR="$REPO_ROOT/results/models"
LOG_FILE="$LOG_DIR/download_yolo_onnx.log"

mkdir -p "$MODEL_DIR_ABS" "$(dirname "$(abs_path "$ENV_FILE")")" "$LOG_DIR"

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_FILE" >&2; }

persist_env_value() {
  local key="$1"
  local value="$2"
  local file_abs
  file_abs="$(abs_path "$ENV_FILE")"
  touch "$file_abs"
  if grep -qE "^${key}=" "$file_abs"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file_abs"
  else
    printf '\n%s=%s\n' "$key" "$value" >> "$file_abs"
  fi
}

if [[ -s "$MODEL_PATH_ABS" && "$FORCE" -eq 0 ]]; then
  log "[OK] YOLO ONNX model already exists: $MODEL_PATH_ABS"
else
  log "=== Download MonitorMe YOLO ONNX model ==="
  log "url=$MODEL_URL"
  log "target=$MODEL_PATH_ABS"
  rm -f "$TMP_PATH"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o "$TMP_PATH" "$MODEL_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$TMP_PATH" "$MODEL_URL"
  else
    echo "ERROR: curl or wget is required to download the YOLO ONNX model." >&2
    exit 1
  fi

  if [[ -n "$MODEL_SHA256" ]]; then
    echo "${MODEL_SHA256}  ${TMP_PATH}" | sha256sum -c -
  fi

  if [[ ! -s "$TMP_PATH" ]]; then
    echo "ERROR: downloaded model is empty: $TMP_PATH" >&2
    exit 1
  fi

  mv "$TMP_PATH" "$MODEL_PATH_ABS"
  log "[OK] Downloaded YOLO ONNX model: $MODEL_PATH_ABS"
fi

ACTUAL_SHA256="$(sha256sum "$MODEL_PATH_ABS" | awk '{print $1}')"

persist_env_value MONITORME_MODEL_DIR "$MODEL_DIR_REL"
persist_env_value MONITORME_DETECTOR_MODEL_ID "$MODEL_ID"
persist_env_value MONITORME_DETECTOR_MODEL_PATH "$MODEL_PATH_ENV"
persist_env_value MONITORME_DETECTOR_MODEL_URL "$MODEL_URL"
if [[ -n "$MODEL_SHA256" ]]; then
  persist_env_value MONITORME_DETECTOR_MODEL_SHA256 "$MODEL_SHA256"
fi

log "[OK] model_id=$MODEL_ID"
log "[OK] model_path=$MODEL_PATH_ENV"
log "[OK] sha256=$ACTUAL_SHA256"
log "[OK] env_file=$(abs_path "$ENV_FILE")"
log "Next: python -m pip install -e '.[api,camera,detector,test]'"
log "Then: ./scripts/validate_node1_c922_yolo_live.sh"

cat <<EOF_SUMMARY
{
  "ok": true,
  "model_id": "$MODEL_ID",
  "model_path": "$MODEL_PATH_ENV",
  "model_sha256": "$ACTUAL_SHA256",
  "env_file": "$(abs_path "$ENV_FILE")",
  "log_file": "$LOG_FILE"
}
EOF_SUMMARY
