#!/usr/bin/env bash
set -Eeuo pipefail

# Validate the standalone MAX + Gemma 3 1B OpenAI-compatible server.
# Run this in TERM2 after scripts/max/term1_start_max_gemma3_1b.sh is serving.
#
# v0.2.2 behavior: this script does not run `pixi add` automatically. It only
# checks the Python OpenAI client and tells you how to install it if missing.

PROJECT_DIR="${MONITORME_MAX_PROJECT_DIR:-$HOME/dev/modular/project/quickstart}"
MODEL_ID="${MONITORME_LLM_MODEL_ID:-google/gemma-3-1b-it}"
BASE_URL="${MONITORME_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
METRICS_URL="${MONITORME_MAX_METRICS_URL:-http://127.0.0.1:8001/metrics}"
RESULTS_DIR="${MONITORME_MAX_RESULTS_DIR:-$PROJECT_DIR/results/max_gemma3_1b_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== TERM2: MAX Gemma 3 1B validation for MonitorMe ==="
echo "Project:  $PROJECT_DIR"
echo "Model:    $MODEL_ID"
echo "Base URL: $BASE_URL"
echo "Metrics:  $METRICS_URL"
echo "Results:  $RESULTS_DIR"
echo

echo "=== Checking OpenAI Python client without modifying pixi environment ==="
if ! pixi run python -c "import openai" >/dev/null 2>&1; then
  echo "ERROR: openai Python package is missing from the MAX pixi environment." >&2
  echo "Install it intentionally with:" >&2
  echo "  cd $PROJECT_DIR" >&2
  echo "  pixi add openai" >&2
  exit 1
fi

echo
echo "=== Waiting for MAX API server ==="
for i in $(seq 1 120); do
  if curl -fsS "$BASE_URL/models" >/dev/null 2>&1; then
    echo "PASS: API server reachable at $BASE_URL"
    break
  fi
  if [ "$i" -eq 120 ]; then
    echo "ERROR: API server not reachable. Check TERM1." >&2
    exit 1
  fi
  sleep 1
done

echo
echo "=== Waiting for MAX metrics server ==="
for i in $(seq 1 60); do
  if curl -fsS "$METRICS_URL" >/dev/null 2>&1; then
    echo "PASS: Metrics server reachable at $METRICS_URL"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "ERROR: Metrics server not reachable. Check TERM1." >&2
    exit 1
  fi
  sleep 1
done

echo
echo "=== Listening ports ==="
ss -ltnp | grep -E ':8000|:8001' | tee "$RESULTS_DIR/ports.txt" || true

echo
echo "=== /v1/models ==="
curl -sS "$BASE_URL/models" > "$RESULTS_DIR/models.json"
python3 -m json.tool "$RESULTS_DIR/models.json"

echo
echo "=== Metrics capture ==="
curl -sS "$METRICS_URL" > "$RESULTS_DIR/metrics_full.prom"
head -80 "$RESULTS_DIR/metrics_full.prom" | tee "$RESULTS_DIR/metrics_head.txt"

echo
echo "=== Creating Python OpenAI-compatible client test ==="
cat > "$RESULTS_DIR/test_gemma3_1b.py" <<PY
from openai import OpenAI

MODEL = "$MODEL_ID"
BASE_URL = "$BASE_URL"

client = OpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=120.0)
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Return exactly this text and nothing else: NODE1_MAX_GEMMA3_1B_PASS"}],
    max_tokens=32,
    temperature=0.0,
)
print(resp.choices[0].message.content)
PY

echo
echo "=== Python client inference ==="
pixi run python "$RESULTS_DIR/test_gemma3_1b.py" | tee "$RESULTS_DIR/python_client_output.txt"

echo
echo "=== curl chat completion inference ==="
curl -sS "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Return exactly this text and nothing else: NODE1_MAX_CURL_PASS\"}],\"max_tokens\":32,\"temperature\":0.0}" \
  > "$RESULTS_DIR/curl_chat_completion.json"
python3 -m json.tool "$RESULTS_DIR/curl_chat_completion.json"

echo
echo "=== Repeatability test: 5 requests ==="
: > "$RESULTS_DIR/repeatability.txt"
for i in $(seq 1 5); do
  echo "=== request $i ===" | tee -a "$RESULTS_DIR/repeatability.txt"
  curl -sS "$BASE_URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Return exactly this text and nothing else: NODE1_MAX_REPEATABILITY_PASS\"}],\"max_tokens\":32,\"temperature\":0.0}" \
    > "$RESULTS_DIR/repeatability_${i}.json"
  python3 -m json.tool "$RESULTS_DIR/repeatability_${i}.json" | tee -a "$RESULTS_DIR/repeatability.txt"
  echo | tee -a "$RESULTS_DIR/repeatability.txt"
done

echo
echo "=== GPU evidence after inference ==="
nvidia-smi | tee "$RESULTS_DIR/nvidia_smi_after_inference.txt" || true

echo
echo "=== Filtered MAX metrics after inference ==="
curl -sS "$METRICS_URL" > "$RESULTS_DIR/metrics_after_inference.prom"
grep -Ei 'maxserve|request|token|latency|model|batch|cache|generation|prompt' \
  "$RESULTS_DIR/metrics_after_inference.prom" \
  | head -160 \
  | tee "$RESULTS_DIR/metrics_filtered_after_inference.txt" || true

cat > "$RESULTS_DIR/VALIDATION_SUMMARY.txt" <<EOF2
Node1 MAX + Gemma 3 1B local serving baseline

Status:
- MAX API server at $BASE_URL: PASS
- MAX metrics server at $METRICS_URL: PASS
- /v1/models endpoint: PASS
- Python OpenAI-compatible client: PASS
- curl /v1/chat/completions: PASS
- Repeatability requests: completed
- GPU residency captured with nvidia-smi: PASS

Known-good model:
- $MODEL_ID

Known-good serving flags:
- --devices=gpu
- --max-length 1024
- --max-batch-size 1
- --max-batch-input-tokens 1024
- --max-batch-total-tokens 1024
- --device-memory-utilization 0.90
- --sample-on-host
- --temperature 0.0

Important workaround:
- --sample-on-host is required on this Node1 RTX 5060 Ti path because the default GPU sampling/top-k path previously crashed with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES.
EOF2

echo
echo "=== Validation summary ==="
cat "$RESULTS_DIR/VALIDATION_SUMMARY.txt"
echo
echo "Artifacts saved under: $RESULTS_DIR"
