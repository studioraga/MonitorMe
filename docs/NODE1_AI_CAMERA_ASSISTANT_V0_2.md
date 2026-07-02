# Node1 AI Camera Assistant v0.2 — Gemma/MAX summaries

This milestone adds an optional local Gemma 3 1B explanation layer through a MAX OpenAI-compatible endpoint. It does **not** move vision or decision-making into Gemma.

```text
C922 /dev/video0
  -> motion gate
  -> YOLO ONNX visual facts
  -> normalized motion_detected / object_detected rows
  -> event contract JSON
  -> deterministic Node1 policy decision
  -> Gemma/MAX strict JSON summary, if configured and valid
  -> deterministic fallback summary, always available
  -> assistant_summaries row
```

## Model split

```text
YOLO ONNX
  role: fast visual facts
  output: labels, confidence, bbox, model_id, event_id

Node1 deterministic policy
  role: decisions/actions
  output: action, reason, severity, duration

Gemma 3 1B via MAX
  role: explanation only
  input: event_contract + policy_decision + artifact metadata
  output: bounded JSON operator summary
```

Gemma never receives raw CCTV frames in v0.2. It receives structured local facts only.

## Environment

Gemma/MAX is disabled by default. To enable it after starting MAX locally:

```bash
export MONITORME_LLM_PROVIDER=max-openai
export MONITORME_ASSISTANT_USE_GEMMA=1
export MONITORME_LLM_BASE_URL=http://127.0.0.1:8000/v1
export MONITORME_LLM_MODEL_ID=google/gemma-3-1b-it
export MONITORME_LLM_API_KEY=EMPTY
export MONITORME_LLM_TEMPERATURE=0.0
```

Recommended MAX command on Node1:

```bash
MODULAR_DEBUG=device-sync-mode max serve \
  --model google/gemma-3-1b-it \
  --devices=gpu \
  --max-length 1024 \
  --max-batch-size 1 \
  --max-batch-input-tokens 1024 \
  --max-batch-total-tokens 1024 \
  --device-memory-utilization 0.90 \
  --sample-on-host \
  --temperature 0.0
```

## Strict JSON contract

Gemma must return exactly one JSON object:

```json
{
  "operator_summary": "...",
  "event_reason": "...",
  "dashboard_tag": "...",
  "recommended_next_step": "...",
  "severity_label": "info|review|urgent",
  "cited_event_ids": ["evt_..."]
}
```

The validator rejects:

```text
missing required keys
unsupported extra keys
invalid severity labels
unknown cited event IDs
identity / face / intent / weapon / suspicious claims without evidence
non-JSON output
```

## Fallback behavior

```text
Gemma disabled       -> deterministic summary
Gemma unreachable    -> deterministic summary + audit fallback warning
Gemma invalid JSON   -> deterministic summary + audit fallback warning
Gemma hallucination  -> deterministic summary + audit fallback warning
Gemma valid JSON     -> Gemma summary stored with summary_source=gemma_max
```

The capture pipeline stays robust: camera capture and evidence storage continue even if MAX is down.

## CLI

```bash
python -m monitor_me.cli --db data/events/monitorme.db llm-health --allow-unconfigured
python -m monitor_me.cli --db data/events/monitorme.db assistant-summarize-event <event_id>
python -m monitor_me.cli --db data/events/monitorme.db summaries --limit 20
```

## API

```text
GET  /assistant/llm/health
POST /assistant/events/{event_id}/summary
GET  /assistant/summaries
GET  /assistant/event-contracts
```

## Validation

```bash
./scripts/validate_node1_ai_camera_assistant_v02.sh
python -m pytest -q
```

Expected:

```text
Node1 AI Camera Assistant v0.2 validation PASSED
32 passed
```

## Node1 MAX helper scripts

The v0.2 source includes MonitorMe-local wrappers around the known-good Node1 MAX/Gemma workflow. They keep MAX installation isolated in the existing pixi quickstart project while making MonitorMe validation repeatable.

```bash
# TERM1: start MAX and keep it running
./scripts/max/term1_start_max_gemma3_1b.sh

# TERM2: validate standalone MAX server
./scripts/max/term2_validate_max_gemma3_1b.sh

# TERM2: validate MonitorMe strict JSON Gemma summaries against live MAX
./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
```

The TERM1 script uses `--sample-on-host` intentionally because this is the known-good RTX 5060 Ti workaround for the MAX nightly fused top-k sampler launch-resource crash path.

For details, see `docs/MAX_GEMMA_NODE1.md`.


## v0.2.2 MAX startup-script safety

MonitorMe v0.2.2 changes the MAX/Gemma helper scripts so they do **not** mutate the external pixi MAX quickstart environment by default. If `scripts/max/term1_start_max_gemma3_1b.sh` reports `unable to locate module 'std'`, run:

```bash
./scripts/max/diagnose_max_gemma3_1b_env.sh
```

That failure is a broken/mismatched MAX/Mojo pixi environment, usually after a rolling nightly package upgrade, not a MonitorMe assistant-code failure. Restore a known-good `pixi.lock` with `pixi install --locked` or intentionally rebuild the quickstart environment. Only set `MONITORME_MAX_PIXI_SYNC=1` when you explicitly want the helper to run `pixi add`.


## v0.2.8 MAX import-root cleanup

Node1 logs showed that `probe_mojo_std_nn_roots.sh` could import `max._kv_cache_ops`, but TERM1 later failed while `pixi run max --help` eagerly compiled another serving extension, `max._distributed_ops`, and could not resolve Mojo packages such as `std`, `comm`, or `layout`.

The MAX helpers now build a cleaner Mojo import path by preferring `$PREFIX/lib/mojo` when it contains `std.mojoc` / `nn.mojoc`, excluding false-positive directories such as `share/locale/nn` and `share/tabset/std`, and overriding stale inherited `MODULAR_MOJO_MAX_IMPORT_PATH` / `MOJO_PACKAGE_PATH` by default. TERM1 also skips `pixi run max --help` by default because that command can trigger optional serving-extension compilation before the actual server starts.

Useful retry sequence:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable_mojo"
export MONITORME_MAX_CLEAR_MOJO_CACHE=0
export MONITORME_MAX_OVERRIDE_MOJO_IMPORT_PATH=1
export MONITORME_MAX_SKIP_HELP_PREFLIGHT=1
./scripts/max/probe_mojo_std_nn_roots.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```

Set `MONITORME_MAX_SKIP_HELP_PREFLIGHT=0` only when intentionally debugging MAX CLI startup itself.
