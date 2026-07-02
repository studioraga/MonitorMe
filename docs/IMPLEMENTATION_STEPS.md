# Implementation Steps

## Completed: Step 17A foundation

- SQLite evidence schema
- normalized events
- model metadata
- audit trail
- evidence packs
- incident reports
- grounded assistant
- FactGuard

## Completed: Step 17B real Node1 C922 capture

- real `/dev/video0` capture through OpenCV/V4L2
- C922 MJPG defaults
- frame-difference motion gate
- real keyframe artifacts
- real capture manifest
- normalized `motion_detected` rows
- policy/audit records
- CLI/API capture controls
- no seeded demo event dependency

## Next: Step 17C local YOLO object rows

- configure real YOLO ONNX model path
- run object detection only after motion gate
- insert normalized `object_detected` child rows
- keep assistant grounded in event/session/frame/model/artifact evidence

## Node1 AI Camera Assistant v0.1 implementation checklist

Implemented in this milestone:

```text
[x] Add YOLO visual-facts client boundary
[x] Add event_contracts table and builder
[x] Add deterministic Node1 capture/action policy
[x] Auto-generate assistant_summaries after motion/YOLO event groups
[x] Expose summary and event-contract CLI commands
[x] Expose summary and event-contract API routes
[x] Include assistant summaries in incident report flow
[x] Add validation script for Assistant v0.1
[x] Add tests proving the assistant does not invent unsupported facts
```

Deferred to v0.2:

```text
[ ] Gemma/MAX OpenAI-compatible client
[ ] strict Gemma JSON output schema
[ ] Gemma output validator
[ ] fallback behavior when MAX is unavailable
[ ] MAX/Gemma health endpoint
```

---

## Node1 AI Camera Assistant v0.2 — Gemma/MAX strict JSON summaries

v0.2 adds an optional local Gemma 3 1B explanation layer served by MAX through an OpenAI-compatible endpoint.

```text
YOLO = fast visual facts
Node1 policy = deterministic decisions/actions
Gemma/MAX = explanation/Q&A over structured event facts only
```

Validation command:

```bash
./scripts/validate_node1_ai_camera_assistant_v02.sh
```

Gemma/MAX is disabled by default. If it is unavailable or returns invalid/unsupported JSON, MonitorMe stores the deterministic summary from v0.1 and records the fallback reason in summary facts and audit logs.

See `docs/NODE1_AI_CAMERA_ASSISTANT_V0_2.md` for the full prompt/output contract and Node1 setup.


## Node1 MAX/Gemma live validation helpers

MonitorMe includes helper scripts for the previously validated Node1 MAX + Gemma 3 1B path:

```bash
./scripts/max/term1_start_max_gemma3_1b.sh
./scripts/max/term2_validate_max_gemma3_1b.sh
./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
```

These scripts use the external pixi MAX quickstart project, preserve `--sample-on-host`, and validate that MonitorMe can accept a strict JSON Gemma summary over local event contracts. See `docs/MAX_GEMMA_NODE1.md`.


### Node1 AI Camera Assistant v0.3

```bash
./scripts/validate_node1_ai_camera_assistant_v03.sh
python -m pytest -q
```

Validates optional local Qwen VLM keyframe analysis after trigger, strict JSON validation, local-only guardrails, failed-analysis storage, and disabled-by-default behavior.
