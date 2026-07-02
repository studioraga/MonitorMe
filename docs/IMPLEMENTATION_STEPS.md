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
