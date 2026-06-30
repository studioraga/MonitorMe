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
