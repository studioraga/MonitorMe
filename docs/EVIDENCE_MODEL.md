# Evidence model

MonitorMe v0.1 uses SQLite as the local evidence source of truth.

## Main tables

| Table | Purpose |
| --- | --- |
| `cameras` | Node1 local camera registry |
| `model_registry` | Detector, text model, and future model metadata |
| `capture_sessions` | Local capture session metadata and policy decision |
| `capture_artifacts` | Keyframe, clip, manifest, report, or other local artifacts |
| `events` | Normalized event and detection facts |
| `assistant_runs` | Assistant question/answer execution records |
| `assistant_summaries` | Evidence-grounded summaries |
| `evidence_packs` | Generated evidence bundle records |
| `incident_reports` | Generated incident reports |
| `event_feedback` | Operator useful/false-positive labels |
| `audit_log` | Audit trail for writes and assistant activity |

## Normalized events

A motion event is a parent row:

```text
event_type = motion_detected
label      = motion
session_id = sess_...
frame_id   = 123
```

Each object detection is a child row:

```text
event_type      = object_detected
label           = person
confidence      = 0.87
bbox_json       = [0.15, 0.20, 0.42, 0.88]
session_id      = sess_...
frame_id        = 123
parent_event_id = evt_...
model_id        = yolo11n-coco-onnx
```

This makes questions like `What person events happened today?` simple and reliable.

## Evidence pack files

For an event, MonitorMe writes:

```text
event.json
related_events.json
capture_session.json
artifacts.json
model_metadata.json
policy_decision.json
audit.json
assistant_summary.json
manifest.json
report.md
```

The manifest records file sizes and SHA-256 hashes.
