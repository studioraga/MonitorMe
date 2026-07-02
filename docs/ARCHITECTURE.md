# MonitorMe architecture

MonitorMe is a fresh standalone project. It is not a patch on the previous CCTV ingest repository.

## Design principle

The assistant is useful only when it is evidence-backed. Therefore, the first implementation target is not a large LLM integration. The first target is a trusted evidence model that later LLM/VLM components can summarize safely.

## Current v0.1 flow

```text
local Node1 evidence producer
  -> MonitorMeDB
  -> normalized events/object detections
  -> evidence tools
  -> assistant answer planner
  -> FactGuard
  -> answer with references
```

## Future camera flow

```text
Node1 C922 /dev/video0
  -> motion gate
  -> YOLO detector
  -> capture session
  -> keyframe/clip artifacts
  -> normalized event rows
  -> MonitorMe assistant
```

## Important separation

```text
capture/event path = fast and deterministic
assistant path     = async or best effort
```

If an LLM/VLM later fails, event capture should still succeed.

## Core modules

- `monitor_me.db.MonitorMeDB`: schema authority and thread-safe SQLite access.
- `monitor_me.event_tools`: event query and evidence reference building.
- `monitor_me.assistant.MonitorMeAssistant`: question handling and grounded answer generation.
- `monitor_me.fact_guard.FactGuard`: rejects unsupported claims.
- `monitor_me.evidence_pack.EvidencePackBuilder`: produces reviewable evidence bundles.
- `monitor_me.report_tools.IncidentReportBuilder`: creates incident reports linked to evidence packs.
- `monitor_me.tracker_tools.TrackerTools`: operator feedback and false-positive tracker.
- `monitor_me.routes.create_app`: optional FastAPI API.

## Evidence-backed answer contract

Every positive answer should be traceable to at least one of:

```text
event_id
session_id
frame_id
camera_id
model_id
artifact path
policy decision
audit_id
evidence pack path
report path
```

Unsupported questions should return a clear evidence limit instead of speculation.

## Node1 AI Camera Assistant v0.1 architecture

```text
C922 /dev/video0
  -> local_capture.py
  -> motion gate
  -> yolo_client.py / yolo_onnx.py
  -> events table
  -> capture_artifacts table
  -> event_contract.py
  -> capture_policy.py
  -> event_contracts table
  -> assistant_summary.py
  -> assistant_summaries table
  -> assistant.py / report_tools.py / evidence_pack.py
```

The architecture intentionally separates model responsibilities:

```text
YOLO ONNX: visual facts only
Node1 policy: deterministic action/severity decision
Assistant summary: safe local text over stored facts
Gemma/MAX v0.2: optional explanation layer over event contracts, not raw frames
```
