# Node1 AI Camera Assistant v0.1

This milestone formalizes the MonitorMe Node1 assistant layer on top of the existing Node1 C922 + YOLO evidence pipeline.

It is intentionally **deterministic**. Gemma/MAX is not used yet. The purpose of v0.1 is to make the facts, policy decisions, event contracts, and assistant summaries stable before adding a local LLM in v0.2.

## Correct model split

```text
YOLO11n ONNX
  role: fast visual facts
  output: class label, confidence, normalized bbox, model_id, frame_id

Node1 deterministic policy
  role: action/severity decision
  output: record_evidence_only, record_motion_only, request_capture_review

Assistant summary service
  role: safe operator text from stored SQLite facts
  output: assistant_summaries rows, incident-report context, /assistant/ask evidence text

Gemma 3 1B via MAX
  role: future v0.2 explanation/Q&A layer
  rule: consumes event contracts only; never raw frames; never decides safety-critical actions

Qwen VLM / SmolVLM2 / SAM 2 / Grounding DINO / CLIP
  role: future optional evidence-enrichment layers after motion/YOLO trigger
```

## End-to-end runtime view

```text
Logitech C922 /dev/video0
  |
  v
OpenCV/V4L2 capture loop
  |
  v
Frame-difference motion gate
  |
  +-- no motion -----------------------------> continue watching
  |
  v
motion_detected parent event
  |
  v
raw keyframe artifact
  |
  v
YOLO ONNX visual-facts client
  |
  +-- no object -----------------------------> motion-only summary
  |
  v
object_detected child events
  |
  v
annotated overlay artifact
  |
  v
event_contracts row
  |
  v
deterministic capture/action policy
  |
  v
assistant_summaries row
  |
  +--> /assistant/ask over event DB
  +--> incident report endpoint
  +--> evidence pack builder
```

## New implementation files

```text
monitor_me/yolo_client.py
  Thin visual-facts boundary over YOLO ONNX detections. This prevents the camera runner from being tightly coupled to a specific detector implementation and prepares the code for later enrichment clients.

monitor_me/event_contract.py
  Builds a strict JSON event contract from the normalized SQLite event rows, child detections, artifacts, model metadata, and privacy flags.

monitor_me/capture_policy.py
  Applies deterministic Node1 policy. v0.1 does not let an LLM decide actions.

monitor_me/assistant_summary.py
  Creates DB-grounded summaries automatically after motion/YOLO events and stores them in assistant_summaries.

migrations/002_node1_assistant_v01.sql
  Adds event_contracts.

tests/test_node1_ai_camera_assistant_v01.py
  Proves event contracts, deterministic policy, automatic summaries, DB-backed assistant answers, incident report summary inclusion, and non-invention behavior.
```

## Event contract schema

Each summarized motion/object event receives a stored contract in `event_contracts`.

```json
{
  "schema_version": "1.0",
  "contract_type": "monitorme.node1_ai_camera_event",
  "event_id": "evt_...",
  "motion_event_id": "evt_...",
  "source_node": "node1",
  "source_kind": "local_v4l2",
  "camera_id": "c922_node1_gate",
  "session_id": "sess_...",
  "event_type": "object_detected",
  "frame_id": 26,
  "label": "person",
  "confidence": 0.90,
  "bbox_xyxy_norm": [0.23, 0.09, 0.77, 0.98],
  "detections": [
    {
      "event_id": "evt_...",
      "class_name": "person",
      "raw_label": "person",
      "canonical_label": "person",
      "confidence": 0.90,
      "bbox_xyxy_norm": [0.23, 0.09, 0.77, 0.98],
      "model_id": "yolo11n-coco-onnx",
      "artifact_id": "art_..."
    }
  ],
  "artifacts": [
    {
      "artifact_id": "art_...",
      "artifact_type": "keyframe",
      "path": "data/captures/sess_.../keyframes/frame_000026.jpg",
      "sha256": "..."
    },
    {
      "artifact_id": "art_...",
      "artifact_type": "annotated_keyframe",
      "path": "data/captures/sess_.../overlays/frame_000026_overlay.jpg",
      "sha256": "..."
    }
  ],
  "privacy": {
    "external_upload": false,
    "face_recognition": false,
    "raw_frame_upload": false,
    "identity_claim": false,
    "intent_claim": false
  }
}
```

## Deterministic policy

v0.1 policy is intentionally simple and explainable:

```text
person confidence >= 0.60
  -> action=request_capture_review
  -> severity=review
  -> duration_sec=90
  -> reason="person confidence ... >= 0.60; review local capture evidence"

object evidence but no person trigger
  -> action=record_evidence_only
  -> severity=info

motion only
  -> action=record_motion_only
  -> severity=info
```

Policy output is stored as `policy_decision_json` in `event_contracts` and copied into assistant summary facts. Gemma v0.2 may explain this policy result, but must not create it.

## Assistant summary behavior

After a motion/YOLO event group is inserted, MonitorMe creates deterministic summaries automatically.

```text
object_detected row inserted
  -> event contract built
  -> deterministic policy evaluated
  -> assistant_summaries row inserted
  -> audit row added
```

Example summary:

```text
Node1 camera c922_node1_gate recorded object_detected evidence at frame_id=26 with labels person=1; highest confidence=0.90. event_id=evt_... session_id=sess_.... Deterministic policy action=request_capture_review because person confidence 0.90 >= 0.60; review local capture evidence.
```

## CLI commands

```bash
python -m monitor_me.cli --db data/events/monitorme.db summaries --limit 20
python -m monitor_me.cli --db data/events/monitorme.db event-contracts --limit 20
python -m monitor_me.cli --db data/events/monitorme.db assistant-summarize-event <event_id>
python -m monitor_me.cli --db data/events/monitorme.db ask "What person events happened today?"
python -m monitor_me.cli --db data/events/monitorme.db ask "What person and guitar events happened today?"
```

## API routes

```text
POST /assistant/events/{event_id}/summary
GET  /assistant/summaries
GET  /assistant/event-contracts
POST /assistant/ask
POST /assistant/reports/incident
```

## Validation

Offline assistant validation:

```bash
./scripts/validate_node1_ai_camera_assistant_v01.sh
```

Live Node1 C922 + YOLO validation:

```bash
./scripts/models/download_yolo_onnx.sh
python -m pip install -e '.[api,camera,detector,test]'
./scripts/validate_node1_c922_yolo_live.sh
```

Inspect DB safely:

```bash
sqlite3 data/events/monitorme.db ".tables"
sqlite3 data/events/monitorme.db "select event_type,label,count(*) from events group by event_type,label;"
sqlite3 data/events/monitorme.db "select event_id,substr(summary_text,1,120) from assistant_summaries limit 5;"
sqlite3 data/events/monitorme.db "select event_id,json_extract(policy_decision_json,'$.action') from event_contracts limit 5;"
```

Do not use `cat data/events/monitorme.db`; it is a binary SQLite database file.

## Safety limits

- No face recognition.
- No identity claim.
- No intent claim.
- No weapon claim unless there is normalized local evidence for that exact label.
- No external upload.
- No raw frames are sent to Gemma in v0.1.
- No LLM participates in deterministic policy decisions in v0.1.

## What v0.2 should add

```text
Gemma/MAX OpenAI-compatible client
strict JSON prompt contract
strict JSON output parser
summary validator
fallback to deterministic summaries
MAX/Gemma health check
more tests proving Gemma output cannot introduce unsupported facts
```
