# MonitorMe

**MonitorMe** is a standalone Node1-local CCTV evidence assistant project.

It is separate from `cctv-ip-or-lan-usb-camera-ingest-ai-inference`. The old CCTV ingest repository was used only as design input. MonitorMe now targets your current hardware layout where the **Logitech C922 is attached directly to Node1 as `/dev/video0`**.

## Current implementation

This repo implements:

```text
Step 17A: evidence-first assistant foundation
Step 17B: real Node1 C922 local capture pipeline
Step 17C: optional real YOLO ONNX detection after motion gate
```

From this point onward, the runtime path does **not** rely on seeded demo CCTV events. It captures from the real local camera, writes local artifacts, inserts normalized SQLite evidence rows, and answers only from those records.

## Goal

MonitorMe answers CCTV questions only when every answer can be backed by local evidence:

- `event_id`
- `session_id`
- `frame_id`
- normalized event rows
- local artifact paths
- model metadata when a model was actually used
- policy decision records
- audit records
- evidence-pack paths
- incident report paths

The assistant is not allowed to invent facts. It does not upload private CCTV frames or event data to external services.

## Step 17C architecture

```text
Logitech C922 on Node1
        |
        v
/dev/video0  (V4L2 / MJPG)
        |
        v
OpenCV local capture runner
        |
        v
Frame-difference motion gate
        |
        +--> real keyframe artifact: data/captures/<session_id>/keyframes/*.jpg
        +--> real capture manifest: data/captures/<session_id>/manifest.json
        +--> normalized SQLite row: event_type=motion_detected, label=motion
        |
        +--> optional real YOLO ONNX detector, enabled only when configured
              |
              +--> normalized child rows: event_type=object_detected
                   label=person or vehicle/etc.
                   parent_event_id=<motion_event_id>
                   model_id=yolo11n-coco-onnx
                   artifact_id=<keyframe_artifact_id>
        |
        v
MonitorMe evidence DB
        |
        +--> capture_sessions
        +--> capture_artifacts
        +--> events
        +--> model_registry
        +--> audit_log
        +--> assistant_runs / assistant_summaries
        +--> evidence_packs / incident_reports / feedback
        |
        v
MonitorMe assistant + FactGuard
        |
        v
evidence-backed answer with event/session/frame/artifact/policy/audit refs
```

Important: Step 17C still does **not** fabricate `person`, `vehicle`, weapon, identity, or intent labels. Object labels appear only when a real enabled YOLO ONNX detector returns them. If the detector is missing or fails, MonitorMe stores the parent motion event and an audit warning, but no object rows.

## Quick start

```bash
cd MonitorMe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[api,camera,test]'

./scripts/validate_step17c_monitorme.sh
```

Expected final line:

```text
=== MonitorMe Step 17C validation PASSED ===
```

That validation does not seed demo CCTV data and does not require a physical camera or ONNX model. It unit-tests the real capture path and proves the normalized object-detection child-row path with an injected fake detector.

## Real Node1 C922 live validation

Your C922 has already shown `/dev/video0` with MJPG/YUYV formats. Use `/dev/video0` first.

```bash
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
source .venv/bin/activate
python -m pip install -e '.[api,camera,test]'

./scripts/validate_node1_c922_live.sh
```

Move in front of the camera during the capture window. If no motion events are emitted, lower the threshold:

```bash
MONITORME_MOTION_THRESHOLD=0.5 ./scripts/validate_node1_c922_live.sh
```

## CLI usage

Initialize DB and default model metadata:

```bash
python -m monitor_me.cli --db data/events/monitorme.db init-db
```

List local camera devices:

```bash
python -m monitor_me.cli camera-devices --probe
```

Run a real C922 capture from `/dev/video0`:

```bash
python -m monitor_me.cli --db data/events/monitorme.db capture-run \
  --camera-id c922_node1_gate \
  --device /dev/video0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --fourcc MJPG \
  --duration-sec 10 \
  --motion-threshold 1.5
```

Download the YOLO ONNX model, then run real C922 capture with YOLO enabled after the motion gate:

```bash
./scripts/models/download_yolo_onnx.sh
python -m pip install -e '.[api,camera,detector,test]'

python -m monitor_me.cli --db data/events/monitorme.db capture-run \
  --camera-id c922_node1_gate \
  --device /dev/video0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --fourcc MJPG \
  --duration-sec 10 \
  --motion-threshold 1.5 \
  --detector-enabled \
  --detector-model-path models/object_detection/yolo11n.onnx \
  --detector-conf-threshold 0.35
```

List normalized motion events:

```bash
python -m monitor_me.cli --db data/events/monitorme.db events \
  --event-type motion_detected \
  --limit 20
```


List normalized object detections:

```bash
python -m monitor_me.cli --db data/events/monitorme.db events \
  --event-type object_detected \
  --limit 20
```

Ask a grounded assistant question:

```bash
python -m monitor_me.cli --db data/events/monitorme.db ask "What motion events happened today?"
```

Build an evidence pack for a real event:

```bash
python -m monitor_me.cli --db data/events/monitorme.db evidence-pack <event_id>
```

Generate an incident report:

```bash
python -m monitor_me.cli --db data/events/monitorme.db incident-report \
  --event-id <event_id> \
  --title "Gate motion event" \
  --severity info
```

Mark feedback:

```bash
python -m monitor_me.cli --db data/events/monitorme.db feedback <event_id> \
  --label false_positive \
  --reason "operator review"
```

## Optional API server

Foreground mode:

```bash
MONITORME_DB=data/events/monitorme.db ./scripts/run_api.sh
```

Background mode:

```bash
./scripts/start_api_background.sh
./scripts/status_api.sh
```

Stop background server:

```bash
./scripts/stop_api.sh
```

Default local endpoint:

```text
http://127.0.0.1:8088
```

Core routes:

```text
GET  /
GET  /health
GET  /camera/devices
POST /camera/capture/run
GET  /events
GET  /models
POST /models/register-defaults
POST /assistant/ask
POST /assistant/events/{event_id}/evidence-pack
POST /assistant/reports/incident
POST /events/{event_id}/feedback
GET  /trackers/false-positives
```

Example real capture API request:

```bash
curl -sS -X POST http://127.0.0.1:8088/camera/capture/run \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id":"c922_node1_gate",
    "device":"/dev/video0",
    "width":1280,
    "height":720,
    "fps":30,
    "fourcc":"MJPG",
    "duration_sec":10,
    "motion_threshold":1.5
  }' | python3 -m json.tool
```


Example real capture API request with YOLO ONNX enabled:

```bash
curl -sS -X POST http://127.0.0.1:8088/camera/capture/run \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id":"c922_node1_gate",
    "device":"/dev/video0",
    "width":1280,
    "height":720,
    "fps":30,
    "fourcc":"MJPG",
    "duration_sec":10,
    "motion_threshold":1.5,
    "detector_enabled":true,
    "detector_model_path":"models/object_detection/yolo11n.onnx",
    "detector_conf_threshold":0.35
  }' | python3 -m json.tool
```

## Privacy rules

MonitorMe follows these default rules:

1. Local evidence first.
2. No external upload of private frames/events.
3. No face recognition.
4. No identity claim.
5. No weapon, intent, or behavior claim unless normalized evidence explicitly exists.
6. No fabricated object labels.
7. Every answer must expose event/session/frame references or clearly say that local evidence is missing.

## Repository layout

```text
MonitorMe/
  monitor_me/
    assistant.py               DB-grounded assistant orchestration
    db.py                      SQLite migrations and data access
    event_tools.py             CCTV event query/evidence helpers
    evidence_pack.py           event/session/model/policy/audit evidence packs
    fact_guard.py              hallucination and unsupported-claim guard
    local_capture.py           real Node1 /dev/video0 capture + motion/YOLO evidence
    yolo_onnx.py               optional local YOLO ONNX detector/postprocess
    llm_client.py              Null/Fake LLM clients; future Gemma/MAX hook
    model_registry.py          detector/text/future-model metadata
    report_tools.py            incident report generation
    routes.py                  optional FastAPI app factory
    tracker_tools.py           false-positive/useful feedback tracking
  migrations/
  scripts/
    validate_step17c_monitorme.sh
    validate_step17b_monitorme.sh
    validate_node1_c922_live.sh
    validate_node1_c922_yolo_live.sh
    run_api.sh
    start_api_background.sh
    status_api.sh
    stop_api.sh
  tests/
  docs/
  data/
    captures/
    events/
    evidence_packs/
    reports/
```

## Suggested commit message

```text
feat: add YOLO ONNX object detection after Node1 motion gate

Add Step 17C optional YOLO ONNX detection after the real Node1 motion gate.
When enabled with a real ONNX model, MonitorMe runs local detection on motion
keyframes only and inserts normalized object_detected child rows with model_id,
confidence, bbox, parent_event_id, keyframe artifact_id, and audit trail. Keep
motion capture resilient when the detector is disabled, missing, or fails, and
add CLI/API detector controls, docs, live validation script, and tests proving no
object labels are fabricated.
```


## Step 17D detector health

MonitorMe v0.1.9 adds `python -m monitor_me.cli detector-health` and `GET /models/detector/health` so Node1 can validate the YOLO ONNX model path, checksum, ONNX Runtime providers, and model input/output metadata before live camera capture. See `docs/STEP17D_DETECTOR_HEALTH.md`.


## Step 17E: evidence visualization overlays

MonitorMe now writes annotated keyframes as separate derived evidence artifacts when YOLO detections are emitted after the motion gate. The original raw keyframe remains unchanged and remains the artifact linked by the `object_detected` rows. The overlay is registered as `artifact_type=annotated_keyframe` and includes rendered labels for `label`, `confidence`, `event_id`, `parent_event_id`, and `model_id`.

Run offline validation:

```bash
./scripts/validate_step17e_monitorme.sh
```

Run live C922 + YOLO validation:

```bash
./scripts/models/download_yolo_onnx.sh
python -m pip install -e '.[api,camera,detector,test]'
./scripts/validate_node1_c922_yolo_live.sh
```

List overlay artifacts:

```bash
python -m monitor_me.cli --db data/events/monitorme.db artifacts \
  --artifact-type annotated_keyframe \
  --limit 20
```
