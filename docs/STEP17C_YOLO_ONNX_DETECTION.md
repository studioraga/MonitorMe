# Step 17C — Real YOLO ONNX Detection After Motion Gate

Step 17C adds optional real object detection to the MonitorMe Node1 local camera pipeline.

The pipeline remains evidence-first and non-fabricating:

```text
/dev/video0 C922 frame
  -> frame-difference motion gate
  -> parent event: motion_detected
  -> keyframe artifact
  -> optional YOLO ONNX detector on the motion keyframe
  -> child events: object_detected
  -> assistant answers from normalized SQLite rows only
```

## Safety rule

MonitorMe never inserts `person`, `vehicle`, or any other object label unless a real enabled detector returns that detection. If the ONNX model is missing, onnxruntime is not installed, or inference fails, MonitorMe still stores the parent `motion_detected` event and writes an audit record such as `detector.unavailable` or `detector.run.failed`.

## Normalized object event rows

A real YOLO child event uses:

```text
event_type      = object_detected
label           = person | vehicle | other canonical label
confidence      = detector score
bbox_json       = normalized [x1, y1, x2, y2]
model_id        = yolo11n-coco-onnx
parent_event_id = parent motion_detected event_id
session_id      = capture session_id
frame_id        = motion keyframe frame_id
artifact_id     = keyframe artifact_id
source_node     = node1
source_kind     = local_v4l2
```

Vehicle-like COCO labels (`car`, `truck`, `bus`, `motorcycle`, `bicycle`, `train`, `boat`) are stored as canonical `label="vehicle"` with the exact model class in `attrs.raw_label`.

## Install detector dependencies

```bash
python -m pip install -e '.[api,camera,detector,test]'
```

The detector extra installs `onnxruntime`. If you want GPU execution, install the appropriate `onnxruntime-gpu` package for your CUDA stack manually, then verify providers in Python.

## Place the ONNX model

Default expected path:

```text
models/object_detection/yolo11n.onnx
```

Recommended setup before installing detector support:

```bash
./scripts/models/download_yolo_onnx.sh
python -m pip install -e '.[api,camera,detector,test]'
```

See `docs/MODEL_SETUP.md` for download options, SHA256 verification, and `.env` persistence.

You can override it:

```bash
export MONITORME_DETECTOR_MODEL_PATH=/absolute/path/to/yolo11n.onnx
```

## CLI capture with detector enabled

```bash
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

Then inspect object events:

```bash
python -m monitor_me.cli --db data/events/monitorme.db events \
  --event-type object_detected \
  --limit 20
```

Ask the assistant:

```bash
python -m monitor_me.cli --db data/events/monitorme.db ask \
  "What person and vehicle events happened today?"
```

## API capture with detector enabled

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

## Live validation

```bash
./scripts/validate_node1_c922_yolo_live.sh
```

The script requires:

- `/dev/video0` present
- a real ONNX model at `MONITORME_DETECTOR_MODEL_PATH`
- `onnxruntime` installed

## Offline validation

```bash
./scripts/validate_step17c_monitorme.sh
```

The offline validation does not require a camera or ONNX model. It uses an injected fake detector only to prove the object-event normalization path, parent/child references, model metadata, artifact references, and audit trail.


## Step 17D detector health

MonitorMe v0.1.8 adds `python -m monitor_me.cli detector-health` and `GET /models/detector/health` so Node1 can validate the YOLO ONNX model path, checksum, ONNX Runtime providers, and model input/output metadata before live camera capture. See `docs/STEP17D_DETECTOR_HEALTH.md`.
