# Step 17E: Evidence visualization overlays

Step 17E adds operator-friendly annotated keyframes while preserving the original raw keyframe as evidence.

## Evidence rule

- Raw keyframes are written under `data/captures/<session_id>/keyframes/`.
- `object_detected` rows continue to point to the raw keyframe artifact via `artifact_id`.
- Annotated overlays are written under `data/captures/<session_id>/overlays/`.
- Overlay artifacts are registered separately as `artifact_type=annotated_keyframe`.
- The overlay is a derived local artifact; it is not a replacement for the raw evidence.

## Overlay labels

Each rendered box includes:

- `label`
- `confidence`
- `event_id`
- `parent_event_id`
- `model_id`

These are exactly the normalized facts MonitorMe is allowed to cite.

## Capture result fields

Step 17E capture results include:

```json
{
  "overlay_artifact_ids": ["art_..."],
  "overlay_paths": ["data/captures/sess_.../overlays/frame_000049_overlay.jpg"]
}
```

## Manifest fields

Each motion keyframe record may include:

```json
{
  "overlay_enabled": true,
  "overlay_artifact_id": "art_...",
  "overlay_path": "data/captures/sess_.../overlays/frame_000049_overlay.jpg",
  "overlay_boxes": [
    {
      "label": "person",
      "confidence": 0.88,
      "event_id": "evt_...",
      "parent_event_id": "evt_...",
      "model_id": "yolo11n-coco-onnx",
      "bbox": [0.25, 0.01, 0.76, 0.99]
    }
  ]
}
```

## CLI

Run capture with overlays enabled, which is the default:

```bash
python -m monitor_me.cli --db data/events/monitorme.db capture-run \
  --camera-id c922_node1_gate \
  --device /dev/video0 \
  --width 1280 --height 720 --fps 30 --fourcc MJPG \
  --duration-sec 10 --motion-threshold 1.5 \
  --detector-enabled \
  --detector-model-path models/object_detection/yolo11n.onnx
```

Disable overlays for a minimal run:

```bash
python -m monitor_me.cli --db data/events/monitorme.db capture-run \
  --detector-enabled --no-overlays
```

List overlays:

```bash
python -m monitor_me.cli --db data/events/monitorme.db artifacts \
  --artifact-type annotated_keyframe \
  --limit 20
```

## API

`POST /camera/capture/run` accepts:

```json
{
  "detector_enabled": true,
  "overlay_enabled": true,
  "overlay_dir_name": "overlays"
}
```

List overlays:

```bash
curl -sS 'http://127.0.0.1:8088/artifacts?artifact_type=annotated_keyframe&limit=20' | python3 -m json.tool
```

## Validation

```bash
./scripts/validate_step17e_monitorme.sh
./scripts/validate_node1_c922_yolo_live.sh
```
