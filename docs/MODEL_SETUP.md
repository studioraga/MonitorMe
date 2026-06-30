# MonitorMe model setup

MonitorMe does not commit model weights to Git. Download the YOLO ONNX model
into the repo-local `models/object_detection/` directory before enabling Step 17C
detector support.

## Download YOLO ONNX

From the repository root:

```bash
./scripts/models/download_yolo_onnx.sh
```

Default output:

```text
models/object_detection/yolo11n.onnx
```

The script writes MonitorMe detector variables to `.env`:

```bash
MONITORME_MODEL_DIR=models/object_detection
MONITORME_DETECTOR_MODEL_ID=yolo11n-coco-onnx
MONITORME_DETECTOR_MODEL_PATH=models/object_detection/yolo11n.onnx
MONITORME_DETECTOR_MODEL_URL=https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx
```

The script also logs to:

```text
results/models/download_yolo_onnx.log
```

## Force re-download

```bash
./scripts/models/download_yolo_onnx.sh --force
```

## Custom path or URL

```bash
./scripts/models/download_yolo_onnx.sh \
  --model-path models/object_detection/yolo11n.onnx \
  --url https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx
```

## Optional SHA256 verification

```bash
./scripts/models/download_yolo_onnx.sh \
  --sha256 <expected_sha256>
```

## Install detector dependencies after the model is present

```bash
python -m pip install -e '.[api,camera,detector,test]'
```

Then run live validation:

```bash
./scripts/validate_node1_c922_yolo_live.sh
```

## Privacy behavior

This script only downloads a public model file. It does not upload private CCTV
frames, events, manifests, or evidence packs.

## Git behavior

The repo keeps `models/object_detection/.gitkeep`, but ignores downloaded ONNX
weights:

```text
models/object_detection/*.onnx
```
