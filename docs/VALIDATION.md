# Validation

## Standard validation

```bash
./scripts/validate_step17b_monitorme.sh
```

This runs the unit/integration tests and does not seed demo CCTV data.

Expected:

```text
=== MonitorMe Step 17B validation PASSED ===
```

## Physical Node1 C922 validation

Run on Node1 with the Logitech C922 connected as `/dev/video0`:

```bash
./scripts/validate_node1_c922_live.sh
```

Move in front of the camera during the capture window. If no motion is emitted:

```bash
MONITORME_MOTION_THRESHOLD=0.5 ./scripts/validate_node1_c922_live.sh
```


## Test dependency note

The `test` extra includes `httpx2>=2.5.0` because the current Starlette/FastAPI TestClient stack requires it. Run `python -m pip install -e '.[api,camera,test]'` before validation.

## YOLO model download script validation

Standard validation also checks that `scripts/models/download_yolo_onnx.sh` is present, shows help, and can persist MonitorMe detector variables without requiring network access when a model file already exists.

For real model download on Node1:

```bash
./scripts/models/download_yolo_onnx.sh
python -m pip install -e '.[api,camera,detector,test]'
./scripts/validate_node1_c922_yolo_live.sh
```


## Step 17D detector health

MonitorMe v0.1.8 adds `python -m monitor_me.cli detector-health` and `GET /models/detector/health` so Node1 can validate the YOLO ONNX model path, checksum, ONNX Runtime providers, and model input/output metadata before live camera capture. See `docs/STEP17D_DETECTOR_HEALTH.md`.


## Step 17E

```bash
./scripts/validate_step17e_monitorme.sh
```

This validates that annotated overlay artifacts are generated from normalized `object_detected` rows while raw keyframes remain unchanged.
