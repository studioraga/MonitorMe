# Step 17D: Detector health and grounded object-query validation

Step 17D adds a pre-flight detector health check before live YOLO capture.
It does not open the camera and does not upload frames. It validates only local
model/runtime state:

- ONNX model path exists
- model file size and SHA-256
- optional expected SHA-256 match
- ONNX Runtime availability and version
- available/selected execution providers
- optional ONNX session load
- model input/output metadata

## CLI

```bash
python -m monitor_me.cli detector-health \
  --model-id yolo11n-coco-onnx \
  --model-path models/object_detection/yolo11n.onnx
```

Use `--skip-load` for file/hash/runtime metadata without opening an ONNX session:

```bash
python -m monitor_me.cli detector-health \
  --model-path models/object_detection/yolo11n.onnx \
  --skip-load
```

Use `--allow-unhealthy` in scripts that should emit a report without failing:

```bash
python -m monitor_me.cli detector-health \
  --model-path results/models/missing.onnx \
  --skip-load \
  --allow-unhealthy
```

## API

```bash
curl -sS 'http://127.0.0.1:8088/models/detector/health?model_path=models/object_detection/yolo11n.onnx&load_model=true' \
  | python3 -m json.tool
```

## Assistant query planner fix

The live Step 17C logs showed that YOLO produced real `person`, `chair`, and
`bed` rows, but the validation prompt asked:

```text
What person and vehicle events happened today?
```

In v0.1.7, any multi-label question with `and` was treated as strict
co-occurrence. In v0.1.8, MonitorMe uses:

- `person + vehicle`, `same clip`, `same session`, `both` = strict correlation
- `person and vehicle events` = union of available person/vehicle events

So if a person exists but no vehicle exists, MonitorMe returns the person
evidence and explicitly states that no local evidence was found for vehicle.
