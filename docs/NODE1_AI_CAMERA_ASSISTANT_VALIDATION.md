# Node1 AI Camera Assistant v0.1 validation guide

This guide documents the exact validation flow for the Node1 AI Camera Assistant v0.1 milestone.

## 1. Install base test/runtime dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[api,camera,test]'
```

## 2. Run offline assistant validation

```bash
./scripts/validate_node1_ai_camera_assistant_v01.sh
```

Expected:

```text
=== MonitorMe Node1 AI Camera Assistant v0.1 validation PASSED ===
```

This validation uses fake injected frames/detections. It does not need a physical camera, ONNX Runtime, YOLO model, Gemma/MAX, network access, or external services.

It validates:

```text
event_contracts table creation
deterministic capture/action policy
automatic assistant summaries
DB-grounded /assistant/ask behavior
incident-report summary inclusion
FactGuard non-invention behavior
person/guitar queries refuse unsupported labels
weapon/identity/intent claims remain blocked
```

## 3. Download YOLO model for live Node1 validation

```bash
./scripts/models/download_yolo_onnx.sh
```

Expected model path:

```text
models/object_detection/yolo11n.onnx
```

The script writes MonitorMe variables to `.env` and logs to `results/models/download_yolo_onnx.log`.

## 4. Install detector support

```bash
python -m pip install -e '.[api,camera,detector,test]'
```

## 5. Run live Node1 C922 + YOLO validation

```bash
./scripts/validate_node1_c922_yolo_live.sh
```

Expected runtime flow:

```text
/dev/video0 C922 opened
YOLO detector health OK
motion_detected rows inserted
object_detected rows inserted when YOLO sees objects
raw keyframe artifacts written
annotated overlay artifacts written
assistant_summaries rows written automatically
event_contracts rows written automatically
assistant answers cite event_id/session_id/frame_id/model_id/artifact paths
```

## 6. Inspect generated artifacts

```bash
ls data/reports/
ls -l data/captures/<session_id>/keyframes/frame_000*
ls -l data/captures/<session_id>/overlays/frame_000*
ls -l data/events/monitorme.db
```

Use `sqlite3`, not `cat`, for DB inspection:

```bash
sqlite3 data/events/monitorme.db ".tables"
sqlite3 data/events/monitorme.db "select event_type,label,count(*) from events group by event_type,label;"
sqlite3 data/events/monitorme.db "select artifact_type,count(*) from capture_artifacts group by artifact_type;"
sqlite3 data/events/monitorme.db "select count(*) from assistant_summaries;"
sqlite3 data/events/monitorme.db "select count(*) from event_contracts;"
sqlite3 data/events/monitorme.db "select event_id,substr(summary_text,1,160) from assistant_summaries order by created_at desc limit 5;"
```

`cat data/events/monitorme.db` prints binary SQLite bytes and is not a useful validation command.

## 7. Ask grounded assistant questions

```bash
python -m monitor_me.cli --db data/events/monitorme.db ask "What person events happened today?"
python -m monitor_me.cli --db data/events/monitorme.db ask "What person and guitar events happened today?"
```

Expected safety behavior:

```text
If person evidence exists, the assistant cites person event IDs.
If guitar evidence does not exist, the assistant says no local evidence for guitar.
The assistant does not infer identity, intent, weapon, or suspicious behavior.
```

## 8. Create/report evidence

```bash
python -m monitor_me.cli --db data/events/monitorme.db evidence-pack <event_id>
python -m monitor_me.cli --db data/events/monitorme.db incident-report --event-id <event_id> --title "Node1 evidence review" --severity review
```

Incident reports include deterministic evidence plus assistant summaries when available.
