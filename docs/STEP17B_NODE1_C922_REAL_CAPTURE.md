# Step 17B: Node1 C922 Real Local Capture

Step 17B moves MonitorMe beyond seeded validation data. The runtime evidence path now begins with the real Logitech C922 attached to Node1 as `/dev/video0`.

## Pipeline

```text
/dev/video0
  -> OpenCV/V4L2 capture
  -> frame-difference motion gate
  -> keyframe artifact
  -> capture manifest
  -> capture_sessions row
  -> capture_artifacts rows
  -> normalized motion_detected event row
  -> audit_log records
  -> assistant answer / evidence pack / incident report
```

## What MonitorMe stores

For each bounded capture run MonitorMe stores:

- a `capture_sessions` row with `session_id`, camera, device, timing, frame counts, dataset path, manifest path, and policy decision;
- one `capture_artifacts` row for every real keyframe emitted by the motion gate;
- one `capture_artifacts` row for the real capture manifest;
- one normalized `events` row per motion event: `event_type=motion_detected`, `label=motion`, `frame_id=<real frame number>`;
- audit records for session creation/update, artifact creation, event insertion, and capture completion/failure.

## What MonitorMe does not do in Step 17B

MonitorMe does not fabricate object labels. It will not claim:

- person
- vehicle
- weapon
- identity
- face recognition
- suspicious behavior
- intent

Those claims require future real detector/VLM integrations that insert normalized evidence rows.

## Live command

```bash
./scripts/validate_node1_c922_live.sh
```

Or manually:

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

If no events appear, move in front of the camera or lower the threshold:

```bash
MONITORME_MOTION_THRESHOLD=0.5 ./scripts/validate_node1_c922_live.sh
```

## API command

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
