# Node1 local camera roadmap

MonitorMe v0.1 creates the evidence and assistant foundation first. The next implementation slice should connect this evidence model to the real C922 camera attached to Node1.

## Planned local camera path

```text
/dev/video0 C922
  -> local frame reader
  -> motion gate
  -> YOLO detector
  -> capture session writer
  -> keyframe/clip artifact writer
  -> MonitorMeDB.insert_motion_with_detections(...)
  -> assistant/evidence/report tools
```

## Why this is separate from v0.1

The assistant must be trustworthy before adding real camera runtime complexity. Once the database contract is stable, any detector can write the same normalized event rows.

## Required next modules

```text
monitor_me/camera/local_v4l2.py
monitor_me/detectors/motion.py
monitor_me/detectors/yolo_onnx.py
monitor_me/capture/session_writer.py
```

## Non-blocking rule

Camera ingest must not wait for LLM/VLM calls. Assistant work should run after event insertion or in a worker queue.


## v0.1.1 API camera-device check

When the API is running:

```bash
MONITORME_DB=data/events/monitorme.db ./scripts/run_api.sh
```

Open these endpoints from another terminal:

```bash
curl -sS http://127.0.0.1:8088/ | python3 -m json.tool
curl -sS http://127.0.0.1:8088/health | python3 -m json.tool
curl -sS http://127.0.0.1:8088/camera/devices | python3 -m json.tool
```

If Node1 shows:

```text
/dev/video0
/dev/video1
```

that is normal for many UVC webcams such as Logitech C922. Confirm the capture node with:

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
v4l2-ctl --device=/dev/video1 --list-formats-ext
```

Pick the node that shows real video formats such as MJPEG or YUYV at 1280x720 or 1920x1080. In many C922 setups this is `/dev/video0`, while `/dev/video1` may be metadata/secondary.
