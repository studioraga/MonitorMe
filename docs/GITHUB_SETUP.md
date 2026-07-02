# GitHub Setup

Suggested repository name:

```text
MonitorMe
```

Suggested commit:

```text
feat: add Node1 C922 real local capture evidence pipeline
```

Detailed commit body:

```text
Add Step 17B real local camera capture for MonitorMe. Capture bounded sessions
from Node1 /dev/video0 using OpenCV/V4L2 with the C922 MJPG profile, run a
local frame-difference motion gate, write real keyframe artifacts and capture
manifests, insert normalized motion_detected rows, update capture session stats,
and preserve policy/audit evidence. Add CLI/API capture controls, event listing,
Node1 live validation script, no-demo Step 17B validation, docs, and tests that
prove MonitorMe does not fabricate object labels or unsupported claims.
```

## Node1 AI Camera Assistant v0.1 commit hygiene

Before committing this milestone, make sure runtime artifacts are not staged:

```bash
git status --short
git status --ignored --short
```

Do not commit:

```text
.venv/
.env
*.db
*.onnx
data/captures/*
data/evidence_packs/*
data/reports/*
results/*
monitorme.egg-info/
.pytest_cache/
```

Do commit:

```text
monitor_me/yolo_client.py
monitor_me/event_contract.py
monitor_me/capture_policy.py
monitor_me/assistant_summary.py
migrations/002_node1_assistant_v01.sql
scripts/validate_node1_ai_camera_assistant_v01.sh
tests/test_node1_ai_camera_assistant_v01.py
docs/NODE1_AI_CAMERA_ASSISTANT_V0_1.md
docs/NODE1_AI_CAMERA_ASSISTANT_VALIDATION.md
README.md and updated docs/*.md
```
