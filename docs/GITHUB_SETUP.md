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
