from __future__ import annotations

from pathlib import Path

import numpy as np

from monitor_me.db import MonitorMeDB
from monitor_me.local_capture import IterableFrameSource, LocalCameraCaptureRunner, LocalCaptureConfig


def motion_frames() -> list[np.ndarray]:
    frame0 = np.zeros((120, 160, 3), dtype=np.uint8)
    frame1 = frame0.copy()
    frame1[30:90, 50:110] = 255
    frame2 = frame1.copy()
    return [frame0, frame1, frame2]


def make_real_motion_capture(tmp_path: Path):
    db = MonitorMeDB(tmp_path / "monitorme.db")
    config = LocalCaptureConfig(
        camera_id="c922_node1_gate",
        device="/dev/video0",
        width=160,
        height=120,
        fps=30,
        duration_sec=5.0,
        max_frames=3,
        motion_threshold=1.0,
        min_event_gap_sec=0.0,
        data_root=str(tmp_path / "data"),
    )
    result = LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(motion_frames())).run()
    events = db.list_events(event_type="motion_detected", label="motion")
    assert result.ok
    assert events
    return db, result, events[0]
