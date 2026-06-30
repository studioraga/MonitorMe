from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any


def _video_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"video(\d+)$", path.name)
    return (int(match.group(1)) if match else 10_000, path.name)


def list_video_devices(dev_root: str | Path = "/dev", probe: bool = False) -> list[dict[str, Any]]:
    """Return discovered Linux V4L2 /dev/video* device nodes.

    This function is intentionally lightweight and safe for MonitorMe v0.1:
    it only enumerates local device nodes by default. When ``probe=True`` and
    ``v4l2-ctl`` is installed, it also captures a bounded format summary.
    """
    root = Path(dev_root)
    devices: list[dict[str, Any]] = []
    for device in sorted(root.glob("video*"), key=_video_sort_key):
        if not re.fullmatch(r"video\d+", device.name):
            continue
        try:
            st = device.stat()
            is_char = stat.S_ISCHR(st.st_mode)
            major = os.major(st.st_rdev) if is_char else None
            minor = os.minor(st.st_rdev) if is_char else None
        except OSError as exc:
            devices.append({"device": str(device), "ok": False, "error": str(exc)})
            continue

        item: dict[str, Any] = {
            "device": str(device),
            "ok": True,
            "is_character_device": is_char,
            "major": major,
            "minor": minor,
        }
        if probe:
            item["v4l2"] = probe_v4l2_device(device)
        devices.append(item)
    return devices


def probe_v4l2_device(device: str | Path, timeout_sec: float = 5.0) -> dict[str, Any]:
    """Probe a V4L2 device with v4l2-ctl when available.

    The raw output is intentionally truncated to keep API responses small.
    """
    binary = shutil.which("v4l2-ctl")
    if not binary:
        return {
            "ok": False,
            "reason": "v4l2-ctl not installed",
            "install_hint": "sudo apt install v4l-utils",
        }
    try:
        result = subprocess.run(
            [binary, f"--device={device}", "--list-formats-ext"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"v4l2-ctl timed out after {timeout_sec}s"}

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "summary": stdout[:6000],
        "stderr": stderr[:2000],
    }


def camera_start_hint(devices: list[dict[str, Any]]) -> str:
    if not devices:
        return "No /dev/video* devices were detected on this host."
    if len(devices) == 1:
        return f"Detected {devices[0]['device']}. Use it as MONITORME_CAMERA_DEVICE after confirming formats."
    return (
        "Multiple /dev/video* nodes detected. Logitech C922 commonly exposes more than one node. "
        "Use v4l2-ctl --list-devices and --list-formats-ext to confirm which node carries MJPEG/YUYV video; "
        "often /dev/video0 is the capture stream and /dev/video1 is metadata/secondary."
    )
