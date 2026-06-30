from pathlib import Path

from monitor_me.camera_devices import camera_start_hint, list_video_devices


def test_list_video_devices_from_fake_dev_root(tmp_path: Path):
    (tmp_path / "video1").write_text("fake")
    (tmp_path / "video0").write_text("fake")
    (tmp_path / "notvideo").write_text("fake")

    devices = list_video_devices(tmp_path)

    assert [Path(item["device"]).name for item in devices] == ["video0", "video1"]
    assert "Multiple /dev/video* nodes" in camera_start_hint(devices)
