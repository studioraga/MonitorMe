from __future__ import annotations

from pathlib import Path

from monitor_me.detector_health import check_detector_health


def test_detector_health_reports_missing_model_without_throwing(tmp_path):
    result = check_detector_health(model_path=tmp_path / "missing.onnx", load_model=False)

    assert result["exists"] is False
    assert result["ok"] is False
    assert result["load"]["requested"] is False
    assert result["privacy"]["camera_opened"] is False
    assert result["next_steps"]


def test_detector_health_checks_hash_without_loading(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not-a-real-onnx")

    result = check_detector_health(model_path=model, expected_sha256="bad", load_model=False)

    assert result["exists"] is True
    assert result["size_bytes"] == len(b"not-a-real-onnx")
    assert result["sha256"]
    assert result["sha256_matches"] is False
    assert result["ok"] is False
