import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from monitor_me.non_llm_gpu_lab import analyze_gray_frames_python, frame_to_gray_u8


def test_python_fallback_rejects_invalid_tile_contract():
    prev = np.zeros((16, 16), dtype=np.uint8)
    curr = prev.copy()

    with pytest.raises(ValueError, match=r"tile_cols \* tile_rows"):
        analyze_gray_frames_python(prev, curr, tile_cols=9, tile_rows=4)

    with pytest.raises(ValueError, match="pixel_threshold"):
        analyze_gray_frames_python(prev, curr, pixel_threshold=300)

    with pytest.raises(ValueError, match="dense_threshold"):
        analyze_gray_frames_python(prev, curr, dense_threshold=99)


def test_python_fallback_sparse_mask_contract():
    prev = np.full((240, 320), 10, dtype=np.uint8)
    curr = prev.copy()
    curr[15:30, 20:40] = 220
    curr[120:135, 160:180] = 200

    out = analyze_gray_frames_python(prev, curr)

    assert out["ok"] is True
    assert out["backend"] == "python_fallback"
    assert out["path"] == "sparse"
    assert out["tile_mask_hex"] == "0x00100001"
    assert out["active_tiles"] == 2


def test_frame_to_gray_u8_accepts_bgr_frame():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255

    gray = frame_to_gray_u8(bgr)

    assert gray.shape == (2, 2)
    assert gray.dtype == np.uint8
    assert int(gray[0, 0]) > 0


def test_native_phase0_selftest_if_built():
    binary = Path("native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab_selftest")
    if not binary.exists():
        pytest.skip("native Phase 0 selftest binary is not built")

    proc = subprocess.run([str(binary)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


def test_native_synthetic_json_has_timing_if_built():
    binary = Path("native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab")
    if not binary.exists():
        pytest.skip("native CPU binary is not built")

    proc = subprocess.run(
        [str(binary), "--mode", "synthetic", "--scenario", "mixed"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["frame"]["path"] == "mixed"
    assert "timing" in payload["frame"]
    assert payload["frame"]["timing"]["total_ms"] >= 0
    assert "timing" in payload["audio"]
