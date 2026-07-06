import json
import os
import subprocess
from pathlib import Path

import pytest


NATIVE_CPU = Path("native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab")
NATIVE_CUDA = Path("native/node1_non_llm_gpu_inference_lab/build/node1_non_llm_gpu_lab")
SELFTEST = Path("native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab_selftest")


def _require_native_cpu():
    if not NATIVE_CPU.exists():
        pytest.skip("native CPU binary is not built")


def _require_native_cuda():
    if not NATIVE_CUDA.exists():
        pytest.skip("native CUDA binary is not built")
    proc = subprocess.run(
        [
            str(NATIVE_CUDA),
            "--mode", "overlay-heavy-synthetic",
            "--scenario", "mixed",
            "--width", "64",
            "--height", "48",
            "--thumbnail-width", "16",
            "--thumbnail-height", "12",
            "--gpu",
            "--include-output",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode == 127 and "libcudart" in combined:
        pytest.skip("native CUDA binary exists but CUDA runtime library is not available in this environment")
    assert proc.returncode == 0, combined
    payload = json.loads(proc.stdout)
    if not payload.get("cuda_compiled"):
        pytest.skip("native binary is not CUDA compiled")
    if payload.get("overlay_heavy_cuda") is None:
        pytest.fail("CUDA native binary did not emit overlay_heavy_cuda for overlay-heavy-synthetic --gpu")


def test_phase6_selftest_if_built():
    if not SELFTEST.exists():
        pytest.skip("native Phase 6 selftest binary is not built")
    proc = subprocess.run([str(SELFTEST)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


@pytest.mark.parametrize(
    ("scenario", "frame_path"),
    [
        ("mixed", "mixed"),
        ("dense", "dense"),
        ("sparse", "sparse"),
    ],
)
def test_native_overlay_heavy_cpu_synthetic_emits_safe_overlay_artifacts(scenario, frame_path):
    _require_native_cpu()
    proc = subprocess.run(
        [
            str(NATIVE_CPU),
            "--mode", "overlay-heavy-synthetic",
            "--scenario", scenario,
            "--width", "64",
            "--height", "48",
            "--thumbnail-width", "16",
            "--thumbnail-height", "12",
            "--include-output",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["mode"] == "overlay-heavy-synthetic"
    assert payload["frame"]["path"] == frame_path
    overlay = payload["overlay_heavy"]
    assert overlay["ok"] is True
    assert overlay["backend"] == "cpu"
    assert overlay["schema"] == "node1_non_llm_overlay_heavy.v0.1"
    assert overlay["facts_only"] is True
    assert "identity" in overlay["note"]
    assert overlay["pixels_processed"] == 64 * 48
    assert overlay["heatmap_elements"] == overlay["pixels_processed"]
    assert overlay["overlay_rgb_elements"] == overlay["pixels_processed"] * 3
    assert overlay["thumbnail_rgb_elements"] == 16 * 12 * 3
    assert len(overlay["heatmap"]) == overlay["pixels_processed"]
    assert len(overlay["overlay_rgb"]) == overlay["pixels_processed"] * 3
    assert len(overlay["thumbnail_rgb"]) == 16 * 12 * 3
    assert overlay["before_after_max_diff"] == overlay["heatmap_max"]
    assert overlay["before_after_abs_mean"] == overlay["heatmap_mean"]
    assert overlay["changed_pixels"] == payload["frame"]["changed_pixels"]
    assert overlay["changed_ratio"] == payload["frame"]["changed_ratio"]
    assert overlay["bytes_read"] == overlay["pixels_processed"] * 2
    assert overlay["bytes_written"] == overlay["pixels_processed"] + overlay["pixels_processed"] * 3 + 16 * 12 * 3
    if scenario == "dense":
        assert overlay["changed_pixels"] == overlay["pixels_processed"]
        assert abs(overlay["lighting_delta"] - 200.0) <= 1e-9


@pytest.mark.parametrize("scenario", ["mixed", "dense", "sparse"])
def test_native_overlay_heavy_cuda_matches_cpu_for_synthetic_scenarios(scenario):
    _require_native_cuda()
    proc = subprocess.run(
        [
            str(NATIVE_CUDA),
            "--mode", "overlay-heavy-synthetic",
            "--scenario", scenario,
            "--width", "64",
            "--height", "48",
            "--thumbnail-width", "16",
            "--thumbnail-height", "12",
            "--gpu",
            "--include-output",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["cuda_compiled"] is True
    cpu = payload["overlay_heavy"]
    gpu = payload["overlay_heavy_cuda"]
    assert cpu["backend"] == "cpu"
    assert gpu["backend"] == "cuda"
    assert cpu["facts_only"] is True
    assert gpu["facts_only"] is True
    assert cpu["heatmap"] == gpu["heatmap"]
    assert cpu["overlay_rgb"] == gpu["overlay_rgb"]
    assert cpu["thumbnail_rgb"] == gpu["thumbnail_rgb"]
    comparison = payload["overlay_heavy_cpu_cuda_comparison"]
    assert comparison["schema"] == "node1_non_llm_overlay_heavy_cpu_cuda_compare.v0.1"
    assert comparison["ok"] is True
    assert comparison["heatmap_equal"] is True
    assert comparison["overlay_equal"] is True
    assert comparison["thumbnail_equal"] is True
    assert comparison["mismatch_count"] == 0
    assert comparison["max_abs_diff"] == 0
    assert comparison["metrics_close"] is True
    assert comparison["facts_only"] is True


def test_monitor_me_cli_gpu_lab_overlay_heavy_synthetic_if_native_cpu_built():
    _require_native_cpu()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CPU)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "0"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-overlay-heavy-synthetic",
            "--scenario", "mixed",
            "--width", "64",
            "--height", "48",
            "--thumbnail-width", "16",
            "--thumbnail-height", "12",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source"] == "native_binary"
    assert payload["overlay_heavy"]["schema"] == "node1_non_llm_overlay_heavy.v0.1"
    assert payload["overlay_heavy"]["backend"] == "cpu"


def test_monitor_me_cli_gpu_lab_overlay_heavy_synthetic_uses_cuda_when_binary_is_cuda_built():
    _require_native_cuda()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CUDA)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "1"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-overlay-heavy-synthetic",
            "--scenario", "mixed",
            "--width", "64",
            "--height", "48",
            "--thumbnail-width", "16",
            "--thumbnail-height", "12",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source"] == "native_binary"
    assert payload["overlay_heavy"]["backend"] == "cpu"
    assert payload["overlay_heavy_cuda"]["backend"] == "cuda"
    assert payload["overlay_heavy_cpu_cuda_comparison"]["ok"] is True
