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
            "--mode", "mixed-region-synthetic",
            "--scenario", "contiguous",
            "--width", "64",
            "--height", "48",
            "--target-width", "4",
            "--target-height", "4",
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
    if payload.get("mixed_region_cuda") is None:
        pytest.fail("CUDA native binary did not emit mixed_region_cuda for mixed-region-synthetic --gpu")


def test_phase4_selftest_if_built():
    if not SELFTEST.exists():
        pytest.skip("native Phase 4 selftest binary is not built")
    proc = subprocess.run([str(SELFTEST)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


@pytest.mark.parametrize(
    ("scenario", "classification", "group_count", "frame_path"),
    [
        ("contiguous", "contiguous", 1, "mixed"),
        ("scattered", "scattered", 16, "mixed"),
        ("dense", "contiguous", 1, "dense"),
    ],
)
def test_native_mixed_region_cpu_synthetic_emits_safe_group_metrics(scenario, classification, group_count, frame_path):
    _require_native_cpu()
    proc = subprocess.run(
        [
            str(NATIVE_CPU),
            "--mode", "mixed-region-synthetic",
            "--scenario", scenario,
            "--width", "64",
            "--height", "48",
            "--target-width", "8",
            "--target-height", "8",
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
    assert payload["mode"] == "mixed-region-synthetic"
    assert payload["frame"]["path"] == frame_path
    assert payload["audio"] is None
    region = payload["mixed_region"]
    assert region["ok"] is True
    assert region["backend"] == "cpu"
    assert region["schema"] == "node1_non_llm_mixed_region.v0.1"
    assert region["facts_only"] is True
    assert "identity" in region["note"]
    assert region["classification"] == classification
    assert region["group_count"] == group_count
    assert region["component_count"] == group_count
    assert len(region["groups"]) == group_count
    assert region["target_width"] == 8
    assert region["target_height"] == 8
    assert region["output_elements"] == group_count * 8 * 8
    assert len(region["normalized"]) == region["output_elements"]
    assert all(0.0 <= float(v) <= 1.0 for v in region["normalized"])


@pytest.mark.parametrize("scenario", ["contiguous", "scattered", "dense"])
def test_native_mixed_region_cuda_matches_cpu_for_synthetic_scenarios(scenario):
    _require_native_cuda()
    proc = subprocess.run(
        [
            str(NATIVE_CUDA),
            "--mode", "mixed-region-synthetic",
            "--scenario", scenario,
            "--width", "64",
            "--height", "48",
            "--target-width", "8",
            "--target-height", "8",
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
    cpu = payload["mixed_region"]
    gpu = payload["mixed_region_cuda"]
    assert cpu["backend"] == "cpu"
    assert gpu["backend"] == "cuda"
    assert cpu["facts_only"] is True
    assert gpu["facts_only"] is True
    assert cpu["groups"] == gpu["groups"]
    assert cpu["normalized"] == gpu["normalized"]
    comparison = payload["mixed_region_cpu_cuda_comparison"]
    assert comparison["schema"] == "node1_non_llm_mixed_region_cpu_cuda_compare.v0.1"
    assert comparison["ok"] is True
    assert comparison["groups_equal"] is True
    assert comparison["output_close"] is True
    assert comparison["mismatch_count"] == 0
    assert float(comparison["max_abs_diff"]) <= 1e-7
    assert comparison["metrics_close"] is True
    assert comparison["facts_only"] is True


def test_monitor_me_cli_gpu_lab_mixed_region_synthetic_if_native_cpu_built():
    _require_native_cpu()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CPU)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "0"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-mixed-region-synthetic",
            "--scenario", "contiguous",
            "--width", "64",
            "--height", "48",
            "--target-width", "8",
            "--target-height", "8",
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
    assert payload["mixed_region"]["schema"] == "node1_non_llm_mixed_region.v0.1"
    assert payload["mixed_region"]["backend"] == "cpu"


def test_monitor_me_cli_gpu_lab_mixed_region_synthetic_uses_cuda_when_binary_is_cuda_built():
    _require_native_cuda()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CUDA)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "1"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-mixed-region-synthetic",
            "--scenario", "contiguous",
            "--width", "64",
            "--height", "48",
            "--target-width", "8",
            "--target-height", "8",
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
    assert payload["mixed_region"]["backend"] == "cpu"
    assert payload["mixed_region_cuda"]["backend"] == "cuda"
    assert payload["mixed_region_cpu_cuda_comparison"]["ok"] is True
