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
            "--mode", "sparse-roi-synthetic",
            "--scenario", "sparse",
            "--width", "32",
            "--height", "24",
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
    if payload.get("sparse_roi_cuda") is None:
        pytest.fail("CUDA native binary did not emit sparse_roi_cuda for sparse-roi-synthetic --gpu")


def test_phase3_selftest_if_built():
    if not SELFTEST.exists():
        pytest.skip("native Phase 3 selftest binary is not built")
    proc = subprocess.run([str(SELFTEST)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


@pytest.mark.parametrize("scenario", ["sparse", "mixed", "dense"])
def test_native_sparse_roi_cpu_synthetic_emits_safe_metrics(scenario):
    _require_native_cpu()
    proc = subprocess.run(
        [
            str(NATIVE_CPU),
            "--mode", "sparse-roi-synthetic",
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
    assert payload["mode"] == "sparse-roi-synthetic"
    assert payload["frame"]["ok"] is True
    assert payload["audio"] is None
    roi = payload["sparse_roi"]
    assert roi["ok"] is True
    assert roi["backend"] == "cpu"
    assert roi["schema"] == "node1_non_llm_sparse_roi.v0.1"
    assert roi["facts_only"] is True
    assert "identity" in roi["note"]
    assert roi["roi_count"] == roi["active_tiles"]
    assert roi["roi_count"] > 0
    assert roi["target_width"] == 8
    assert roi["target_height"] == 8
    assert roi["output_elements"] == roi["roi_count"] * 8 * 8
    assert len(roi["rois"]) == roi["roi_count"]
    assert len(roi["normalized"]) == roi["output_elements"]
    assert all(0.0 <= float(v) <= 1.0 for v in roi["normalized"])


@pytest.mark.parametrize("scenario", ["sparse", "mixed", "dense"])
def test_native_sparse_roi_cuda_matches_cpu_for_synthetic_scenarios(scenario):
    _require_native_cuda()
    proc = subprocess.run(
        [
            str(NATIVE_CUDA),
            "--mode", "sparse-roi-synthetic",
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
    cpu = payload["sparse_roi"]
    gpu = payload["sparse_roi_cuda"]
    assert cpu["backend"] == "cpu"
    assert gpu["backend"] == "cuda"
    assert cpu["facts_only"] is True
    assert gpu["facts_only"] is True
    assert cpu["rois"] == gpu["rois"]
    assert cpu["normalized"] == gpu["normalized"]
    comparison = payload["sparse_roi_cpu_cuda_comparison"]
    assert comparison["schema"] == "node1_non_llm_sparse_roi_cpu_cuda_compare.v0.1"
    assert comparison["ok"] is True
    assert comparison["rois_equal"] is True
    assert comparison["output_close"] is True
    assert comparison["mismatch_count"] == 0
    assert float(comparison["max_abs_diff"]) <= 1e-7
    assert comparison["metrics_close"] is True
    assert comparison["facts_only"] is True


def test_monitor_me_cli_gpu_lab_sparse_roi_synthetic_if_native_cpu_built():
    _require_native_cpu()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CPU)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "0"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-sparse-roi-synthetic",
            "--scenario", "sparse",
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
    assert payload["sparse_roi"]["schema"] == "node1_non_llm_sparse_roi.v0.1"
    assert payload["sparse_roi"]["backend"] == "cpu"


def test_monitor_me_cli_gpu_lab_sparse_roi_synthetic_uses_cuda_when_binary_is_cuda_built():
    _require_native_cuda()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CUDA)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "1"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-sparse-roi-synthetic",
            "--scenario", "sparse",
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
    assert payload["sparse_roi"]["backend"] == "cpu"
    assert payload["sparse_roi_cuda"]["backend"] == "cuda"
    assert payload["sparse_roi_cpu_cuda_comparison"]["ok"] is True
