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
            "--mode", "dense-full-frame-synthetic",
            "--scenario", "dense",
            "--width", "64",
            "--height", "48",
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
    if payload.get("dense_full_frame_cuda") is None:
        pytest.fail("CUDA native binary did not emit dense_full_frame_cuda for dense-full-frame-synthetic --gpu")


def test_phase5_selftest_if_built():
    if not SELFTEST.exists():
        pytest.skip("native Phase 5 selftest binary is not built")
    proc = subprocess.run([str(SELFTEST)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


@pytest.mark.parametrize(
    ("scenario", "frame_path"),
    [
        ("dense", "dense"),
        ("mixed", "mixed"),
        ("sparse", "sparse"),
    ],
)
def test_native_dense_full_frame_cpu_synthetic_emits_safe_reduction_metrics(scenario, frame_path):
    _require_native_cpu()
    proc = subprocess.run(
        [
            str(NATIVE_CPU),
            "--mode", "dense-full-frame-synthetic",
            "--scenario", scenario,
            "--width", "64",
            "--height", "48",
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
    assert payload["mode"] == "dense-full-frame-synthetic"
    assert payload["frame"]["path"] == frame_path
    dense = payload["dense_full_frame"]
    assert dense["ok"] is True
    assert dense["backend"] == "cpu"
    assert dense["schema"] == "node1_non_llm_dense_full_frame.v0.1"
    assert dense["facts_only"] is True
    assert "identity" in dense["note"]
    assert dense["pixels_processed"] == 64 * 48
    assert dense["histogram_total"] == dense["pixels_processed"]
    assert len(dense["diff_histogram"]) == 256
    assert len(dense["normalized"]) == dense["pixels_processed"]
    assert all(0.0 <= float(v) <= 1.0 for v in dense["normalized"])
    assert dense["bytes_read"] == dense["pixels_processed"] * 2
    assert dense["bytes_written"] == dense["pixels_processed"] * 4 + 256 * 8
    if scenario == "dense":
        assert dense["changed_pixels"] == dense["pixels_processed"]
        assert dense["diff_histogram"][200] == dense["pixels_processed"]
        assert abs(dense["lighting_delta"] - 200.0) <= 1e-9


@pytest.mark.parametrize("scenario", ["dense", "mixed", "sparse"])
def test_native_dense_full_frame_cuda_matches_cpu_for_synthetic_scenarios(scenario):
    _require_native_cuda()
    proc = subprocess.run(
        [
            str(NATIVE_CUDA),
            "--mode", "dense-full-frame-synthetic",
            "--scenario", scenario,
            "--width", "64",
            "--height", "48",
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
    cpu = payload["dense_full_frame"]
    gpu = payload["dense_full_frame_cuda"]
    assert cpu["backend"] == "cpu"
    assert gpu["backend"] == "cuda"
    assert cpu["facts_only"] is True
    assert gpu["facts_only"] is True
    assert cpu["diff_histogram"] == gpu["diff_histogram"]
    assert cpu["normalized"] == gpu["normalized"]
    comparison = payload["dense_full_frame_cpu_cuda_comparison"]
    assert comparison["schema"] == "node1_non_llm_dense_full_frame_cpu_cuda_compare.v0.1"
    assert comparison["ok"] is True
    assert comparison["histogram_equal"] is True
    assert comparison["normalized_close"] is True
    assert comparison["mismatch_count"] == 0
    assert float(comparison["max_abs_diff"]) <= 1e-7
    assert comparison["reductions_close"] is True
    assert comparison["facts_only"] is True


def test_monitor_me_cli_gpu_lab_dense_full_frame_synthetic_if_native_cpu_built():
    _require_native_cpu()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CPU)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "0"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-dense-full-frame-synthetic",
            "--scenario", "dense",
            "--width", "64",
            "--height", "48",
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
    assert payload["dense_full_frame"]["schema"] == "node1_non_llm_dense_full_frame.v0.1"
    assert payload["dense_full_frame"]["backend"] == "cpu"


def test_monitor_me_cli_gpu_lab_dense_full_frame_synthetic_uses_cuda_when_binary_is_cuda_built():
    _require_native_cuda()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CUDA)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "1"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-dense-full-frame-synthetic",
            "--scenario", "dense",
            "--width", "64",
            "--height", "48",
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
    assert payload["dense_full_frame"]["backend"] == "cpu"
    assert payload["dense_full_frame_cuda"]["backend"] == "cuda"
    assert payload["dense_full_frame_cpu_cuda_comparison"]["ok"] is True
