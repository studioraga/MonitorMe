import json
import os
import subprocess
from pathlib import Path

import pytest


NATIVE_CUDA = Path("native/node1_non_llm_gpu_inference_lab/build/node1_non_llm_gpu_lab")


def _require_native_cuda():
    if not NATIVE_CUDA.exists():
        pytest.skip("native CUDA binary is not built")
    probe = subprocess.run(
        [str(NATIVE_CUDA), "--mode", "isp-synthetic", "--isp-filter", "blur", "--width", "8", "--height", "8", "--gpu"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    combined = probe.stdout + probe.stderr
    if probe.returncode == 127 and "libcudart" in combined:
        pytest.skip("native CUDA binary exists but CUDA runtime library is not available in this environment")
    assert probe.returncode == 0, combined
    payload = json.loads(probe.stdout)
    if not payload.get("cuda_compiled"):
        pytest.skip("native binary is not CUDA compiled")
    if payload.get("isp_cuda") is None:
        pytest.fail("CUDA native binary did not emit isp_cuda for isp-synthetic --gpu")


@pytest.mark.parametrize("filter_name", ["blur", "sharpen", "edge", "sobel-x", "sobel-y", "sobel-mag"])
def test_native_isp_cuda_matches_cpu_for_synthetic_filters(filter_name):
    _require_native_cuda()
    proc = subprocess.run(
        [
            str(NATIVE_CUDA),
            "--mode", "isp-synthetic",
            "--isp-filter", filter_name,
            "--width", "32",
            "--height", "24",
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
    assert payload["isp"]["backend"] == "cpu"
    assert payload["isp_cuda"]["backend"] == "cuda"
    assert payload["isp"]["filter"] == filter_name
    assert payload["isp_cuda"]["filter"] == filter_name
    assert payload["isp"]["facts_only"] is True
    assert payload["isp_cuda"]["facts_only"] is True
    assert payload["isp"]["output"] == payload["isp_cuda"]["output"]
    comparison = payload["isp_cpu_cuda_comparison"]
    assert comparison["schema"] == "node1_non_llm_isp_cpu_cuda_compare.v0.1"
    assert comparison["ok"] is True
    assert comparison["output_equal"] is True
    assert comparison["mismatch_count"] == 0
    assert comparison["max_abs_diff"] == 0
    assert comparison["metrics_close"] is True
    assert comparison["facts_only"] is True


def test_monitor_me_cli_gpu_lab_isp_synthetic_uses_cuda_when_binary_is_cuda_built():
    _require_native_cuda()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CUDA)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "1"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-isp-synthetic",
            "--filter", "sobel-mag",
            "--width", "32",
            "--height", "24",
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
    assert payload["isp"]["backend"] == "cpu"
    assert payload["isp_cuda"]["backend"] == "cuda"
    assert payload["isp_cpu_cuda_comparison"]["ok"] is True
