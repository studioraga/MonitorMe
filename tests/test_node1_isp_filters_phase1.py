import json
import os
import subprocess
from pathlib import Path

import pytest


NATIVE_CPU = Path("native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab")
SELFTEST = Path("native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab_selftest")


def _require_native_cpu():
    if not NATIVE_CPU.exists():
        pytest.skip("native CPU binary is not built")


def test_phase1_selftest_if_built():
    if not SELFTEST.exists():
        pytest.skip("native Phase 1 selftest binary is not built")
    proc = subprocess.run([str(SELFTEST)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


@pytest.mark.parametrize("filter_name", ["blur", "sharpen", "edge", "sobel-x", "sobel-y", "sobel-mag"])
def test_native_isp_synthetic_filters_emit_safe_metrics(filter_name):
    _require_native_cpu()
    proc = subprocess.run(
        [
            str(NATIVE_CPU),
            "--mode", "isp-synthetic",
            "--isp-filter", filter_name,
            "--width", "16",
            "--height", "12",
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
    assert payload["frame"] is None
    assert payload["audio"] is None
    isp = payload["isp"]
    assert isp["ok"] is True
    assert isp["schema"] == "node1_non_llm_isp_filters.v0.1"
    assert isp["backend"] == "cpu"
    assert isp["filter"] == filter_name
    assert isp["width"] == 16
    assert isp["height"] == 12
    assert isp["pixels_processed"] == 16 * 12
    assert isp["bytes_read"] == 16 * 12
    assert isp["bytes_written"] == 16 * 12
    assert isp["facts_only"] is True
    assert "identity" in isp["note"]
    assert len(isp["output"]) == 16 * 12
    assert isp["timing"]["total_ms"] >= 0


def test_native_isp_pgm_roundtrip(tmp_path):
    _require_native_cpu()
    input_pgm = tmp_path / "input.pgm"
    output_pgm = tmp_path / "output.pgm"
    pixels = bytes([0, 10, 20, 30, 40, 50, 60, 70, 80])
    input_pgm.write_bytes(b"P5\n3 3\n255\n" + pixels)

    proc = subprocess.run(
        [
            str(NATIVE_CPU),
            "--mode", "isp-pgm",
            "--input", str(input_pgm),
            "--output", str(output_pgm),
            "--isp-filter", "blur",
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
    assert payload["isp"]["ok"] is True
    assert payload["isp"]["filter"] == "blur"
    assert payload["isp_input"] == str(input_pgm)
    assert payload["isp_output"] == str(output_pgm)
    assert output_pgm.exists()
    assert output_pgm.read_bytes().startswith(b"P5\n3 3\n255\n")


def test_monitor_me_cli_gpu_lab_isp_synthetic_if_native_built():
    _require_native_cpu()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CPU)
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-isp-synthetic",
            "--filter", "sobel-mag",
            "--width", "16",
            "--height", "12",
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
    assert payload["isp"]["schema"] == "node1_non_llm_isp_filters.v0.1"
    assert payload["isp"]["filter"] == "sobel-mag"
