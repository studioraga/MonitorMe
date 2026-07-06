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
            "--mode", "audiobox-synthetic",
            "--audio-samples", "8192",
            "--audio-window-samples", "1024",
            "--audio-max-windows", "8",
            "--max-lag", "96",
            "--sync-drift-samples", "48",
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
    if payload.get("audiobox_cuda") is None:
        pytest.fail("CUDA native binary did not emit audiobox_cuda for audiobox-synthetic --gpu")


def test_phase7_selftest_if_built():
    if not SELFTEST.exists():
        pytest.skip("native Phase 7 selftest binary is not built")
    proc = subprocess.run([str(SELFTEST)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


@pytest.mark.parametrize("drift", [0, 64, -64])
def test_native_audiobox_cpu_synthetic_emits_safe_audio_metrics(drift):
    _require_native_cpu()
    proc = subprocess.run(
        [
            str(NATIVE_CPU),
            "--mode", "audiobox-synthetic",
            "--audio-samples", "32768",
            "--sample-rate", "48000",
            "--audio-window-samples", "1024",
            "--audio-max-windows", "32",
            "--silence-threshold", "0.02",
            "--onset-threshold", "0.08",
            "--max-lag", "128",
            "--sync-drift-samples", str(drift),
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
    assert payload["mode"] == "audiobox-synthetic"
    audio = payload["audiobox"]
    assert audio["ok"] is True
    assert audio["backend"] == "cpu"
    assert audio["schema"] == "node1_non_llm_audiobox.v0.1"
    assert audio["facts_only"] is True
    assert "speaker identity" in audio["note"]
    assert audio["samples"] == 32768
    assert audio["sample_rate"] == 48000
    assert audio["windows"] == 32
    assert len(audio["rms"]) == 32
    assert len(audio["peaks"]) == 32
    assert len(audio["correlation_scores"]) == 257
    assert audio["active_windows"] > 0
    assert audio["silent_windows"] > 0
    assert audio["onset_count"] >= 1
    assert audio["max_peak"] > 0.35
    assert audio["sync_drift_samples"] == drift
    assert abs(audio["sync_drift_ms"] - (1000.0 * drift / 48000.0)) < 1e-4
    assert audio["sync_correlation_abs"] > 0.99
    assert audio["bytes_read"] == 32768 * 4 * 2
    assert audio["bytes_written"] == (len(audio["rms"]) + len(audio["peaks"]) + len(audio["correlation_scores"])) * 4


@pytest.mark.parametrize("drift", [0, 64, -64])
def test_native_audiobox_cuda_matches_cpu_for_synthetic_drifts(drift):
    _require_native_cuda()
    proc = subprocess.run(
        [
            str(NATIVE_CUDA),
            "--mode", "audiobox-synthetic",
            "--audio-samples", "32768",
            "--sample-rate", "48000",
            "--audio-window-samples", "1024",
            "--audio-max-windows", "32",
            "--silence-threshold", "0.02",
            "--onset-threshold", "0.08",
            "--max-lag", "128",
            "--sync-drift-samples", str(drift),
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
    cpu = payload["audiobox"]
    gpu = payload["audiobox_cuda"]
    assert cpu["backend"] == "cpu"
    assert gpu["backend"] == "cuda"
    assert cpu["facts_only"] is True
    assert gpu["facts_only"] is True
    assert cpu["sync_drift_samples"] == drift
    assert gpu["sync_drift_samples"] == drift
    comparison = payload["audiobox_cpu_cuda_comparison"]
    assert comparison["schema"] == "node1_non_llm_audiobox_cpu_cuda_compare.v0.1"
    assert comparison["ok"] is True
    assert comparison["rms_close"] is True
    assert comparison["peaks_close"] is True
    assert comparison["correlation_close"] is True
    assert comparison["masks_equal"] is True
    assert comparison["drift_equal"] is True
    assert comparison["metrics_close"] is True
    assert comparison["mismatch_count"] == 0
    assert float(comparison["max_abs_diff"]) <= 1e-4
    assert comparison["facts_only"] is True


def test_monitor_me_cli_gpu_lab_audiobox_synthetic_if_native_cpu_built():
    _require_native_cpu()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CPU)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "0"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-audiobox-synthetic",
            "--audio-samples", "32768",
            "--window-samples", "1024",
            "--max-lag", "128",
            "--sync-drift-samples", "64",
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
    assert payload["audiobox"]["schema"] == "node1_non_llm_audiobox.v0.1"
    assert payload["audiobox"]["backend"] == "cpu"
    assert payload["audiobox"]["sync_drift_samples"] == 64


def test_monitor_me_cli_gpu_lab_audiobox_synthetic_uses_cuda_when_binary_is_cuda_built():
    _require_native_cuda()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CUDA)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "1"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-audiobox-synthetic",
            "--audio-samples", "32768",
            "--window-samples", "1024",
            "--max-lag", "128",
            "--sync-drift-samples", "64",
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
    assert payload["audiobox"]["backend"] == "cpu"
    assert payload["audiobox_cuda"]["backend"] == "cuda"
    assert payload["audiobox_cpu_cuda_comparison"]["ok"] is True
