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


def test_phase8_selftest_if_built():
    if not SELFTEST.exists():
        pytest.skip("native Phase 8 selftest binary is not built")
    proc = subprocess.run([str(SELFTEST)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


@pytest.mark.parametrize("clips", [8, 12, 20])
def test_native_storage_batch_synthetic_emits_manifest_plan_key_moments_and_timeline(clips):
    _require_native_cpu()
    proc = subprocess.run(
        [
            str(NATIVE_CPU),
            "--mode", "storage-batch-synthetic",
            "--clips", str(clips),
            "--max-batch-bytes", "1600000",
            "--max-batch-clips", "3",
            "--key-moments", "4",
            "--min-key-gap-ms", "1000",
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
    assert payload["mode"] == "storage-batch-synthetic"
    storage = payload["storage_batch"]
    assert storage["ok"] is True
    assert storage["backend"] == "cpu"
    assert storage["schema"] == "node1_non_llm_storage_batch.v0.1"
    assert storage["facts_only"] is True
    assert "no visual" in storage["note"]
    assert storage["manifest_entries"] == clips
    assert storage["clip_count"] == clips
    assert storage["batch_count"] >= 1
    assert storage["key_moment_count"] == min(4, clips)
    assert storage["planned_read_bytes"] == storage["total_manifest_bytes"]
    assert len(storage["manifest"]) == clips
    assert len(storage["batches"]) == storage["batch_count"]
    assert len(storage["key_moments"]) == storage["key_moment_count"]
    assert storage["timeline"]["clip_count"] == clips
    assert storage["timeline"]["total_bytes"] == storage["total_manifest_bytes"]
    assert storage["timeline"]["timeline_span_ms"] > 0
    assert storage["key_moments"][0]["clip_id"] == "clip_3"
    for batch in storage["batches"]:
        assert batch["clip_count"] >= 1
        assert batch["total_bytes"] > 0
        assert len(batch["clip_indices"]) == batch["clip_count"]


def test_native_storage_batch_manifest_scan(tmp_path):
    _require_native_cpu()
    manifest = tmp_path / "clips.csv"
    manifest.write_text(
        "clip_id,path,start_ms,duration_ms,bytes,motion_score,audio_score,lighting_delta,changed_pixels\n"
        "a,clips/a.mkv,0,1000,500000,0.1,0.0,2,100\n"
        "b,clips/b.mkv,1200,1000,600000,0.9,0.2,30,30000\n"
        "c,clips/c.mkv,2600,1000,700000,0.2,0.8,12,12000\n"
        "d,clips/d.mkv,4000,1000,800000,0.7,0.7,50,50000\n"
    )
    proc = subprocess.run(
        [
            str(NATIVE_CPU),
            "--mode", "storage-batch-manifest",
            "--manifest", str(manifest),
            "--max-batch-bytes", "1200000",
            "--max-batch-clips", "2",
            "--key-moments", "2",
            "--min-key-gap-ms", "1000",
            "--include-output",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    storage = payload["storage_batch"]
    assert payload["ok"] is True
    assert payload["mode"] == "storage-batch-manifest"
    assert storage["ok"] is True
    assert storage["manifest_entries"] == 4
    assert storage["batch_count"] == 3
    assert storage["key_moment_count"] == 2
    assert storage["key_moments"][0]["clip_id"] == "d"
    assert storage["planned_read_bytes"] == 2600000


def test_monitor_me_cli_gpu_lab_storage_batch_synthetic_if_native_cpu_built():
    _require_native_cpu()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CPU)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "0"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-storage-batch-synthetic",
            "--clips", "12",
            "--max-batch-bytes", "1600000",
            "--max-batch-clips", "3",
            "--key-moments", "4",
            "--min-key-gap-ms", "1000",
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
    storage = payload["storage_batch"]
    assert storage["schema"] == "node1_non_llm_storage_batch.v0.1"
    assert storage["backend"] == "cpu"
    assert storage["clip_count"] == 12
    assert storage["key_moment_count"] == 4
