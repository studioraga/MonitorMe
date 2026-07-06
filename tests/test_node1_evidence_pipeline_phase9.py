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


def test_phase9_selftest_if_built():
    if not SELFTEST.exists():
        pytest.skip("native Phase 9 selftest binary is not built")
    proc = subprocess.run([str(SELFTEST)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


@pytest.mark.parametrize("clips", [8, 12, 20])
def test_native_evidence_pipeline_synthetic_emits_fingerprints_dedup_latency_and_safety(clips):
    _require_native_cpu()
    proc = subprocess.run(
        [
            str(NATIVE_CPU),
            "--mode", "evidence-pipeline-synthetic",
            "--clips", str(clips),
            "--max-batch-bytes", "1600000",
            "--max-batch-clips", "3",
            "--key-moments", "4",
            "--min-key-gap-ms", "1000",
            "--dedup-hamming-threshold", "0",
            "--fingerprint-width", "16",
            "--fingerprint-height", "16",
            "--fingerprint-cycle", "6",
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
    assert payload["mode"] == "evidence-pipeline-synthetic"
    evidence = payload["evidence_pipeline"]
    assert evidence["ok"] is True
    assert evidence["backend"] == "cpu"
    assert evidence["schema"] == "node1_non_llm_evidence_pipeline.v0.1"
    assert evidence["facts_only"] is True
    assert "no object" in evidence["note"]
    assert evidence["manifest_entries"] == clips
    assert evidence["fingerprint_count"] == clips
    assert evidence["storage_batch"]["clip_count"] == clips
    assert evidence["storage_batch"]["planned_read_bytes"] == evidence["storage_batch"]["total_manifest_bytes"]
    assert evidence["planned_read_bytes"] == evidence["total_manifest_bytes"]
    assert evidence["batch_count"] == payload["storage_batch"]["batch_count"]
    assert evidence["key_moment_count"] <= 4
    assert evidence["unique_clip_count"] + evidence["duplicate_clip_count"] == clips
    assert evidence["duplicate_group_count"] >= 1
    assert len(evidence["fingerprints"]) == clips
    assert len(evidence["duplicate_groups"]) == evidence["duplicate_group_count"]
    assert evidence["fingerprints"][0]["fingerprint_hex"].startswith("0x")
    assert len(evidence["fingerprints"][0]["histogram16"]) == 16
    assert evidence["timeline"]["clip_count"] == clips
    assert evidence["latency"]["total_ms"] >= 0
    assert evidence["latency"]["planned_read_mb"] > 0
    assert evidence["safety"]["ok"] is True
    assert evidence["safety"]["facts_only"] is True
    assert evidence["safety"]["no_semantic_claims"] is True
    assert evidence["safety"]["violation_count"] == 0
    assert evidence["safety"]["batch_plan_ok"] is True
    assert evidence["safety"]["fingerprint_ok"] is True
    assert evidence["safety"]["dedup_ok"] is True


def test_native_evidence_pipeline_manifest_scan(tmp_path):
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
            "--mode", "evidence-pipeline-manifest",
            "--manifest", str(manifest),
            "--max-batch-bytes", "1200000",
            "--max-batch-clips", "2",
            "--key-moments", "2",
            "--min-key-gap-ms", "1000",
            "--dedup-hamming-threshold", "0",
            "--fingerprint-cycle", "2",
            "--include-output",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    evidence = payload["evidence_pipeline"]
    assert payload["ok"] is True
    assert payload["mode"] == "evidence-pipeline-manifest"
    assert evidence["ok"] is True
    assert evidence["manifest_entries"] == 4
    assert evidence["storage_batch"]["batch_count"] == 3
    assert evidence["key_moment_count"] == 2
    assert evidence["storage_batch"]["planned_read_bytes"] == 2600000
    assert evidence["safety"]["ok"] is True


def test_monitor_me_cli_gpu_lab_evidence_pipeline_synthetic_if_native_cpu_built():
    _require_native_cpu()
    env = os.environ.copy()
    env["MONITORME_GPU_LAB_BIN"] = str(NATIVE_CPU)
    env["MONITORME_GPU_LAB_PREFER_CUDA"] = "0"
    proc = subprocess.run(
        [
            "python", "-m", "monitor_me.cli",
            "gpu-lab-evidence-pipeline-synthetic",
            "--clips", "12",
            "--max-batch-bytes", "1600000",
            "--max-batch-clips", "3",
            "--key-moments", "4",
            "--min-key-gap-ms", "1000",
            "--dedup-hamming-threshold", "0",
            "--fingerprint-cycle", "6",
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
    evidence = payload["evidence_pipeline"]
    assert evidence["schema"] == "node1_non_llm_evidence_pipeline.v0.1"
    assert evidence["backend"] == "cpu"
    assert evidence["manifest_entries"] == 12
    assert evidence["fingerprint_count"] == 12
    assert evidence["safety"]["ok"] is True
