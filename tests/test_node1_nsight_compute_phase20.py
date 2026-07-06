from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "nsight_compute" / "node1_gpu_lab_phase20_profile_plan.json"
SCRIPT = ROOT / "native" / "node1_non_llm_gpu_inference_lab" / "scripts" / "profile_node1_gpu_lab_nsight_compute.py"
SELFTEST = ROOT / "native" / "node1_non_llm_gpu_inference_lab" / "scripts" / "run_node1_gpu_lab_phase20_nsight_compute_selftest.sh"


def test_phase20_nsight_compute_plan_is_synthetic_and_cuda_scoped() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    assert plan["schema"] == "monitorme.node1_nsight_compute_profile_plan.v0.1"
    assert plan["phase"] == 20
    assert plan["safety"]["synthetic_inputs_only"] is True
    assert plan["safety"]["external_upload"] is False
    assert plan["safety"]["raw_frame_upload"] is False
    assert plan["safety"]["media_decode"] is False
    assert plan["safety"]["semantic_claims"] is False

    workloads = {item["name"]: item for item in plan["workloads"]}
    assert set(workloads) == {
        "isp_sobel_mag",
        "sparse_roi",
        "mixed_region",
        "dense_full_frame",
        "overlay_heavy",
        "audiobox",
    }
    for workload in workloads.values():
        args = workload["args"]
        assert "--gpu" in args
        assert any("synthetic" in str(arg) for arg in args)
        joined = " ".join(args)
        assert "data/captures" not in joined
        assert "http://" not in joined
        assert "https://" not in joined


def test_phase20_nsight_compute_dry_run_builds_expected_commands(tmp_path: Path) -> None:
    output_dir = tmp_path / "ncu"
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--dry-run",
            "--workload",
            "dense_full_frame",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["schema"] == "monitorme.node1_nsight_compute_profile_run.v0.1"
    assert payload["mode"] == "dry_run"
    assert payload["phase"] == 20
    assert payload["cuda_arch"] == os.environ.get("CMAKE_CUDA_ARCHITECTURES", "120")
    assert payload["workloads"] == ["dense_full_frame"]
    assert payload["output_dir"] == str(output_dir)
    assert payload["source_scope"]["synthetic_inputs_only"] is True
    assert payload["source_scope"]["native_rerun"] is True
    assert payload["source_scope"]["media_decode"] is False
    assert payload["privacy"]["facts_only"] is True
    assert payload["privacy"]["external_upload"] is False
    assert payload["privacy"]["raw_frame_upload"] is False

    commands = payload["commands"]
    assert "-DNODE1_NON_LLM_ENABLE_CUDA=ON" in commands["cmake_configure"]
    assert any(str(part).endswith("nvcc") for part in commands["cmake_configure"])
    workload_commands = commands["workload_commands"]
    assert len(workload_commands) == 1
    workload = workload_commands[0]
    assert workload["name"] == "dense_full_frame"
    assert workload["report_path"].endswith("dense_full_frame.ncu-rep")
    assert workload["baseline_json"].endswith("dense_full_frame.baseline.json")
    assert "--mode" in workload["baseline_command"]
    assert "dense-full-frame-synthetic" in workload["baseline_command"]
    ncu_command = workload["ncu_command"]
    assert ncu_command[0] == "ncu"
    assert "--target-processes" in ncu_command
    assert "all" in ncu_command
    assert "--export" in ncu_command
    assert "--gpu" in ncu_command
    assert "dense-full-frame-synthetic" in workload["ncu_command_string"]
    assert not output_dir.exists(), "dry-run must not create profiler output directories"


def test_phase20_nsight_compute_dry_run_all_workloads_has_no_external_or_runtime_paths(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--dry-run",
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    blob = json.dumps(payload)

    assert payload["workload_count"] == 6
    assert "isp_sobel_mag" in payload["workloads"]
    assert "sparse_roi" in payload["workloads"]
    assert "mixed_region" in payload["workloads"]
    assert "dense_full_frame" in payload["workloads"]
    assert "overlay_heavy" in payload["workloads"]
    assert "audiobox" in payload["workloads"]
    assert "http://" not in blob
    assert "https://" not in blob
    assert "data/captures" not in blob
    assert payload["privacy"]["raw_frame_upload"] is False


def test_phase20_selftest_and_docs_reference_dry_run_safety() -> None:
    assert SELFTEST.exists()
    selftest_text = SELFTEST.read_text(encoding="utf-8")
    assert "--dry-run" in selftest_text
    assert "Phase 20 Nsight Compute profiling selftest PASS" in selftest_text

    readme = (ROOT / "native" / "node1_non_llm_gpu_inference_lab" / "README.md").read_text(encoding="utf-8")
    assert "Phase 20" in readme
    assert "Nsight Compute" in readme
    assert "--dry-run" in readme
    assert "--execute" in readme
