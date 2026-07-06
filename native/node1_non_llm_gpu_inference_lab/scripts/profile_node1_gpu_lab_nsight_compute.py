#!/usr/bin/env python3
"""Build and optionally run the Node1 non-LLM GPU lab Nsight Compute pass.

The default mode is dry-run so CI and CPU-only hosts can validate the exact
profiling command plan without requiring an NVIDIA GPU or ncu installation.
Use --execute on Node1 to build the CUDA binary and collect .ncu-rep reports.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "monitorme.node1_nsight_compute_profile_run.v0.1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_plan(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    if plan.get("schema") != "monitorme.node1_nsight_compute_profile_plan.v0.1":
        raise SystemExit(f"unsupported Nsight Compute plan schema: {plan.get('schema')!r}")
    return plan


def _select_workloads(plan: dict[str, Any], requested: list[str]) -> list[dict[str, Any]]:
    workloads = list(plan.get("workloads", []))
    if not requested or requested == ["all"]:
        return workloads
    by_name = {str(w.get("name")): w for w in workloads}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in requested:
        if name == "all":
            selected.extend(workloads)
        elif name in by_name:
            selected.append(by_name[name])
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"unknown workload(s): {', '.join(missing)}")
    # Preserve caller order but remove duplicates.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in selected:
        name = str(item.get("name"))
        if name not in seen:
            seen.add(name)
            unique.append(item)
    return unique


def _command_string(command: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in command)


def _build_commands(args: argparse.Namespace, root: Path, output_dir: Path, workloads: list[dict[str, Any]]) -> dict[str, Any]:
    lab_dir = root / "native" / "node1_non_llm_gpu_inference_lab"
    build_dir = Path(args.build_dir) if args.build_dir else lab_dir / "build"
    bin_path = build_dir / "node1_non_llm_gpu_lab"
    cmake_configure = [
        "cmake",
        "-S",
        str(lab_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DNODE1_NON_LLM_ENABLE_CUDA=ON",
        f"-DCMAKE_CUDA_COMPILER={args.cuda_compiler}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
    ]
    cmake_build = ["cmake", "--build", str(build_dir), f"-j{os.cpu_count() or 1}"]

    ncu_prefix = [args.ncu_bin, "--target-processes", "all"]
    if args.ncu_metrics:
        ncu_prefix.extend(["--metrics", args.ncu_metrics])
    else:
        ncu_prefix.extend(["--set", args.ncu_set])
    if args.force_overwrite:
        ncu_prefix.append("--force-overwrite")

    workload_commands: list[dict[str, Any]] = []
    for workload in workloads:
        name = str(workload["name"])
        report_path = output_dir / f"{name}.ncu-rep"
        text_path = output_dir / f"{name}.ncu.txt"
        baseline_json = output_dir / f"{name}.baseline.json"
        args_list = [str(part) for part in workload["args"]]
        baseline_command = [str(bin_path), *args_list]
        ncu_command = [*ncu_prefix, "--export", str(report_path), str(bin_path), *args_list]
        workload_commands.append(
            {
                "name": name,
                "title": workload.get("title", name),
                "phase_origin": workload.get("phase_origin"),
                "description": workload.get("description", ""),
                "baseline_json": str(baseline_json),
                "report_path": str(report_path),
                "text_path": str(text_path),
                "baseline_command": baseline_command,
                "baseline_command_string": _command_string(baseline_command),
                "ncu_command": ncu_command,
                "ncu_command_string": _command_string(ncu_command),
            }
        )
    return {
        "lab_dir": str(lab_dir),
        "build_dir": str(build_dir),
        "bin_path": str(bin_path),
        "cmake_configure": cmake_configure,
        "cmake_configure_string": _command_string(cmake_configure),
        "cmake_build": cmake_build,
        "cmake_build_string": _command_string(cmake_build),
        "workload_commands": workload_commands,
    }


def _run(command: list[str], *, cwd: Path, stdout_path: Path | None = None) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    if stdout_path is None:
        completed = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
        stdout = completed.stdout
        stderr = completed.stderr
    else:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as f:
            completed = subprocess.run(command, cwd=str(cwd), text=True, stdout=f, stderr=subprocess.PIPE, check=False)
        stdout = ""
        stderr = completed.stderr
    finished = datetime.now(timezone.utc).isoformat()
    return {
        "command": command,
        "command_string": _command_string(command),
        "returncode": completed.returncode,
        "started_at": started,
        "completed_at": finished,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
        "stdout_path": str(stdout_path) if stdout_path else None,
    }


def _safe_manifest_base(args: argparse.Namespace, root: Path, plan: dict[str, Any], output_dir: Path, workloads: list[dict[str, Any]], commands: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "schema": SCHEMA,
        "mode": "execute" if args.execute else "dry_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": 20,
        "plan_schema": plan.get("schema"),
        "profile_plan": str(args.config),
        "repo_root": str(root),
        "output_dir": str(output_dir),
        "cuda_arch": str(args.cuda_arch),
        "ncu_bin": args.ncu_bin,
        "ncu_set": args.ncu_set if not args.ncu_metrics else None,
        "ncu_metrics": args.ncu_metrics,
        "workload_count": len(workloads),
        "workloads": [w.get("name") for w in workloads],
        "commands": commands,
        "source_scope": {
            "synthetic_inputs_only": True,
            "native_rerun": True,
            "media_decode": False,
            "external_upload": False,
            "raw_frame_upload": False,
            "semantic_claims": False,
            "destructive_actions": False,
        },
        "privacy": {
            "facts_only": True,
            "external_upload": False,
            "raw_frame_upload": False,
            "media_decode": False,
            "semantic_claims": False,
        },
        "notes": [
            "Dry-run mode constructs commands only and does not create profiler output.",
            "Execute mode must run locally on Node1 with CUDA 13.3, RTX 5060 Ti, and Nsight Compute available.",
            "Profiler output belongs under results/ and must not be staged into git.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    default_config = root / "configs" / "nsight_compute" / "node1_gpu_lab_phase20_profile_plan.json"
    parser = argparse.ArgumentParser(description="Run or dry-run the Node1 Nsight Compute profiling pass.")
    parser.add_argument("--config", default=str(default_config), help="Path to the Phase 20 Nsight Compute plan JSON.")
    parser.add_argument("--output-dir", default=None, help="Profiler output directory. Defaults under results/node1_gpu_lab/nsight_compute/<timestamp>.")
    parser.add_argument("--workload", action="append", default=[], help="Workload name to include; repeat or use all.")
    parser.add_argument("--dry-run", action="store_true", help="Print a JSON command plan without executing. This is the default.")
    parser.add_argument("--execute", action="store_true", help="Build CUDA binary and run Nsight Compute. Requires Node1 GPU + ncu.")
    parser.add_argument("--build", dest="build", action="store_true", default=True, help="Build the CUDA binary before profiling.")
    parser.add_argument("--no-build", dest="build", action="store_false", help="Skip the CUDA build step.")
    parser.add_argument("--build-dir", default=None, help="CMake build directory; defaults to native lab build/.")
    parser.add_argument("--cuda-compiler", default=os.environ.get("CMAKE_CUDA_COMPILER", "/usr/local/cuda-13.3/bin/nvcc"))
    parser.add_argument("--cuda-arch", default=os.environ.get("CMAKE_CUDA_ARCHITECTURES", "120"))
    parser.add_argument("--ncu-bin", default=os.environ.get("NCU_BIN", "ncu"))
    parser.add_argument("--ncu-set", default=os.environ.get("NCU_SET", "full"))
    parser.add_argument("--ncu-metrics", default=os.environ.get("NCU_METRICS", ""), help="Optional explicit comma-separated metric list; overrides --ncu-set.")
    parser.add_argument("--force-overwrite", action="store_true", default=True, help="Pass --force-overwrite to ncu exports.")
    args = parser.parse_args(argv)

    if args.execute and args.dry_run:
        raise SystemExit("choose only one of --dry-run or --execute")
    if not args.execute:
        args.dry_run = True

    plan = _load_plan(Path(args.config))
    workloads = _select_workloads(plan, args.workload or ["all"])
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = root / str(plan.get("default_output_dir", "results/node1_gpu_lab/nsight_compute")) / stamp
    commands = _build_commands(args, root, output_dir, workloads)
    manifest = _safe_manifest_base(args, root, plan, output_dir, workloads, commands)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "phase20_nsight_compute_manifest.json"
    ncu_path = shutil.which(args.ncu_bin)
    if not ncu_path:
        manifest["ok"] = False
        manifest["status"] = "ncu_missing"
        manifest["error"] = f"Nsight Compute binary not found: {args.ncu_bin}"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2

    run_results: list[dict[str, Any]] = []
    if args.build:
        run_results.append({"step": "cmake_configure", **_run(commands["cmake_configure"], cwd=root)})
        if run_results[-1]["returncode"] != 0:
            manifest["ok"] = False
            manifest["status"] = "cmake_configure_failed"
            manifest["run_results"] = run_results
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 2
        run_results.append({"step": "cmake_build", **_run(commands["cmake_build"], cwd=root)})
        if run_results[-1]["returncode"] != 0:
            manifest["ok"] = False
            manifest["status"] = "cmake_build_failed"
            manifest["run_results"] = run_results
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 2

    for item in commands["workload_commands"]:
        baseline_path = Path(item["baseline_json"])
        baseline_result = _run(item["baseline_command"], cwd=root, stdout_path=baseline_path)
        run_results.append({"step": "baseline", "workload": item["name"], **baseline_result})
        if baseline_result["returncode"] != 0:
            manifest["ok"] = False
            manifest["status"] = f"baseline_failed:{item['name']}"
            break
        ncu_result = _run(item["ncu_command"], cwd=root, stdout_path=Path(item["text_path"]))
        run_results.append({"step": "ncu", "workload": item["name"], **ncu_result})
        if ncu_result["returncode"] != 0:
            manifest["ok"] = False
            manifest["status"] = f"ncu_failed:{item['name']}"
            break

    manifest["run_results"] = run_results
    manifest.setdefault("status", "completed" if manifest["ok"] else "failed")
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
