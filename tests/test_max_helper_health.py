from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

from monitor_me.llm_client import GemmaMaxConfig, gemma_max_health


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps({"data": [{"id": "google/gemma-3-1b-it"}]}).encode("utf-8")


def test_gemma_max_health_can_probe_models_endpoint():
    cfg = GemmaMaxConfig(enabled=True, base_url="http://127.0.0.1:8000/v1")
    with patch("urllib.request.urlopen", return_value=_Response()):
        out = gemma_max_health(cfg, probe=True)

    assert out["ok"] is True
    assert out["probe"]["ok"] is True
    assert out["probe"]["models"] == ["google/gemma-3-1b-it"]
    assert out["privacy"]["raw_frame_upload"] is False


def test_gemma_max_health_reports_unconfigured_without_probe():
    cfg = GemmaMaxConfig(enabled=False)
    out = gemma_max_health(cfg, probe=False)

    assert out["ok"] is False
    assert out["enabled"] is False
    assert out["probe"]["requested"] is False


def test_stable_max_recovery_helper_exists():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    helper = root / "scripts" / "max" / "create_max_gemma3_1b_stable_py312_env.sh"
    text = helper.read_text()
    assert helper.exists()
    assert "https://conda.modular.com/max" in text
    assert "quickstart_py312_stable" in text


def test_term1_preserves_mojo_cache_by_default():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    helper = root / "scripts" / "max" / "term1_start_max_gemma3_1b.sh"
    text = helper.read_text()
    assert 'MONITORME_MAX_CLEAR_MOJO_CACHE:-0' in text
    assert 'MONITORME_MAX_CLEAR_MOJO_CACHE=1' in text


def test_term1_uses_clean_mojo_import_path_and_skips_help_by_default():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    helper = root / "scripts" / "max" / "term1_start_max_gemma3_1b.sh"
    text = helper.read_text()
    assert 'MONITORME_MAX_OVERRIDE_MOJO_IMPORT_PATH:-1' in text
    assert 'MONITORME_MAX_SKIP_HELP_PREFLIGHT:-1' in text
    assert 'Skipping pixi run max --help preflight' in text
    assert 'share/locale' in text and 'share/tabset' in text
    assert 'std.mojoc' in text and 'nn.mojoc' in text


def test_probe_uses_clean_mojo_import_roots():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    helper = root / "scripts" / "max" / "probe_mojo_std_nn_roots.sh"
    text = helper.read_text()
    assert 'clean Mojo import roots used' in text
    assert 'std.mojoc' in text
    assert 'share/locale' in text and 'share/tabset' in text


def test_term1_uses_pixi_env_wrapper_for_mojo_import_roots():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    helper = root / "scripts" / "max" / "term1_start_max_gemma3_1b.sh"
    text = helper.read_text()
    assert "pixi_env_run()" in text
    assert "pixi run env" in text
    assert "MODULAR_MOJO_MAX_IMPORT_PATH=$MODULAR_MOJO_MAX_IMPORT_PATH" in text
    assert "MOJO_PACKAGE_PATH=${MOJO_PACKAGE_PATH:-$MODULAR_MOJO_MAX_IMPORT_PATH}" in text
    assert "MAX_SERVE_ARGS=(" in text
    assert "MODULAR_DEBUG=device-sync-mode pixi_env_run max" in text


def test_probe_uses_pixi_env_wrapper_for_mojo_import_roots():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    helper = root / "scripts" / "max" / "probe_mojo_std_nn_roots.sh"
    text = helper.read_text()
    assert "pixi_env_run()" in text
    assert "pixi run env" in text
    assert "MODULAR_MOJO_MAX_IMPORT_PATH=$MODULAR_MOJO_MAX_IMPORT_PATH" in text


def test_term1_disables_sample_on_host_by_default_and_allows_opt_in():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    helper = root / "scripts" / "max" / "term1_start_max_gemma3_1b.sh"
    text = helper.read_text()
    assert 'MONITORME_MAX_SAMPLE_ON_HOST:-0' in text
    assert 'MAX_SERVE_ARGS+=(--sample-on-host)' in text
    assert '--sample-on-host \\\n' not in text


def test_term1_can_prebuild_mojo_caches_before_max_serve():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    helper = root / "scripts" / "max" / "term1_start_max_gemma3_1b.sh"
    text = helper.read_text()
    assert 'MONITORME_MAX_PREBUILD_MOJO_CACHES:-1' in text
    assert 'prebuild_mojo_cache_for_module "max._distributed_ops" "distributed_ops"' in text
    assert 'mojo build "$src"' in text
    assert '-I "$prefix/lib/mojo"' in text
    assert '-I "$max_site"' in text


def test_patch_pixi_activation_env_helper_exists_and_term1_references_kernel_error():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    patcher = root / "scripts" / "max" / "patch_max_pixi_activation_env.sh"
    term1 = root / "scripts" / "max" / "term1_start_max_gemma3_1b.sh"
    patcher_text = patcher.read_text()
    term1_text = term1.read_text()

    assert patcher.exists()
    assert "[activation.env]" in patcher_text
    assert "MODULAR_MOJO_MAX_IMPORT_PATH" in patcher_text
    assert "MOJO_PACKAGE_PATH" in patcher_text
    assert "failed to resolve built-in kernel package paths" in patcher_text
    assert "MAXG_addKernelPackage" in patcher_text
    assert "MONITORME_MAX_PATCH_PIXI_ACTIVATION_ENV:-0" in term1_text
    assert "patch_max_pixi_activation_env.sh" in term1_text


def test_term1_uses_single_lib_mojo_runtime_mode_by_default() -> None:
    script = (ROOT / "scripts" / "max" / "term1_start_max_gemma3_1b.sh").read_text()
    assert "MONITORME_MAX_RUNTIME_MOJO_IMPORT_MODE:-single-lib-mojo" in script
    assert "Runtime Mojo import path" in script

def test_pixi_activation_env_uses_lib_mojo_only_for_runtime_kernel_root() -> None:
    script = (ROOT / "scripts" / "max" / "patch_max_pixi_activation_env.sh").read_text()
    assert 'IMPORT_ROOTS="$LIB_MOJO"' in script
    assert 'IMPORT_ROOTS="$LIB_MOJO:$MAX_SITE"' not in script
