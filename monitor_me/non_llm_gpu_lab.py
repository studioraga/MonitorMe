from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GPU_LAB_MODEL_ID = "node1-non-llm-gpu-lab-v0.1"
GPU_LAB_SCHEMA = "monitorme.node1_non_llm_gpu_profile.v0.1"


@dataclass(frozen=True)
class GpuLabConfig:
    enabled: bool = False
    binary_path: str = ""
    tile_cols: int = 8
    tile_rows: int = 4
    pixel_threshold: int = 30
    sparse_threshold: int = 8
    dense_threshold: int = 24
    prefer_cuda: bool = True
    allow_python_fallback: bool = True
    timeout_sec: float = 5.0

    @staticmethod
    def default_binary_path() -> str:
        repo_root = Path(__file__).resolve().parents[1]
        return str(repo_root / "native" / "node1_non_llm_gpu_inference_lab" / "build" / "node1_non_llm_gpu_lab")

    @classmethod
    def from_env(cls, *, enabled: bool | None = None) -> "GpuLabConfig":
        return cls(
            enabled=bool(int(os.getenv("MONITORME_GPU_LAB_ENABLED", "0"))) if enabled is None else enabled,
            binary_path=os.getenv("MONITORME_GPU_LAB_BIN", ""),
            tile_cols=int(os.getenv("MONITORME_GPU_LAB_TILE_COLS", "8")),
            tile_rows=int(os.getenv("MONITORME_GPU_LAB_TILE_ROWS", "4")),
            pixel_threshold=int(os.getenv("MONITORME_GPU_LAB_PIXEL_THRESHOLD", "30")),
            sparse_threshold=int(os.getenv("MONITORME_GPU_LAB_SPARSE_THRESHOLD", "8")),
            dense_threshold=int(os.getenv("MONITORME_GPU_LAB_DENSE_THRESHOLD", "24")),
            prefer_cuda=bool(int(os.getenv("MONITORME_GPU_LAB_PREFER_CUDA", "1"))),
            allow_python_fallback=bool(int(os.getenv("MONITORME_GPU_LAB_ALLOW_PYTHON_FALLBACK", "1"))),
            timeout_sec=float(os.getenv("MONITORME_GPU_LAB_TIMEOUT_SEC", "5.0")),
        )

    @property
    def resolved_binary_path(self) -> str:
        return self.binary_path or self.default_binary_path()


def _path_name(active_tiles: int, sparse_threshold: int, dense_threshold: int) -> str:
    if active_tiles <= sparse_threshold:
        return "sparse"
    if active_tiles >= dense_threshold:
        return "dense"
    return "mixed"


def _popcount(value: int) -> int:
    return int(value & 0xFFFFFFFF).bit_count()


def _hex32(value: int) -> str:
    return f"0x{value & 0xFFFFFFFF:08X}"


def frame_to_gray_u8(frame: Any) -> Any:
    """Convert a BGR/RGB/gray numpy-like frame to uint8 grayscale without OpenCV.

    MonitorMe already uses OpenCV for capture. This helper avoids requiring OpenCV in
    the bridge and keeps the native module input contract as simple raw gray bytes.
    """

    import numpy as np  # type: ignore

    arr = np.asarray(frame)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # OpenCV capture frames are BGR. The weighting is intentionally simple;
        # the lab is about workload routing, not visual recognition quality.
        b = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        r = arr[:, :, 2].astype(np.float32)
        gray = (0.114 * b + 0.587 * g + 0.299 * r).clip(0, 255)
    else:
        raise ValueError(f"unsupported frame shape for GPU lab bridge: {arr.shape}")
    return gray.astype(np.uint8, copy=False)


def _validate_tile_contract(
    *,
    width: int,
    height: int,
    tile_cols: int,
    tile_rows: int,
    pixel_threshold: int,
    sparse_threshold: int,
    dense_threshold: int,
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if tile_cols <= 0 or tile_rows <= 0:
        raise ValueError("tile grid must be positive")
    if tile_cols * tile_rows > 32:
        raise ValueError("tile_cols * tile_rows must be <= 32 because the tile mask is uint32_t")
    if pixel_threshold < 0 or pixel_threshold > 255:
        raise ValueError("pixel_threshold must be in [0, 255]")
    if sparse_threshold < 0 or dense_threshold < 0 or sparse_threshold >= dense_threshold:
        raise ValueError("sparse_threshold must be lower than dense_threshold")
    if dense_threshold > tile_cols * tile_rows:
        raise ValueError("dense_threshold cannot exceed tile count")


def analyze_gray_frames_python(
    previous_gray: Any,
    current_gray: Any,
    *,
    tile_cols: int = 8,
    tile_rows: int = 4,
    pixel_threshold: int = 30,
    sparse_threshold: int = 8,
    dense_threshold: int = 24,
) -> dict[str, Any]:
    """Pure Python/NumPy fallback matching the C++ tile-mask contract."""

    import numpy as np  # type: ignore

    prev = np.asarray(previous_gray, dtype=np.uint8)
    curr = np.asarray(current_gray, dtype=np.uint8)
    if prev.shape != curr.shape or prev.ndim != 2:
        raise ValueError("previous/current frames must be same-shape grayscale arrays")
    height, width = int(prev.shape[0]), int(prev.shape[1])
    _validate_tile_contract(
        width=width,
        height=height,
        tile_cols=tile_cols,
        tile_rows=tile_rows,
        pixel_threshold=pixel_threshold,
        sparse_threshold=sparse_threshold,
        dense_threshold=dense_threshold,
    )

    diff = np.abs(curr.astype(np.int16) - prev.astype(np.int16)) > int(pixel_threshold)
    tile_counts: list[int] = [0 for _ in range(tile_cols * tile_rows)]
    mask = 0
    changed_pixels = int(np.count_nonzero(diff))
    ys, xs = np.nonzero(diff)
    for y, x in zip(ys.tolist(), xs.tolist()):
        tile_x = min((int(x) * tile_cols) // width, tile_cols - 1)
        tile_y = min((int(y) * tile_rows) // height, tile_rows - 1)
        tile = tile_y * tile_cols + tile_x
        tile_counts[tile] += 1
        mask |= 1 << tile

    low_mask = mask & 0x0000FFFF
    high_mask = (mask >> 16) & 0x0000FFFF
    active_tiles = _popcount(mask)
    return {
        "ok": True,
        "backend": "python_fallback",
        "width": width,
        "height": height,
        "tile_cols": tile_cols,
        "tile_rows": tile_rows,
        "pixel_threshold": pixel_threshold,
        "sparse_threshold": sparse_threshold,
        "dense_threshold": dense_threshold,
        "tile_mask": mask,
        "tile_mask_hex": _hex32(mask),
        "low_half_mask_hex": _hex32(low_mask),
        "high_half_mask_hex": _hex32(high_mask),
        "active_tiles": active_tiles,
        "low_half_active_tiles": _popcount(low_mask),
        "high_half_active_tiles": _popcount(high_mask),
        "changed_pixels": changed_pixels,
        "changed_ratio": changed_pixels / max(width * height, 1),
        "path": _path_name(active_tiles, sparse_threshold, dense_threshold),
        "tile_changed_pixels": tile_counts,
    }


class Node1NonLLMGpuLabRunner:
    """Safe Python bridge around the optional native C++/CUDA sidecar."""

    def __init__(self, config: GpuLabConfig | None = None):
        self.config = config or GpuLabConfig.from_env()

    @property
    def binary_path(self) -> Path:
        return Path(self.config.resolved_binary_path)

    def health(self, *, probe: bool = False) -> dict[str, Any]:
        binary = self.binary_path
        result: dict[str, Any] = {
            "ok": binary.exists() and os.access(binary, os.X_OK),
            "schema": GPU_LAB_SCHEMA,
            "model_id": GPU_LAB_MODEL_ID,
            "binary_path": str(binary),
            "available": binary.exists() and os.access(binary, os.X_OK),
            "python_fallback_available": True,
            "enabled": self.config.enabled,
        }
        if probe and result["available"]:
            result["probe"] = self.run_synthetic(scenario="sparse")
            result["ok"] = bool(result["probe"].get("ok"))
        return result

    def run_synthetic(self, *, scenario: str = "mixed") -> dict[str, Any]:
        binary = self.binary_path
        if not binary.exists():
            return {
                "ok": False,
                "schema": GPU_LAB_SCHEMA,
                "error": f"native binary not found: {binary}",
                "binary_path": str(binary),
            }
        cmd = [
            str(binary),
            "--mode", "synthetic",
            "--scenario", scenario,
            "--tile-cols", str(self.config.tile_cols),
            "--tile-rows", str(self.config.tile_rows),
            "--pixel-threshold", str(self.config.pixel_threshold),
            "--sparse-threshold", str(self.config.sparse_threshold),
            "--dense-threshold", str(self.config.dense_threshold),
        ]
        if self.config.prefer_cuda:
            cmd.append("--gpu")
        return self._run_json(cmd)


    def run_sparse_roi_synthetic(
        self,
        *,
        scenario: str = "sparse",
        width: int = 320,
        height: int = 240,
        target_width: int = 16,
        target_height: int = 16,
        max_rois: int = 32,
    ) -> dict[str, Any]:
        binary = self.binary_path
        if not binary.exists():
            return {
                "ok": False,
                "schema": GPU_LAB_SCHEMA,
                "error": f"native binary not found: {binary}",
                "binary_path": str(binary),
            }
        cmd = [
            str(binary),
            "--mode", "sparse-roi-synthetic",
            "--scenario", scenario,
            "--width", str(width),
            "--height", str(height),
            "--tile-cols", str(self.config.tile_cols),
            "--tile-rows", str(self.config.tile_rows),
            "--pixel-threshold", str(self.config.pixel_threshold),
            "--sparse-threshold", str(self.config.sparse_threshold),
            "--dense-threshold", str(self.config.dense_threshold),
            "--target-width", str(target_width),
            "--target-height", str(target_height),
            "--max-rois", str(max_rois),
        ]
        if self.config.prefer_cuda:
            cmd.append("--gpu")
        result = self._run_json(cmd)
        result["schema"] = GPU_LAB_SCHEMA
        result["source"] = "native_binary"
        result["binary_path"] = str(binary)
        return result


    def run_mixed_region_synthetic(
        self,
        *,
        scenario: str = "contiguous",
        width: int = 320,
        height: int = 240,
        target_width: int = 16,
        target_height: int = 16,
        max_groups: int = 32,
    ) -> dict[str, Any]:
        binary = self.binary_path
        if not binary.exists():
            return {
                "ok": False,
                "schema": GPU_LAB_SCHEMA,
                "error": f"native binary not found: {binary}",
                "binary_path": str(binary),
            }
        cmd = [
            str(binary),
            "--mode", "mixed-region-synthetic",
            "--scenario", scenario,
            "--width", str(width),
            "--height", str(height),
            "--tile-cols", str(self.config.tile_cols),
            "--tile-rows", str(self.config.tile_rows),
            "--pixel-threshold", str(self.config.pixel_threshold),
            "--sparse-threshold", str(self.config.sparse_threshold),
            "--dense-threshold", str(self.config.dense_threshold),
            "--target-width", str(target_width),
            "--target-height", str(target_height),
            "--max-groups", str(max_groups),
        ]
        if self.config.prefer_cuda:
            cmd.append("--gpu")
        result = self._run_json(cmd)
        result["schema"] = GPU_LAB_SCHEMA
        result["source"] = "native_binary"
        result["binary_path"] = str(binary)
        return result

    def run_isp_synthetic(self, *, filter_name: str = "sobel-mag", width: int = 64, height: int = 48) -> dict[str, Any]:
        binary = self.binary_path
        if not binary.exists():
            return {
                "ok": False,
                "schema": GPU_LAB_SCHEMA,
                "error": f"native binary not found: {binary}",
                "binary_path": str(binary),
            }
        cmd = [
            str(binary),
            "--mode", "isp-synthetic",
            "--isp-filter", filter_name,
            "--width", str(width),
            "--height", str(height),
        ]
        if self.config.prefer_cuda:
            cmd.append("--gpu")
        result = self._run_json(cmd)
        result["schema"] = GPU_LAB_SCHEMA
        result["source"] = "native_binary"
        result["binary_path"] = str(binary)
        return result

    def analyze_frames(self, *, previous_frame: Any, current_frame: Any) -> dict[str, Any]:
        prev_gray = frame_to_gray_u8(previous_frame)
        curr_gray = frame_to_gray_u8(current_frame)
        fallback = lambda: analyze_gray_frames_python(
            prev_gray,
            curr_gray,
            tile_cols=self.config.tile_cols,
            tile_rows=self.config.tile_rows,
            pixel_threshold=self.config.pixel_threshold,
            sparse_threshold=self.config.sparse_threshold,
            dense_threshold=self.config.dense_threshold,
        )

        binary = self.binary_path
        if not binary.exists() or not os.access(binary, os.X_OK):
            if not self.config.allow_python_fallback:
                return {
                    "ok": False,
                    "schema": GPU_LAB_SCHEMA,
                    "error": f"native binary not available: {binary}",
                    "binary_path": str(binary),
                }
            frame_result = fallback()
            return {
                "ok": True,
                "schema": GPU_LAB_SCHEMA,
                "source": "python_fallback",
                "native_binary_available": False,
                "binary_path": str(binary),
                "frame": frame_result,
                "frame_cuda": None,
            }

        with tempfile.TemporaryDirectory(prefix="monitorme_gpu_lab_") as tmp:
            tmp_path = Path(tmp)
            prev_path = tmp_path / "prev.gray"
            curr_path = tmp_path / "curr.gray"
            prev_gray.tofile(prev_path)
            curr_gray.tofile(curr_path)
            height, width = int(prev_gray.shape[0]), int(prev_gray.shape[1])
            cmd = [
                str(binary),
                "--mode", "analyze-raw-gray",
                "--prev", str(prev_path),
                "--curr", str(curr_path),
                "--width", str(width),
                "--height", str(height),
                "--tile-cols", str(self.config.tile_cols),
                "--tile-rows", str(self.config.tile_rows),
                "--pixel-threshold", str(self.config.pixel_threshold),
                "--sparse-threshold", str(self.config.sparse_threshold),
                "--dense-threshold", str(self.config.dense_threshold),
            ]
            if self.config.prefer_cuda:
                cmd.append("--gpu")
            result = self._run_json(cmd)
            result["schema"] = GPU_LAB_SCHEMA
            result["source"] = "native_binary"
            result["binary_path"] = str(binary)
            return result

    def _run_json(self, cmd: list[str]) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.config.timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return {"ok": False, "schema": GPU_LAB_SCHEMA, "error": f"GPU lab timeout: {exc}"}
        except OSError as exc:
            return {"ok": False, "schema": GPU_LAB_SCHEMA, "error": str(exc)}

        try:
            payload = json.loads(proc.stdout.strip() or "{}")
        except json.JSONDecodeError:
            payload = {"ok": False, "error": "native binary did not return JSON", "stdout": proc.stdout}
        payload["returncode"] = proc.returncode
        if proc.stderr:
            payload["stderr"] = proc.stderr.strip()
        if proc.returncode != 0:
            payload["ok"] = False
        return payload


def gpu_lab_health(*, probe: bool = False, enabled: bool | None = None) -> dict[str, Any]:
    return Node1NonLLMGpuLabRunner(GpuLabConfig.from_env(enabled=enabled)).health(probe=probe)
