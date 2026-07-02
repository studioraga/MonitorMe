from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .vlm_client import _is_loopback_url, image_file_to_data_url, parse_json_object


SMOLVLM2_SCHEMA_VERSION = "monitorme.smolvlm2_short_clip.v0.4.1"
VISIBLE_SCENE_VALUES = {"indoor", "outdoor", "unclear"}
PRESENCE_VALUES = {"visible", "not_visible", "unclear"}
MOTION_CLAIM_VALUES = {"single_frame_only_no_motion_claim"}
SAFE_OBSERVATION_VALUES = {
    "single frame reviewed",
    "visible content unclear",
    "observable scene present",
}


class ShortClipVLMClient(Protocol):
    model_id: str

    def analyze_clip(
        self,
        *,
        clip_manifest_path: str | Path,
        event: dict[str, Any],
        related_events: list[dict[str, Any]],
        clip_artifact: dict[str, Any],
        frame_artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SmolVLM2Config:
    """Configuration for a local SmolVLM2 OpenAI-compatible short-clip endpoint.

    v0.4.1 is experimental and disabled by default. It sends only locally stored
    short clip frame bundles after a trigger. Remote endpoints are rejected by
    default because MonitorMe is designed for local CCTV evidence processing.

    The default frame/token settings are tuned for a small local SmolVLM2 model
    served with vLLM at a 4096-token context. Increase only after live validation.
    """

    base_url: str = "http://127.0.0.1:8004/v1"
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    api_key: str = "EMPTY"
    timeout_sec: float = 180.0
    max_tokens: int = 300
    temperature: float = 0.0
    max_frames: int = 1
    enabled: bool = False
    allow_remote: bool = False

    @classmethod
    def from_env(cls) -> "SmolVLM2Config":
        provider = (
            os.environ.get("MONITORME_SMOLVLM2_PROVIDER")
            or os.environ.get("MONITORME_SHORT_CLIP_VLM_PROVIDER")
            or ""
        ).strip().lower()
        enabled_raw = os.environ.get("MONITORME_ASSISTANT_USE_SMOLVLM2", "").strip().lower()
        enabled = provider in {"smolvlm2", "smolvlm2-openai", "openai-compatible-video-vlm"} or enabled_raw in {"1", "true", "yes", "on"}
        return cls(
            base_url=os.environ.get("MONITORME_SMOLVLM2_BASE_URL", "http://127.0.0.1:8004/v1").rstrip("/"),
            model_id=os.environ.get("MONITORME_SMOLVLM2_MODEL_ID", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
            api_key=os.environ.get("MONITORME_SMOLVLM2_API_KEY", "EMPTY"),
            timeout_sec=float(os.environ.get("MONITORME_SMOLVLM2_TIMEOUT_SEC", "180")),
            max_tokens=int(os.environ.get("MONITORME_SMOLVLM2_MAX_TOKENS", "300")),
            temperature=float(os.environ.get("MONITORME_SMOLVLM2_TEMPERATURE", "0.0")),
            max_frames=int(os.environ.get("MONITORME_SMOLVLM2_MAX_FRAMES", "1")),
            enabled=enabled,
            allow_remote=os.environ.get("MONITORME_SMOLVLM2_ALLOW_REMOTE", "").strip().lower() in {"1", "true", "yes", "on"},
        )


class SmolVLM2OpenAIClient:
    """OpenAI-compatible SmolVLM2 client for constrained short clip experiments.

    The client uses vLLM native structured output constraints instead of freeform
    captions. This makes SmolVLM2 a bounded visual-state probe, not an evidence
    narrator. The result is still companion context only; YOLO and deterministic
    policy remain the source of truth.
    """

    prompt_version = "monitorme-smolvlm2-short-clip-structured-v0.4.1"

    def __init__(self, config: SmolVLM2Config | None = None):
        self.config = config or SmolVLM2Config.from_env()
        self.model_id = self.config.model_id
        if self.config.enabled and not self.config.allow_remote and not _is_loopback_url(self.config.base_url):
            raise ValueError("SmolVLM2 endpoint must be loopback/local unless MONITORME_SMOLVLM2_ALLOW_REMOTE=1")

    def analyze_clip(
        self,
        *,
        clip_manifest_path: str | Path,
        event: dict[str, Any],
        related_events: list[dict[str, Any]],
        clip_artifact: dict[str, Any],
        frame_artifacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        manifest = load_clip_manifest(clip_manifest_path)
        prompt = build_smolvlm2_short_clip_prompt(
            event=event,
            related_events=related_events,
            clip_artifact=clip_artifact,
            frame_artifacts=frame_artifacts,
            clip_manifest=manifest,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in manifest.get("frames", [])[: max(1, int(self.config.max_frames))]:
            path = frame.get("path") if isinstance(frame, dict) else None
            if path and Path(path).exists():
                content.append({"type": "image_url", "image_url": {"url": image_file_to_data_url(path)}})

        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "structured_outputs": {
                "json": build_smolvlm2_short_clip_schema(event=event, clip_artifact=clip_artifact),
            },
        }
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:  # noqa: S310 - local configurable endpoint
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"SmolVLM2 request failed: HTTP Error {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"SmolVLM2 request failed: {exc}") from exc
        try:
            content_text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("SmolVLM2 response did not contain choices[0].message.content") from exc
        parsed = parse_json_object(content_text)
        return validate_smolvlm2_short_clip_json(
            parsed,
            event=event,
            related_events=related_events,
            clip_artifact=clip_artifact,
            frame_artifacts=frame_artifacts,
            clip_manifest=manifest,
        )


def build_default_smolvlm2_client() -> ShortClipVLMClient | None:
    config = SmolVLM2Config.from_env()
    if not config.enabled:
        return None
    return SmolVLM2OpenAIClient(config)


def load_clip_manifest(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("short clip manifest must be a JSON object")
    return obj


def build_smolvlm2_short_clip_schema(*, event: dict[str, Any], clip_artifact: dict[str, Any]) -> dict[str, Any]:
    """Build the vLLM structured-output schema used for SmolVLM2.

    event_id and artifact_id are const-bound so the model cannot invent or drift
    away from the evidence references supplied by MonitorMe.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "event_id",
            "artifact_id",
            "visible_scene",
            "person_like_presence",
            "vehicle_like_presence",
            "motion_claim",
            "safe_observation",
            "unsupported_claims",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": SMOLVLM2_SCHEMA_VERSION},
            "event_id": {"type": "string", "const": str(event.get("event_id"))},
            "artifact_id": {"type": "string", "const": str(clip_artifact.get("artifact_id"))},
            "visible_scene": {"type": "string", "enum": sorted(VISIBLE_SCENE_VALUES)},
            "person_like_presence": {"type": "string", "enum": sorted(PRESENCE_VALUES)},
            "vehicle_like_presence": {"type": "string", "enum": sorted(PRESENCE_VALUES)},
            "motion_claim": {"type": "string", "enum": sorted(MOTION_CLAIM_VALUES)},
            "safe_observation": {"type": "string", "enum": sorted(SAFE_OBSERVATION_VALUES)},
            "unsupported_claims": {
                "type": "array",
                "maxItems": 0,
                "items": {"type": "string"},
            },
        },
    }


def build_smolvlm2_short_clip_prompt(
    *,
    event: dict[str, Any],
    related_events: list[dict[str, Any]],
    clip_artifact: dict[str, Any],
    frame_artifacts: list[dict[str, Any]],
    clip_manifest: dict[str, Any],
) -> str:
    facts = {
        "trigger_event": _event_ref(event),
        "related_event_ids": [row.get("event_id") for row in related_events if row.get("event_id")],
        "clip_artifact": _artifact_ref(clip_artifact),
        "frame_artifact_ids": [row.get("artifact_id") for row in frame_artifacts if row.get("artifact_id")],
        "clip_manifest": {
            "schema": clip_manifest.get("schema"),
            "session_id": clip_manifest.get("session_id"),
            "camera_id": clip_manifest.get("camera_id"),
            "trigger_event_id": clip_manifest.get("trigger_event_id"),
            "trigger_frame_id": clip_manifest.get("trigger_frame_id"),
            "frame_count": len(clip_manifest.get("frames", [])),
            "sampled_frames_sent_to_model": "limited by MONITORME_SMOLVLM2_MAX_FRAMES",
        },
    }
    return (
        "Return only the constrained JSON object required by the supplied structured_outputs schema. "
        "Use only the supplied local frame image(s) and evidence refs. "
        "Do not identify any person. Do not infer name, age, gender, job, nationality, ethnicity, website, social media, intent, threat, weapon, suspicious behavior, crime, or private attributes. "
        "This experiment is a companion visual-state probe only; YOLO and deterministic policy remain source of truth. "
        "For single-frame smoke tests, motion_claim must be single_frame_only_no_motion_claim. "
        "Set unsupported_claims to an empty array. "
        "Evidence refs: "
        f"{json.dumps(facts, sort_keys=True)}"
    )


def _event_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row.get("event_id"),
        "parent_event_id": row.get("parent_event_id"),
        "camera_id": row.get("camera_id"),
        "session_id": row.get("session_id"),
        "frame_id": row.get("frame_id"),
        "event_type": row.get("event_type"),
        "label": row.get("label"),
        "confidence": row.get("confidence"),
        "bbox": row.get("bbox"),
        "model_id": row.get("model_id"),
        "artifact_id": row.get("artifact_id"),
    }


def _artifact_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_id": row.get("artifact_id"),
        "artifact_type": row.get("artifact_type"),
        "path": row.get("path"),
        "sha256": row.get("sha256"),
        "media_type": row.get("media_type"),
    }


def validate_smolvlm2_short_clip_json(
    data: dict[str, Any],
    *,
    event: dict[str, Any],
    related_events: list[dict[str, Any]],
    clip_artifact: dict[str, Any],
    frame_artifacts: list[dict[str, Any]],
    clip_manifest: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("SmolVLM2 short clip output must be a JSON object")
    required = {
        "schema_version",
        "event_id",
        "artifact_id",
        "visible_scene",
        "person_like_presence",
        "vehicle_like_presence",
        "motion_claim",
        "safe_observation",
        "unsupported_claims",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"SmolVLM2 short clip JSON missing required keys: {missing}")
    extra = sorted(set(data) - required)
    if extra:
        raise ValueError(f"SmolVLM2 short clip JSON contains unsupported keys: {extra}")
    if data["schema_version"] != SMOLVLM2_SCHEMA_VERSION:
        raise ValueError(f"SmolVLM2 schema_version must be {SMOLVLM2_SCHEMA_VERSION}")
    if str(data.get("event_id")) != str(event.get("event_id")):
        raise ValueError("SmolVLM2 analysis must cite the supplied trigger event_id")
    if str(data.get("artifact_id")) != str(clip_artifact.get("artifact_id")):
        raise ValueError("SmolVLM2 analysis must cite the supplied short clip artifact_id")
    if data.get("visible_scene") not in VISIBLE_SCENE_VALUES:
        raise ValueError("SmolVLM2 visible_scene must be indoor, outdoor, or unclear")
    if data.get("person_like_presence") not in PRESENCE_VALUES:
        raise ValueError("SmolVLM2 person_like_presence must be visible, not_visible, or unclear")
    if data.get("vehicle_like_presence") not in PRESENCE_VALUES:
        raise ValueError("SmolVLM2 vehicle_like_presence must be visible, not_visible, or unclear")
    if data.get("motion_claim") not in MOTION_CLAIM_VALUES:
        raise ValueError("SmolVLM2 motion_claim must be single_frame_only_no_motion_claim")
    if data.get("safe_observation") not in SAFE_OBSERVATION_VALUES:
        raise ValueError("SmolVLM2 safe_observation is not in the allowed bounded values")
    unsupported = data.get("unsupported_claims")
    if unsupported != []:
        raise ValueError("SmolVLM2 unsupported_claims must be an empty array")

    text = json.dumps(data, sort_keys=True).lower()
    blocked_terms = [
        "recognized",
        "identity",
        "identified as",
        "face recognition",
        "name",
        "age",
        "gender",
        "occupation",
        "nationality",
        "website",
        "social media",
        "facebook",
        "profile",
        "intent",
        "suspicious",
        "threat",
        "dangerous",
        "criminal",
        "weapon",
        "gun",
        "knife",
    ]
    blocked_hits = [term for term in blocked_terms if term in text]
    if blocked_hits:
        raise ValueError(f"SmolVLM2 analysis contains unsupported claim terms: {blocked_hits}")
    data["_validated"] = True
    return data


def smolvlm2_health(config: SmolVLM2Config | None = None, *, probe: bool = False) -> dict[str, Any]:
    cfg = config or SmolVLM2Config.from_env()
    local_endpoint = _is_loopback_url(cfg.base_url)
    probe_result: dict[str, Any] = {"requested": bool(probe), "ok": None, "error": None, "models": []}
    if probe:
        request = urllib.request.Request(f"{cfg.base_url}/models", headers={"Authorization": f"Bearer {cfg.api_key}"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(cfg.timeout_sec, 10.0)) as response:  # noqa: S310 - local configurable endpoint
                body = json.loads(response.read().decode("utf-8"))
            models = body.get("data", []) if isinstance(body, dict) else []
            probe_result = {"requested": True, "ok": True, "error": None, "models": [item.get("id") for item in models if isinstance(item, dict) and item.get("id")]}
        except Exception as exc:  # pragma: no cover - exercised by Node1 live validation
            probe_result = {"requested": True, "ok": False, "error": str(exc), "models": []}
    configured = bool(cfg.enabled)
    ok = configured and (cfg.allow_remote or local_endpoint) and (not probe or bool(probe_result.get("ok")))
    return {
        "ok": ok,
        "enabled": configured,
        "provider": "smolvlm2-openai-compatible" if configured else "not_configured",
        "base_url": cfg.base_url,
        "model_id": cfg.model_id,
        "local_endpoint": local_endpoint,
        "allow_remote": cfg.allow_remote,
        "max_frames": cfg.max_frames,
        "max_tokens": cfg.max_tokens,
        "structured_outputs": True,
        "schema_version": SMOLVLM2_SCHEMA_VERSION,
        "privacy": {
            "runs_after_trigger_only": True,
            "external_upload": bool(configured and not local_endpoint and cfg.allow_remote),
            "raw_frame_upload": False,
            "short_clip_upload_to_local_vlm": bool(configured),
            "experimental": True,
        },
        "probe": probe_result,
    }
