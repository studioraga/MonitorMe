from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class KeyframeVLMClient(Protocol):
    model_id: str

    def analyze_keyframe(
        self,
        *,
        image_path: str | Path,
        event: dict[str, Any],
        related_events: list[dict[str, Any]],
        artifact: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class QwenVLMConfig:
    """Configuration for a local Qwen VLM OpenAI-compatible endpoint.

    The client is disabled by default. When enabled, it sends only stored
    keyframe artifacts after a local trigger. Remote endpoints are rejected by
    default because MonitorMe is designed for local CCTV evidence processing.
    """

    base_url: str = "http://127.0.0.1:8002/v1"
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    api_key: str = "EMPTY"
    timeout_sec: float = 120.0
    max_tokens: int = 384
    temperature: float = 0.0
    enabled: bool = False
    allow_remote: bool = False

    @classmethod
    def from_env(cls) -> "QwenVLMConfig":
        provider = os.environ.get("MONITORME_VLM_PROVIDER", "").strip().lower()
        enabled_raw = os.environ.get("MONITORME_ASSISTANT_USE_QWEN_VLM", "").strip().lower()
        enabled = provider in {"qwen-vlm", "qwen-openai", "openai-compatible-vlm", "vlm-openai"} or enabled_raw in {"1", "true", "yes", "on"}
        return cls(
            base_url=os.environ.get("MONITORME_VLM_BASE_URL", "http://127.0.0.1:8002/v1").rstrip("/"),
            model_id=os.environ.get("MONITORME_VLM_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct"),
            api_key=os.environ.get("MONITORME_VLM_API_KEY", "EMPTY"),
            timeout_sec=float(os.environ.get("MONITORME_VLM_TIMEOUT_SEC", "120")),
            max_tokens=int(os.environ.get("MONITORME_VLM_MAX_TOKENS", "384")),
            temperature=float(os.environ.get("MONITORME_VLM_TEMPERATURE", "0.0")),
            enabled=enabled,
            allow_remote=os.environ.get("MONITORME_VLM_ALLOW_REMOTE", "").strip().lower() in {"1", "true", "yes", "on"},
        )


def _is_loopback_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def image_file_to_data_url(path: str | Path) -> str:
    p = Path(path)
    media_type = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{data}"


class QwenVLMOpenAIClient:
    """OpenAI-compatible Qwen VLM client for local keyframe analysis.

    It asks for strict JSON only and accepts only a bounded visual-facts schema.
    VLM output is stored as companion visual context and must not override YOLO
    labels, deterministic policy, event IDs, or operator evidence decisions.
    """

    prompt_version = "monitorme-qwen-keyframe-json-v0.3"

    def __init__(self, config: QwenVLMConfig | None = None):
        self.config = config or QwenVLMConfig.from_env()
        self.model_id = self.config.model_id
        if self.config.enabled and not self.config.allow_remote and not _is_loopback_url(self.config.base_url):
            raise ValueError("Qwen VLM endpoint must be loopback/local unless MONITORME_VLM_ALLOW_REMOTE=1")

    def analyze_keyframe(
        self,
        *,
        image_path: str | Path,
        event: dict[str, Any],
        related_events: list[dict[str, Any]],
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = build_qwen_keyframe_prompt(event=event, related_events=related_events, artifact=artifact)
        payload = {
            "model": self.config.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_file_to_data_url(image_path)}},
                    ],
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
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
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Qwen VLM request failed: {exc}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Qwen VLM response did not contain choices[0].message.content") from exc
        parsed = parse_json_object(content)
        return validate_qwen_keyframe_json(parsed, event=event, related_events=related_events, artifact=artifact)


def build_default_keyframe_vlm() -> KeyframeVLMClient | None:
    config = QwenVLMConfig.from_env()
    if not config.enabled:
        return None
    return QwenVLMOpenAIClient(config)


def build_qwen_keyframe_prompt(*, event: dict[str, Any], related_events: list[dict[str, Any]], artifact: dict[str, Any]) -> str:
    facts = {
        "trigger_event": _event_ref(event),
        "related_events": [_event_ref(row) for row in related_events],
        "artifact": {
            "artifact_id": artifact.get("artifact_id"),
            "artifact_type": artifact.get("artifact_type"),
            "path": artifact.get("path"),
            "sha256": artifact.get("sha256"),
            "media_type": artifact.get("media_type"),
        },
    }
    return (
        "You are MonitorMe's local Node1 keyframe visual-facts assistant.\n"
        "Analyze only the provided keyframe image and the supplied JSON refs.\n"
        "Return cautious visual observations, not decisions. Do not infer identity, intent, threat, crime, face recognition, or suspicious behavior.\n"
        "Do not create new event IDs, artifact IDs, frame IDs, cameras, timestamps, bounding boxes, policy actions, or YOLO detections.\n"
        "If unsure, say unknown. Keep counts approximate and avoid claims that require tracking beyond the single image.\n"
        "Return one JSON object only with exactly these keys:\n"
        "schema_version, scene_summary, visible_entities, text_visible, image_quality, safety_notes, cited_event_ids, cited_artifact_ids, limitations.\n"
        "schema_version must be monitorme.qwen_vlm_keyframe.v0.3.\n"
        "visible_entities must be a JSON array of objects with keys: label, description, confidence_hint, location_hint.\n"
        "confidence_hint must be one of: low, medium, high, unknown.\n"
        "cited_event_ids and cited_artifact_ids must contain only IDs from supplied JSON refs.\n\n"
        "JSON refs:\n"
        f"{json.dumps(facts, indent=2, sort_keys=True)}"
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


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("VLM output is not JSON")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("VLM output JSON must be an object")
    return obj


def validate_qwen_keyframe_json(
    data: dict[str, Any],
    *,
    event: dict[str, Any],
    related_events: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "scene_summary",
        "visible_entities",
        "text_visible",
        "image_quality",
        "safety_notes",
        "cited_event_ids",
        "cited_artifact_ids",
        "limitations",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Qwen VLM keyframe JSON missing required keys: {missing}")
    extra = sorted(set(data) - required)
    if extra:
        raise ValueError(f"Qwen VLM keyframe JSON contains unsupported keys: {extra}")
    if data["schema_version"] != "monitorme.qwen_vlm_keyframe.v0.3":
        raise ValueError("Qwen VLM schema_version must be monitorme.qwen_vlm_keyframe.v0.3")
    for key in ["scene_summary", "text_visible", "image_quality", "safety_notes"]:
        if not isinstance(data.get(key), str):
            raise ValueError(f"Qwen VLM field must be a string: {key}")
    if not isinstance(data.get("limitations"), list) or not all(isinstance(item, str) for item in data["limitations"]):
        raise ValueError("Qwen VLM limitations must be a JSON array of strings")
    entities = data.get("visible_entities")
    if not isinstance(entities, list):
        raise ValueError("Qwen VLM visible_entities must be a JSON array")
    allowed_entity_keys = {"label", "description", "confidence_hint", "location_hint"}
    for idx, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError(f"Qwen VLM visible_entities[{idx}] must be an object")
        if set(entity) != allowed_entity_keys:
            raise ValueError(f"Qwen VLM visible_entities[{idx}] keys must be {sorted(allowed_entity_keys)}")
        if entity.get("confidence_hint") not in {"low", "medium", "high", "unknown"}:
            raise ValueError("Qwen VLM confidence_hint must be low, medium, high, or unknown")
        for key in ["label", "description", "location_hint"]:
            if not isinstance(entity.get(key), str):
                raise ValueError(f"Qwen VLM visible entity field must be a string: {key}")

    cited_event_ids = data.get("cited_event_ids")
    if not isinstance(cited_event_ids, list) or not all(isinstance(item, str) for item in cited_event_ids):
        raise ValueError("Qwen VLM cited_event_ids must be a JSON array of strings")
    allowed_event_ids = {str(event.get("event_id"))}
    allowed_event_ids.update(str(row.get("event_id")) for row in related_events if row.get("event_id"))
    unknown_events = sorted(set(cited_event_ids) - allowed_event_ids)
    if unknown_events:
        raise ValueError(f"Qwen VLM cited_event_ids contain unknown event IDs: {unknown_events}")
    if not cited_event_ids and event.get("event_id"):
        raise ValueError("Qwen VLM analysis must cite at least one supplied event_id")

    cited_artifact_ids = data.get("cited_artifact_ids")
    if not isinstance(cited_artifact_ids, list) or not all(isinstance(item, str) for item in cited_artifact_ids):
        raise ValueError("Qwen VLM cited_artifact_ids must be a JSON array of strings")
    allowed_artifact_ids = {str(artifact.get("artifact_id"))} if artifact.get("artifact_id") else set()
    allowed_artifact_ids.update(str(row.get("artifact_id")) for row in related_events if row.get("artifact_id"))
    unknown_artifacts = sorted(set(cited_artifact_ids) - allowed_artifact_ids)
    if unknown_artifacts:
        raise ValueError(f"Qwen VLM cited_artifact_ids contain unknown artifact IDs: {unknown_artifacts}")
    if artifact.get("artifact_id") and str(artifact.get("artifact_id")) not in cited_artifact_ids:
        raise ValueError("Qwen VLM analysis must cite the supplied keyframe artifact_id")

    text = json.dumps(data, sort_keys=True).lower()
    blocked_terms = [
        "recognized",
        "identity",
        "identified as",
        "face recognition",
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
        raise ValueError(f"Qwen VLM analysis contains unsupported claim terms: {blocked_hits}")
    data["_validated"] = True
    return data


def qwen_vlm_health(config: QwenVLMConfig | None = None, *, probe: bool = False) -> dict[str, Any]:
    cfg = config or QwenVLMConfig.from_env()
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
        "provider": "qwen-vlm-openai-compatible" if configured else "not_configured",
        "base_url": cfg.base_url,
        "model_id": cfg.model_id,
        "local_endpoint": local_endpoint,
        "allow_remote": cfg.allow_remote,
        "privacy": {
            "runs_after_trigger_only": True,
            "external_upload": bool(configured and not local_endpoint and cfg.allow_remote),
            "raw_frame_upload": False,
            "keyframe_upload_to_local_vlm": bool(configured),
        },
        "probe": probe_result,
    }
