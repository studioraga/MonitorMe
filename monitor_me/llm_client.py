from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    model_id: str

    def summarize(self, question: str, evidence: list[dict[str, Any]]) -> str:
        ...


class EventSummaryLLMClient(Protocol):
    model_id: str

    def summarize_event_contract(
        self,
        *,
        event_contract: dict[str, Any],
        policy_decision: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ...


@dataclass
class NullLLMClient:
    """Safe local fallback used when Gemma/MAX is not configured."""

    model_id: str = "deterministic-null-llm"

    def summarize(self, question: str, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return "I do not have local evidence for that request."
        labels = sorted({str(item.get("label")) for item in evidence if item.get("label")})
        return f"Local evidence contains {len(evidence)} referenced item(s) with labels: {', '.join(labels) or 'none'}."


@dataclass
class FakeLLMClient:
    """Test double that can be configured to return safe or unsafe text."""

    response: str
    model_id: str = "fake-test-llm"

    def summarize(self, question: str, evidence: list[dict[str, Any]]) -> str:
        return self.response


@dataclass(frozen=True)
class GemmaMaxConfig:
    """Config for MAX's local OpenAI-compatible chat-completions endpoint."""

    base_url: str = "http://127.0.0.1:8000/v1"
    model_id: str = "google/gemma-3-1b-it"
    api_key: str = "EMPTY"
    timeout_sec: float = 60.0
    max_tokens: int = 192
    temperature: float = 0.0
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "GemmaMaxConfig":
        provider = os.environ.get("MONITORME_LLM_PROVIDER", "").strip().lower()
        enabled_raw = os.environ.get("MONITORME_ASSISTANT_USE_GEMMA", "").strip().lower()
        enabled = provider in {"max", "max-openai", "gemma-max", "openai-compatible"} or enabled_raw in {"1", "true", "yes", "on"}
        return cls(
            base_url=os.environ.get("MONITORME_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/"),
            model_id=os.environ.get("MONITORME_LLM_MODEL_ID", "google/gemma-3-1b-it"),
            api_key=os.environ.get("MONITORME_LLM_API_KEY", "EMPTY"),
            timeout_sec=float(os.environ.get("MONITORME_LLM_TIMEOUT_SEC", "60")),
            max_tokens=int(os.environ.get("MONITORME_LLM_MAX_TOKENS", "192")),
            temperature=float(os.environ.get("MONITORME_LLM_TEMPERATURE", "0.0")),
            enabled=enabled,
        )


class GemmaMaxClient:
    """Local Gemma/MAX client using OpenAI-compatible chat completions.

    The client asks Gemma for strict JSON only. It does not decide capture actions
    and it never receives raw CCTV frames; it receives stored event contracts,
    deterministic policy decisions, and artifact metadata only.
    """

    prompt_version = "monitorme-gemma-summary-json-v0.2"

    def __init__(self, config: GemmaMaxConfig | None = None):
        self.config = config or GemmaMaxConfig.from_env()
        self.model_id = self.config.model_id

    def summarize_event_contract(
        self,
        *,
        event_contract: dict[str, Any],
        policy_decision: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = build_gemma_event_summary_prompt(event_contract, policy_decision, evidence)
        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": prompt}],
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
            raise RuntimeError(f"Gemma/MAX request failed: {exc}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemma/MAX response did not contain choices[0].message.content") from exc
        parsed = parse_json_object(content)
        return validate_gemma_summary_json(parsed, event_contract=event_contract, policy_decision=policy_decision, evidence=evidence)


def build_default_event_summary_llm() -> EventSummaryLLMClient | None:
    config = GemmaMaxConfig.from_env()
    if not config.enabled:
        return None
    return GemmaMaxClient(config)


def build_gemma_event_summary_prompt(event_contract: dict[str, Any], policy_decision: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    allowed = {
        "event_contract": event_contract,
        "policy_decision": policy_decision,
        "evidence_refs": evidence,
    }
    return (
        "You are MonitorMe's local Node1 CCTV evidence summarizer.\n"
        "Use only the JSON facts supplied below. Do not infer identity, intent, danger, weapons, face recognition, or suspicious behavior.\n"
        "Do not create new event IDs, labels, cameras, timestamps, model IDs, bounding boxes, or actions.\n"
        "The deterministic policy already decided the action; explain it, do not override it.\n"
        "Return one JSON object only, with exactly these keys:\n"
        "operator_summary, event_reason, dashboard_tag, recommended_next_step, severity_label, cited_event_ids.\n"
        "severity_label must be one of: info, review, urgent.\n"
        "cited_event_ids must be a JSON array containing only event_id values from the supplied facts.\n"
        "Keep operator_summary under 3 sentences.\n\n"
        "JSON facts:\n"
        f"{json.dumps(allowed, indent=2, sort_keys=True)}"
    )


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
            raise ValueError("LLM output is not JSON")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM output JSON must be an object")
    return obj


def validate_gemma_summary_json(
    data: dict[str, Any],
    *,
    event_contract: dict[str, Any],
    policy_decision: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    required = {
        "operator_summary",
        "event_reason",
        "dashboard_tag",
        "recommended_next_step",
        "severity_label",
        "cited_event_ids",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Gemma summary missing required keys: {missing}")
    extra = sorted(set(data) - required)
    if extra:
        raise ValueError(f"Gemma summary contains unsupported keys: {extra}")
    for key in ["operator_summary", "event_reason", "dashboard_tag", "recommended_next_step", "severity_label"]:
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"Gemma summary field must be a non-empty string: {key}")
    severity = data["severity_label"].strip().lower()
    if severity not in {"info", "review", "urgent"}:
        raise ValueError("Gemma severity_label must be one of: info, review, urgent")
    data["severity_label"] = severity
    cited = data.get("cited_event_ids")
    if not isinstance(cited, list) or not all(isinstance(item, str) for item in cited):
        raise ValueError("Gemma cited_event_ids must be a JSON array of strings")
    allowed_event_ids = {str(event_contract.get("event_id"))}
    if event_contract.get("motion_event_id"):
        allowed_event_ids.add(str(event_contract.get("motion_event_id")))
    allowed_event_ids.update(str(item.get("event_id")) for item in evidence if item.get("event_id"))
    unknown = sorted(set(cited) - allowed_event_ids)
    if unknown:
        raise ValueError(f"Gemma cited_event_ids contain unknown event IDs: {unknown}")
    if not cited and event_contract.get("event_id"):
        raise ValueError("Gemma summary must cite at least one supplied event_id")

    supplied_labels = {str(event_contract.get("label") or "").lower()}
    supplied_labels.update(str(det.get("class_name") or "").lower() for det in event_contract.get("detections") or [])
    supplied_labels.update(str(item.get("label") or "").lower() for item in evidence)
    text = "\n".join(str(data[key]) for key in ["operator_summary", "event_reason", "dashboard_tag", "recommended_next_step"]).lower()
    blocked = ["weapon", "gun", "knife", "face", "recognized", "identity", "intent", "suspicious", "threat"]
    blocked_hits = [term for term in blocked if term in text]
    if blocked_hits:
        raise ValueError(f"Gemma summary contains unsupported claim terms: {blocked_hits}")

    # Keep the next step bounded to the deterministic policy action vocabulary.
    policy_action = str(policy_decision.get("action") or "")
    if policy_action and policy_action not in data["recommended_next_step"] and policy_action.replace("_", " ") not in data["recommended_next_step"].lower():
        # Allow concise human wording but require the policy action to remain in facts for audit.
        data["recommended_next_step"] = f"{data['recommended_next_step']} (policy_action={policy_action})"
    data["_validated"] = True
    return data

def gemma_max_health(config: GemmaMaxConfig | None = None, *, probe: bool = False) -> dict[str, Any]:
    cfg = config or GemmaMaxConfig.from_env()
    probe_result: dict[str, Any] = {"requested": bool(probe), "ok": None, "error": None, "models": []}
    if probe:
        request = urllib.request.Request(
            f"{cfg.base_url}/models",
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(cfg.timeout_sec, 10.0)) as response:  # noqa: S310 - local configurable endpoint
                body = json.loads(response.read().decode("utf-8"))
            models = body.get("data", []) if isinstance(body, dict) else []
            probe_result = {
                "requested": True,
                "ok": True,
                "error": None,
                "models": [item.get("id") for item in models if isinstance(item, dict) and item.get("id")],
            }
        except Exception as exc:  # pragma: no cover - exercised by Node1 live validation
            probe_result = {"requested": True, "ok": False, "error": str(exc), "models": []}
    configured = bool(cfg.enabled)
    ok = configured and (not probe or bool(probe_result.get("ok")))
    return {
        "ok": ok,
        "enabled": configured,
        "provider": "max-openai-compatible" if configured else "not_configured",
        "base_url": cfg.base_url,
        "model_id": cfg.model_id,
        "timeout_sec": cfg.timeout_sec,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "probe": probe_result,
        "privacy": {
            "raw_frame_upload": False,
            "external_upload": False,
            "input": "structured event contracts, policy decisions, and artifact metadata only",
        },
        "next_steps": [
            "Start MAX with scripts/max/term1_start_max_gemma3_1b.sh." if configured else "Set MONITORME_LLM_PROVIDER=max-openai or MONITORME_ASSISTANT_USE_GEMMA=1 to enable Gemma/MAX summaries.",
            "Run scripts/max/term2_validate_max_gemma3_1b.sh to validate the standalone MAX server.",
            "Run scripts/max/term2_validate_monitorme_gemma_v02_live.sh to validate MonitorMe strict JSON summaries against MAX.",
        ],
    }
