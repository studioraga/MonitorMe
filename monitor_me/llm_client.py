from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    model_id: str

    def summarize(self, question: str, evidence: list[dict[str, Any]]) -> str:
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
