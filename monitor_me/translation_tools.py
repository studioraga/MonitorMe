from __future__ import annotations

LANGUAGE_LABELS = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
}


def operator_friendly_alert(text: str, *, language: str = "en") -> str:
    """Return an operator-friendly alert wrapper.

    v0.1 intentionally avoids machine translation dependencies. It preserves the
    original evidence-backed text and labels the requested language so a future
    local translator model can replace this safely.
    """
    label = LANGUAGE_LABELS.get(language, language)
    if language == "en":
        return text
    return f"[{label} operator alert - translation model not enabled in v0.1] {text}"
