from __future__ import annotations

from datetime import datetime, timezone


def now_iso() -> str:
    """Return an ISO-8601 timestamp with seconds precision and local offset."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
