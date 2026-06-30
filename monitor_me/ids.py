from __future__ import annotations

import uuid


def new_id(prefix: str) -> str:
    """Return a compact, sortable-enough identifier with a domain prefix."""
    clean = prefix.strip().lower().replace("_", "-")
    return f"{clean}_{uuid.uuid4().hex[:16]}"
