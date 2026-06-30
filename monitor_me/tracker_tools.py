from __future__ import annotations

from .db import MonitorMeDB


class TrackerTools:
    def __init__(self, db: MonitorMeDB):
        self.db = db

    def mark_event(self, event_id: str, *, label: str, reason: str = "", operator: str = "operator") -> str:
        return self.db.create_feedback(event_id, label=label, reason=reason, operator=operator)

    def false_positive_tracker(self, *, limit: int = 100) -> list[dict]:
        return self.db.list_feedback(label="false_positive", limit=limit)

    def all_feedback(self, *, limit: int = 100) -> list[dict]:
        return self.db.list_feedback(limit=limit)
