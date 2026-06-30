from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyDecision:
    decision: str
    reason: str
    policy_version: str = "monitorme-node1-local-v0.1"
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "details": self.details or {},
        }


def allow_node1_local_camera(*, camera_id: str, device: str = "/dev/video0") -> PolicyDecision:
    return PolicyDecision(
        decision="allow",
        reason="Node1 local camera evidence capture is enabled for MonitorMe v0.1",
        details={"camera_id": camera_id, "device": device, "external_upload": False},
    )
