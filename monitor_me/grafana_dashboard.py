"""Local Grafana/Prometheus integration for the operator dashboard.

This module exposes facts-only metrics derived from the same persisted SQLite
readback used by the operator dashboard. It never decodes media, reads frame
pixels, reruns native analysis, uploads artifacts, or emits semantic claims.
"""

from __future__ import annotations

import json
import re
from typing import Any

METRICS_SCHEMA = "monitorme.operator_dashboard_prometheus.v0.1"
GRAFANA_DASHBOARD_SCHEMA = "monitorme.operator_grafana_dashboard.v0.1"


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _bool_value(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _sanitize_metric_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_:]", "_", str(name))
    if not cleaned or cleaned[0].isdigit():
        cleaned = "monitorme_" + cleaned
    return cleaned


def _label_value(value: Any, *, maximum: int = 96) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\").replace("\n", " ").replace('"', '\\"')
    if len(text) > maximum:
        text = text[:maximum]
    return text


def _labels(labels: dict[str, Any] | None = None) -> str:
    if not labels:
        return ""
    parts = []
    for key in sorted(labels):
        if labels[key] is None:
            continue
        parts.append(f'{_sanitize_metric_name(key)}="{_label_value(labels[key])}"')
    return "{" + ",".join(parts) + "}" if parts else ""


class _MetricWriter:
    def __init__(self) -> None:
        self._lines: list[str] = []
        self._declared: set[str] = set()

    def header(self) -> None:
        self._lines.extend(
            [
                "# MonitorMe operator dashboard Prometheus exposition",
                f'# schema="{METRICS_SCHEMA}"',
                '# source="persisted_sqlite_evidence_index_rows"',
                '# facts_only="true"',
                '# media_decode="false"',
                '# external_upload="false"',
                '# native_rerun="false"',
            ]
        )

    def gauge(self, name: str, help_text: str, value: Any, labels: dict[str, Any] | None = None) -> None:
        metric = _sanitize_metric_name(name)
        if metric not in self._declared:
            self._lines.append(f"# HELP {metric} {help_text}")
            self._lines.append(f"# TYPE {metric} gauge")
            self._declared.add(metric)
        self._lines.append(f"{metric}{_labels(labels)} {_as_float(value):.12g}")

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"


def build_operator_dashboard_prometheus_metrics(context: dict[str, Any]) -> str:
    """Render operator dashboard facts as Prometheus text exposition."""
    writer = _MetricWriter()
    writer.header()

    cards = context.get("cards") if isinstance(context.get("cards"), dict) else {}
    privacy = context.get("privacy") if isinstance(context.get("privacy"), dict) else {}
    charts = context.get("charts") if isinstance(context.get("charts"), dict) else {}
    chart_privacy = charts.get("privacy") if isinstance(charts.get("privacy"), dict) else {}

    card_metrics = {
        "profile_count": "Evidence profiles visible in the operator dashboard.",
        "fingerprint_count": "Evidence fingerprints visible in the operator dashboard.",
        "media_fingerprint_count": "Media-derived fingerprints visible in the operator dashboard.",
        "synthetic_fingerprint_count": "Synthetic fingerprints visible in the operator dashboard.",
        "duplicate_group_count": "Duplicate groups visible in the operator dashboard.",
        "duplicate_clip_count": "Duplicate clips visible in the operator dashboard.",
        "key_moment_count": "Key moments visible in the operator dashboard.",
        "safety_violation_count": "Safety validator violations visible in the operator dashboard.",
        "retention_run_count": "Evidence retention audit rows visible in the operator dashboard.",
        "scheduler_run_count": "Scheduled retention audit rows visible in the operator dashboard.",
        "rebuild_run_count": "Evidence index rebuild audit rows visible in the operator dashboard.",
        "retention_schedule_enabled": "Whether the default evidence retention schedule is enabled.",
    }
    for key, help_text in card_metrics.items():
        writer.gauge(f"monitorme_operator_{key}", help_text, cards.get(key, 0))

    for row in charts.get("profile_points") or []:
        labels = {
            "profile_id": row.get("profile_id"),
            "session_id": row.get("session_id"),
            "camera_id": row.get("camera_id"),
            "profile_label": row.get("label"),
        }
        writer.gauge("monitorme_operator_profile_fingerprints", "Fingerprints per evidence profile.", row.get("fingerprints"), labels)
        writer.gauge("monitorme_operator_profile_key_moments", "Key moments per evidence profile.", row.get("key_moments"), labels)
        writer.gauge("monitorme_operator_profile_duplicate_groups", "Duplicate groups per evidence profile.", row.get("duplicate_groups"), labels)
        writer.gauge("monitorme_operator_profile_latency_total_ms", "Total evidence pipeline latency per evidence profile in milliseconds.", row.get("latency_total_ms"), labels)

    for row in charts.get("fingerprint_composition") or []:
        writer.gauge(
            "monitorme_operator_fingerprint_composition",
            "Fingerprint composition buckets for the selected evidence profile.",
            row.get("value"),
            {"bucket": row.get("label"), "unit": row.get("unit")},
        )

    for row in charts.get("latency_breakdown_ms") or []:
        writer.gauge(
            "monitorme_operator_latency_breakdown_ms",
            "Latency breakdown for the selected evidence profile in milliseconds.",
            row.get("value"),
            {"stage": row.get("label")},
        )

    for row in charts.get("key_moment_timeline") or []:
        labels = {"rank": row.get("rank"), "clip_id": row.get("clip_id"), "reason": row.get("reason")}
        writer.gauge("monitorme_operator_key_moment_start_ms", "Key moment start timestamp in milliseconds.", row.get("start_ms"), labels)
        writer.gauge("monitorme_operator_key_moment_priority_score", "Facts-only key moment priority score.", row.get("priority_score"), labels)
        writer.gauge("monitorme_operator_key_moment_motion_score", "Facts-only motion score stored for key moments.", row.get("motion_score"), labels)
        writer.gauge("monitorme_operator_key_moment_lighting_delta", "Facts-only lighting delta stored for key moments.", row.get("lighting_delta"), labels)
        writer.gauge("monitorme_operator_key_moment_changed_pixels", "Changed-pixel count stored for key moments.", row.get("changed_pixels"), labels)

    for row in charts.get("fingerprint_hamming_sample") or []:
        writer.gauge(
            "monitorme_operator_fingerprint_nearest_hamming",
            "Nearest Hamming distance sample for selected fingerprints.",
            row.get("nearest_hamming"),
            {"clip_index": row.get("clip_index"), "fingerprint_source": row.get("fingerprint_source"), "from_media": str(bool(row.get("from_media"))).lower()},
        )

    for row in charts.get("operation_audit") or []:
        writer.gauge(
            "monitorme_operator_operation_audit_runs",
            "Retention, scheduler, and rebuild audit run counts by status bucket.",
            row.get("value"),
            {"bucket": row.get("label")},
        )

    for row in charts.get("safety_checks") or []:
        writer.gauge(
            "monitorme_operator_safety_check_ok",
            "Safety check status for selected evidence profile, 1 means ok.",
            _bool_value(row.get("ok")),
            {"check": row.get("label")},
        )

    privacy_flags = {
        "facts_only": privacy.get("facts_only", True),
        "external_upload": privacy.get("external_upload", False),
        "raw_frame_upload": privacy.get("raw_frame_upload", False),
        "media_decode_in_dashboard": privacy.get("media_decode_in_dashboard", False),
        "media_decode_in_api": privacy.get("media_decode_in_api", False),
        "semantic_claims": privacy.get("semantic_claims", False),
        "identity": privacy.get("identity", False),
        "intent": privacy.get("intent", False),
        "speech_content": privacy.get("speech_content", False),
        "external_chart_assets": chart_privacy.get("external_chart_assets", False),
        "client_side_chart_library": chart_privacy.get("client_side_chart_library", False),
        "native_rerun": chart_privacy.get("native_rerun", False),
    }
    for flag, value in privacy_flags.items():
        writer.gauge("monitorme_operator_privacy_flag", "Operator dashboard privacy and safety flags, 1 means enabled/true.", _bool_value(value), {"flag": flag})

    writer.gauge("monitorme_operator_dashboard_info", "Operator dashboard integration metadata.", 1, {"schema": METRICS_SCHEMA, "source": "persisted_sqlite_evidence_index_rows"})
    return writer.text()


def build_grafana_dashboard_definition() -> dict[str, Any]:
    """Return a Grafana dashboard model backed by the Prometheus metrics endpoint."""
    datasource = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}

    def panel(panel_id: int, title: str, expr: str, x: int, y: int, w: int = 8, h: int = 7, *, unit: str = "short", panel_type: str = "timeseries") -> dict[str, Any]:
        return {
            "id": panel_id,
            "title": title,
            "type": panel_type,
            "datasource": datasource,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
            "targets": [{"refId": "A", "expr": expr, "legendFormat": "{{profile_label}}{{bucket}}{{stage}}{{check}}{{flag}}"}],
            "options": {"legend": {"showLegend": True, "placement": "bottom"}, "tooltip": {"mode": "single"}},
        }

    panels = [
        panel(1, "Evidence profiles", "monitorme_operator_profile_count", 0, 0, 6, 4, panel_type="stat"),
        panel(2, "Fingerprints", "monitorme_operator_fingerprint_count", 6, 0, 6, 4, panel_type="stat"),
        panel(3, "Key moments", "monitorme_operator_key_moment_count", 12, 0, 6, 4, panel_type="stat"),
        panel(4, "Safety violations", "monitorme_operator_safety_violation_count", 18, 0, 6, 4, panel_type="stat"),
        panel(5, "Profiles by fingerprint count", "monitorme_operator_profile_fingerprints", 0, 4, 12, 7, panel_type="barchart"),
        panel(6, "Fingerprint composition", "monitorme_operator_fingerprint_composition", 12, 4, 12, 7, panel_type="piechart"),
        panel(7, "Latency breakdown", "monitorme_operator_latency_breakdown_ms", 0, 11, 12, 7, unit="ms", panel_type="barchart"),
        panel(8, "Key moment priority", "monitorme_operator_key_moment_priority_score", 12, 11, 12, 7),
        panel(9, "Fingerprint nearest Hamming", "monitorme_operator_fingerprint_nearest_hamming", 0, 18, 12, 7, panel_type="barchart"),
        panel(10, "Retention / scheduler / rebuild audit", "monitorme_operator_operation_audit_runs", 12, 18, 12, 7, panel_type="barchart"),
        panel(11, "Safety checks", "monitorme_operator_safety_check_ok", 0, 25, 12, 7, panel_type="barchart"),
        panel(12, "Privacy flags", "monitorme_operator_privacy_flag", 12, 25, 12, 7, panel_type="barchart"),
    ]
    return {
        "schemaVersion": 39,
        "uid": "monitorme-operator-dashboard",
        "title": "MonitorMe Operator Evidence Dashboard",
        "tags": ["monitorme", "node1", "evidence", "facts-only"],
        "timezone": "browser",
        "refresh": "30s",
        "editable": True,
        "graphTooltip": 0,
        "templating": {
            "list": [
                {
                    "name": "DS_PROMETHEUS",
                    "type": "datasource",
                    "query": "prometheus",
                    "current": {"text": "Prometheus", "value": "Prometheus"},
                }
            ]
        },
        "annotations": {"list": []},
        "panels": panels,
        "time": {"from": "now-6h", "to": "now"},
        "metadata": {
            "schema": GRAFANA_DASHBOARD_SCHEMA,
            "source_metrics_endpoint": "/operator/dashboard/metrics",
            "facts_only": True,
            "external_upload": False,
            "media_decode": False,
            "native_rerun": False,
        },
    }


def build_grafana_dashboard_envelope() -> dict[str, Any]:
    return {
        "ok": True,
        "schema": GRAFANA_DASHBOARD_SCHEMA,
        "dashboard": build_grafana_dashboard_definition(),
        "prometheus": {
            "scrape_path": "/operator/dashboard/metrics",
            "job_name": "monitorme_operator_dashboard",
        },
        "privacy": {
            "facts_only": True,
            "external_upload": False,
            "raw_frame_upload": False,
            "media_decode_in_api": False,
            "native_rerun": False,
            "semantic_claims": False,
        },
    }


if __name__ == "__main__":  # pragma: no cover - helper for regenerating static dashboard JSON.
    print(json.dumps(build_grafana_dashboard_definition(), indent=2, sort_keys=True))
