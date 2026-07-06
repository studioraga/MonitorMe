"""Read-only local operator dashboard for facts-only evidence index summaries."""

from __future__ import annotations

from html import escape
from typing import Any

from .db import MonitorMeDB


def _safe_limit(value: int, *, default: int, minimum: int = 1, maximum: int = 100) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(parsed, maximum))


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f} ms"
    except Exception:
        return str(value)


def _fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _summary_count(summary: dict[str, Any], name: str) -> int:
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    return _as_int(counts.get(name))




def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _clip_float(value: Any, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    parsed = _as_float(value)
    return max(minimum, min(parsed, maximum))


def _chart_bar(label: str, value: Any, *, unit: str = "count") -> dict[str, Any]:
    return {"label": str(label), "value": _as_float(value), "unit": unit}


def _build_dashboard_charts(
    *,
    summaries: list[dict[str, Any]],
    selected_summary: dict[str, Any] | None,
    retention_runs: list[dict[str, Any]],
    scheduler_runs: list[dict[str, Any]],
    rebuild_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build local chart-ready facts for the operator dashboard.

    The chart data is a deterministic projection of already persisted SQLite rows.
    It does not fetch artifacts, decode media, rerun native analysis, or call any
    external charting service.
    """
    profile_points: list[dict[str, Any]] = []
    for idx, item in enumerate(summaries):
        counts = item.get("counts") if isinstance(item.get("counts"), dict) else {}
        latency = item.get("latency") if isinstance(item.get("latency"), dict) else {}
        profile_points.append(
            {
                "profile_id": item.get("profile_id"),
                "session_id": item.get("session_id"),
                "camera_id": item.get("camera_id"),
                "ordinal": idx + 1,
                "label": f"P{idx + 1}",
                "fingerprints": _as_int(counts.get("fingerprint_count")),
                "key_moments": _as_int(counts.get("key_moment_count")),
                "duplicate_groups": _as_int(counts.get("duplicate_group_count")),
                "duplicate_clips": _as_int(counts.get("duplicate_clip_count")),
                "latency_total_ms": _as_float(latency.get("total_ms")),
            }
        )

    selected_counts = selected_summary.get("counts") if isinstance((selected_summary or {}).get("counts"), dict) else {}
    selected_latency = selected_summary.get("latency") if isinstance((selected_summary or {}).get("latency"), dict) else {}
    selected_timeline = selected_summary.get("timeline") if isinstance((selected_summary or {}).get("timeline"), dict) else {}
    fingerprints = selected_summary.get("fingerprints") if isinstance((selected_summary or {}).get("fingerprints"), list) else []
    key_moments = selected_summary.get("key_moments") if isinstance((selected_summary or {}).get("key_moments"), list) else []

    fingerprint_composition = [
        _chart_bar("Media", selected_counts.get("media_fingerprint_count"), unit="fingerprints"),
        _chart_bar("Synthetic", selected_counts.get("synthetic_fingerprint_count"), unit="fingerprints"),
    ]
    if _as_int(selected_counts.get("duplicate_clip_count")):
        fingerprint_composition.append(_chart_bar("Duplicate clips", selected_counts.get("duplicate_clip_count"), unit="clips"))

    latency_breakdown = [
        _chart_bar("Manifest", selected_latency.get("manifest_scan_ms"), unit="ms"),
        _chart_bar("Batch plan", selected_latency.get("batch_plan_ms"), unit="ms"),
        _chart_bar("Fingerprint", selected_latency.get("fingerprint_ms"), unit="ms"),
        _chart_bar("Dedup", selected_latency.get("dedup_ms"), unit="ms"),
        _chart_bar("Key select", selected_latency.get("key_selection_ms"), unit="ms"),
        _chart_bar("Safety", selected_latency.get("safety_validation_ms"), unit="ms"),
    ]

    key_moment_timeline = []
    for item in key_moments:
        key_moment_timeline.append(
            {
                "rank": _as_int(item.get("rank")),
                "clip_id": item.get("clip_id"),
                "start_ms": _as_int(item.get("start_ms")),
                "duration_ms": _as_int(item.get("duration_ms")),
                "priority_score": _as_float(item.get("priority_score")),
                "motion_score": _as_float(item.get("motion_score")),
                "lighting_delta": _as_float(item.get("lighting_delta")),
                "changed_pixels": _as_int(item.get("changed_pixels")),
                "reason": item.get("reason"),
            }
        )

    hamming_points = []
    for fp in fingerprints:
        hamming_points.append(
            {
                "clip_index": _as_int(fp.get("clip_index")),
                "nearest_hamming": _as_int(fp.get("nearest_hamming")),
                "fingerprint_source": fp.get("fingerprint_source"),
                "from_media": bool(fp.get("from_media")),
            }
        )

    def count_status(rows: list[dict[str, Any]], status: str) -> int:
        return sum(1 for row in rows if str(row.get("status") or "") == status)

    operation_audit = [
        {"label": "Retention dry-run", "value": count_status(retention_runs, "dry_run"), "unit": "runs"},
        {"label": "Retention completed", "value": count_status(retention_runs, "completed"), "unit": "runs"},
        {"label": "Scheduler dry-run", "value": count_status(scheduler_runs, "dry_run"), "unit": "runs"},
        {"label": "Scheduler skipped", "value": count_status(scheduler_runs, "skipped"), "unit": "runs"},
        {"label": "Rebuild dry-run", "value": count_status(rebuild_runs, "dry_run"), "unit": "runs"},
        {"label": "Rebuild completed", "value": count_status(rebuild_runs, "completed"), "unit": "runs"},
    ]

    safety = selected_summary.get("safety") if isinstance((selected_summary or {}).get("safety"), dict) else {}
    safety_checks = [
        {"label": "Manifest", "ok": bool(safety.get("manifest_ok", selected_summary is not None))},
        {"label": "Fingerprints", "ok": bool(safety.get("fingerprint_ok", selected_summary is not None))},
        {"label": "Dedup", "ok": bool(safety.get("dedup_ok", selected_summary is not None))},
        {"label": "Key moments", "ok": bool(safety.get("key_moments_ok", selected_summary is not None))},
        {"label": "Timeline", "ok": bool(safety.get("timeline_ok", selected_summary is not None))},
        {"label": "Facts only", "ok": bool(safety.get("facts_only", selected_summary is not None))},
    ]

    return {
        "schema": "monitorme.operator_dashboard_charts.v0.1",
        "source": "persisted_sqlite_evidence_index_rows",
        "profile_points": profile_points,
        "fingerprint_composition": fingerprint_composition,
        "latency_breakdown_ms": latency_breakdown,
        "key_moment_timeline": key_moment_timeline,
        "fingerprint_hamming_sample": hamming_points,
        "operation_audit": operation_audit,
        "safety_checks": safety_checks,
        "timeline_span_ms": _as_int(selected_timeline.get("timeline_span_ms")),
        "privacy": {
            "facts_only": True,
            "external_upload": False,
            "raw_frame_upload": False,
            "media_decode_in_dashboard": False,
            "native_rerun": False,
            "external_chart_assets": False,
            "client_side_chart_library": False,
            "semantic_claims": False,
        },
    }

def build_operator_dashboard_context(
    db: MonitorMeDB,
    *,
    session_id: str | None = None,
    camera_id: str | None = None,
    profile_id: str | None = None,
    limit: int = 10,
    fingerprint_limit: int = 5,
    retention_limit: int = 5,
) -> dict[str, Any]:
    """Build the facts-only dashboard model from persisted evidence-index rows.

    The dashboard is intentionally read-only. It does not decode media, fetch
    artifact contents, upload data, or run retention deletes.
    """
    safe_limit = _safe_limit(limit, default=10, maximum=50)
    safe_fingerprint_limit = _safe_limit(fingerprint_limit, default=5, maximum=50)
    safe_retention_limit = _safe_limit(retention_limit, default=5, maximum=50)
    summaries = db.summarize_evidence_pipeline_profiles(
        profile_id=profile_id,
        session_id=session_id,
        camera_id=camera_id,
        limit=safe_limit,
        detailed=True,
    )
    selected_profile_id = profile_id or (str(summaries[0].get("profile_id")) if summaries else None)
    selected_summary: dict[str, Any] | None = None
    if selected_profile_id:
        selected_summary = db.get_evidence_pipeline_summary(
            selected_profile_id,
            include_fingerprints=True,
            include_dedup_groups=True,
            include_key_moments=True,
            fingerprint_limit=safe_fingerprint_limit,
            detailed=True,
        )

    retention_runs = db.list_evidence_retention_runs(limit=safe_retention_limit)
    retention_schedule = db.get_evidence_retention_schedule("default")
    scheduler_runs = db.list_evidence_retention_scheduler_runs(limit=safe_retention_limit)
    rebuild_runs = db.list_evidence_index_rebuild_runs(limit=safe_retention_limit)
    charts = _build_dashboard_charts(
        summaries=summaries,
        selected_summary=selected_summary,
        retention_runs=retention_runs,
        scheduler_runs=scheduler_runs,
        rebuild_runs=rebuild_runs,
    )
    profile_count = len(summaries)
    fingerprint_count = sum(_summary_count(item, "fingerprint_count") for item in summaries)
    media_fingerprint_count = sum(_summary_count(item, "media_fingerprint_count") for item in summaries)
    synthetic_fingerprint_count = sum(_summary_count(item, "synthetic_fingerprint_count") for item in summaries)
    duplicate_group_count = sum(_summary_count(item, "duplicate_group_count") for item in summaries)
    duplicate_clip_count = sum(_summary_count(item, "duplicate_clip_count") for item in summaries)
    key_moment_count = sum(_summary_count(item, "key_moment_count") for item in summaries)
    safety_violation_count = sum(
        _as_int((item.get("ingestion") or {}).get("violation_count")) for item in summaries if isinstance(item.get("ingestion"), dict)
    )
    return {
        "ok": True,
        "schema": "monitorme.operator_dashboard.v0.1",
        "filters": {
            "session_id": session_id,
            "camera_id": camera_id,
            "profile_id": profile_id,
            "limit": safe_limit,
            "fingerprint_limit": safe_fingerprint_limit,
            "retention_limit": safe_retention_limit,
        },
        "cards": {
            "profile_count": profile_count,
            "fingerprint_count": fingerprint_count,
            "media_fingerprint_count": media_fingerprint_count,
            "synthetic_fingerprint_count": synthetic_fingerprint_count,
            "duplicate_group_count": duplicate_group_count,
            "duplicate_clip_count": duplicate_clip_count,
            "key_moment_count": key_moment_count,
            "safety_violation_count": safety_violation_count,
            "retention_run_count": len(retention_runs),
            "scheduler_run_count": len(scheduler_runs),
            "rebuild_run_count": len(rebuild_runs),
            "retention_schedule_enabled": 1 if (retention_schedule or {}).get("enabled") else 0,
        },
        "evidence_pipeline_summaries": summaries,
        "selected_profile_id": selected_profile_id,
        "selected_summary": selected_summary,
        "retention_runs": retention_runs,
        "retention_schedule": retention_schedule,
        "retention_scheduler_runs": scheduler_runs,
        "evidence_index_rebuild_runs": rebuild_runs,
        "charts": charts,
        "links": {
            "json_data": "/operator/dashboard/data",
            "api_summaries": "/evidence/pipeline/summaries",
            "api_retention_plan": "/evidence/pipeline/retention/plan",
            "api_retention_schedule": "/evidence/pipeline/retention/schedule",
            "api_rebuild_plan": "/evidence/pipeline/rebuild/plan",
            "prometheus_metrics": "/operator/dashboard/metrics",
            "grafana_dashboard_json": "/operator/dashboard/grafana/dashboard.json",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
        "privacy": {
            "facts_only": True,
            "external_upload": False,
            "raw_frame_upload": False,
            "media_decode_in_dashboard": False,
            "media_decode_in_api": False,
            "semantic_claims": False,
            "identity": False,
            "intent": False,
            "speech_content": False,
            "destructive_actions_from_dashboard": False,
            "scheduled_retention_visible": True,
            "scheduled_retention_apply_from_dashboard": False,
            "evidence_index_rebuild_visible": True,
            "evidence_index_rebuild_apply_from_dashboard": False,
            "operator_dashboard_charts": True,
            "operator_dashboard_external_chart_assets": False,
            "operator_dashboard_client_side_chart_library": False,
            "operator_dashboard_prometheus_metrics": True,
            "operator_dashboard_metrics_external_upload": False,
            "operator_dashboard_grafana_dashboard_json": True,
            "operator_dashboard_grafana_external_datasource": False,
        },
    }




def _fmt_num(value: Any, *, decimals: int = 2) -> str:
    try:
        number = float(value or 0.0)
    except Exception:
        return str(value)
    if abs(number - int(number)) < 0.000001:
        return str(int(number))
    return f"{number:.{decimals}f}".rstrip("0").rstrip(".")


def _chart_empty(message: str) -> str:
    return f'<div class="chart-empty">{_e(message)}</div>'


def _render_bar_chart(title: str, rows: list[dict[str, Any]], *, value_key: str = "value", max_rows: int = 8) -> str:
    usable = [row for row in rows[:max_rows] if _as_float(row.get(value_key)) >= 0]
    if not usable:
        return f'<section class="chart-card"><h3>{_e(title)}</h3>{_chart_empty("No chart data available.")}</section>'
    max_value = max((_as_float(row.get(value_key)) for row in usable), default=0.0) or 1.0
    bars: list[str] = []
    for row in usable:
        label = row.get("label") or row.get("profile_id") or row.get("clip_id") or "-"
        value = _as_float(row.get(value_key))
        pct = max(2.0, min(100.0, (value / max_value) * 100.0)) if max_value else 0.0
        unit = row.get("unit") or ""
        bars.append(
            '<div class="bar-row">'
            f'<div class="bar-label">{_e(label)}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct:.2f}%"></div>'
            '</div>'
            f'<div class="bar-value">{_e(_fmt_num(value))}{(" " + _e(unit)) if unit else ""}</div>'
            '</div>'
        )
    return f'<section class="chart-card"><h3>{_e(title)}</h3><div class="bar-chart">{"".join(bars)}</div></section>'


def _render_profile_chart(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f'<section class="chart-card"><h3>Profiles by fingerprint count</h3>{_chart_empty("No profiles to chart.")}</section>'
    data = []
    for row in rows[:10]:
        data.append({"label": row.get("label") or row.get("profile_id"), "value": row.get("fingerprints"), "unit": "fingerprints"})
    return _render_bar_chart("Profiles by fingerprint count", data, max_rows=10)


def _render_timeline_chart(rows: list[dict[str, Any]], *, timeline_span_ms: int = 0) -> str:
    if not rows:
        return f'<section class="chart-card wide"><h3>Key moment timeline</h3>{_chart_empty("No key moments to chart.")}</section>'
    span = max(timeline_span_ms, max((_as_int(row.get("start_ms")) for row in rows), default=0), 1)
    width = 640
    height = 140
    top = 22
    base_y = 112
    axis = f'<line x1="24" y1="{base_y}" x2="{width - 18}" y2="{base_y}" class="svg-axis" />'
    points = []
    for row in rows[:12]:
        start = _as_int(row.get("start_ms"))
        priority = _clip_float(row.get("priority_score"), minimum=0.0, maximum=1.0)
        x = 24 + ((width - 42) * start / span)
        y = base_y - max(8, priority * 78)
        rank = _as_int(row.get("rank"))
        reason = row.get("reason") or "key_moment"
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" class="svg-dot"><title>rank {rank}: {_e(reason)} at {start} ms</title></circle>'
            f'<text x="{x:.1f}" y="{max(top, y - 10):.1f}" text-anchor="middle" class="svg-label">{rank}</text>'
        )
    label = f'<text x="24" y="132" class="svg-label">0 ms</text><text x="{width - 18}" y="132" text-anchor="end" class="svg-label">{_e(span)} ms</text>'
    svg = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Key moment timeline">{axis}{"".join(points)}{label}</svg>'
    return f'<section class="chart-card wide"><h3>Key moment timeline</h3><div class="svg-chart">{svg}</div></section>'


def _render_hamming_chart(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f'<section class="chart-card"><h3>Fingerprint nearest-Hamming sample</h3>{_chart_empty("No fingerprint sample to chart.")}</section>'
    data = []
    for row in rows[:10]:
        data.append({"label": f"#{_as_int(row.get('clip_index'))}", "value": row.get("nearest_hamming"), "unit": "bits"})
    return _render_bar_chart("Fingerprint nearest-Hamming sample", data, max_rows=10)


def _render_safety_checks(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f'<section class="chart-card"><h3>Safety checks</h3>{_chart_empty("No safety check data available.")}</section>'
    items = []
    for row in rows:
        ok = bool(row.get("ok"))
        items.append(
            '<div class="check-row">'
            f'<span class="check-dot {"ok" if ok else "bad"}"></span>'
            f'<span>{_e(row.get("label"))}</span>'
            f'<strong>{_e("ok" if ok else "attention")}</strong>'
            '</div>'
        )
    return f'<section class="chart-card"><h3>Safety checks</h3><div class="check-chart">{"".join(items)}</div></section>'


def _render_chart_grid(charts: dict[str, Any]) -> str:
    if not isinstance(charts, dict):
        return ""
    parts = [
        _render_profile_chart(charts.get("profile_points") or []),
        _render_bar_chart("Fingerprint composition", charts.get("fingerprint_composition") or []),
        _render_bar_chart("Latency breakdown", charts.get("latency_breakdown_ms") or []),
        _render_timeline_chart(charts.get("key_moment_timeline") or [], timeline_span_ms=_as_int(charts.get("timeline_span_ms"))),
        _render_hamming_chart(charts.get("fingerprint_hamming_sample") or []),
        _render_bar_chart("Retention / rebuild audit", charts.get("operation_audit") or []),
        _render_safety_checks(charts.get("safety_checks") or []),
    ]
    return '<section class="section"><h2>Charts</h2><div class="chart-grid">' + "".join(parts) + "</div></section>"

def render_operator_dashboard_html(context: dict[str, Any]) -> str:
    cards = context.get("cards") if isinstance(context.get("cards"), dict) else {}
    filters = context.get("filters") if isinstance(context.get("filters"), dict) else {}
    summaries = context.get("evidence_pipeline_summaries") or []
    selected = context.get("selected_summary") if isinstance(context.get("selected_summary"), dict) else None
    retention_runs = context.get("retention_runs") or []
    retention_schedule = context.get("retention_schedule") if isinstance(context.get("retention_schedule"), dict) else {}
    scheduler_runs = context.get("retention_scheduler_runs") or []
    rebuild_runs = context.get("evidence_index_rebuild_runs") or []
    charts = context.get("charts") if isinstance(context.get("charts"), dict) else {}
    chart_grid = _render_chart_grid(charts)

    def card(label: str, value: Any, note: str = "") -> str:
        return (
            '<section class="card">'
            f'<div class="card-label">{_e(label)}</div>'
            f'<div class="card-value">{_e(value)}</div>'
            f'<div class="card-note">{_e(note)}</div>'
            '</section>'
        )

    summary_rows = []
    for item in summaries:
        counts = item.get("counts") if isinstance(item.get("counts"), dict) else {}
        ingestion = item.get("ingestion") if isinstance(item.get("ingestion"), dict) else {}
        latency = item.get("latency") if isinstance(item.get("latency"), dict) else {}
        profile_id = item.get("profile_id")
        summary_rows.append(
            "<tr>"
            f"<td><code>{_e(profile_id)}</code></td>"
            f"<td><code>{_e(item.get('session_id'))}</code></td>"
            f"<td>{_e(item.get('camera_id'))}</td>"
            f"<td>{_e(counts.get('fingerprint_count'))}</td>"
            f"<td>{_e(counts.get('key_moment_count'))}</td>"
            f"<td>{_e(counts.get('duplicate_group_count'))}</td>"
            f"<td>{_e(_fmt_bool(ingestion.get('real_media_ingestion')))}</td>"
            f"<td>{_e(_fmt_bool(ingestion.get('safety_ok')))}</td>"
            f"<td>{_e(_fmt_ms(latency.get('total_ms')))}</td>"
            f"<td><a href=\"/evidence/pipeline/profiles/{_e(profile_id)}/summary\">JSON</a></td>"
            "</tr>"
        )
    summary_table = "".join(summary_rows) or '<tr><td colspan="10">No evidence profiles found. Run capture-run with evidence pipeline enabled.</td></tr>'

    key_rows = []
    fp_rows = []
    group_rows = []
    if selected:
        for km in selected.get("key_moments") or []:
            key_rows.append(
                "<tr>"
                f"<td>{_e(km.get('rank'))}</td>"
                f"<td><code>{_e(km.get('clip_id'))}</code></td>"
                f"<td>{_e(km.get('reason'))}</td>"
                f"<td>{_e(km.get('start_ms'))}</td>"
                f"<td>{_e(km.get('priority_score'))}</td>"
                f"<td>{_e(km.get('lighting_delta'))}</td>"
                "</tr>"
            )
        for fp in selected.get("fingerprints") or []:
            fp_rows.append(
                "<tr>"
                f"<td>{_e(fp.get('clip_index'))}</td>"
                f"<td><code>{_e(fp.get('clip_id'))}</code></td>"
                f"<td>{_e(fp.get('fingerprint_source'))}</td>"
                f"<td>{_e(fp.get('decoded_width'))}×{_e(fp.get('decoded_height'))}</td>"
                f"<td><code>{_e(fp.get('fingerprint_hex'))}</code></td>"
                f"<td>{_e(fp.get('nearest_hamming'))}</td>"
                "</tr>"
            )
        for group in selected.get("dedup_groups") or []:
            group_rows.append(
                "<tr>"
                f"<td>{_e(group.get('group_id'))}</td>"
                f"<td><code>{_e(group.get('representative_clip_id'))}</code></td>"
                f"<td>{_e(group.get('group_size'))}</td>"
                f"<td>{_e(group.get('duplicate_count'))}</td>"
                f"<td>{_e(group.get('min_hamming'))}/{_e(group.get('max_hamming'))}</td>"
                "</tr>"
            )
    key_table = "".join(key_rows) or '<tr><td colspan="6">No key moments in selected profile.</td></tr>'
    fp_table = "".join(fp_rows) or '<tr><td colspan="6">No fingerprint sample in selected profile.</td></tr>'
    group_table = "".join(group_rows) or '<tr><td colspan="5">No duplicate groups in selected profile.</td></tr>'

    retention_rows = []
    for run in retention_runs:
        retention_rows.append(
            "<tr>"
            f"<td><code>{_e(run.get('run_id'))}</code></td>"
            f"<td>{_e(run.get('status'))}</td>"
            f"<td>{_e(_fmt_bool(run.get('dry_run')))}</td>"
            f"<td>{_e(run.get('profiles_selected'))}</td>"
            f"<td>{_e(run.get('created_at'))}</td>"
            "</tr>"
        )
    retention_table = "".join(retention_rows) or '<tr><td colspan="5">No retention runs recorded.</td></tr>'

    scheduler_rows = []
    for run in scheduler_runs:
        scheduler_rows.append(
            "<tr>"
            f"<td><code>{_e(run.get('scheduler_run_id'))}</code></td>"
            f"<td>{_e(run.get('status'))}</td>"
            f"<td>{_e(run.get('reason'))}</td>"
            f"<td>{_e(_fmt_bool(run.get('dry_run')))}</td>"
            f"<td>{_e(run.get('checked_at'))}</td>"
            "</tr>"
        )
    scheduler_table = "".join(scheduler_rows) or '<tr><td colspan="5">No scheduler runs recorded.</td></tr>'

    rebuild_rows = []
    for run in rebuild_runs:
        rebuild_rows.append(
            "<tr>"
            f"<td><code>{_e(run.get('run_id'))}</code></td>"
            f"<td>{_e(run.get('status'))}</td>"
            f"<td>{_e(_fmt_bool(run.get('dry_run')))}</td>"
            f"<td>{_e(run.get('profiles_rebuilt'))}</td>"
            f"<td>{_e(run.get('profiles_failed'))}</td>"
            f"<td>{_e(run.get('created_at'))}</td>"
            "</tr>"
        )
    rebuild_table = "".join(rebuild_rows) or '<tr><td colspan="6">No evidence index rebuild runs recorded.</td></tr>'

    schedule_html = (
        "<dl>"
        f"<dt>enabled</dt><dd>{_e(_fmt_bool(retention_schedule.get('enabled')))}</dd>"
        f"<dt>cadence</dt><dd>{_e(retention_schedule.get('cadence') or '-')}</dd>"
        f"<dt>dry run</dt><dd>{_e(_fmt_bool(retention_schedule.get('dry_run', True)))}</dd>"
        f"<dt>next run after</dt><dd>{_e(retention_schedule.get('next_run_after') or '-')}</dd>"
        f"<dt>last run</dt><dd>{_e(retention_schedule.get('last_run_at') or '-')}</dd>"
        "</dl>"
    )

    selected_profile_id = context.get("selected_profile_id") or ""
    session_filter = filters.get("session_id") or ""
    camera_filter = filters.get("camera_id") or ""

    return f"""<!doctype html>
<html lang="en" data-no-external-assets="true">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MonitorMe Operator Dashboard</title>
  <style>
    :root {{ color-scheme: light; --bg:#f6f8fb; --card:#fff; --ink:#14213d; --muted:#5f6b7a; --line:#d8dee9; --ok:#146c43; --warn:#8a5a00; }}
    body {{ margin:0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:24px 32px; background:#101828; color:white; }}
    h1 {{ margin:0 0 6px; font-size:28px; }}
    h2 {{ margin-top:28px; font-size:20px; }}
    main {{ padding:24px 32px 48px; }}
    .notice {{ background:#e8f5e9; border:1px solid #b7dfbd; color:#1b5e20; padding:12px 16px; border-radius:10px; margin-bottom:18px; }}
    .filters, .links {{ color:var(--muted); font-size:14px; margin:8px 0; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:14px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; box-shadow:0 1px 2px rgba(16,24,40,.05); }}
    .card-label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
    .card-value {{ font-size:26px; font-weight:700; margin-top:4px; }}
    .card-note {{ color:var(--muted); font-size:12px; margin-top:3px; min-height:16px; }}
    .chart-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:14px; }}
    .chart-card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; box-shadow:0 1px 2px rgba(16,24,40,.05); }}
    .chart-card.wide {{ grid-column: 1 / -1; }}
    .chart-card h3 {{ margin:0 0 12px; font-size:16px; }}
    .bar-row {{ display:grid; grid-template-columns: 110px minmax(100px, 1fr) 96px; gap:10px; align-items:center; margin:8px 0; }}
    .bar-label, .bar-value {{ font-size:13px; color:var(--muted); }}
    .bar-value {{ text-align:right; color:var(--ink); }}
    .bar-track {{ height:12px; background:#eef2f7; border-radius:999px; overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:999px; background:#175cd3; }}
    .chart-empty {{ color:var(--muted); font-size:14px; padding:10px 0; }}
    .svg-chart svg {{ width:100%; height:160px; background:#fbfdff; border:1px solid var(--line); border-radius:10px; }}
    .svg-axis {{ stroke:#9aa4b2; stroke-width:1; }}
    .svg-dot {{ fill:#175cd3; stroke:#0b3b8c; stroke-width:1; }}
    .svg-label {{ fill:#5f6b7a; font-size:11px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .check-row {{ display:grid; grid-template-columns:18px 1fr auto; gap:8px; align-items:center; padding:7px 0; border-bottom:1px solid var(--line); }}
    .check-dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
    .check-dot.ok {{ background:var(--ok); }}
    .check-dot.bad {{ background:#b42318; }}
    table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
    th, td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-size:14px; }}
    th {{ background:#eef2f7; font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#465466; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; }}
    a {{ color:#175cd3; text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    .section {{ margin-top:22px; }}
    .read-only {{ color:var(--warn); }}
    footer {{ padding:18px 32px; color:var(--muted); font-size:13px; }}
  </style>
</head>
<body>
  <header>
    <h1>MonitorMe Operator Dashboard</h1>
    <div>Local Node1 evidence pipeline summaries and retention status</div>
  </header>
  <main>
    <div class="notice">
      <strong>Facts-only / local-only UI.</strong>
      This dashboard reads persisted SQLite evidence-index rows only. It does not decode media in the request path, upload raw frames, infer identity, infer intent, inspect speech content, or emit semantic claims.
      <span class="read-only">Retention actions are not executed from this page.</span>
    </div>
    <div class="filters">
      Filters: session=<code>{_e(session_filter or '-')}</code>, camera=<code>{_e(camera_filter or '-')}</code>, selected profile=<code>{_e(selected_profile_id or '-')}</code>
    </div>
    <div class="links">
      JSON: <a href="/operator/dashboard/data">dashboard data</a> ·
      <a href="/evidence/pipeline/summaries">evidence summaries</a> ·
      <a href="/evidence/pipeline/retention/plan">retention plan</a> ·
      <a href="/evidence/pipeline/rebuild/plan">rebuild plan</a> ·
      <a href="/operator/dashboard/metrics">Prometheus metrics</a> ·
      <a href="/operator/dashboard/grafana/dashboard.json">Grafana dashboard JSON</a> ·
      <a href="/docs">OpenAPI docs</a>
    </div>
    <div class="grid">
      {card('Profiles', cards.get('profile_count', 0), 'persisted evidence profiles')}
      {card('Fingerprints', cards.get('fingerprint_count', 0), f"media={cards.get('media_fingerprint_count', 0)} synthetic={cards.get('synthetic_fingerprint_count', 0)}")}
      {card('Key Moments', cards.get('key_moment_count', 0), 'ranked local facts')}
      {card('Dedup Groups', cards.get('duplicate_group_count', 0), f"duplicate clips={cards.get('duplicate_clip_count', 0)}")}
      {card('Safety Violations', cards.get('safety_violation_count', 0), 'expected 0')}
      {card('Retention Runs', cards.get('retention_run_count', 0), 'dry-run/apply audit')}
      {card('Rebuild Runs', cards.get('rebuild_run_count', 0), 'from retained artifacts')}
    </div>
    {chart_grid}
    <section class="section">
      <h2>Evidence pipeline profiles</h2>
      <table><thead><tr><th>Profile</th><th>Session</th><th>Camera</th><th>Fingerprints</th><th>Key Moments</th><th>Dedup Groups</th><th>Real Media</th><th>Safety</th><th>Total</th><th>API</th></tr></thead><tbody>{summary_table}</tbody></table>
    </section>
    <section class="section">
      <h2>Selected profile key moments</h2>
      <table><thead><tr><th>Rank</th><th>Clip</th><th>Reason</th><th>Start ms</th><th>Priority</th><th>Lighting Δ</th></tr></thead><tbody>{key_table}</tbody></table>
    </section>
    <section class="section">
      <h2>Selected profile fingerprint sample</h2>
      <table><thead><tr><th>Index</th><th>Clip</th><th>Source</th><th>Decode</th><th>Fingerprint</th><th>Nearest Hamming</th></tr></thead><tbody>{fp_table}</tbody></table>
    </section>
    <section class="section">
      <h2>Selected profile dedup groups</h2>
      <table><thead><tr><th>Group</th><th>Representative</th><th>Size</th><th>Duplicates</th><th>Hamming</th></tr></thead><tbody>{group_table}</tbody></table>
    </section>
    <section class="section">
      <h2>Retention run audit</h2>
      <table><thead><tr><th>Run</th><th>Status</th><th>Dry Run</th><th>Profiles Selected</th><th>Created</th></tr></thead><tbody>{retention_table}</tbody></table>
    </section>
    <section class="section">
      <h2>Evidence index rebuild audit</h2>
      <table><thead><tr><th>Run</th><th>Status</th><th>Dry Run</th><th>Profiles Rebuilt</th><th>Failed</th><th>Created</th></tr></thead><tbody>{rebuild_table}</tbody></table>
    </section>
  </main>
  <footer>MonitorMe operator dashboard · local SQLite readback · no external assets</footer>
</body>
</html>"""
