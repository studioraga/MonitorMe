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
        },
        "evidence_pipeline_summaries": summaries,
        "selected_profile_id": selected_profile_id,
        "selected_summary": selected_summary,
        "retention_runs": retention_runs,
        "links": {
            "json_data": "/operator/dashboard/data",
            "api_summaries": "/evidence/pipeline/summaries",
            "api_retention_plan": "/evidence/pipeline/retention/plan",
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
        },
    }


def render_operator_dashboard_html(context: dict[str, Any]) -> str:
    cards = context.get("cards") if isinstance(context.get("cards"), dict) else {}
    filters = context.get("filters") if isinstance(context.get("filters"), dict) else {}
    summaries = context.get("evidence_pipeline_summaries") or []
    selected = context.get("selected_summary") if isinstance(context.get("selected_summary"), dict) else None
    retention_runs = context.get("retention_runs") or []

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
      <a href="/docs">OpenAPI docs</a>
    </div>
    <div class="grid">
      {card('Profiles', cards.get('profile_count', 0), 'persisted evidence profiles')}
      {card('Fingerprints', cards.get('fingerprint_count', 0), f"media={cards.get('media_fingerprint_count', 0)} synthetic={cards.get('synthetic_fingerprint_count', 0)}")}
      {card('Key Moments', cards.get('key_moment_count', 0), 'ranked local facts')}
      {card('Dedup Groups', cards.get('duplicate_group_count', 0), f"duplicate clips={cards.get('duplicate_clip_count', 0)}")}
      {card('Safety Violations', cards.get('safety_violation_count', 0), 'expected 0')}
      {card('Retention Runs', cards.get('retention_run_count', 0), 'dry-run/apply audit')}
    </div>
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
  </main>
  <footer>MonitorMe operator dashboard · local SQLite readback · no external assets</footer>
</body>
</html>"""
