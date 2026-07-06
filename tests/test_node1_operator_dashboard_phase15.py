from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from monitor_me.db import MonitorMeDB
from monitor_me.operator_dashboard import build_operator_dashboard_context, render_operator_dashboard_html
from monitor_me.routes import create_app
from tests.test_node1_evidence_api_phase13 import _seed_evidence_index


def test_operator_dashboard_routes_are_registered(tmp_path):
    app = create_app(str(tmp_path / "monitorme.db"))
    paths = {route.path for route in app.routes}

    assert "/operator/dashboard" in paths
    assert "/operator/dashboard/data" in paths
    root = TestClient(app).get("/")
    assert root.status_code == 200, root.text
    routes = root.json()["routes"]
    assert routes["operator_dashboard"] == "GET /operator/dashboard"
    assert routes["operator_dashboard_data"] == "GET /operator/dashboard/data"
    privacy = root.json()["privacy"]
    assert privacy["operator_dashboard_enabled"] is True
    assert privacy["operator_dashboard_external_assets"] is False
    assert privacy["operator_dashboard_media_decode"] is False
    assert privacy["operator_dashboard_destructive_actions"] is False


def test_operator_dashboard_data_is_facts_only_readback(tmp_path):
    db_path = tmp_path / "monitorme.db"
    ids = _seed_evidence_index(db_path)
    client = TestClient(create_app(str(db_path)))

    response = client.get("/operator/dashboard/data", params={"session_id": ids["session_id"], "fingerprint_limit": 1})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    assert payload["schema"] == "monitorme.operator_dashboard.v0.1"
    assert payload["cards"]["profile_count"] == 1
    assert payload["cards"]["fingerprint_count"] == 2
    assert payload["cards"]["media_fingerprint_count"] == 2
    assert payload["cards"]["synthetic_fingerprint_count"] == 0
    assert payload["cards"]["key_moment_count"] == 1
    assert payload["cards"]["duplicate_group_count"] == 1
    assert payload["cards"]["safety_violation_count"] == 0
    assert payload["selected_profile_id"] == ids["profile_id"]
    assert payload["selected_summary"]["profile_id"] == ids["profile_id"]
    assert payload["selected_summary"]["fingerprints_truncated"] is True
    assert payload["selected_summary"]["fingerprints"][0]["fingerprint_source"] == "decoded_keyframe"
    assert payload["privacy"]["facts_only"] is True
    assert payload["privacy"]["external_upload"] is False
    assert payload["privacy"]["media_decode_in_dashboard"] is False
    assert payload["privacy"]["destructive_actions_from_dashboard"] is False


def test_operator_dashboard_html_renders_local_read_only_ui(tmp_path):
    db_path = tmp_path / "monitorme.db"
    ids = _seed_evidence_index(db_path)
    client = TestClient(create_app(str(db_path)))

    response = client.get("/operator/dashboard", params={"session_id": ids["session_id"], "fingerprint_limit": 1})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "MonitorMe Operator Dashboard" in html
    assert "Facts-only / local-only UI" in html
    assert ids["profile_id"] in html
    assert ids["session_id"] in html
    assert "decoded_keyframe" in html
    assert "priority_score" in html
    assert "No external assets" not in html  # footer uses lower-case; avoid brittle title-case check.
    assert "data-no-external-assets=\"true\"" in html
    assert "Retention actions are not executed from this page" in html
    assert "https://" not in html
    assert "http://" not in html
    assert "/evidence/pipeline/profiles/" in html


def test_operator_dashboard_renderer_handles_empty_database(tmp_path):
    db = MonitorMeDB(tmp_path / "monitorme.db")
    context = build_operator_dashboard_context(db, limit=5, fingerprint_limit=2)
    html = render_operator_dashboard_html(context)

    assert context["ok"] is True
    assert context["cards"]["profile_count"] == 0
    assert "No evidence profiles found" in html
    assert "facts-only" in html.lower()
    assert context["privacy"]["media_decode_in_dashboard"] is False
    db.close()
