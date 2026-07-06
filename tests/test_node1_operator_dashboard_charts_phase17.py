from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from monitor_me.db import MonitorMeDB
from monitor_me.operator_dashboard import build_operator_dashboard_context, render_operator_dashboard_html
from monitor_me.routes import create_app
from tests.test_node1_evidence_api_phase13 import _seed_evidence_index


def test_operator_dashboard_chart_context_is_facts_only(tmp_path):
    db_path = tmp_path / "monitorme.db"
    ids = _seed_evidence_index(db_path)
    db = MonitorMeDB(db_path)

    context = build_operator_dashboard_context(db, session_id=ids["session_id"], fingerprint_limit=2)
    charts = context["charts"]

    assert charts["schema"] == "monitorme.operator_dashboard_charts.v0.1"
    assert charts["source"] == "persisted_sqlite_evidence_index_rows"
    assert charts["privacy"]["facts_only"] is True
    assert charts["privacy"]["external_upload"] is False
    assert charts["privacy"]["media_decode_in_dashboard"] is False
    assert charts["privacy"]["native_rerun"] is False
    assert charts["privacy"]["external_chart_assets"] is False
    assert charts["privacy"]["client_side_chart_library"] is False

    assert charts["profile_points"][0]["fingerprints"] == 2
    assert charts["profile_points"][0]["key_moments"] == 1
    assert charts["fingerprint_composition"][0] == {"label": "Media", "value": 2.0, "unit": "fingerprints"}
    assert charts["fingerprint_composition"][1] == {"label": "Synthetic", "value": 0.0, "unit": "fingerprints"}
    assert {row["label"] for row in charts["latency_breakdown_ms"]} >= {"Fingerprint", "Dedup", "Key select", "Safety"}
    assert charts["key_moment_timeline"][0]["rank"] == 1
    assert charts["fingerprint_hamming_sample"][0]["fingerprint_source"] == "decoded_keyframe"
    assert all(item["ok"] for item in charts["safety_checks"])
    assert context["privacy"]["operator_dashboard_charts"] is True
    assert context["privacy"]["operator_dashboard_external_chart_assets"] is False
    db.close()


def test_operator_dashboard_html_renders_local_charts_without_external_assets(tmp_path):
    db_path = tmp_path / "monitorme.db"
    ids = _seed_evidence_index(db_path)
    client = TestClient(create_app(str(db_path)))

    response = client.get("/operator/dashboard", params={"session_id": ids["session_id"], "fingerprint_limit": 2})

    assert response.status_code == 200, response.text
    html = response.text
    assert "<h2>Charts</h2>" in html
    assert "Profiles by fingerprint count" in html
    assert "Fingerprint composition" in html
    assert "Latency breakdown" in html
    assert "Key moment timeline" in html
    assert "Fingerprint nearest-Hamming sample" in html
    assert "Retention / rebuild audit" in html
    assert "Safety checks" in html
    assert "bar-fill" in html
    assert "<svg" in html
    assert "data-no-external-assets=\"true\"" in html
    assert "https://" not in html
    assert "http://" not in html
    assert "<script" not in html.lower()


def test_operator_dashboard_data_route_exposes_chart_model(tmp_path):
    db_path = tmp_path / "monitorme.db"
    ids = _seed_evidence_index(db_path)
    client = TestClient(create_app(str(db_path)))

    root = client.get("/")
    assert root.status_code == 200
    privacy = root.json()["privacy"]
    assert privacy["operator_dashboard_charts"] is True
    assert privacy["operator_dashboard_external_chart_assets"] is False
    assert privacy["operator_dashboard_client_side_chart_library"] is False

    response = client.get("/operator/dashboard/data", params={"session_id": ids["session_id"], "fingerprint_limit": 2})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["charts"]["schema"] == "monitorme.operator_dashboard_charts.v0.1"
    assert payload["charts"]["fingerprint_composition"][0]["value"] == 2.0
    assert payload["charts"]["privacy"]["external_chart_assets"] is False
