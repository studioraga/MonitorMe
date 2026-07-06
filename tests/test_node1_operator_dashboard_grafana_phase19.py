from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from monitor_me.grafana_dashboard import (
    build_grafana_dashboard_envelope,
    build_operator_dashboard_prometheus_metrics,
)
from monitor_me.operator_dashboard import build_operator_dashboard_context
from monitor_me.routes import create_app
from tests.test_node1_evidence_api_phase13 import _seed_evidence_index


def test_operator_dashboard_prometheus_metrics_are_facts_only(tmp_path: Path) -> None:
    db_path = tmp_path / "monitorme.db"
    ids = _seed_evidence_index(db_path)
    client = TestClient(create_app(str(db_path)))

    response = client.get("/operator/dashboard/metrics", params={"session_id": ids["session_id"], "fingerprint_limit": 2})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert 'schema="monitorme.operator_dashboard_prometheus.v0.1"' in body
    assert 'facts_only="true"' in body
    assert 'media_decode="false"' in body
    assert 'external_upload="false"' in body
    assert 'native_rerun="false"' in body
    assert "monitorme_operator_profile_count 1" in body
    assert "monitorme_operator_fingerprint_count 2" in body
    assert "monitorme_operator_key_moment_count 1" in body
    assert "monitorme_operator_profile_fingerprints" in body
    assert 'session_id="sess_api_evidence"' in body
    assert 'camera_id="c922_node1_gate"' in body
    assert "monitorme_operator_latency_breakdown_ms" in body
    assert 'stage="Fingerprint"' in body
    assert "monitorme_operator_safety_check_ok" in body
    assert 'check="Facts only"' in body
    assert 'flag="external_upload"} 0' in body
    assert 'flag="media_decode_in_dashboard"} 0' in body
    assert 'flag="native_rerun"} 0' in body


def test_grafana_dashboard_json_route_and_static_config_are_local(tmp_path: Path) -> None:
    client = TestClient(create_app(str(tmp_path / "monitorme.db")))

    root = client.get("/")
    assert root.status_code == 200, root.text
    routes = root.json()["routes"]
    assert routes["operator_dashboard_metrics"] == "GET /operator/dashboard/metrics"
    assert routes["operator_grafana_dashboard"] == "GET /operator/dashboard/grafana/dashboard.json"
    privacy = root.json()["privacy"]
    assert privacy["operator_dashboard_prometheus_metrics"] is True
    assert privacy["operator_dashboard_metrics_external_upload"] is False
    assert privacy["operator_dashboard_grafana_dashboard_json"] is True
    assert privacy["operator_dashboard_grafana_external_datasource"] is False

    response = client.get("/operator/dashboard/grafana/dashboard.json")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    assert payload["schema"] == "monitorme.operator_grafana_dashboard.v0.1"
    assert payload["prometheus"]["scrape_path"] == "/operator/dashboard/metrics"
    assert payload["privacy"]["facts_only"] is True
    assert payload["privacy"]["media_decode_in_api"] is False
    dashboard = payload["dashboard"]
    assert dashboard["uid"] == "monitorme-operator-dashboard"
    assert dashboard["metadata"]["source_metrics_endpoint"] == "/operator/dashboard/metrics"
    exprs = {target["expr"] for panel in dashboard["panels"] for target in panel.get("targets", [])}
    assert "monitorme_operator_profile_count" in exprs
    assert "monitorme_operator_fingerprint_count" in exprs
    assert "monitorme_operator_latency_breakdown_ms" in exprs
    assert "monitorme_operator_operation_audit_runs" in exprs
    assert "monitorme_operator_privacy_flag" in exprs

    static_dashboard = Path("configs/grafana/dashboards/monitorme_operator_dashboard.json")
    static_prometheus = Path("configs/prometheus/monitorme_operator_dashboard.yml")
    static_provider = Path("configs/grafana/provisioning/dashboards/monitorme_operator_dashboard.yml")
    assert static_dashboard.exists()
    assert static_prometheus.exists()
    assert static_provider.exists()
    static_payload = json.loads(static_dashboard.read_text(encoding="utf-8"))
    assert static_payload["uid"] == "monitorme-operator-dashboard"
    assert "/operator/dashboard/metrics" in static_dashboard.read_text(encoding="utf-8")
    assert "127.0.0.1:8088" in static_prometheus.read_text(encoding="utf-8")
    assert "monitorme" in static_provider.read_text(encoding="utf-8")


def test_operator_dashboard_links_grafana_without_external_assets(tmp_path: Path) -> None:
    db_path = tmp_path / "monitorme.db"
    ids = _seed_evidence_index(db_path)
    client = TestClient(create_app(str(db_path)))

    html = client.get("/operator/dashboard", params={"session_id": ids["session_id"], "fingerprint_limit": 2}).text

    assert "/operator/dashboard/metrics" in html
    assert "/operator/dashboard/grafana/dashboard.json" in html
    assert "Prometheus metrics" in html
    assert "Grafana dashboard JSON" in html
    assert "https://" not in html
    assert "http://" not in html
    assert "<script" not in html.lower()


def test_prometheus_builder_handles_empty_context(tmp_path: Path) -> None:
    client = TestClient(create_app(str(tmp_path / "monitorme.db")))
    response = client.get("/operator/dashboard/data")
    assert response.status_code == 200
    metrics = build_operator_dashboard_prometheus_metrics(response.json())
    assert "monitorme_operator_profile_count 0" in metrics
    assert "monitorme_operator_dashboard_info" in metrics
    envelope = build_grafana_dashboard_envelope()
    assert envelope["privacy"]["native_rerun"] is False
    assert envelope["dashboard"]["metadata"]["facts_only"] is True
