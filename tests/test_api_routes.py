import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from monitor_me.routes import create_app


def test_api_has_root_camera_capture_and_assistant_routes(tmp_path):
    app = create_app(str(tmp_path / "monitorme.db"))
    paths = {route.path for route in app.routes}

    assert "/" in paths
    assert "/health" in paths
    assert "/camera/devices" in paths
    assert "/camera/capture/run" in paths
    assert "/events" in paths
    assert "/artifacts" in paths
    assert "/demo/seed" not in paths


def test_api_post_json_body_routes_do_not_expect_query_req(tmp_path, monkeypatch):
    # Regression coverage for FastAPI/Pydantic on Python 3.12 where local
    # request models plus postponed annotations can otherwise be treated as a
    # query parameter named `req`. The explicit Body(...) annotations must make
    # JSON POSTs work exactly like the README curl commands.
    import monitor_me.routes as routes

    class FakeCaptureRun:
        def as_dict(self):
            return {
                "ok": True,
                "camera_id": "c922_node1_gate",
                "device": "/dev/video0",
                "session_id": "sess_api_test",
                "dataset_path": "data/captures/sess_api_test",
                "manifest_path": "data/captures/sess_api_test/manifest.json",
                "frames_seen": 0,
                "frames_written": 0,
                "motion_event_ids": [],
                "object_event_ids": [],
                "artifact_ids": [],
                "artifact_paths": [],
                "started_at": "2026-06-29T00:00:00+05:30",
                "ended_at": "2026-06-29T00:00:01+05:30",
                "error": None,
            }

    class FakeRunner:
        def __init__(self, db, config):
            self.db = db
            self.config = config

        def run(self):
            return FakeCaptureRun()

    monkeypatch.setattr(routes, "LocalCameraCaptureRunner", FakeRunner)
    app = routes.create_app(str(tmp_path / "monitorme.db"))
    client = TestClient(app)

    capture = client.post(
        "/camera/capture/run",
        json={
            "camera_id": "c922_node1_gate",
            "device": "/dev/video0",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "fourcc": "MJPG",
            "duration_sec": 1,
            "motion_threshold": 1.5,
        },
    )
    assert capture.status_code == 200, capture.text
    assert capture.json()["ok"] is True
    assert capture.json()["session_id"] == "sess_api_test"

    ask = client.post("/assistant/ask", json={"question": "What motion events happened today?"})
    assert ask.status_code == 200, ask.text
    assert "answer" in ask.json()


def test_api_detector_health_route_reports_missing_model(tmp_path):
    app = create_app(str(tmp_path / "monitorme.db"))
    client = TestClient(app)

    response = client.get("/models/detector/health", params={"model_path": str(tmp_path / "missing.onnx"), "load_model": "false"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["exists"] is False
    assert body["ok"] is False
    assert body["privacy"]["camera_opened"] is False
