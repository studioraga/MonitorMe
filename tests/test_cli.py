from __future__ import annotations

import subprocess
import sys

from tests.helpers import make_real_motion_capture


def test_cli_init_events_and_ask_without_demo_seed(tmp_path):
    db_path = tmp_path / "monitorme.db"
    db, _, event = make_real_motion_capture(tmp_path)
    db.close()

    subprocess.run([sys.executable, "-m", "monitor_me.cli", "--db", str(db_path), "init-db"], check=True, capture_output=True, text=True)
    events = subprocess.run([sys.executable, "-m", "monitor_me.cli", "--db", str(db_path), "events", "--event-type", "motion_detected"], check=True, capture_output=True, text=True)
    assert event["event_id"] in events.stdout
    ask = subprocess.run([sys.executable, "-m", "monitor_me.cli", "--db", str(db_path), "ask", "What motion events happened today?"], check=True, capture_output=True, text=True)
    assert "event_id=" in ask.stdout
    assert "session_id=" in ask.stdout
