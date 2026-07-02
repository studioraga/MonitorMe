PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS event_contracts (
    contract_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    parent_event_id TEXT,
    session_id TEXT,
    camera_id TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT '1.0',
    contract_json TEXT NOT NULL,
    policy_decision_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES events(event_id),
    FOREIGN KEY(parent_event_id) REFERENCES events(event_id),
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
);

CREATE INDEX IF NOT EXISTS idx_event_contracts_event ON event_contracts(event_id, created_at);
CREATE INDEX IF NOT EXISTS idx_event_contracts_session ON event_contracts(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_event_contracts_camera ON event_contracts(camera_id, created_at);
