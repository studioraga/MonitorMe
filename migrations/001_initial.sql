PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT,
    source_node TEXT NOT NULL DEFAULT 'node1',
    source_kind TEXT NOT NULL DEFAULT 'local_v4l2',
    device TEXT NOT NULL DEFAULT '/dev/video0',
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'local',
    version TEXT,
    path TEXT,
    sha256 TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS capture_sessions (
    session_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    source_node TEXT NOT NULL DEFAULT 'node1',
    source_kind TEXT NOT NULL DEFAULT 'local_v4l2',
    device TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    manifest_path TEXT,
    dataset_path TEXT,
    frames_seen INTEGER NOT NULL DEFAULT 0,
    frames_written INTEGER NOT NULL DEFAULT 0,
    bytes_written INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    policy_decision_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
);

CREATE TABLE IF NOT EXISTS capture_artifacts (
    artifact_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    media_type TEXT,
    size_bytes INTEGER,
    sha256 TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    parent_event_id TEXT,
    camera_id TEXT NOT NULL,
    session_id TEXT,
    frame_id INTEGER,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info',
    label TEXT,
    confidence REAL,
    bbox_json TEXT,
    track_id TEXT,
    zone_id TEXT,
    source_node TEXT NOT NULL DEFAULT 'node1',
    source_kind TEXT NOT NULL DEFAULT 'local_v4l2',
    model_id TEXT,
    artifact_id TEXT,
    attrs_json TEXT NOT NULL DEFAULT '{}',
    caption TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(parent_event_id) REFERENCES events(event_id),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id),
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id),
    FOREIGN KEY(model_id) REFERENCES model_registry(model_id),
    FOREIGN KEY(artifact_id) REFERENCES capture_artifacts(artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_type_label_ts ON events(event_type, label, ts);
CREATE INDEX IF NOT EXISTS idx_events_session_frame ON events(session_id, frame_id);
CREATE INDEX IF NOT EXISTS idx_events_parent ON events(parent_event_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON capture_artifacts(session_id);

CREATE TABLE IF NOT EXISTS assistant_runs (
    run_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'rejected')),
    model_id TEXT,
    prompt_version TEXT NOT NULL DEFAULT 'monitorme-v0.1',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    answer TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY(model_id) REFERENCES model_registry(model_id)
);

CREATE TABLE IF NOT EXISTS assistant_summaries (
    summary_id TEXT PRIMARY KEY,
    run_id TEXT,
    event_id TEXT,
    session_id TEXT,
    camera_id TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    facts_json TEXT NOT NULL DEFAULT '{}',
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    model_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES assistant_runs(run_id),
    FOREIGN KEY(event_id) REFERENCES events(event_id),
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id),
    FOREIGN KEY(model_id) REFERENCES model_registry(model_id)
);

CREATE TABLE IF NOT EXISTS evidence_packs (
    pack_id TEXT PRIMARY KEY,
    event_id TEXT,
    session_id TEXT,
    camera_id TEXT NOT NULL,
    pack_path TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    sha256 TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES events(event_id),
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
);

CREATE TABLE IF NOT EXISTS incident_reports (
    report_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    start_ts TEXT,
    end_ts TEXT,
    title TEXT NOT NULL,
    severity TEXT NOT NULL,
    event_ids_json TEXT NOT NULL DEFAULT '[]',
    session_ids_json TEXT NOT NULL DEFAULT '[]',
    evidence_pack_ids_json TEXT NOT NULL DEFAULT '[]',
    report_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
);

CREATE TABLE IF NOT EXISTS event_feedback (
    feedback_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('useful', 'false_positive', 'needs_review', 'duplicate', 'bad_bbox', 'wrong_label')),
    reason TEXT,
    operator TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_event ON event_feedback(event_id);
CREATE INDEX IF NOT EXISTS idx_feedback_label_created ON event_feedback(label, created_at);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    actor TEXT,
    outcome TEXT NOT NULL,
    camera_id TEXT,
    event_id TEXT,
    session_id TEXT,
    report_id TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id),
    FOREIGN KEY(event_id) REFERENCES events(event_id),
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id),
    FOREIGN KEY(report_id) REFERENCES incident_reports(report_id)
);
