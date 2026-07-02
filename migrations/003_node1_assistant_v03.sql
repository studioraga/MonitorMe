PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS vlm_keyframe_analyses (
    analysis_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    parent_event_id TEXT,
    session_id TEXT,
    camera_id TEXT NOT NULL,
    frame_id INTEGER,
    artifact_id TEXT,
    artifact_path TEXT NOT NULL,
    model_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed', 'skipped')),
    analysis_json TEXT NOT NULL DEFAULT '{}',
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES events(event_id),
    FOREIGN KEY(parent_event_id) REFERENCES events(event_id),
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id),
    FOREIGN KEY(artifact_id) REFERENCES capture_artifacts(artifact_id),
    FOREIGN KEY(model_id) REFERENCES model_registry(model_id)
);

CREATE INDEX IF NOT EXISTS idx_vlm_keyframe_analyses_event ON vlm_keyframe_analyses(event_id, created_at);
CREATE INDEX IF NOT EXISTS idx_vlm_keyframe_analyses_session ON vlm_keyframe_analyses(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_vlm_keyframe_analyses_camera ON vlm_keyframe_analyses(camera_id, created_at);
CREATE INDEX IF NOT EXISTS idx_vlm_keyframe_analyses_artifact ON vlm_keyframe_analyses(artifact_id, created_at);
