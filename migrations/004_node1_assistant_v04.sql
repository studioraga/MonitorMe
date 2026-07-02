PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS smolvlm2_clip_experiments (
    experiment_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    parent_event_id TEXT,
    session_id TEXT,
    camera_id TEXT NOT NULL,
    trigger_frame_id INTEGER,
    clip_artifact_id TEXT,
    clip_path TEXT NOT NULL,
    model_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed', 'skipped')),
    experiment_json TEXT NOT NULL DEFAULT '{}',
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES events(event_id),
    FOREIGN KEY(parent_event_id) REFERENCES events(event_id),
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id),
    FOREIGN KEY(clip_artifact_id) REFERENCES capture_artifacts(artifact_id),
    FOREIGN KEY(model_id) REFERENCES model_registry(model_id)
);

CREATE INDEX IF NOT EXISTS idx_smolvlm2_clip_experiments_event ON smolvlm2_clip_experiments(event_id, created_at);
CREATE INDEX IF NOT EXISTS idx_smolvlm2_clip_experiments_session ON smolvlm2_clip_experiments(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_smolvlm2_clip_experiments_camera ON smolvlm2_clip_experiments(camera_id, created_at);
CREATE INDEX IF NOT EXISTS idx_smolvlm2_clip_experiments_artifact ON smolvlm2_clip_experiments(clip_artifact_id, created_at);
