PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS evidence_retention_runs (
    run_id TEXT PRIMARY KEY,
    dry_run INTEGER NOT NULL DEFAULT 1 CHECK (dry_run IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'planned',
    policy_json TEXT NOT NULL DEFAULT '{}',
    cutoff_at TEXT,
    profiles_scanned INTEGER NOT NULL DEFAULT 0,
    profiles_selected INTEGER NOT NULL DEFAULT 0,
    fingerprints_selected INTEGER NOT NULL DEFAULT 0,
    dedup_groups_selected INTEGER NOT NULL DEFAULT 0,
    key_moments_selected INTEGER NOT NULL DEFAULT 0,
    index_payload_bytes_estimate INTEGER NOT NULL DEFAULT 0,
    db_size_before_bytes INTEGER NOT NULL DEFAULT 0,
    db_size_after_bytes INTEGER NOT NULL DEFAULT 0,
    wal_checkpoint INTEGER NOT NULL DEFAULT 0 CHECK (wal_checkpoint IN (0, 1)),
    vacuum_requested INTEGER NOT NULL DEFAULT 0 CHECK (vacuum_requested IN (0, 1)),
    vacuum_completed INTEGER NOT NULL DEFAULT 0 CHECK (vacuum_completed IN (0, 1)),
    selected_profiles_json TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_evidence_retention_runs_created ON evidence_retention_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_retention_runs_status ON evidence_retention_runs(status, dry_run);
