PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS evidence_index_rebuild_runs (
    run_id TEXT PRIMARY KEY,
    dry_run INTEGER NOT NULL DEFAULT 1 CHECK (dry_run IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'planned',
    filters_json TEXT NOT NULL DEFAULT '{}',
    artifact_root TEXT NOT NULL DEFAULT '',
    replace_existing INTEGER NOT NULL DEFAULT 0 CHECK (replace_existing IN (0, 1)),
    candidates_scanned INTEGER NOT NULL DEFAULT 0,
    candidates_selected INTEGER NOT NULL DEFAULT 0,
    profiles_rebuilt INTEGER NOT NULL DEFAULT 0,
    profiles_skipped INTEGER NOT NULL DEFAULT 0,
    profiles_failed INTEGER NOT NULL DEFAULT 0,
    fingerprints_rebuilt INTEGER NOT NULL DEFAULT 0,
    dedup_groups_rebuilt INTEGER NOT NULL DEFAULT 0,
    key_moments_rebuilt INTEGER NOT NULL DEFAULT 0,
    selected_events_json TEXT NOT NULL DEFAULT '[]',
    errors_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_evidence_index_rebuild_runs_created ON evidence_index_rebuild_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_index_rebuild_runs_status ON evidence_index_rebuild_runs(status, dry_run);
