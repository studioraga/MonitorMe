PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS evidence_retention_schedule (
    schedule_id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0 CHECK (enabled IN (0, 1)),
    cadence TEXT NOT NULL DEFAULT 'daily',
    older_than_days INTEGER,
    keep_last_per_camera INTEGER NOT NULL DEFAULT 1,
    keep_last_per_session INTEGER NOT NULL DEFAULT 1,
    profile_id TEXT,
    session_id TEXT,
    camera_id TEXT,
    limit_profiles INTEGER NOT NULL DEFAULT 1000,
    dry_run INTEGER NOT NULL DEFAULT 1 CHECK (dry_run IN (0, 1)),
    compact INTEGER NOT NULL DEFAULT 1 CHECK (compact IN (0, 1)),
    vacuum INTEGER NOT NULL DEFAULT 0 CHECK (vacuum IN (0, 1)),
    next_run_after TEXT,
    last_checked_at TEXT,
    last_run_at TEXT,
    last_retention_run_id TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_retention_scheduler_runs (
    scheduler_run_id TEXT PRIMARY KEY,
    schedule_id TEXT NOT NULL,
    forced INTEGER NOT NULL DEFAULT 0 CHECK (forced IN (0, 1)),
    due INTEGER NOT NULL DEFAULT 0 CHECK (due IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'checked',
    reason TEXT NOT NULL DEFAULT '',
    retention_run_id TEXT,
    policy_json TEXT NOT NULL DEFAULT '{}',
    dry_run INTEGER NOT NULL DEFAULT 1 CHECK (dry_run IN (0, 1)),
    compact INTEGER NOT NULL DEFAULT 1 CHECK (compact IN (0, 1)),
    vacuum INTEGER NOT NULL DEFAULT 0 CHECK (vacuum IN (0, 1)),
    checked_at TEXT NOT NULL,
    next_run_after TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    completed_at TEXT,
    FOREIGN KEY(schedule_id) REFERENCES evidence_retention_schedule(schedule_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evidence_retention_schedule_enabled ON evidence_retention_schedule(enabled, next_run_after);
CREATE INDEX IF NOT EXISTS idx_evidence_retention_scheduler_runs_checked ON evidence_retention_scheduler_runs(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_retention_scheduler_runs_schedule ON evidence_retention_scheduler_runs(schedule_id, checked_at DESC);

INSERT OR IGNORE INTO evidence_retention_schedule(
    schedule_id, enabled, cadence, older_than_days, keep_last_per_camera,
    keep_last_per_session, limit_profiles, dry_run, compact, vacuum,
    next_run_after, notes, created_at, updated_at
)
VALUES(
    'default', 0, 'daily', 30, 1,
    1, 1000, 1, 1, 0,
    NULL, 'disabled by default; scheduled retention defaults to dry-run',
    datetime('now'), datetime('now')
);
