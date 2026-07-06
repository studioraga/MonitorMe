PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS evidence_pipeline_profiles (
    profile_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    session_id TEXT,
    camera_id TEXT NOT NULL,
    manifest_artifact_id TEXT,
    profile_artifact_id TEXT,
    manifest_csv_path TEXT,
    profile_path TEXT,
    native_schema TEXT,
    capture_manifest_rows INTEGER NOT NULL DEFAULT 0,
    fingerprint_count INTEGER NOT NULL DEFAULT 0,
    media_fingerprint_count INTEGER NOT NULL DEFAULT 0,
    synthetic_fingerprint_count INTEGER NOT NULL DEFAULT 0,
    real_media_ingestion INTEGER NOT NULL DEFAULT 0 CHECK (real_media_ingestion IN (0, 1)),
    duplicate_group_count INTEGER NOT NULL DEFAULT 0,
    duplicate_clip_count INTEGER NOT NULL DEFAULT 0,
    unique_clip_count INTEGER NOT NULL DEFAULT 0,
    key_moment_count INTEGER NOT NULL DEFAULT 0,
    planned_read_bytes INTEGER NOT NULL DEFAULT 0,
    total_manifest_bytes INTEGER NOT NULL DEFAULT 0,
    safety_ok INTEGER NOT NULL DEFAULT 0 CHECK (safety_ok IN (0, 1)),
    violation_count INTEGER NOT NULL DEFAULT 0,
    facts_only INTEGER NOT NULL DEFAULT 1 CHECK (facts_only IN (0, 1)),
    timeline_json TEXT NOT NULL DEFAULT '{}',
    latency_json TEXT NOT NULL DEFAULT '{}',
    safety_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id) ON DELETE SET NULL,
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id),
    FOREIGN KEY(manifest_artifact_id) REFERENCES capture_artifacts(artifact_id) ON DELETE SET NULL,
    FOREIGN KEY(profile_artifact_id) REFERENCES capture_artifacts(artifact_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_profiles_session_created ON evidence_pipeline_profiles(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_profiles_camera_created ON evidence_pipeline_profiles(camera_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_profiles_safety ON evidence_pipeline_profiles(safety_ok, violation_count);

CREATE TABLE IF NOT EXISTS evidence_fingerprints (
    fingerprint_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    session_id TEXT,
    camera_id TEXT NOT NULL,
    clip_id TEXT NOT NULL,
    clip_index INTEGER NOT NULL DEFAULT -1,
    path TEXT,
    start_ms INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    from_media INTEGER NOT NULL DEFAULT 0 CHECK (from_media IN (0, 1)),
    fingerprint_source TEXT NOT NULL DEFAULT 'metadata_synthetic',
    decoded_width INTEGER NOT NULL DEFAULT 0,
    decoded_height INTEGER NOT NULL DEFAULT 0,
    ahash64 TEXT,
    dhash64 TEXT,
    fingerprint64 TEXT,
    fingerprint_hex TEXT,
    histogram_json TEXT NOT NULL DEFAULT '[]',
    histogram_bins INTEGER NOT NULL DEFAULT 0,
    duplicate_group INTEGER NOT NULL DEFAULT -1,
    duplicate_of INTEGER NOT NULL DEFAULT -1,
    nearest_hamming INTEGER NOT NULL DEFAULT -1,
    fingerprint_score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES evidence_pipeline_profiles(profile_id) ON DELETE CASCADE,
    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id) ON DELETE SET NULL,
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_fingerprints_profile_clip ON evidence_fingerprints(profile_id, clip_index);
CREATE INDEX IF NOT EXISTS idx_evidence_fingerprints_session_media ON evidence_fingerprints(session_id, from_media);
CREATE INDEX IF NOT EXISTS idx_evidence_fingerprints_hex ON evidence_fingerprints(fingerprint_hex);
CREATE INDEX IF NOT EXISTS idx_evidence_fingerprints_duplicate ON evidence_fingerprints(profile_id, duplicate_group);

CREATE TABLE IF NOT EXISTS evidence_dedup_groups (
    dedup_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    session_id TEXT,
    camera_id TEXT NOT NULL,
    group_id INTEGER NOT NULL,
    representative_clip_id TEXT,
    representative_clip_index INTEGER NOT NULL DEFAULT -1,
    group_size INTEGER NOT NULL DEFAULT 0,
    duplicate_count INTEGER NOT NULL DEFAULT 0,
    min_hamming INTEGER NOT NULL DEFAULT -1,
    max_hamming INTEGER NOT NULL DEFAULT -1,
    clip_ids_json TEXT NOT NULL DEFAULT '[]',
    clip_indices_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES evidence_pipeline_profiles(profile_id) ON DELETE CASCADE,
    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id) ON DELETE SET NULL,
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id),
    UNIQUE(profile_id, group_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_dedup_profile ON evidence_dedup_groups(profile_id, group_id);
CREATE INDEX IF NOT EXISTS idx_evidence_dedup_session ON evidence_dedup_groups(session_id, duplicate_count DESC);

CREATE TABLE IF NOT EXISTS evidence_key_moments (
    key_moment_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    session_id TEXT,
    camera_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    clip_id TEXT NOT NULL,
    clip_index INTEGER NOT NULL DEFAULT -1,
    start_ms INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    priority_score REAL NOT NULL DEFAULT 0.0,
    motion_score REAL NOT NULL DEFAULT 0.0,
    audio_score REAL NOT NULL DEFAULT 0.0,
    lighting_delta REAL NOT NULL DEFAULT 0.0,
    changed_pixels INTEGER NOT NULL DEFAULT 0,
    duplicate_group INTEGER NOT NULL DEFAULT -1,
    created_at TEXT NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES evidence_pipeline_profiles(profile_id) ON DELETE CASCADE,
    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE CASCADE,
    FOREIGN KEY(session_id) REFERENCES capture_sessions(session_id) ON DELETE SET NULL,
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id),
    UNIQUE(profile_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_evidence_key_moments_profile_rank ON evidence_key_moments(profile_id, rank);
CREATE INDEX IF NOT EXISTS idx_evidence_key_moments_session_start ON evidence_key_moments(session_id, start_ms);
