# MonitorMe API

Start locally:

```bash
MONITORME_DB=data/events/monitorme.db ./scripts/run_api.sh
```

Root:

```text
GET /
```

Health:

```text
GET /health
```

Camera discovery:

```text
GET /camera/devices
GET /camera/devices?probe=true
```

Run a bounded real C922 capture:

```text
POST /camera/capture/run
```

Body:

```json
{
  "camera_id": "c922_node1_gate",
  "device": "/dev/video0",
  "width": 1280,
  "height": 720,
  "fps": 30,
  "fourcc": "MJPG",
  "duration_sec": 10,
  "motion_threshold": 1.5
}
```

List evidence events:

```text
GET /events
GET /events?event_type=motion_detected&label=motion&limit=20
```

Ask the grounded assistant:

```text
POST /assistant/ask
```

Body:

```json
{"question":"What motion events happened today?"}
```

Build evidence pack:

```text
POST /assistant/events/{event_id}/evidence-pack
```

Build incident report:

```text
POST /assistant/reports/incident
```

Body:

```json
{"event_ids":["evt_..."],"title":"Gate motion event","severity":"info"}
```

Operator feedback:

```text
POST /events/{event_id}/feedback
```

Body:

```json
{"label":"false_positive","reason":"operator review","operator":"operator"}
```

## Node1 AI Camera Assistant v0.1 routes

The v0.1 assistant milestone adds DB-backed summary and event-contract endpoints. These routes operate only on local MonitorMe SQLite evidence.

```text
POST /assistant/events/{event_id}/summary
GET  /assistant/summaries
GET  /assistant/event-contracts
POST /assistant/ask
POST /assistant/reports/incident
```

Create or refresh a deterministic summary for one event:

```bash
curl -sS -X POST http://127.0.0.1:8088/assistant/events/<event_id>/summary \
  | python3 -m json.tool
```

List summaries:

```bash
curl -sS 'http://127.0.0.1:8088/assistant/summaries?limit=20' \
  | python3 -m json.tool
```

List event contracts:

```bash
curl -sS 'http://127.0.0.1:8088/assistant/event-contracts?limit=20' \
  | python3 -m json.tool
```

Ask over the local evidence DB:

```bash
curl -sS -X POST http://127.0.0.1:8088/assistant/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What person events happened today?"}' \
  | python3 -m json.tool
```

Safety behavior:

```text
The assistant may cite event_id, session_id, frame_id, label, confidence, bbox, model_id, artifact paths, policy decision, and audit IDs.
The assistant must not invent identity, intent, face recognition, weapon, danger, suspicious behavior, or object labels absent from the event DB.
```

---

## Node1 AI Camera Assistant v0.2 — Gemma/MAX strict JSON summaries

v0.2 adds an optional local Gemma 3 1B explanation layer served by MAX through an OpenAI-compatible endpoint.

```text
YOLO = fast visual facts
Node1 policy = deterministic decisions/actions
Gemma/MAX = explanation/Q&A over structured event facts only
```

Validation command:

```bash
./scripts/validate_node1_ai_camera_assistant_v02.sh
```

Gemma/MAX is disabled by default. If it is unavailable or returns invalid/unsupported JSON, MonitorMe stores the deterministic summary from v0.1 and records the fallback reason in summary facts and audit logs.

See `docs/NODE1_AI_CAMERA_ASSISTANT_V0_2.md` for the full prompt/output contract and Node1 setup.


## Node1 MAX/Gemma live validation helpers

MonitorMe includes helper scripts for the previously validated Node1 MAX + Gemma 3 1B path:

```bash
./scripts/max/term1_start_max_gemma3_1b.sh
./scripts/max/term2_validate_max_gemma3_1b.sh
./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
```

These scripts use the external pixi MAX quickstart project, preserve `--sample-on-host`, and validate that MonitorMe can accept a strict JSON Gemma summary over local event contracts. See `docs/MAX_GEMMA_NODE1.md`.

## Evidence pipeline summary API

Phase 13 exposes the persisted Phase 12 evidence index through local-only FastAPI summary endpoints. These routes read the SQLite evidence index tables and return facts-only metadata; they do not decode media, upload frames, or emit semantic audio/visual claims.

```text
GET /evidence/pipeline/summaries
GET /evidence/pipeline/sessions/{session_id}/summary
GET /evidence/pipeline/profiles/{profile_id}/summary
GET /evidence/pipeline/profiles/{profile_id}/fingerprints
GET /evidence/pipeline/profiles/{profile_id}/dedup-groups
GET /evidence/pipeline/profiles/{profile_id}/key-moments
```

List compact summaries:

```bash
curl -sS 'http://127.0.0.1:8088/evidence/pipeline/summaries?session_id=<session_id>&limit=10' \
  | python3 -m json.tool
```

Fetch the latest detailed summary for one capture session:

```bash
curl -sS 'http://127.0.0.1:8088/evidence/pipeline/sessions/<session_id>/summary?latest_only=true' \
  | python3 -m json.tool
```

Fetch one profile summary with key moments, dedup groups, and an optional bounded fingerprint sample:

```bash
curl -sS 'http://127.0.0.1:8088/evidence/pipeline/profiles/<profile_id>/summary?include_fingerprints=true&fingerprint_limit=20' \
  | python3 -m json.tool
```

Fetch detail rows:

```bash
curl -sS 'http://127.0.0.1:8088/evidence/pipeline/profiles/<profile_id>/fingerprints?from_media=true&limit=20' | python3 -m json.tool
curl -sS 'http://127.0.0.1:8088/evidence/pipeline/profiles/<profile_id>/dedup-groups' | python3 -m json.tool
curl -sS 'http://127.0.0.1:8088/evidence/pipeline/profiles/<profile_id>/key-moments' | python3 -m json.tool
```

Safety contract:

```text
The evidence summary API returns persisted profile, fingerprint, duplicate-group, key-moment, timeline, latency, and safety-validator facts only.
It does not decode media in the API request path.
It does not upload raw frames.
It does not infer person identity, speech content, behavior, intent, weapons, danger, or suspiciousness.
```
