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
