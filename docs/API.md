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
