# Node1 AI Camera Assistant v0.3: Qwen VLM keyframe analysis after trigger

Node1 Assistant v0.3 adds optional Qwen VLM analysis for stored keyframes after a
local motion/YOLO trigger. The feature is disabled by default and follows the
same MonitorMe evidence-first rule used in v0.1 and v0.2.

## Scope

- Run Qwen VLM only after a trigger event has created a local keyframe artifact.
- Send only the stored keyframe to a configured local OpenAI-compatible VLM
  endpoint.
- Store VLM output as companion visual context in `vlm_keyframe_analyses`.
- Require strict JSON with cited event/artifact IDs.
- Reject identity, intent, threat, weapon, face-recognition, and suspicious-behavior claims.
- Keep YOLO detections and deterministic policy as the source of truth.
- Keep v0.1/v0.2 deterministic fallback behavior unchanged.

## Data model

Migration `003_node1_assistant_v03.sql` adds:

```text
vlm_keyframe_analyses
  analysis_id
  event_id / parent_event_id / session_id / camera_id / frame_id
  artifact_id / artifact_path
  model_id
  status: completed | failed | skipped
  analysis_json
  source_refs_json
  error
  created_at
```

The table stores VLM output separately from YOLO `object_detected` rows so the
assistant never confuses low-trust VLM observations with detector facts.

## Environment

Qwen VLM is disabled unless explicitly enabled:

```bash
export MONITORME_VLM_PROVIDER=qwen-vlm
export MONITORME_ASSISTANT_USE_QWEN_VLM=1
export MONITORME_VLM_BASE_URL="http://127.0.0.1:8002/v1"
export MONITORME_VLM_MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
export MONITORME_VLM_API_KEY="EMPTY"
```

Remote VLM endpoints are rejected by default. To use a non-loopback endpoint,
`MONITORME_VLM_ALLOW_REMOTE=1` must be set explicitly. For private CCTV evidence,
keep the endpoint local.

## CLI

```bash
python -m monitor_me.cli --db data/events/monitorme.db vlm-health --allow-unconfigured
python -m monitor_me.cli --db data/events/monitorme.db capture-run --detector-enabled --vlm-enabled
python -m monitor_me.cli --db data/events/monitorme.db vlm-analyze-event <event_id>
python -m monitor_me.cli --db data/events/monitorme.db vlm-analyses --session-id <session_id>
```

## API

```text
GET  /assistant/vlm/health
POST /assistant/events/{event_id}/vlm-analysis
GET  /assistant/vlm-analyses
```

## Validation

```bash
./scripts/validate_node1_ai_camera_assistant_v03.sh
python -m pytest -q
```

The validation uses fake frames, detector output, and a fake Qwen VLM client. It
does not require a camera, Qwen server, MAX/Gemma, or external network access.
