# Node1 AI Camera Assistant v0.4/v0.4.1 — SmolVLM2 short clip experiments

v0.4 adds an experimental SmolVLM2 short clip path after a local MonitorMe trigger.
v0.4.1 tightens this path after live RTX 5060 Ti testing by switching from freeform
caption-style JSON to vLLM `structured_outputs` with a constrained enum-style schema.

The feature is disabled by default and is designed for local-only experiments on Node1.

## What v0.4/v0.4.1 does

1. Motion gate emits a local `motion_detected` event.
2. Optional YOLO emits child `object_detected` rows.
3. MonitorMe writes normal keyframe evidence and assistant summaries.
4. If `smolvlm2_enabled=true`, MonitorMe writes a small local short clip bundle:
   - sampled JPEG frames from the trigger context window
   - a `short_clip_manifest` JSON artifact
5. SmolVLM2 receives only selected local frame artifact(s) through a local OpenAI-compatible endpoint.
6. v0.4.1 asks vLLM to constrain the answer with `structured_outputs={"json": schema}`.
7. The validated result is stored in `smolvlm2_clip_experiments`.

## v0.4.1 structured output shape

SmolVLM2 is treated as a bounded visual-state probe, not a narrator. The accepted
experiment JSON has only these fields:

```json
{
  "schema_version": "monitorme.smolvlm2_short_clip.v0.4.1",
  "event_id": "evt_...",
  "artifact_id": "art_...",
  "visible_scene": "indoor | outdoor | unclear",
  "person_like_presence": "visible | not_visible | unclear",
  "vehicle_like_presence": "visible | not_visible | unclear",
  "motion_claim": "single_frame_only_no_motion_claim",
  "safe_observation": "single frame reviewed | visible content unclear | observable scene present",
  "unsupported_claims": []
}
```

`event_id` and `artifact_id` are const-bound in the structured-output schema. The
model cannot invent alternate evidence IDs and still pass validation.

## Safety boundaries

- Disabled by default.
- Local loopback endpoint by default.
- Remote endpoint rejected unless explicitly allowed.
- Runs only after a local trigger and stored short clip artifact.
- Produces companion bounded visual-state observations only.
- Does not create YOLO detections.
- Does not override policy decisions.
- Rejects identity, face recognition, person profiles, age, gender, occupation,
  nationality, intent, threat, suspicious-behavior, weapon, gun, and knife claims.
- Requires the supplied trigger `event_id` and short-clip `artifact_id`.

## Environment

For the RTX 5060 Ti / 4096-token SmolVLM2 smoke test, keep frame count low:

```bash
export MONITORME_SMOLVLM2_PROVIDER=smolvlm2-openai
export MONITORME_ASSISTANT_USE_SMOLVLM2=1
export MONITORME_SMOLVLM2_BASE_URL="http://127.0.0.1:8003/v1"
export MONITORME_SMOLVLM2_MODEL_ID="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
export MONITORME_SMOLVLM2_API_KEY="EMPTY"
export MONITORME_SMOLVLM2_MAX_FRAMES=1
export MONITORME_SMOLVLM2_MAX_TOKENS=300
export MONITORME_SMOLVLM2_TEMPERATURE=0.0
```

The short clip bundle may contain more frames, but `MONITORME_SMOLVLM2_MAX_FRAMES`
controls how many are sent to SmolVLM2.

## CLI

```bash
python -m monitor_me.cli smolvlm2-health --allow-unconfigured

python -m monitor_me.cli --db data/events/monitorme.db capture-run \
  --detector-enabled \
  --smolvlm2-enabled \
  --smolvlm2-clip-frame-count 8 \
  --duration-sec 10

python -m monitor_me.cli --db data/events/monitorme.db smolvlm2-experiments --limit 20
```

Manual analysis for an existing event with a stored short clip manifest:

```bash
python -m monitor_me.cli \
  --db data/events/monitorme.db \
  smolvlm2-analyze-event \
  evt_5c5723a591a64ac9
```

## API

- `GET /assistant/smolvlm2/health`
- `POST /assistant/events/{event_id}/smolvlm2-experiment`
- `GET /assistant/smolvlm2-experiments`

## Validation

```bash
./scripts/validate_node1_ai_camera_assistant_v04.sh
./scripts/validate_node1_ai_camera_assistant_v041.sh
python -m pytest -q
```
