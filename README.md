# MonitorMe

**MonitorMe** is a standalone Node1-local CCTV evidence assistant project.

It is separate from `cctv-ip-or-lan-usb-camera-ingest-ai-inference`. The old CCTV ingest repository was used only as design input. MonitorMe now targets your current hardware layout where the **Logitech C922 is attached directly to Node1 as `/dev/video0`**.

## Current implementation

This repo implements:

```text
Step 17A: evidence-first assistant foundation
Step 17B: real Node1 C922 local capture pipeline
Step 17C: optional real YOLO ONNX detection after motion gate
Step 17D: detector health and object query grounding
Step 17E: evidence overlays with raw-keyframe preservation
Node1 AI Camera Assistant v0.1: event contracts, deterministic policy, automatic summaries
Node1 AI Camera Assistant v0.2: optional Gemma/MAX strict JSON summaries with deterministic fallback
```

From this point onward, the runtime path does **not** rely on seeded demo CCTV events. It captures from the real local camera, writes local artifacts, inserts normalized SQLite evidence rows, and answers only from those records.

## Goal

MonitorMe answers CCTV questions only when every answer can be backed by local evidence:

- `event_id`
- `session_id`
- `frame_id`
- normalized event rows
- local artifact paths
- model metadata when a model was actually used
- policy decision records
- audit records
- evidence-pack paths
- incident report paths

The assistant is not allowed to invent facts. It does not upload private CCTV frames or event data to external services.

## Step 17C architecture

```text
Logitech C922 on Node1
        |
        v
/dev/video0  (V4L2 / MJPG)
        |
        v
OpenCV local capture runner
        |
        v
Frame-difference motion gate
        |
        +--> real keyframe artifact: data/captures/<session_id>/keyframes/*.jpg
        +--> real capture manifest: data/captures/<session_id>/manifest.json
        +--> normalized SQLite row: event_type=motion_detected, label=motion
        |
        +--> optional real YOLO ONNX detector, enabled only when configured
              |
              +--> normalized child rows: event_type=object_detected
                   label=person or vehicle/etc.
                   parent_event_id=<motion_event_id>
                   model_id=yolo11n-coco-onnx
                   artifact_id=<keyframe_artifact_id>
        |
        v
MonitorMe evidence DB
        |
        +--> capture_sessions
        +--> capture_artifacts
        +--> events
        +--> model_registry
        +--> audit_log
        +--> assistant_runs / assistant_summaries
        +--> evidence_packs / incident_reports / feedback
        |
        v
MonitorMe assistant + FactGuard
        |
        v
evidence-backed answer with event/session/frame/artifact/policy/audit refs
```

Important: Step 17C still does **not** fabricate `person`, `vehicle`, weapon, identity, or intent labels. Object labels appear only when a real enabled YOLO ONNX detector returns them. If the detector is missing or fails, MonitorMe stores the parent motion event and an audit warning, but no object rows.

## Quick start

```bash
cd MonitorMe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[api,camera,test]'

./scripts/validate_step17c_monitorme.sh
```

Expected final line:

```text
=== MonitorMe Step 17C validation PASSED ===
```

That validation does not seed demo CCTV data and does not require a physical camera or ONNX model. It unit-tests the real capture path and proves the normalized object-detection child-row path with an injected fake detector.


## Node1 MAX/Gemma helper scripts

MonitorMe now carries the known-good Node1 MAX + Gemma 3 1B startup and validation scripts under `scripts/max/`. These leverage the existing pixi MAX project at `~/dev/modular/project/quickstart` and preserve the RTX 5060 Ti workaround `--sample-on-host`.

```text
TERM1
  ./scripts/max/term1_start_max_gemma3_1b.sh
      -> starts MAX serving google/gemma-3-1b-it on 127.0.0.1:8000
      -> keeps metrics on 127.0.0.1:8001

TERM2
  ./scripts/max/term2_validate_max_gemma3_1b.sh
      -> validates /v1/models, /metrics, Python client, curl client, repeatability, nvidia-smi

TERM2 after standalone MAX passes
  ./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
      -> validates MonitorMe llm-health --probe
      -> creates a local validation event
      -> requires strict Gemma JSON summary_source=gemma_max
```

See `docs/MAX_GEMMA_NODE1.md` for full setup and troubleshooting.

## Node1 AI Camera Assistant v0.2 — Gemma/MAX explanation layer

Gemma/MAX is optional and disabled by default. When enabled, MonitorMe sends only local structured facts to Gemma:

```text
YOLO visual facts + event contract + deterministic policy + artifact metadata
        |
        v
Gemma 3 1B via MAX OpenAI-compatible /v1/chat/completions
        |
        v
strict JSON summary validator
        |
        +--> valid: store Gemma operator summary
        +--> invalid/unavailable: store deterministic fallback summary
```

Gemma does not receive raw CCTV frames and does not decide actions. It explains facts and policy that already exist in SQLite.

Enable Gemma/MAX after starting the local MAX server:

```bash
export MONITORME_LLM_PROVIDER=max-openai
export MONITORME_ASSISTANT_USE_GEMMA=1
export MONITORME_LLM_BASE_URL=http://127.0.0.1:8000/v1
export MONITORME_LLM_MODEL_ID=google/gemma-3-1b-it
export MONITORME_LLM_API_KEY=EMPTY
export MONITORME_LLM_TEMPERATURE=0.0
```

Check configuration:

```bash
python -m monitor_me.cli --db data/events/monitorme.db llm-health --allow-unconfigured
```

Validate v0.2 without requiring MAX to be running:

```bash
./scripts/validate_node1_ai_camera_assistant_v02.sh
```

## Real Node1 C922 live validation

Your C922 has already shown `/dev/video0` with MJPG/YUYV formats. Use `/dev/video0` first.

```bash
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
source .venv/bin/activate
python -m pip install -e '.[api,camera,test]'

./scripts/validate_node1_c922_live.sh
```

Move in front of the camera during the capture window. If no motion events are emitted, lower the threshold:

```bash
MONITORME_MOTION_THRESHOLD=0.5 ./scripts/validate_node1_c922_live.sh
```

## CLI usage

Initialize DB and default model metadata:

```bash
python -m monitor_me.cli --db data/events/monitorme.db init-db
```

List local camera devices:

```bash
python -m monitor_me.cli camera-devices --probe
```

Run a real C922 capture from `/dev/video0`:

```bash
python -m monitor_me.cli --db data/events/monitorme.db capture-run \
  --camera-id c922_node1_gate \
  --device /dev/video0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --fourcc MJPG \
  --duration-sec 10 \
  --motion-threshold 1.5
```

Download the YOLO ONNX model, then run real C922 capture with YOLO enabled after the motion gate:

```bash
./scripts/models/download_yolo_onnx.sh
python -m pip install -e '.[api,camera,detector,test]'

python -m monitor_me.cli --db data/events/monitorme.db capture-run \
  --camera-id c922_node1_gate \
  --device /dev/video0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --fourcc MJPG \
  --duration-sec 10 \
  --motion-threshold 1.5 \
  --detector-enabled \
  --detector-model-path models/object_detection/yolo11n.onnx \
  --detector-conf-threshold 0.35
```

List normalized motion events:

```bash
python -m monitor_me.cli --db data/events/monitorme.db events \
  --event-type motion_detected \
  --limit 20
```


List normalized object detections:

```bash
python -m monitor_me.cli --db data/events/monitorme.db events \
  --event-type object_detected \
  --limit 20
```

Ask a grounded assistant question:

```bash
python -m monitor_me.cli --db data/events/monitorme.db ask "What motion events happened today?"
```

Build an evidence pack for a real event:

```bash
python -m monitor_me.cli --db data/events/monitorme.db evidence-pack <event_id>
```

Generate an incident report:

```bash
python -m monitor_me.cli --db data/events/monitorme.db incident-report \
  --event-id <event_id> \
  --title "Gate motion event" \
  --severity info
```

Mark feedback:

```bash
python -m monitor_me.cli --db data/events/monitorme.db feedback <event_id> \
  --label false_positive \
  --reason "operator review"
```


## Node1 non-LLM GPU inference lab v0.1

MonitorMe now includes an optional native C++/CUDA sidecar under `native/node1_non_llm_gpu_inference_lab/` for deterministic CPU/GPU workload profiling without an LLM in the inference core. It computes frame-difference tile masks, classifies frames into `sparse`, `mixed`, or `dense` routes, and can also analyze batched float32 audio energy windows.

The sidecar is disabled by default. When enabled during `capture-run`, it runs only after a local motion/keyframe trigger and stores `gpu_workload_profiled` child events with routing metadata such as `tile_mask_hex`, `active_tiles`, `changed_ratio`, and CUDA availability. It does not emit person, identity, intent, or behavior claims.

Build on Node1 with CUDA 13.3:

```bash
cd native/node1_non_llm_gpu_inference_lab
./scripts/build_node1_gpu_lab.sh
```

Enable during capture:

```bash
python -m monitor_me.cli --db data/events/monitorme.db capture-run \
  --camera-id c922_node1_gate \
  --device /dev/video0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --fourcc MJPG \
  --duration-sec 10 \
  --motion-threshold 1.5 \
  --gpu-lab-enabled \
  --gpu-lab-binary native/node1_non_llm_gpu_inference_lab/build/node1_non_llm_gpu_lab
```

See `docs/NODE1_NON_LLM_GPU_INFERENCE_LAB.md`.

## Optional API server

Foreground mode:

```bash
MONITORME_DB=data/events/monitorme.db ./scripts/run_api.sh
```

Background mode:

```bash
./scripts/start_api_background.sh
./scripts/status_api.sh
```

Stop background server:

```bash
./scripts/stop_api.sh
```

Default local endpoint:

```text
http://127.0.0.1:8088
```

Core routes:

```text
GET  /
GET  /health
GET  /camera/devices
POST /camera/capture/run
GET  /events
GET  /models
POST /models/register-defaults
POST /assistant/ask
POST /assistant/events/{event_id}/evidence-pack
POST /assistant/reports/incident
POST /events/{event_id}/feedback
GET  /trackers/false-positives
```

Example real capture API request:

```bash
curl -sS -X POST http://127.0.0.1:8088/camera/capture/run \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id":"c922_node1_gate",
    "device":"/dev/video0",
    "width":1280,
    "height":720,
    "fps":30,
    "fourcc":"MJPG",
    "duration_sec":10,
    "motion_threshold":1.5
  }' | python3 -m json.tool
```


Example real capture API request with YOLO ONNX enabled:

```bash
curl -sS -X POST http://127.0.0.1:8088/camera/capture/run \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id":"c922_node1_gate",
    "device":"/dev/video0",
    "width":1280,
    "height":720,
    "fps":30,
    "fourcc":"MJPG",
    "duration_sec":10,
    "motion_threshold":1.5,
    "detector_enabled":true,
    "detector_model_path":"models/object_detection/yolo11n.onnx",
    "detector_conf_threshold":0.35
  }' | python3 -m json.tool
```

## Privacy rules

MonitorMe follows these default rules:

1. Local evidence first.
2. No external upload of private frames/events.
3. No face recognition.
4. No identity claim.
5. No weapon, intent, or behavior claim unless normalized evidence explicitly exists.
6. No fabricated object labels.
7. Every answer must expose event/session/frame references or clearly say that local evidence is missing.

## Repository layout

```text
MonitorMe/
  monitor_me/
    assistant.py               DB-grounded assistant orchestration
    db.py                      SQLite migrations and data access
    event_tools.py             CCTV event query/evidence helpers
    evidence_pack.py           event/session/model/policy/audit evidence packs
    fact_guard.py              hallucination and unsupported-claim guard
    local_capture.py           real Node1 /dev/video0 capture + motion/YOLO evidence
    yolo_onnx.py               optional local YOLO ONNX detector/postprocess
    llm_client.py              Null/Fake LLM clients; future Gemma/MAX hook
    model_registry.py          detector/text/future-model metadata
    report_tools.py            incident report generation
    routes.py                  optional FastAPI app factory
    tracker_tools.py           false-positive/useful feedback tracking
  migrations/
  scripts/
    validate_step17c_monitorme.sh
    validate_step17b_monitorme.sh
    validate_node1_c922_live.sh
    validate_node1_c922_yolo_live.sh
    run_api.sh
    start_api_background.sh
    status_api.sh
    stop_api.sh
  tests/
  docs/
  data/
    captures/
    events/
    evidence_packs/
    reports/
```

## Suggested commit message

```text
feat: add YOLO ONNX object detection after Node1 motion gate

Add Step 17C optional YOLO ONNX detection after the real Node1 motion gate.
When enabled with a real ONNX model, MonitorMe runs local detection on motion
keyframes only and inserts normalized object_detected child rows with model_id,
confidence, bbox, parent_event_id, keyframe artifact_id, and audit trail. Keep
motion capture resilient when the detector is disabled, missing, or fails, and
add CLI/API detector controls, docs, live validation script, and tests proving no
object labels are fabricated.
```


## Step 17D detector health

MonitorMe v0.1.9 adds `python -m monitor_me.cli detector-health` and `GET /models/detector/health` so Node1 can validate the YOLO ONNX model path, checksum, ONNX Runtime providers, and model input/output metadata before live camera capture. See `docs/STEP17D_DETECTOR_HEALTH.md`.


## Step 17E: evidence visualization overlays

MonitorMe now writes annotated keyframes as separate derived evidence artifacts when YOLO detections are emitted after the motion gate. The original raw keyframe remains unchanged and remains the artifact linked by the `object_detected` rows. The overlay is registered as `artifact_type=annotated_keyframe` and includes rendered labels for `label`, `confidence`, `event_id`, `parent_event_id`, and `model_id`.

Run offline validation:

```bash
./scripts/validate_step17e_monitorme.sh
```

Run live C922 + YOLO validation:

```bash
./scripts/models/download_yolo_onnx.sh
python -m pip install -e '.[api,camera,detector,test]'
./scripts/validate_node1_c922_yolo_live.sh
```

List overlay artifacts:

```bash
python -m monitor_me.cli --db data/events/monitorme.db artifacts \
  --artifact-type annotated_keyframe \
  --limit 20
```


## Node1 AI Camera Assistant v0.1

MonitorMe now has the first formal **Node1 AI Camera Assistant** layer on top of the local C922 + YOLO evidence pipeline. This milestone keeps the model split intentionally strict:

```text
YOLO11n ONNX              = fast visual facts: labels, confidence, bbox, frame_id
Node1 deterministic policy = actions: record only, review capture, bounded severity
Assistant summary service  = safe operator text from stored SQLite evidence
Gemma/MAX                  = next milestone, explanation only, not raw vision
VLM/SAM/GroundingDINO/CLIP = later evidence-enrichment layers after trigger
```

Runtime flow:

```text
C922 /dev/video0
  -> motion gate
  -> YOLO ONNX on motion keyframe
  -> motion_detected parent event
  -> object_detected child events
  -> event_contracts row
  -> deterministic policy_decision_json
  -> assistant_summaries row
  -> /assistant/ask over event DB
  -> incident report / evidence pack
```

New implementation files:

```text
monitor_me/yolo_client.py          # visual-facts client boundary
monitor_me/event_contract.py       # strict Node1 event contract builder
monitor_me/capture_policy.py       # deterministic Node1 action/severity policy
monitor_me/assistant_summary.py    # automatic DB-grounded summaries
migrations/002_node1_assistant_v01.sql
tests/test_node1_ai_camera_assistant_v01.py
docs/NODE1_AI_CAMERA_ASSISTANT_V0_1.md
```

New validation:

```bash
./scripts/validate_node1_ai_camera_assistant_v01.sh
```

Useful post-validation commands:

```bash
python -m monitor_me.cli --db data/events/monitorme.db summaries --limit 20
python -m monitor_me.cli --db data/events/monitorme.db event-contracts --limit 20
python -m monitor_me.cli --db data/events/monitorme.db ask "What person events happened today?"
```

Use `sqlite3`, not `cat`, to inspect the SQLite DB:

```bash
sqlite3 data/events/monitorme.db ".tables"
sqlite3 data/events/monitorme.db "select event_type,label,count(*) from events group by event_type,label;"
```

See `docs/NODE1_AI_CAMERA_ASSISTANT_V0_1.md` and `docs/NODE1_AI_CAMERA_ASSISTANT_VALIDATION.md`.


## v0.2.2 MAX startup-script safety

MonitorMe v0.2.2 changes the MAX/Gemma helper scripts so they do **not** mutate the external pixi MAX quickstart environment by default. If `scripts/max/term1_start_max_gemma3_1b.sh` reports `unable to locate module 'std'`, run:

```bash
./scripts/max/diagnose_max_gemma3_1b_env.sh
```

That failure is a broken/mismatched MAX/Mojo pixi environment, usually after a rolling nightly package upgrade, not a MonitorMe assistant-code failure. Restore a known-good `pixi.lock` with `pixi install --locked` or intentionally rebuild the quickstart environment. Only set `MONITORME_MAX_PIXI_SYNC=1` when you explicitly want the helper to run `pixi add`.

### MAX/Gemma environment recovery note

If the external MAX pixi quickstart fails with Mojo `unable to locate module 'std'`,
use the non-destructive Python 3.12 recovery helper:

```bash
./scripts/max/create_max_gemma3_1b_py312_env.sh
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312"
./scripts/max/term1_start_max_gemma3_1b.sh
```

See `docs/MAX_GEMMA_NODE1.md` for details.


## v0.2.4 MAX Mojo import-path hardening

If a clean Python 3.12 MAX pixi workspace still fails at `max serve` with `unable to locate module 'std'` or `unable to locate module 'nn'`, use the v0.2.4 helpers. The TERM1 and diagnostic scripts now discover Mojo package roots under `.pixi/envs/default`, export `MODULAR_MOJO_MAX_IMPORT_PATH` and `MOJO_PACKAGE_PATH`, clear stale `__mojocache__` directories by default, and preflight both `max._core_mojo` and `max._kv_cache_ops` before starting the server.

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312"
./scripts/max/diagnose_max_gemma3_1b_env.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```

Compiled Mojo caches are preserved by default. Set `MONITORME_MAX_CLEAR_MOJO_CACHE=1` only if you intentionally want to force recompilation while debugging stale cache artifacts.

## v0.2.5 stable MAX workspace fallback

If the Python 3.12 nightly workspace still fails before serving with:

```text
max._kv_cache_ops import failed
unable to locate module 'std'
unable to locate module 'nn'
```

then the current `max-nightly` solve is internally mismatched for serving on this
Node1 setup. Do not keep retrying `pixi add` against the same nightly workspace.
Create a separate stable-channel MAX workspace instead:

```bash
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
source .venv/bin/activate

./scripts/max/create_max_gemma3_1b_stable_py312_env.sh

export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable"
./scripts/max/term1_start_max_gemma3_1b.sh
```

Then validate from TERM2:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable"
./scripts/max/term2_validate_max_gemma3_1b.sh
./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
```

MonitorMe v0.2 remains safe if MAX is unavailable: Gemma/MAX summaries are optional
and deterministic summaries remain the fallback path.

### MAX/Gemma recovery note for v0.2.6

If both the nightly and stable MAX pixi workspaces fail during `max._kv_cache_ops` compilation with missing Mojo `std` or `nn`, use the explicit-Mojo stable workspace helper:

```bash
./scripts/max/create_max_gemma3_1b_stable_py312_with_mojo_env.sh
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable_mojo"
./scripts/max/probe_mojo_std_nn_roots.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```

The assistant remains usable without MAX through deterministic summaries and fallback validation.


## v0.2.7 MAX cache-preserving TERM1 fix

Node1 full logs showed a cache-sensitive MAX/Mojo behavior:

```text
probe_mojo_std_nn_roots.sh -> max._kv_cache_ops: PASS
term1_start_max_gemma3_1b.sh -> Clear Mojo cache: 1 -> max._kv_cache_ops compile FAIL
```

So TERM1 now preserves compiled Mojo caches by default:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable_mojo"
./scripts/max/probe_mojo_std_nn_roots.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```

Only clear caches when deliberately debugging stale compiled artifacts:

```bash
MONITORME_MAX_CLEAR_MOJO_CACHE=1 ./scripts/max/term1_start_max_gemma3_1b.sh
```


## v0.2.8 MAX import-root cleanup

Node1 logs showed that `probe_mojo_std_nn_roots.sh` could import `max._kv_cache_ops`, but TERM1 later failed while `pixi run max --help` eagerly compiled another serving extension, `max._distributed_ops`, and could not resolve Mojo packages such as `std`, `comm`, or `layout`.

The MAX helpers now build a cleaner Mojo import path by preferring `$PREFIX/lib/mojo` when it contains `std.mojoc` / `nn.mojoc`, excluding false-positive directories such as `share/locale/nn` and `share/tabset/std`, and overriding stale inherited `MODULAR_MOJO_MAX_IMPORT_PATH` / `MOJO_PACKAGE_PATH` by default. TERM1 also skips `pixi run max --help` by default because that command can trigger optional serving-extension compilation before the actual server starts.

Useful retry sequence:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable_mojo"
export MONITORME_MAX_CLEAR_MOJO_CACHE=0
export MONITORME_MAX_OVERRIDE_MOJO_IMPORT_PATH=1
export MONITORME_MAX_SKIP_HELP_PREFLIGHT=1
./scripts/max/probe_mojo_std_nn_roots.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```

Set `MONITORME_MAX_SKIP_HELP_PREFLIGHT=0` only when intentionally debugging MAX CLI startup itself.


## v0.2.10 MAX pixi env wrapper fix

Node1 v0.2.8 logs showed the clean Mojo import roots were discovered and `max._kv_cache_ops` passed, but `max serve` still failed while compiling `max._distributed_ops` with missing `std`, `comm`, and `layout`. The helper now invokes MAX/Mojo-sensitive commands through `pixi run env MODULAR_MOJO_MAX_IMPORT_PATH=... MOJO_PACKAGE_PATH=...` so the clean Mojo package roots are applied after pixi activation/dependency scripts, not only exported in the parent shell.

Use the same stable explicit-Mojo workspace and retry TERM1 with:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable_mojo"
export MONITORME_MAX_CLEAR_MOJO_CACHE=0
export MONITORME_MAX_OVERRIDE_MOJO_IMPORT_PATH=1
export MONITORME_MAX_SKIP_HELP_PREFLIGHT=1
./scripts/max/probe_mojo_std_nn_roots.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```


### v0.2.10 MAX startup correction

Node1 validation showed that manual Mojo compilation with explicit `-I` paths can prebuild MAX serving extensions, while `max serve` failed only when Python `mojo.importer` had to compile an uncached extension. The TERM1 helper now prebuilds known MAX Mojo caches by default and no longer passes `--sample-on-host` by default because MAX 26.4.0 reports `No such option --sample-on-host`. Use `MONITORME_MAX_SAMPLE_ON_HOST=1` only with MAX builds that support that flag.

### MAX/Gemma v0.2.11 activation-env helper

If MAX starts but the worker crashes with `failed to resolve built-in kernel
package paths` or `MAXG_addKernelPackage: failed to import kernels from ''`, patch
the external MAX Pixi workspace once:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable_mojo"
./scripts/max/patch_max_pixi_activation_env.sh
```

This writes `MODULAR_MOJO_MAX_IMPORT_PATH` and `MOJO_PACKAGE_PATH` to the MAX
workspace `[activation.env]` so MAX child worker processes inherit the Mojo
package roots from Pixi activation.


### v0.2.12 MAX runtime Mojo kernel-root narrowing

Node1 logs showed that the MAX server can progress past Mojo cache prebuild, Hugging Face access, Gemma pipeline configuration, and worker startup, then fail during graph custom-op registration with `MAXG_addKernelPackage: failed to import kernels from ''`. v0.2.12 keeps `MODULAR_MOJO_MAX_IMPORT_PATH`/`MOJO_PACKAGE_PATH` focused on the real compiled Mojo package root (`.pixi/envs/default/lib/mojo`) for the runtime process while cache prebuilds still compile with explicit `-I lib/mojo -I site-packages/max`. This avoids exporting a broad `lib/mojo:site-packages/max` runtime path that can confuse built-in kernel package resolution.


### v0.2.13 pytest helper fix

Fixes a test-only regression in `tests/test_max_helper_health.py` where newly added
MAX helper regression tests referenced `ROOT` without defining it. Runtime MAX/Gemma
startup behavior is unchanged from v0.2.12.


## Node1 AI Camera Assistant v0.3

Adds optional Qwen VLM keyframe analysis after a local motion/YOLO trigger. Qwen VLM is disabled by default, must use a local OpenAI-compatible endpoint unless explicitly allowed, stores strict JSON in `vlm_keyframe_analyses`, and cannot override YOLO facts or deterministic policy. See `docs/NODE1_AI_CAMERA_ASSISTANT_V0_3.md`.


## Node1 AI Camera Assistant v0.4 — SmolVLM2 short clip experiments

v0.4 adds optional local-only SmolVLM2 short clip experiments after a motion/YOLO trigger. MonitorMe writes a short local clip bundle from sampled trigger-context frames, stores it as evidence, and asks SmolVLM2 for strictly validated companion temporal observations. It is disabled by default and cannot create detections, override policy, or make identity/intent/threat claims.

Validation:

```bash
./scripts/validate_node1_ai_camera_assistant_v01.sh
./scripts/validate_node1_ai_camera_assistant_v02.sh
./scripts/validate_node1_ai_camera_assistant_v03.sh
./scripts/validate_node1_ai_camera_assistant_v04.sh
python -m pytest -q
```

## Node1 AI Camera Assistant v0.4.1 — constrained SmolVLM2 short clip experiments

SmolVLM2 short clip experiments are optional and disabled by default. v0.4.1
switches the SmolVLM2 path from freeform captions to vLLM `structured_outputs`
with a constrained JSON schema. The model stores only bounded visual-state fields:
`visible_scene`, `person_like_presence`, `vehicle_like_presence`, `motion_claim`,
`safe_observation`, and `unsupported_claims`.

For Node1 RTX 5060 Ti smoke testing with the 4096-token SmolVLM2 server context:

```bash
export MONITORME_SMOLVLM2_PROVIDER=smolvlm2-openai
export MONITORME_ASSISTANT_USE_SMOLVLM2=1
export MONITORME_SMOLVLM2_BASE_URL=http://127.0.0.1:8003/v1
export MONITORME_SMOLVLM2_MODEL_ID=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
export MONITORME_SMOLVLM2_API_KEY=EMPTY
export MONITORME_SMOLVLM2_MAX_FRAMES=1
export MONITORME_SMOLVLM2_MAX_TOKENS=300
export MONITORME_SMOLVLM2_TEMPERATURE=0.0
```

Run validation:

```bash
./scripts/validate_node1_ai_camera_assistant_v04.sh
./scripts/validate_node1_ai_camera_assistant_v041.sh
python -m pytest -q
```

### Node1 non-LLM GPU lab Phase 1: CPU ISP filters

The native `node1_non_llm_gpu_inference_lab` now includes a CPU ISP
memory-locality lab with a true rolling 3-row line buffer. Supported filters are
`blur`, `sharpen`, `edge`, `sobel-x`, `sobel-y`, and `sobel-mag`, with binary
PGM/PPM input/output support for deterministic validation.

Example:

```bash
cd native/node1_non_llm_gpu_inference_lab
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j"$(nproc)"
./build-cpu/node1_non_llm_gpu_lab_selftest
./build-cpu/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-mag --width 64 --height 48
```

MonitorMe CLI wrapper:

```bash
MONITORME_GPU_LAB_BIN=native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab \
  python -m monitor_me.cli gpu-lab-isp-synthetic --filter sobel-mag --width 64 --height 48
```

The ISP path emits only image-processing workload metrics and no object,
identity, behavior, or intent claims.

### Node1 non-LLM GPU lab Phase 2: CUDA ISP filters and metrics

Phase 2 adds a CUDA ISP path that validates against the Phase 1 CPU rolling
line-buffer baseline. The native binary keeps the CPU result in `isp` and, when
compiled with CUDA and invoked with `--gpu`, emits `isp_cuda` plus an
`isp_cpu_cuda_comparison` object.

Implemented CUDA ISP work:

- shared-memory tiled 3x3 filters with halo loading
- CUDA Sobel X, Sobel Y, and Sobel magnitude paths
- CUDA blur, sharpen, and edge filters for CPU-vs-CUDA parity
- CUDA reduction metrics for output mean/edge energy, focus score, noise score,
  lighting delta, saturation pixels, and saturation ratio
- facts-only CPU/CUDA comparison metadata

Example:

```bash
cd native/node1_non_llm_gpu_inference_lab
./scripts/build_node1_gpu_lab.sh
./build/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-mag --width 64 --height 48 --gpu --include-output
./scripts/run_node1_gpu_lab_phase2_isp_cuda_selftest.sh
```

MonitorMe CLI wrapper with CUDA binary:

```bash
MONITORME_GPU_LAB_BIN=native/node1_non_llm_gpu_inference_lab/build/node1_non_llm_gpu_lab \
MONITORME_GPU_LAB_PREFER_CUDA=1 \
  python -m monitor_me.cli gpu-lab-isp-synthetic --filter sobel-mag --width 64 --height 48
```

Validation should confirm `isp_cpu_cuda_comparison.ok=true`,
`output_equal=true`, `metrics_close=true`, `mismatch_count=0`, and
`max_abs_diff=0` for all Phase 1 filters. The ISP CUDA path remains workload
metadata only and does not emit object, person, identity, behavior, intent,
weapon, or suspiciousness claims.


### Node1 non-LLM GPU lab Phase 3 sparse ROI

Phase 3 adds a native sparse ROI crop/resize/normalize path. It walks active motion tiles, creates ROI rectangles, resizes each ROI to a fixed target, normalizes to float32, and compares CPU/CUDA output when a CUDA binary is used.

```bash
MONITORME_GPU_LAB_BIN=native/node1_non_llm_gpu_inference_lab/build/node1_non_llm_gpu_lab \
MONITORME_GPU_LAB_PREFER_CUDA=1 \
  python -m monitor_me.cli gpu-lab-sparse-roi-synthetic \
  --scenario sparse \
  --width 320 \
  --height 240 \
  --target-width 16 \
  --target-height 16
```

The sparse ROI result is facts-only workload metadata and does not emit object, person, identity, behavior, intent, weapon, or suspiciousness claims.

### Node1 non-LLM GPU lab Phase 4

Phase 4 adds the mixed-region path: connected tile components, contiguous vs
scattered classification, and grouped crop/resize/normalize batching with CPU
and CUDA parity checks. The MonitorMe CLI command is:

```bash
python -m monitor_me.cli gpu-lab-mixed-region-synthetic --scenario scattered
```

Use `MONITORME_GPU_LAB_PREFER_CUDA=1` with a CUDA-built native binary to emit
`mixed_region_cuda` and `mixed_region_cpu_cuda_comparison`.

### Node1 non-LLM GPU lab Phase 5

Phase 5 adds the dense full-frame path for high-activity frames. It runs full-frame absolute difference, a 256-bin diff histogram, reductions for previous/current/diff means, lighting delta, changed-pixel counts, and dense uint8-to-float32 normalization with CPU/CUDA parity checks.

```bash
python -m monitor_me.cli gpu-lab-dense-full-frame-synthetic --scenario dense
```

Use `MONITORME_GPU_LAB_PREFER_CUDA=1` with a CUDA-built native binary to emit `dense_full_frame_cuda` and `dense_full_frame_cpu_cuda_comparison`. The dense path remains facts-only workload metadata and does not emit object, person, identity, behavior, intent, weapon, or suspiciousness claims.

### Node1 non-LLM GPU lab Phase 6

Phase 6 adds the overlay-heavy path for visual artifact generation. It produces
a deterministic motion heatmap, alpha-blended RGB overlay, thumbnail RGB output,
and before/after comparison metrics with CPU/CUDA parity checks.

```bash
python -m monitor_me.cli gpu-lab-overlay-heavy-synthetic --scenario mixed
```

Use `MONITORME_GPU_LAB_PREFER_CUDA=1` with a CUDA-built native binary to emit
`overlay_heavy_cuda` and `overlay_heavy_cpu_cuda_comparison`. The overlay-heavy
path remains facts-only workload metadata and does not emit object, person,
identity, behavior, intent, weapon, or suspiciousness claims.

### Node1 non-LLM GPU lab Phase 7

Phase 7 adds the AudioBox path for facts-only audio signal processing. It computes per-window RMS, peak, silence masks, onset masks, and bounded cross-correlation sync drift between a primary and reference track. The native mode is:

```bash
python -m monitor_me.cli gpu-lab-audiobox-synthetic --sync-drift-samples 64
```

When the CUDA build is available and `MONITORME_GPU_LAB_PREFER_CUDA=1`, the result includes `audiobox`, `audiobox_cuda`, and `audiobox_cpu_cuda_comparison`. The AudioBox path reports signal metrics only and does not infer speech content, speaker identity, behavior, or intent.

### Node1 non-LLM GPU lab Phase 8

Phase 8 adds the storage batch planner and clip sampler path. It scans clip manifest metadata, builds a deterministic batch read plan, selects key moments, and emits clip timeline features. The native mode is:

```bash
python -m monitor_me.cli gpu-lab-storage-batch-synthetic --clips 12
```

The result includes `storage_batch` facts with batch count, planned read bytes, key moments, timeline span, gap metrics, and manifest metadata when `--include-output` is used natively. This path is metadata-only and does not decode media or infer visual/audio semantics, identity, behavior, or intent.

### Node1 non-LLM GPU lab Phase 9

Phase 9 adds a combined evidence pipeline for the post-storage-planning roadmap items: visual fingerprint workload vectors, evidence dedup grouping, dedup-aware key-moment selection, storage batch-read plan reuse, latency/throughput counters, safety validation, native/Python bridge expansion, CLI exposure, and documentation.

The native mode is:

```bash
python -m monitor_me.cli gpu-lab-evidence-pipeline-synthetic \
  --clips 12 \
  --max-batch-bytes 1600000 \
  --max-batch-clips 3 \
  --key-moments 4 \
  --min-key-gap-ms 1000 \
  --dedup-hamming-threshold 0 \
  --fingerprint-cycle 6
```

The result includes `evidence_pipeline`, nested `storage_batch`, `fingerprints`, `duplicate_groups`, dedup-aware `key_moments`, `timeline`, `latency`, and `safety`. The path is facts-only and does not decode media or make object, identity, speech-content, behavior, or intent claims.

### Node1 non-LLM GPU lab Phase 10

Phase 10 integrates the Phase 9 evidence pipeline into real `capture-run` sessions. When `--evidence-pipeline-enabled` is used, MonitorMe converts stored local motion keyframes into a facts-only evidence CSV manifest, runs the native evidence pipeline in manifest mode, stores the JSON profile as a capture artifact, and inserts a session-level `evidence_pipeline_indexed` event.

The capture-run integration remains local and metadata-only. It does not decode media through the evidence pipeline and does not emit object, person, identity, speech-content, behavior, intent, weapon, suspiciousness, or semantic audio/visual claims.

Example:

```bash
python -m monitor_me.cli capture-run \
  --camera-id c922_node1_gate \
  --device /dev/video0 \
  --duration-sec 10 \
  --evidence-pipeline-enabled \
  --evidence-pipeline-binary native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab \
  --evidence-pipeline-max-batch-bytes 1600000 \
  --evidence-pipeline-max-batch-clips 3 \
  --evidence-pipeline-key-moments 4 \
  --evidence-pipeline-min-key-gap-ms 1000 \
  --evidence-pipeline-dedup-hamming-threshold 0
```

Generated artifacts include:

- `evidence_pipeline_manifest_csv`
- `evidence_pipeline_profile`

The final capture manifest records `evidence_pipeline_event_ids`, `evidence_pipeline_artifact_ids`, and a compact `evidence_pipeline.last_result` summary.

### Phase 11: real media fingerprint ingestion after decoded keyframes

Phase 11 upgrades the capture-run evidence pipeline from manifest-only synthetic workload fingerprints to explicit decoded-keyframe fingerprint ingestion. When `--evidence-pipeline-enabled` is used, stored local JPEG keyframes are decoded after capture, resized to the configured fingerprint shape, and converted into facts-only fingerprint columns in `evidence_pipeline/capture_evidence_manifest.csv`.

The evidence CSV now includes optional real media fingerprint columns:

```text
fingerprint_source,decoded_width,decoded_height,ahash64,dhash64,fingerprint64,histogram16
```

The native evidence pipeline consumes those precomputed values in `evidence-pipeline-manifest` mode. It reports `real_media_ingestion=true`, `media_fingerprint_count`, `synthetic_fingerprint_count`, and per-fingerprint `from_media` / `fingerprint_source` facts. If real decode is disabled or unavailable, it falls back to the existing manifest-metadata synthetic fingerprint path.

Real fingerprint decoding is local-only and limited to already-stored keyframes. It does not upload frames, identify people, infer intent, decode speech, or emit semantic visual/audio claims.

Disable real keyframe decoding while keeping evidence indexing enabled:

```bash
python -m monitor_me.cli capture-run \
  --camera-id c922_node1_gate \
  --device /dev/video0 \
  --duration-sec 10 \
  --evidence-pipeline-enabled \
  --evidence-pipeline-no-real-fingerprints
```

Validate Phase 11:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase11_real_media_fingerprint_selftest.sh
python -m pytest -q tests/test_node1_capture_real_fingerprint_phase11.py
```

### Phase 12: evidence index persistence / DB migration

Phase 12 persists the capture-run evidence pipeline profile into normalized SQLite tables instead of relying only on the large JSON artifact and `events.attrs_json`. The migration `005_evidence_index_persistence.sql` creates queryable rows for evidence profiles, per-keyframe fingerprints, duplicate groups, and key moments.

After `--evidence-pipeline-enabled`, capture-run still writes:

```text
capture_evidence_manifest.csv
evidence_pipeline_profile.json
evidence_pipeline_indexed event
```

It now also persists:

```text
evidence_pipeline_profiles
evidence_fingerprints
evidence_dedup_groups
evidence_key_moments
```

Read back persisted evidence index rows:

```bash
python -m monitor_me.cli evidence-index --session-id <session_id>
python -m monitor_me.cli evidence-fingerprints --session-id <session_id> --from-media --limit 20
python -m monitor_me.cli evidence-dedup-groups --session-id <session_id>
python -m monitor_me.cli evidence-key-moments --session-id <session_id>
```

The persisted index remains facts-only. It stores storage paths, hashes, fingerprint integers, histogram bins, duplicate accounting, key-moment metadata, safety validator output, and latency/throughput counters. It does not store identity, behavior, intent, suspiciousness, object claims, or speech-content claims.

Validate Phase 12:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase12_evidence_index_selftest.sh
python -m pytest -q tests/test_node1_evidence_index_phase12.py
```

### Phase 13 — Evidence pipeline summary API

Phase 13 exposes the persisted evidence index from Phase 12 through local FastAPI endpoints for UI/dashboard consumption:

```text
GET /evidence/pipeline/summaries
GET /evidence/pipeline/sessions/{session_id}/summary
GET /evidence/pipeline/profiles/{profile_id}/summary
GET /evidence/pipeline/profiles/{profile_id}/fingerprints
GET /evidence/pipeline/profiles/{profile_id}/dedup-groups
GET /evidence/pipeline/profiles/{profile_id}/key-moments
```

These endpoints summarize evidence profile metadata, decoded-keyframe fingerprint rows, duplicate groups, key moments, timeline facts, latency/throughput facts, and safety-validator status from SQLite. They are local-only and facts-only: no media decode occurs inside the API request path, no frame upload occurs, and no identity/intent/speech-content/weapon/suspiciousness claims are emitted.

Validation:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase13_evidence_api_selftest.sh
python -m pytest -q tests/test_node1_evidence_api_phase13.py
```

### Phase 14 — Evidence index retention / compaction policies

Phase 14 adds retention and compaction controls for the persisted evidence index introduced in Phase 12 and exposed through the Phase 13 API. The policy layer prunes only normalized evidence-index rows:

- `evidence_pipeline_profiles`
- `evidence_fingerprints`
- `evidence_dedup_groups`
- `evidence_key_moments`

It deliberately does **not** delete source events, capture sessions, artifact rows, keyframe JPEGs, capture manifests, evidence CSV manifests, or evidence profile JSON artifacts. The goal is to keep the query index bounded while preserving the original local evidence trail for audit/rebuild.

The migration `006_evidence_index_retention.sql` adds `evidence_retention_runs`, which records every dry-run/apply operation with policy JSON, selected profile counts, child-row counts, estimated index payload size, compaction flags, and DB size before/after.

CLI usage:

```bash
python -m monitor_me.cli evidence-retention-plan \
  --older-than-days 30 \
  --keep-last-per-camera 1 \
  --keep-last-per-session 1 \
  --camera-id c922_node1_gate

python -m monitor_me.cli evidence-retention-apply \
  --dry-run \
  --older-than-days 30 \
  --keep-last-per-camera 1 \
  --keep-last-per-session 1

python -m monitor_me.cli evidence-retention-apply \
  --yes \
  --older-than-days 30 \
  --keep-last-per-camera 1 \
  --keep-last-per-session 1 \
  --vacuum

python -m monitor_me.cli evidence-retention-runs --limit 20
```

API usage:

```bash
curl -sS 'http://127.0.0.1:8088/evidence/pipeline/retention/plan?older_than_days=30&keep_last_per_camera=1&keep_last_per_session=1' | python3 -m json.tool
curl -sS -X POST 'http://127.0.0.1:8088/evidence/pipeline/retention/apply?dry_run=true&older_than_days=30' | python3 -m json.tool
curl -sS -X POST 'http://127.0.0.1:8088/evidence/pipeline/retention/apply?dry_run=false&confirm=true&older_than_days=30' | python3 -m json.tool
curl -sS 'http://127.0.0.1:8088/evidence/pipeline/retention/runs?limit=20' | python3 -m json.tool
```

Safety contract:

- retention is local-only
- retention deletes index rows only
- retention does not decode media
- retention does not upload frames
- retention does not create semantic claims
- retention keeps events/artifacts/files intact so the index can be rebuilt later

Validate Phase 14:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase14_evidence_retention_selftest.sh
python -m pytest -q tests/test_node1_evidence_retention_phase14.py
```

---

## Phase 15 — Operator dashboard / UI integration

MonitorMe now includes a local read-only operator dashboard for the persisted evidence pipeline index.

Start the local API, then open:

```text
http://127.0.0.1:8088/operator/dashboard
```

JSON context:

```text
GET /operator/dashboard/data
```

The dashboard summarizes local facts only:

```text
evidence profiles
media vs synthetic fingerprint counts
key moments
dedup groups
safety validator status
latency/throughput facts
retention run audit records
```

Safety boundaries:

```text
no external upload
no raw frame upload
no API-time media decode
no external web assets
no identity / intent / speech-content claims
no destructive retention action from the dashboard
```
