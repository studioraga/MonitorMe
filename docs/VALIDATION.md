# Validation

## Standard validation

```bash
./scripts/validate_step17b_monitorme.sh
```

This runs the unit/integration tests and does not seed demo CCTV data.

Expected:

```text
=== MonitorMe Step 17B validation PASSED ===
```

## Physical Node1 C922 validation

Run on Node1 with the Logitech C922 connected as `/dev/video0`:

```bash
./scripts/validate_node1_c922_live.sh
```

Move in front of the camera during the capture window. If no motion is emitted:

```bash
MONITORME_MOTION_THRESHOLD=0.5 ./scripts/validate_node1_c922_live.sh
```


## Test dependency note

The `test` extra includes `httpx2>=2.5.0` because the current Starlette/FastAPI TestClient stack requires it. Run `python -m pip install -e '.[api,camera,test]'` before validation.

## YOLO model download script validation

Standard validation also checks that `scripts/models/download_yolo_onnx.sh` is present, shows help, and can persist MonitorMe detector variables without requiring network access when a model file already exists.

For real model download on Node1:

```bash
./scripts/models/download_yolo_onnx.sh
python -m pip install -e '.[api,camera,detector,test]'
./scripts/validate_node1_c922_yolo_live.sh
```


## Step 17D detector health

MonitorMe v0.1.8 adds `python -m monitor_me.cli detector-health` and `GET /models/detector/health` so Node1 can validate the YOLO ONNX model path, checksum, ONNX Runtime providers, and model input/output metadata before live camera capture. See `docs/STEP17D_DETECTOR_HEALTH.md`.


## Step 17E

```bash
./scripts/validate_step17e_monitorme.sh
```

This validates that annotated overlay artifacts are generated from normalized `object_detected` rows while raw keyframes remain unchanged.

## Node1 AI Camera Assistant v0.1 validation

Install the base validation dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[api,camera,test]'
```

Run the offline assistant milestone validation:

```bash
./scripts/validate_node1_ai_camera_assistant_v01.sh
```

Expected:

```text
=== MonitorMe Node1 AI Camera Assistant v0.1 validation PASSED ===
```

Then run live C922 + YOLO validation on Node1:

```bash
./scripts/models/download_yolo_onnx.sh
python -m pip install -e '.[api,camera,detector,test]'
./scripts/validate_node1_c922_yolo_live.sh
```

The live validation should produce:

```text
motion_detected rows
object_detected rows when YOLO detects objects
keyframe artifacts
overlay artifacts
assistant_summaries rows
event_contracts rows
```

Inspect the SQLite database with `sqlite3`, not `cat`:

```bash
sqlite3 data/events/monitorme.db ".tables"
sqlite3 data/events/monitorme.db "select event_type,label,count(*) from events group by event_type,label;"
sqlite3 data/events/monitorme.db "select count(*) from assistant_summaries;"
sqlite3 data/events/monitorme.db "select count(*) from event_contracts;"
```

See `docs/NODE1_AI_CAMERA_ASSISTANT_VALIDATION.md` for the full validation sequence.

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


## v0.2.2 MAX startup-script safety

MonitorMe v0.2.2 changes the MAX/Gemma helper scripts so they do **not** mutate the external pixi MAX quickstart environment by default. If `scripts/max/term1_start_max_gemma3_1b.sh` reports `unable to locate module 'std'`, run:

```bash
./scripts/max/diagnose_max_gemma3_1b_env.sh
```

That failure is a broken/mismatched MAX/Mojo pixi environment, usually after a rolling nightly package upgrade, not a MonitorMe assistant-code failure. Restore a known-good `pixi.lock` with `pixi install --locked` or intentionally rebuild the quickstart environment. Only set `MONITORME_MAX_PIXI_SYNC=1` when you explicitly want the helper to run `pixi add`.

## MAX/Gemma recovery validation for Python 3.14 / Mojo stdlib failure

If MAX fails with `unable to locate module 'std'`, first run:

```bash
./scripts/max/diagnose_max_gemma3_1b_env.sh
```

If the diagnostic shows Python 3.14 or a missing `pixi.lock`, create a clean
Python 3.12 MAX workspace:

```bash
./scripts/max/create_max_gemma3_1b_py312_env.sh
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312"
./scripts/max/term1_start_max_gemma3_1b.sh
```

Then validate from a second terminal:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312"
./scripts/max/term2_validate_max_gemma3_1b.sh
./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
```



## v0.2.4 MAX Mojo import-path hardening

If a clean Python 3.12 MAX pixi workspace still fails at `max serve` with `unable to locate module 'std'` or `unable to locate module 'nn'`, use the v0.2.4 helpers. The TERM1 and diagnostic scripts now discover Mojo package roots under `.pixi/envs/default`, export `MODULAR_MOJO_MAX_IMPORT_PATH` and `MOJO_PACKAGE_PATH`, clear stale `__mojocache__` directories by default, and preflight both `max._core_mojo` and `max._kv_cache_ops` before starting the server.

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312"
./scripts/max/diagnose_max_gemma3_1b_env.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```

Compiled Mojo caches are preserved by default. Set `MONITORME_MAX_CLEAR_MOJO_CACHE=1` only if you intentionally want to force recompilation while debugging stale cache artifacts.

## v0.2.5 validation path when max-nightly fails at `_kv_cache_ops`

If TERM1 or the diagnostic script fails with:

```text
max._kv_cache_ops
unable to locate module 'std'
unable to locate module 'nn'
```

then run the stable-channel recovery:

```bash
./scripts/max/create_max_gemma3_1b_stable_py312_env.sh
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable"
./scripts/max/term1_start_max_gemma3_1b.sh
```

Then in TERM2:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable"
./scripts/max/term2_validate_max_gemma3_1b.sh
./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
```

### MAX/Gemma std/nn package-root recovery

If `max serve` fails while compiling `max._kv_cache_ops` with `unable to locate module 'std'` or `unable to locate module 'nn'`, create the explicit-Mojo stable workspace:

```bash
./scripts/max/create_max_gemma3_1b_stable_py312_with_mojo_env.sh
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable_mojo"
./scripts/max/probe_mojo_std_nn_roots.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```

MonitorMe v0.2 remains safe if MAX is unavailable because assistant summaries fall back to deterministic event-contract summaries.


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


### v0.2.12 MAX runtime Mojo kernel-root narrowing

Node1 logs showed that the MAX server can progress past Mojo cache prebuild, Hugging Face access, Gemma pipeline configuration, and worker startup, then fail during graph custom-op registration with `MAXG_addKernelPackage: failed to import kernels from ''`. v0.2.12 keeps `MODULAR_MOJO_MAX_IMPORT_PATH`/`MOJO_PACKAGE_PATH` focused on the real compiled Mojo package root (`.pixi/envs/default/lib/mojo`) for the runtime process while cache prebuilds still compile with explicit `-I lib/mojo -I site-packages/max`. This avoids exporting a broad `lib/mojo:site-packages/max` runtime path that can confuse built-in kernel package resolution.


### v0.2.13 pytest helper fix

Fixes a test-only regression in `tests/test_max_helper_health.py` where newly added
MAX helper regression tests referenced `ROOT` without defining it. Runtime MAX/Gemma
startup behavior is unchanged from v0.2.12.


### Node1 AI Camera Assistant v0.3

```bash
./scripts/validate_node1_ai_camera_assistant_v03.sh
python -m pytest -q
```

Validates optional local Qwen VLM keyframe analysis after trigger, strict JSON validation, local-only guardrails, failed-analysis storage, and disabled-by-default behavior.


## Node1 AI Camera Assistant v0.4 — SmolVLM2 short clip experiments

Adds optional local-only SmolVLM2 experiments over stored short clip bundles after a trigger. Disabled by default. v0.4.1 constrains SmolVLM2 with vLLM `structured_outputs` JSON schema, stores bounded visual-state fields in `smolvlm2_clip_experiments`, and rejects identity, person-profile, intent, threat, weapon, and unsupported evidence-ID claims.

Validation command:

```bash
./scripts/validate_node1_ai_camera_assistant_v04.sh
./scripts/validate_node1_ai_camera_assistant_v041.sh
```
