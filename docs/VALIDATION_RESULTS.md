# Validation Results

Latest packaged validation:

```text
=== MonitorMe Step 17B validation ===
........... [100%]
=== MonitorMe Step 17B validation PASSED ===
```

The Step 17B validation suite does not seed demo CCTV data. It validates the real capture runner using a test-only frame source so CI/development machines without a physical C922 can still prove persistence, artifacts, assistant grounding, FactGuard, API routes, and reports.

Node1 physical validation should be run on the machine with the Logitech C922:

```bash
./scripts/validate_node1_c922_live.sh
```

Expected physical output:

- a completed `capture_sessions` row;
- real keyframe artifacts under `data/captures/<session_id>/keyframes/` if motion is detected;
- real `motion_detected` event rows;
- assistant answer containing `event_id`, `session_id`, and `frame_id`.

## Node1 AI Camera Assistant v0.1 validation result

Package-level validation passed in the sandbox:

```text
./scripts/validate_node1_ai_camera_assistant_v01.sh
10 passed
```

Full pytest passed:

```text
27 passed
```

The uploaded Node1 runtime database also showed that live validation produced MonitorMe assistant data:

```text
events: 9
assistant_summaries: 11
event_contracts: 9
capture_artifacts: 9
artifact types: keyframe, annotated_keyframe, capture_manifest
```

Use `sqlite3` for database inspection. Do not use `cat data/events/monitorme.db` because the database is binary.

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

## v0.2.3 MAX environment recovery helper

Added a stronger MAX/Mojo preflight and clean Python 3.12 pixi workspace creation
script after Node1 logs showed `max serve` could fail with `unable to locate
module 'std'` even though `max --help` succeeded. The new helper avoids rewriting
the existing quickstart and creates `$HOME/dev/modular/project/quickstart_py312`
by default.



## v0.2.4 MAX Mojo import-path hardening

If a clean Python 3.12 MAX pixi workspace still fails at `max serve` with `unable to locate module 'std'` or `unable to locate module 'nn'`, use the v0.2.4 helpers. The TERM1 and diagnostic scripts now discover Mojo package roots under `.pixi/envs/default`, export `MODULAR_MOJO_MAX_IMPORT_PATH` and `MOJO_PACKAGE_PATH`, clear stale `__mojocache__` directories by default, and preflight both `max._core_mojo` and `max._kv_cache_ops` before starting the server.

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312"
./scripts/max/diagnose_max_gemma3_1b_env.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```

Compiled Mojo caches are preserved by default. Set `MONITORME_MAX_CLEAR_MOJO_CACHE=1` only if you intentionally want to force recompilation while debugging stale cache artifacts.

## v0.2.5 MAX stable-channel recovery helper

Added `scripts/max/create_max_gemma3_1b_stable_py312_env.sh` for cases where the
Python 3.12 max-nightly workspace is created successfully but fails serving-time
Mojo compilation in `max._kv_cache_ops` with missing `std` or `nn` modules.

The helper creates a separate stable-channel workspace at:

```text
~/dev/modular/project/quickstart_py312_stable
```

MonitorMe validation remains independent of MAX availability. If MAX is down or
invalid, Node1 AI Camera Assistant v0.2 stores deterministic fallback summaries.

## v0.2.6 MAX explicit-Mojo fallback

Observed Node1 state: MonitorMe v0.2.5 validation passed, but the stable MAX workspace still failed during serving-time compilation of `max._kv_cache_ops` with missing Mojo `std` and `nn` modules. v0.2.6 adds a third recovery workspace that installs `modular` and explicit `mojo` together, plus a probe script for package roots.


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

## v0.2.11 validation note

v0.2.10 Node1 logs showed MonitorMe validation passing while MAX progressed to a
new failure stage: the Gemma server started, the worker initialized, local model
weights were found, and then MAX graph/kernel registration failed with:

```text
failed to resolve built-in kernel package paths
MAXG_addKernelPackage: failed to import kernels from ''
```

v0.2.11 adds a helper to patch the external MAX Pixi workspace with
`[activation.env]` entries for `MODULAR_MOJO_MAX_IMPORT_PATH` and
`MOJO_PACKAGE_PATH`. This is intentionally separated from the MonitorMe Python
runtime so the MAX workspace mutation is explicit and reviewable.


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
