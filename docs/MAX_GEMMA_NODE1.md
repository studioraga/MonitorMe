# Node1 MAX + Gemma 3 1B serving for MonitorMe v0.2

This document captures the known-good Node1 MAX + Gemma 3 1B serving path that MonitorMe uses for optional v0.2 assistant summaries.

MonitorMe keeps the model split strict:

```text
YOLO ONNX
  -> fast visual facts: label, confidence, bbox, model_id

Node1 deterministic policy
  -> decisions: action, reason, severity, review duration

Gemma 3 1B through MAX
  -> explanation only over event_contract + policy_decision + artifact metadata
  -> no raw CCTV frames
  -> no identity, face, intent, danger, weapon, or suspicious-behavior claims
```

## Prerequisite

The helper scripts assume your existing MAX pixi project exists here:

```text
~/dev/modular/project/quickstart
```

Override when needed:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart"
```

The MAX project should contain `pixi.toml` and have access to:

```text
google/gemma-3-1b-it
```

If Hugging Face access is missing, run once inside the MAX project:

```bash
cd ~/dev/modular/project/quickstart
pixi run hf auth login
```

Then accept/request Gemma access with the same Hugging Face account.

## TERM1: start MAX

From the MonitorMe repo:

```bash
./scripts/max/term1_start_max_gemma3_1b.sh
```

Keep this terminal running.

The script uses your known-good Node1 RTX 5060 Ti serving flags:

```text
MODULAR_DEBUG=device-sync-mode
max serve
  --model google/gemma-3-1b-it
  --devices=gpu
  --max-length 1024
  --max-batch-size 1
  --max-batch-input-tokens 1024
  --max-batch-total-tokens 1024
  --device-memory-utilization 0.90
  --sample-on-host
  --temperature 0.0
```

`--sample-on-host` is intentionally preserved because this was the known-good workaround for the RTX 5060 Ti / MAX nightly GPU fused top-k sampler launch-resource crash path.

## TERM2: validate standalone MAX

From the MonitorMe repo in a second terminal:

```bash
./scripts/max/term2_validate_max_gemma3_1b.sh
```

This validates:

```text
127.0.0.1:8000 /v1/models
127.0.0.1:8001 /metrics
Python OpenAI-compatible client
curl /v1/chat/completions
5 repeatability requests
nvidia-smi after inference
filtered MAX metrics
```

Artifacts are saved under the MAX project results directory, for example:

```text
~/dev/modular/project/quickstart/results/max_gemma3_1b_YYYYmmdd_HHMMSS/
```

## TERM2: validate MonitorMe + live MAX strict JSON summaries

After standalone MAX passes, validate MonitorMe's Gemma path:

```bash
source .venv/bin/activate
python -m pip install -e '.[api,camera,test]'
./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
```

This script:

```text
checks MAX /v1/models
exports MonitorMe Gemma settings
runs monitor_me.cli llm-health --probe
creates a local validation event in SQLite using synthetic frames and an injected detector
calls Gemma/MAX through the real MonitorMe v0.2 client
requires summary_source=gemma_max
writes validation artifacts under results/max_gemma_monitorme_v02_*/
```

The synthetic validation frame source does not open `/dev/video0`; it is only used to exercise the MonitorMe event-contract and summary path. Real camera validation remains:

```bash
./scripts/validate_node1_c922_yolo_live.sh
```

## Environment used by MonitorMe

```bash
export MONITORME_LLM_PROVIDER=max-openai
export MONITORME_ASSISTANT_USE_GEMMA=1
export MONITORME_LLM_BASE_URL=http://127.0.0.1:8000/v1
export MONITORME_LLM_MODEL_ID=google/gemma-3-1b-it
export MONITORME_LLM_API_KEY=EMPTY
export MONITORME_LLM_TEMPERATURE=0.0
export MONITORME_LLM_TIMEOUT_SEC=120
export MONITORME_LLM_MAX_TOKENS=192
```

## Health checks

Configuration-only health:

```bash
python -m monitor_me.cli --db data/events/monitorme.db llm-health --allow-unconfigured
```

Configured endpoint probe:

```bash
MONITORME_LLM_PROVIDER=max-openai \
MONITORME_ASSISTANT_USE_GEMMA=1 \
python -m monitor_me.cli --db data/events/monitorme.db llm-health --probe
```

API equivalent:

```bash
curl -sS 'http://127.0.0.1:8088/assistant/llm/health?probe=true' | python3 -m json.tool
```

## Troubleshooting

If TERM1 fails before serving:

```text
pixi.toml not found
  -> MONITORME_MAX_PROJECT_DIR is wrong or the MAX quickstart project has not been created.

Hugging Face login missing
  -> run pixi run hf auth login in the MAX quickstart project.

Cannot access google/gemma-3-1b-it
  -> accept/request Gemma access on Hugging Face with the same account.
```

If TERM2 cannot reach the API:

```text
Check TERM1 is still running.
Check port 8000 is listening with: ss -ltnp | grep :8000
Check MAX logs: ~/dev/modular/project/quickstart/max_gemma3_1b_sample_on_host.log
```

If MonitorMe falls back to deterministic summaries:

```text
Gemma returned invalid JSON, cited unknown event IDs, used an invalid severity,
or introduced unsupported claims such as identity, face recognition, intent,
weapons, danger, threat, or suspicious behavior.
```

Fallback is safe and expected when Gemma/MAX is unavailable or invalid; it means MonitorMe preserved evidence flow and did not trust unsupported text.


## v0.2.2 startup-script safety update

The MonitorMe MAX helper scripts are intentionally conservative. They **do not** run `pixi add modular huggingface_hub openai` during every startup anymore. The earlier behavior could mutate a previously working MAX quickstart environment when the `max-nightly` channel advanced. On Node1 this showed up as a Mojo compile failure:

```text
unable to locate module 'std'
from std.os import abort
```

That error is an external MAX/Mojo pixi environment problem, not a MonitorMe event DB or assistant-summary problem. It usually means the MAX/Mojo packages inside `$HOME/dev/modular/project/quickstart/.pixi` are mismatched or partially upgraded.

Use this read-only diagnostic first:

```bash
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
./scripts/max/diagnose_max_gemma3_1b_env.sh
```

Only intentionally change the MAX pixi environment when you choose to do so. To allow the helper to sync packages explicitly:

```bash
MONITORME_MAX_PIXI_SYNC=1 ./scripts/max/term1_start_max_gemma3_1b.sh
```

Recommended recovery when Mojo cannot locate `std`:

```bash
cd ~/dev/modular/project/quickstart
# Best path if you have a known-good lock file:
git checkout -- pixi.toml pixi.lock
pixi install --locked

# Then from MonitorMe:
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
./scripts/max/term1_start_max_gemma3_1b.sh
```

If you do not have a known-good `pixi.lock`, recreate the quickstart environment using the exact MAX recipe/version that previously validated on Node1, then keep `pixi.lock` under version control or backup.

## Distinguishing expected validation results from failures

These results are expected and are not MonitorMe code failures:

```text
llm-health --probe -> connection refused
```

This means MAX is not running yet or Gemma is not enabled. Start MAX in TERM1 first.

```text
curl http://127.0.0.1:8088/... -> connection refused
```

This means the MonitorMe FastAPI server is not running. Start it with:

```bash
MONITORME_DB=data/events/monitorme.db ./scripts/run_api.sh
```

```text
object_events=0, motion_event_ids=[]
```

This means the camera validation completed but no motion/object crossed the threshold during the capture window. Move in front of the camera or lower the motion threshold.

## Recovery: quickstart is not a git repo and MAX cannot locate Mojo `std`

If `git checkout -- pixi.toml pixi.lock` fails with:

```text
fatal: not a git repository (or any of the parent directories): .git
```

then the external MAX quickstart directory cannot be restored with Git. If
`pixi install --locked` still leaves `max serve` failing with:

```text
error: unable to locate module 'std'
from std.os import abort
```

then the current pixi solve is internally mismatched. In the observed Node1
failure, the quickstart environment had selected Python 3.14 and MAX/Mojo nightly
packages. `max --help` worked, but `max serve` failed when MAX imported
`max._core_mojo`.

Use the clean Python 3.12 recovery path:

```bash
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
source .venv/bin/activate

./scripts/max/create_max_gemma3_1b_py312_env.sh

export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312"
./scripts/max/term1_start_max_gemma3_1b.sh
```

In TERM2:

```bash
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
source .venv/bin/activate
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312"

./scripts/max/term2_validate_max_gemma3_1b.sh
./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
```

The helper creates a separate workspace by default, so the old broken
`~/dev/modular/project/quickstart` directory is not overwritten.



## v0.2.4 MAX Mojo import-path hardening

If a clean Python 3.12 MAX pixi workspace still fails at `max serve` with `unable to locate module 'std'` or `unable to locate module 'nn'`, use the v0.2.4 helpers. The TERM1 and diagnostic scripts now discover Mojo package roots under `.pixi/envs/default`, export `MODULAR_MOJO_MAX_IMPORT_PATH` and `MOJO_PACKAGE_PATH`, clear stale `__mojocache__` directories by default, and preflight both `max._core_mojo` and `max._kv_cache_ops` before starting the server.

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312"
./scripts/max/diagnose_max_gemma3_1b_env.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```

Compiled Mojo caches are preserved by default. Set `MONITORME_MAX_CLEAR_MOJO_CACHE=1` only if you intentionally want to force recompilation while debugging stale cache artifacts.

## v0.2.5 stable-channel fallback for broken nightly MAX solves

If `quickstart_py312` was created successfully but `diagnose_max_gemma3_1b_env.sh`
shows that `max._kv_cache_ops` fails with `unable to locate module 'std'` or
`unable to locate module 'nn'`, then the installed MAX/Mojo nightly package set is
not usable for `max serve` on this Node1 environment.

In that case, keep the failing nightly workspace for diagnostics and create a
separate stable-channel workspace:

```bash
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
source .venv/bin/activate

./scripts/max/create_max_gemma3_1b_stable_py312_env.sh
```

This creates:

```text
~/dev/modular/project/quickstart_py312_stable
```

Start MAX with:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable"
./scripts/max/term1_start_max_gemma3_1b.sh
```

Validate from TERM2:

```bash
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable"
./scripts/max/term2_validate_max_gemma3_1b.sh
./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
```

Why stable: the Modular package docs describe both nightly and stable builds;
nightly has newer features but can change frequently, while stable is the better
choice when the current nightly solve is broken.

## v0.2.6 recovery: stable workspace with explicit Mojo package

If both the nightly Python 3.12 workspace and the stable Python 3.12 workspace fail with:

```text
unable to locate module 'std'
unable to locate module 'nn'
max._kv_cache_ops
```

then create a third workspace that explicitly installs the standalone `mojo` Conda package alongside `modular`:

```bash
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
source .venv/bin/activate

./scripts/max/create_max_gemma3_1b_stable_py312_with_mojo_env.sh

export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable_mojo"
./scripts/max/probe_mojo_std_nn_roots.sh
./scripts/max/term1_start_max_gemma3_1b.sh
```

In TERM2:

```bash
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
source .venv/bin/activate
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable_mojo"

./scripts/max/term2_validate_max_gemma3_1b.sh
./scripts/max/term2_validate_monitorme_gemma_v02_live.sh
```

This keeps all failed workspaces intact for evidence:

```text
quickstart                  original workspace
quickstart_py312            nightly Python 3.12 workspace
quickstart_py312_stable     stable Python 3.12 modular-only workspace
quickstart_py312_stable_mojo stable Python 3.12 modular + explicit mojo workspace
```


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

## v0.2.11: MAX graph/kernel package path crash

Node1 v0.2.10 logs showed that MAX can now pass the earlier Mojo importer phase,
start `max serve`, load the Gemma pipeline configuration, start the worker, and
find local Hugging Face weights. The next failure was inside MAX graph/kernel
registration:

```text
error: failed to resolve built-in kernel package paths
ValueError: MAXG_addKernelPackage: failed to import kernels from ''
```

This is different from the earlier `unable to locate module 'std'` importer
failure. Manual `mojo build` with explicit `-I` paths proved the installed Mojo
packages are present. For this MAX 26.4.0 path, set the Mojo package roots in the
Pixi workspace activation environment so child worker processes inherit them from
Pixi itself:

```bash
cd ~/dev/pub/ai-sys1/MonitorME/MonitorMe
export MONITORME_MAX_PROJECT_DIR="$HOME/dev/modular/project/quickstart_py312_stable_mojo"
./scripts/max/patch_max_pixi_activation_env.sh
```

Then retry TERM1 with cache preservation and cache prebuild enabled:

```bash
export MONITORME_MAX_CLEAR_MOJO_CACHE=0
export MONITORME_MAX_OVERRIDE_MOJO_IMPORT_PATH=1
export MONITORME_MAX_SKIP_HELP_PREFLIGHT=1
export MONITORME_MAX_SAMPLE_ON_HOST=0
export MONITORME_MAX_PREBUILD_MOJO_CACHES=1
./scripts/max/term1_start_max_gemma3_1b.sh
```


### v0.2.12 MAX runtime Mojo kernel-root narrowing

Node1 logs showed that the MAX server can progress past Mojo cache prebuild, Hugging Face access, Gemma pipeline configuration, and worker startup, then fail during graph custom-op registration with `MAXG_addKernelPackage: failed to import kernels from ''`. v0.2.12 keeps `MODULAR_MOJO_MAX_IMPORT_PATH`/`MOJO_PACKAGE_PATH` focused on the real compiled Mojo package root (`.pixi/envs/default/lib/mojo`) for the runtime process while cache prebuilds still compile with explicit `-I lib/mojo -I site-packages/max`. This avoids exporting a broad `lib/mojo:site-packages/max` runtime path that can confuse built-in kernel package resolution.


### v0.2.13 pytest helper fix

Fixes a test-only regression in `tests/test_max_helper_health.py` where newly added
MAX helper regression tests referenced `ROOT` without defining it. Runtime MAX/Gemma
startup behavior is unchanged from v0.2.12.
