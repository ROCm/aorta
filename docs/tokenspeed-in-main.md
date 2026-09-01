# TokenSpeed under AORTA — what is in `main` today

Verified against `ROCm/aorta` `main` at commit `7f82258`, which contains
[#404](https://github.com/ROCm/aorta/pull/404) (`8e14505`, probe integration) and
[#407](https://github.com/ROCm/aorta/pull/407) (`7f82258`, serving workload).
Everything below was read out of the tree at that commit; nothing here describes
unmerged work.

## What this is

[TokenSpeed](https://github.com/lightseekorg/tokenspeed) is a third-party,
AMD-optimized LLM inference engine. AORTA can now triage it on gfx950: run its
kernels, run its own operator test suites, bring its server up, benchmark how
fast it serves, and put its JIT-compiled kernels through AORTA's sanitizer
pipeline.

No TokenSpeed source is vendored. The engine is consumed only through its
published container image (`lightseekorg/tokenspeed-amd`), driven through its
public CLIs (`tokenspeed serve`, `tokenspeed bench serve`,
`python -m tokenspeed_kernel.benchmark`, `python -m tokenspeed_kernel.numerics`)
and the test tree the image already ships at `/workspace`.

Two of the five routes carry a *verdict* only; one carries *numbers*. That split
is structural, not an oversight: `mode: probe` runs on AORTA's built-in
`_subprocess` path, which wraps an opaque command and has no metrics channel, so
a serving latency cannot come out of it. Only a workload class can populate
`WorkloadResult.metrics`, and that is what `tokenspeed_serve` exists for.

## What you get, route by route

| Route | Recipe mode | Question it answers | Verdict comes from | Cost |
|---|---|---|---|---|
| Kernel probe | `probe` | Do the kernels compute the right bytes, and how fast? | exit code (script re-reads the exported JSON) | ~17 s per trial |
| Suite probe | `probe` | Do TokenSpeed's own operator suites pass? | exit code (script requires ≥1 executed test) | ~1–4 min per suite |
| Serving probe | `probe` | Does a model come up and generate a token? | exit code | ~5 min per trial, noisy |
| `tokenspeed_serve` | `triage` | How fast does it serve? | metrics + served-request audit | minutes per cell |
| Sanitizer harvest | `sanitizer` | Do the JIT kernels pass Waitcheck / ConSan? | `sanitizer_report.json` | seconds to harvest |

The three probe routes need no workload class and nothing that has to be kept in
step with the engine's output format — that is deliberate, and it is what makes
them a demonstration of AORTA triaging an engine it knows nothing about.
`host_launch.sh` is the only entry point AORTA sees on those routes, and
`TS_ENTRY` selects which in-container script it runs.

### Kernel probe

Drives TokenSpeed's kernels through TokenSpeed's own
`tokenspeed_kernel.benchmark` / `.numerics` harnesses. No weights, no server, no
readiness polling — the cheap path, and the one to reach for first.

The matrix is **solution × ROCm knob**: rows pin a `gemm.mm` solution (Gluon vs
torch) via the sidecar's `TS_KERNEL_NAME`, columns apply a launch knob
(`hip_launch_blocking`).

The verdict deliberately does not come from the exit code.
`tokenspeed_kernel.benchmark` ends in an unconditional `return 0` — it exits
successfully even when `--verify` finds a numerics mismatch. `ts_kernel_probe.sh`
re-reads the exported JSON and fails the trial itself (exit 32) when any record
has `numerics_passed == false`. (`numerics_passed: null` means *not checked*, not
failed — `torch_mm` reports null because it is the reference everything else is
compared against.)

### Suite probe

Runs TokenSpeed's own pytest operator suites. This is the only route that reaches
the non-GEMM families — attention, MoE, quantization, sampling, transform —
because those suites build their own inputs rather than depending on the
benchmark harness's input-generator registry.

The trade-off is that these suites *assert*; they do not measure. There are no
TFLOPs or latency figures here. What you get is a correctness verdict per
operator family, plus a populated JIT cache that the sanitizer route can turn
into a Waitcheck corpus covering attention and MoE rather than only gemm.

These suites skip heavily and pytest exits 0 when everything skips or is
deselected, which would turn a cell that proved nothing into a green one. So
`ts_pytest_probe.sh` counts executed tests from the JUnit report and fails the
trial when none ran (`ts_pytest_nothing_executed`, exit 41).

### Serving probe

Bring-up triage only: does the engine come up on this stack, and can it produce a
token. It cannot report a latency or a throughput.

`tokenspeed serve` never exits, so handing it directly to AORTA would always hit
`timeout_per_trial` and classify as `tier1:timeout` — an *error*, not a verdict.
`ts_serve_probe.sh` gives the probe something that terminates.

It is worth knowing that `tokenspeed serve` is an orchestrator, not one server:
it spawns an smg gateway (the OpenAI-compatible `/v1` surface, on `--port`) in
front of a gRPC engine, plus a TokenSpeed control server owning `/health` and
`/health_generate` on `--port + 1`. The script passes `--control-port`
explicitly, checks readiness on the control port, and sends the completion to the
**gateway** port — the path a real client takes, so a gateway that is up but not
wired to the engine still fails.

### `tokenspeed_serve` workload

The metrics route. Per trial: one containerised server and N measured bench runs
against it.

1. Stage `ts_bench_serve.sh` into a node-local `work_dir` and syntax-check it.
2. `docker run` the TokenSpeed image, forwarding the cell's mitigation
   environment.
3. In-container: start `tokenspeed serve`, poll `/health` then
   `/health_generate`, run `tokenspeed bench serve` `warmup_steps` times
   (discarded) and `steps` times (measured), tear the server down.
4. On the host: parse each exported JSON, average the scalars across steps, audit
   the served-request counts, apply any perf gates, return a verdict.

The load generator is TokenSpeed's own `tokenspeed bench serve` — the harness AMD
publishes its numbers with — so the numbers stay comparable to upstream's rather
than drifting from them.

`steps` is the number of **measured bench repetitions against one server**, not
the number of servers started. Weight load costs minutes and a bench step costs
about a second, so re-serving per step would mostly measure model load.

### Sanitizer harvest

TokenSpeed's kernels are Gluon/Triton, so they do not exist as committed
binaries — they are JIT-compiled into the Triton cache on first use. AORTA's
sanitizer pipeline consumes code objects by path plus a SHA-256 identity
(`source.kind: kernel_list`). `harvest_code_objects.py` bridges the two: run the
kernel once with a clean cache, collect the `.hsaco`, inventory it with
`rj_waitcheck --list-kernels`, and write a ready-to-run recipe. No sanitizer
changes were needed for any of this.

The harvested objects and generated recipes are a **run area, not something to
commit** — a harvested object is specific to the image, the GPU target and the
shapes that were compiled. Re-harvest instead; it takes seconds.

## Committed recipes

Ten files in `recipes/tokenspeed/`: eight recipes and two mitigation sidecars.

### Recipes

| File | Route | What it runs |
|---|---|---|
| `tokenspeed-kernel-gemm-smoke.yaml` | kernel probe | 2 `gemm.mm` solutions (Gluon, torch; both bf16) × 2 launch-knob columns, `trials: 3`. Deliberately a smoke, not a survey. |
| `tokenspeed-kernel-suites-smoke.yaml` | suite probe | 8 suites × 2 diagnostic columns, `trials: 1`. |
| `tokenspeed-serve-probe-smoke.yaml` | serving probe | Qwen3-0.6B bring-up, 2 mitigation cells (`none`, `hsa_no_scratch_reclaim`), `timeout_per_trial: 1800`. |
| `tokenspeed-serve-bench-smoke.yaml` | `tokenspeed_serve` | Qwen3-0.6B, 2 mitigation cells. The shape check to run first. |
| `tokenspeed-serve-models.yaml` | `tokenspeed_serve` | Qwen3 0.6B / 1.7B / 4B / 8B, identical load. |
| `tokenspeed-serve-load.yaml` | `tokenspeed_serve` | `max_concurrency` 1 / 8 / 32 / 64, plus prefill-heavy (ISL 4096) and decode-heavy (OSL 1024) shapes. |
| `tokenspeed-serve-gptoss.yaml` | `tokenspeed_serve` | `openai/gpt-oss-20b`, 2 mitigation cells. TokenSpeed's canonical AMD benchmark model. |
| `tokenspeed-serve-gptoss-tp.yaml` | `tokenspeed_serve` | `openai/gpt-oss-20b` at `--tensor-parallel-size` 1 / 2, GPUs pinned per cell. |

### Mitigation sidecars

These are not recipes. Because `mode: probe` runs one argv across every cell,
per-cell selection has to travel as environment, so a sidecar "mitigation" here
is really a selector.

| File | Carries | Used by the committed recipe |
|---|---|---|
| `tokenspeed-kernel-sidecar.json` | 5 `gemm.mm` kernel selectors (`ts_gemm_gluon_bf16`, `ts_gemm_torch_bf16`, `ts_gemm_triton_fp8_blockscale`, `ts_gemm_torch_fp8_blockscale`, `ts_gemm_all_bf16`) | 2 of 5 |
| `tokenspeed-pytest-sidecar.json` | 12 suite selectors, including two whole-directory sweeps (`ts_suite_amd_ops_all`, `ts_suite_upstream_ops_all`) | 8 of 12 |

Widening either matrix is a one-line edit to the recipe's `mitigation_axis` — the
selectors are already committed.

### On digest pinning

**The five `tokenspeed_serve` recipes are digest-pinned.** Each carries

```yaml
image: lightseekorg/tokenspeed-amd@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78
```

which is also the workload class's `_DEFAULT_IMAGE`. The recipes say why: a
registry can retarget a date tag, so `:nightly-20260714` would let the image
change underneath a blessed baseline while the recipe still read as pinned.

**The three probe recipes are not, and structurally cannot be.** On `mode: probe`
the image is chosen by the `TS_IMAGE` environment variable, outside the recipe;
the committed run instructions set it to the tag
`lightseekorg/tokenspeed-amd:nightly-20260714`. Provenance is recovered
after the fact instead: `host_launch.sh` prints `host_launch: image_digest=...`
on every trial, so each run records the content it actually used even when it was
invoked by tag. If you need a probe run to be exactly reproducible, export
`TS_IMAGE` as the digest yourself.

That digest is the content every measured number in this document was taken
against.

## Code surface

### Entry point

One entry in `pyproject.toml`'s `aorta.workloads` group:

```toml
tokenspeed_serve = "aorta.workloads.tokenspeed_serve:TokenSpeedServeWorkload"
```

The probe routes register nothing — they run on the platform-internal
`_subprocess` workload that was already there.

`src/aorta/workloads/tokenspeed_serve.py` is 2943 lines.

### Scripts in `src/aorta/workloads/tokenspeed/`

`README.md` in that directory is the per-file reference. Exit codes are a
documented interface: the recipes' `custom_patterns` and both test suites depend
on them, and `ts_bench_serve.sh` uses the 50–55 band precisely so a triage log
never leaves it ambiguous whether a verdict came from it or from
`ts_serve_probe.sh` (20–23).

| File | Runs on | Purpose |
|---|---|---|
| `host_launch.sh` | host | The opaque command `aorta sweep run` wraps. Turns the cell's `AORTA_ENV_FILE` into `docker run --env-file` and mints the per-trial `TS_RUN_TOKEN`. `TS_ENTRY` picks the in-container script. |
| `stage_scripts.sh` | host | Mirrors the directory to a node-local path the docker daemon can read, and syntax-checks it. |
| `harvest_code_objects.py` | host | Runs a kernel — via the benchmark harness or `--pytest-suite` — with a clean Triton cache, collects the JIT-compiled `.hsaco`, and emits a `mode: sanitizer` recipe pinning each object by SHA-256. `--consan` additionally emits one loader shim and one single-kernel ConSan recipe per object. |
| `ts_kernel_probe.sh` | container | Kernel numerics and/or benchmark; re-reads the exported JSON to reach its own verdict. |
| `ts_pytest_probe.sh` | container | Runs one of TokenSpeed's own operator suites; requires at least one executed test. |
| `ts_serve_probe.sh` | container | Serving bring-up: start, poll readiness on the control port, one completion against the gateway port, tear the process group down. |
| `ts_bench_serve.sh` | container | Serving benchmark. Launched by the `tokenspeed_serve` workload class, not by `host_launch.sh`. |
| `list_harness_coverage.py` | container | Surveys which TokenSpeed operators its own numerics/benchmark harness can actually drive. Substantiates the "only `gemm.mm`" constraint. |
| `map_kernel_test_coverage.py` | container | Surveys which registered kernels TokenSpeed's own suites actually exercise, by instrumenting registry lookups while pytest runs. Substantiates the coverage table. |

The two survey tools read TokenSpeed private attributes (`_by_name`,
`_INPUT_GENERATORS`, `_STANDARD_SHAPES`) because the registries expose no public
enumeration. They fail loudly rather than reporting nothing if those move, and
they exist so the coverage claims can be re-checked against a newer image instead
of being trusted indefinitely.

## How to run it

### One-time setup

**Prerequisites.** gfx950. TokenSpeed's platform check accepts gfx950 and
gfx1250 and raises on anything else, so MI300X (gfx942) cannot run any of this.
Everything measured here is gfx950 (MI355X), ROCm 7.0.2.2. Docker is required;
`tokenspeed_serve` fails in `setup()` if `docker` is not on `PATH` or if
`/dev/kfd` is not readable and writable.

**1. Allocate a node. Do not run on the head node.**

```bash
salloc --no-shell -p interactive -N1 -t 12:00:00 -J tokenspeed-aorta
srun --jobid=<id> --overlap bash -lc '...'
```

**2. Work around a stale Docker config.** If `~/.docker/config.json` holds a
Docker Hub PAT that no longer authenticates, docker uses it instead of falling
back to anonymous, and even `docker pull hello-world` fails with `authentication
required`. Override it rather than editing the file:

```bash
mkdir -p /tmp/dockercfg-anon && echo '{}' > /tmp/dockercfg-anon/config.json
DOCKER_CONFIG=/tmp/dockercfg-anon \
  docker pull lightseekorg/tokenspeed-amd:nightly-20260714
```

The documented `:latest` tag does not exist; `nightly-20260714` and `tml` are
what is published. To reproduce the numbers in this document exactly, pull by
digest:

```
lightseekorg/tokenspeed-amd@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78
```

**3. Keep everything node-local.** The docker daemon runs as root, so on a
root-squashed NFS export `-v /home/...` fails with `mkdir /home/<user>:
permission denied`. Scripts, the HF cache and probe output all have to live
node-locally — `/tmp/ts-work/...`. `host_launch.sh`, `stage_scripts.sh` and
`harvest_code_objects.py` each resolve the path against `/proc/mounts` and refuse
any network fstype up front. Note `/tmp` is per-node: an `rm -rf /tmp/...` on the
login node does not clear the compute node's copy.

**4. Provision a Python that travels with its venv,** if the compute nodes
predate AORTA's floor of 3.10. A venv built on the login node also breaks,
because its `bin/python` resolves to the node's interpreter and then cannot find
its own `site-packages`.

```bash
uv python install 3.11
uv venv --python 3.11 ~/ts-aorta/.venv311
uv pip install --python ~/ts-aorta/.venv311/bin/python -e path/to/aorta
```

**5. Stage the scripts node-locally, on the compute node.**

```bash
srun --jobid=$JOBID --overlap \
  bash src/aorta/workloads/tokenspeed/stage_scripts.sh
```

Re-run after **any** edit. A stale staging directory fails every cell in a matrix
identically, or — worse — passes while silently running the previous version of
the script. Staging from the login node writes a directory the trials never see.

Steps 1–5 are needed for the three probe routes and the sanitizer route. The
`tokenspeed_serve` workload stages `ts_bench_serve.sh` itself from the installed
package, so it needs steps 1–4 only.

### Kernel probe

```bash
export TS_IMAGE=lightseekorg/tokenspeed-amd:nightly-20260714
export TS_SCRIPTS_DIR=/tmp/ts-work/scripts
export TS_ENTRY=ts_kernel_probe.sh TS_GPUS=1

aorta sweep run \
  --recipe recipes/tokenspeed/tokenspeed-kernel-gemm-smoke.yaml \
  --mitigations-file recipes/tokenspeed/tokenspeed-kernel-sidecar.json \
  --output-dir /tmp/ts-work/kernel-out --ticket TOKENSPEED-KERNEL \
  -- bash /tmp/ts-work/scripts/host_launch.sh
```

`env_passthrough_mode: file` is set in the recipe and is mandatory: `docker run`
does not inherit `os.environ`, so `inherit` would make every cell silently run
the same default kernel.

### Suite probe

```bash
export TS_IMAGE=lightseekorg/tokenspeed-amd:nightly-20260714
export TS_SCRIPTS_DIR=/tmp/ts-work/scripts
export TS_ENTRY=ts_pytest_probe.sh TS_GPUS=0

aorta sweep run \
  --recipe recipes/tokenspeed/tokenspeed-kernel-suites-smoke.yaml \
  --mitigations-file recipes/tokenspeed/tokenspeed-pytest-sidecar.json \
  --output-dir /tmp/ts-work/suites-out --ticket TOKENSPEED-SUITES \
  -- bash /tmp/ts-work/scripts/host_launch.sh
```

Expert-parallel (`_ep_`) solutions skip when too few GPUs are visible, so run with
the full node exposed before concluding a kernel has no test.

### Serving probe

```bash
export TS_IMAGE=lightseekorg/tokenspeed-amd:nightly-20260714
export TS_SCRIPTS_DIR=/tmp/ts-work/scripts
export TS_MODEL=Qwen/Qwen3-0.6B TS_GPUS=0

aorta sweep run \
  --recipe recipes/tokenspeed/tokenspeed-serve-probe-smoke.yaml \
  --output-dir /tmp/ts-work/probe-out --ticket TOKENSPEED-PHASE0 \
  -- bash /tmp/ts-work/scripts/host_launch.sh
```

### Serving benchmark

No `TS_*` exports and no staging step — the recipe carries its own configuration
and the workload stages its script.

```bash
# validate only -- no GPU, no container:
aorta sweep run --recipe recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml --dry-run

# real run (gfx950/gfx1250 + docker):
aorta sweep run --recipe recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml \
  --output-dir /tmp/ts-serve-smoke
```

The other four serving recipes run the same way. For `gpt-oss-20b`, pre-warm the
HF cache first: the **full** snapshot is required (~40 GB, and excluding
`original/` fails during weight loading rather than at download time), and it
must be warmed **as the uid the trial will run as**, because `run_as_current_user`
defaults to true and a cache populated by a root container leaves the run failing
with `PermissionError` on the snapshot directory. The repo is ungated, so no HF
token is needed.

### Sanitizers

First provision the RocJITsu binaries (see
`src/aorta/instrumentation/rocjitsu_sanitizers/README.md`; needs a GitHub token
with `actions:read`):

```bash
python3 scripts/sanitizers/download_sanitizer_artifacts.py --dest ./rocjitsu-sanitizers
export ROCJITSU_PREBUILT="$PWD/rocjitsu-sanitizers"
```

`ROCJITSU_PREBUILT` has to be exported in the shell that runs `aorta sweep run`,
not only the one that harvests. Without it the run completes with
`overall verdict: not_checked` and every kernel `not_checked`, which reads as a
sanitizer failure rather than a missing binary.

Harvest gemm and run Waitcheck:

```bash
python3 src/aorta/workloads/tokenspeed/harvest_code_objects.py \
  --image lightseekorg/tokenspeed-amd:nightly-20260714 \
  --kernel gluon_mm_a16w16_gfx950 --dtype bf16 --dtype-role a \
  --dest /tmp/ts-work/sanitizer-run --gpus 2 \
  --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck" \
  --docker-config /tmp/dockercfg-anon

aorta sweep run \
  --recipe /tmp/ts-work/sanitizer-run/waitcheck-gluon_mm_a16w16_gfx950.yaml \
  --output-dir /tmp/ts-work/sanitizer-out
```

`--kernel` / `--op` drive the benchmark harness, so they can only ever harvest
`gemm.mm`. `--pytest-suite` drives one of TokenSpeed's own suites instead, which
is the only way to compile the attention, MoE, quantization, sampling and
transform kernels — and therefore the only way to get them under Waitcheck:

```bash
python3 src/aorta/workloads/tokenspeed/harvest_code_objects.py \
  --image lightseekorg/tokenspeed-amd:nightly-20260714 \
  --pytest-suite tokenspeed-kernel/test/ops/test_attention.py \
  --pytest-k mha_prefill \
  --dest /tmp/ts-work/attn-run --gpus 5 \
  --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck" \
  --docker-config /tmp/dockercfg-anon

aorta sweep run \
  --recipe /tmp/ts-work/attn-run/waitcheck-test_attention.yaml \
  --output-dir /tmp/ts-work/attn-sanitizer-out
```

The `--pytest-k mha_prefill` filter is load-bearing — see [Known limits](#known-limits).

For ConSan, add `--consan` and run one recipe per code object (ConSan takes
exactly one object per run, and a TokenSpeed kernel compiles to several
shape-specialized objects):

```bash
python3 src/aorta/workloads/tokenspeed/harvest_code_objects.py \
    --image lightseekorg/tokenspeed-amd:nightly-20260714 \
    --kernel gluon_mm_a16w16_gfx950 --dtype bf16 --dtype-role a \
    --dest /tmp/ts-work/consan-gemm \
    --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck" \
    --consan

for recipe in /tmp/ts-work/consan-gemm/consan/consan-*.yaml; do
    aorta sweep run --recipe "$recipe" \
        --output-dir "/tmp/ts-work/consan-out/$(basename "$recipe" .yaml)"
done
```

`--consan-limit N` caps the fan-out, which matters because an attention harvest
yields 20 objects and each is a separate ConSan run.

## How to test it without a GPU

Both suites are GPU-free and Docker-free by design: `shutil.which`, the
`/dev/kfd` probe and the `subprocess` entry points are monkeypatched.

```bash
python -m pytest tests/probe/test_tokenspeed_probe.py tests/workloads/test_tokenspeed_serve.py tests/ci -q --timeout=300
```

Measured on `main` at `7f82258`, Python 3.10.12, pytest 9.1.1:
**741 passed in 350.01 s (5 m 50 s)**, comprising

| Suite | Tests |
|---|---|
| `tests/probe/test_tokenspeed_probe.py` | 202 |
| `tests/workloads/test_tokenspeed_serve.py` | 382 |
| `tests/ci/test_dashboard_and_alert.py` | 112 |
| `tests/ci/test_eval_lib.py` | 21 |
| `tests/ci/test_nightly_eval.py` | 24 |

**What they prove.** The probe suite covers script syntax, the guardrails (NFS
refusal, missing entry script, missing selector), input validation on the
documented settings, recipe and sidecar wellformedness, per-trial output naming,
that every recipe axis entry resolves through the real mitigation registry, and
that nothing in the recipe directory is gitignored. Two gates are covered against
stubs rather than real kernels, because both guard against a *silent* pass the
shipped code cannot be made to produce on demand: the **numerics gate** (against
a synthetic benchmark export) and the **nothing-executed gate** (against a stub
suite of only-skipped and deselected tests). The pytest-probe tests run the real
pytest against a temporary stub workspace.

The serving suite covers what the workload owns — config validation, env/argv
construction, export parsing, aggregation and the verdict — with the
served-request audit, the exit-code mapping and the gates getting the most
attention. Every committed recipe is loaded through the real recipe parser,
resolved against the real mitigation registry, and pushed through the workload's
own validation, so an out-of-range value fails on the CPU gate rather than on a
GPU node. An *unknown* config key is only a warning at runtime (a config carrying
a key some other tool reads should not be fatal), but the recipe test treats that
warning as fatal for the recipes in this repo — a typo in a committed recipe
fails the CPU gate.

**The `--dry-run` path** validates a recipe end to end without a GPU or a
container: it resolves the workload, the trial isolation policy, every cell's
merged `workload_config`, the mitigations and the confound threshold, and prints
them. All five `tokenspeed_serve` recipes were confirmed clean under `--dry-run`
on `main`. It proves the recipe is well-formed and resolvable; it does not
execute the workload's own `setup()` validation, so it is a weaker check than the
CPU test suite, not a stronger one.

**What neither covers:** nothing here exercises a GPU, a container, or the
TokenSpeed engine. Every measured number in this document came from a real gfx950
run and cannot be reproduced from the CPU suites.

## What comes out, and where to look

### Probe routes

`mode: probe` carries a verdict but no metrics. For the kernel probe, per-shape
numbers land on **stdout** as `TS_KERNEL_METRIC` lines (`median_latency_us`,
`p99_latency_us`, `tflops`, `bandwidth_gb_s`, `numerics_passed`). They are **not**
in `result.json`'s `metrics` dict — the probe path has no such channel.

The `Mean step (ms)` column in `matrix.md` is whole-container walltime (docker
start + JIT + benchmark, ~17 s), not kernel time; do not read it as a perf signal.

### `tokenspeed_serve`

- `perf.md` — per-cell step timing and the serving metric table.
- `matrix.md` — pass/fail per cell.
- `cells/<cell>/tokenspeed_serve/trial_*.json` — `result.metrics`, including
  `steps` (per-step detail), `result_files` and `server_log`.
- `matrix.json` — `metrics_summary` per cell.

Metric names are `tokenspeed bench serve`'s own, passed through verbatim,
averaged across the measured steps:

| Metric | Meaning |
|---|---|
| `median_ttft_ms`, `p50/p90/p99_ttft_ms`, `mean_ttft_ms`, `std_ttft_ms` | Time to first token |
| `median_tpot_ms`, `p*_tpot_ms` | Time per output token |
| `median_itl_ms`, `p*_itl_ms` | Inter-token latency |
| `median_e2el_ms`, `p*_e2el_ms` | End-to-end request latency |
| `output_throughput`, `tokens_per_sec` | Output tokens/sec |
| `total_token_throughput` | Input + output tokens/sec |
| `request_throughput` | Requests/sec |
| `max_output_tokens_per_s`, `max_concurrent_requests` | Peaks |
| `duration`, `total_input_tokens`, `total_output_tokens` | Work done per step |
| `completed_total`, `failed_total` | Served-request audit (sums, not means) |
| `server_startup_sec` | Bring-up time, from the script's marker |
| `container_elapsed_sec` | `docker run` wall clock, excluding the VRAM drain wait |

`tokens_per_sec` is an alias for `output_throughput`, present because AORTA's CI
gating allowlist already knew that name. TTFT is deliberately **not** aliased to
`prefill_latency_ms`: TTFT includes queueing delay, so gating it under that name
would gate a differently-defined quantity.

`elapsed_sec` covers the whole cell including the post-exit wait for VRAM to be
released; `container_elapsed_sec` is the `docker run` duration on its own.

A note on `median_itl_ms`: it is often near zero while `p99_itl_ms` is tens of
milliseconds. The gateway delivers several tokens per SSE chunk, so most recorded
inter-token gaps are ~0 and the real gaps show up in the tail. `median_tpot_ms`
is the better-behaved per-token metric.

### Sanitizer route

The CLI line is terse (`overall verdict: pass (execution complete)`). The verdict
worth reading is in `sanitizer_report.json` under `checks[].kernel_results[]`.
Check `state: ran` and a populated `code_object_sha256`, not just the overall
verdict: `not_checked` also renders as a non-failure, so "no findings" and "never
looked" are indistinguishable from the top-level line alone. `checks[].backend`
records the `rj_waitcheck` binary's own SHA-256, so a report identifies both sides
of the analysis. `coverage` is `[]` — do not expect per-instruction detail.

## Configuration surface (`tokenspeed_serve`)

The workload accepts 38 `workload_config` keys; the full list is in the
`TokenSpeedServeWorkload` docstring. `steps` is reserved — it is a top-level
recipe key, not a `workload_config` one. The load-bearing ones:

| Key | Default | Notes |
|---|---|---|
| `image` | the pinned digest above | |
| `model` | `Qwen/Qwen3-0.6B` | Any HF id the container can load. |
| `served_model_name` | `model` | The id both sides use. Reserved in `serve_args` so the halves cannot diverge. |
| `num_prompts` | `64` | Requests per bench step. Also the audit target. |
| `input_len` / `output_len` | `1024` / `128` | Random-dataset ISL/OSL. Not sent and reported as `null` for `sharegpt`. |
| `dataset` | `random` | `random` or `sharegpt`. |
| `dataset_path` | — | Host path to a ShareGPT JSON. Required for `sharegpt`, rejected for `random`. |
| `max_concurrency` | unbounded | In-flight request cap. |
| `request_rate` | `inf` | The quoted string `"inf"` submits everything at once; an unquoted infinite float is rejected. |
| `warmup_steps` | `1` | Discarded bench steps. |
| `num_warmups` | `1` | Warmup requests *within* a bench step. |
| `ignore_eos` | `true` | Holds OSL fixed so cells do equal work. `false` is only accepted for `sharegpt`. |
| `seed` | `0` | |
| `percentile_metrics` | `ttft,tpot,itl,e2el` | |
| `metric_percentiles` | `50,90,99` | |
| `work_dir` | `/tmp/ts-work-serve` | Must be node-local. Scratch and the HF cache are per-uid beneath it, at `<work_dir>/u<uid>`. |
| `hf_home` | `<work_dir>/u<uid>/hf` | Set it to share one pre-populated cache between users. |
| `hf_token_env` | `HF_TOKEN` | The *name* of a host env var; the value never appears in a recipe or in argv. |
| `hf_offline` | `false` | Serve strictly from a pre-populated cache on a node with no egress. |
| `hip_visible_devices` | unset | Which GPUs the container sees. A visibility filter, not an allocation. |
| `exclusive_gpus` | `false` | Assert that no other job shares this node's GPUs. Only then does unreleased VRAM fail the trial. |
| `ready_timeout_sec` | `900` | Raise it for big models on a cold cache. (`gpt-oss` recipes use 1800 and 2400.) |
| `bench_timeout_sec` | `1800` | |
| `teardown_grace_sec` | `45` | Must be 5–3600, and must strictly exceed any explicit `--drain-timeout`. |
| `network` | `host` | A bridged container on a node with IPv4 forwarding disabled cannot reach the HF Hub. |
| `shm_size` | `16g` | Effective, because the IPC namespace is private. |
| `port` / `control_port` | `auto` | Free ports, picked per trial. Explicit values must be in 1024..65535 and must differ. |
| `serve_args` / `bench_args` | — | Extra flags. May not shadow the flags the workload owns. |
| `docker_args` | — | Extra docker flags. May not displace a generated option. |
| `run_as_current_user` | `true` | Keeps the HF cache and exported JSON deletable by the caller. |
| `keep_work_dir` | `true` | |
| `gates` | none | Optional per-trial perf gates. |

### Perf gates

`gates` enforces bounds per trial, so a recipe fails immediately when serving
degrades rather than waiting for a nightly comparison:

```yaml
workload_config:
  gates:
    max_median_ttft_ms: 500
    max_p99_tpot_ms: 10
    min_output_throughput: 1000
```

The accepted keys, verified against `_GATE_SPECS`, are exactly:
`max_median_ttft_ms`, `max_p99_ttft_ms`, `max_median_tpot_ms`, `max_p99_tpot_ms`,
`max_median_itl_ms`, `max_p99_itl_ms`, `max_median_e2el_ms`, `max_p99_e2el_ms`,
`min_output_throughput`, `min_total_token_throughput`, `min_request_throughput`.
Anything else is rejected at validation, and a gate naming a metric the bench did
not report fails the trial rather than being skipped — so a recipe cannot believe
it is gated when it is not.

### What could be gated in the nightly

`scripts/ci/eval_lib.py`'s `_METRIC_POLICIES` allowlist on `main` contains, for
serving: `median_ttft_ms`, `p99_ttft_ms`, `median_tpot_ms`, `p99_tpot_ms`,
`median_itl_ms`, `p99_itl_ms`, `median_e2el_ms`, `p99_e2el_ms` as `max`; and
`output_throughput`, `total_token_throughput`, `request_throughput` as `min`.
`tokens_per_sec` was already there as `min` (which is why the alias exists).
Unlisted metrics are recorded for trends but never gated.

**Nothing is gated today.** No serving recipe is in the nightly matrix, and that
matrix lives in `aorta-internal`.

## Measured results

All figures below are reproduced from the docs on `main`. Environment: one gfx950
(MI355X), ROCm 7.0.2.2, image
`lightseekorg/tokenspeed-amd@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78`.

These are what the integration produced, not a performance claim about
TokenSpeed — one node, one seed, tiny request counts.

### Kernel probe

4 cells × 3 trials, all 12 passing in 3 m 30 s. Figures are the median across
three trials at the 4096³ shape; the spread across trials was under 1.5 µs in
every cell.

| Solution | Diagnostic | p50 @ 4096³ | TFLOPs |
|---|---|---|---|
| `gluon_mm_a16w16_gfx950` | none | 107.2 µs | 1282 |
| `gluon_mm_a16w16_gfx950` | `hip_launch_blocking` | 139.6 µs | 984 |
| `torch_mm` | none | 106.9 µs | 1286 |
| `torch_mm` | `hip_launch_blocking` | 131.3 µs | 1047 |

`hip_launch_blocking` costs ~30% throughput, which is the expected shape of
serializing every launch — the axis behaving correctly, not a regression. It does
mean the two columns are not comparable as perf numbers, only within a column.

Each trial covers 5 shapes (M ∈ {1, 16, 128, 512, 4096} at N=K=4096). Small-M
shapes are bandwidth-bound (~1.6–1.8 TB/s at M ≤ 16) and large-M shapes
compute-bound.

### Suite probe

16 cells (8 suites × 2 diagnostics), all passing, 3 m 48 s of summed per-cell wall
clock. Counts were identical in both columns, so only the `none` column is shown;
the walltime is in-container pytest time and excludes container start and JIT.

| Suite | Passed | Skipped | Walltime |
|---|---|---|---|
| `test_attention.py` | 46 | 84 | 19 s |
| `test_attention_dsa.py` | 13 | 0 | 11 s |
| `test_attention_gdn.py` | 4 | 2 | 10 s |
| `test_attention_mla.py` | 6 | 0 | 8 s |
| `test_moe_gluon_bf16_gfx950.py` | 9 | 0 | 7 s |
| `test_quantization.py` | 17 | 9 | 5 s |
| `test_sampling_gluon_gfx950.py` | 20 | 0 | 5 s |
| `test_transform.py` | 1 | 1 | 5 s |

`hip_launch_blocking` changed no verdict and no count, which is the expected
result for assertion suites. It is kept as a column so that when a suite does go
red, the two cells immediately separate "the kernel is wrong" from "the
pipelining around it is wrong".

### Kernel coverage

Two independent surveys, both on `nightly-20260714`.

`list_harness_coverage.py`: 21 operator families over **38 registered kernel
names**, of which **1 family — `gemm.mm` (9 kernels) — can actually be run**
through the benchmark harness. (38 is the count of distinct names in
`KernelRegistry._by_name`. Summing the per-family lists gives 40 instead, because
a name registered under two operator keys is listed under both.)

`map_kernel_test_coverage.py`, over all 38 registered kernels:

| Status | Kernels | Detail |
|---|---|---|
| **Entered by the suites** | **20** | attention 15, quantization 3, transform 1, `moe.apply` 1 |
| Looked up but never entered | 0 | none on this image |
| Candidate-only | 3 | `gluon_*_moe_apply` variants — the operator is reached, this implementation is not selected |
| Reached only by the benchmark harness | 9 | `gemm.mm` (of which `cublaslt_mm_nvfp4` is a cuBLASLt path that cannot run here at all) |
| Reached only by a direct-import suite | 1 | `gluon_argmax_gfx950` |
| No executing test anywhere | 5 | `triton_*_moe_apply` variants |

Read `covered` as "this implementation ran", not "asserted correct". Two blind
spots make this an under-count rather than an over-count:
`tokenspeed-kernel-amd/test/ops` imports implementations directly and never
consults the registry, and `_ep_` solutions skip when too few GPUs are visible.

### Serving probe

Both cells pass, `failure_detectors_fired: []`, ~10–11 min for the 2-cell matrix.
Teardown exited cleanly on SIGTERM every time, with no escalation to SIGKILL.

Startup to `/health` is the dominant cost and it is **noisy**: 189, 276, 285, 291,
316 and 319 seconds across six runs of the same recipe on the same node — a 1.7×
spread with nothing changed between them. Do not treat a slow bring-up as a signal
without repeats.

That this is minutes for a *0.6B* model is the surprising part, and it is not
weight loading: it is dominated by allocating the default 250 GB KV cache and a
537 GB pinned host tier, both roughly independent of model size.

### Serving: across models

`tokenspeed-serve-models.yaml`, 32 requests per step, ISL 512 / OSL 128,
concurrency 8, `trials: 1`, `steps: 3`, `warmup_steps: 1`, `ignore_eos: true`. All
four cells passed with all 384 requests served and none failed. Numbers are means
across the three measured steps.

| Model | Startup (s) | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s | Total tok/s |
|---|---|---|---|---|---|
| Qwen3-0.6B | 379 | 46.3 | 1.94 | 3538 | 17688 |
| Qwen3-1.7B | 283 | 53.3 | 2.04 | 3269 | 16346 |
| Qwen3-4B | 289 | 56.9 | 3.77 | 1905 | 9527 |
| Qwen3-8B | 328 | 61.1 | 4.46 | 1625 | 8124 |

Read the startup column as a floor, not a measurement: it is dominated by weight
loading and Triton compilation against whatever the node's caches already hold,
which is why the smallest model posts the largest number.

### Serving: gpt-oss-20b

`tokenspeed-serve-gptoss.yaml`. 21B parameters with MXFP4-packed MoE experts,
served from a single MI355X. Both cells passed, 96/96 requests served per cell,
none failed. Same load as the model sweep.

| Cell | Startup (s) | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|---|
| baseline | 316 | 67.1 | 7.61 | 994 |
| `hsa_no_scratch_reclaim` | 286 | 66.7 | 7.49 | 1008 |

Against Qwen3-0.6B on the same node and load: TTFT ~1.45×, TPOT ~3.9×, output
throughput ~0.28×. A reasonable shape for a 21B MoE against a 0.6B dense model —
decode is where the cost lands. The mitigation is within noise of baseline, as it
was at every smaller size.

### Serving: tensor parallelism

`tokenspeed-serve-gptoss-tp.yaml`. Both cells passed, 96/96 requests each, none
failed. Same load as above.

| Cell | Startup (s) | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|---|
| `tp1` | 199 | 67.2 | 7.63 | 991 |
| `tp2` | 282 | 65.6 | 7.23 | 1036 |

TP=1 reproduces the single-GPU numbers to within a percent, which is the control
this axis needs. TP=2 buys about 4.5% throughput for a second GPU — a poor trade,
and an unsurprising one: at 21B with MXFP4 the model already fits in one MI355X.
The result here is that the path works and reports coherently, not that it is
fast.

### Serving: across load shapes

`tokenspeed-serve-load.yaml`, Qwen3-0.6B, as committed: 16/32/128/256/64/64
prompts per step over six cells, three measured steps each, so 1680 requests. All
six cells passed and all 1680 were served; none failed.

| Cell | Concurrency | Prompts/step | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Output tok/s | Requests/s |
|---|---|---|---|---|---|---|---|
| conc-1 | 1 | 16 | 47.6 | 210 | 1.50 | 537 | 4.2 |
| conc-8 | 8 | 32 | 45.9 | 194 | 1.91 | 3631 | 28.4 |
| conc-32 | 32 | 128 | 60.5 | 315 | 2.01 | 12770 | 99.8 |
| conc-64 | 64 | 256 | 77.4 | 426 | 2.49 | 20125 | 157.2 |
| isl-4096-osl-128 | 16 | 64 | 64.0 | 440 | 2.95 | 4639 | 36.2 |
| isl-128-osl-1024 | 16 | 64 | 47.4 | 2139 | 2.05 | 7634 | 7.5 |

The concurrency rows show a curve that is bending but has not flattened:
throughput returns 6.8× for the first 8× of cap (537 → 3631 tok/s), 3.5× for the
next 4× (→ 12770), and 1.58× for the last 2× (→ 20125). Per unit of cap that is
0.85, 0.88, then 0.79 — so 64 is past the efficient point without being the
ceiling, and p99 TTFT has risen 35% (315 → 426 ms) to buy that last 58%.

The two shape rows separate the regimes. Prefill-heavy (ISL 4096) raises TPOT to
2.95 ms; decode-heavy (OSL 1024) leaves TPOT alone (2.05 ms) but pushes p99 TTFT
to 2.1 s, because a short request queued behind long decodes waits for them.

`num_prompts` rises with the cap on purpose, and the table is only readable
because it does. Keep the two in step when adding cells, and re-measure the whole
table when you change either — the rows are not comparable across configurations.

### Sanitizer harvest

**Waitcheck, gemm.** One `gluon_mm_a16w16_gfx950` benchmark emits two
shape-specialized objects, and both pass Waitcheck.

**Waitcheck, attention.** `--pytest-suite tokenspeed-kernel/test/ops/test_attention.py
--pytest-k mha_prefill` harvests **20 code objects and all 20 pass Waitcheck**
(`state: ran`, `verdict: pass`, no findings), covering `_mha_prefill` ×6,
`_mha_prefill_sliding` ×4 and `_fwd_kernel` ×10.

Dropping the `--pytest-k` filter harvests 61 objects, and 24 of them — every
`_mha_decode` and `_mha_extend` specialization — fail with `waitcheck analysis
failed ... decode failed while building CFG: Invalid instruction opcode:
DDF48000`. That is this Waitcheck build's instruction decoder reaching an opcode
it does not know, not a finding about the kernel.

**ConSan, gemm.** Works, with full static instrumentation:

```
pass | consan ran/pass | access 208/208 | analysis_complete=True | _mfma_lds_largem_kernel
pass | consan ran/pass | access  77/77  | analysis_complete=True | _mfma_lds_mediumm_kernel
```

**ConSan, attention.** Does not work. All 20 harvested objects discover their
sites, report them supported, and then fail to lower every one of them — **2502
sites in total across the 20 objects**, `access_patched=0`,
`lowering_reason=instrumentation_patch_missing`. Per-object counts vary; only the
total is a reliable figure. The one object measured in detail would go from
`0/232` to `232/232` if the upstream defect were fixed.

Re-harvesting the same kernel on the same image and node reproduces byte-identical
digests, which is what makes a report comparable across runs.

## Known limits

Each of these is documented on `main` and a colleague will hit them.

**Only `gemm.mm` is drivable through the benchmark harness.** The harness needs
three registries to line up — a kernel, an input generator, and a shape list —
and only `gemm.mm` has all three. `--op attention.mha_prefill` dies with
`KeyError: No standard shapes registered for attention.mha_prefill. Known:
gemm.mm` before any kernel launches. This is an upstream TokenSpeed gap;
the maintainer's answer is in
[tokenspeed#1244](https://github.com/lightseekorg/tokenspeed/issues/1244) — "we
are missing a lot of input generator implementations because most numerical
correctness testing is done via unit tests", and the right fix is generalising
each generator, not renaming it. The recipes name the gap as its own detector
(`ts_kernel_no_input_generator`, `ts_kernel_no_standard_shapes`), so pointing one
at an unsupported operator is self-diagnosing.

This is **not** the binding constraint on kernel coverage — the suite probe reaches
attention, MoE, quantization, sampling and transform. Use the benchmark harness
when you want numbers and the suites when you want coverage.

**ConSan cannot instrument the attention kernels.** Waitcheck passes on all 20,
so these kernels are reachable; ConSan simply cannot instrument them yet. Filed
as [ROCm/aorta#405](https://github.com/ROCm/aorta/issues/405) here and as
[rocm-systems#10955](https://github.com/ROCm/rocm-systems/issues/10955) upstream,
which is where the defect is.

**No dynamic race evidence is available anywhere.** The ConSan loader runs in
`load` mode: it loads and instruments the object but never dispatches it, so
there is no dispatch packet and no records. Every ConSan lane here is
static-coverage only. This is why the recipes default to `consan_policy:
lenient` — `strict` sets `RJ_CONSAN_MOI_REQUIRE_RECORDS` and fails closed with
`combined_hook_exit_86` however healthy the run was. Tracked as
[rocm-systems#10966](https://github.com/ROCm/rocm-systems/issues/10966), which is
a **separate** gap from the lowering defect above: even a fully patched object
records nothing under this flow. Getting dynamic evidence is not a policy change
— it needs the RocJITsu entry-point allowlist that
`src/aorta/instrumentation/rocjitsu_sanitizers/README.md` already tracks as
unavailable.

**`peak_vram_mib` is `null` on the probe path.** The work happens inside a
container under a PID AORTA is not watching.

**`env.json` is host-scoped on the probe path,** and reports `torch not
importable`, `triton not importable`. ROCm and PyTorch live in the container, so
the environment AORTA records is not the environment the workload ran in.

**`ignore_eos: false` is rejected on the random dataset.** `tokenspeed bench
serve` overwrites the flag unconditionally for `dataset_name == "random"` on an
OpenAI-compatible backend, and this workload benches the gateway with `--backend
openai` — so there is no argv that turns EOS back on. Rather than let a trial
report `ignore_eos: false` while serving at a pinned length, the combination is
rejected during validation, on the host and again in `ts_bench_serve.sh`. The
route that does work is the request payload:

```yaml
bench_args: ["--extra-body", '{"ignore_eos": false}']
```

Expect cells to stop doing equal work once you do this. `dataset: sharegpt` is
unaffected — the rule is keyed on the dataset name, so `ignore_eos: false` is
honoured there.

**gfx950 only, in practice.** TokenSpeed's platform check accepts gfx950 and
gfx1250 and raises on anything else, so MI300X (gfx942) cannot run any of this.
Everything measured is gfx950; gfx1250 is accepted by the platform check but
untested here.

**TP=4 does not come up,** reproducibly on this image, with an out-of-memory
raised while `FlatMemoryExecutor` builds its host-side KV mirror
(`torch.AcceleratorError: CUDA error: out of memory`, from `flat_host_mirror.py`).
Not yet diagnosed further; the cell is left out of the recipe rather than shipped
red, because ranks that die this way hold their GPU memory long enough to poison
whatever runs next. Container shared memory is *not* ruled out — earlier runs that
appeared to rule it out predate the IPC fix and compared one host `/dev/shm` with
itself.

**A tensor-parallel teardown returns before the GPUs are free.** Measured after a
*passing* TP=2 cell: the container is gone, `docker ps -a` is empty, `rocm-smi
--showpids` reports no KFD processes — and one GPU still holds 256 GB of its 309
GB. Only rank 0's device is released promptly; the rest clears on its own after
roughly 30–45 s. The workload samples per-GPU VRAM before the run and waits for it
to come back afterwards, up to 5 minutes. Growth is only *attributed* — i.e. turned
into a failed trial — when `exclusive_gpus: true`, because on a shared node a
co-tenant produces the same delta. Ranks that die during *startup* hold their
memory past the 5-minute wait and need `rocm-smi --gpureset -d N`.

**`sharegpt` has never been run on hardware.** The plumbing is implemented and
tested; there are no numbers from it.

**Nothing is gated in the nightly.** The serving metric names are in the CI
allowlist and are gateable once baselines exist, but no serving recipe is in the
nightly matrix and that matrix lives in `aorta-internal`.

### Container-namespace traps, already handled

These bit during bring-up, are easy to reintroduce, and have regression tests.
Worth knowing so the symptoms are recognisable:

- **Per-trial filenames cannot use `$$`.** Each `docker run` gets a fresh PID
  namespace, so the entry script is always PID 1 and every trial would write the
  same filename. This was observed as a 12-trial matrix producing exactly one
  export file — and the matrix still passed. `host_launch.sh` mints
  `TS_RUN_TOKEN` host-side instead.
- **The container must not run as root,** or a root-owned Triton cache leaves the
  caller unable to delete its own run area. `--user` alone is not enough: Triton
  calls `getpass.getuser()`, so `USER` and `HOME` are passed too.
- **An unwritable JIT cache is reported as an unsupported GPU.** TokenSpeed
  raises `RuntimeError: Triton is not supported on the current platform` when the
  real error is a `PermissionError` from `os.makedirs` on the cache directory.
  That message sends you to the GPU, and the GPU is fine.
- **`tokenspeed bench serve` exits 0 when every request fails.** Both layers
  re-read the export and require `failed == 0` and `completed == num_prompts`.
- **The orchestrator's gateway budget is 60 s** by default, and a cold start
  exceeds it, producing a wall of 503s that reads like a broken engine. The
  workload defaults `--gateway-startup-timeout` to its own `ready_timeout_sec`.
- **The first bench step measures Triton compilation, not serving** (6.2 s vs 1.1 s
  on Qwen3-0.6B; TTFT inflated tenfold). `warmup_steps` (default 1) runs whole
  discarded bench steps; `num_warmups` does not fix this, because it warms
  requests *within* an invocation.

## Where to go for depth

Paths are relative to the repository root.

- [`docs/tokenspeed.md`](tokenspeed.md) — the three probe routes, the
  sanitizer path, the coverage surveys, and the full set of bring-up constraints.
- [`docs/tokenspeed-serving.md`](tokenspeed-serving.md) — the
  `tokenspeed_serve` workload: full configuration reference, every measured
  table, and a long "things that will bite you" section covering the failure
  modes the workload now handles by default.
- [`src/aorta/workloads/tokenspeed/README.md`](../src/aorta/workloads/tokenspeed/README.md)
  — the per-file script reference and the exit-code contract.

One stale sentence to be aware of while reading: `docs/tokenspeed.md`'s ConSan
section says [#408](https://github.com/ROCm/aorta/pull/408) "is on `main`, not on
this branch". That phrasing was written on the PR branch and survived the
squash-merge — #408 landed as `e61f455` and **is** on `main`, so a run from `main`
reports `consan_coverage_incomplete` with the real counts, not the old
`consan_output_parse_error`.

Further work exists on unmerged branches and is deliberately out of scope here.
