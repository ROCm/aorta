# TokenSpeed serving benchmarks under AORTA

The `tokenspeed_serve` workload measures how fast TokenSpeed *serves*: time to
first token, time per output token, inter-token latency, and throughput, per
recipe cell, aggregated into `perf.md` and `matrix.json`.

This is the metrics counterpart to the probe routes in
[TokenSpeed under AORTA](tokenspeed.md). Those routes answer correctness and
bring-up questions through an exit code; `mode: probe` has no metrics channel, so
a serving *number* cannot come out of them. Only a workload class can populate
`WorkloadResult.metrics`, which is what the sweep machinery aggregates.

| Route | Recipe mode | Answers | Verdict from |
|---|---|---|---|
| [Kernel probe](tokenspeed.md#kernel-probe) | `probe` | do the kernels compute the right bytes? | exit code |
| [Suite probe](tokenspeed.md#suite-probe) | `probe` | do the operator suites pass? | exit code |
| [Serving probe](tokenspeed.md#serving-probe) | `probe` | does a model come up and generate? | exit code |
| **`tokenspeed_serve`** | `triage` | **how fast does it serve?** | metrics + served-request audit |

## What it does

Per trial, one containerised server and N measured bench runs against it:

1. Stage `ts_bench_serve.sh` into a node-local `work_dir` and syntax-check it.
2. `docker run` the TokenSpeed image, forwarding the cell's mitigation
   environment.
3. Inside the container: start `tokenspeed serve`, poll `/health` then
   `/health_generate`, run `tokenspeed bench serve` `warmup_steps` times
   (discarded) and `steps` times (measured), tear the server down.
4. On the host: parse each exported JSON, average the scalars across steps,
   audit the served-request counts, apply any perf gates, return a verdict.

The load generator is TokenSpeed's own `tokenspeed bench serve` — the harness
AMD publishes its numbers with. This workload does not reimplement one, so the
numbers stay comparable to upstream's rather than drifting from them.

`steps` is the number of **measured bench repetitions against one server**, not
the number of servers started. Weight load costs minutes and a bench step costs
about a second, so re-serving per step would mostly measure model load.

## Quick run

```bash
# validate only -- no GPU, no container:
aorta sweep run --recipe recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml --dry-run

# real run (gfx950/gfx1250 + docker):
aorta sweep run --recipe recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml \
  --output-dir /tmp/ts-serve-smoke
```

Then read:

- `perf.md` — per-cell step timing and the serving metric table.
- `matrix.md` — pass/fail per cell.
- `cells/<cell>/tokenspeed_serve/trial_*.json` — `result.metrics`, including
  `steps` (per-step detail), `result_files` and `server_log`.

## Recipes

| Recipe | Shape |
|---|---|
| `tokenspeed-serve-bench-smoke.yaml` | Qwen3-0.6B, two mitigation cells. The shape check to run first. |
| `tokenspeed-serve-models.yaml` | Qwen3 0.6B / 1.7B / 4B / 8B, identical load. |
| `tokenspeed-serve-load.yaml` | Concurrency 1→64, plus prefill-heavy and decode-heavy shapes. |

In the multi-model and load recipes the cells differ by *workload*, not by
mitigation, so `matrix.md`'s confound ratio (a step-time comparison against the
baseline cell) is not meaningful — a bigger model or a heavier load is
legitimately slower. Read `perf.md` for those.

## Metrics emitted

Names are `tokenspeed bench serve`'s own, passed through verbatim, averaged
across the measured steps:

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

`tokens_per_sec` is an alias for `output_throughput`, present because AORTA's CI
gating allowlist already knew that name. TTFT is deliberately **not** aliased to
`prefill_latency_ms`: TTFT includes queueing delay, so gating it under that name
would gate a differently-defined quantity.

`step_times_ms` is the per-step bench `duration`, so `perf.md`'s step-timing
columns describe the bench runs.

A note on `median_itl_ms`: it is often near zero while `p99_itl_ms` is tens of
milliseconds. That is not a bug — the gateway delivers several tokens per SSE
chunk, so most recorded inter-token gaps are ~0 and the real gaps show up in the
tail. `median_tpot_ms` is the better-behaved per-token metric.

## Measured results

Run on one gfx950 (MI355X), ROCm 7.0.2.2, image
`lightseekorg/tokenspeed-amd@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78`
(the digest the recipes pin, resolved from `:nightly-20260714`), `trials: 1`,
`steps: 3`, `warmup_steps: 1`, `ignore_eos: true`, otherwise exactly the
committed recipes. Numbers are means across the three measured steps. These are
what the integration produced, not a performance claim about TokenSpeed — one
node, one seed, tiny request counts.

### Across models (`tokenspeed-serve-models.yaml`)

32 requests per step, ISL 512 / OSL 128, concurrency 8. All four cells passed
with all 384 requests served and none failed.

| Model | Startup (s) | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s | Total tok/s |
|---|---|---|---|---|---|
| Qwen3-0.6B | 379 | 46.3 | 1.94 | 3538 | 17688 |
| Qwen3-1.7B | 283 | 53.3 | 2.04 | 3269 | 16346 |
| Qwen3-4B | 289 | 56.9 | 3.77 | 1905 | 9527 |
| Qwen3-8B | 328 | 61.1 | 4.46 | 1625 | 8124 |

Per-token cost and latency rise with parameter count and throughput falls, which
is the expected shape — the point of the run is that all four sizes come up,
serve, and report coherently through the same code path.

Read the startup column as a floor, not a measurement: it is dominated by weight
loading and Triton compilation against whatever the node's caches already hold,
which is why the smallest model here posts the largest number. It is reported
because `ready_timeout_sec` has to cover it, not because it scales with anything.

### Across load shapes (`tokenspeed-serve-load.yaml`)

Qwen3-0.6B, as committed: 16/32/128/256/64/64 prompts per step over the six
cells, three measured steps each, so 1680 requests. All six cells passed and all
1680 were served; none failed.

| Cell | Concurrency | Prompts/step | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Output tok/s | Requests/s |
|---|---|---|---|---|---|---|---|
| conc-1 | 1 | 16 | 47.6 | 210 | 1.50 | 537 | 4.2 |
| conc-8 | 8 | 32 | 45.9 | 194 | 1.91 | 3631 | 28.4 |
| conc-32 | 32 | 128 | 60.5 | 315 | 2.01 | 12770 | 99.8 |
| conc-64 | 64 | 256 | 77.4 | 426 | 2.49 | 20125 | 157.2 |
| isl-4096-osl-128 | 16 | 64 | 64.0 | 440 | 2.95 | 4639 | 36.2 |
| isl-128-osl-1024 | 16 | 64 | 47.4 | 2139 | 2.05 | 7634 | 7.5 |

The concurrency rows are the useful part, and what they show is a curve that is
bending but has not yet flattened: throughput returns 6.8× for the first 8× of
cap (537 → 3631 tok/s), 3.5× for the next 4× (→ 12770), and 1.58× for the last
2× (→ 20125). Per unit of cap that is 0.85, 0.88, then 0.79 — so 64 is past the
efficient point without being the ceiling, and p99 TTFT has risen 35% (315 → 426
ms) to buy that last 58%. Finding where that trade stops being worth it, not the
peak number, is what the recipe exists for; on this node it would take cells
beyond 64 to bracket it.

The two shape rows separate the regimes. Prefill-heavy (ISL 4096) raises TPOT to
2.95 ms — attention over a long context — while decode-heavy (OSL 1024) leaves
TPOT alone (2.05 ms) but pushes p99 TTFT to 2.1 s, because a short request queued
behind long decodes waits for them.

`num_prompts` rises with the cap on purpose, and the table is only readable
because it does: 32 requests against a cap of 64 drain before the server reaches
steady state, so a flat request count across these cells would have the
high-concurrency rows describing the ramp rather than the throughput. Keep the
two in step when adding cells, and re-measure the whole table when you change
either — the rows are not comparable across configurations.

## Configuration

Full key list is in the `TokenSpeedServeWorkload` docstring. The ones that
matter most:

| Key | Default | Notes |
|---|---|---|
| `model` | `Qwen/Qwen3-0.6B` | Any HF id the container can load. |
| `num_prompts` | `64` | Requests per bench step. Also the audit target. |
| `input_len` / `output_len` | `1024` / `128` | Random-dataset ISL/OSL. |
| `max_concurrency` | unbounded | In-flight request cap. |
| `request_rate` | `inf` | `inf` submits everything at once. |
| `warmup_steps` | `1` | Discarded bench steps. See below. |
| `num_warmups` | `1` | Warmup requests *within* a bench step. |
| `ignore_eos` | `true` | Holds OSL fixed so cells do equal work. |
| `work_dir` | `/tmp/ts-work-serve` | Must be node-local. |
| `ready_timeout_sec` | `900` | Raise it for big models on a cold cache. |
| `network` | `host` | See below. |
| `port` / `control_port` | `auto` | Free ports, picked per trial. |
| `gates` | none | Optional per-trial perf gates. |

### Perf gates

AORTA gates metrics against blessed baselines in the *nightly*, after the fact.
For a recipe that should fail immediately when serving degrades, `gates` enforces
bounds per trial:

```yaml
workload_config:
  gates:
    max_median_ttft_ms: 500
    max_p99_tpot_ms: 10
    min_output_throughput: 1000
```

A gate naming a metric the bench did not report fails the trial rather than
being skipped, so a recipe cannot believe it is gated when it is not.

## Things that will bite you

Every item here is something that failed during bring-up on gfx950, and the
workload now handles by default. They are documented because the *symptoms* point
somewhere other than the cause.

### `tokenspeed bench serve` exits 0 when every request fails

`metrics.failed` is printed and written to the JSON, and never consulted for an
exit code. A run where the engine refused every request looks identical to a
clean one to any caller reading `$?` — and it still produces plausible TTFT and
throughput numbers, computed from whatever trickle succeeded.

Both layers therefore re-read the export and require `failed == 0` and
`completed == num_prompts`: the script fails the step (exit 55), and the workload
re-checks independently, so the guard survives someone editing the script's exit
codes. This is the same trap `tokenspeed_kernel.benchmark --verify` sets for
[the kernel probe](tokenspeed.md#the-verdict-does-not-come-from-the-exit-code).

Counting requests is not sufficient on its own, though. An export carrying only
`completed` and `failed` satisfies both checks, and since `gates` is empty by
default the cell would go green having measured no duration, no TTFT and no
throughput at all. Every measured step must therefore also carry finite values
for `duration` (greater than zero), `mean_ttft_ms`, `median_ttft_ms`,
`output_throughput`, `request_throughput` and `total_token_throughput`, plus
`mean_tpot_ms` / `median_tpot_ms` whenever `output_len > 1` — TPOT averages
inter-token gaps, of which a single-token response has none. A step missing any
of them is reported as `result_json_unusable` rather than as a pass. The
percentiles are deliberately not required: they depend on `percentile_metrics`
and `metric_percentiles`, so demanding them would fail a cell for a legitimate
recipe choice.

### Ports and timeouts are validated before anything computes with them

`ts_bench_serve.sh` already checked its counts, but the ports and timeouts reach
arithmetic and `seq` first, and each misbehaves somewhere other than where the
mistake is: `TS_PORT=abc` aborts inside `$(( PORT + 1 ))` with a bash arithmetic
error rather than the documented usage exit 64, `TS_PORT=65535` derives a control
port of 65536 that cannot be bound and so reads as a server that failed to start,
and a zero `TS_READY_TIMEOUT` leaves the readiness loop empty — reporting a
server that never became ready without having waited at all. All of them now exit
64 with the setting named, before the run area is created. The two ports must also
differ, since readiness is checked on the control port and the load is sent to the
gateway; sharing one would let a half-wired server be benchmarked.

The workload validates the same values on the host, so a recipe never reaches
these. They matter because the script is also meant to be runnable by hand, which
is what keeps the audit trustworthy independently of the Python layer.

### A mitigation cannot redefine the host/container protocol

Cell mitigations are forwarded into the container, which is the whole point of
the matrix — but a handful of variables are how this workload and
`ts_bench_serve.sh` agree on what the run is: `TS_NUM_PROMPTS`, `TS_BENCH_STEPS`,
`TS_RUN_TOKEN`, `TS_OUT_DIR`, the model and shape variables, and the ports. Only
the container would learn about an override, so `TS_NUM_PROMPTS=999` would have
the script request 999 requests while the host still audited against the recipe's
count — failing a healthy cell for a served-request shortfall — and a redefined
`TS_RUN_TOKEN` would have it write exports under a name the host's glob never
matches, failing the cell for finding no export. Both read as engine faults and
are neither, so a mitigation that sets one is rejected up front with the
`workload_config` field to use instead. Any other `TS_*` knob is forwarded
normally.

### The orchestrator's gateway budget is 60s, and a cold start exceeds it

`tokenspeed serve` starts the engine and an smg gateway, then waits for the
gateway to reach `/readiness` — `OrchestratorOpts.gateway_startup_timeout`,
default 60s. Downloading weights and JIT-compiling Gluon kernels takes minutes on
a fresh node, so the gateway's gRPC health check to the engine keeps timing out
and the orchestrator tears everything down.

What you see is a wall of 503s and `gRPC not reachable (tried sglang, vllm,
trtllm, mlx, tokenspeed)`, which reads like a broken engine rather than an
expired budget. The workload defaults `--gateway-startup-timeout` to its own
`ready_timeout_sec` so the outer deadline is the binding one and a slow start is
reported as a slow start.

### The first bench step measures Triton compilation, not serving

On Qwen3-0.6B the first bench invocation against a fresh server took 6.2s against
1.1s for every later one. Rolled into the metrics that one outlier dominates the
mean step time and inflates TTFT tenfold (465ms vs 47ms), so a cell looks like a
regression purely because it went first.

`num_warmups` does not fix this — it warms requests *within* an invocation, not
the compile cache. `warmup_steps` (default 1) runs whole discarded bench steps,
whose exports use a `bench-warmup.` prefix the host never globs.

### `--user <uid>` breaks torch's cache directory

Running the container as the calling uid keeps the HF cache and the exported JSON
deletable by the caller — without it they are root-owned and the next run cannot
clean up. But the uid has no passwd entry in the image, and torch's
`cache_dir()` computes its default via `getpass.getuser()`, which raises
`KeyError: getpwuid(): uid not found` at *import* of `torch._dynamo`.

The workload sets `TORCHINDUCTOR_CACHE_DIR` explicitly (so the default is never
computed) plus `USER`/`LOGNAME` (which `getpass.getuser()` prefers), and
redirects `HF_HOME`, `TRITON_CACHE_DIR` and `XDG_CACHE_HOME` into the mount. An
unwritable Triton cache is especially worth avoiding: it surfaces as "Triton is
not supported on the current platform".

### Bridged containers may have no route to the HF Hub

On a node with IPv4 forwarding disabled, a bridged container cannot resolve
`huggingface.co`, and the failure arrives as `LocalEntryNotFoundError` from
`snapshot_download` — which reads like a bad model id. `network` defaults to
`host` for that reason. Set `hf_offline: true` to serve strictly from a
pre-populated cache on a node with no egress.

Host networking means the gateway port binds on the *host*, so `port` defaults to
`auto` and is resolved per trial; otherwise two users of one node collide.

### `work_dir` must be node-local

An NFS home under root-squash cannot be bind-mounted: `docker run -v
/home/<user>/...` fails with a permission error on the mount point itself. The
workload stages the script from the installed package into `work_dir` for this
reason, so no manual staging step is needed. Keeping `work_dir` stable across
runs is also what stops every run from re-downloading weights.

### A timeout kills the docker client, not the server

`subprocess.run(timeout=...)` kills the local `docker run` process. The container
belongs to the daemon and keeps running, and `--rm` only fires once a container
exits — so a timed-out cell hands the next one a live TokenSpeed still holding the
GPU and the gateway port, and the next cell fails for a reason that appears
nowhere in its own logs. The container is therefore named
`aorta-ts-serve-<run-token>` and force-removed on timeout and on interruption. A
SIGKILL to the workload still leaks it, since no handler survives one, but the
runner sends SIGTERM first and that is the window this uses.

### Mitigations must be forwarded explicitly

The dispatcher resolves a cell's mitigations into `config["_aorta_trial_env"]`
and applies them to the workload process — but the engine runs in a container,
which does not inherit them. Forwarding uses the platform's `docker_env_flags`
helper. Without it both cells of a mitigation A/B benchmark the same
configuration and report a spurious "no effect".

## CI gating

The serving metric names are in `scripts/ci/eval_lib.py`'s `_METRIC_POLICIES`
allowlist (`median_*_ms` / `p99_*_ms` as `max`, the throughputs as `min`), so
they are gateable once baselines exist. Unlisted metrics are recorded for trends
but never gated.

To gate a serving recipe in the nightly: add it to
`config/ci/nightly_eval_matrix.yaml`, let it run record-only, then bless it via
the `refresh-baselines` workflow. See [ci-nightly-eval.md](ci-nightly-eval.md).

## Tests

```bash
python -m pytest tests/workloads/test_tokenspeed_serve.py -q
```

GPU-free and Docker-free: `shutil.which`, the `/dev/kfd` probe and
`subprocess.run` are monkeypatched, so what is covered is what this workload
owns — config validation, env/argv construction, export parsing, aggregation and
the verdict. The served-request audit, the exit-code mapping and the gates get
the most attention. Every committed recipe is also loaded through the real
recipe parser, resolved against the real mitigation registry, and pushed through
the workload's own validation, so a typo'd key fails on the CPU gate rather than
on a GPU node.

## Not done yet

- **`gpt-oss-20b`.** TokenSpeed's canonical AMD benchmark model. The workload has
  no size-specific logic, but it has not been run.
- **Multi-GPU serving.** `serve_args: --tensor-parallel-size N` should work and
  is untested; RCCL mitigations would become relevant.
- **Blessed nightly baselines.** The metrics are gateable but no serving recipe
  is in the nightly matrix yet, so nothing is gated.
- **`sharegpt` dataset.** Only the self-contained `random` dataset is wired;
  `--dataset-name sharegpt` needs a dataset file staged into the container.
