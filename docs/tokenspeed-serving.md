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
`lightseekorg/tokenspeed-amd:nightly-20260714`, `trials: 1`, `steps: 3`,
`warmup_steps: 1`, `ignore_eos: true`. Numbers are means across the three
measured steps. These are what the integration produced, not a performance
claim about TokenSpeed — one node, one seed, tiny request counts.

### Across models (`tokenspeed-serve-models.yaml`)

32 requests, ISL 512 / OSL 128, concurrency 8. All four cells passed with every
request served.

| Model | Startup (s) | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s | Total tok/s |
|---|---|---|---|---|---|
| Qwen3-0.6B | 358 | 45.4 | 1.92 | 3573 | 17867 |
| Qwen3-1.7B | 280 | 53.5 | 2.03 | 3273 | 16364 |
| Qwen3-4B | 325 | 58.2 | 3.81 | 1883 | 9414 |
| Qwen3-8B | 322 | 60.7 | 4.48 | 1620 | 8098 |

Latency and per-token cost rise with parameter count and throughput falls, which
is the expected shape — the point of the run is that all four sizes come up,
serve, and report coherently through the same code path.

### Across load shapes (`tokenspeed-serve-load.yaml`)

Qwen3-0.6B. All six cells passed; 1656 requests served in total, none failed.

| Cell | Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Output tok/s | Requests/s |
|---|---|---|---|---|---|---|
| conc-1 | 1 | 47.6 | 209 | 1.49 | 540 | 4.2 |
| conc-8 | 8 | 43.7 | 195 | 1.91 | 3623 | 28.3 |
| conc-32 | 32 | 60.6 | 316 | 2.02 | 12713 | 99.3 |
| conc-64 | 64 | 80.5 | 951 | 2.49 | 15726 | 122.9 |
| isl-4096-osl-128 | 16 | 65.1 | 476 | 2.95 | 4427 | 34.6 |
| isl-128-osl-1024 | 16 | 49.7 | 2116 | 2.02 | 7695 | 7.5 |

The concurrency rows are the useful part: throughput scales nearly linearly to
32 (540 → 12713 tok/s) and then saturates, gaining only 24% from 32 → 64 while
p99 TTFT triples (316 → 951 ms). That knee, not the peak number, is what the
recipe exists to find.

The two shape rows separate the regimes. Prefill-heavy (ISL 4096) raises TPOT to
2.95 ms — attention over a long context — while decode-heavy (OSL 1024) leaves
TPOT alone but pushes p99 TTFT to 2.1 s, because a short request queued behind
long decodes waits for them.

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
