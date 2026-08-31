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
| `tokenspeed-serve-gptoss.yaml` | gpt-oss-20b, two mitigation cells. TokenSpeed's canonical AMD benchmark model. |
| `tokenspeed-serve-gptoss-tp.yaml` | gpt-oss-20b at `--tensor-parallel-size` 1 / 2. |

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
| `container_elapsed_sec` | `docker run` wall clock, excluding the VRAM drain wait |

`tokens_per_sec` is an alias for `output_throughput`, present because AORTA's CI
gating allowlist already knew that name. TTFT is deliberately **not** aliased to
`prefill_latency_ms`: TTFT includes queueing delay, so gating it under that name
would gate a differently-defined quantity.

`step_times_ms` is the per-step bench `duration`, so `perf.md`'s step-timing
columns describe the bench runs.

`elapsed_sec` covers the whole cell, including the post-exit wait for VRAM to be
released. That wait blocks the trial and the GPU stays unusable throughout it, so
stopping the clock at container exit reported a cell that held the node for
minutes longer than its own `elapsed_sec`, and a sweep budget summed from those
came out short by the drain time of every cell. `container_elapsed_sec` is the
`docker run` duration on its own, for when benchmark time is what is wanted.

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

### gpt-oss-20b (`tokenspeed-serve-gptoss.yaml`)

TokenSpeed's canonical AMD benchmark model, and the size at which the numbers
start to mean something: 21B parameters with MXFP4-packed MoE experts, served
from a single MI355X. Both cells passed, 96/96 requests served per cell, none
failed. Same load as the model sweep above (32 requests per step, ISL 512 / OSL
128, concurrency 8), so the rows are directly comparable.

| Cell | Startup (s) | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|---|
| baseline | 316 | 67.1 | 7.61 | 994 |
| `hsa_no_scratch_reclaim` | 286 | 66.7 | 7.49 | 1008 |

Against Qwen3-0.6B on the same node and load: TTFT ~1.45×, TPOT ~3.9×, output
throughput ~0.28×. That is a reasonable shape for a 21B MoE against a 0.6B dense
model — decode is where the cost lands, prefill much less so — and the mitigation
is within noise of baseline, as it was at every smaller size.

TokenSpeed supports the architecture first-class (`runtime/models/gpt_oss.py`,
with a dedicated MXFP4 weight-loading path), and the repo is ungated, so no HF
token is needed. Two practical notes, both learned the hard way:

- The **full** snapshot is required, ~40 GB. Excluding `original/` fails during
  weight loading rather than at download time.
- Pre-warm the cache **as the uid the trial will run as**. `run_as_current_user`
  defaults to true, so a cache populated by a root container leaves the run
  failing with `PermissionError` on the snapshot directory — the same EPERM
  described in [`--user <uid>` breaks torch's cache
  directory](#--user-uid-breaks-torchs-cache-directory), arriving by a different
  route.

### Tensor parallelism (`tokenspeed-serve-gptoss-tp.yaml`)

Both cells passed, 96/96 requests each, none failed. Same load as above.

| Cell | Startup (s) | TTFT p50 (ms) | TPOT p50 (ms) | Output tok/s |
|---|---|---|---|---|
| `tp1` | 199 | 67.2 | 7.63 | 991 |
| `tp2` | 282 | 65.6 | 7.23 | 1036 |

TP=1 reproduces the single-GPU numbers above to within a percent, which is the
control this axis needs — without it a change in the multi-GPU path could not be
told apart from a change in the model or the image.

TP=2 buys about 4.5% throughput for a second GPU. That is a poor trade, and an
unsurprising one: at 21B with MXFP4 the model already fits in one MI355X, so the
second rank relieves no real constraint and pays collective cost on every step.
The interesting result here is that the path works and reports coherently, not
that it is fast.

**TP=4 does not come up.** Reproducibly, on this image, with an out-of-memory
raised while `FlatMemoryExecutor` builds its host-side KV mirror:

```
File ".../tokenspeed/runtime/cache/flat_host_mirror.py", line 127, in __init__
  torch.zeros(
torch.AcceleratorError: CUDA error: out of memory
```

One thing it is not: contamination from an earlier cell — the GPUs were reset
clean immediately before the run. The node has 3 TB of host RAM and eight 309 GB
cards, so neither host memory nor VRAM is scarce, and the error naming CUDA for
what the traceback shows to be a host allocation is part of why this took a while
to pin down.

Container shared memory is *not* ruled out, though it reads that way in earlier
notes. Those runs compared `shm_size: 16g` against `256g` and saw an identical
failure — but they predate the IPC fix below, and under the `--ipc host` in force
at the time docker ignored `--shm-size` entirely and both runs used the same host
`/dev/shm`. The comparison established nothing. Now that `shm_size` is effective
it is worth re-running before drawing any conclusion.

Not yet diagnosed further; the cell is left out of the recipe rather than shipped
red, because ranks that die this way hold their GPU memory long enough to poison
whatever runs next.

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
| `served_model_name` | `model` | The id both sides use: the bench asks for it, and when it differs from `model` the server is told to register it. Reserved in `serve_args` so the halves cannot diverge. |
| `num_prompts` | `64` | Requests per bench step. Also the audit target. |
| `input_len` / `output_len` | `1024` / `128` | Random-dataset ISL/OSL. Not sent and reported as `null` for `sharegpt`. |
| `dataset` | `random` | `random` or `sharegpt`. See [Datasets](#datasets). |
| `dataset_path` | — | Host path to a ShareGPT JSON. Required for `sharegpt`, rejected for `random`. |
| `max_concurrency` | unbounded | In-flight request cap. |
| `request_rate` | `inf` | The quoted string `"inf"` submits everything at once; an unquoted infinite float is rejected. See below. |
| `warmup_steps` | `1` | Discarded bench steps. See below. |
| `num_warmups` | `1` | Warmup requests *within* a bench step. |
| `ignore_eos` | `true` | Holds OSL fixed so cells do equal work. `false` is only accepted for `sharegpt`; on `random` the bench CLI pins it regardless, so the combination is rejected. See below. |
| `work_dir` | `/tmp/ts-work-serve` | Must be node-local. Scratch and the HF cache are per-uid beneath it, at `<work_dir>/u<uid>`. See below. |
| `hf_home` | `<work_dir>/u<uid>/hf` | Set it to share one pre-populated cache between users; see below for why that has to be deliberate. |
| `hip_visible_devices` | unset | Which GPUs the container sees. A visibility filter, not an allocation. |
| `exclusive_gpus` | `false` | Assert that no other job shares this node's GPUs. Only then does unreleased VRAM fail the trial. |
| `ready_timeout_sec` | `900` | Raise it for big models on a cold cache. |
| `network` | `host` | See below. |
| `port` / `control_port` | `auto` | Free ports, picked per trial. Explicit values must be in 1024..65535 and must differ. |
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

`completed` and `failed` are reported as `completed_total` / `failed_total` —
sums across the trial's measured steps, since "how many requests did this trial
serve" is the question and a mean hides one bad step among good ones. The raw
per-step counters are deliberately kept out of the metric aggregate: published as
means they would sit beside the sums in `perf.md`, so a three-step 32-prompt
trial would show `completed: 32` and `completed_total: 96` in the same table,
which reads as a discrepancy rather than as two units. Per-step counts stay in
the trial JSON, where a shortfall is attributable to the step that caused it.

The comparison is against zero exactly, in both layers. A negative `failed` is
not a run that did better than none-failed; it is an export that cannot be
believed, and reading it as "no failures" would admit `completed ==
num_prompts, failed == -1` with the metrics computed from whatever produced the
-1. Two independent audits only help if neither of them is the lenient one.

Both layers also check the counters' type exactly, not with `isinstance`. `bool`
is a subclass of `int` in Python and `json` decodes `true`/`false` into it, so
with the legitimate `num_prompts: 1` an export carrying `completed: true,
failed: false` compares equal to 1 and 0 — counters that measured nothing,
satisfying the guard.

Counting requests is not sufficient on its own, though. An export carrying only
`completed` and `failed` satisfies both checks, and since `gates` is empty by
default the cell would go green having measured no duration, no TTFT and no
throughput at all. Every measured step must therefore also carry finite values
for `duration` (greater than zero), `mean_ttft_ms`, `median_ttft_ms`,
`output_throughput`, `request_throughput` and `total_token_throughput`, plus
`mean_tpot_ms` / `median_tpot_ms` whenever a second output token can exist —
TPOT averages inter-token gaps, of which a single-token response has none. A step
missing any of them is reported as `result_json_unusable` rather than as a pass.

"Whenever a second output token can exist" is decided from `output_len > 1` only
for `random` with `ignore_eos: true`, the one configuration that actually pins
the output length. Everywhere else it is decided from the export, by comparing
`total_output_tokens` against `completed` — see
[Datasets](#datasets) for why.

All of them must be strictly *positive*, not merely finite. A step that served
requests took time, produced tokens and had a latency, so zero or negative is a
broken measurement rather than a fast one — and the negative case is the one that
matters, because it is finite enough to pass a naive check and a negative latency
makes a `max_*` gate read as an improvement.

The percentiles are deliberately not required: they depend on
`percentile_metrics` and `metric_percentiles`, so demanding them would fail a
cell for a legitimate recipe choice. They are, however, only published as a
scalar when *every* measured step carries them. Averaging over whichever steps
happen to have a key changes what the number means: a `max_p99_tpot_ms` gate over
three steps, with `p99_tpot_ms` in one, would otherwise evaluate that single step
while reading as a three-step aggregate — a gate passing on a third of the
evidence, when the caller was promised a missing metric instead. Partially
present keys are listed under `partial_metrics` and remain visible per step.

When a step does fail, the message explaining it is on **stdout** — every
`TS_BENCH_FAIL` diagnostic is printed, not raised — so `stdout_tail` is retained
on the nonzero-exit and incomplete-step failures as well as on `no_bench_export`.
Keeping only `stderr_tail` was survivable while a failure also produced no
records, since the `no_bench_export` branch carries stdout; but a later step
failing after earlier ones exported never reaches that branch, and the trial then
reported an exit code with the message that explains it discarded.

### A tensor-parallel teardown returns before the GPUs are actually free

`docker run` returning is not the same event as the device memory coming back.
Measured on gfx950 after a **passing** `--tensor-parallel-size 2` cell: the
container is gone, `docker ps -a` is empty, `rocm-smi --showpids` reports no KFD
processes — and one GPU still holds 256 GB of its 309 GB. Only rank 0's device is
released promptly. The rest clears on its own after roughly 30–45 s.

Nothing about the trial that causes this looks wrong, which is what makes it
dangerous. aorta starts the next cell immediately, so the cost lands there: the
next tensor-parallel cell dies during startup with

```
torch.AcceleratorError: CUDA error: out of memory
```

naming a device it never chose, for a model that fits comfortably. That is
exactly how the `tp4` cell of `tokenspeed-serve-gptoss-tp.yaml` first failed —
and read as "TP=4 does not work on this stack", which was wrong.

So the workload samples per-GPU VRAM before the run and waits for it to come back
afterwards, up to 5 minutes, before returning the result. The common case exits
on the first sample and costs nothing. If memory is still outstanding when
waiting stops helping, the trial fails with `gpu_memory_not_reclaimed` naming the
GPU and the `rocm-smi --gpureset -d N` needed to reclaim it — a real leak is
worth a red cell, because the alternative is a green one that breaks the next.

The comparison is against a pre-run sample rather than an absolute ceiling, but a
delta only shows that memory grew — it does not show who allocated it. On a
shared node a co-tenant starting a job mid-trial produces the same delta, and
blaming it here would redden a healthy cell and point `rocm-smi --gpureset` at
someone else's device.

So growth is only **attributed** when something outside this workload says the
GPUs were its own: `exclusive_gpus: true`, the recipe author asserting a
scheduler allocation or a machine they have to themselves. Without it the wait
still happens — waiting helps the next cell whoever owns the memory — but the
outcome is a logged warning rather than a failure, and the log says what to set.

`hip_visible_devices` does **not** carry that meaning, and used to be read as if
it did. It is a visibility filter on this process tree, not an allocation: it
narrows which devices are *watched*, since the container could not have allocated
anywhere else, while leaving the same physical GPU open to a co-tenant for the
whole trial. A cell that pins its devices and nothing more is therefore still
unattributed. `tokenspeed-serve-gptoss-tp.yaml` pins its GPUs per cell — that is
what makes the watch specific; the claim of exclusivity is a separate statement.

VRAM is read from `/sys/class/drm/card*/device/mem_info_vram_used` — no
subprocess, no PATH assumption — and a node that does not expose it simply skips
the check, since a missing measurement is not evidence of a leak.

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

"The same values" has to include the *ranges*, in both directions. Where the host
was more permissive than the script — explicit ports below 1024, a
`ready_timeout_sec` above 86400, a `teardown_grace_sec` outside 5–3600 — the
recipe passed `setup()`, took a GPU node, and then exited 64 inside the
container. The result read as a workload failure with the violated bound visible
only in the container log, when it was a configuration error that could have been
caught before anything was allocated.

The lower bound on `teardown_grace_sec` is a relationship rather than a taste:
the gateway's `--drain-timeout` is derived from it and has to finish *inside* it,
or teardown escalates to SIGKILL while the gateway is still draining — the
delayed-VRAM-release failure the drain exists to prevent, arriving through the
mechanism meant to prevent it. Below 5 there is no positive drain that fits.

The same relationship is enforced when the drain is set explicitly. `serve_args:
["--drain-timeout", "60"]` against the default 45-second grace put the drain
back outside the window, and `--drain-timeout` is a *serve* flag, so the guard
that rejects owned *bench* flags never saw it. An explicit value is now required
to be at least 1 and strictly less than `teardown_grace_sec` — rejected on the
host and again in the script, since `TS_DRAIN_TIMEOUT` can also arrive from a
mitigation. Raise `teardown_grace_sec` if a gateway genuinely needs longer.

The two layers also have to accept the same *spellings*, not only the same
range. `int()` reads `"+5"`, `" 5 "`, `"3_0"` and the zero-padded `"08"`; the
script's `require_uint` reads none of them, so those recipes passed the host and
then exited 64 in the container — where the failure reads as a script problem on
a recipe the host had already approved. The host now requires the same plain
unsigned decimal, and rejects rather than normalises it: `08` more likely means 8
than octal 010, and the recipe should say which.

Integer fields are also checked as integers rather than coerced with `int()`,
which accepted two shapes of malformed recipe and ran a *different* load instead
of failing. `num_prompts: true` becomes 1 because `bool` is an `int` subclass,
and `num_prompts: 1.9` truncates to 1 — in both cases the cell then reports
`num_prompts: 1` for a value nobody wrote. A recipe that cannot be read the way
it was written must not be run the way it was not.

### A mitigation cannot redefine the host/container protocol

Cell mitigations are forwarded into the container, which is the whole point of
the matrix — but a mitigation may not redefine a value this workload sets itself,
because only the container would learn about the new one. Three shapes of
failure, in increasing order of how badly they mislead:

- `TS_NUM_PROMPTS=999` has the script request 999 while the host still audits
  against the recipe's count, so a healthy cell fails for a served-request
  shortfall.
- `TS_RUN_TOKEN` has the script write exports under a name the host's glob never
  matches, so the cell fails for finding no export at all.
- `TS_MAX_CONCURRENCY=1` changes the load actually applied while the host keeps
  reporting the configured value. Nothing fails — the cell passes, carrying a
  number that describes a run that did not happen. This is the worst case, and
  the reason the guard covers reported values and not only audited ones.

The owned set is the union of two things: the environment the workload actually
built, and `_PROTOCOL_ENV_KEYS`. Computing it from the environment means anything
added there later is covered without a second place to remember. The declared set
is needed as well, because for some keys the workload's setting *is* absence —
`max_concurrency` defaults to unbounded and expresses that by setting no
`TS_MAX_CONCURRENCY` at all, so a computed-only set guarded the configured case
and left the default one open, which is the case most cells run. A test asserts
every declared key is one the workload sets under some configuration, so the
declaration cannot overreach and forbid a legitimate mitigation.

Any `TS_*` knob the workload does not set — an engine
tunable, an attention backend — is forwarded normally; all 22 mitigations in
aorta's registry pass through untouched.

### Unlimited has to be written as a quoted `"inf"`

`request_rate: "inf"`, `"+inf"` and `"Infinity"` mean submit everything at once.
An infinite *float* is rejected, which looks pedantic and is not: YAML reads
`.inf` and `1.0e999` as the same value, and so does `float("1e999")`, so by the
time the value arrives there is no way to tell a deliberate unlimited from a
finite rate whose exponent was mistyped. Accepting it turned that typo into the
heaviest load the harness can generate while the trial went on reporting the
rate the recipe asked for — a green cell describing a run that did not happen.
The quotes cost the deliberate case nothing and the accident cannot produce them.

### `ignore_eos: false` has never reached the random dataset

`tokenspeed bench serve` decides this for itself. In `bench.py`, the flag is
honoured first:

```python
if args.disable_ignore_eos:
    args.ignore_eos = False
```

and then, further down the same function and after the tokenizer is loaded, the
dataset rule overwrites it unconditionally:

```python
if args.dataset_name == "random" and args.backend in OPENAI_COMPATIBLE_BACKENDS:
    args.ignore_eos = True
```

`OPENAI_COMPATIBLE_BACKENDS` is `{"openai", "tokenspeed"}` and this workload
benches the gateway with `--backend openai`, so on `dataset: random` there is no
argv that turns EOS back on: omitting `--ignore-eos` does not, and neither does
`--disable-ignore-eos`. Every request goes out with `ignore_eos` in its payload
and runs to `output_len`.

The config table used to present `ignore_eos` as a plain boolean, so a recipe
setting it to `false` on the random dataset ran at a pinned length while the
trial reported `ignore_eos: false` — the reported configuration is not the one
that ran, and nothing in the export contradicts it. That is the same shape as a
`bench_args` override of `--max-concurrency`, and it is treated the same way:
the combination is **rejected** during validation, on the host and again in
`ts_bench_serve.sh`, rather than warned about.

The route that does work is the request payload:

```yaml
bench_args: ["--extra-body", '{"ignore_eos": false}']
```

`_update_payload_common` writes the forced `ignore_eos` into the payload and
*then* merges `extra_body` over it, and `--extra-body` is not one of the flags
this workload reserves. Expect the cells to stop doing equal work once you do
this — output lengths become whatever the model chooses, so `perf.md` is
comparing runs of different sizes. That is why the default is `true`.

`dataset: sharegpt` is unaffected. The rule is keyed on the dataset name, so
`ignore_eos: false` is honoured there and stays accepted.

### Extra arguments cannot shadow the flags the workload owns

`serve_args` and `bench_args` reach `tokenspeed serve` and `tokenspeed bench
serve` as extra flags, and both CLIs take the *last* occurrence of a repeated
option. The extras arrive after the flags the workload sets, so a caller's value
silently won — and each of these fails somewhere that does not mention the cause:

| In | Consequence |
|---|---|
| `serve_args: ["--port", "9000"]` | the gateway starts on 9000 while readiness is polled on the resolved port; a healthy server reads as one that never came up |
| `bench_args: ["--output-file", ...]` | the export lands where neither the in-container audit nor the host's glob looks; a completed benchmark reads as a missing result |
| `bench_args: ["--num-prompts", "4"]` | the bench runs 4 while the host audits against the recipe's count |
| `bench_args: ["--max-concurrency", "1"]` | the applied load changes while the host still publishes the configured cap; both request audits pass, so the cell goes **green** describing a run that did not happen |

That last row is the worst of them and the reason the list covers the load
controls — `--max-concurrency`, `--request-rate`, `--num-warmups`,
`--ignore-eos`, `--seed` — and not only the export plumbing. Nothing detects it:
every request completes, none fail, and the reported configuration is simply not
the one that ran. They are reserved even at their defaults, where the script
appends nothing at all, since a guard that covers only the configured case leaves
the case most cells run unguarded.

### The list form survives the trip into the container

Both fields are lists in the recipe, and they reach the container as a JSON
array in `TS_SERVE_ARGS` / `TS_BENCH_ARGS` rather than as a space-joined string.
Joining and re-splitting threw away the boundaries the list form exists to
express: `bench_args: ["--extra-body", '{"a": 1}']` arrived as three arguments,
and a value containing `*` or `?` was glob-expanded against the container's
filesystem before TokenSpeed ever saw it.

A string is accepted too, and it is parsed the way a shell would parse it, not
with `split()`. Under `split()` there was no way to write an argument containing
whitespace: `serve_args: '--extra-body "{\"a\": 1}"'` became three arguments, so
the string form could express something other than what it said and the checks
below saw tokens nobody wrote. An unbalanced quote is a config error rather than
a guess at what was meant.

`ts_bench_serve.sh` decodes the array through a NUL-separated stream — the one
delimiter that cannot occur inside an argument — into a Bash array, and the
owned-flag guard above runs over that array, so a flag is matched as an argument
rather than as text that might appear inside someone's `--extra-body` JSON.
Setting either variable by hand means writing JSON:
`TS_BENCH_ARGS='["--foo","bar baz"]'`. Anything else exits 64 saying so.

The guard also rejects **abbreviations**. Python's argparse resolves any
unambiguous prefix by default, so `--max-conc 1` sets `--max-concurrency`
without ever matching an exact-spelling denylist — the same green-cell-wrong-run
outcome as the spaced form, arriving by a spelling the list did not contain. We
could not confirm the upstream CLI's `allow_abbrev` setting from outside the
image, so the guard fails closed: any extra argument that is a strict prefix of
an owned flag is refused. That costs nothing if abbreviation is disabled
upstream, since such a spelling would then just be an unrecognised flag, and a
CLI with abbreviation enabled could not offer a strict prefix of an owned flag
as a distinct option anyway — it would be ambiguous with the owned one. Flags
that merely *contain* an owned flag as their own prefix, like `--seed-offset`,
are unaffected.

Reordering would fix the precedence but leave the override silently ignored,
which is its own trap, so `ts_bench_serve.sh` rejects these by name with exit 64
and names the `workload_config` field instead. Extras that do not collide —
`--tp`, `--goodput`, engine tunables — are passed through as before.

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

### `work_dir` is a shared root; scratch beneath it is per-uid

Scripts and exports go to `<work_dir>/u<uid>`, not to `work_dir` itself. A fixed
path in `/tmp` is owned by whichever user got there first, at that user's umask,
so the second user on the node failed while creating `scripts/` — a permission
error with nothing to do with their recipe — and where the mode did permit
writing, `keep_work_dir: false` deleted the other user's exports mid-run. Every
recipe in this repo names the same `/tmp/ts-work-serve`, so this is applied to
the configured value and not only to the default. The root is created `1777`
when the workload creates it, for the same reason `/tmp` is: the sticky bit lets
everyone create their own subdirectory while stopping *other users* removing
another's.

With one exception the sticky bit does not cover, which is why the root's
*ownership* is checked as well: the owner of a sticky directory may rename or
remove anything in it however it is owned. Whoever ran first owns the root, so a
co-tenant who got there first could swap this trial's `u<uid>` for a directory of
their own after the per-uid check had approved it. A root owned by root (an
administrator provisioned it) or by the current user is accepted; one owned by
another unprivileged user is refused, naming both remedies — set `work_dir` to a
path you own, or have an administrator create the shared root root-owned and
`1777` so it is shareable without being anyone's to rearrange.

The HF cache is per-uid too, at `<work_dir>/u<uid>/hf`. Sharing it by default
was tried and withdrawn: a cache is only shareable if later users can *write* it
— the failure a second user hits is a cache miss, not a read — and making the
parent world-writable does not achieve that, because `huggingface_hub` creates
`hub`, `.locks` and each model directory at the creating user's umask. A
world-writable model cache is also something any local user can pre-populate
with entries a later run would load.

Sharing a pre-warmed cache is still worth doing on a busy node, since a gpt-oss
snapshot is roughly 40 GB. It just has to be deliberate: point `hf_home` at a
directory an administrator populated, where read-only is the intended mode
rather than an accident of who ran first.

### An HF token is passed by name, never by value

Gated models need a token, read from the host environment variable named by
`hf_token_env` (default `HF_TOKEN`) so the value is never in a recipe. It is
forwarded to the container as a bare `docker run -e HF_TOKEN`, with no value:
docker takes it from its own environment, which the workload populates when it
spawns the client. The value-carrying spelling, `-e HF_TOKEN=<value>`, would put
the token in the docker client's argv, and `/proc/<pid>/cmdline` is readable by
every user on the node for as long as the trial runs — minutes to hours on a
serving benchmark, on shared nodes. `/proc/<pid>/environ` is readable only by
the owning uid and root, which is where it lives instead.

### A timeout kills the docker client, not the server

`subprocess.run(timeout=...)` kills the local `docker run` process. The container
belongs to the daemon and keeps running, and `--rm` only fires once a container
exits — so a timed-out cell hands the next one a live TokenSpeed still holding the
GPU and the gateway port, and the next cell fails for a reason that appears
nowhere in its own logs. The container is therefore named
`aorta-ts-serve-<run-token>` and force-removed on timeout and on interruption.

Interruption includes a signal, which needs its own handler rather than an
`except` clause: under Python's default disposition SIGTERM terminates the
interpreter *without raising*, so `except BaseException` around the `docker run`
call never runs and the case most likely to strand a container — a cancelled
sweep, an expired job budget — was the one case not covered. SIGTERM and SIGHUP
are trapped for the duration of the run, and the handler removes the container,
installs `SIG_DFL` and re-raises the signal at itself, so a supervisor still
sees death by signal (exit 143) rather than a swallowed one. `SIG_DFL`
specifically, rather than whatever disposition was there before: under `nohup`
SIGHUP is inherited as `SIG_IGN`, so restoring the previous one meant the
re-sent signal was ignored and the handler returned into the benchmark with its
container already removed underneath it. A
SIGKILL still leaks the container, since no handler survives one, but the runner
sends SIGTERM first and that is the window this uses.

Every cleanup path reaps the `docker run` client *before* removing the container,
which is why the run is launched with `Popen` rather than `run()` — the handler
needs a client it can reach. Removing first left a window in which the removal
reported "No such container" because the daemon had not created it yet, and the
surviving client went on to create it after this process was gone: an orphan on
the GPU produced by the cleanup path itself. Reaping the client first closes it,
since nothing is then left that could still create the container. Draining the
client on the way out also recovers what it had written, so a timeout still
reports the bring-up log that explains it instead of discarding the pipe.

Three details of that are load-bearing, and each of them was a bug first:

- **The lock is reentrant.** A Python signal handler runs in the main thread,
  which is the thread holding the lock while the client is spawned and published,
  so a plain `Lock` made the handler wait for a release only the code it had
  interrupted could give. The process then sat there until SIGKILL, container
  still running.
- **Cleanup runs inside the termination context**, not in an `except` clause
  after it. Cleanup takes over a minute in the worst case — terminate grace, pipe
  drain, `docker rm -f` — and with the handlers already restored a signal in that
  window killed the process outright and stranded the container.
- **The removal is retried when the client could not be reaped.** That is only
  possible while `Popen` itself is running, since the object to reap does not
  exist until it returns; such a client can create the container just after the
  first removal finds nothing, so three passes a second apart cover it. When the
  client *was* reaped, one pass is enough and only one is made.
- **The removal runs on every path**, including the one where the client exited
  on its own. A completed `communicate()` proves the *client* is gone, not the
  container: a client that lost its connection to the daemon, was OOM-killed, or
  had the daemon restart under it returns an exit code while the named container
  keeps serving and holding the GPUs. `--rm` cannot help there, because it fires
  when the container exits, which is the thing that did not happen. `docker rm
  -f` on an already-removed container is one call reporting "No such container",
  so the ordinary path pays a single no-op — the same trade `host_launch.sh`
  makes with its unconditional EXIT cleanup.

Removal failures are warned about with docker's exit code and stderr. `docker rm
-f` reports an unreachable daemon or a permission problem by exit status rather
than by raising, so checking only for exceptions meant the one outcome worth
knowing about — the container is still up, still holding the GPU — was the one
that produced no output. "No such container" is logged at debug instead: `--rm`
usually gets there first, and a trial that failed before the container existed
has nothing to leak.

### `docker_args` cannot displace a generated option

Extra arguments are spliced in after the ones the workload generates, and docker
takes the *last* occurrence of a repeated option — so `docker_args` was a way to
quietly turn off the guarantees above. `--name` is the sharpest: the container
then runs under a name the cleanup path does not know, so a timed-out trial leaks
a live server while the workload reports having removed it. `-v ...:/ts-out`
sends the exports somewhere the host does not look, `--entrypoint` means the
bench script never runs, and `-e TS_RUN_TOKEN=...` reaches the same
desynchronisation the mitigation guard rejects while bypassing that guard
entirely. Two more with less obvious symptoms: a later `--ipc` or `--shm-size`
replaces what TokenSpeed's scheduler sizes its shared memory against, and fails
at load time with an error about shared memory rather than about `docker_args`;
`--rm=false` leaves every completed container behind, so a sweep fills the node
up while no single trial looks wrong.

`--name`, `--entrypoint`, `-v`/`--volume`, `--mount`, `--network`, `--user`,
`--env-file`, `--ipc`, `--shm-size`, `--rm`, `--device`, `--group-add` and
`--security-opt` are therefore rejected as configuration errors, as is any `-e`
naming a protocol variable.

`--detach`/`-d` is rejected for a different reason: it does not displace an
option, it removes the client this workload supervises the run through. `docker
run -d` returns 0 immediately, so the trial reports success with no exports
while the container keeps benchmarking and holding the GPU — and because nothing
raised and nothing timed out, no cleanup path runs. The combined short cluster
`-dit` is caught too, since `-d` there never appears as its own token. Everything else still passes through — that is what
the field is for.

Two details the guard has to get right, because either one makes it avoidable
rather than strict. Docker accepts a value attached to a short option, so `-u0:0`
and `-v/tmp:/ts-out` are the spaced forms under another spelling — and so are
`-e=NAME=value`, where the first `=` is docker's separator, and `-ieNAME=value`,
where a boolean letter precedes the option that takes the value. Every token is
parsed once, into the option it sets plus whatever value is attached to it,
because reading it three partial ways left gaps between them: `-e=` extracted an
empty name and matched nothing, and a cluster scan that stopped at `e` or `v`
without reading the remainder accepted the same override it rejects spaced. And
the `-e` half is compared against the declared
protocol floor, not only the variables this run populated: `max_concurrency`
defaults to unbounded and sets no `TS_MAX_CONCURRENCY`, so checking only what is
present left the default configuration open to exactly the mislabelled pass
described above.

The same reasoning applies inside the container: `serve_args` and `bench_args`
may not set the flags `ts_bench_serve.sh` derives itself (`--port`,
`--base-url`, `--num-prompts`, `--output-file` and the rest), since those are
what tie the export back to the cell that asked for it.

### The IPC namespace is private, which is what makes `shm_size` mean anything

Docker applies `ShmSize` only when it creates the `/dev/shm` mount itself. Under
`--ipc host` the container gets the host's mount and `--shm-size` is ignored
without a word — so `shm_size` was a documented setting that did nothing, and the
TP=4 note above "ruled out" shared memory by comparing one host mount with
itself.

Nothing needed host IPC. Every TokenSpeed process — orchestrator, engine,
scheduler, gateway — is forked inside this one container and already shares its
private IPC namespace, which is the same reasoning `harvest_code_objects.py`
records for its own `--shm-size`. Host IPC additionally exposes node-wide shared
memory and semaphores to a third-party image for no gain, so it is not passed and
`shm_size` (default `16g`) is what sizes the mount.

### Auto-resolved ports must not collide with each other

`port: auto` binds an ephemeral port, records it and closes the probe socket, so
by the time the control port is resolved the kernel is free to hand back that
very port — leaving the two equal, which `ts_bench_serve.sh` rejects as a usage
error. A valid `auto` configuration would fail intermittently, on a node under
no unusual load. Resolution therefore retries, holding each candidate open until
one lands outside the set already claimed.

The mixed case needs the same care: with `port: auto` and an explicit
`control_port`, the gateway is resolved with that explicit value already in its
`avoid` set. Resolving it blind let the kernel hand the gateway the very port the
control endpoint was configured to use — most likely when the explicit value sits
in the ephemeral range, which is where anyone picking "a high free port" would
put it — and the equality check then rejected a configuration that was valid.

Explicit ports are held to 1024..65535, matching what `ts_bench_serve.sh`
enforces: the container runs unprivileged and cannot bind a reserved port. The
host used to accept 1..65535, so such a recipe passed `setup()` and was then
guaranteed to fail with the script's exit 64 — after occupying a node, and
reported as a workload failure rather than the configuration error it is.

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

GPU-free and Docker-free: `shutil.which`, the `/dev/kfd` probe and the
`subprocess` entry points are monkeypatched, so what is covered is what this workload
owns — config validation, env/argv construction, export parsing, aggregation and
the verdict. The served-request audit, the exit-code mapping and the gates get
the most attention. Every committed recipe is also loaded through the real
recipe parser, resolved against the real mitigation registry, and pushed through
the workload's own validation, so an out-of-range value fails on the CPU gate
rather than on a GPU node.

An *unknown* key is only a warning at runtime, on purpose: a config carrying a
key some other tool reads should not be fatal. That made the gate weaker than it
looked, though — a recipe saying `num_prompt` passed validation and then ran the
whole matrix at the default request count, silently, with plausible numbers. So
the recipe test treats that warning as fatal for the recipes in this repo, which
are ours to keep correct. A typo in a committed recipe fails the CPU gate; a
typo in someone's local recipe still warns and runs.

## Datasets

`dataset: random` (the default) generates its prompts from `input_len` and
`output_len`, which is what makes it self-contained and reproducible — every
existing recipe uses it.

`dataset: sharegpt` benches against real conversation lengths, which is what
makes a number comparable to a published one. It requires `dataset_path`
pointing at a ShareGPT JSON file **on the host**, which the workload mounts
read-only at `/ts-data/dataset.json`. That is deliberately stricter than the
bench CLI, which downloads ShareGPT from the Hub when given no path — convenient,
and wrong here three ways over: the recipe stops being reproducible (the URL is
not pinned and the file is not content-addressed), a bridged container often has
no route to the Hub, and the download lands *inside the measured window*, so the
first trial reports slower for a reason that is not the engine.

Both layers validate it, and both do so before a server starts — a dataset
problem found after the model has loaded costs minutes and reads as a bench
failure rather than the staging mistake it is.

What gets mounted is a copy under the work root, named by the digest of its
contents, not the path the recipe gave. The path is validated with the caller's
credentials, but the mount is resolved by the docker daemon, and on the NFS home
this integration expects — the reason the work root has to be node-local at all —
a file its author can read is squashed to nobody for the daemon. The container
then gets an empty mount or an error, after the model has loaded, which is
exactly the failure the up-front check exists to prevent. Copying moves the read
to a directory where the daemon is root. Addressing the copy by content means a
sweep over an unchanged dataset pays for it once, an edited dataset is never
served from the previous copy, and concurrent runs cannot half-write each
other's.

`input_len`/`output_len` are dropped for `sharegpt`, since it takes its lengths
from the conversations. They are not sent to the container, and they are
published as `null` rather than as the recipe's defaults: reporting them would
label the result with a shape the run did not have, and a matrix mixing the two
datasets would compare those labels as though they meant the same thing.

The TPOT audit follows from the same question — does the configuration actually
determine the output length? Only `random` does, and it always does, since EOS
is ignored there whatever the recipe says (see [`ignore_eos: false` has never
reached the random dataset](#ignore_eos-false-has-never-reached-the-random-dataset)),
so the audit asks whether `output_len` exceeds 1. For `sharegpt` it asks the
export instead, whether more output tokens were produced than requests
completed: the lengths come from the conversations rather than from the recipe,
and with `ignore_eos: false` — which only `sharegpt` can express — the model
stops at its first EOS token, which for a short prompt can be immediately, so
every request may emit exactly one token and TPOT is genuinely undefined. Keying
off `output_len` there rejected a correct export.

## Not done yet

- **Blessed nightly baselines.** The metrics are gateable but no serving recipe
  is in the nightly matrix yet, so nothing is gated. The matrix lives in
  `aorta-internal`, where perf gating is still awaiting review.
- **`sharegpt` measured on hardware.** The plumbing is tested; no run has been
  made against a real ShareGPT file, so there are no numbers from it yet.
- **TP=4 and above.** TP 1 and 2 work; 4 fails to come up, diagnosed as far as
  the host-side KV mirror allocation but no further (see above). The RCCL
  mitigations that would become relevant at wider TP are untouched.
- **Why crashed TP ranks keep their memory.** A clean teardown clears in 30-45s
  and the workload waits it out. Ranks that die during startup hold theirs past
  a 5-minute wait and need `rocm-smi --gpureset`. Worth understanding, since it
  is what turns one failed TP cell into a failed sweep.
