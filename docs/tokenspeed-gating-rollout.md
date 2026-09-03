# Turning on nightly perf gating for TokenSpeed serving

[`tokenspeed_serve`](tokenspeed-serving.md) reports TTFT, TPOT, ITL and
throughput, and the serving metric names are already in the nightly's gating
allowlist. A serving recipe is now in `config/ci/nightly_eval_matrix.yaml`, but
nothing is gated, because no serving baseline has been blessed. This document is
the sequence for changing that, and the reasoning behind the numbers it picks.

The reason it is a document rather than a commit is that we do not yet have a
window to derive thresholds from. A threshold derived from a single observation
encodes whichever night it was taken on, and the nightly then fails on the
difference between two healthy runs. That failure is worse than no gate: it
trains everyone to ignore the alert, and the first real regression arrives into a
channel nobody reads.

Measuring the cell rather than reasoning about it has since made that case
stronger and the first bless smaller. One cell-run in thirteen carries a
five-second compile excursion in its first measured step, which three of the four
originally-proposed gates cannot survive — see [the smoke cell
section](#the-smoke-cell-is-not-unconditionally-clean-measured-13-cell-runs).

See [ci-nightly-eval.md](ci-nightly-eval.md) for how the nightly works in
general; this only covers what is specific to serving.

## Record-only is the absence of a baseline, not a matrix field

Worth stating plainly, because the phrase suggests a setting somewhere and there
isn't one. `nightly_eval.py` looks each `(entry, cell)` up in
`config/ci/regression_baselines.yaml` by the key `entry::cell`; a cell with no
key gets `compare_to_baseline(harvested, None)`, which returns `record`. So an
entry added to the matrix is record-only by construction, and stops being
record-only the moment a `refresh-baselines` PR merges a key for it.

Two consequences that decide the shape of this rollout:

- **Record-only defers performance gating only.** A cell that errors or fails is
  a `fail` with or without a baseline — the harness is fail-closed by design, and
  `docs/ci-nightly-eval.md` says so. An entry may therefore only be added to the
  matrix once it can actually *pass* on the runner. Adding one that cannot is not
  a soft landing; it is a nightly that is red every night. That constraint is
  what shaped this entry: it declares `needs_docker_daemon: true` so that a
  runner without the socket **skips** it rather than failing — see [What blocks
  this today](#what-blocks-this-today).
- **Blessing is all-or-nothing per refresh, unless it is scoped.**
  `refresh_baselines.py` rebuilds the whole baseline file and `--perf-gate` armed
  every entry in it. Running it to bless serving would, in the same PR, derive
  step-time ceilings for `gpu_smoke`, `inference_offline`, `training_ddp`,
  `training_fsdp`, `race` and `llm_determinism` — sixteen cells, each from a
  single observation, none of which anyone has variance data for. That is the
  failure this document is about, inflicted on six unrelated workloads as a side
  effect. `--perf-gate-entry` (added with this document) scopes it.

## What we actually know about the variance

Every number below is measured, on one gfx950 (MI355X) node against the image
digest the recipes pin. There is no synthetic data here, and no more of it than
this — that is the whole problem.

**Bring-up is the noisy one, and it is very noisy.** From
[tokenspeed.md](tokenspeed.md):

> Startup to `/health` is the dominant cost and it is **noisy**: 189, 276, 285,
> 291, 316 and 319 seconds across six runs of the same recipe on the same node —
> a 1.7× spread with nothing changed between them. So do not treat a slow
> bring-up as a signal without repeats, and do not tune `timeout_per_trial` close
> to an observed number; the recipe's 1800 s leaves deliberate headroom.

The multi-model sweep adds a seventh observation for the same 0.6B model at
379 s, and [tokenspeed-serving.md](tokenspeed-serving.md) declines to treat the
column as a measurement at all:

> Read the startup column as a floor, not a measurement: it is dominated by
> weight loading and Triton compilation against whatever the node's caches
> already hold, which is why the smallest model here posts the largest number. It
> is reported because `ready_timeout_sec` has to cover it, not because it scales
> with anything.

The four-model table is the evidence for that last clause: 379, 283, 289 and 328
seconds for Qwen3 at 0.6B, 1.7B, 4B and 8B. Bring-up does not even order by
model size.

**The compile cache is a second, larger excursion, and one mitigation stands
between it and the metrics:**

> On Qwen3-0.6B the first bench invocation against a fresh server took 6.2s
> against 1.1s for every later one. Rolled into the metrics that one outlier
> dominates the mean step time and inflates TTFT tenfold (465ms vs 47ms), so a
> cell looks like a regression purely because it went first.

`warmup_steps: 1` is supposed to discard that step. **It does not always**, and
that is the single most important number in this document — see the next
section, which measures it on the cell the staged entry actually runs. That
measurement is why the recipe now sets `warmup_steps: 2`.

**Steady-state serving metrics, by contrast, reproduce well.** The docs do not
say this in one sentence, but they contain two independent repeats:

- `tokenspeed-serve-models.yaml::qwen3-0.6b` and
  `tokenspeed-serve-load.yaml::conc-8` run a byte-identical measurement
  configuration — Qwen3-0.6B, ISL 512 / OSL 128, concurrency 8, 32 prompts, 3
  measured steps, 1 warmup step, `ignore_eos`, seed 0 — in two separate sweeps.
- `tokenspeed-serve-gptoss.yaml::baseline` and
  `tokenspeed-serve-gptoss-tp.yaml::tp1`, of which the doc says "TP=1 reproduces
  the single-GPU numbers above to within a percent, which is the control this
  axis needs".

| Metric | Qwen3-0.6B, two sweeps | spread | gpt-oss-20b, two sweeps | spread |
|---|---|---|---|---|
| `median_ttft_ms` | 46.3 / 45.9 | 0.87% | 67.1 / 67.2 | 0.15% |
| `median_tpot_ms` | 1.94 / 1.91 | 1.57% | 7.61 / 7.63 | 0.26% |
| `output_throughput` | 3538 / 3631 | 2.63% | 994 / 991 | 0.30% |
| `server_startup_sec` | 379 / — | — | 316 / 199 | 59% |

The same node reports serving rates to within 3% across sweeps and bring-up time
to within 59%. Those two facts are what the per-metric table below is derived
from.

There is also a useful contrast from the kernel side of the integration, where
the repo already distinguishes a tight metric from a loose one: the GEMM probe's
"spread across trials was under 1.5 µs in every cell". Not everything TokenSpeed
measures is noisy; bring-up is.

### The smoke cell is not unconditionally clean (measured, 13 cell-runs)

Everything above is a claim quoted from another document. This section is a
measurement of the exact cell the staged nightly entry runs, taken from the
`step_times_ms` and `metrics_summary` recorded in every `matrix.json` on disk for
`TOKENSPEED-SERVE-SMOKE`: six sweeps, 13 cell-runs with step times, 39 steps.

> **These numbers were all taken at `warmup_steps: 1`, which the recipe no
> longer uses.** The recipe now sets `warmup_steps: 2`, precisely because of what
> this section measures. So read everything below as **the rationale for that
> change, not as the baseline to bless against**: the table is the evidence that
> a step-0 excursion reaches the metrics at `warmup_steps: 1`, and it is retained
> for that purpose. It is *not* a prediction of what the record-only window will
> show, because the window is taken at a different setting — one whose whole
> purpose is to remove the excursion these thirteen cell-runs contain. Nothing
> here is deleted or restated at the new setting; there is no measurement at
> `warmup_steps: 2` yet, and inventing one would be worse than having none. The
> ten-night window is what produces it. See step 3 of
> [the rollout sequence](#the-rollout-sequence).

**Twelve of the thirteen are very clean.** Tighter than the cross-sweep numbers
above, because these are the same recipe on the same node:

| Metric | Clean range over 12 cell-runs | Spread |
|---|---|---|
| `median_ttft_ms` | 43.65 – 46.87 | 7.37% |
| `median_tpot_ms` | 1.89 – 1.95 | 3.08% |
| `output_throughput` | 3502.53 – 3646.20 | 4.10% |
| `p99_itl_ms` | 34.00 – 35.84 | 5.43% |
| step time | 1108 – 1178 ms | 6.3% |

**The thirteenth is the compile-cache excursion, and `warmup_steps: 1` did not
catch it.** `serve-smoke3::no-scratch-reclaim` recorded its three measured steps,
in order, as:

```
6193.4 ms,  1140.2 ms,  1142.6 ms
```

The excursion is the **first measured step** — after the warmup step was already
discarded. Its cell then reported `median_ttft_ms` 465.30 against a clean 43.65–
46.87, which is 10.26× the clean mean and the same 465 vs 47 the serving doc
quotes. So the 10× TTFT excursion is not a hypothetical a recipe edit could
introduce; it is present, once, in the data we already have, at **1 cell-run in
13 (7.7%)**.

What it does to the four gates the table below proposes, anchoring each on the
worst of the twelve clean runs and applying the plan's own margins:

| Gate | Anchor | Threshold | Excursion | Verdict |
|---|---|---|---|---|
| `step_time_ms.max` | 1169.50 | 1461.88 | 2825.40 | **fail** |
| `median_ttft_ms` | 46.87 | 58.59 | 465.30 | **fail** |
| `output_throughput` | 3502.53 | 2977.15 | 2612.83 | **fail** |
| `median_tpot_ms` | 1.95 | 2.44 | 1.93 | pass |
| `p99_itl_ms` | 35.84 | 44.80 | 34.96 | pass |

Three of the four proposed gates fire on a run that is not a regression. Those
verdicts are `compare_to_baseline`'s, not arithmetic done here —
`tests/ci/test_eval_lib.py` runs the measured numbers through the real comparator.

**Why the latency-per-token metrics survive it, and why that is the useful
signal.** The excursion is a fixed ~5 s of compilation added to one step. It
therefore lands on anything derived from step *duration* — the step-time mean,
and throughput, which is tokens over that duration — and on TTFT, because the
first request of that step waits for the compile. It does not touch
`median_tpot_ms` (1.01× clean) or `p99_itl_ms` (1.00× clean), because those are
measured *between tokens*, after the compile has happened. A metric whose
definition excludes the excursion is immune to it by construction, not by luck.

**This is a different failure from the concurrency-64 one, and the difference
decides the fix.** The matrix work found ~8% of bench steps at concurrency 64
stalling by a fixed ~0.92 s with no positional pattern — the `serve-load::conc-64`
cell on disk reads `1621, 2286, 2595` ms, bimodal in a way no warmup setting can
remove. The smoke cell's excursion is at position 0 every time it appears, which
makes it **deterministically avoidable**: it is cache warming, and one more
discarded step removes it. The two look alike in a summary statistic (both make a
three-step mean untrustworthy) and are opposite in what to do about them.

Finally, the mitigation A/B is consistent with all of the above. On gpt-oss the
`hsa_no_scratch_reclaim` cell landed at TTFT 66.7 vs 67.1 baseline, TPOT 7.49 vs
7.61, throughput 1008 vs 994 — the doc calls it "within noise of baseline, as it
was at every smaller size", and the size of that "noise" is the 0.3–2.6% band
above, not the 59% one.

## Per-metric: gate, record-only, or never

Direction is fixed by `_METRIC_POLICIES` in `scripts/ci/eval_lib.py` (`max` for
latencies, `min` for throughputs). What this table decides is which of them get a
*bound* on the first bless.

Only that. The verdicts below are about the nightly's blessed baselines, which
are derived by applying a margin to an observed value, and they say nothing
about the workload's own `gates:` block (`_GATE_SPECS` in
`src/aorta/workloads/tokenspeed_serve.py`), which enforces absolute numbers a
recipe writes out per trial. The two differ exactly where the margin does the
damage: `max_p99_itl_ms: 50` is a stated ceiling on tail stalls and is a
perfectly good gate, whereas a *baseline* ITL ceiling is `observed × 1.25`,
which for an observation near zero is near zero. So "Never" in this table means
"never armed automatically from a baseline", not "not gateable" — and a metric
listed here as record-only can still carry a hand-written per-trial bound today.

> **Revised by measurement.** The four-gate set below was derived before the
> step-0 excursion was measured on the smoke cell. Three of those four fire on
> it. The `First bless` column now reflects that; the `Why` column keeps the
> original reasoning, because it was not wrong about the noise — it was wrong
> about `warmup_steps: 1` making the excursion unreachable.

| Metric | Policy | First bless | Why |
|---|---|---|---|
| `median_tpot_ms` | max | **Gate** | Reproduced to 1.57% and 0.26% across sweeps, 3.08% over 12 same-node cell-runs, and the docs already name it "the better-behaved per-token metric". It is steady-state decode cost with no queueing term, which is why it is the tightest number we have — and it is measured between tokens, so the step-0 compile excursion does not enter it (1.01× on the excursion run). The one gate the measurement leaves standing. |
| `p99_itl_ms` | max | **Gate** | Promoted from record-only for the same reason: 1.00× on the excursion run, 5.43% clean spread. It is the useful half of the ITL pair, it catches tail stalls that a median cannot, and it is the only other metric whose definition excludes the excursion. Gating it and `median_tpot_ms` together covers per-token latency at both the centre and the tail without touching anything duration-derived. |
| `mean_step_time_ms` (`step_time_ms.max`) | max | **Record-only** (was: gate) | The bench step `duration`, and therefore the metric the excursion hits hardest: 2825 ms against a 1462 ms ceiling, 2.4× over. It is still the bound `--perf-gate` always writes, so arming it is the *default* — which is exactly why the rollout sequence below has to prune it by hand until `warmup_steps` is proven to cover the excursion. |
| `output_throughput` | min | **Record-only** (was: gate) | Reproduced to 2.63% and 0.30%, and it is the headline number — but it is tokens over the step duration, so the excursion drags it to 0.73× clean and through a 0.85 floor. Nothing is wrong with the metric; it simply cannot be gated while a five-second compile can land inside the window it divides by. |
| `median_ttft_ms` | max | **Record-only** (was: gate) | The original entry said "if one gate flaps, expect it to be this one", and that was right for the wrong reason — not a flap but a 10.26× excursion, 465.30 against a 58.59 ceiling. It carries queueing delay the others do not, and its first request is the one that waits for the compile. |
| `p99_ttft_ms` | max | **Record-only** | A p99 over 32 requests is the 32nd of 32 order statistics — effectively the maximum, and we have no repeat measurement of it. The load sweep shows it moving 194 → 2139 ms across shapes and 315 → 426 ms for a 2× concurrency change, so it is responsive to things a gate should not fire on. Promote on evidence from the record-only window. |
| `p99_tpot_ms`, `p99_e2el_ms` | max | **Record-only** | Same order-statistic argument, same absence of repeat data. (`p99_itl_ms` was in this row and has been promoted to the gate set above — it now has 13 same-node cell-runs behind it, not zero repeats, and it is one of the two metrics the step-0 excursion leaves alone.) |
| `median_e2el_ms` | max | **Record-only** | End-to-end latency is TTFT plus OSL × TPOT, so gating it adds a third alarm for an event two gates already catch, and its reason line is the least specific of the three. |
| `request_throughput` | min | **Record-only** | Under `ignore_eos` at fixed OSL it is `output_throughput / output_len` — fully determined by a metric already gated. |
| `total_token_throughput`, `tokens_per_sec` | min | **Record-only** | Restatements of `output_throughput` at fixed ISL/OSL (`tokens_per_sec` is documented as an alias of it). Gating them costs nothing but produces three reason lines for one event, which makes triage slower rather than safer. |
| `median_itl_ms` | max | **Never** | Measured at ~0, because the gateway delivers several tokens per SSE chunk. A margin is multiplicative, so an observation of 0.0 blesses a ceiling of 0.0 and every later run with any inter-token gap at all fails. `eval_lib` already warned about this in a comment; it is now enforced by `_NO_AUTO_GATE`. |
| `server_startup_sec` | — | **Never** | 189–379 s in the docs; **180–415 s measured** over the 13 smoke cell-runs, so the spread is wider than the quoted one, not narrower. It does not order by model size. Not in the allowlist and must stay out: the allowlist is what `--perf-gate` arms from, so adding it *is* gating it. |
| `container_elapsed_sec` | — | **Never** | Dominated by bring-up; same argument. |
| `duration`, `total_input_tokens`, `total_output_tokens` | — | **Never** | Work-done counters. The token totals are pinned by the recipe, so a bound on them restates the configuration rather than measuring the stack. |
| `completed_total`, `failed_total` | — | **Never** | Already enforced, harder, elsewhere: the bench script and the workload independently require `failed == 0` and `completed == num_prompts`, so a shortfall fails the *cell* and reddens the nightly with no baseline involved. A metric bound here would be a third and weaker copy of a check that already fails closed. |
| `max_output_tokens_per_s`, `max_concurrent_requests` | — | **Never** | Single-sample maxima — the noisiest available summary of a distribution. |
| `mean_*_ms`, `std_*_ms`, `p50_*`, `p90_*` | — | **Never** | Deliberately absent from the allowlist already ("means are the noisiest summary of a latency distribution and the least useful thing to gate on"). Keep them absent. |

Two gates, then — `median_tpot_ms` and `p99_itl_ms`, the per-token pair — and
everything else charted, including three metrics that would have been gated
before the excursion was measured. That is a deliberately smaller first bless
than the plan originally proposed: the two that remain are the two whose
definitions exclude the failure mode we can actually demonstrate, and the three
that were dropped can be promoted from the record-only window as soon as it shows
the excursion is gone. The gap between "gated" and
"invisible" is covered by the dashboard's *What changed* view, which reports any
metric that moved more than 10% between the two most recent runs without failing
the job. A 10% throughput drift is real, is not a gate breach under these
margins, and is exactly what that view exists to surface.

## How many record-only runs, and what the threshold comes from

**Take ten nightlies before blessing. Derive each bound from the window's
extremum — the maximum for a `max` metric, the minimum for a `min` metric — not
from its mean, and not from one night's observation.**

The statistic matters more than the count, so take that first. Deriving from the
mean is the intuitive choice and it is wrong here, because the variation we have
measured is not jitter around a centre. The startup series is 189 against a
cluster of 276–379: a single environmental excursion, not a spread. A mean is
precisely the statistic that hides one, and a bound built on it is breached by
the next occurrence. The extremum is the statistic that asks the question we
actually care about — *how bad has a healthy night ever been?*

That also fixes the current tooling's real defect, which is not the margin size
but the anchor. `refresh_baselines.py --perf-gate` derives `value × 1.25` from
whatever single run was under way, so the bound depends on which night the
operator pressed the button. Feeding the seven known startup observations through
`compare_to_baseline` shows what that costs:

| Blessed on | Ceiling (×1.25) | Later runs breaching |
|---|---|---|
| 189 s | 236.2 | 6 of 6 |
| 276 s | 345.0 | 1 of 6 |
| 285 s | 356.2 | 1 of 6 |
| 291 s | 363.8 | 1 of 6 |
| 316 s / 319 s / 379 s | 395.0 / 398.8 / 473.8 | 0 of 6 |

Four of the seven possible blessing nights produce a gate that breaches, and one
of them reddens every subsequent run. The window maximum (379 → 473.8) breaches
none — but only because the window contained the excursion, which is the argument
for the count.

Ten is chosen from that, not from convention. Scoring every *n*-run subset of the
series against the runs it did not contain gives the residual breach rate on an
unseen night:

| Window size | Mean breach rate on unseen runs | Worst window |
|---|---|---|
| 1 | 21.4% | 6 breaches |
| 2 | 5.7% | 1 breach |
| 3 | 2.9% | 1 breach |
| 4 | 1.0% | 1 breach |
| 5 and up | 0% | none |

The nightly runs roughly 250 times a year, so one false alarm per quarter needs a
per-run breach rate under about 1.1% — which this series reaches at n=4, on the
noisiest metric the workload produces. The metrics we are actually gating are
10–16× tighter than that one. Zero at n=5 is an artefact of a seven-point sample
rather than a real floor, so the honest reading is that the risk is small by 5
and the remaining reason to go further is calendar coverage: ten nightlies is two
full weeks, long enough to contain a runner reimage, a Docker Hub re-pull, a
cold HF cache or a Dependabot ROCm bump — the once-a-week-ish events that produce
excursions in the first place. **Five is the floor; below it the breach rate on
known-noisy data is 2.9% per run, about nine false alarms a quarter.**

### The extremum anchor has no good answer on a bimodal cell

That whole derivation assumes the window's extremum is a *healthy worst case*.
The step-0 excursion breaks the assumption, and it is worth being explicit about
why, because it is the reason the gate set shrank rather than the window growing.

With a bimodal cell the ten-night window either contains an excursion or it does
not, and both outcomes are bad:

| Window | `median_ttft_ms` anchor | Ceiling | Consequence |
|---|---|---|---|
| No excursion (12 of 13 nights) | 46.87 | 58.59 | The excursion, when it lands, reads as a 10× regression. False alarm. |
| Contains one (1 in 13) | 465.30 | 581.63 | A genuine 2× TTFT regression — 47 → 94 ms — passes comfortably. No detection power at all. |

Enlarging the window does not resolve this; it only makes the second row more
likely. The extremum is the right statistic for a unimodal metric with occasional
environmental excursions, which is what the startup series is. It is the wrong
statistic for a metric with two modes, because "the worst a healthy night has
been" is not a single number any more.

So the correct response to a bimodal cell is not a cleverer threshold. It is
either to gate a metric the second mode does not reach — which is what
`median_tpot_ms` and `p99_itl_ms` are — or to remove the second mode. For the
step-0 excursion the second option is genuinely available, because the excursion
is positional: raising `warmup_steps` from 1 to 2 discards it by construction.

**That has now been done** — the recipe sets `warmup_steps: 2` — which is what
should let the three duration-derived metrics be promoted later. The cost is that
it changes the measurement: every number in this document was taken at
`warmup_steps: 1`, so none of them is a baseline any more, and the ten-night
record-only window has to be taken afresh at the new setting before anything is
blessed. That is a deliberate trade. Carrying the excursion into the window
instead would have meant either blessing a bimodal cell or spending ten nights
establishing a distribution we already intended to change.

For the concurrency-64 stall no such fix exists — it has no position to discard —
which is why that cell stays out of the nightly entirely rather than being gated
on a narrower metric.

The margins themselves need no change. `--step-time-margin 0.25` and
`--throughput-margin 0.15` are already right for these metrics, and the
simulation says so: against the measured cross-sweep spread, every one of the
twelve bless-one-run-check-the-other combinations passes. Against the window
extremum the separation is wide — worst observed TPOT noise 1.57% versus a 25%
detection threshold, a factor of 16 — while a regression the size of a genuine
stack change is caught:

| Scenario | `median_tpot_ms` vs 2.425 | `output_throughput` vs 3007.3 |
|---|---|---|
| the other measured night | 1.910 → pass | 3631 → pass |
| +10% / −10% | 2.134 → pass | 3184 → pass |
| +30% / −25% | 2.522 → **fail** | 2654 → **fail** |
| +94% / −46% (the 0.6B → 4B step) | 3.770 → **fail** | 1905 → **fail** |

Those rows are in `tests/ci/test_eval_lib.py`, run against the real comparator,
so the claim is checked rather than asserted.

## What blocks this today

The matrix entry is live, and what remains is a decision rather than an
engineering task.

`nightly_eval.py` runs *inside* the `aorta-ci-gpu` container
(`eval-reusable.yml` → `docker_cmd.sh exec`), and `tokenspeed_serve` runs the
TokenSpeed engine in a container of its own. So the entry needs a Docker client
and a route to a daemon in there. It had neither, and the workload's `setup()`
raised `'docker' not on PATH` before anything else happened. Record-only does not
help — it defers perf bounds, not failures.

Both pieces now exist, the image has been built, and a cell has been run to
completion from inside the container (see
[Verification](#verification-status)). The client half is therefore done and
proven. The daemon half is a per-lane opt-in that no lane sets, because it grants
effective root on the runner and that is the CI owner's call.

The entry lives in `entries` with `needs_docker_daemon: true`, which is what makes
that safe: on a runner without a socket it skips with the reason recorded, rather
than failing. `tests/ci/test_nightly_eval.py` covers both halves of that — the
skip when no daemon is reachable, and the run when one is — so the day a lane
flips the flag is a configuration change, not a discovery.

### The enabling change

**1. A Docker client in the CI image.** `docker/install_docker_cli.py`, invoked
from `Dockerfile.ci-gpu`, fetches the pinned static tarball, checks its sha256
before extracting, and extracts exactly one member — `docker/docker`. The same
tarball ships `dockerd`, `containerd` and `runc`; naming the member is what keeps
a daemon out of the image, and a build-time check fails the build if any of them
reaches `PATH`. The engine container is a **sibling**, started by the host daemon
next to `aorta-ci-gpu`, not nested inside it.

**2. A route to the daemon, opt-in per lane.** `docker-compose.docker-socket.yaml`
is an override, which is the mechanism the base compose already documents for
optional mounts ("Do not add a volume here"), so the default container still has
no route to the daemon. `rocm-ci-setup` takes a `docker-socket` input, default
`false`, and adds the override's `-f` only when it is `true`.

That input is reachable from a caller workflow through a matching `docker_socket`
`workflow_call` input on `eval-reusable.yml`, also defaulting to `false` and
forwarded to the setup step. The default is what matters here: `eval-reusable.yml`
is shared by the nightly and by `bump-validate.yml`, so the choice has to be the
caller's rather than the reusable workflow's — setting it centrally would grant
the socket to a PR-triggered lane as a side effect of enabling the nightly.
`bump-validate.yml` therefore pins it `false` explicitly, and
`sanitizers-nightly.yml` never sees it, using `rocm-ci-setup` directly with no
`docker-socket` argument. **No lane currently sets it `true`** — see the security
note below.

**3. `work_dir` must be an explicitly configured shared path.** This is the detail
most likely to be missed, because nothing reports it as a path problem.

With the socket mounted, the `-v` sources in the `docker run` the workload builds
are strings the **host daemon** resolves against the **host** filesystem. The
workload builds them from `work_dir` — `<work_dir>/u<uid>/{scripts,out,hf}` — using
paths as seen from *inside* `aorta-ci-gpu`. Those are different namespaces. A
missing bind source is not an error to the daemon: it **creates** the directory on
the host and mounts that. So the engine container comes up with an empty
`/ts-scripts` and dies on a missing script, or — worse — with an empty writable
`/ts-out`, and the harvest reports no results for a run that really happened.

The default `work_dir` does **not** fix this by being `/tmp/ts-work-serve` on both
sides. `/tmp` inside the CI container is the container's own `/tmp`, not the
host's, so the two are different directories that share a name — which is the
failure above with the confusing property that the path looks right in every log.

What the nightly needs, concretely:

- The socket override bind-mounts `${TS_SERVE_WORK_DIR:-/tmp/ts-work-serve}` at
  **the same path** inside the container. Source and target are identical on
  purpose, and a test asserts they stay identical; that is what makes one string
  name one directory on both sides of the boundary.
- The recipe sets `work_dir` to that same path **explicitly**. It happens to equal
  the workload default, but a default that must agree with a mount is a
  coincidence, not a configuration: change either one alone and the run breaks
  quietly.
- The path must be node-local. The workload already rejects an NFS home, for the
  root-squash reason documented in
  [tokenspeed-serving.md](tokenspeed-serving.md), and that reason is stronger
  here, not weaker.

**Measured, and more specific than the above was.** Demonstrating this on a
gfx950 node produced two failures worth writing down, because neither is what the
guidance as written would have led you to expect.

*Every* bind source must be resolvable by the daemon, not only `work_dir`. The
first attempt put `AORTA_WORKSPACE` on an autofs-mounted NFS home and the
container never started:

```
Error response from daemon: error while creating mount source path
'/home/.../wt-gating': mkdir /home/...: permission denied
```

The daemon resolves *all* `-v` sources in the host namespace, so a checkout on an
automounted or root-squashed filesystem fails the same way a `work_dir` there
would — and it fails at container create, before any of the workload's own
checks can produce a better message. On a GitHub runner the workspace is local
disk and this does not arise; on any node where `/home` is networked, the repo
has to be staged locally first. Worth knowing before debugging it as a socket
problem, which is what it looks like.

**The work root must be owned by the uid the container runs as, and the obvious
way to create it gets that wrong.** `aorta-ci-gpu` runs as root, so the workload
checks `u0` and requires the root itself to be root-owned. A `work_dir` created
by an ordinary host-side run is owned by *that* user, and the containerised run
then refuses it — the node used here already had a `/tmp/ts-work-serve` at
`uid=100550 mode=1777` left by earlier host-side sweeps, which uid 0 must reject.
The demonstration used a fresh path created *through the daemon* so it landed
root-owned, which is exactly what the workload's own error message advises an
administrator to do.

The operational trap in that is worth stating on its own: **the ten record-only
runs cannot share a `work_dir` with the containerised nightly** if they are taken
by hand as an ordinary user. Either take them as root, or give the two a separate
`work_dir` and accept the cold HF cache on the first nightly.

**The uid has to agree too**, and this is a second way the same boundary bites.
The workload uses a per-uid scratch root and then *verifies* it: `<work_dir>/u<uid>`
must be a real directory (not a symlink), owned by the running uid, and not group-
or world-writable. Inside `aorta-ci-gpu` the process is **root** (`user: root` in
the base compose), so it writes and checks `u0`. That works — a root-owned
directory created through the shared mount is root-owned on the host too, so the
host daemon and the checking process agree. But it holds only because both sides
are uid 0. Change `user:` in the base compose, run the harness as a non-root uid,
or turn on userns-remap on the daemon, and `<work_dir>/u<uid>` is created by one
uid and inspected by another: the check fails with an ownership error that does
not mention containers, uids-across-a-boundary, or the mount. Anyone changing
either should expect that error to be the symptom.

One consequence worth stating: with `run_as_current_user` defaulting true, the
engine container also runs as uid 0, so exports land on the host owned by root,
outside the workspace that `eval-reusable.yml`'s "Reclaim results ownership" step
chowns. They are cleaned per trial unless `keep_work_dir` is set, but a runner
that fills `/tmp` with root-owned scratch is a plausible future complaint.

### Security: this grants effective root on the runner

Stated plainly, because it is the part most easily lost in a diff: a process that
can talk to the Docker daemon can ask it for a privileged container that
bind-mounts `/`. That is root on the host, with no exploit involved. Mounting the
socket into `aorta-ci-gpu` therefore hands the host to anything running in that
container — including every package the nightly installs from an index.

The mitigating argument is real but partial. The container is already
`privileged: true` with `seccomp=unconfined` on a self-hosted runner, so on a
dedicated, trusted, single-tenant node this widens an already-wide posture rather
than opening a new one. The argument fails on a shared or multi-tenant runner.

**If the CI owner is not willing to accept it, the recommended alternative is to
run the serving workload outside the CI container entirely** — as a step on the
runner host, which already has a client and the socket, publishing its results
JSON into the harness's results directory. The nightly keeps the socket out of
the container, and the only cost is that this one cell is invoked differently
from the other sixteen. A socket proxy allowlisting just the container
create/start/logs/remove calls is a middle option; rootless Docker is a third.
Nested docker-in-docker is not on the list: it needs `privileged` too, so it
trades no privilege away, and it gives the engine a different filesystem view,
which breaks exactly the bind mounts described above.

This branch deliberately leaves the decision open: the mechanism is in place and
switched off, and turning it on is one flag plus a named sign-off.

### What is still missing

**One thing, and it is a decision rather than a piece of work: no lane enables
the socket.** That is the sign-off described above, and it is deliberately not
this branch's to give.

Everything else on this list has been done. The entry is therefore in `entries`
rather than `pending_entries`, carrying `needs_docker_daemon: true` — so on a
runner without the socket it **skips**, with the reason in the results, instead
of failing on `Cannot connect to the Docker daemon`. When the nightly lane sets
`docker_socket: true` the entry starts running with no further edit. Promoting it
without that flag would have been the mistake the matrix file's own header warns
about: an entry may only be added once it can actually pass on the runner.

### Verification status

Built and run on a gfx950 (MI355X) node on 2026-09-02.

**The image builds and is client-only.** `docker compose --env-file .env.ci -f
docker-compose.build.yaml build` succeeds, producing a 51.9 GB `aorta:ci-gpu`.
The build's own `RUN` checks pass — `docker client check: OK --
/usr/local/bin/docker, no daemon on PATH` — and the assertion holds when
re-checked against the built image rather than during it: `dockerd`,
`containerd`, `runc`, `containerd-shim-runc-v2`, `docker-proxy` and
`docker-init` are all absent from `PATH`, and `docker --version` reports
**29.7.2**. That is what makes "client only" a property of the artifact and not
of the Dockerfile's intent.

**The client reaches the host daemon from inside the container.** With the
override mounted, `docker exec aorta-ci-gpu docker ps` returns the running
container list, exit 0.

**A cell completes.** Two sweeps of `tokenspeed-serve-bench-smoke.yaml` were
driven from inside `aorta-ci-gpu`, each starting the TokenSpeed engine as a
sibling container: four cell-runs, all passed, `failed_total` 0 and
`completed_total` 96 throughout, engine containers cleaned up afterwards. The
twelve measured steps were 1108–1149 ms with no step-0 excursion, and every
metric landed within 5% of the host-side clean envelope — so the container
boundary does not move the measurement, which is what lets a nightly baseline be
compared against the host-side runs the variance analysis is built from.

Two configuration requirements were discovered by doing this rather than by
reading, and are written up under [work_dir](#the-enabling-change) above: all
bind sources must be resolvable by the daemon (an autofs NFS checkout is not),
and the work root must be root-owned because the container runs as uid 0.

Also verified earlier, and still true: the installer rejects a wrong sha256
without leaving a binary behind; the client negotiates against an older (29.1.3)
daemon; `docker compose config` resolves all three mounts with the scratch mount
identical on both sides.

Not verified, because it is the thing awaiting sign-off: the entry has never run
under `nightly_eval.py` on a CI runner, only under `aorta sweep run` in the
container by hand. The skip path is covered by tests rather than by a runner.

To reproduce:

```bash
cd docker
bash ../scripts/ci/docker_compose.sh --env-file .env.ci -f docker-compose.build.yaml build
docker run --rm aorta:ci-gpu docker --version

# work root must be root-owned; creating it through the daemon is the easy way
docker run --rm -v /tmp:/mnt busybox:1.37 mkdir -p /mnt/ts-work-serve

export TS_SERVE_WORK_DIR=/tmp/ts-work-serve
bash ../scripts/ci/docker_compose.sh --env-file .env.ci \
  -f docker-compose.build.yaml -f docker-compose.docker-socket.yaml up -d
docker exec aorta-ci-gpu docker ps
docker exec aorta-ci-gpu aorta sweep run \
  --recipe recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml \
  --output-dir "$TS_SERVE_WORK_DIR/out" --strict
```

## The rollout sequence

Steps 1–2 are **done**. Step 3 cannot start until a lane enables the socket,
because until then the entry skips rather than records. Everything from step 3 is
the part this document is really specifying.

**1. ~~Promote the entry.~~ Done.** It is in `entries` with `min_gpus: 1`,
`timeout_sec: 3600` and `needs_docker_daemon: true`.

`timeout_sec: 3600` rather than the 1800 default: two bring-ups at the observed
spread plus the teardown VRAM drain is around 15 minutes before the image pull
and any cold-cache weight download, and a timeout is an unconditional entry
failure rather than a slow record. Check the job's own `timeout-minutes: 150` in
`eval-reusable.yml` still has room. Measured end to end from inside the
container, a full two-cell sweep took **12 minutes** on a warm cache, so the
budget is right but most of it is bring-up.

**2. ~~Verify it before merging.~~ Done**, and rerun these after any change:

```bash
python -m pytest tests/ci -q --timeout=180
python -m pytest tests/workloads/test_tokenspeed_serve.py -q
aorta sweep run --recipe recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml --dry-run
```

**2a. Enable the socket on the nightly lane.** The plumbing is in place; the
decision is not this branch's to make — see
[Security](#security-this-grants-effective-root-on-the-runner). It is one line in
`.github/workflows/nightly-eval.yml`, plus a named sign-off:

```yaml
    uses: ./.github/workflows/eval-reusable.yml
    with:
      docker_socket: true
```

The `docker_socket` input on `eval-reusable.yml` defaults to `false` and is set
per caller, so this arms the nightly lane and nothing else. `bump-validate.yml`
sets it `false` explicitly: that lane runs PR head code on the same self-hosted
runner and must keep no route to the host daemon. `sanitizers-nightly.yml` does
not go through `eval-reusable.yml` at all — it uses the `rocm-ci-setup` action
directly and passes no `docker-socket`, so it inherits the action default of
`false`. Setting the input inside `eval-reusable.yml` instead of per caller would
hand the socket to every lane at once and is the mistake this shape exists to
prevent.

Until the sign-off lands, leave it `false`: the entry reports `skip` with
`needs a docker daemon: ...` in its reasons, which is visible on the dashboard
and costs the nightly nothing.

**3. Let it record for ten nightlies, at `warmup_steps: 2`.** It will report
`recording` on the dashboard.

> The window is only valid at the setting the gate will run at, and that setting
> is now `warmup_steps: 2` in
> `recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml`. Confirm that before
> counting nights, and restart the count if it changes underneath the window.
> None of the numbers earlier in this document can substitute for any of these
> ten: they were measured at `warmup_steps: 1`, and the change was made
> specifically to alter the behaviour they describe. Expect the window to be
> *cleaner* than the 13-cell-run table — that is the change working — but do not
> assume it; the point of the window is to measure it rather than predict it.

Watch the *Workloads* view and write the numbers down; the ten values
of `median_tpot_ms` and `p99_itl_ms` per cell are the input to step 4, and the
ten of `median_ttft_ms`, `output_throughput` and `mean_step_time_ms` are what
decides whether the three record-only metrics can be promoted later. Two cells,
so twenty observations of each. A `fail` during this window is a real failure —
the entry is unbaselined but the harness is fail-closed — and must be fixed
rather than waited out.

**4. Check the window before blessing, and check it for bimodality first.**
Compute, per cell and per metric, the extremum and the ratio of extremum to
median. Two separate checks:

- *Spread.* If either gated metric shows a window spread above about 10%, do not
  bless it — the margin was sized for a 3% spread and a 10% one means something
  is varying that we have not identified. Demote it in step 5 and investigate.
- *Bimodality.* Look at the per-step times, not only the per-cell summary. If any
  night's first measured step stands well clear of its other two, the compile
  excursion is still reachable and the three record-only metrics stay
  record-only regardless of how good their spread looks — a clean ten-night
  window over a bimodal cell is the exact situation where the extremum anchor
  gives a threshold with no detection power. See
  [the bimodal-cell section](#the-extremum-anchor-has-no-good-answer-on-a-bimodal-cell).

If the window shows no excursion in twenty cell-runs, that is reasonable evidence
it has stopped happening, and the three can be promoted with the same margins.

This window is the first measurement at `warmup_steps: 2`, so it is also the
test of whether that change did what it was supposed to. Two outcomes worth
telling apart: no excursion in twenty cell-runs is the expected result and clears
the three duration-derived metrics for promotion in step 7; an excursion that
still appears at position 0 means two discarded steps are not enough to cover
the compile, which is new information and should be investigated rather than
absorbed by raising `warmup_steps` again.

**5. Bless, scoped to this entry only.**

```
Actions -> Refresh baselines -> Run workflow
```

with the dispatch form filled in as

| Input | Value |
|---|---|
| `perf_gate` | `true` |
| `perf_gate_entry` | `tokenspeed_serve_smoke` |
| `step_time_margin` | `0.25` (default) |
| `throughput_margin` | `0.15` (default) |

which the workflow turns into

```bash
python scripts/ci/refresh_baselines.py --perf-gate \
  --perf-gate-entry tokenspeed_serve_smoke
```

The scope is not optional. Without it the same PR arms step-time ceilings for
every other entry in the matrix from that one run, so leaving `perf_gate_entry`
empty with `perf_gate: true` logs a warning on the job. It is a warning rather
than a hard failure because an unscoped refresh is the legitimate end state once
every workload has variance data behind it — but during rollout it is not what
you want. A misspelled entry name is rejected rather than silently scoping to
nothing, and setting `perf_gate_entry` without `perf_gate` fails immediately
instead of after the wheel install.

`perf_gate_entry` accepts several names, comma- or space-separated, mapping to
one `--perf-gate-entry` each. For this rollout it should be exactly one.

**6. Correct the PR diff by hand, then merge.** The refresher derives bounds from
the *single* run it just did, and from every auto-gateable metric it observed.
Two edits are needed, both mechanical:

- Replace each bound with one derived from the ten-run window: `max × 1.25` for a
  `max` metric, `min × 0.85` for a `min` metric.
- Delete the keys this document lists as record-only — `p99_ttft_ms`,
  `p99_tpot_ms`, `median_e2el_ms`, `p99_e2el_ms`, `request_throughput`,
  `total_token_throughput`, `tokens_per_sec`, and now also `median_ttft_ms`,
  `output_throughput` **and the `step_time_ms.max` entry** — leaving
  `median_tpot_ms` and `p99_itl_ms`. `median_itl_ms` will not be there;
  `_NO_AUTO_GATE` keeps the refresher from writing it.

Deleting `step_time_ms.max` is the one that needs attention, because it is the
bound `--perf-gate` always writes and the only one on this list that is not a
metric key. Leaving it in by inattention arms the gate the measured excursion
breaches hardest — 2825 ms against a 1462 ms ceiling. It is the most likely
mistake in this whole sequence.

This hand-editing is the honest cost of the current tooling, and it is bounded:
ten keys across two cells, on a PR a human reviews anyway, and it is meant to
shrink as the record-only metrics are promoted on evidence. It got larger rather
than smaller when the excursion was measured, which strengthens the case below
for a per-metric scope.

**7. Promote the record-only metrics on evidence, not on schedule.** After
another ten nightlies under the live gate there are twenty more observations.
Take them in two groups, because they are blocked on different things:

- `median_ttft_ms`, `output_throughput` and `step_time_ms.max` are blocked on the
  step-0 excursion, not on their spread — which is already good enough. The
  `warmup_steps: 2` change is intended to remove the blocker, so promote them
  once the step-3 window shows twenty cell-runs with no excursion. That window
  *is* the re-taken one; there is no separate re-take to schedule.
- The remaining `p99_*` metrics are blocked on having no repeat data. Promote any
  whose window spread is comparable to its median's. `p99_ttft_ms` gated is worth
  more than `median_ttft_ms` gated, because tail latency is what a serving
  regression damages first — we just have no basis for a bound on it yet.

If the hand-editing recurs every refresh rather than converging, the fix is a
per-metric scope alongside `--perf-gate-entry`. That is the point to build it —
not before, when the right metric set is still a guess.

## When a gate fires

The alert names the cell, the metric, the observed value and the bound. Three
things it can be, and they are distinguishable without a GPU:

**A stack change.** Check the dashboard's *What changed* line first: it reports
whether PyTorch, ROCm or HIP moved that night. If one did, and the direction of
the metric move is plausible for it, this is a stack effect and the answer is
either an upstream bug report or a re-bless. Do not re-bless silently — a bump
that costs 20% of decode throughput is exactly the result the nightly exists to
produce, and `docs/ci-nightly-eval.md` is explicit that the baseline diff *is*
the bump's impact.

**A noise event.** The two cells of this recipe differ only by mitigation and run
minutes apart on the same node, which is the reason this recipe was chosen. Use
it: if `baseline` and `no-scratch-reclaim` both moved by a similar amount, the
node or the stack changed. If exactly one moved, and the mitigation has never
shown an effect before at any model size, that is a single-cell excursion — a
co-tenant, a thermal event, a cold cache. Confirm by looking at the previous
nights on the dashboard's run history; a noise event is one red cell in a column
of green, a regression is a column that turns red and stays red. Also check
`server_startup_sec` for that cell: it is not gated, but a bring-up at the top of
its range is a good indicator that the node was busy.

**A real regression.** Everything else — both cells moved, no toolchain change,
and the next night reproduces it. Reproduce locally with the recipe as committed
and bisect against the pinned image digest. Note that the image is pinned by
digest precisely so this case cannot be caused by TokenSpeed changing underneath
the baseline.

If you cannot tell which of the three within one working day, revert to
record-only and keep collecting. An unexplained gate is worth less than the
`recording` state it came from.

## Reverting to record-only

Delete the `metrics` and `step_time_ms` keys for the affected cells from
`config/ci/regression_baselines.yaml`, leaving `passed: true`. That returns those
cells to record-only immediately — the comparator treats an absent bound as
nothing to check — while keeping correctness gating and the fail-closed behaviour
intact. It is a small, reviewable, obviously-correct diff, which is what you want
at the point where a gate is misbehaving.

Do **not** revert by deleting the matrix entry. That stops the measurement as
well as the gate, and the record-only data is the thing needed to size a better
bound.

Do **not** revert by widening the margin as a first move. A margin wide enough to
absorb an unexplained excursion is usually too wide to catch a regression, and
the widening tends to be permanent. Go back to record-only, find out what moved,
then re-bless from a longer window.

One whole-file caveat: any later `refresh-baselines` run rewrites
`regression_baselines.yaml` completely, so a hand-reverted cell will be re-armed
by the next unscoped `--perf-gate` refresh. Scope those runs with
`--perf-gate-entry`, or check the diff for cells you had deliberately demoted.

## Assumptions

The user referenced `Aorta planning & Updates.docx`, which is not reachable from
this machine. Everything above was derived from the repository, and where that
was not enough a decision was made and is recorded here.

- **Nightly cadence and volume.** ~250 nightly runs a year, and one false alarm
  per quarter is the tolerable rate. This sets the window length; a stricter
  target needs a longer window, not a wider margin.
- **The recipe.** `tokenspeed-serve-bench-smoke.yaml` was chosen over the other
  four serving recipes as the cheapest that still answers something — Qwen3-0.6B
  is ~1.2 GB of weights against ~40 GB for the gpt-oss pair, and two cells rather
  than four or six. The tie-breaker was triage rather than cost: it is the only
  serving recipe whose cells differ *only* by mitigation, so the pair is a
  same-night control. `tokenspeed-serve-gptoss.yaml` is the recipe whose numbers
  matter most (it is TokenSpeed's canonical AMD benchmark) and is the right
  second entry once this one has been gated for a month.
- **Cost.** Measured at **12 minutes** of runner time per nightly for the two
  cells on a warm HF cache, driven from inside the CI container — close to the 15
  minutes originally estimated. Not separately budgeted with anyone.
- **`--perf-gate-entry` over hand-pruning.** Scoping was added as code because
  the alternative recurs on every refresh, for sixteen cells belonging to other
  people's workloads, in a diff where the omission looks identical to the
  inclusion. The per-metric pruning in step 6 was left manual for the mirror-image
  reason: it is ten keys in one entry, on a PR a human reviews anyway, and
  building a flag for it now would fix a metric set we are explicitly planning to
  change.
- **`needs_docker_daemon` over leaving the entry staged.** A launch was
  demonstrated, so the entry has earned promotion; but the socket is still off by
  design, and an entry in `entries` that cannot reach a daemon fails every night.
  Rather than choose between a stale `pending_entries` row and a red nightly, the
  capability was made declarable, exactly as `min_gpus` already is for GPU count.
  The cost is one field and one probe; the alternative was for the promotion to
  wait on a security decision it does not actually depend on.
- **`warmup_steps` raised to 2.** Reversed from "left at 1" earlier in this
  branch. The step-0 excursion is positional, so one more discarded step removes
  it by construction, and it is measured at 1 cell-run in 13 — frequent enough
  that a ten-night window taken at 1 would probably contain one and produce
  either a false alarm or a threshold with no detection power. The objection to
  bundling it was that it changes the measurement and invalidates every number
  here; that is true and is now the accepted cost, because those numbers were
  never going to be the bless baseline — the window is, and the window has not
  been taken yet. Changing the setting *before* the window costs nothing;
  changing it after would have cost ten nights. The variance table is kept as
  the rationale.
- **`median_itl_ms`.** Kept in the allowlist and excluded from auto-blessing,
  rather than removed. Removing it would make a legitimate hand-written bound
  impossible; the problem is only ever with a bound derived from a margin.
- **TP and multi-GPU serving.** Out of scope. TP=4 does not come up on this
  image, and TP=2 buys 4.5% throughput for a second GPU, so neither is worth a
  gate before the single-GPU one has proven itself.
