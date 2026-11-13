# Rewards: the building blocks and how to compose them

A reward is two things: the checks that run on an answer (each check of one candidate is a *leg*, run by a *check
adapter*), and a *reward function* that reads the legs' results and gives a number. This page lists every check
adapter and reward function that exists, built in or in the examples, and shows how to put them together in a profile
and a run config, and how to write a new one. The contract behind it is in [ADAPTING.md](ADAPTING.md) ("Your own
reward function", "Checks and profiles"); paths are relative to `aorlta/`.

What every leg result says (`LegResult` in `src/aorlta/schemas.py`):

| state | meaning |
|---|---|
| `clean` | the check passed |
| `finding` | a sanitizer found a defect, with the findings to back it |
| `fail` | the check failed and blames the candidate, with a `reason` |
| `unsettled` | the check could not decide (an error, a timeout the adapter cannot blame on the candidate, missing inputs); never clean |
| `not_run`, `skipped_by_plan` | the leg did not run (a leg it `requires` was not clean, or the planner skipped it) |

What every reward function keeps (the service checks it and unscores anything else, with an alarm): the reward is
`None` (unscored, with a `reason`) or a float in [-1.55, 1.0] whose parts are exactly `base`, `broke_healthy` and
`length` and add up to it; a reward that claims to be clean has every profile leg clean; an unscored sample is never
counted clean; `decide` on a partial record never gives an answer `compute` on the full record would contradict.

## The check adapters

The name is what a profile's pool sets as `adapter:`; the leg is what its stage lists in `legs:`.

### Format and parse

| adapter | leg | needs | states and reasons |
|---|---|---|---|
| `standin.parse` (built in, `src/aorlta/reward/adapters/standin_parse.py`) | `gate.parse` | nothing (CPU) | `clean`: the last ```` ```cpp ```` block of the reply, written to the job's directory as `source.hip`; `fail` `parse`: no such block; `unsettled` `adapter_error` |

The rollout has parses of its own (`rollout.parse_adapter`), which decide whether an answer ends the chain or gets a
repair turn; they are not legs. `standin.parse` takes the last ```` ```cpp ```` block; `hipk.parse`
(`examples/hip-kernels/hip_kernels.py`) also rejects a block that does not define the task's kernel (`no_code_block`,
`no_kernel`), and `qa.parse` (`examples/qa/qa_task.py`) a reply with no `Answer:` line (`no_answer`). A rejected
answer gets the parse's message as a repair turn (the feedback template's `build_apply`).

### Compiles

| adapter | leg | needs | states and reasons |
|---|---|---|---|
| `standin.hipcc` (built in, `adapters/standin_hipcc.py`) | `gate.compile` | `hipcc`, `llvm-readelf` (settings `hipcc`, `readelf`); CPU only, the offload arch is always explicit (leg settings `offload_arch`, `flags`, which must hold `-c`) | `clean`, with a trusted build fact (the launched kernels' LDS) when it can be read; `fail` `compile_error`, with the error count and their classes, never their text; `unsettled` `timeout`, `worker_died`, `adapter_error` (a launch failure, a crash, a kill) |

### Matches a reference

| adapter | leg | needs | states and reasons |
|---|---|---|---|
| `hipk.run` (`examples/hip-kernels/hip_kernels.py`) | `ref.native` | `hipcc` and a GPU per worker (settings `hipcc`, `gpus`: worker n uses `gpus[(n - 1) % len(gpus)]`); leg settings `target`, `run_timeout_s`; the task's spec in the pool's `tasks.json` | builds the candidate with a generated host program, runs it once and compares every output element with a host reference, and every guard element past an array's end. `clean`: every element within the task's `atol + rtol·|want|`, nothing written past an end; `fail` `wrong_output` (a mismatch or a write past the end), `crash` (a signal, a HIP error), `timeout` (past `run_timeout_s`), `bad_signature` (the harness does not build and the compiler names the kernel or `main`); `unsettled` `harness_build_failed`, `run_not_started`, `no_result`, `no_time_left`, `adapter_error` |
| `qa.match` (`examples/qa/qa_task.py`) | `ref.answer` | nothing (CPU); optional `judge`: a command, and `judge_timeout_s` | see "Exact or normalised match" |
| `hipk.time` (`examples/hip-kernels/hip_kernels.py`) | `ref.speedup` | `hipcc`, `llvm-readelf` and a GPU of its own | see "Timing" below |

### Exact or normalised match

| adapter | leg | needs | states and reasons |
|---|---|---|---|
| `qa.match` (`examples/qa/qa_task.py`) | `ref.answer` | nothing; optional `judge` (a command as a list) | the reply's last `Answer:` line, normalised (NFKC, lower case, outer quotes and backticks, a trailing full stop, commas between digits and runs of space removed), against the reference and its aliases. `clean`: it matches, or the judge prints `{"verdict": "correct"}`; `fail` `wrong_answer` (or the judge says `wrong`), `no_answer` (no `Answer:` line); `unsettled` `judge_no_verdict` (the judge gave anything else, failed or timed out), `adapter_error` |

The judge reads `{"question", "reference", "answer"}` as JSON on stdin and is asked only about an answer that does not
match. That is also the shape of a reward that is an external tool's verdict: one leg that calls the tool, with no
verdict unsettled.

### Sanitizer verdicts

These run aorta's sanitizer backends (a checkout named by `aorta_src`) and map aorta's verdict: PASS is `clean`, FAIL
(and waitcheck's WARN) with findings is `finding`, and ERROR, NOT_CHECKED, TIMED_OUT, or a verdict with no findings
to back it is `unsettled` with reason `checker_not_settled:<state>/<verdict>[: <aorta's reason code>]`. Findings carry
codes, severities and kernel names, never message text.

| adapter | leg | needs | also unsettled |
|---|---|---|---|
| `standin.rj_waitcheck` (`adapters/standin_rj_waitcheck.py`) | `san.waitcheck` | CPU; rocjitsu's `rj_waitcheck` (`binary`), a libstdc++ with GLIBCXX_3.4.31 (`ld_library_path`), aorta (`aorta_src`), `hipcc` for a template kernel's instantiation; leg settings `target`, `max_diagnostics` | `harness_unsupported: ...`, `harness_build_failed` (a template kernel with no instance), `worker_died`, `adapter_error`. A capped scan is still a `finding` (with `diagnostics_truncated`) |
| `standin.aorta_consan` (`adapters/standin_aorta.py`) | `san.consan` | a GPU; rocjitsu's DBI hook (`hook`), aorta, `hipcc`; a trusted control (`control_source`, with `controls: true`) | `consan_preflight_not_settled:<verdict>`, `check_timeout: ...` (past `reward.gpu_check_s`), `harness_unsupported`, `harness_build_failed`, `worker_died`, `adapter_error`. NOT_CHECKED (nothing to watch: no LDS, barrier, atomic or fence) is unsettled |
| `standin.aorta_asan` (`adapters/standin_aorta.py`) | `san.asan` | a GPU, the ROCm ASan runtime, aorta, `hipcc` (built `xnack+ -fsanitize=address`); a trusted control; batched (`batched: true`, up to `reward.asan.max_packages` per request) | as ConSan; a PASS on a node whose canary is NOT_VERIFIED is NOT_CHECKED in aorta, so unsettled |

A kernel-only source (no `main`) gets the stand-in launch harness (`adapters/standin_harness.py`), one run per kernel:
the leg is a finding if any kernel's run found something, clean only if every run is clean, and unsettled otherwise.
The GPU legs run only inside a Slurm job (`jobs/aorlta-run.sbatch`, `jobs/aorlta-reward-standin.sbatch`).

`fake` and `fake-asan` (`adapters/fake.py`) are a deterministic fake of every leg, with injected latencies, failures,
stalls and dirty controls, for testing the service; they are never a training result.

### Timing

`hipk.time`, leg `ref.speedup`, times a correct candidate against its task's reference kernel
(`reference/<id>.hip` in the pool) and asks whether it is faster by a required ratio. The statistics are generic, in
`src/aorlta/reward/timing.py`; the adapter is the hip-kernels example's, and is the pattern for another task's.

How it measures. The candidate is compiled on its own (`launcher_source`: its source, then one C function that
launches its kernel with the task's grid and block), so nothing it defines, a macro included, reaches the timing code.
Its object's symbol table (`llvm-readelf --syms`) must not define `main`, a `hip*`, `hsa_*` or `__hip*` name, or any
function with C linkage other than the launcher and a kernel stub: such a symbol would take the place of the HIP
runtime's own, `hipEventElapsedTime` say, in the linked program. The harness (`timing_source`) holds the reference
kernel and the host program. It runs each kernel once on fresh inputs and compares the outputs with the host
reference; times one batch of each and stops if the projected GPU time is over `gpu_budget_s`; then takes `warmup +
repeats` rounds, each a fresh checked run of the candidate and one timed batch (hipEvent around `launches`
back-to-back launches) of each kernel, the order alternating by round, the inputs reset before each batch.

How it decides (`timing.judge`). The first `warmup` samples of each side are dropped. Of the rest: each side's median
and relative MAD (`median(|x - median|) / median`); `noisy` if either relative MAD is above `max_rel_mad`; the noise
floor `max(min_noise_floor, MAD_candidate + MAD_reference)`; the speedup `s = median_reference / median_candidate`.

| outcome | leg |
|---|---|
| `s >= ratio * (1 + floor)` | `clean`, with a measurement record (kind `measurement`: the speedup, the ratio, the floor, the medians and MADs) |
| `s <= ratio * (1 - floor)` (measured, correct, not faster by the ratio) | `fail` `not_faster`, with its record |
| in between | `unsettled` `timing_within_noise` |
| either side too spread | `unsettled` `timing_noisy` |
| fewer than `repeats` samples, or one that is not a positive finite number | `unsettled` `timing_too_few`, `timing_invalid` |
| wrong output under the timing harness | `unsettled` `timing_wrong_output` (`reference_wrong_output` for the reference) |
| over the GPU budget, a timeout, a crash, a HIP error | `unsettled` `too_slow_to_measure`, `timing_timeout`, `timing_crash`, `timing_hip_error` |
| the harness does not build, a reserved symbol, settings that cannot decide | `unsettled` `harness_build_failed`, `symbol_check_failed` or `defines_reserved_symbol`, `adapter_error` |

Correctness comes first twice over: the profile runs `ref.speedup` only after `ref.native` is clean (`requires`), and
the harness checks the candidate's output on fresh inputs before every round. Nothing the timing cannot decide is
clean. What the timing does not see: a kernel that behaves differently on repeated launches without a reset (a timed
batch's later launches run on the outputs of the earlier ones, unchecked), or one that is correct at the task's size
and not at the timing size (the harness checks it at the timing size, so that is unsettled, not clean).

Leg settings (`examples/hip-kernels/hipk-speed.yaml`): `ratio` 1.1, `warmup` 5, `repeats` 30, `launches` 20,
`max_rel_mad` 0.05, `min_noise_floor` 0.02, `gpu_budget_s` 10, `run_timeout_s` 60, `target`, and `scalars`, the
problem sizes for the timing by the tasks' scalar names (`n` = 2^24 + 3, `rows` × `cols` = 3001 × 5171). They are part
of the cache key, so changing one re-times. Give the timing worker a GPU of its own, apart from the engines and the
correctness runs.

The sizes matter. Measured on one MI350X with these settings, a check takes 2 to 3 s:

| | median per launch | relative MAD | speedup | leg |
|---|---|---|---|---|
| each of the 8 reference kernels against itself | 28 to 51 µs | 0.06% to 0.6% | 0.999 to 1.002 | `fail` `not_faster` (`within_noise` at ratio 1.0) |
| `vector_add`, thread 0 of each block doing its 256 elements | 802 µs against 33 µs | under 0.4% | 0.042 | `fail` `not_faster` |
| `transpose`, one thread row of each 16 × 16 block working | 97 µs against 51 µs | under 0.3% | 0.53 | `fail` `not_faster` |
| the real kernels against those slow ones as references | | under 0.3% | 25.9 and 1.90 | `clean` |
| `vector_add` and `transpose` at the tasks' own sizes, against themselves | 2.6 µs | 5% to 6% | | `unsettled` `timing_noisy` |

At the tasks' own sizes (n = 100003) a launch is about its own overhead, so the times barely depend on the kernel and
spread by about 5%.

## The reward functions

The name is what `reward.reward_fn` (and the profile's `reward_fn:`) sets. Every one adds the length term, `-0.05 ×
response_tokens / max_new_tokens`, to its base; the tables give the base.

### `standin.v11` (built in, `src/aorlta/reward/rewardfn/standin_v11.py`)

Reads the nine `gate.*` legs, `ref.native`, `ref.edge` and the three `san.*` legs, in that order (Appendix B's
ladder). Scores `standin-static`, `standin-gpu` and `fake`.

| the first leg that decides | base |
|---|---|
| a gate `fail` | -1.0 (-0.9 for `compile_error`); an unknown gate reason is unscored |
| `ref.native` / `ref.edge` `fail` `wrong_output` / `edge_mismatch` | -0.7 |
| a `fail` `crash` / `fault` / `timeout` whose trusted controls were clean | -0.8 (unscored if the bracket was not clean; a ConSan-only timeout leaves its property unscored) |
| a confirmed sanitizer finding | -0.5 × (sanitizers that found / sanitizers that ran), each weight 1.0 |
| everything clean, with enough clean repeats and a clean compile | +1.0 for a frozen reference, +0.5 otherwise |

Then -0.5 (`broke_healthy`) if the task's parent was healthy and this answer is not clean. An unsettled gate or
reference leg leaves the sample unscored unless a sanitizer finding is confirmed; so does a sanitizer that did not
settle (`sanitizer_not_settled`), and a clean that is not proven (`clean_not_proven`). Clean: a positive base.

### `hipk.v1` (`examples/hip-kernels/hip_kernels.py`)

Reads `gate.parse`, `gate.compile`, `ref.native`; profile `hipk-gpu`.

| outcome | base |
|---|---|
| `gate.parse` fails | -1.0 |
| `gate.compile` `compile_error` (another compile failure) | -0.9 (-1.0) |
| `ref.native` `bad_signature` | -0.8 |
| `ref.native` `crash` or `timeout` | -0.6 |
| `ref.native` `wrong_output` | -0.5 |
| every leg clean | +1.0 |

Any other state or reason (unsettled, a finding, not run, an unknown `fail` reason) is unscored. Clean: a positive
base.

### `qa.v1` (`examples/qa/qa_task.py`)

Reads `ref.answer`; profile `qa-match`. +1.0 clean, -0.5 `wrong_answer`, -1.0 `no_answer`; anything else unscored.

### `WithSpeedup` and `hipk.speed` (timing)

`WithSpeedup(inner, name=..., leg="ref.speedup")` (`src/aorlta/reward/rewardfn/speedup.py`) wraps any reward function
and adds the timing leg. Whatever the inner function scores as not clean stands as it is (a wrong kernel is never
timed); a clean inner reward needs the timing leg and its base is replaced by the measured speedup's grade
(`timing.speedup_base`; `broke_healthy` and `length` stay the inner function's):

| the timing leg | base |
|---|---|
| `clean`: faster than the reference by `ratio`, past the noise floor | `0.75 + 0.25 × min(1, log2(s / ratio))`: 0.75 just past the threshold, 1.0 at twice the ratio |
| `fail` `not_faster`: correct, measured, not faster by the ratio | `0.5 × s / ratio`: below 0.5, so above every wrong answer and below every one that clears the ratio |
| anything else, or a record that contradicts the state | unscored (`ref.speedup:<reason>`, `ref.speedup:no_measurement`) |

Clean: a base of at least 0.75. An inner reward that is not clean but has a base of 0.75 or more is unscored
(`inner_base_overlaps_clean`), so that this holds for every reward it passes through; hipk.v1 and standin.v11 have
none. `hipk.speed` is `WithSpeedup(HipkV1(), name="hipk.speed")`, profile `hipk-speed`.

A measured, correct, slower kernel is settled and never clean: its score is positive and grows with its speed, so a
group of correct answers still gets different rewards, and GRPO can learn from them which is faster.

## Composing them

The checks are a profile; the run config names the profile and the reward function, and lists the plugins that
register any name either uses. The hip-kernels example with the timing leg, `examples/hip-kernels/hipk-speed.yaml`:

```yaml
name: hipk-speed
reward_fn: hipk.speed            # must equal the run config's reward.reward_fn
clean_repeats: 1                 # settled-clean repeats a clean reward needs
materialize: true                # write each candidate's files: real adapters read them
pools:                           # one worker pool per check: its adapter, resource, timeout name and workers
  gates: {resource: local_cpu, adapter: standin.parse, timeout: gate, workers: 4}
  compile: {resource: local_cpu, adapter: standin.hipcc, timeout: compile, workers: 8}
  run: {resource: own_gpu, adapter: hipk.run, timeout: sim_run, workers: 3}
  time: {resource: own_gpu, adapter: hipk.time, timeout: sim_run, workers: 1}
stages:                          # run in order, each only while the reward function answers NeedMore
  - {name: S0, pool: gates, legs: [gate.parse]}
  - {name: S1, pool: compile, legs: [gate.compile]}
  - {name: S2, pool: run, legs: [ref.native]}
  - {name: S3, pool: time, legs: [ref.speedup]}
requires:                        # a leg runs only after these are clean; otherwise it is not_run
  gate.compile: [gate.parse]
  ref.native: [gate.compile]
  ref.speedup: [ref.native]
leg_settings:                    # given to the adapter with each job, and part of the cache key
  gate.compile: {offload_arch: gfx950, flags: ["-c", "-O1"]}
  ref.native: {target: gfx950, run_timeout_s: 20}
  ref.speedup: {target: gfx950, ratio: 1.1, warmup: 5, repeats: 30, launches: 20, max_rel_mad: 0.05,
                min_noise_floor: 0.02, gpu_budget_s: 10, run_timeout_s: 60,
                scalars: {n: "(1 << 24) + 3", rows: "3001", cols: "5171"}}
adapters:                        # each adapter's own settings; ${AORLTA_ROCM_ROOT} is that variable, or /opt/rocm
  standin.hipcc: {hipcc: "${AORLTA_ROCM_ROOT}/bin/hipcc", readelf: "${AORLTA_ROCM_ROOT}/llvm/bin/llvm-readelf"}
  hipk.run: {hipcc: "${AORLTA_ROCM_ROOT}/bin/hipcc", gpus: [4, 5, 6]}
  hipk.time: {hipcc: "${AORLTA_ROCM_ROOT}/bin/hipcc", readelf: "${AORLTA_ROCM_ROOT}/llvm/bin/llvm-readelf", gpus: [7]}
```

and in the run config (`configs/hipk-speed-score.yaml`):

```yaml
plugins: [examples/hip-kernels/hip_kernels.py]   # registers hipk-speed, hipk.run, hipk.time and hipk.speed
reward:
  profile: hipk-speed
  reward_fn: hipk.speed
```

To mix blocks: a profile may name any adapter, built in or registered, in any pool, and a stage may list any of its
legs; the reward function must list every leg of the profile in its `legs` (the service builds the final record from
them, and a clean reward needs every profile leg clean). So adding a sanitizer leg to `hipk-gpu` means a pool for
`standin.rj_waitcheck`, a stage for `san.waitcheck` that `requires: [gate.compile]`, and a reward function that reads
it. Adding timing to another task's reward is one line: `register_reward_fn("my.speed", lambda profile:
WithSpeedup(MyV1(profile), name="my.speed"))`, plus a timing adapter for the task's kernels and its stage after the
correctness leg. A pool's `timeout` names one of `reward.timeouts` (`gate`, `compile`, `sim_run`, `asan_request`);
a slow check needs that timeout raised to match.

## Writing a new one

A check adapter (`CheckAdapter`, `src/aorlta/reward/adapters/base.py`) has `name`, `version`, `legs`, `resource`,
`run(job) -> LegResult`, `controls()` and `health()`. Build results with `make_result` and `unsettled`, run
subprocesses with `ProcessRunner` (the watchdog can kill them), write files in `run_dir(job)`, and read the job's
files from `job.inputs` (`source` is the parsed source, `response` the reply). Register a factory, `factory(ctx)` for
one worker, with `plugins.register_check_adapter(name, factory)`; `ctx.settings` is the profile's `adapters:` entry,
`ctx.pools` the run's prompt pool directories (where your task's specs are), `ctx.timeout_s` the time it has.

Rules every adapter keeps:

- `run` never raises; anything unexpected is `unsettled` (`adapter_error`).
- Anything it cannot settle (an error, missing inputs, a check it could not finish, a measurement it cannot trust)
  is `unsettled`, never `clean`.
- Only an adapter, never the service, may blame the candidate for a timeout (`fail` with reason `timeout`), and only
  when the timeout is the candidate's.
- A `finding` has findings; a `clean` leg has none, except records: a trusted build fact (`trusted_build`) or a
  measurement (`measurement`).
- Findings and reasons carry codes, kernel names, offsets and numbers, never source text or program output.
- Its `version` changes when what it checks changes: results are cached by the candidate, the task, the leg, its
  settings and the adapter's name and version.

A reward function (`RewardFn`, `src/aorlta/reward/rewardfn/base.py`) has `name`, `version`, `code_sha256`, `legs`,
`compute`, `decide` and `claims_clean`, and `settings` if its scoring reads configuration. `hipk.v1` and `qa.v1` are
short ones to copy: a `_base` that walks the legs in order and returns a score, a reason to leave the sample
unscored, or the first pending leg; `compute` and `decide` both built on it, so they cannot disagree. Register
`factory(profile)` with `plugins.register_reward_fn(name, factory)`, and the profile with
`plugins.register_profile(name, path)`. Test it as `tests/examples/test_hip_kernels.py` and
`tests/reward/test_timing.py` do: every combination of leg outcomes (or Hypothesis-drawn records) keeps the
service's invariants, and `decide` on every partial record agrees with `compute`.
