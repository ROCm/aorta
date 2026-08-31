# RL post-training an AORTA-expert model, with TokenSpeed as the rollout engine

The requirement, as stated in the meeting: take a reasonably-sized open-source
model (Qwen3-class was named), RL post-train it until it is expert in one narrow
domain — "AORTA-like topics and AORTA-like actions" — and use TokenSpeed as the
inference engine for the rollout phase. The pitch is that this is cheap on
training compute and expensive on inference, so it needs no cluster, and that
inference-heavy-compute-light is exactly where deploying TokenSpeed pays off. The
end state is a model that internal agent tooling can call when it needs AORTA
expertise.

This document works out what of that belongs in this repository, what TokenSpeed
can already do, what the training signal would be, and whether the cost claim
survives contact with the measured numbers. It is a plan, not a report of work
done: the only part built so far is the rollout-shaped serving support described
under [What was built](#what-was-built).

The short version:

- **The RL loop does not belong in aorta.** AORTA is a benchmarking and triage
  harness. Its job here is to stand up, validate and measure the rollout engine.
  The trainer belongs in a separate effort built on verl or slime.
- **TokenSpeed already has the hard part.** It ships an RL online weight-sync
  control plane — `/init_weight_transfer_engine`, `/start_weight_update`,
  `/update_weights`, `/finish_weight_update`, `/pause`, `/resume` — with NCCL and
  CUDA-IPC transports, and names verl / slime / AReaL / miles as the trainers it
  is for. This is not a gap to work around; it is the reason the engine choice is
  defensible.
- **The strongest training signal in this repo is recipe synthesis**, because a
  generated recipe either loads, validates and dry-runs or it does not. That is a
  graded machine-checkable reward requiring no human labelling. The second
  strongest is triage classification, where `aorta.probe.classifier` is a
  deterministic labeller that turns every archived probe run into a free example.
- **The cost claim is half right.** "No cluster needed" holds with room to spare:
  a plausible 400-iteration run generates ~419M tokens, which is ~2.5 hours of
  generation across one 8-GPU MI355X node. "Light on compute" does not hold on a
  FLOPs basis — the update is roughly 5x the generation's FLOPs — it holds only
  because decode utilises the hardware so much worse. The wall-clock split is
  more like 70/30 than 95/5, which caps what optimising the rollout half can buy.
- **"CIA" and "Sleuth" do not appear anywhere in this repository.** The internal
  consumer that does exist is `aorta agent`'s `LiteLLMProposer`, and it is a good
  enough seam to design against. Whether that is what was meant is the first
  assumption to confirm.

Related: [TokenSpeed under AORTA](tokenspeed.md) for the probe routes and the
container's operational hazards, [TokenSpeed serving benchmarks](tokenspeed-serving.md)
for the workload this extends and every measured number quoted below,
[Aorta Probe Agent](agent/aorta-probe-agent.md) for the agent loop that is the
candidate consumer.

## 1. Scope: what belongs where

AORTA runs benchmarks and triages failures. It has no optimiser, no dataset
plumbing, no checkpoint management and no gradient anywhere in it, and a
`Workload` is a thing that runs, passes or fails, and reports metrics. An RL
trainer is none of those shapes. Bending one into the other would produce a
trainer that is worse than verl and a harness that is worse than aorta.

So the split, and the seam:

| Concern | Where | Why |
|---|---|---|
| The RL algorithm (GRPO/PPO, advantages, optimiser, checkpoints) | Separate repo, built on **verl** or **slime** | TokenSpeed names both as supported trainers; reimplementing either is months of work to arrive behind where they are |
| The reward functions | Separate repo, but **importing aorta as a library** | The rewards are "does this recipe load", "does this triage read match the classifier" — they are calls into `aorta.triage.recipe` and `aorta.probe.classifier`, which is what makes them cheap |
| Standing the rollout engine up, and proving it serves | **aorta** (`tokenspeed_serve`) | Already does exactly this for fixed-length serving; rollout is a load shape it did not express |
| Measuring rollout throughput, length distribution, bring-up | **aorta** | This is the measurement the cost model above depends on, and nothing else in the stack reports it |
| Weight-transfer correctness and cost | **aorta**, later | A probe route, not a workload — see [Phase 2](#phase-2-validate-the-weight-sync-path) |
| Serving the finished model for agents to call | Neither; **ops** | An inference deployment, not a benchmark |

The load-bearing consequence is that the reward functions import aorta rather
than living in it. That keeps the dependency pointing the right way: the trainer
needs to know what a valid recipe is, and aorta already knows, but aorta must not
grow a dependency on a training framework to say so. It also means the reward
functions get aorta's own tests for free — a reward that accepts an invalid
recipe is a reward that teaches the model to write invalid recipes, and the
cheapest defence against that is that the validator is the same code the CPU gate
runs.

**Alternative rejected: put a `rl_rollout` workload in aorta that drives the
whole loop.** It would need an optimiser, a checkpoint store, and a way to hold
state across trials — and `Workload` has no place to put any of it, because a
trial is deliberately independent of every other trial. The version of this that
could be built would be a subprocess launcher for someone else's trainer, which
is worth less than the recipe that launches it directly.

**Alternative rejected: keep the reward functions inside aorta as a new module.**
Tempting, since the validators are already here. But rewards are a property of
the training run, not of the harness: they get tuned every few days early on, and
each tuning would be a commit to a repo that gates a nightly. The seam is better
placed at aorta's existing public API.

## 2. What a rollout loop needs from the engine, and what TokenSpeed has

An RL post-training loop spends most of its wall clock generating. Per iteration
it needs the engine to: take a batch of prompts; return several sampled
completions for each; sample at a temperature above zero so the group actually
differs; stop on EOS so length is the policy's choice; report token counts
honestly; and then **accept updated weights without a full restart**.

Findings below are from reading the CLI and runtime source inside the pinned
image (`lightseekorg/tokenspeed-amd@sha256:60c12e37…`), on a gfx950 node.

| Requirement | Status | Where |
|---|---|---|
| Several sampled completions per prompt | **Yes**, `n` in `SamplingParams`, reachable from the bench via `--extra-body` | `runtime/sampling/sampling_params.py`, `bench.py:1820` |
| Temperature > 0 | **Yes**, and the bench no longer sets one — the server's default applies unless `--extra-body` carries it | `bench.py:1916-1919` |
| EOS-respecting generation | **Yes, but not through the flag** — see below | `bench.py:1911-1912` |
| Generated-token counts | **Yes**, from the response's `usage.completion_tokens` | `bench.py:1258-1270` |
| Per-request lengths | **Yes**, `output_lens`, but only with `--save-detailed` | `bench.py:1639,1972` |
| Weight reload without cold start | **Yes, purpose-built** | `runtime/engine/weight_transfer/`, `runtime/entrypoints/control_server.py:399+` |
| Logprobs for the trainer's importance ratios | **Yes**, `/generate` with `return_logprob` | `control_server.py:262-290` |
| OpenAI *and* SGLang dialects | **Yes**, both, deliberately | `control_server.py:209` names "sglang-native /generate clients (e.g. slime, verl)" |

### The weight-reload answer, which is better than expected

TokenSpeed ships a weight-transfer control plane whose own docstring reads
"Implements the HTTP weight-transfer API that RL trainers (verl / slime / AReaL /
miles) drive to update model weights in place during online serving." The
lifecycle is `init / start / update / finish / pause / resume / is_paused`, the
metadata travels over HTTP on the control port, and the tensors travel
out-of-band over NCCL broadcast (trainer and engine on separate GPUs) or CUDA-IPC
(trainer and engine colocated), selected by
`--weight-transfer-config '{"backend": "nccl"}'`.

This is the requirement most likely to have been missing, and it is not missing.
It also settles the engine choice on more than throughput: an engine without it
would force a cold start per iteration, and the cost of that is not marginal.
Bring-up on gfx950 was measured at 189-319 seconds. At 250 seconds and 400
iterations that is 100,000 seconds — **27.8 hours of doing nothing but loading
weights**, against roughly 2.5 hours of actual generation for the same run (see
[cost](#5-cost-does-the-claim-hold)). Cold-starting per iteration would make
bring-up eleven times the cost of the work.

What is **not** established: that the path works on this image, on gfx950, at the
tensor-parallel widths a real run would use. Nothing in this repository has
exercised it, and `docs/tokenspeed-serving.md` records that TP=4 does not come up
at all on this image. Validating it is [Phase 2](#phase-2-validate-the-weight-sync-path).

### The EOS trap, which cost the most to find

`tokenspeed bench serve` contains this, after argument parsing:

```python
if args.dataset_name == "random" and args.backend in OPENAI_COMPATIBLE_BACKENDS:
    args.ignore_eos = True
```

On the random dataset, against an OpenAI-compatible backend — which is every
configuration the `tokenspeed_serve` workload runs — **`ignore_eos` is forced on
regardless of the flags.** Omitting `--ignore-eos` does not turn it off. The
`--disable-ignore-eos` flag does not reach it. So the flag is not a way to ask
for EOS-respecting generation, and a rollout built on it would have run at a
pinned output length while reporting itself as EOS-respecting: every completion
exactly `output_len` tokens, a length distribution with no distribution in it,
and a token volume that was a property of the recipe rather than of the policy.
That is a green cell describing a run that did not happen, arriving from upstream
rather than from a recipe.

What does reach it is the request body. The payload builder writes
`payload["ignore_eos"]` from the forced flag and *then* applies `extra_body` over
the payload:

```python
if request_func_input.ignore_eos:
    payload["ignore_eos"] = request_func_input.ignore_eos
if request_func_input.extra_body:
    payload.update(request_func_input.extra_body)
```

So `--extra-body '{"ignore_eos": false, "n": 4, "temperature": 1.0}'` is the
whole mechanism: it is the only route to the sampling parameters *and* the only
route to EOS. This is why rollout mode sends `ignore_eos: false` inside the body
and does not rely on the flag, and why `--extra-body` becomes an owned flag in
that mode.

A side effect worth recording: this also means the existing workload's
`ignore_eos: false` setting has never had any effect on the `random` dataset. The
setting is not removed, because it is still meaningful for `sharegpt`, and
because the export-derived TPOT branch it selects is the conservative one. But
nobody should read an existing `ignore_eos: false` cell as having respected EOS.

### The `n` accounting question, which is left open on purpose

The bench derives a request's output length from the response's
`usage.completion_tokens`, falling back to tokenizing the concatenated text. With
`n > 1`, whether the gateway reports that summed over all choices or only for the
first is the server's decision, and it cannot be read from the client source.

The consequence is a factor of `n` on every throughput number, so it is not
something to assume. Rollout mode therefore reports
`mean_output_tokens_per_request` — a name that is true on either reading — and
both committed recipes carry an `n=1` control cell so the ratio is measurable
from the recipe's own output. Roughly `n`x between the control and the `n=4` cell
means all samples are counted and the throughput figures cover the whole rollout;
roughly flat means they describe one sample per prompt and the real token volume
is up to `n`x higher than reported.

Two further things degrade under `n > 1` and should not be read as if they had
not. The client reads `choices[0]` from every SSE chunk and concatenates, so with
interleaved choices `generated_text` is a concatenation of all samples and every
recorded inter-token gap is an aggregate across them. **TTFT and throughput stay
meaningful; TPOT and ITL do not.** They are still reported, because suppressing
them per-mode would make the metric set depend on the configuration, but a
rollout cell's TPOT is not a per-token latency.

## 3. What "CIA" and "Sleuth" are

They are not in this repository. `rg -i` across all of `docs/`, `src/`, `config/`,
`recipes/` and `tests/` finds no `CIA` and no `sleuth` — not as identifiers, not
in prose, not in configuration.

What does exist, and is a plausible referent, is **`aorta agent`** — a
closed-loop mitigation-search agent documented in
[agent/aorta-probe-agent.md](agent/aorta-probe-agent.md). Its shape matters,
because it is the shape a post-trained model would have to fit:

- It calls an LLM through `LiteLLMProposer`, which is `litellm.completion(model=…,
  messages=…, response_format={"type": "json_object"})`.
- It asks for strict JSON: `category` (one of eight autopsy labels), `hypothesis`,
  `next_mitigations` (registered names only), `confidence`, `stop`.
- Every proposal is re-validated: names must resolve through
  `aorta.registry.get_mitigation`, and the pass/fail verdict comes from the
  deterministic classifier, never from the model.

That last property is what makes this a good first consumer rather than a risky
one: the model is advisory. A bad proposal costs a wasted probe cell, not a wrong
verdict. Pointing it at a self-hosted model is a small change — LiteLLM routes
`openai/<name>` to any OpenAI-compatible base URL, which is exactly what
TokenSpeed's gateway serves — with one gap: `LiteLLMProposer` passes no
`api_base`, so today it depends on `OPENAI_API_BASE` being set in the
environment. Threading an explicit `--llm-api-base` through is a few lines and
belongs with this work rather than in it.

If "CIA" and "Sleuth" are different systems, their interfaces need to be
established before anything is trained, because the output format is not a detail
— it is most of what the reward function checks. Recorded as
[assumption A1](#assumptions-to-confirm-with-manoj).

## 4. The domain and the training signal

"AORTA-like topics and AORTA-like actions" is a good niche for this precisely
because the actions are checkable. The repository is a corpus: 51 recipes, 22
registered mitigations, 11 workload classes with their configuration schemas, 35
documents, a five-tier deterministic failure classifier, and a triage output
format. An agent good at AORTA-like actions produces valid recipes, correct
triage reads, and correct mitigation proposals — and the first and third of those
can be scored by a machine, exactly.

Ordered by how much of the reward is automatic:

### 4.1 Recipe synthesis — the strongest signal, and the one to build first

**Prompt:** a natural-language benchmarking or triage intent. "Write a recipe that
finds where TokenSpeed serving throughput stops scaling with concurrency on one
MI355X." "Write a probe recipe that isolates an RCCL hang to a single mitigation."

**Completion:** a recipe YAML.

**Reward, graded, entirely automatic:**

| Tier | Check | Cost |
|---|---|---|
| 1 | Parses as YAML | microseconds |
| 2 | `aorta.triage.recipe.load_recipe` accepts it | milliseconds |
| 3 | Every mitigation resolves in the registry | milliseconds |
| 4 | The named workload's own validation accepts every cell, with unknown-key warnings treated as fatal | milliseconds |
| 5 | `aorta sweep run --dry-run` succeeds | ~1 second |
| 6 | The recipe expresses the *asked-for* shape (does the concurrency axis vary? is `num_prompts` scaled with it?) | rubric, partly automatic |

Tiers 1-5 are the same code path the CPU gate already runs on every committed
recipe — `tests/workloads/test_tokenspeed_serve.py` does precisely this — so the
reward function is a few dozen lines and inherits its correctness from tests that
already exist. Tier 6 is where a human or a stronger judge model is needed, and
it is also where most of the value is: a recipe that loads but measures the wrong
thing is the failure mode that matters, and it is the one this repo's own docs
spend the most words on.

This is the asset. It is rare to have a domain where the primary artifact is
machine-verifiable at this granularity, and it should carry most of the training
signal even if it is not the headline demo.

### 4.2 Triage classification — free labels at scale

`aorta.probe.classifier` is deterministic, and it is the source of truth for
pass/fail by design. So every probe run ever archived is a labelled example:
input is the cell's logs and report, label is the classifier's verdict plus its
`failure_detectors_fired` list plus the autopsy category those map to.

**No human labelling at all.** The reward is exact agreement with the classifier.
The risk is the mirror image of the opportunity: a model trained to agree with a
regex-and-tier classifier learns the classifier, not the failure. That is worth
having anyway — it is what makes the model useful inside the agent loop, where
the classifier is the arbiter — but it must not be described as diagnosis.

The practical constraint is corpus size. This needs a body of archived probe
runs, and how many exist is not something this repository can answer. Recorded as
[assumption A4](#assumptions-to-confirm-with-manoj).

### 4.3 Mitigation proposal — automatic validity, expensive correctness

Validity is free: the name resolves in the registry of 22 or it does not.
Correctness — did the proposed mitigation actually make the repro pass? — is
answerable without a human, by running the probe cell, but each answer costs a
GPU node for minutes. That makes it a good *evaluation* metric and a poor
*training* reward, and a small offline dataset of (symptom, detectors, winning
mitigation) triples harvested from past probe runs is the affordable
approximation.

### 4.4 Diagnosis and doc question-answering — needs humans

"Given this report, what went wrong and why" is the thing that would impress in a
demo and the thing with no automatic reward. The `docs/` corpus supports
retrieval-augmented answering, but grading a free-text explanation needs a human
or a judge model, and judge-model rewards on a narrow technical domain are where
reward hacking shows up first. Deliberately last.

### 4.5 What this implies about the run

Two things follow that a naive plan would get wrong.

**The base model must already be able to produce YAML that parses.** With rewards
concentrated in tiers 1-5, a policy that fails tier 1 gets a flat zero on nearly
every sample, every group's advantage is zero, and nothing is learned. Qwen3-8B
clears this comfortably; Qwen3-0.6B is borderline and is the wrong size for the
demo even though it is the right size for measuring the engine. A short
supervised warm-up on the 51 committed recipes before any RL is the cheap
insurance, and it is also the honest baseline to compare against — if supervised
fine-tuning on 51 recipes gets most of the way, the RL half needs to justify
itself.

**The reward must reject a memorised recipe.** Tiers 1-5 are all satisfied by
reproducing a committed recipe verbatim, which is the first thing a policy will
find. The prompt distribution has to ask for shapes that do not exist in the
corpus, and tier 6 has to check that the recipe answers the question asked.

## 5. Cost: does the claim hold?

Arithmetic from the measured numbers in
[docs/tokenspeed-serving.md](tokenspeed-serving.md) — one MI355X (gfx950), ROCm
7.0.2.2, the pinned image.

**A plausible run.** Domain specialisation on a narrow task does not need a large
run. Take 256 prompts per iteration, a GRPO group size of 8, a mean completion
of 512 tokens (AORTA recipes and triage reads are structured artifacts, not
long-form reasoning traces), and 400 iterations:

```
tokens per iteration = 256 prompts x 8 samples x 512 tokens =  1,048,576
total generated       = 400 iterations x 1,048,576          = 419,430,400
```

**Generation time.** The closest measured shape is the decode-heavy cell of
`tokenspeed-serve-load.yaml` (ISL 128 / OSL 1024, concurrency 16): **7,634
output tok/s** on Qwen3-0.6B. For Qwen3-8B the only measurement available is
1,625 tok/s at concurrency 8, ISL 512 / OSL 128 — a low-concurrency shape, so
using it directly overstates the cost. Qwen3-0.6B gained 3.52x going from
concurrency 8 to 32 (3,631 → 12,770 tok/s); applying that factor to the 8B is
**not measured** and is flagged as such.

| Model | tok/s | Basis | 419M tokens, 1 GPU | across 8 GPUs |
|---|---|---|---|---|
| Qwen3-0.6B | 7,634 | measured, decode-heavy | 15.3 h | 1.9 h |
| Qwen3-8B (conservative) | 1,625 | measured, conc 8 | 71.7 h | 9.0 h |
| Qwen3-8B (scaled) | ~5,720 | 1,625 x 3.52, **extrapolated** | 20.4 h | 2.5 h |

`419,430,400 / 5,720 = 73,327 s = 20.4 h`, and rollout generation is embarrassingly
parallel across data-parallel workers, so one 8-GPU node divides it.

**Verdict on "no large cluster": confirmed, with room.** Between 2 and 9 hours of
generation on a single node for the whole run. Even the conservative 8B figure
fits inside a working day. The claim is not marginal.

**Verdict on "light on compute": not as stated.** On a FLOPs basis the update is
*larger* than the generation. With `N` parameters and `T` generated tokens,
generation costs about `2NT` (one forward per token), while a GRPO iteration
costs about `6NT` for the policy's forward-and-backward plus roughly `2NT` each
for the reference and old-policy forwards — call it `10NT`, **five times the
generation's FLOPs.**

The claim survives only through utilisation. Decode is memory-bandwidth-bound and
sequential, running at single-digit-percent of peak; the update is compute-bound
and parallel over the whole sequence, running at tens of percent. Taking 5% for
decode and 40% for training, generation's share of wall clock is
`(2/5) / (2/5 + 10/40)` ≈ 62%, or about 70% once the scheduling tail is included
— the iteration cannot finish until the *longest* completion in the batch does,
which is why the length distribution this work reports is not a curiosity.

So: **inference-heavy, yes; training-negligible, no.** The distinction has a
consequence worth stating plainly, because it bounds the pitch. If generation is
70% of the iteration, making the rollout engine twice as fast makes the run
1.54x faster, not 2x. TokenSpeed is the right thing to deploy here and the
biggest single lever, and it is still an Amdahl-bounded one.

**And the bring-up figure is the sharpest number in this document.** 400
iterations x 250 s of cold start = **27.8 hours**, against 2.5 hours of
generation. Any loop that restarts the engine per iteration spends 91% of its
life loading weights. That, and not throughput, is what makes the weight-transfer
control plane the load-bearing feature.

## 6. What was built

The smallest thing that is useful regardless of how the RL loop lands: rollout
support in the existing `tokenspeed_serve` workload. No new workload class — the
mode differs from ordinary serving in what the engine is asked to do per request,
not in how it is stood up, supervised, torn down or audited, and every one of
those is where the workload's complexity lives. A second class would have
duplicated the VRAM-release wait, the container-cleanup handlers, the per-uid
scratch and the protocol guards, which is how two copies of a guard become one
guard and one hole.

`rollout: true` sends `rollout_samples` sampled completions per prompt at
`temperature`, stopping on EOS, and reports the generated-length distribution.
Configuration is documented in
[tokenspeed-serving.md](tokenspeed-serving.md#rollout-mode); two digest-pinned
recipes are committed (`tokenspeed-serve-rollout-smoke.yaml`,
`tokenspeed-serve-rollout.yaml`).

### The audit, which needed the most thought

The existing served-request audit requires `completed == num_prompts` and
`failed == 0`. **Neither is relaxed.** `completed` counts requests, not
completions — `bench.py` increments it once per `RequestFuncOutput` — so it is
still `num_prompts` however large `n` is, and if the export turns out to count
completions instead, the audit fails loudly and says so, which is information
worth having rather than a guard worth pre-emptively loosening.

What EOS-respecting generation *does* break is the sufficiency of those two
checks. With `ignore_eos` the output length is pinned, so a served request is a
known amount of work. Without it the model decides, and a policy that emits EOS
immediately serves every request, fails none, and takes real time doing it —
producing finite positive `duration`, TTFT and all three throughputs. Every
existing guard passes. The cell goes green having generated about one token per
prompt, and reports the resulting throughput as a measurement.

For a fixed-length benchmark that state is unreachable, which is why the audit
never covered it. For a rollout it is a routine RL failure: entropy collapse, a
length penalty that overshot, a tokenizer whose EOS lands first. So rollout mode
adds one guard — `min_mean_output_tokens`, a floor on
`total_output_tokens / completed`, default 8 — enforced in the container (exit
56, `rollout_output_too_short`) and independently on the host, like every other
audit here. It is computed from `total_output_tokens` and `completed`, the two
fields every export version carries, rather than from `output_lens`, which
depends on `--save-detailed`; a verdict must not hinge on an optional field. A
floor above `output_len` is rejected at config time, since `output_len` becomes
the `max_tokens` cap and such a recipe could never pass — it would fail as a
collapsed policy rather than as the arithmetic error it is.

The default of 8 is not a judgement about good completion length. It is far below
anything worth training on and far above the 1-2 tokens a collapsed policy
produces, which is the only distinction it is being asked to make.

The TPOT audit needed **no change**, which is worth recording because it looked
like it would. `_missing_core_metrics` already decides whether TPOT can exist
from the export (`total_output_tokens > completed`) whenever the configuration
does not pin the output length, and rollout mode is exactly that case. The
existing branch is correct for it.

### `--extra-body` is reserved in rollout mode only

Every other owned bench flag is reserved unconditionally, because for several of
them the workload's setting is absence and a guard covering only the configured
case leaves the default one open. `--extra-body` inverts that reasoning. Outside
rollout the workload sends no sampling parameters at all, so there is nothing for
a caller's `--extra-body` to shadow, and reserving it would forbid the only route
to a sampling knob — a documented use of `bench_args`. Inside rollout the
sampling body *is* what the trial reports as `rollout_samples` and `temperature`,
and the extras arrive last, so a caller's copy would win and change the
completions generated while the cell kept publishing the configured ones.

Reserved exactly where the workload generates one, in other words, and the
generated case is the whole mode — so within rollout the reservation is
unconditional in the way the others are.

A pre-existing hole is left open and named rather than half-fixed: outside
rollout, `bench_args: ["--extra-body", '{"n": 8}']` multiplies the load eightfold
with nothing in the report saying so. That is true today, was true before this
change, and closing it properly means validating the body's contents against the
values the workload publishes. Listed under [known gaps](#known-gaps).

### Two existing tests were deliberately changed

- `_run_script_audit`'s harness now defines `MIN_MEAN_OUTPUT_TOKENS`, because the
  in-container audit function gained a parameter.
- `test_the_documented_protocol_floor_is_actually_owned` gained a rollout
  configuration, because the floor gained six keys and that test's contract is
  that every reserved key is one the workload really sets under *some*
  configuration.

Neither weakens an assertion; both extend a fixture to cover a wider contract.

## 7. Phased path from nothing to a model an agent can call

### Phase 0 — rollout-shaped measurement (done)

Rollout mode plus two recipes, so the throughput and length numbers the cost
model rests on can be measured instead of extrapolated. Gate: CPU suites, ruff,
`bash -n`, `--dry-run`, and a real gfx950 run of the smoke recipe.

### Phase 1 — measure the real rollout shape

Run `tokenspeed-serve-rollout.yaml` on gfx950 and fill in a measured-results
section: throughput at `n` = 1/4/8, the length distribution at a 1024- and a
4096-token allowance, and the `n=1`-vs-`n=4` comparison that resolves the
`usage.completion_tokens` question. Repeat on Qwen3-8B, which is the size the
demo would actually use and the one whose concurrency scaling is currently
extrapolated. **Output: the cost model in section 5 stops containing an
extrapolation.**

### Phase 2 — validate the weight-sync path

A probe route, not a workload: exercise `/init_weight_transfer_engine` →
`/start_weight_update` → `/update_weights` → `/finish_weight_update` against a
served model, with a trivial "trainer" that broadcasts a known perturbation, and
assert the served outputs change. Measure the update's wall clock, which is the
number that replaces 250 s of bring-up in the cost model. Verdict from an exit
code, in the band `ts_serve_probe.sh` uses.

Do this **before** committing to a trainer. It is the single assumption the whole
architecture rests on, it is cheap to test, and if it fails on this image the plan
changes shape rather than schedule.

### Phase 3 — reward functions and a supervised baseline

In the separate repo, importing aorta: the graded recipe reward of section 4.1
and the classifier-agreement reward of 4.2. Then supervised fine-tuning of
Qwen3-8B on the 51 committed recipes and whatever probe archive exists, and
measurement of tier-1-through-5 pass rates. **This is the gate on the whole
effort:** if SFT gets most of the way, the RL phase has to justify itself against
that number rather than against zero.

### Phase 4 — the RL loop

verl or slime, with TokenSpeed as the rollout engine over the weight-transfer
path, colocated (`ipc`) on one 8-GPU node. GRPO rather than PPO: no value network
to fit, which matters when the reward is a graded checker rather than a learned
model. Success is tier-1-5 pass rate and classifier agreement on held-out
prompts, against the Phase 3 baseline.

### Phase 5 — serve it, and point the agent at it

Deploy behind TokenSpeed's OpenAI-compatible gateway, add `--llm-api-base` to
`aorta agent`, and run the agent's own evaluation with the post-trained model as
proposer against the same loop driven by a general model. The comparison is
mitigation-search iterations to convergence — the agent's existing outcome
labels (`converged`, `exhausted_candidates`) are the metric, and no new
evaluation harness is needed.

## Assumptions to confirm with Manoj

**A1 — "CIA" and "Sleuth".** Neither appears in this repository. This plan
assumes the intended first consumer is something shaped like `aorta agent`'s
`LiteLLMProposer`: an OpenAI-compatible endpoint returning strict JSON with a
constrained `category` and registered mitigation names. If CIA and Sleuth are
different systems, their output contract is needed before training, because the
output format is most of what the reward checks.

**A2 — "AORTA-like actions" means producing AORTA artifacts.** This plan reads
the niche as recipe synthesis, triage classification and mitigation proposal,
because those are what the repository can score automatically. If what was meant
is conversational expertise about AORTA — answering questions from the docs — the
machine-checkable reward largely disappears and the effort is a
retrieval-augmented-generation problem rather than an RL one.

**A3 — Model size.** Qwen3-8B is assumed: the largest Qwen3 in the measured set,
and comfortably able to emit parseable YAML, which the reward design requires.
Qwen3-0.6B is used in the committed recipes because they measure the engine, not
the model. If the demo needs a specific size, the cost table changes.

**A4 — A probe archive exists.** The classifier-agreement signal needs a body of
archived probe runs with logs and reports. This repository has the classifier but
no corpus. How many real runs are retained, and where, decides whether 4.2 is a
main signal or a footnote.

**A5 — "Self-serving to Toyota" is a separate demo.** The transcript mentions
Toyota alongside AORTA as candidate domains. This plan addresses only the AORTA
domain, on the grounds that it is the one with an automatic reward. A
customer-facing vertical would need its own reward design and probably human
labelling.

**A6 — The rollout engine may take the whole node.** The colocated (`ipc`)
configuration assumes trainer and engine share GPUs on one node. If the engine
must be disaggregated onto separate GPUs (`nccl`), the node count doubles and the
"no cluster" claim gets tighter.

## Known gaps

- **The weight-transfer path is unexercised here.** Everything in section 2 about
  it is read from source, not run. Phase 2 exists for this, and it is the
  assumption most worth breaking early.
- **`usage.completion_tokens` under `n > 1` is unresolved.** The recipes are built
  to answer it; until they run on hardware, every rollout throughput number is
  ambiguous by a factor of up to `n`.
- **TPOT and ITL are not meaningful under `n > 1`.** The client concatenates all
  choices and treats every chunk gap as an inter-token interval. Still reported,
  because suppressing metrics per-mode would make the metric set depend on the
  configuration; not to be read as per-token latency.
- **`--extra-body` outside rollout can still change the load silently.** Named
  above; a pre-existing hole this change neither widens nor closes.
- **No perf gates on rollout metrics.** `mean_output_tokens_per_request` and
  `generated_tokens_p50` are in the CI gating allowlist, so they are gateable
  once baselines exist, but no rollout recipe is in the nightly matrix and
  nothing is gated. Gating a *length* is also a judgement call: under
  EOS-respecting generation the length is the model's choice, so a lower bound
  protects against the engine returning less text while an upper bound would
  redden a run for producing more.
- **The 8B concurrency figure is extrapolated.** Section 5 marks it; Phase 1
  removes it.
- **`ignore_eos: false` has never worked on the `random` dataset.** Documented in
  section 2 rather than fixed, because the fix is the rollout body and the
  setting remains meaningful for `sharegpt`.
