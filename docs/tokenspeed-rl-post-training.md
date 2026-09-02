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
- **TokenSpeed has the hard part designed, and neither transport works.** It
  ships an RL online weight-sync control plane —
  `/init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`,
  `/finish_weight_update`, `/pause`, `/resume` — with NCCL and CUDA-IPC
  transports, and names verl / slime / AReaL / miles as the trainers it is for.
  Measured on gfx950: the control plane works and is effectively free (1–5 ms
  against a 322 s cold start), `ipc` raises `NotImplementedError`, and `nccl`
  **returns success while transferring nothing**, loading uninitialised device
  memory into the model
  ([Phase 2b](#phase-2b-the-nccl-data-plane-does-not-transfer), filed upstream
  as [tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373)).
  The engine choice is still defensible on the shape of the API; the loop is
  blocked on an upstream fix.
- **The strongest training signal in this repo is recipe synthesis**, because a
  generated recipe either loads, validates and dry-runs or it does not. That is a
  graded machine-checkable reward requiring no human labelling. The second
  strongest is triage classification, where `aorta.probe.classifier` is a
  deterministic labeller that turns every archived probe run into a free example.
- **The cost claim is half right.** "No cluster needed" holds with room to spare:
  a plausible 400-iteration run generates ~419M tokens, which is ~2.5 hours of
  generation across one 8-GPU MI355X node — or ~3.3 hours once `ipc`'s absence
  forces the trainer onto 2 of those 8 GPUs, which costs a third of the
  generation throughput and **no second node**
  ([5b](#5b-what-disaggregation-actually-costs)). "Light on compute" does not
  hold on a FLOPs basis — the update is roughly 5x the generation's FLOPs — it
  holds only because decode utilises the hardware so much worse. The wall-clock
  split is more like 70/30 than 95/5, which caps what optimising the rollout
  half can buy.
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
| Weight reload without cold start | **Yes, purpose-built — but `nccl` only**, `ipc` raises `NotImplementedError` ([Phase 2](#phase-2-validate-the-weight-sync-path)) | `runtime/engine/weight_transfer/`, `runtime/entrypoints/control_server.py:399+` |
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

**The heading above is wrong, and running it is what established that.** Two
phases of measurement have inverted this section's conclusion, and it is left
standing because the reasoning that produced it — an engine with this API is the
right engine — is still correct, while the assumption underneath it was not.

[Phase 2](#phase-2-validate-the-weight-sync-path) found the control plane works
on this image on gfx950 and costs 1–5 ms against a 322 s cold start, but that
**`ipc` is not implemented** — its receive path raises — so of the two backends
named above only `nccl` is wired, and the colocated deployment this section
implies is unavailable.

[Phase 2b](#phase-2b-the-nccl-data-plane-does-not-transfer) then found that
`nccl` **does not transfer either.** With a real trainer peer joined to the
group, `/update_weights` returns `200 {"message": "Weights updated"}` having
moved no tensor at all, and loads uninitialised device memory into the model.
The served completion changes, so the failure impersonates success; only pushing
the original weights back and watching them *not* come back reveals it.

So the honest position is that the API exists, its shape is right, its control
plane is fast, and **neither of its two transports works on this build**. The
27.8-hour argument above still holds — it is why this matters — but the fix it
depends on is upstream and unbuilt, tracked as
[tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373).
TP > 1 is blocked behind the same defect; `docs/tokenspeed-serving.md`
separately records that TP=4 does not come up at all on this image.

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

### What was measured on gfx950

Two probes on one MI355X against the pinned image, Qwen3-0.6B.

**The rollout path works end to end.** The smoke shape — 8 prompts, `n=4`,
`temperature: 1.0`, 256-token allowance — served 8 requests with none failed,
exit 0, at 6,461 output tok/s. The export carried `output_lens` under
`save_detailed`, and the workload's floor and audits passed.

**`usage.completion_tokens` sums across all sampled choices.** This was the open
question and it is now closed. That run reported 1,024 generated tokens *per
request* against a per-choice cap of 256: `4 x 256 = 1024`, and no single choice
can exceed its own cap, so the count is necessarily the sum over all four.
Rollout throughput figures therefore describe the whole rollout, not one sample
of it.

**EOS-respecting generation through the request body works.** A direct probe of
`/v1/chat/completions` with `ignore_eos: false, n: 4` returned four choices, all
with `finish_reason: "stop"`, totalling 788 tokens against an 800-token ceiling —
completions that ended on EOS at genuinely variable lengths, not at the cap. This
is the mechanism the whole mode depends on, and it is the one that could not be
established by reading source.

The metric name stays `mean_output_tokens_per_request` rather than becoming
per-sample. The measurement above is one gateway version on one model; the name
is true regardless, the `n=1` control cell in both recipes keeps the ratio
observable if it ever changes, and that cell independently measures whether
sharing a prefill across a larger group is cheaper per completion.

### The random dataset cannot show a length distribution

The same smoke run reported `output_lens` of exactly `[1024] x 8` — every
completion at the cap. That is not EOS being ignored; the probe above shows it is
respected. It is the prompts. `dataset: random` generates sequences of random
tokens, and a model given gibberish has no reason to emit an end-of-sequence
token, so every completion runs to `output_len` and the length distribution
collapses to a constant.

The consequence is worth stating plainly, because it limits what the committed
recipes can show: **on `dataset: random`, rollout mode measures throughput at a
rollout-shaped request pattern, but its `generated_tokens_*` distribution is an
artifact of the cap rather than a property of the policy.** Throughput,
concurrency behaviour and the sample-count comparison are all still valid — those
depend on the request shape, not on where generation stops.

A real length distribution needs prompts a model would answer. `dataset:
sharegpt` is the option already supported and is the cheapest next step; a
prompt set drawn from the actual rollout task — the recipe-synthesis prompts of
[section 4](#4-the-domain-and-the-training-signal) — is the one that would
actually predict the RL run's cost, and it needs a dataset loader that neither
the workload nor the bench has today. Recorded under
[known gaps](#known-gaps).

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

### 5b. What disaggregation actually costs

`backend: ipc` is unimplemented (Phase 2), so trainer and engine cannot share
GPUs and the figures above — which assumed all 8 GPUs generating — need
revising. An earlier draft of this document said that "doubles the nodes". **That
was wrong, and the correction is in our favour.**

`nccl` requires separate *GPUs*, not separate *nodes*. The Phase 2b peer ran on
GPU 7 of the same node as the engine on GPU 0, over intra-node RCCL. So the cost
of losing `ipc` is paid in devices out of the node's eight, not in a second
node.

**What each side wants for a Qwen3-8B-class model.** MI355X carries 288 GB
HBM3e per GPU, 8 per node, which is what makes this comfortable:

| Side | Component | Memory |
|---|---|---|
| Trainer | policy weights (bf16) | 16 GB |
| | gradients (bf16) | 16 GB |
| | Adam moments (fp32 m + v) | 64 GB |
| | fp32 master weights | 32 GB |
| | frozen reference policy (bf16) | 16 GB |
| | activations, 512-token sequences | ~20–40 GB |
| | **trainer total** | **~165–185 GB** |
| Engine | weights per data-parallel replica (bf16) | 16 GB |
| | KV cache + workspace per replica | ~30–60 GB |

The trainer fits on a single 288 GB GPU on paper. Giving it **two** is the
recommendation — headroom for activations at longer sequences, and FSDP across a
pair rather than betting the run on a single-device fit. GRPO needs no value
network, which is what keeps this off four or more.

That leaves **6 GPUs generating instead of 8**, and generation is
embarrassingly parallel across data-parallel replicas, so the cost is linear:

| Layout | Generating GPUs | Generation (8B scaled) | Generation (8B conservative) | Iteration wall clock | Node-hours |
|---|---|---|---|---|---|
| colocated `ipc` (unavailable) | 8 | 2.5 h | 9.0 h | 3.6 h | **3.6** |
| **disaggregated, one node (6 + 2)** | 6 | 3.3 h | 12.0 h | 4.8 h | **4.8** |
| disaggregated, two nodes (8 + 2) | 8 | 2.5 h | 9.0 h | 3.6 h | **7.1** |

Iteration wall clock divides generation by the ~70% share section 5 derives, so
it covers the training step too; node-hours multiply that by the nodes held.

**The recommendation is the middle row: keep it on one node.** Losing a quarter
of the generating GPUs costs ~33% more wall clock; buying that back with a
second node costs ~48% more node-hours and leaves 6 GPUs idle on the second
node. The one-node layout is the cheaper trade on both counts, and it is also
the simpler thing to schedule on a preemptible partition.

**So "no large cluster" survives, and it was never close.** The honest revision
is 3.6 → 4.8 node-hours for the scaled 8B estimate, and 12.9 → 17.1 for the
conservative one. A 400-iteration run still fits on **one 8-GPU node** inside a
day, which is what the claim actually asserted. What disaggregation costs is a
quarter of the node's generation capacity, not a second machine.

Two caveats on those numbers. The 8B throughput is extrapolated from a
concurrency-8 measurement by a factor measured on Qwen3-0.6B (section 5), so the
conservative column is the one to plan against. And all of this is contingent on
the weight transport working at all — Phase 2b found it does not, and no
topology fixes that.

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

Rollout mode plus two recipes, validated on gfx950 as described in
[section 2](#what-was-measured-on-gfx950). Gate: CPU suites, ruff, `bash -n`,
`--dry-run`, and a real run of the smoke shape.

### Phase 1 — prompts that stop, and the 8B numbers

Two things, in order. First, prompts that induce EOS, since `dataset: random`
cannot: run the rollout recipes against ShareGPT to get a length distribution
that is a property of the model, then decide whether a task-shaped prompt set is
worth the dataset plumbing. Second, run the sweep on Qwen3-8B — the size the
demo would use, and the one whose concurrency scaling section 5 currently
extrapolates. **Output: the cost model stops containing an extrapolation, and
the mean-completion-length assumption behind it becomes a measurement.**

### Phase 2 — validate the weight-sync path

**Partly done, and it changed the plan's shape.** Measured on gfx950
(`cv350-rck-g03-c10-18`, one GPU, Qwen3-0.6B, the image the recipes pin), by
driving the control plane against a live `tokenspeed serve`.

**The control plane is real, always on, and effectively free.** `tokenspeed
serve` assigns `--rl-control-port` itself, so the weight routes are live on the
control port without asking. Against a **322 s** cold start for a 0.6B model at
TP=1 — the top of the 189-319 s band section 2 records, for the smallest model
in play:

| Operation | Wall clock |
|---|---|
| `/pause` (`abort`, `wait`, `keep`) | 1–5 ms |
| `/resume` | ~1 ms |
| `/is_paused`, `/get_world_size` | ~1 ms |
| full `init → start → update → finish` | ~5 ms |
| cold start, for comparison | **322 s** |

That is five orders of magnitude, and it is the number the cost model needed:
the control plane is not what a weight update will cost. Generation was verified
before, between and after — the server served identical completions throughout
and never restarted, so **pause/resume against a running server is settled**.

**The colocated path is not available.** `backend: ipc` accepts `init` and
`start`, parses `update_info` in full, and then raises `NotImplementedError` —
see [A6](#assumptions-to-confirm-with-manoj). Phase 4's "colocated (`ipc`) on
one 8-GPU node" has to become disaggregated `nccl` — which costs GPUs inside the
node, not a second node, since `nccl` runs intra-node quite happily
(see [5b](#5b-what-disaggregation-actually-costs)). This is exactly the
assumption that was worth breaking early, and it broke.

**Two contract details a trainer will hit.** The lifecycle guards are real —
start-before-init, update-with-no-active-update, finish-with-none-active and
double-start all refuse — but they surface as **500**, not the **409** the
manager's own docstring promises ("The HTTP layer maps this to 409 Conflict"; it
does not). Bad `update_info` — an unknown key, `update_kind: sparse` — is also a
500 rather than a 400, because the handlers validate only that the field is
present and let manager errors through. A trainer that distinguishes retryable
conflicts from bugs by status code will misclassify both. Malformed requests the
*handlers* do check (`init_info` missing, an invalid pause mode) are correctly
400 with useful messages.

**The surface is wider than section 2 lists.** Also present: `/get_world_size`,
`/is_paused`, and a complete SGLang dialect — `/init_weights_update_group`,
`/update_weights_from_distributed`, `/update_weights_from_tensor`,
`/update_weights_from_disk`, `/pause_generation`, `/continue_generation`,
`/release_memory_occupation`, `/resume_memory_occupation`. slime and verl's
SGLang rollout should drive this unchanged, which widens the trainer choice.

### Phase 2b — the NCCL data plane does not transfer

Filed upstream as
[tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373).

The control plane above is real. **The transport is not.** A trainer peer was
stood up on an 8-GPU gfx950 node (engine on GPU 0, peer on GPU 7, disjoint
`HIP_VISIBLE_DEVICES`, TP=1, Qwen3-0.6B) and driven through the full
`init → start → update → finish`. Every call returns 200. Nothing moves.

**The finding.** `/update_weights` answers
`200 {"message": "Weights updated"}` while transferring nothing, and **loads
uninitialised device memory into the model**. The served completion changes,
which is why this is dangerous: it looks like a working weight update.

```
prompt "The capital of France is", temperature 0, TP=1 (identical at TP=2)
  baseline          " Paris. The capital of France is also the capital of ..."
  after "perturb"   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  after "restore"   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"   <- should equal baseline
```

**Why the restore half is what caught it.** A perturbation-only test passes
here. The completion changes, `/update_weights` returns 200, and the obvious
conclusion — "the broadcast landed" — is wrong. Pushing the *original* weights
back is what breaks the tie: a faithful transport returns the baseline
completion exactly, and this one does not move at all. Any future check of this
path should be a round trip for that reason.

**The evidence that nothing was received**, in the order it narrowed:

1. **The peer's broadcast never drains.** It logs per tensor and never completes
   its first `dist.broadcast`. Whatever the engine did, it did not match the
   sender.
2. **The peer's helper is correct.** Two ranks using the same group construction
   broadcast to each other correctly over RCCL 2.27.7 on the same node and image
   — a poisoned receive buffer is replaced by the sender's pattern. So the
   protocol reimplementation is not the fault.
3. **No RCCL communicator is ever created by the engine.** It logs
   `weight-update group joined: rank=1 world_size=2` and later
   `weight-update group destroyed`, with no communicator init between them.
4. **`/update_weights` is precisely the call that corrupts the model.** Walking
   the lifecycle one step at a time, generating after each: `init` leaves the
   completion identical, a `start`/`finish` bracket with no update leaves it
   identical, and the *first* `/update_weights` degrades it — with
   `flush_cache: false`, so the cache flush is not the cause.
5. **Reversing the roles fails the same way.** Initialising with
   `rank_offset: 0` makes the engine rank 0 and therefore the broadcast root,
   with the peer receiving into a poisoned buffer. The engine's update returns
   200 in 0.350 s; the peer's receive never drains and the poison is never
   replaced. So this is not a disagreement about which end is the root — the
   engine posts no collective in either direction.

Together: `init_weights_update_group` genuinely forms the torch process group
(the peer's rendezvous completes and the engine reports the right rank), and
`update_weights_from_distributed` genuinely reaches the model and calls
`load_weights` — but the `dist.broadcast` that should fill the receive buffer
never executes, so `torch.empty`'s contents are what gets loaded. On the second
update the allocator hands back the same block, which is why "restore" produces
a byte-identical degenerate completion rather than a different one.

This is worse than `ipc`. `ipc` refuses honestly with `NotImplementedError`;
`nccl` reports success and silently destroys the policy. A trainer would see
rewards collapse after the first weight sync and have no reason to suspect the
transport.

**Timing, for what it is worth.** The control-plane cost is not the obstacle:
`init` 6–8 ms, `start` and `finish` ~1 ms each, `/update_weights` 0.25–0.85 s
for a 311 MB tensor on the first call and ~2 ms after — against the **322 s**
cold start it would replace. If the transport worked at these latencies, the
27.8-hour restart bill in section 5 would collapse to minutes and 400 iterations
would be comfortable. The ratio is not what blocks this; correctness is.

**TP = 2 fails identically**, which is worth knowing: the defect is not a
function of tensor-parallel width. Engine on GPUs 0–1, peer on GPU 7,
`world_size: 3`, `rank_offset: 1`:

| | TP=1 | TP=2 |
|---|---|---|
| `/get_world_size` | 1 | 2 |
| workers joined | rank 1 | ranks 1 **and** 2 |
| `init` | 6 ms | 357 ms |
| `/update_weights` (first) | 246 ms | 518 ms |
| `/update_weights` (second) | 2 ms | 7 ms |
| completion changed by "perturb" | yes | yes |
| completion restored by "restore" | **no** | **no** |
| peer's broadcast drained | **no** | **no** |
| RCCL communicators created | **0** | **0** |

The rank layout is exactly as documented — both engine workers join at
`rank_offset + i`, taking ranks 1 and 2 of a world of 3 — so the *addressing*
scales correctly and only the transfer is missing. Sharded receive proper
(whether `load_weights` slices a full unsharded tensor correctly per rank)
remains genuinely unevaluated, because it cannot be reached until a tensor
arrives.

**What would be needed.** This is an upstream fix, not a configuration change —
the metadata contract is right, the group forms, the call reaches the model, and
the collective does not happen. Reproducing it takes
[`examples/rl/nccl_weight_peer.py`](../examples/rl/nccl_weight_peer.py) and
[`examples/rl/nccl_roundtrip_check.py`](../examples/rl/nccl_roundtrip_check.py)
against the pinned image; the round-trip check reports
`HTTP_OK_BUT_WEIGHTS_UNCHANGED` or `CHANGED_BUT_NOT_FAITHFUL` rather than
`PROVEN`, and is the regression test for the fix. Until then **no online RL loop
can run on this build at any topology**, and Phase 4 needs either an upstream
weight-transfer fix or a fallback that pays the cold start.

The metadata contract, confirmed against the image's own source:
`init_info: {master_address, master_port, rank_offset, world_size, group_name?}`
and `update_info: {names, dtype_names, shapes, packed?, packed_buffer_size_bytes?,
packed_num_buffers?, group_name?, flush_cache?}`. Engine worker `i` takes rank
`rank_offset + i`; shapes are the **unsharded** checkpoint shapes at every TP
degree, because the receive side hands the tensors to the model's own
`load_weights`, which applies the sharding.

### Phase 3 — reward functions and a supervised baseline

In the separate repo, importing aorta: the graded recipe reward of section 4.1
and the classifier-agreement reward of 4.2. Then supervised fine-tuning of
Qwen3-8B on the 51 committed recipes and whatever probe archive exists, and
measurement of tier-1-through-5 pass rates. **This is the gate on the whole
effort:** if SFT gets most of the way, the RL phase has to justify itself against
that number rather than against zero.

### Phase 4 — the RL loop

verl or slime, with TokenSpeed as the rollout engine over the weight-transfer
path. **Disaggregated (`nccl`), not colocated:** Phase 2 established that the
`ipc` receive path is not implemented, so trainer and engine need separate GPUs
— 6 generating plus 2 training on one 8-GPU node, which costs ~33% of the
generation throughput and no extra machine
([5b](#5b-what-disaggregation-actually-costs)). GRPO rather than PPO: no value network
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

Everything that could be settled by reading the repository or running hardware
has been. What is left needs a decision or an artifact that is not ours, and
these four block Phase 4 rather than merely refining it:

| # | Blocked on | Why it blocks | Costs if wrong |
|---|---|---|---|
| [A1](#a1) | The CIA / Sleuth output contract | The reward checks output *format* first; neither system appears in this repository, so there is nothing to check against | Rewrite of the reward's outer layer; the tier ladder survives |
| [A2](#a2) | What "AORTA-like actions" means | Producing artifacts is automatically scorable; conversational expertise is not | If it means conversation, this is a RAG problem and the RL case largely dissolves |
| [A3](#a3) | Model choice | Sets the memory and throughput arithmetic in [5](#5-cost-does-the-claim-hold) and [5b](#5b-what-disaggregation-actually-costs) | The cost table, not the design |
| [A7](#a7) | Who stands up the trainer | No trainer exists here; verl/slime integration, the GRPO loop and its checkpointing are outside this repository | Phase 4 cannot start |

Two further items are blocked but softer: [A4](#a4) (does a probe archive exist)
decides whether triage classification is a main signal or a footnote — the code
path for it is built and tested against synthetic fixtures, so only the corpus
is missing — and [A5](#a5) (whether Toyota is a separate demo) affects scope
rather than feasibility.

One item that is **not** blocked on Manoj and should be raised anyway: the
`nccl` weight transport does not work on this image
([Phase 2b](#phase-2b-the-nccl-data-plane-does-not-transfer)). That is an
upstream TokenSpeed defect, it blocks the loop at every topology and model size,
and it is ours to file rather than his to decide. Filed as
[tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373).

<a id="a1"></a>
**A1 — "CIA" and "Sleuth".** Neither appears in this repository. This plan
assumes the intended first consumer is something shaped like `aorta agent`'s
`LiteLLMProposer`: an OpenAI-compatible endpoint returning strict JSON with a
constrained `category` and registered mitigation names. If CIA and Sleuth are
different systems, their output contract is needed before training, because the
output format is most of what the reward checks.

<a id="a2"></a>
**A2 — "AORTA-like actions" means producing AORTA artifacts.** This plan reads
the niche as recipe synthesis, triage classification and mitigation proposal,
because those are what the repository can score automatically. If what was meant
is conversational expertise about AORTA — answering questions from the docs — the
machine-checkable reward largely disappears and the effort is a
retrieval-augmented-generation problem rather than an RL one.

<a id="a3"></a>
**A3 — Model size.** Qwen3-8B is assumed: the largest Qwen3 in the measured set,
and comfortably able to emit parseable YAML, which the reward design requires.
Qwen3-0.6B is used in the committed recipes because they measure the engine, not
the model. If the demo needs a specific size, the cost table changes.

<a id="a4"></a>
**A4 — A probe archive exists.** The classifier-agreement signal needs a body of
archived probe runs with logs and reports. This repository has the classifier but
no corpus. How many real runs are retained, and where, decides whether 4.2 is a
main signal or a footnote.

<a id="a5"></a>
**A5 — "Self-serving to Toyota" is a separate demo.** The transcript mentions
Toyota alongside AORTA as candidate domains. This plan addresses only the AORTA
domain, on the grounds that it is the one with an automatic reward. A
customer-facing vertical would need its own reward design and probably human
labelling.

<a id="a6"></a>
**A6 — The rollout engine may take the whole node. RESOLVED, against us.** The
colocated (`ipc`) configuration assumed trainer and engine could share GPUs on
one node. They cannot, on this image: the IPC receive path is not implemented.
Driving `/update_weights` under `backend: ipc` parses the metadata in full and
then raises

```
NotImplementedError: IPC weight receive is not yet implemented on the worker
side; use backend='nccl' for now.
```

which matches the source (`weight_transfer/manager.py`, the `else` branch of
`update()`). `nccl` is therefore the only wired backend and the engine must be
disaggregated onto separate GPUs — but intra-node, so the "no cluster" claim
survives at a cost of ~33% of generation throughput rather than a second
machine ([5b](#5b-what-disaggregation-actually-costs)). Measured in
[Phase 2](#phase-2-validate-the-weight-sync-path). The `nccl` transport then
turned out not to transfer at all
([Phase 2b](#phase-2b-the-nccl-data-plane-does-not-transfer)), which blocks the
loop regardless of topology.

<a id="a7"></a>
**A7 — Someone has to stand up a trainer.** Nothing in this repository trains
anything, and nothing here proposes to: the reward functions
([`examples/rl/`](../examples/rl/)) are deliberately scorers that a trainer
calls, not a training loop. Phase 4 needs a verl or slime deployment, a GRPO
configuration, checkpointing, and an owner for the run itself. That is a
separate piece of work with a separate owner, and until it has one the phased
path stops at Phase 3 no matter what the engine does.

## Known gaps

- **The weight transport does not work, and that is now measured rather than
  assumed.** The control plane, its lifecycle guards, its validation and
  pause/resume are measured against a live server in
  [Phase 2](#phase-2-validate-the-weight-sync-path). The transport is measured
  in [Phase 2b](#phase-2b-the-nccl-data-plane-does-not-transfer) and it is
  broken: `/update_weights` returns 200 having moved nothing and loads
  uninitialised device memory into the model, filed upstream as
  [tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373).
  `ipc` is settled and negative.
  So "weights can be updated in place" is **established for the control plane
  and refuted for the transport** — the opposite of the previous position,
  which assumed the transport worked because the metadata contract did.
- **TP > 1 weight receive is unrun**, because it is blocked rather than
  deferred: sharded receive cannot be evaluated while single-rank receive
  transfers nothing. It is the first thing to run after an upstream fix.
- **The committed recipes cannot show a real length distribution**, because
  `dataset: random` prompts do not induce EOS. Their throughput and sample-count
  numbers are valid; `generated_tokens_*` will read as a constant at the cap.
  ShareGPT is the cheap fix, task-shaped prompts the right one.
- **TPOT and ITL are not meaningful under `n > 1`.** The client concatenates all
  choices and treats every chunk gap as an inter-token interval. Still reported,
  because suppressing metrics per-mode would make the metric set depend on the
  configuration; not to be read as per-token latency.
- **`--extra-body` outside rollout can still change the load silently.** Named
  above; a pre-existing hole this change neither widens nor closes.
- **No perf gates on rollout metrics, by decision.** An earlier revision of this
  branch added `mean_output_tokens_per_request` and `generated_tokens_p50` to
  `_METRIC_POLICIES` in `scripts/ci/eval_lib.py`. They have been removed again.
  That table is not a description of which metrics exist, it is the set
  `refresh_baselines --perf-gate` arms automatically, from a single observation
  and a flat margin — so listing a name elects to gate it.

  A generated length is the wrong thing to arm that way. Under EOS-respecting
  generation at temperature 1.0 it is the policy's choice and varies between
  runs by construction, so a margin-derived floor flaps; and on the `random`
  dataset both rollout recipes use, nothing induces EOS, so the value sits near
  `output_len * n` and a floor under it measures the recipe rather than the
  engine. The invariant actually worth enforcing — a collapsed policy answering
  every request with an immediate EOS — is already covered by
  `min_mean_output_tokens`, a hand-set per-step floor that fails the trial as
  `rollout_output_too_short`. A threshold someone chose beats one derived from a
  baseline. Both metrics remain captured for trends, which is the right
  treatment for a diagnostic.
- **The 8B concurrency figure is extrapolated.** Section 5 marks it; Phase 1
  removes it.
- **`ignore_eos: false` reaches the `random` dataset only under `rollout`.** The
  bench CLI forces EOS to be ignored for that dataset after parsing, so no argv
  reaches it; main rejects the combination outright for that reason. Rollout is
  exempt because it takes the one route that does work — `ignore_eos: false` in
  `--extra-body`, which is merged over the forced value, and which rollout
  reserves so no `bench_args` copy can shadow it. Outside rollout the rejection
  stands, and `sharegpt` is unaffected either way.
