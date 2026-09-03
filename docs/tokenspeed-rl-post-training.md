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
under [What was built](#6-what-was-built).

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
  ([Phase 2b](#phase-2b--the-nccl-data-plane-does-not-transfer), filed upstream
  as [tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373)).
  The engine choice is still defensible on the shape of the API; the loop is
  blocked on an upstream fix.
- **The domain is debugging, and the consumer is CIA with aorta inside it.**
  Both were open when this was first written; [A2](#a2) settled the domain and a
  CIA architecture diagram settled the consumer
  ([3.0](#30-what-cia-is-and-where-a-post-trained-model-sits-in-it)). CIA is the
  Cluster Intelligence Agent, exposing aorta over MCP, API and CLI, driving
  Waitcheck / ConSan / ASAN / UBSan / ROCgdb / RocJITsu against MI355, with an
  AORTA chatbot front end that shows **output logs, root cause and the fix**.
- **The output contract is a pair, and the two free rewards map onto it.** Root
  cause is triage classification, labelled deterministically by
  `aorta.probe.classifier`; fix is mitigation correctness. That pair is now the
  spine of [4](#4-the-domain-and-the-training-signal), with proposal validity as
  a free gate over both, and recipe synthesis demoted from headline to
  capability check. The demotion costs the best reward this repository can
  offer, which is worth stating plainly rather than absorbing quietly.
- **The corpus is now the critical path, and it is nearly empty.** Counting
  probe *and* sanitizer artifacts: 18 archived labelled runs, of which exactly
  **one** evidences a real defect. But the repository already ships deliberate
  race reproducers with committed expected verdicts, so the corpus is *generable*
  cheaply even though it does not exist
  ([4.6](#46-what-the-corpus-actually-contains)). Not one archived artifact
  comes from a workload the diagram names, so stratifying by workload family
  matters as much as covering failure categories.
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
- **The consumer contract is pinned from source, and the integration seam is
  measured.** `aorta agent`'s `LiteLLMProposer` is the contract that exists
  today; its prompt, schema and validation are transcribed in
  [3.1](#31-what-the-current-consumer-sends), and
  [3.4](#34-the-current-contract-versus-the-target-one) reconciles it with the
  chatbot's root-cause-and-fix target — they are the same pair, compressed.
  Pointing the agent at a self-hosted model needs **no code change** — two
  environment variables — verified against a mock OpenAI endpoint rather than
  assumed.

Related: [TokenSpeed under AORTA](tokenspeed.md) for the probe routes and the
container's operational hazards, [TokenSpeed serving benchmarks](tokenspeed-serving.md)
for the workload this extends and every measured number quoted below,
[Aorta Probe Agent](agent/aorta-probe-agent.md) for the agent loop that is the
first consumer, and whose "Relationship to Cluster-Scale Agent Systems" section
already anticipates being invoked by a cluster agent — which is what CIA is.

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
| The reward functions | Separate repo, but **importing aorta as a library** | The rewards are "is this proposal on contract", "does this triage read match the classifier", "did this mitigation fix it" — calls into `aorta.agent`, `aorta.probe.classifier` and `aorta.registry`, which is what makes them cheap |
| Standing the rollout engine up, and proving it serves | **aorta** (`tokenspeed_serve`) | Already does exactly this for fixed-length serving; rollout is a load shape it did not express |
| Measuring rollout throughput, length distribution, bring-up | **aorta** | This is the measurement the cost model above depends on, and nothing else in the stack reports it |
| Weight-transfer correctness and cost | **aorta**, later | A probe route, not a workload — see [Phase 2](#phase-2--validate-the-weight-sync-path) |
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
| Weight reload without cold start | **Yes, purpose-built — but `nccl` only**, `ipc` raises `NotImplementedError` ([Phase 2](#phase-2--validate-the-weight-sync-path)) | `runtime/engine/weight_transfer/`, `runtime/entrypoints/control_server.py:399+` |
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

[Phase 2](#phase-2--validate-the-weight-sync-path) found the control plane works
on this image on gfx950 and costs 1–5 ms against a 322 s cold start, but that
**`ipc` is not implemented** — its receive path raises — so of the two backends
named above only `nccl` is wired, and the colocated deployment this section
implies is unavailable.

[Phase 2b](#phase-2b--the-nccl-data-plane-does-not-transfer) then found that
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

## 3. The consumer: CIA, with aorta inside it

Two answers settled this section. Asked what the model is for
([A2](#a2)), the answer was:

> "debugging vertical, can we post train to become a good model for use with
> aorta llm agent?"

And an architecture diagram supplied for **CIA — the Cluster Intelligence
Agent** — settled what CIA is ([A1](#a1)). It is not a separate system
consuming aorta from outside. **AORTA is the box inside CIA**, and Sleuth is
described as something similar.

### 3.0 What CIA is, and where a post-trained model sits in it

```
   CIA: Cluster Intelligence Agent
   ┌─────────────────────────────────┐        Waitcheck  ◄── highlighted
   │  M   A   C  │                  │        ConSan     ◄── highlighted
   │  C   P   L  │      AORTA       │───────  ASAN                        MI355
   │  P   I   I  │                  │        UBSan          ────────────►
   └─────────────────────────────────┘        ROCgdb
            │  workloads  │                   RocJITsu                    Log
            └─────────────┘                   ...                      (3rd phase)

   Front end is an AORTA chatbot for input, and shows
   output logs, root cause and the fix.

   workloads: Fremont (Gsplat, MIOpen conv2D, ...), Ads, TBD
```

Four things in that picture change this plan.

**CIA exposes aorta over MCP, API and CLI.** Three interfaces down its left
edge. The CLI is what `aorta agent` already is. MCP is the interesting one for
serving: if CIA reaches aorta over MCP, a post-trained model could be reached
through that path rather than only through the proposer's LiteLLM call. Noted as
an integration option and **not designed for** — there is nothing in this tree
that serves aorta over MCP today, and speculating about a schema that does not
exist is how the previous version of this document went wrong about CIA.

This is also not a new idea in the repository. `docs/agent/aorta-probe-agent.md`
already anticipates it, distinguishing fleet-wide cluster intelligence from the
probe agent's single-repro search and recording "cluster agent could invoke
`aorta agent` with a frozen argv" as the future interop path. The diagram
confirms that shape.

**The output contract is a pair: root cause *and* fix.** That is the annotation
verbatim — the chatbot "will show output logs, root cause and the fix". It is
more than the mitigation-name proposal the current proposer asks for, and the
relationship between the two is worked out in
[3.4](#34-the-current-contract-versus-the-target-one).

**The evidence is sanitizer and debugger output, not just probe verdicts.**
Waitcheck and ConSan are highlighted in the diagram, with ASAN, UBSan, ROCgdb
and RocJITsu behind them, all pointed at MI355 hardware. That makes this
project's sanitizer work on-domain rather than adjacent, and it changes the
corpus inventory materially — see
[4.6](#46-what-the-corpus-actually-contains).

**The workloads are not TokenSpeed.** Fremont (with Gsplat and MIOpen conv2D
beneath it), Ads, and TBD. TokenSpeed is the workload this document was written
around and it does not appear in the diagram at all. The generalisation
consequences are in [4.6](#46-what-the-corpus-actually-contains); the short
version is that a model trained only on TokenSpeed triage is being trained on
the wrong distribution.

`Log` is marked as a **3rd phase** on the right of the diagram. Taken only as a
sequencing hint: log analysis is planned later, so a reward built on structured
detector and sanitizer output rather than raw log text is aligned with the
current phase. Not read as more than that.

### 3.1 What the current consumer sends

What follows is the contract `aorta agent` enforces **today**, read from
`src/aorta/agent/llm.py`, `policy.py` and `loop.py` rather than summarised. It
is the machine-checkable contract, and it is a subset of the chatbot's target
contract rather than the same thing —
[3.4](#34-the-current-contract-versus-the-target-one) reconciles them.

One `litellm.completion` call per iteration, with
`response_format={"type": "json_object"}` and two messages. The system message
is a fixed string:

```
You are an AORTA probe agent. Propose ONLY registered mitigation names from the
candidate list. Never propose shell commands or argv. Return strict JSON with
keys: category, hypothesis, next_mitigations (list of strings), confidence
(0-1), stop (bool). category must be one of: ['checkpoint_race',
'illegal_mem', 'launch_error', 'oom_fragment', 'perf_regression', 'rccl_hang',
'thermal_throttle', 'unknown'].
```

The user message is `json.dumps(..., indent=2)` of exactly four keys —
`symptom`, `cell_summaries`, `candidates`, `already_tried`. `candidates` is
already narrowed to what is still available (allowlist minus tried minus the
`none` baseline), so the model is never offered a mitigation it cannot use.

`cell_summaries` is the entire evidence the model gets, and it is narrow.
`_read_cell_summaries` builds one dict per probe cell with six keys and nothing
else:

```json
{
  "cell_name": "none-none",
  "verdict": "fail",
  "failure_detectors_fired": ["tier2:hang", "tier4:collective_timeout"],
  "warn_detectors_fired": [],
  "capture": {"stderr_tail": "..."},
  "exit_code": null
}
```

That matters for the reward design more than anything else in this section: the
model is not given raw logs, it is given the classifier's own detector IDs plus
a capture excerpt. Detectors are unioned across trials, and the `capture` shown
is from the first *failing* trial rather than `trial_0`, so a single bad trial
in an otherwise passing cell is not hidden. The task is therefore "read the
classifier's evidence and act on it", not "read a log".

### 3.2 What it demands back, and what it does with a bad answer

Five keys: `category`, `hypothesis`, `next_mitigations`, `confidence`, `stop`
(plus an optional `stop_reason`). But the *stated* contract and the *enforced*
contract differ, and the gap is where a reward has to live.

`AgentStep.from_dict` repairs rather than rejects. A `stop` that is not a real
JSON boolean becomes `False` (so `"stop": "false"` cannot prematurely end the
search — `bool("false")` is `True`, which the code comments call out). A
`next_mitigations` that is not a list becomes `[]` rather than being exploded
into characters by `list("tf32_off")`. A non-numeric `confidence` becomes `0.0`.
A null or blank `category` becomes `"unknown"`. Those defences are correct for a
serving path — the audit trail has to survive a bad provider — but they mean the
consumer reports success on input it silently repaired.

Then there are three ways a proposal can end the run:

| What the model sends | What happens | How loud |
|---|---|---|
| Unparseable, or a non-object | Caught, becomes a safe stop with `stop_reason: agent_requested`; the parse error is recorded in `hypothesis` | Silent to the operator |
| A mitigation name that is not in the offered candidates | Dropped by `filtered = [m for m in step.next_mitigations if m in remaining]` before validation. If that empties the list, `run_agent_loop` reads it as a decision to stop and reports `agent_stop` with the model's own hypothesis as the recommended action | **Entirely silent** |
| A `category` outside the closed set | `AgentPolicy.validate_step` raises `PolicyViolation`; the loop catches it and reports `policy_stop` | Loud |

The middle row is the important one. A hallucinated-but-plausible mitigation
name — `rccl_p2p_disable`, say, which sounds exactly like the 22 registered
names but is not one of them — does not error. It ends the search early, and
the report blames the model's own hypothesis. This is the same shape of trap as
the `nccl` transport defect in [Phase 2b](#phase-2b--the-nccl-data-plane-does-not-transfer):
a failure that returns success. It is the reason the format reward in
[4.1](#41-the-gate-on-both-halves--proposal-validity) is scored
explicitly rather than delegated to the consumer.

What the model never does is decide pass/fail. `aggregate_cell_verdict` sets the
verdict, `winning_mitigation` detects the fix, and a passing `none-none`
baseline short-circuits the loop *before* the proposer is consulted at all. The
model is advisory: a bad proposal costs a wasted probe cell, never a wrong
verdict. That is what makes this a safe first consumer.

### 3.3 Pointing it at a self-hosted model: no code change

This was recorded as an open integration gap. It is now measured, and the answer
is better than expected.

`LiteLLMProposer` passes neither `api_base` nor `api_key`, so it inherits
LiteLLM's environment resolution. Setting two variables is sufficient:

```bash
export OPENAI_API_BASE=http://<engine-host>:<port>/v1
export OPENAI_API_KEY=unused-but-must-be-set
aorta agent --llm-backend litellm --llm-model openai/<served-model-name> ...
```

Verified against a mock OpenAI-compatible endpoint: the request arrives at
`POST /v1/chat/completions` on the self-hosted address, with
`response_format: {"type": "json_object"}` intact, and the reply parses back
into an `AgentStep` unchanged. Both `openai/<name>` and a bare model name route
there. The `openai/` prefix is stripped on the wire, so **the name after the
slash must match what the engine advertises**, not what the CLI default says.

Three caveats, none blocking:

- `OPENAI_API_BASE` is process-global. It redirects *every* OpenAI-routed call
  in the process, so a hosted judge model and a self-hosted policy cannot
  coexist in one run. An explicit `--llm-api-base` (about eight lines through
  `LiteLLMProposer.__init__` and the CLI) removes that limitation and makes the
  endpoint appear in the audit log. Recommended, but not required to integrate,
  so it is recorded here rather than done as part of this work.
- With the variable set, a `--llm-model gpt-4o-mini` default silently goes to
  the local engine. The flag would say one thing and the traffic do another,
  which the agent report cannot currently distinguish.
- The serving engine must accept `response_format: {"type": "json_object"}`.
  TokenSpeed's OpenAI-compatible route is what would serve this, and that
  parameter is a dependency of the integration rather than an optional extra —
  without it the call errors rather than degrading.

### 3.4 The current contract versus the target one

These are not the same contract, and the difference decides how much of the
reward can stay automatic. Treat the proposer's JSON schema as the **current**
contract — it exists, it is enforced, and it is machine-checkable — and the
chatbot's root-cause-and-fix as the **target** one.

| | Current (`aorta agent` proposer) | Target (CIA chatbot) |
|---|---|---|
| Root cause | `category`, one of eight closed labels | Free-text root cause, shown to an operator |
| Fix | `next_mitigations`, registered names only | "The fix", presumably prose plus an action |
| Evidence shown | `cell_summaries`: detector IDs, verdict, capture excerpt, exit code | Output logs from Waitcheck / ConSan / ASAN / UBSan / ROCgdb / RocJITsu |
| Consumer | A search loop that runs the next probe cell | A human reading a chat response |
| Gradable | Fully, by code in this tree | Only the parts that reduce to a closed set |

The good news first: the two contracts **agree on structure**. The current
schema's `category` is a compressed root cause and `next_mitigations` is a
compressed fix, so a model trained on the proposer's contract is being trained
on the target contract's skeleton, not on something orthogonal. The pair is the
same pair.

Where they diverge is worth naming plainly, because it is the ceiling on
automatic reward:

**The target's root cause is free text; the current one is a closed label.** A
category from eight options is gradable by exact match. "The GEMM kernel is
missing an `s_waitcnt` before reading LDS at offset 0x2f0" is not, by any
mechanism in [4.1](#41-the-gate-on-both-halves--proposal-validity) or
[4.2](#42-the-root-cause-half--triage-classification).
It is the [4.5](#45-free-text-diagnosis--needs-humans) problem, which has no
automatic reward.

**The target's fix is an action, not a name.** `next_mitigations` is one of 22
registry entries, checkable in microseconds. "Add an `s_waitcnt vmcnt(0)`
before the load" is a code change, and verifying it means applying it and
re-running — the expensive path of
[4.3](#43-the-fix-half--mitigation-correctness).

**The target sees richer evidence.** The current proposer gets detector IDs; the
chatbot gets sanitizer and debugger output. That is *more* structure, not less —
a ConSan `Finding` carries a `code`, `severity`, `kernel_name`, `code_object`
and `entry_offset` (`instrumentation/rocjitsu_sanitizers/models.py`) — so this
divergence is an opportunity rather than a problem. Attribution against a
finding's `code` and `kernel_name` is exactly as gradable as attribution against
a detector ID.

**The practical consequence, and the recommendation taken:** train against the
current contract, and treat the target's free-text halves as a presentation
layer over it rather than as a separate training objective. A model that emits a
correct `category` plus correct cited evidence plus a correct registered
mitigation has produced a root cause and a fix — it has produced them in
structured form, which a chatbot front end can render into prose, and which a
reward can grade exactly. Training directly on the prose form would forfeit
every automatic label this plan is built on, in exchange for a judge model on a
narrow technical domain, which is where reward hacking shows up first.

What that leaves genuinely open is the **response schema the chatbot renders**:
whether it expects structured fields it formats, or free text it displays. If
the former, the target contract collapses almost entirely into the current one
and this plan needs no change. If the latter, a presentation layer has to be
written, and the free-text quality of it is ungraded. Recorded in
[A1](#a1) as the remaining piece.

## 4. The domain and the training signal

The domain is **debugging**: triage, diagnosis and failure analysis
([A2](#a2)), delivered through CIA's chatbot as **a root cause and a fix**
([3.0](#30-what-cia-is-and-where-a-post-trained-model-sits-in-it)). That pair is
the spine of this section, because the output contract *is* the pair and the two
free rewards this repository can offer map onto it one to one:

| Half of the contract | Reward | Labelled by | GPU per sample |
|---|---|---|---|
| **Root cause** | Triage classification: the verdict, and the evidence that justifies it ([4.2](#42-the-root-cause-half--triage-classification)) | `aorta.probe.classifier`, deterministically | none |
| **Fix** | Mitigation correctness: did the proposal make the repro pass ([4.3](#43-the-fix-half--mitigation-correctness)) | A probe cell's own verdict | minutes, or none if harvested offline |

Both halves sit behind one shared gate — the proposal is well-formed, its
category is in the closed set, its mitigation resolves in the registry
([4.1](#41-the-gate-on-both-halves--proposal-validity)) — which is free and
teaches nothing on its own.

That correspondence is the design, not a convenience. It also explains why the
ranking changed. The previous version of this section ranked recipe synthesis
first, on the grounds that a generated recipe either loads, validates and
dry-runs or it does not — the most exactly machine-checkable reward the
repository can offer. That ranking optimised for *how automatic the reward is*.
The domain answer points at the loop's own task instead, and the honest
consequence is that the best reward available is no longer the most relevant
one. The cost of that demotion is stated in
[4.4](#44-recipe-synthesis--now-a-capability-check-not-the-domain) rather than
absorbed quietly.

Full ranking, with status:

| # | Signal | Labels from | GPU per sample | Status |
|---|---|---|---|---|
| [4.1](#41-the-gate-on-both-halves--proposal-validity) | Proposal format and validity | The consumer's own code | none | Built |
| [4.2](#42-the-root-cause-half--triage-classification) | Triage classification | `aorta.probe.classifier` | none | Built, no corpus |
| [4.3](#43-the-fix-half--mitigation-correctness) | Mitigation correctness | A probe cell's verdict | minutes | Not built, needs archive |
| [4.4](#44-recipe-synthesis--now-a-capability-check-not-the-domain) | Recipe synthesis | Loader + dry-run | ~1 s | Built, demoted |
| [4.5](#45-free-text-diagnosis--needs-humans) | Free-text diagnosis | Humans or a judge | n/a | Deliberately last |

### 4.1 The gate on both halves — proposal validity

**Prompt:** the `cell_summaries` payload from [3.1](#31-what-the-current-consumer-sends).
**Completion:** the five-key JSON object of [3.2](#32-what-it-demands-back-and-what-it-does-with-a-bad-answer).

This is the outer layer, and it is the cheapest reward in the plan: strict JSON,
a `category` inside the closed autopsy set, and mitigation names that resolve in
the registry *and* sit inside the candidate set the loop offered. Zero GPU,
microseconds per sample, no labelling — the contract is code.

It is scored despite the consumer already validating, because the consumer
validates quietly. As [3.2](#32-what-it-demands-back-and-what-it-does-with-a-bad-answer)
sets out, an invented mitigation name is silently filtered and ends the search
as `agent_stop`; a mistyped field is silently repaired. A reward that delegated
to the consumer would score both as success. So the ladder in
[`examples/rl/proposal_reward.py`](../examples/rl/proposal_reward.py) measures
the contract as *stated*, while recording what the consumer would actually do
with each proposal:

| Tier | Check | Reward |
|---|---|---|
| 1 | Parses as a JSON object | 0.2 |
| 2 | The five demanded keys, with the demanded types | 0.4 |
| 3 | `category` is in the closed set | 0.6 |
| 4 | A non-empty mitigation list, every name in the registry | 0.8 |
| 5 | Every name still available, `confidence` in [0, 1] | 1.0 |

It imports `AUTOPSY_CATEGORIES`, `AgentStep`, `AgentPolicy` and
`get_mitigation` rather than restating any of them, so a new category or
mitigation changes the reward in the same commit — the same seam discipline the
other two scorers use.

The ceiling is the point, and it is asserted as a test: a policy that returns
one fixed valid proposal every time scores **1.0** and would be accepted by the
loop every time, while diagnosing nothing. Format is a gate, not a signal. It
belongs first because it is free and because failing it wastes a real search,
not because it teaches anything.

### 4.2 The root-cause half — triage classification

**Prompt:** a failing run's `cell_summaries`.
**Completion:** the verdict, and the detectors that justify it.

This is the **root-cause half** of the chatbot's contract, and the primary
substantive reward: given the evidence the agent will actually hand a model, say
what happened and why. `aorta.probe.classifier` is deterministic and is the
source of truth for pass/fail by design, so every archived probe run is a
labelled example with **no human labelling at all**.

It is the root-cause half in compressed form rather than in the chatbot's prose
form — a verdict plus cited evidence, not "the GEMM kernel is missing an
`s_waitcnt`" — and [3.4](#34-the-current-contract-versus-the-target-one) argues
that is the right thing to train on: the structured form is what a front end
renders, and it is the only form a reward can grade exactly.

**The sanitizer path is a second, richer label source for this same half**, and
the CIA diagram makes it on-domain rather than adjacent. A `SanitizerReport`
carries an `overall_verdict` from the same three-way-plus vocabulary
(`pass`/`warn`/`fail`/`not_checked`/`error`) and a tuple of `Finding` objects,
each with a `code`, a `severity` (`warning`/`race`/`error`), a `message`, and
optionally a `kernel_name`, `code_object` and `entry_offset`
(`instrumentation/rocjitsu_sanitizers/models.py`). That is *more* structure than
a detector ID, not less: attribution against a finding's `code` and
`kernel_name` grades exactly as cleanly as attribution against
`tier4:collective_timeout`, and it points at a specific kernel and offset, which
is much closer to a root cause an engineer would accept.

`triage_reward.py` now labels from `sanitizer_report.json` as well as
`result.json`, and `--runs` collects both from one tree. The seam is *stronger*
on this side: `SanitizerReport.from_dict` recomputes `overall_verdict` as the
max-ranked check verdict and **raises** when the stored value contradicts the
recomputation, so a rotted report cannot be trained on at all — it fails to
load and is named. On the probe side the same disagreement is only flagged as
`stale`.

Two deliberate choices. The verdict vocabulary stays the sanitizer's own
(`pass`/`warn`/`fail`/`not_checked`/`error`) rather than being mapped onto the
probe's three-way split, because `warn` has no probe equivalent and inventing
one would be a judgement the tools did not make. And cited evidence is
namespaced — `waitcheck:wait_hazard`, not `wait_hazard` — so attribution reads
like a detector ID and cannot be confused with one.

This is the only part of the reward stack that runs on real archived data
today, because the six committed survey reports are the only real labelled
evidence in the tree ([4.6](#46-what-the-corpus-actually-contains)).

[`examples/rl/triage_reward.py`](../examples/rl/triage_reward.py) scores two
terms — verdict exactness at 0.6 and attribution F1 at 0.4 — and recomputes
every label from the recorded detector IDs through `partition_detectors` and
`verdict_from_detectors` rather than trusting the stored `verdict` field. That
catches corpus rot: an archived run whose stored verdict disagrees with today's
precedence rules is flagged `stale` and reported instead of silently trained on.

Two things about this reward are worth stating precisely, because both are easy
to get wrong:

**Attribution is what stops "right answer, wrong reason".** Verdict alone is a
three-way choice and a policy can guess it from surface cues — a negative exit
code, the word "timeout" — then invent a justification. F1 over the cited
detector IDs docks exactly that. The fixtures include a run where
`tier3:vram_growth` fired as a *warn* alongside a genuine segfault, and a policy
that cites every detector it can see, warns included, scores lower than one that
cites only the failure signals. Advisory detectors are evidence about the run,
not justification for the verdict, and only `tier3:vram_growth` is advisory —
`tier3:thermal_throttle`, which reads like a performance note, is a failure
detector.

**`category` is deliberately not scored here, even though the contract demands
it.** There is no deterministic labeller for the autopsy category. The only
detector-to-category mapping in the tree is `_infer_category_from_detectors`, a
keyword heuristic used solely by the offline `FakeLLMProposer` — it reads
`"tier2" in joined` as `rccl_hang`, for instance, which is true of any tier-2
hang whatever caused it. Scoring against it would train the model to reproduce a
crude heuristic and call that success. Category correctness needs either human
labels or a rule table with an owner, and until it has one the reward stops at
verdict and attribution, which are genuinely derived. This is the one piece of
the consumer contract that remains unbacked by a label — see [A1](#a1).

The fixtures are shaped like the vertical: one failure per autopsy category the
contract enumerates, using only detector IDs the tiers can actually emit, which
is pinned by a test that collects the real vocabulary from the tier modules'
constants. What is missing is the corpus, and only the corpus —
[4.6](#46-what-the-corpus-actually-contains).

### 4.3 The fix half — mitigation correctness

**Prompt:** a failing run's evidence.
**Completion:** the mitigation that fixes it.

This is the **fix half**, and the one whose reward costs real hardware: "did the
proposed mitigation make the repro pass?" is answered by running the probe cell,
which is a GPU node for minutes per sample. As a training reward at that price
it is unusable; per-sample cost has to come down by orders of magnitude, not
percentages.

Note the same compression as the root-cause half. The chatbot's "fix" is
presumably an action or a code change; a registered mitigation name is the
closed-set form of one. For the environment-knob failures aorta's registry
covers — 22 of them, from `tf32_off` to `hsa_no_sdma` — the name *is* the fix.
For a missing `s_waitcnt` in a kernel it is not, and nothing in this plan grades
a patch. That boundary is worth being explicit about: this reward teaches
"which knob", not "which line".

**Can it be made cheap offline?** Yes in principle, and the mechanism already
exists. A probe run's own artifacts record the answer: `winning_mitigation`
parses the `<mitigation>-<diagnostic>` cell directory name back out, and the
loop writes a `converged` event naming the winner into `agent_log.jsonl`. So any
completed agent search yields one labelled `(evidence → winning mitigation)`
example for free, and a probe matrix yields one per cell — a mitigation was
applied to a failing cell and the outcome was recorded. Harvesting that costs no
new GPU time at all. The cost moves entirely to *assembling the archive*.

What it would take:

- A harvester that walks archived run directories and emits
  `(cell_summaries_before, winning_mitigation)` pairs from `matrix.json`,
  per-cell `result.json` files and `agent_log.jsonl` `converged` events. This is
  a few hundred lines and needs no hardware. It is not written, because there is
  nothing to point it at yet.
- The archive itself. This is the blocker, and it is worse than "small": see
  below.
- A reward that accepts *any* mitigation which made the cell pass, not just the
  one the historical search happened to land on first. Probe matrices often
  contain several passing cells, and treating the archived winner as uniquely
  correct would punish a right answer for being a different right answer.

Until the archive exists this stays an evaluation metric rather than a training
reward, which is the same conclusion as before — but for a different and more
tractable reason. It is no longer "inherently too expensive"; it is "cheap once
a corpus exists", and the corpus is now the thing to buy.

### 4.4 Recipe synthesis — now a capability check, not the domain

This is the demotion, and it should be said plainly rather than absorbed: recipe
synthesis was ranked first in the previous version of this plan and is the best
reward the repository can offer. A generated recipe either parses, loads,
resolves its mitigations, passes the workload's own validation and dry-runs, or
it does not, at five graded tiers with microsecond-to-second cost and no
labelling. Nothing in the debugging vertical is that exactly checkable.

It is no longer the headline because it is not the task. Writing a recipe is an
authoring skill; the agent loop never asks for one. Keeping it first would have
meant training hardest on the thing easiest to grade rather than the thing that
was asked for — which is the classic way a reward design goes wrong.

It keeps a real role, in two places. First, as a **capability check**: a policy
that cannot emit valid YAML against a schema will not reliably emit valid JSON
against one either, and tiers 1–5 are a cheap, harshly-graded probe of exactly
that. Second, as the **memorisation canary**: the novelty gate in
[`examples/rl/recipe_reward.py`](../examples/rl/recipe_reward.py) already
demonstrates a policy earning full tier marks by reproducing a committed recipe
verbatim, and refuses it. That failure mode is not specific to recipes — a
debugging policy can equally learn to emit the most common `(category,
mitigation)` pair in the corpus — so the gate stays as the worked example of the
problem.

The scorer is **not** being rebuilt. It stays as it is, in the role it now has.

### 4.5 Free-text diagnosis — needs humans

"Given this report, what went wrong and why" is the thing that would impress in a
demo and the thing with no automatic reward. The `docs/` corpus supports
retrieval-augmented answering, but grading a free-text explanation needs a human
or a judge model, and judge-model rewards on a narrow technical domain are where
reward hacking shows up first. Deliberately last.

Note that the `hypothesis` field in the proposal contract is exactly this
problem in miniature: it is free text, the consumer stores it verbatim, and it
becomes the operator's recommended action when the search stops. It is
ungradable by any of the mechanisms above, and it is deliberately left
unscored — a proposal earns full marks in [4.1](#41-the-gate-on-both-halves--proposal-validity)
with a useless hypothesis, which is a known and accepted hole.

### 4.6 What the corpus actually contains

[A4](#a4) was previously filed as a soft assumption: "does a probe archive
exist?", affecting whether triage classification is a main signal or a footnote.
With the vertical settled on debugging, both [4.2](#42-the-root-cause-half--triage-classification)
and [4.3](#43-the-fix-half--mitigation-correctness)
depend on it, so it is now the critical path. It was surveyed rather than
assumed. The result is worse than "small".

The first pass at this counted probe artifacts only. The CIA diagram makes the
sanitizer fleet on-domain — Waitcheck and ConSan are the two highlighted tools —
so the inventory was re-run to include sanitizer reports. That changes the count
and, more importantly, changes the outlook.

**Probe artifacts**, under this node's run areas, excluding source trees:

| Artifact | Count | Usable as a label |
|---|---|---|
| `result.json` (per-trial probe results) | 12 | **No** — all 12 are `verdict: pass` |
| ...of which fired any failure detector | 0 | — |
| ...distinct cells represented | 2 | — |
| `matrix.json` | 16 | Metadata only; 3 are perf-sweep schema, not probe cells |
| `agent_log.jsonl` / `agent_report.md` | 0 | No completed agent search exists |

**Sanitizer artifacts**, committed in `recipes/sanitizers/survey/reports/`:

| Report | Overall verdict | Findings |
|---|---|---|
| `gemm_f32_waitcheck` | `warn` | 64, all `wait_hazard`, severity `warning` |
| `lds_reduce_waitcheck` | `pass` | 0 |
| `tiny_vecadd_waitcheck` | `pass` | 0 |
| `gemm_f32_consan` | `error` | 0 — ConSan did not run in that environment |
| `lds_reduce_consan` | `error` | 0 |
| `tiny_vecadd_consan` | `error` | 0 |

So the revised count is **18 labelled runs, not 12** — and exactly **one of them
evidences a real defect**: the 64 `wait_hazard` findings in
`gemm_f32_waitcheck`, which are 64 findings from one scenario rather than 64
independent examples. Three of the six sanitizer reports are `error`, meaning
the tool did not complete, which is the sanitizer analogue of a probe `error`
verdict and is not a diagnosis. No committed report evidences a ConSan race.
None of the findings carry a `kernel_name`, so the finest-grained attribution
the schema allows is not populated in the data that exists.

The headline conclusion therefore stands, slightly softened: **there is
essentially no labelled failure corpus** — one scenario, not zero, across 18
runs. Every other archived result is a passing smoke, timing or survey run,
which is unsurprising in hindsight: these are runs made to validate a harness,
and a harness is validated by making it succeed.

**What genuinely improves is the generative story, and it improves a lot.** The
sanitizer path ships deliberate known-bad reproducers and committed ground truth
for them:

- `recipes/sanitizers/fixtures/repro/consan_lds_race.hip` and
  `consan_lds_race_2wave.hip` — intentional LDS races.
- `recipes/sanitizers/fixtures/expected/verdict_baselines.json` — committed
  expected verdicts: `consan_racy` → `fail` with an "auto replay diagnostic"
  finding shape, `consan_clean` → `pass`, `waitcheck_gemm` → `warn` with a
  "missing s_waitcnt" shape.
- Eleven `daily-*` sanitizer recipes that drive them, plus a
  `sanitizers-nightly` workflow that runs them.

That is the expensive half of corpus generation already paid for, at least for
the race and wait-hazard categories: a reproducer whose failure is intentional
has ground truth by construction, and the expected verdict is in the repository
to check against. Running those recipes produces labelled failures on demand.
It is a materially better position than the probe path, where no reproducer for
any autopsy category exists.

So the shortfall is still most of the thing. A usable first corpus needs, per
category, on the order of 30 distinct failing runs to train on and a held-out
set on top; call it 240 failing examples as a floor, against roughly one
scenario today. But the path to the first few dozen is committed code rather
than new engineering.

### The workload breadth problem

The diagram's workloads are **Fremont** (with Gsplat and MIOpen conv2D beneath
it), **Ads**, and TBD. None of them appear in this repository: `rg -i` finds no
`fremont`, no `gsplat`, and no `ads` workload; `miopen` appears only in
environment probing and Buck introspection, not as a workload class. The
workload classes that do exist are `gpu_smoke`, `hrx`, `hrx_perf`, `inference`,
`llm_determinism`, `race`, `tokenspeed`, `tokenspeed_serve` and `training`.

And the archive that exists covers fewer still. The 12 probe results are
`training` (emulated DDP and timing) and `gpu_smoke`; the perf sweeps are
`tokenspeed_serve`; the sanitizer reports are three synthetic HIP kernels
(`gemm_f32`, `lds_reduce`, `tiny_vecadd`). **Not one archived artifact comes
from a workload named in the diagram.**

What that means for generalisation is worth stating rather than hoping about. A
model trained solely on TokenSpeed serving triage learns the detector
vocabulary, which transfers — the classifier tiers are workload-independent by
construction — but it also learns the *co-occurrence statistics* of one
workload's failures, and those do not transfer. An RCCL collective timeout in a
tensor-parallel LLM server and a correctness fault in a MIOpen convolution
present through different detectors, with different capture text, at different
tiers. A policy that has only seen the former will read the latter's evidence
against the wrong prior.

The mitigation is cheap to state and not cheap to execute: the corpus has to be
**stratified by workload family, not just by category**. Concretely, the
recommendation taken here is that the synthetic generator below should emit each
failure signature under at least two distinct workload shapes, and that any
real-hardware corpus should include at least one non-TokenSpeed family before
it is used for training. That is a constraint on the corpus design rather than
extra node-hours, so it costs nothing to adopt now and is expensive to retrofit.

It also argues against over-fitting this plan to TokenSpeed at all. TokenSpeed
is the rollout *engine* — that is
[Phase 2](#phase-2--validate-the-weight-sync-path)'s subject and it remains
blocked on [tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373)
— but it is not the debugging *domain*, and the diagram makes that explicit by
not mentioning it.

**What generating it would cost.** Three routes now, not two — the sanitizer
reproducers add a middle option that is both cheap and real:

*Synthetic, ~0 node-hours.* Probe mode runs arbitrary `subprocess_argv`, and
`recipes/probe/probe-template-bash.yaml` exists for exactly that. A script that
exits non-zero, stalls past the hang window, or prints the stderr signatures the
tier-4 patterns match produces genuine `result.json` files with real detector
firings, on CPU, in seconds. 240 cells is minutes of wall clock and perhaps two
days of authoring the generators. The objection — that this trains
pattern-matching on the classifier's vocabulary rather than on real failures —
is weaker here than it first looks, because [4.2](#42-the-root-cause-half--triage-classification)
already concedes that agreeing with the classifier *is* the task inside the
agent loop. What synthetic data cannot supply is realistic `capture` text and
realistic co-occurrence between detectors, and those are precisely what
distinguishes a plausible read from a correct one.

*Sanitizer reproducers, ~2–4 node-hours, and genuinely real.* The committed
`consan_lds_race*.hip` fixtures and the eleven `daily-*` sanitizer recipes
produce true tool output — real ConSan replay diagnostics, real Waitcheck
hazards — against ground truth already in the repository. This is the best
labelled data available per hour spent, and it needs no new engineering: run the
recipes, archive the `sanitizer_report.json` files, extend `triage_reward.py` to
label from them. It covers races and wait hazards well and covers nothing else,
so it is a strong start rather than a corpus.

*Real, ~24 node-hours plus the hard part.* Genuine failures across the remaining
categories need genuinely broken GPU workloads. At three trials per cell and
roughly two minutes per cell including startup, 240 failing cells is about 24
node-hours — cheap, and not the real cost. The real cost is **authoring
reliably-reproducing failures for the categories the sanitizer fixtures do not
cover**, which is engineering time on hardware and is the line item to plan
around. Some are easy to induce (`oom_fragment`, `launch_error`);
`checkpoint_race` and `thermal_throttle` are not, and
`perf_regression` has no detector that evidences it at all.

The recommendation, taken rather than deferred, and revised by the sanitizer
finding:

1. **Run the committed sanitizer reproducers and archive their reports.** A few
   node-hours, no new engineering, and it yields *real* tool output against
   ground truth already in the repository. This is now the first step, ahead of
   the synthetic generator, because it is nearly as cheap and not synthetic.
   The scoring half is already done: `triage_reward.py` labels from
   `sanitizer_report.json` today, and
   `--runs recipes/sanitizers/survey` scores the six committed reports.
2. **Build the synthetic generator** for the categories the fixtures do not
   cover, stratified across at least two workload shapes per signature. This
   unblocks the reward end-to-end at essentially no cost and turns "no corpus"
   into "a corpus of known limitations".
3. **Buy real runs** for the categories where synthetic capture text is least
   defensible, and for at least one workload family from the diagram.

Do not wait for an archive to appear — nothing in the survey above suggests one
is accumulating, and three of the six existing sanitizer reports are `error`
because the tool was unavailable, which is itself a sign that these artifacts
are not being produced under conditions anyone is curating for reuse.

### 4.7 What this implies about the run

Three things follow that a naive plan would get wrong.

**The base model must already clear the format gate.** With
[4.1](#41-the-gate-on-both-halves--proposal-validity) as the outer
layer, a policy that cannot emit a strict JSON object gets a flat 0.2 on nearly
every sample, every group's advantage is zero, and nothing is learned. This is
the same argument the previous version of this plan made about YAML, and it
transfers directly — which is the residual value of
[4.4](#44-recipe-synthesis--now-a-capability-check-not-the-domain) as a
capability check. Qwen3-8B clears it comfortably; Qwen3-0.6B is borderline and
is the wrong size for the demo even though it is the right size for measuring
the engine. A short supervised warm-up is the cheap insurance, and also the
honest baseline: if supervised fine-tuning on a few hundred proposals gets most
of the way, the RL half needs to justify itself.

**The reward must reject a memorised answer.** In the recipe domain this was
verbatim corpus copying, which the novelty gate refuses. In the debugging domain
it is duller and harder to see: a policy can learn the corpus's most common
`(category, mitigation)` pair and emit it unconditionally. That scores full marks
at [4.1](#41-the-gate-on-both-halves--proposal-validity) — asserted as
a test — and beats the always-`pass` floor at
[4.2](#42-the-root-cause-half--triage-classification) whenever
the corpus is skewed. Both scorers therefore print a degenerate-policy baseline
next to any real score, and a reward must never be read on its own. The
synthetic corpus of [4.6](#46-what-the-corpus-actually-contains) has to be
balanced across categories for this reason, not merely large.

**Nothing above needs the weight transport.** All three built scorers are
offline functions over archived artifacts, so the reward work proceeds in
parallel with [tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373)
rather than behind it. The corpus is the binding constraint, and it needs no
engine at all.

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

Three scorers now exist in [`examples/rl/`](../examples/rl/), all offline and
all importing aorta rather than restating it: the proposal-contract ladder
([4.1](#41-the-gate-on-both-halves--proposal-validity)), the
classifier-agreement reward
([4.2](#42-the-root-cause-half--triage-classification)), and
the recipe reward in its reduced role
([4.4](#44-recipe-synthesis--now-a-capability-check-not-the-domain)).

What Phase 3 still needs, in order:

1. **The corpus generator** ([4.6](#46-what-the-corpus-actually-contains)). No
   hardware, no decisions, and nothing substantive can be measured without it.
   This is the first task, ahead of any training.
2. **The mitigation-outcome harvester**
   ([4.3](#43-the-fix-half--mitigation-correctness)),
   which turns archived probe matrices into `(evidence → winning mitigation)`
   pairs. Cheap, but pointless before 1.
3. **Supervised fine-tuning** of Qwen3-8B on the assembled corpus, measuring
   proposal-tier pass rates and triage reward against the printed degenerate
   baselines. **This is the gate on the whole effort:** if SFT gets most of the
   way, the RL phase has to justify itself against that number rather than
   against zero.

None of this is blocked by
[tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373), which
is why it is worth doing now.

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
has been. Two answers arrived since the first draft: [A2](#a2) settled the
domain, and a CIA architecture diagram substantially settled [A1](#a1). Between
them they promoted [A4](#a4) from a soft item to the critical path. What is left
needs a decision or an artifact that is not ours:

| # | Blocked on | Why it blocks | Costs if wrong |
|---|---|---|---|
| [A4](#a4) | **A corpus of failing runs** | Both substantive rewards need labelled failures; the survey found 18 archived runs and **one** real defect among them ([4.6](#46-what-the-corpus-actually-contains)) | Nothing substantive can be trained. This is now the critical path |
| [A1](#a1) | The chatbot's response schema; what Sleuth is; a labeller for the autopsy `category` | Structured-versus-free-text decides whether the reward stays fully automatic ([3.4](#34-the-current-contract-versus-the-target-one)) | A presentation layer, and the category term stays unscored; verdict and attribution survive intact |
| [A3](#a3) | Model choice | Sets the memory and throughput arithmetic in [5](#5-cost-does-the-claim-hold) and [5b](#5b-what-disaggregation-actually-costs) | The cost table, not the design |
| [A7](#a7) | Who stands up the trainer | No trainer exists here; verl/slime integration, the GRPO loop and its checkpointing are outside this repository | Phase 4 cannot start |

[A5](#a5) (whether Toyota is a separate demo) remains open but affects scope
rather than feasibility.

A4 is worth separating from the rest, because it is the only one that does not
need a decision from anyone — it needs work, the work needs no hardware, and
[4.6](#46-what-the-corpus-actually-contains) recommends a route and takes it.
The others genuinely wait on someone else.

One item that is **not** blocked on Manoj and should be raised anyway: the
`nccl` weight transport does not work on this image
([Phase 2b](#phase-2b--the-nccl-data-plane-does-not-transfer)). That is an
upstream TokenSpeed defect, it blocks the loop at every topology and model size,
and it is ours to file rather than his to decide. Filed as
[tokenspeed#1373](https://github.com/lightseekorg/tokenspeed/issues/1373).

<a id="a1"></a>
**A1 — the consumer. SUBSTANTIALLY ANSWERED by the CIA architecture diagram.**
CIA is the **Cluster Intelligence Agent**, and AORTA is the box inside it — not
a separate system consuming aorta from outside. CIA exposes aorta over MCP, API
and CLI; a workloads box sits beneath it; and it drives a tool fleet —
Waitcheck and ConSan highlighted, then ASAN, UBSan, ROCgdb, RocJITsu — against
MI355 hardware. The front end is an AORTA chatbot that takes input and shows
**output logs, root cause and the fix**. Sleuth is described as something
similar. The full reading, and what each part changes, is
[3.0](#30-what-cia-is-and-where-a-post-trained-model-sits-in-it).

This resolves what the previous version of this document could only guess at.
It also vindicates the guess: the contract designed against — `aorta agent`'s
proposer, transcribed in
[3.1](#31-what-the-current-consumer-sends) — is the right skeleton, because the
chatbot's root-cause-and-fix pair is the same pair in prose form
([3.4](#34-the-current-contract-versus-the-target-one)).

Three things remain genuinely open, none of them blocking:

- **The response schema the chatbot renders.** Whether it expects structured
  fields it formats into prose, or free text it displays verbatim. If
  structured, the target contract collapses almost entirely into the current one
  and nothing here changes. If free text, a presentation layer has to be written
  and its quality is ungraded. This is the one that would change the plan.
- **What Sleuth actually is.** Described as "something similar" to CIA, which is
  enough to not design around it and not enough to design for it.
- **The autopsy `category` has no deterministic labeller.** Unchanged by the
  diagram, and the narrowest of the three. The contract demands one of eight
  categories, but nothing in the tree derives the correct category from a run's
  evidence; the only mapping is `_infer_category_from_detectors`, a keyword
  heuristic used solely by the offline `FakeLLMProposer`, and training against
  it would teach the heuristic rather than the diagnosis
  ([4.2](#42-the-root-cause-half--triage-classification)). Closing it needs
  human category labels or a rule table someone owns. Until then the reward
  scores verdict and attribution, which are genuinely derived, and leaves
  category unscored.

Also noted and deliberately not designed for: **MCP is a plausible serving
path.** If CIA reaches aorta over MCP, a post-trained model might be reached
that way rather than through the proposer's LiteLLM call. Nothing in this tree
serves aorta over MCP today, so this is recorded as an integration option only.

<a id="a2"></a>
**A2 — the domain. ANSWERED.** Asked whether the target is producing AORTA
artifacts or conversational expertise, the answer was:

> "debugging vertical, can we post train to become a good model for use with
> aorta llm agent?"

So: the domain is debugging — triage, diagnosis, failure analysis — and the
consumer is `aorta agent`. Both halves of the question are settled by that one
sentence, and neither answer is the one this plan originally optimised for. The
consequences are worked through in
[4](#4-the-domain-and-the-training-signal): the reward ranking changes, recipe
synthesis is demoted to a capability check, and the corpus becomes the critical
path. The RAG risk this assumption was hedging against does not materialise —
"for use with aorta llm agent" is an artifact-producing task with a code
contract, which is the good case.

<a id="a3"></a>
**A3 — Model size.** Qwen3-8B is assumed: the largest Qwen3 in the measured set,
and comfortably able to emit parseable YAML, which the reward design requires.
Qwen3-0.6B is used in the committed recipes because they measure the engine, not
the model. If the demo needs a specific size, the cost table changes.

<a id="a4"></a>
**A4 — a corpus of failing runs. SURVEYED: ESSENTIALLY EMPTY, BUT GENERABLE.
Now the critical path.** This was previously a soft assumption about whether triage
classification would be a main signal or a footnote. With the debugging vertical
settled it is load-bearing for both substantive rewards, so it was measured.
Counting probe artifacts **and** the sanitizer reports the CIA diagram makes
on-domain: **18 archived labelled runs, of which exactly one evidences a real
defect** — 64 `wait_hazard` findings from a single Waitcheck scenario. The 12
probe results are all `pass` with no detectors fired; of the 6 sanitizer
reports, 2 are `pass` and 3 are `error` because ConSan did not run. Zero
completed agent searches.

The full survey, the workload-breadth problem, and what generating a corpus
would cost by each of three routes are in
[4.6](#46-what-the-corpus-actually-contains). Two things changed once sanitizer
artifacts were included:

- The count went from 12 to 18 and from zero real defects to one. Materially
  better than "nothing", still not a corpus.
- The **generative** position improved a lot. The repository already ships
  deliberate race reproducers (`consan_lds_race.hip`,
  `consan_lds_race_2wave.hip`), committed expected verdicts for them
  (`fixtures/expected/verdict_baselines.json`), eleven `daily-*` sanitizer
  recipes and a nightly workflow. That is the expensive half of corpus
  generation already paid for, for races and wait hazards.

Also established, and a constraint on the corpus rather than on hardware: **not
one archived artifact comes from a workload the diagram names.** Fremont, Gsplat
and Ads appear nowhere in the tree, and MIOpen only in environment probing. A
model trained on TokenSpeed serving triage learns a workload-independent
detector vocabulary but a workload-specific set of co-occurrence priors, so the
corpus must be stratified by workload family and not only by failure category.

The recommendation, already taken in 4.6: run the committed sanitizer
reproducers first and label from `sanitizer_report.json`, then build a synthetic
generator for the uncovered categories, then buy real runs — including at least
one non-TokenSpeed family.

What is *not* blocked on anyone: this needs no decision and no hardware. It
needs someone to do it, and it is the highest-value unblocked work in the plan.

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
[Phase 2](#phase-2--validate-the-weight-sync-path). The `nccl` transport then
turned out not to transfer at all
([Phase 2b](#phase-2b--the-nccl-data-plane-does-not-transfer)), which blocks the
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

- **There is almost no corpus of failing runs, and that is now the binding
  constraint.** Surveyed rather than assumed, across probe and sanitizer
  artifacts: 18 archived labelled runs, of which **one** evidences a real defect
  (64 `wait_hazard` findings from a single Waitcheck scenario); the 12 probe
  results are all `pass`, and 3 of 6 sanitizer reports are `error` because
  ConSan did not run ([4.6](#46-what-the-corpus-actually-contains), [A4](#a4)).
  Both substantive rewards are written and tested and neither can be evaluated
  on real data. This gap is ahead of the engine defect in priority, because it
  needs no hardware and no decision from anyone — only work, and the committed
  race reproducers make the first increment cheap.
- **No archived artifact comes from a workload CIA names.** Fremont, Gsplat and
  Ads are absent from the tree; MIOpen appears only in environment probing. The
  detector vocabulary is workload-independent, but failure co-occurrence
  statistics are not, so a corpus stratified only by category would train a
  TokenSpeed-shaped prior. Recorded as a corpus-design constraint in
  [4.6](#46-what-the-corpus-actually-contains) rather than as extra node-hours.
- **The autopsy `category` is unscored, because nothing can label it.** The
  proposal contract demands one of eight categories and no deterministic
  labeller for it exists; the sole mapping in the tree is a keyword heuristic
  used by the offline fake proposer. Training against that would teach the
  heuristic, so the triage reward stops at verdict and attribution
  ([A1](#a1)). This is a genuine hole in an otherwise fully-labelled signal, and
  it needs human labels or an owned rule table to close.
- **The format reward can be saturated without diagnosing anything.** A fixed
  valid proposal scores 1.0 at
  [4.1](#41-the-gate-on-both-halves--proposal-validity) and is accepted
  by the agent loop every time. This is asserted as a test rather than left
  implicit, and it is why the format term is a gate and never the reward.
- **The weight transport does not work, and that is now measured rather than
  assumed.** The control plane, its lifecycle guards, its validation and
  pause/resume are measured against a live server in
  [Phase 2](#phase-2--validate-the-weight-sync-path). The transport is measured
  in [Phase 2b](#phase-2b--the-nccl-data-plane-does-not-transfer) and it is
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
