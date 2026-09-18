# First end-to-end run of the sanitizer-selection-and-routing reward

The "test it concretely before claiming it" step for the use case
[4.0](tokenspeed-rl-post-training.md#40-the-six-prioritised-use-cases-and-how-ready-each-one-is)
ranks first. A real model was served on TokenSpeed, driven through `aorta
agent`'s own proposer against the nine committed sanitizer scenarios, and every
output scored with both committed graders.

**The headline is not the score. It is that the score is 1.0000 and means
nothing.** Qwen3-8B tops out the proposal ladder on all 45 samples, and so do
two fixed templates that read no input at all. Every one of the nine
per-scenario groups has zero within-group spread, so a GRPO advantage computed
over them is identically zero. The reward cannot train a policy because it
cannot tell any two policies apart.

Three things have to change before this use case is trainable, and none of them
is more corpus. They are in [What to fix](#5-what-to-fix).

> **Status, 2026-09-08.** Both reward-design fixes have landed, and the rollout
> now samples. Three things changed the picture:
>
> 1. **The reward is no longer saturated.** The ladder spans 0.5400 to 1.0000
>    where it was flat at 1.0000.
> 2. **The rollout was not sampling at all**, and the temperature was not the
>    fix. The engine's sampling backend defaults to `greedy` on non-NVIDIA
>    hardware and ignores `temperature`, `top_p` and `seed` *silently*
>    ([§5.4](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default)).
>    With `--sampling-backend triton` the effective sample size goes from 9 to
>    45, and **within-group spread is non-zero on all nine groups** of the wide
>    candidate-set condition — so a GRPO gradient exists for the first time.
> 3. **The model is now beaten by a two-line constant on both conditions.** It
>    beat one under greedy decoding by 0.0111, and that margin turned out to be
>    an artifact of a single deterministic completion. This is the honest
>    headline and it is not good news.
>
> §1 through §4 are the record of the original run, with corrections marked
> where later measurement overturned them — including
> [§2](#2-does-the-format-gate-hold-yes--but-the-serving-stack-holds-it-not-the-model),
> whose format-gate conclusion is one tier off. [§5](#5-what-to-fix) is the
> record of the fixes, the numbers, and the failures. The caveats in
> [§7](#7-what-nine-scenarios-can-and-cannot-support) apply unchanged: the
> effective n is now genuinely 45 draws, but they are still **9 prompts**, and
> nothing here has an error bar worth quoting.

## 0. Run provenance

| | |
|---|---|
| node | `cv350-rck-g03-c08-08.rck.dcgpu`, partition `meta64`, gfx950, Slurm job 33582 |
| model | `Qwen/Qwen3-8B`, 16 GB in 5 safetensors shards, bf16, TP 1, one GPU |
| engine | `lightseekorg/tokenspeed-amd@sha256:60c12e37…` (the digest pinned in `recipes/tokenspeed/tokenspeed-serve-rollout.yaml`) |
| served name | `Qwen/Qwen3-8B`, read back from `/v1/models` rather than assumed |
| litellm model id | `openai/Qwen/Qwen3-8B` — the `openai/` prefix is stripped on the wire, leaving exactly the advertised name |
| proposer | `aorta.agent.llm.LiteLLMProposer`, unmodified |
| corpus | `examples/rl/corpus/triage.jsonl`, 9 scenarios, `aorta_commit` `ba73dab` |
| samples | 5 completions per scenario = 45 proposals, plus 9 triage answers, per condition |
| bring-up | 171 s from `docker run` to `/health_generate`; 0.63 s median per proposal, 113 median completion tokens, 0 transport errors |
| scripts | `examples/rl/serve_for_rollouts.sh`, `examples/rl/run_e2e.py` |
| raw results | `/apps/vikhande/rl-e2e/results/{faithful,hard,triage-*}.json` |

**Second run, 2026-09-08 (sampled).** Same node, model, engine digest, corpus
and scripts; Slurm job 33781. Adds `--sampling-backend triton` and a rollout
temperature, which is what the original run was missing
([§5.4](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default)).
Bring-up 300 s cold; four driver conditions plus two probes in 12 min total.

| | |
|---|---|
| conditions | `{faithful,hard} x {t=0.7, t=1.0}`, 9 scenarios x 5 samples each |
| raw results | `/apps/vikhande/rl-e2e/results/{faithful,hard}-t{0.7,1.0}.json` |
| probes | `examples/rl/probe_seed.py`, and the raw-HTTP and engine probes under `/apps/vikhande/rl-e2e/` |
| re-score | `examples/rl/rescore_e2e.py` (offline, no GPU) |

```bash
examples/rl/serve_for_rollouts.sh hold      # now defaults --sampling-backend triton
python examples/rl/run_e2e.py \
    --corpus examples/rl/corpus/triage.jsonl \
    --base-url http://127.0.0.1:8000/v1 \
    --model openai/Qwen/Qwen3-8B --samples 5 \
    --temperature 0.7 \
    --out results/faithful-t0.7.json
```

Reproduce:

```bash
examples/rl/serve_for_rollouts.sh hold          # in its own Slurm step
python examples/rl/run_e2e.py \
    --corpus examples/rl/corpus/triage.jsonl \
    --base-url http://127.0.0.1:8000/v1 \
    --model openai/Qwen/Qwen3-8B --samples 5 \
    --out results/faithful.json
```

### The proposer was driven, not imitated

This was the load-bearing design constraint and it held: the proposals come
from `LiteLLMProposer.propose()`, so the system prompt, the user payload, the
`response_format` and the post-filter are all aorta's. Two environment
variables were enough, as [3.1](tokenspeed-rl-post-training.md#31-what-the-current-consumer-sends)
predicted — no code change.

One wrinkle worth recording. `propose()` returns an `AgentStep`, which
`from_dict` has already coerced and the proposer has already filtered, and that
is the wrong input for the ladder: tiers 1 and 2 measure the *raw* wire text,
and by the time an `AgentStep` exists a truncated object has become a safe stop
and a string `stop` has become `False`. So `litellm.completion` is wrapped
rather than replaced — the proposer calls it exactly as it ships, and the
wrapper records the request and the raw response on the way past. The recorded
request is in every results file and shows only `model`, `response_format` and
the two message roles, which is what proves the contract went out unaltered.

## 1. The blocker found on the way in: a default server cannot serve `aorta agent` at all

`LiteLLMProposer.propose` sends `response_format={"type": "json_object"}` on
every call. TokenSpeed answers that with an HTTP 500:

```
Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not
supported when the server is launched with --grammar-backend none
```

`ServerArgs.grammar_backend` defaults to `"none"`, and the only other choice
the engine offers is `"xgrammar"`. So `aorta agent --llm-backend=litellm`
pointed at a stock `tokenspeed serve` fails on **every** request, and it fails
by raising out of `propose()` rather than degrading — the proposer's only
`except` is for a missing `litellm` import.

This does not show up in any existing `tokenspeed-serve-*` recipe, because
`tokenspeed bench serve` drives `/v1/completions` and never asks for a response
format. The integration is measured, the *consumer's* use of it is not.

The fix is one flag, and `examples/rl/serve_for_rollouts.sh` now defaults it on.
Everything below was measured with `--grammar-backend xgrammar`.

> **This was not the only such default.** `--sampling-backend` defaults to
> `greedy` on non-NVIDIA hardware and makes sampled rollouts impossible, and it
> is the harder of the two to find because it never fails: the request is
> accepted and the argmax comes back. See
> [§5.4](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default).
> The generalisation is that the serving recipes configure this engine for
> *benchmarking*, where determinism and no grammar are both reasonable, and
> nothing in the tree configured it for *being consumed by an agent* until
> these two flags were set.

Two smaller bring-up traps, both already solved in `tokenspeed_serve.py` and
both re-discovered the hard way before its env block was read. Under `--user`,
an unset `TRITON_CACHE_DIR`/`HOME` surfaces as *"Triton is not supported on the
current platform"* — a broken-GPU-stack message on a node whose GPUs are fine —
and an unset `TORCHINDUCTOR_CACHE_DIR` raises `KeyError: getpwuid()` at import
of `torch._dynamo`. A third is Slurm's, not aorta's: a detached container
started inside an `srun` step is SIGKILLed when that step ends, so the endpoint
dies the moment bring-up returns. Hence `hold`.

## 2. Does the format gate hold? Yes — but the serving stack holds it, not the model

100% of 45 samples reach tier 5. Not one malformed object, missing key, wrong
type, out-of-set category, unregistered name or unavailable name.

| tier | what it tests | n | share |
|---|---|---:|---:|
| 0 | did not parse | 0 | 0% |
| 1 | parses, but schema wrong | 0 | 0% |
| 2 | schema right, category outside the set | 0 | 0% |
| 3 | category in set, name unregistered or list empty | 0 | 0% |
| 4 | name registered but not offered, or bad confidence | 0 | 0% |
| 5 | on contract | **45** | **100%** |

The plan's assertion that Qwen3-8B "clears it comfortably" is therefore
confirmed — with one large caveat that changes what it means. Tiers 1 and 2 are
**enforced by grammar-constrained decoding**, not produced by the model: with
`response_format=json_object` and xgrammar, the first emitted token must open a
JSON object, so malformed output is structurally impossible. A useful
side-effect is that Qwen3's reasoning trace cannot appear either — the `<think>`
block a Qwen3-class model would ordinarily emit ahead of its answer is
unreachable, `reasoning_content` came back empty on all 45, and the `--no-think`
condition built into the runner turned out to be unnecessary.

So the correct reading is: **the format half of the reward is satisfied by
configuration, and contributes no training signal at all.** That is not the
GRPO collapse the plan feared — all samples failing the format check and
scoring the same low value — but it is the same disease with the sign flipped.
All samples *pass* and score the same high value. The advantage is zero either
way.

> **Corrected 2026-09-08**, after re-running with a sampled rollout
> ([§5.4](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default)).
> Two claims above are too strong, and the boundary is one tier off.
>
> **What survives.** `parse_rate` is **1.0000 at every temperature tested, up
> to 1.2** — not one unparseable completion in 180 sampled proposals. Grammar
> -constrained decoding really does make malformed JSON structurally
> impossible, so tier 0 is unreachable and the `<think>` block stays
> unreachable with it.
>
> **What does not.** The grammar enforces **tier 1, not tier 2**. `json_object`
> mode constrains output to a syntactically valid JSON *object*; it says nothing
> about which keys that object carries. `{}` satisfies it perfectly. Under
> greedy decoding the model never emitted `{}`, which is why the distinction was
> invisible — but at temperature 1.0 it does, on 1 of 45 samples on the corpus
> candidate set and 2 of 45 on the full registry (the literal completions are
> `{}` and `{\n\n\n}`). So `schema_rate` falls to 0.9778 and 0.9556, and the
> table's "missing key: 0" is a property of greedy decoding, not of the gate.
>
> **And so "contributes no training signal at all" is false once the rollout
> samples.** At temperature 0.7 the format tiers still contribute nothing
> (`schema_rate` 1.0000), but the *ladder* does: 5 of 45 and 2 of 45 samples
> fall to tier 3 by proposing an empty mitigation list, and one names
> `debug_hip_dynamic queues_2` — a real registered name with a space where an
> underscore belongs, which is exactly the tier-4 hallucination the reward was
> built to catch and which greedy decoding never produced. The failure modes
> [§4](#4-what-the-model-actually-gets-wrong) called "real, and this model just
> does not produce them" are produced by this model. It needed a temperature,
> not a different model.
>
> The revised reading: **tiers 0–1 are enforced by configuration and carry no
> signal; tier 2 upward carries signal as soon as the rollout samples, and the
> amount depends on the temperature.** At 0.7 the signal is in tiers 3–5; at 1.0
> tier 2 starts contributing too, but by way of degenerate `{}` completions that
> are not worth training on.

## 3. Does the scoring discriminate? No. It is saturated, with no headroom

This is the finding that matters, and it is one of the two outcomes the brief
named as bad news.

| policy | reads the input? | mean proposal reward |
|---|---|---:|
| known-perfect oracle | — | 1.0000 |
| **Qwen3-8B, 45 samples** | **yes** | **1.0000** |
| `abstain_and_shotgun` template | no | 1.0000 |
| `abstain_and_pick_first` template | no | 1.0000 |
| `always_prose` template | no | 0.0000 |

The two abstain templates are two-line constants: `category: "unknown"`, an
empty hypothesis, the offered names, `confidence: 0.5`. They score exactly what
the model scores and exactly what the oracle scores. **A reward that assigns the
same value to Qwen3-8B, to a constant, and to a perfect answer has measured
nothing about any of them.**

Per-scenario groups, both conditions:

| scenario | n | mean | min | max | spread |
|---|---:|---:|---:|---:|---:|
| `consan-clean` | 5 | 1.000 | 1.000 | 1.000 | 0.000 |
| `consan-racy` | 5 | 1.000 | 1.000 | 1.000 | 0.000 |
| `consan-gemm` | 5 | 1.000 | 1.000 | 1.000 | 0.000 |
| `consan-lds-dispatch` | 5 | 1.000 | 1.000 | 1.000 | 0.000 |
| `consan-tiny` | 5 | 1.000 | 1.000 | 1.000 | 0.000 |
| `waitcheck` | 5 | 1.000 | 1.000 | 1.000 | 0.000 |
| `waitcheck-gemm` | 5 | 1.000 | 1.000 | 1.000 | 0.000 |
| `waitcheck-lds-dispatch` | 5 | 1.000 | 1.000 | 1.000 | 0.000 |
| `waitcheck-tiny` | 5 | 1.000 | 1.000 | 1.000 | 0.000 |

**9 of 9 groups degenerate.** Under GRPO the advantage of every sample in every
group is exactly zero, so the policy-gradient term is zero and the update is
whatever the KL penalty alone dictates. This does not train slowly. It does not
train.

Widening the candidate set from the corpus's 2 available names to all 20 in the
registry changed nothing — still 45/45 at tier 5, still 9/9 degenerate. The
saturation is a property of the ladder, not of how narrow the choice was.

### Updated 2026-09-08: two causes, both now addressed, one criterion still open

The 9-of-9 above had **two independent causes**, and this section attributed all
of it to the ladder. The second was that the rollout was not sampling at all:
every group's five completions were byte-identical, so the spread was zero for a
reason no reward could touch
([§5.4](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default)).

With the reward graded ([§5.1](#51-implemented-unknown-no-longer-earns-full-marks),
[§5.2](#52-implemented-a-precision-term-on-the-mitigation-block)) *and* the
rollout sampling, on the same nine scenarios and five samples each:

| condition | model mean | distinct completions | effective n | groups with non-zero spread |
|---|---:|---:|---:|---:|
| greedy, 2 offered | 1.0000 | 9/45 | 9 | **0 / 9** |
| greedy, 20 offered | 1.0000 | 9/45 | 9 | **0 / 9** |
| t=0.7, 2 offered | 0.8689 | 45/45 | 45 | 5 / 9 |
| t=0.7, 20 offered | 0.7299 | 45/45 | 45 | **9 / 9** |
| t=1.0, 2 offered | 0.8822 | 45/45 | 45 | 4 / 9 |
| t=1.0, 20 offered | 0.7046 | 44/45 | 45 | **9 / 9** |

Per-scenario within-group spread on the recommended setting (t=0.7, 20 offered),
against 0.000 everywhere before:

| scenario | mean | spread | scenario | mean | spread |
|---|---:|---:|---|---:|---:|
| `consan-clean` | 0.7400 | 0.3333 | `waitcheck` | 0.6800 | 0.1667 |
| `consan-gemm` | 0.8000 | 0.3600 | `waitcheck-gemm` | 0.6724 | 0.1667 |
| `consan-lds-dispatch` | 0.6867 | 0.2667 | `waitcheck-lds-dispatch` | 0.6694 | 0.2667 |
| `consan-racy` | 0.8149 | 0.1524 | `waitcheck-tiny` | 0.7400 | 0.0667 |
| `consan-tiny` | 0.7655 | 0.3200 | | | |

**No collapsed groups at any temperature**, and the effective n is now genuinely
45 rather than 9. So the GRPO advantage is non-zero on every group of the wide
condition: the policy-gradient term exists.

Two things this does *not* fix, both in
[§5.3](#53-what-the-re-score-showed-after-sampling):

- **The narrow condition still degenerates on 4–5 of 9 groups.** With two names
  offered, a proposal that abstains and names one or two of them scores exactly
  0.9 whatever else it says, so the reward has too few distinguishable states
  for sampling to separate. The spread that does appear comes from samples
  falling to tier 3. Widening the candidate set is what makes the precision term
  produce a continuum, which is why the 20-name condition reaches 9 of 9.
- **The model still loses to a constant** — and now on both conditions, where
  under greedy decoding it beat one of them. That is the substance of criterion
  1's failure and is discussed there.

### On the 0.629 floor

The brief's degenerate-policy reference of **0.629 does not reproduce.** The
always-`pass`-no-detectors floor on these nine scenarios is **0.5333**, from
`triage_reward.py --corpus examples/rl/corpus/triage.jsonl`, and the arithmetic
checks independently of the grader: always-`pass` is right on 4 of the 9, and
cites correctly on those 4 plus the 2 zero-finding `error` runs (an empty-empty
citation is 1.0 by the scorer's convention), so the reward is
`0.6·(4/9) + 0.4·(6/9) = 0.5333`. The synthetic fixtures give 0.2222 and the six
archived survey reports give 0.5333; nothing in the tree gives 0.629. Use 0.533
for this corpus.

It does not affect the conclusion. The observed 1.0000 is above every floor and
level with the ceiling, which is the problem.

## 4. What the model actually gets wrong

Nothing the graders can see — which is the point. What the *outputs* show is
another matter, and it is visible only in fields the reward ignores.

**Abstention, 40 of 45 (88.9%).** The model answered `category: "unknown"` on
eight of the nine scenarios. `unknown` is a member of `AUTOPSY_CATEGORIES`, so
it passes tier 3 and reaches 1.0. Tier 3 asks whether the category is *in the
set*, and the set contains a legal way to decline — so the tier is satisfiable
by refusing the task it exists to test.

The one scenario that got a real category was `consan-racy`, on all 5 samples:
`checkpoint_race`, for an intra-wave LDS race between instructions `0x8` and
`0x28`. That is the closest of the eight available names and still not right,
which is [the blocker](tokenspeed-rl-post-training.md#the-blocker-read-this-before-spending-another-node-hour)
showing through from the other side: there is no autopsy category for a
kernel-level data race, so the honest answers here are `unknown` and a wrong
label. The model picked both, and the reward paid 1.0 for each.

**Shotgunning.** On the corpus's 2-name set, 20 of 45 named both. On the
20-name registry the lists ran to 1, 3, 7, 9 and **11 mitigations at once** —
all registered, all offered, so all scored 1.0. The ladder has no precision
term: a proposal naming 11 candidates is worth exactly what the single right
name is worth. Most-proposed names were `debug_hip_dynamic_queues_1` (40/45),
`debug_clr_no_batch_cpu_sync` (35/45) and `debug_hip_dynamic_queues_2` (35/45),
chosen at much the same rate across LDS races, GEMM wait hazards and
zero-finding tool errors alike — which is not a targeted choice.

> **Corrected 2026-09-08**, while implementing [§5](#5-what-to-fix). This
> paragraph previously read "3, 4, 7, 9, 11 and in one case **17 mitigations at
> once**", and gave the cost as "17 iterations". Neither reproduces from
> `hard.json`: the observed list lengths are 1, 3, 7, 9, 11, the maximum is
> **11**, no proposal named 4 or 17, and the counts are per-scenario constants
> (all five samples of a scenario name the same list — see the
> [sampling defect](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default)).
> The cost model was also wrong in a way that mattered for the fix: `loop.py`
> appends **every** proposed name to the mitigation axis and runs a probe cell
> for each, then charges the whole proposal **one** unit of the iteration
> budget. So a k-name proposal costs k GPU cells inside a single iteration, and
> `check_iteration_budget` does not restrain shotgunning at all. That is a
> stronger argument for the precision term than the original sentence, and it is
> what `precision_credit` is derived from.

**`consumer_outcome`: all 45 `accepted`, 0 `silent_stop`, 0 `policy_stop`.** The
asymmetry the brief asked about — a proposal scored 0.8 that the real loop would
silently stop on — did not arise, because nothing scored below 1.0. The
committed *synthetic* corpus is the useful contrast: its 45 hand-built
proposals average 0.6 and split 18 `accepted` / 18 `silent_stop` / 9
`policy_stop`. The failure modes the ladder was designed around are real and
the ladder catches them; this model just does not produce them.

> **Corrected 2026-09-08.** The last sentence is wrong: this model does produce
> them, once the rollout samples
> ([§5.4](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default)).
> At t=0.7 the wide condition yields 2 `silent_stop` of 45 and the narrow one 5
> of 45; at t=1.0, 6 and 3. They arrive by two routes the ladder was built for:
> an **empty `next_mitigations` list**, which `run_agent_loop` reads as a
> decision to stop searching (2–5 samples per condition), and a **hallucinated
> name** — `debug_hip_dynamic queues_2`, a real registered mitigation with a
> space where an underscore belongs, which the proposer silently filters. A
> third category, `illegal_mem`, also appears (1–3 samples), so the claimed set
> is no longer just `unknown` and `checkpoint_race`. Greedy decoding hid all of
> this: it showed the single most likely completion per prompt, and that
> completion was always on contract. The abstention rate barely moves
> (37–39 of 45 against 40), so the headline of this section stands.

## 5. What to fix

Items 1 and 2 were **implemented on 2026-09-08** and the recorded proposals
re-scored under them; this section is now the record of that rather than a plan.
Items 3 and 4 are unchanged and still open. Item 5 is new, and it is the one
that changes the outlook.

### 5.1 Implemented: `unknown` no longer earns full marks

`proposal_reward.py` grades tier 3 instead of passing it. A category in the
closed set that is *not* `unknown` earns the whole 0.2 step; `unknown` earns
`ABSTENTION_CREDIT` (0.5) of it. Declining therefore costs **0.1**, and an
abstaining proposal tops out at 0.9 rather than 1.0.

The dock is deliberately small, and the reason is the trap in this fix. Both
options this section originally offered were rejected:

- **Excluding `unknown` from tier 3's accepted set** drops an abstention to 0.4
  while *any* in-set category — right or wrong — still earns 1.0. That is a 0.6
  gradient pointing straight at "invent a confident label", and on this corpus
  it is not hypothetical: [§4](#4-what-the-model-actually-gets-wrong) establishes
  that 8 of 9 scenarios have **no correct category available**, so the policy it
  trains is one that emits a wrong label instead of an honest `unknown`. That is
  worse for an operator and worse for the loop, which routes on the category. It
  would also put the reward in disagreement with `AgentPolicy.validate_step`,
  which accepts `unknown` — the exact drift `proposal_reward.py`'s docstring is
  built to avoid.
- **Gating the credit on "the evidence genuinely does not support a category"**
  is *not implementable*. That predicate needs the per-scenario category labels
  that do not exist. It is item 3 wearing a different hat, so this section
  offered as an alternative to fix 1 something that is really a restatement of
  fix 3.

A third candidate, not in the original list, was rejected on the data:
**coupling the credit to `confidence`**, docking an abstention that also claims
certainty. It is attractive — incoherence is checkable without labels, and it
does not push the policy toward a confident wrong label, because committing and
abstaining stay equally available. But the recorded model abstains at confidence
0.6–0.95 while both constant templates abstain at exactly 0.5, so the term ranks
a two-line constant **above** the model on 8 of 9 scenarios. It inverts the one
comparison this exercise exists to make. Worth revisiting once a correctness
signal exists to anchor it.

What survives is still not clean, and the residue should be stated rather than
buried: **partial credit keeps a wrong-but-specific label worth more than an
honest abstention** (a full step against half a step). It shrinks the perverse
gradient from 0.6 to 0.1 rather than removing it. Its real merit is graceful
degradation — when the category set is widened to cover kernel-level races, the
same term becomes correctness-sensitive with no rewrite.

### 5.2 Implemented: a precision term on the mitigation block

The tier 4–5 block is scaled by `precision_credit(k) = min(1, 2/k)` for a
proposal costing k probe cells. The cost model is the loop's, not a preference:
`loop.py` runs a probe cell per name and charges one unit of iteration budget
for the whole proposal, so k cells is k GPU cells and the budget does not
restrain it.

**k is the cell count, not the written length**, and the two differ: review
caught that `AgentPolicy.validate_step` drops `none` and collapses repeats
before `run_agent_loop` iterates, so charging for the raw list priced work that
never happens. `probe_cells` now mirrors the consumer. **None of the numbers
below move**: across all 1,185 proposals in the recorded rollouts, zero contain
a repeated name and zero contain `none`, so written length and cell count were
equal everywhere they were measured.

The free pair is what neutralises the perversity a brevity term invites. With no
correctness signal the reward cannot tell a right name from a wrong one, so any
brevity term makes a 1-name proposal beat a 2-name proposal that *contains* the
right name. `1/len(next_mitigations)` — the form this section suggested — puts a
0.2 reward cliff exactly there, the largest single step the term can produce. At
`FREE_MITIGATIONS = 2` the two tie instead, so the reward never pays a policy to
drop a correct name in order to look decisive.

**It relocates the perversity rather than removing it, and that has to be said
plainly.** At three names and up, a single confident wrong name still outscores
a list containing the right one. No function of the list's *shape* can fix that;
it needs a correctness signal. See [§5.5](#55-is-the-contracts-fix-half-worth-a-gpu-now).

One consequence is deliberate and worth flagging for anyone reading a score:
**the reward is no longer a pure function of the tier.** A wide enough sweep at
tier 5 can score below a precise proposal at tier 4, or below a tier-3 miss.
Tier and reward are therefore reported separately by `Score`.

### 5.3 What the re-score showed, after sampling

Read [§5.4](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default)
first: the numbers below are from rollouts that actually sample, which the
originals did not. Recommended setting **t=0.7**, both candidate-set conditions,
9 scenarios x 5 samples, re-scored under the graded reward.

| policy | reads input? | t=0.7, 2 offered | t=0.7, 20 offered |
|---|---|---:|---:|
| `oracle_contract_perfect` | no | **1.0000** | **1.0000** |
| `honest_abstainer` | no | 0.9000 | 0.9000 |
| `abstain_and_pick_first` | no | 0.9000 | 0.9000 |
| `abstain_and_shotgun` | no | 0.9000 | 0.5400 |
| **Qwen3-8B, 45 samples** | **yes** | **0.8689** | **0.7299** |
| `always_prose` | no | 0.0000 | 0.0000 |

The four criteria, at both temperatures and both conditions:

| # | criterion | t=0.7 narrow | t=0.7 wide | t=1.0 narrow | t=1.0 wide |
|---|---|:-:|:-:|:-:|:-:|
| 1 | abstain templates strictly below the model | fail | fail | fail | fail |
| 2 | reference at or near the top | **pass** | **pass** | **pass** | **pass** |
| 3 | non-zero within-group spread | fail (5/9) | **pass (9/9)** | fail (4/9) | **pass (9/9)** |
| 4 | model between the constants and the reference | fail | fail | fail | fail |

**Criterion 3 — the one this was for — now passes on the wide candidate set, at
both temperatures, on all nine groups.** No group collapsed to a single
completion at any temperature, and the effective sample size is 45. A GRPO
advantage over these groups is non-zero, so the policy-gradient term exists for
the first time. On the narrow set it still fails on 4–5 of 9 groups, for the
mechanical reason in [§3](#3-does-the-scoring-discriminate-no-it-is-saturated-with-no-headroom):
two offered names give the reward too few distinguishable states.

**Criterion 1 now fails on both conditions, where under greedy decoding it
passed on the narrow one.** This is the result worth dwelling on, and it is
worse news than the earlier report, correctly. The greedy pass was 0.9111
against 0.9000 — a margin of 0.0111 that came entirely from one scenario,
`consan-racy`, where the single deterministic completion happened to commit to a
category. With sampling, the model commits on that scenario only some of the
time and occasionally proposes an empty mitigation list elsewhere, so the mean
falls to 0.8689 (t=0.7) and 0.8822 (t=1.0) — **below the 0.9000 that a two-line
constant earns for declining to classify and naming one mitigation.**

So the earlier 3-of-4 was partly an artifact of a single deterministic
completion, exactly as suspected. The honest position is now 2 of 4 on the wide
condition and 1 of 4 on the narrow one — and the criterion that flipped from
pass to fail is the one measuring whether the reward prefers a real model to a
constant. It does not. Criterion 4 fails as a direct consequence of criterion 1
in every condition.

That is a stronger argument for the contract's fix half than the greedy run
made, not a weaker one: with genuine sampling and a real spread to learn from,
what the reward would now teach the policy is *to abstain and name one thing*,
because that is what scores 0.9. See
[§5.5](#55-is-the-contracts-fix-half-worth-a-gpu-now).

The always-`pass` triage floor is **0.5333**, unchanged, as it must be — both
reward fixes are inside `proposal_reward.py`.

### 5.3.1 The original greedy re-score, for the record

`examples/rl/rescore_e2e.py` re-scores the recorded raw completions offline — no
GPU, no server — reading `raw` and never the stored reward, so the "before"
column is the original run's own number.

```bash
python examples/rl/rescore_e2e.py \
    /apps/vikhande/rl-e2e/results/{faithful,hard}.json
```

Ordered by the faithful condition, which is the primary one — its candidate set
is the corpus's. Note that `hard` **reorders** the ladder, and that reordering is
the substance of criterion 1's failure below.

| policy | reads input? | faithful (2 offered) | hard (20 offered) |
|---|---|---:|---:|
| `oracle_contract_perfect` | no | **1.0000** | **1.0000** |
| **Qwen3-8B, 45 samples** | **yes** | **0.9111** | **0.7174** |
| `honest_abstainer` | no | 0.9000 | 0.9000 |
| `abstain_and_pick_first` | no | 0.9000 | 0.9000 |
| `abstain_and_shotgun` | no | 0.9000 | 0.5400 |
| `always_prose` | no | 0.0000 | 0.0000 |

Before: every row except `always_prose` was 1.0000. The four acceptance
criteria, honestly:

| # | criterion | faithful | hard |
|---|---|:-:|:-:|
| 1 | both abstain templates strictly below the model | **pass** | **fail** |
| 2 | contract-perfect reference at or near the top | **pass** | **pass** |
| 3 | non-zero within-group spread | **fail** | **fail** |
| 4 | model between the constants and the reference | **pass** | **fail** |

Three of four on the faithful condition, two of four on the widened one. What
each failure means:

**Criterion 1 fails on `hard`, and the way it fails is the finding.**
`abstain_and_pick_first` scores 0.9000 against the model's 0.7174. A two-line
constant that declines to classify and names one mitigation beats a real model,
*because* the model hedges and the constant does not. The precision term added
to punish hedging handed the win to a constant. Note also that
`abstain_and_pick_first` and `honest_abstainer` score **identically** — they are
the same policy up to the hypothesis text, and this reward does not read the
hypothesis. Criterion 4 fails on `hard` as a direct consequence of criterion 1.

**Criterion 1 passes on `faithful` by 0.0111, and the margin is worth
distrusting.** It comes entirely from `consan-racy`, the single scenario where
the model committed to a category instead of abstaining — and that category,
`checkpoint_race`, is *wrong*. So the model's whole advantage over a constant on
the primary condition is one confidently wrong label. The perversity is not
theoretical; it is what the passing number is made of.

**Criterion 2 passes, with a caveat that undercuts the word "perfect".** The
reference commits to a category and spends one cell, so it scores 1.0. But on 8
of 9 scenarios its category is *knowingly wrong*, because the closed set has no
name for the failure. There is no policy on this corpus that is both honest and
top-scoring: the ceiling is 1.0, the honest ceiling is 0.9, and the 0.1 between
them is what the reward currently charges for telling the truth.

**Criterion 3 fails, and no reward change can fix it** — see below.

The always-`pass` degenerate floor is **0.5333, unchanged**, and it must be:
fixes 1 and 2 are entirely inside `proposal_reward.py`, and the floor is a
`triage_reward.py` number. Confirmed by re-running
`triage_reward.py --corpus examples/rl/corpus/triage.jsonl`, and it still checks
independently as `0.6·(4/9) + 0.4·(6/9)`. The brief's 0.629 still reproduces
nowhere.

### 5.4 The rollout was not sampling, and the cause was a server default

**All five samples in every scenario group were byte-identical.** The 45
originally recorded proposals are 9 distinct completions, each repeated five
times, in both conditions.

The first explanation was that `LiteLLMProposer.propose` sends no `temperature`
and `run_e2e.py` injected none (`temperature_injected: null`). That is true and
it was not the cause. **Passing a temperature changed nothing**, and chasing why
found something worse.

#### It was not the client

litellm was suspected of dropping `temperature` when it equals the OpenAI
default of 1.0, which would have explained a run at `--temperature 1.0` coming
back byte-identical to the greedy one. Measured against a local echo server
rather than argued: litellm 1.100.0 puts `temperature`, `seed` and flattened
`extra_body` keys on the wire faithfully at every value, 1.0 included. Also
worth recording because it misleads: the `requests` block in every results file
is the *kwargs litellm was called with*, not the JSON that reached the engine,
so it cannot be used to prove what was sent.

#### It was a platform-conditional server default

`tokenspeed/runtime/sampling/registry.py`, in the pinned image:

```python
def _get_default_backend_name() -> str:
    if current_platform().is_nvidia:
        return "flashinfer"
    return "greedy"
```

On this hardware `is_nvidia` is false, so the sampling backend defaults to
**`greedy`**, which ignores `temperature`, `top_p`, `top_k` and `seed`
altogether. `/get_server_info` on the control port confirmed
`"sampling_backend":"greedy"` on the original run's configuration.

The decisive measurement was `n=8` in a *single* request at temperature 1.2:
eight identical choices, same batch, same cache state, same kernels. Counting
distinct completions across separate requests cannot establish this on its own,
because separate requests at temperature 0 also differ occasionally — 1–2 of 8 —
from floating-point non-determinism in greedy decoding as batch composition
varies. That noise is what an earlier reading of a 0.6 run mistook for sampling.
It is not sampling: it is argmax jitter, it is not controllable, and it is not a
draw from the policy distribution, which is what GRPO needs.

**Nothing warns.** The request is accepted, HTTP 200 comes back, and the
completion is the argmax. There is no failure for a caller to notice — only
completions that are all the same, which reads as a property of the model.

This is the same shape as
[§1](#1-the-blocker-found-on-the-way-in-a-default-server-cannot-serve-aorta-agent-at-all)
and arguably worse: `--grammar-backend none` fails loudly on every request,
while this one succeeds quietly forever. Both are defaults that make a stock
server unable to serve the use case, and both are invisible to the serving
recipes, which drive `/v1/completions` and want reproducible output anyway.
Being NVIDIA-conditional, it would never appear on an NVIDIA CI lane.

#### The fix, and what it bought

`examples/rl/serve_for_rollouts.sh` now passes `--sampling-backend triton` and
reads the value back from `/get_server_info` after bring-up, warning if the
engine reports `greedy` when something else was asked for — the same
"read it back rather than assume it" the script already applies to the served
model name. `--sampling-backend` accepts `greedy | triton | triton_full |
flashinfer | flashinfer_full`; `flashinfer` is CUDA-only, so `triton` is the
portable choice on ROCm.

Measured on the reconfigured engine, distinct completions from 8 separate
requests on the proposer's own prompt:

| temperature | greedy backend | triton backend |
|---|---:|---:|
| 0.0 | 2/8 | 1/8 |
| 0.3 | 1/8 | **8/8** |
| 0.6 | 1/8 | **8/8** |
| 0.7 | 1/8 | **8/8** |
| 0.8 | 1/8 | **8/8** |
| 1.0 | 1/8 | **8/8** |
| 1.2 | 1/8 | **8/8** |

Temperature 0 still collapses, correctly. Everything above it samples.

#### Two residual defects, reported rather than worked around

- **Seeds are still not honoured.** With the triton backend, the same seed twice
  gives *different* completions and different seeds diverge — i.e. the parameter
  is accepted and ignored, so the diversity is real but not reproducible.
  Measured for the OpenAI-standard `seed`, for `sampling_seed` (the name
  TokenSpeed's SGLang-compat layer shims onto its own seed) and for a plain
  `seed` in `extra_body`; none reproduces. `run_e2e.py` therefore derives a
  per-sample seed and sends it, and it currently has no effect: the plumbing is
  correct and the engine is where the gap is. A rollout is repeatable only at
  the level of its aggregate statistics, not sample by sample.
- **`n > 1` in one request returns identical choices** at every temperature,
  even with sampling working. Separate requests are unaffected, which is what
  the driver issues, so this costs throughput rather than correctness — one
  request per sample instead of one batched request per group.

Neither is filed upstream yet.

#### Where the temperature lives, and why not in the proposer

In `run_e2e.py`, as `--temperature`, off by default. Not in
`LiteLLMProposer`, which is the production path `aorta agent` uses to diagnose
real failures. For a diagnostic tool reproducibility is a feature: the same
evidence should yield the same recommendation, an operator should be able to
re-run a triage and get the same answer, and the nightly matrix already carries
an `llm_determinism` entry. Making the shipped agent stochastic to serve a
training harness would be a real behaviour change to the consumer, introduced
invisibly and for the wrong reason. Sampling diversity is a property of how
training data is collected, so it is threaded through the driver as an explicit
parameter and the proposer is untouched — asserted by a test that reads the
proposer's source for `temperature` and `seed`.

**0.7 is the recommended value.** Qwen3's own model card recommends temperature
0.7 for non-thinking generation, and the reasoning trace is unreachable here
anyway because grammar-constrained decoding cannot emit `<think>`
([§2](#2-does-the-format-gate-hold-yes--but-the-serving-stack-holds-it-not-the-model)).
It is inside the usual GRPO rollout range, it produced 8/8 distinct completions,
it reaches 9 of 9 groups with spread on the wide condition, and unlike 1.0 it
keeps `schema_rate` at 1.0000 rather than emitting degenerate `{}` completions
on 2–4% of samples. 1.0 was measured alongside it and is reported throughout;
it buys slightly more spread on the narrow condition and pays for it in
`{}` completions that are not worth training on.

#### Why this mattered more than it first looked

A reward is a function of the completion, and the loop state is constant inside
a group, so **identical completions earn identical rewards under any reward
function whatsoever.** Within-group spread was therefore exactly zero for
reasons that had nothing to do with saturation, and criterion 3 was
unsatisfiable on the original data before a single line was changed.
[§3](#3-does-the-scoring-discriminate-no-it-is-saturated-with-no-headroom)
attributed 9-of-9 degenerate groups to the reward's ceiling; that was one of two
independent causes, and this was the other.

The distinction is pinned by tests rather than argued: five copies of one
completion give zero spread, and four differing completions on the same
scenario and loop state give non-zero spread under the same grader. So the
reward was never what blocked criterion 3. `run_e2e.py` now reports
`distinct_completions` and `collapsed_groups` next to `degenerate_groups`, and
`effective_n`, so the two failure modes can never again be confused in a
results file: a collapsed group is a sampling defect, a degenerate group with
several distinct completions is the reward saturating.

`rescore_e2e.py` reports `distinct_completions` per group, fails under
`--check-determinism` when any group collapses to one, and reports
`spread_across_on_contract_policies` — the range the reward achieves on one
scenario across the policies that clear the format gate, **0.1000** on the
narrow condition and **0.4600** on the wide one, against 0.0000 before. That
was the substitute measurement while criterion 3 was unreachable. It is *not*
criterion 3, and now that real within-group spread exists it should be read as
what it is: a statement about the reward's range across policies, not about a
GRPO gradient.

### 5.5 Is the contract's fix half worth a GPU now?

**Yes — and the case is much stronger than before this exercise.**

The general shape of the two implemented fixes is the argument. With no
correctness signal, every available term scores **form**, and each form-based
term is gameable in its own direction:

| term | what it rewards | how it is gamed |
|---|---|---|
| category membership (before) | naming anything in the set | abstain — `unknown` is in the set |
| abstention credit (fix 1) | committing to a category | commit to a *wrong* category |
| precision (fix 2) | short mitigation lists | name one wrong thing confidently |
| calibration (rejected) | coherent confidence | emit a humble constant |

Fixing one direction opens another, and the re-score measured the trade rather
than predicting it: fix 2 broke the shotgun tie (`abstain_and_shotgun` fell from
1.0000 to 0.5400) and in the same move let a one-name constant beat the model.
Every row in that table is gameable because every row scores the *shape* of a
proposal, and shape is all that is observable without running the mitigation.

The plan deferred the fix half — scoring whether a proposed mitigation actually
resolves the reproducer — because it needs a probe cell and a GPU. That
deferral now looks like the expensive choice:

- It is the only term that is **not** gameable by form, because it is checked by
  execution rather than inspection.
- It is the only term that makes fix 2's residual perversity go away, rather
  than relocating it: with a resolve signal, a list containing the right name
  beats one wrong name on merit, and precision becomes a tie-break instead of
  the whole mitigation signal.
- It does **not** need the category labels of item 3, so it is unblocked today.
  Uniquely among the open items, it can be built without waiting on anything.
- The scaffolding it needs already exists and is measured: the corpus is built
  from real reproducers, and `run_e2e.py` plus the serving script take under two
  minutes on one GPU ([§8](#8-is-the-sanitizer-selection-use-case-ready)).

The honest summary is that fixes 1 and 2 bought range — 1.0000 → 0.9111 and
0.7174, with a real ladder underneath — and bought no *truth*. That was the
cheapest way to find out that form is not enough, which is what it was for. But
form is now demonstrably exhausted, and further work on form-only terms should
be expected to relocate perversities rather than remove them.

**Sharpened 2026-09-08 by the sampled rollout.** The argument above is about
incentives; there is now a measurement. With sampling working, the model scores
**0.8689** and `abstain_and_pick_first` — two lines, reads nothing — scores
**0.9000**. So the gradient a GRPO run would follow today is not merely
uninformative, it points at *abstain and name one mitigation*. Every form-based
term in the table is satisfied best by that policy, and no reordering of them
changes it, because the thing that would distinguish "abstained because there is
no right category" from "abstained because it pays 0.9" is not a property of the
proposal's shape.

That is the whole case in one comparison. A gradient that exists and points the
wrong way is worse than no gradient, and it is only fixable by scoring something
the policy cannot fake by looking cheap — which means running the mitigation.

### 5.6 Still open, unchanged

3. **The category axis still has no labels**, so tier 3 cannot be made
   *correct*-sensitive rather than *membership*-sensitive. This is the same
   blocker, unchanged, and it now has a measured consequence rather than a
   predicted one: 8 of 9 scenarios have no right answer available in the closed
   set. Widening the set to cover kernel-level races is the smallest fix. Fix 1
   is built to become correctness-sensitive the moment this lands.
4. **Default `--grammar-backend xgrammar` wherever a recipe serves a model an
   agent will call**, or make `LiteLLMProposer` degrade when `response_format`
   is refused. Right now the two halves of the integration disagree and only
   the untested half is configured. Not filed yet.
5. ~~**Set a non-zero rollout temperature.**~~ **Done**
   ([§5.4](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default)) —
   and the temperature was not the fix on its own. The engine's sampling backend
   defaults to `greedy` on non-NVIDIA hardware and ignores sampling parameters
   silently, so `--sampling-backend triton` was the load-bearing change.
   Criterion 3 now passes on the wide candidate set.
6. **Two engine defects found on the way, neither filed.** Seeds are accepted
   and ignored even with a sampling backend, so a rollout is not reproducible
   sample-by-sample; and `n > 1` in one request returns identical choices, so
   a group costs N requests instead of one batched call. Details and the
   measurements in [§5.4](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default).

## 6. Triage: verdict versus attribution

The split the brief asked for, across four payload conditions. `blind` withholds
each check's own verdict; `hide-status` withholds `execution_status`.

| condition | verdict accuracy | attribution F1 | reward |
|---|---:|---:|---:|
| full payload (faithful) | 100% (9/9) | 1.0000 | 1.0000 |
| `blind` only | 100% (9/9) | 1.0000 | 1.0000 |
| `hide-status` only | 100% (9/9) | 1.0000 | 1.0000 |
| `blind` + `hide-status` | 77.8% (7/9) | 1.0000 | 0.8667 |

**Attribution F1 is 1.0000 in every condition, including every ablation.** So
the specific failure the attribution term exists to catch — guessing verdicts
from surface cues while inventing justifications — was not observed, and could
not have been: the finding codes are handed to the model in the payload, so
citing them is copying rather than justifying. The term is well-designed and
this corpus cannot exercise it. Six of the nine scenarios have zero findings, so
their correct citation is the empty list, which the scorer counts as perfect
attribution by convention.

The verdict half is **doubly redundant**, which the ablation isolates cleanly:
the overall verdict is recoverable from the per-check verdicts *or* from
`execution_status`, independently. Only when both are withheld does the model
miss, and then it misses on exactly `consan-gemm` and `consan-tiny` — the two
zero-finding `combined_hook_exit_86` cases — where it answered `pass`. That is
the reasonable read of "no findings"; the `error` verdict lives entirely in
`execution_status`. So the fail-versus-error split the reward weights most
heavily is decided by a field the findings say nothing about, and attribution
carries no information about it at all.

So the honest summary of the triage half: not a diagnosis task as currently
posed, an extraction task with a redundant answer key. It is *not* saturated in
the way the proposal ladder is — the 77.8% condition shows it can be got wrong
— but the failure requires deleting evidence the real report contains.

## 7. What nine scenarios can and cannot support

They can support the claims above about **saturation**, because saturation is a
ceiling effect and a ceiling is visible in any sample size. 45 of 45 at tier 5,
with two constants tying the model, is not a statement about a distribution — it
is a demonstration that the reward's range collapsed on every point tested. The
same goes for the blocker in [§1](#1-the-blocker-found-on-the-way-in-a-default-server-cannot-serve-aorta-agent-at-all):
one 500 from the real proposer settles it.

They cannot support anything with an error bar. The 88.9% abstention rate is 40
of 45 draws over 9 prompts, so the effective n is 9, not 45 — and one of those
nine (`consan-racy`) accounts for all five non-abstentions. Verdict accuracy of
77.8% is 7 of 9; its confidence interval spans roughly 45–95% and the two
failures are the same tool defect twice, not two independent observations. The
verdict mix (4 `pass`, 2 `warn`, 2 `error`, 1 `fail`) is not a real distribution
of anything, and `warn` has no probe equivalent at all.

Nothing here is a benchmark and none of these numbers should be quoted as a
model evaluation. What this run establishes is a property of the *reward*, which
is what it was for.

> **Still true after the sampled re-run, with one clause updated.** The
> "effective n is 9, not 45" above was doubly true of the original run: 9
> prompts, and the 45 draws collapsed to 9 distinct completions. Sampling fixed
> the second half only — the effective n is now genuinely 45 draws, and
> `effective_n` in each results file records it — but **it is still 9 prompts**,
> and the prompts are what the confidence interval depends on. Five draws from
> one scenario are five draws from one scenario; they narrow the estimate of
> that scenario's mean and tell you nothing new about the corpus.
>
> So every caveat in this section applies unchanged to the new numbers, and two
> of them get sharper rather than weaker:
>
> - The model-versus-constant gap that criterion 1 turns on is **0.0311**
>   (0.8689 against 0.9000). That is a difference over 9 prompts with no error
>   bar, and its sign is the whole conclusion. It is stable across both
>   temperatures and both candidate-set conditions, which is the only reason it
>   is worth stating — not because 45 draws made it significant.
> - The abstention rate barely moved under sampling (37–39 of 45 against 40),
>   and `consan-racy` still accounts for most non-abstentions. One scenario
>   carrying a headline is exactly what nine prompts cannot support.
>
> The within-group spread results are the exception, for the same reason
> saturation was: "9 of 9 groups have non-zero spread" is a statement about
> whether a gradient exists at all, not an estimate of its size.

## 8. Is the sanitizer-selection use case ready?

**No — but it is now blocked on something cheap, and it was not before.**

The scaffolding works. Serving is one script, the consumer needs no code change,
the graders run on real model output with no conversion pass, and the whole
54-example loop takes under two minutes on one GPU. That was the part in doubt
and it is no longer in doubt.

What is not ready is the reward, and calling it ready on the strength of a 1.0
would have been exactly backwards. As it stands the proposal ladder would train
nothing: every group's advantage is zero, and the policy it cannot distinguish
from Qwen3-8B is a two-line constant that reads no input. Fixes 1 and 2 in
[§5](#5-what-to-fix) are small, local to `proposal_reward.py`, and testable
against the committed corpus without a node. Fix 3 is the category-labelling
blocker, unchanged and still first among the open questions.

**Updated 2026-09-08, after fixes 1 and 2 and a sampled rollout.** Still no —
but only one blocker is left, and it is no longer a mystery.

Two of the three things standing in the way have been cleared. The reward has
range and a real ladder. The rollout samples, so **the GRPO advantage is
non-zero on all nine groups** of the wide condition and the effective n is 45
rather than 9 — the mechanical precondition for training at all, which was
absent before and absent for a reason nobody had identified
([§5.4](#54-the-rollout-was-not-sampling-and-the-cause-was-a-server-default)).

What is left is one thing, and sampling made the case for it sharper rather than
softer:

**The mitigation half needs a resolve signal.** Form-based terms are
demonstrably exhausted — each fixes a perversity by creating another, and the
re-score measured that trade rather than predicting it
([§5.5](#55-is-the-contracts-fix-half-worth-a-gpu-now)). The evidence is now
much more direct than an argument about incentives: with real sampling and a
real gradient, **the model scores 0.8689 and a two-line constant that declines
to classify scores 0.9000.** A policy trained on this reward today would learn
to abstain and name one mitigation, because that is what the reward pays best.
The gradient exists and points the wrong way, which is worse than no gradient
and much easier to act on.

So the recommendation is unchanged in direction and sharper again in content:
this row is still the right starting slice, the scaffolding is now genuinely
complete end to end, and the next node-hour should go to the probe cell that
scores whether a proposed mitigation resolves the reproducer. Nothing else on
the list will change the sign of that gradient.

The recommendation from
[4.0](tokenspeed-rl-post-training.md#40-the-six-prioritised-use-cases-and-how-ready-each-one-is)
— that this row is the right starting slice because it is the only one scaffolded
end to end — survives. It was the cheapest place to discover that the reward is
the hard part, and that is worth more than the 1.0 it produced.
