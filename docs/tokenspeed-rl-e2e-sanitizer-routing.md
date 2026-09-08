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

> **Status, 2026-09-08.** The two reward-design fixes have landed and the
> recorded proposals have been re-scored against them: the ladder now spans
> 0.5400 to 1.0000 where it was flat at 1.0000, and the model no longer ties a
> constant on the primary condition. It does not clear the whole acceptance
> test — two of the four criteria fail on the widened condition, and one of them
> fails for a reason no reward can address. §1 through §4 below are left as the
> record of the original run; [§5](#5-what-to-fix) is the record of the fixes,
> the numbers, and the failures. The caveats in
> [§7](#7-what-nine-scenarios-can-and-cannot-support) apply unchanged to the new
> numbers — the effective n is still 9, not 45, and now demonstrably so.

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
> [sampling defect](#54-a-second-cause-of-zero-advantage-that-no-reward-can-fix)).
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
k-name proposal. The cost model is the loop's, not a preference: `loop.py` runs
a probe cell per name and charges one unit of iteration budget for the whole
proposal, so k names is k GPU cells and the budget does not restrain it.

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

### 5.3 What the re-score showed

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

### 5.4 A second cause of zero advantage that no reward can fix

**All five samples in every scenario group are byte-identical.** The 45 recorded
proposals are 9 distinct completions, each repeated five times, in both
conditions.

`LiteLLMProposer.propose` sends no `temperature` and `run_e2e.py` injected none
(`temperature_injected: null`, and the recorded request carries only `model`,
`response_format` and the messages), so the engine decoded at its default and
sampling was effectively greedy.

This matters more than it first looks. A reward is a function of the completion,
and the loop state is constant inside a group, so **identical completions earn
identical rewards under any reward function whatsoever.** Within-group spread is
therefore exactly zero for reasons that have nothing to do with saturation, and
criterion 3 was unsatisfiable on this data before a single line was changed.
[§3](#3-does-the-scoring-discriminate-no-it-is-saturated-with-no-headroom)
attributed 9-of-9 degenerate groups to the reward's ceiling; that was one of two
independent causes, and this is the other.

The distinction is pinned by a pair of tests rather than argued: five copies of
one completion give zero spread, and four differing completions on the same
scenario and loop state give non-zero spread under the same grader. So the
reward is not what blocks criterion 3.

`rescore_e2e.py` reports `distinct_completions` per group and fails under
`--check-determinism` when any group collapses to one. As a substitute
measurement it reports `spread_across_on_contract_policies` — the range the
reward achieves on one scenario across the policies that clear the format gate,
**0.1000** on faithful and **0.4600** on hard, against 0.0000 before. That
demonstrates the reward regained discriminative power. It is *not* criterion 3
and must not be reported as if it were: it needs several policies, whereas GRPO
needs one policy sampled several times.

**Consequence for the trainer:** a rollout must set a non-zero temperature, or
GRPO gets a zero advantage regardless of the reward. This is a one-line change
to the rollout path, and it needs a GPU to re-measure, so it is not done here.

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
5. **Set a non-zero rollout temperature** ([§5.4](#54-a-second-cause-of-zero-advantage-that-no-reward-can-fix)).
   Without it there is no within-group spread and so no GRPO gradient, whatever
   the reward says.

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

**Updated 2026-09-08, after fixes 1 and 2.** Still no — but the blocker moved,
and it moved onto something that costs a GPU rather than something free. The
reward now has range and a real ladder, and the two constants no longer tie the
model on the primary condition. Two things stand between here and a training
run, and neither is a reward-design change:

1. **The rollout must sample.** Every group's five completions came back
   byte-identical, so the GRPO advantage is zero for reasons no reward can
   touch ([§5.4](#54-a-second-cause-of-zero-advantage-that-no-reward-can-fix)).
   One line, one GPU, one re-measure.
2. **The mitigation half needs a resolve signal.** Form-based terms are now
   demonstrably exhausted — each one fixes a perversity by creating another, and
   the re-score measured that trade
   ([§5.5](#55-is-the-contracts-fix-half-worth-a-gpu-now)). The contract's fix
   half is the only term that is not gameable by inspection, and unlike fix 3 it
   is unblocked today.

So the recommendation is unchanged in direction and sharper in content: this row
is still the right starting slice, and the next node-hour spent on it should go
to the probe cell, not to another reward term.

The recommendation from
[4.0](tokenspeed-rl-post-training.md#40-the-six-prioritised-use-cases-and-how-ready-each-one-is)
— that this row is the right starting slice because it is the only one scaffolded
end to end — survives. It was the cheapest place to discover that the reward is
the hard part, and that is worth more than the 1.0 it produced.
