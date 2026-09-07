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
20-name registry the lists ran to 3, 4, 7, 9, 11 and in one case **17
mitigations at once** — all registered, all offered, so all scored 1.0. The
ladder has no precision term: a proposal naming 17 candidates is worth exactly
what the single right name is worth, and it would cost the agent loop 17
iterations to work through. Most-proposed names were
`debug_hip_dynamic_queues_1` (39/45) and `debug_hip_dynamic_queues_2` (35/45),
chosen at much the same rate across LDS races, GEMM wait hazards and
zero-finding tool errors alike — which is not a targeted choice.

**`consumer_outcome`: all 45 `accepted`, 0 `silent_stop`, 0 `policy_stop`.** The
asymmetry the brief asked about — a proposal scored 0.8 that the real loop would
silently stop on — did not arise, because nothing scored below 1.0. The
committed *synthetic* corpus is the useful contrast: its 45 hand-built
proposals average 0.6 and split 18 `accepted` / 18 `silent_stop` / 9
`policy_stop`. The failure modes the ladder was designed around are real and
the ladder catches them; this model just does not produce them.

## 5. What to fix

In priority order. The first two are reward-design changes, cheap and testable
without a GPU.

1. **Stop paying full marks for `unknown`.** Either exclude it from tier 3's
   accepted set, or gate it on the evidence genuinely not supporting a
   category. At 88.9% abstention this single change is the difference between a
   saturated reward and a graded one.
2. **Add a precision term to the mitigation tiers.** Score
   `1/len(next_mitigations)` or similar, so hedging across the candidate set
   costs something. A 17-name proposal must not tie a 1-name one.
3. **The category axis still has no labels**, so tier 3 cannot be made
   *correct*-sensitive rather than *membership*-sensitive. This is the same
   blocker, unchanged, and it now has a measured consequence rather than a
   predicted one: 8 of 9 scenarios have no right answer available in the closed
   set. Widening the set to cover kernel-level races is the smallest fix.
4. **Default `--grammar-backend xgrammar` wherever a recipe serves a model an
   agent will call**, or make `LiteLLMProposer` degrade when `response_format`
   is refused. Right now the two halves of the integration disagree and only
   the untested half is configured.

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

The recommendation from
[4.0](tokenspeed-rl-post-training.md#40-the-six-prioritised-use-cases-and-how-ready-each-one-is)
— that this row is the right starting slice because it is the only one scaffolded
end to end — survives. It was the cheapest place to discover that the reward is
the hard part, and that is worth more than the 1.0 it produced.
