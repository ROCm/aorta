# RL post-training seams

Demonstrations of the interfaces an RL post-training effort would need from
aorta. **Nothing here trains anything**, nothing here is wired into CI, and
nothing here is a workload. Each file exists to show that one seam is real and
to make its cost and its limits concrete.

Background and the plan these support: [`docs/tokenspeed-rl-post-training.md`](../../docs/tokenspeed-rl-post-training.md).

| | What it demonstrates | Needs |
|---|---|---|
| [`proposal_reward.py`](proposal_reward.py) | A reward for the shape of an `aorta agent` proposal, scored through the agent's own contract code | nothing — no GPU, no container |
| [`triage_reward.py`](triage_reward.py) | A reward for triage classification, labelled by aorta's own verdict resolver | nothing — no GPU, no container |
| [`recipe_reward.py`](recipe_reward.py) | A graded reward for recipe synthesis, computed by calling aorta's own validators, with a novelty gate that refuses to pay for corpus copies | nothing — no GPU, no container |
| [`probe_weight_transfer.py`](probe_weight_transfer.py) | Whether an RL iteration costs a weight update or a cold restart: drives TokenSpeed's weight-sync control plane and times it | a running `tokenspeed serve` |
| [`nccl_weight_peer.py`](nccl_weight_peer.py) | The trainer half of the `nccl` weight-transfer protocol: joins the engine's group and broadcasts tensors | two GPUs, the TokenSpeed image |
| [`nccl_roundtrip_check.py`](nccl_roundtrip_check.py) | Whether a weight update *actually changes the weights*: perturb, then restore, comparing greedy completions | the above plus a running engine |

## `probe_weight_transfer.py`

The Phase 2 probe route. Point it at a running `tokenspeed serve` and it records
what every weight-transfer endpoint accepts and returns, and how long the
operations in an RL loop's inner cycle take.

```bash
python examples/rl/probe_weight_transfer.py \
  --control http://127.0.0.1:8001 \
  --gateway http://127.0.0.1:8000 \
  --model Qwen/Qwen3-0.6B \
  --out weight_transfer_probe.json
```

Stdlib only, so it runs from a compute node's system Python without a venv.

What one run established on gfx950 (one GPU, Qwen3-0.6B, the image the recipes
pin) is written up in
[`docs/tokenspeed-rl-post-training.md`](../../docs/tokenspeed-rl-post-training.md)
under Phase 2. The short version: pause/resume costs 1–5 ms against a 322 s cold
start, the server serves identical completions throughout without restarting,
the lifecycle guards refuse correctly but report 500 where their own docstring
promises 409 — and `backend: ipc` parses its metadata and then raises
`NotImplementedError`, so the colocated deployment is not available and an RL
run has to disaggregate over `nccl`.

It does **not** exercise the NCCL tensor broadcast itself, which needs a trainer
peer joining the process group. The metadata contract for it is exercised; the
transport is not.

## `recipe_reward.py`

The seam: **a reward function that lives in a trainer repo but imports aorta**,
so the code grading the model is the code the CPU gate runs.

The alternative — a reward that reimplements "is this recipe valid" against a
copy of the schema — drifts. When it drifts, the model is optimised against a
validator that no longer describes the product, which is a reward that has
quietly stopped measuring the thing it is named after. Importing the real
validator makes that failure impossible: if aorta's rules change, the reward
changes in the same commit.

Candidate recipe text in, tier out. Tiers are cumulative and each is a real
call into aorta:

| Tier | Check | Called |
|---|---|---|
| 1 | Parses as YAML | `yaml.safe_load` |
| 2 | Schema and cells accepted | `aorta.triage.recipe.load_recipe` |
| 3 | Mitigations and environments resolve | `aorta.registry.get_mitigation` / `get_environment` |
| 4 | Every cell's merged config validates | the named workload's own validation |
| 5 | No cell logged an unknown-key warning | captured from `logging` |

Run the built-in demonstration, which grades six candidates spanning every
tier:

```bash
python examples/rl/recipe_reward.py
```

Grade real files, or check a candidate against the committed corpus:

```bash
python examples/rl/recipe_reward.py recipes/tokenspeed/tokenspeed-serve-load.yaml
python examples/rl/recipe_reward.py --check-memorisation --json candidate.yaml
```

### Why tier 5 is separate from tier 4

`tokenspeed_serve` *warns* about an unknown `workload_config` key and carries on
with the default. So a two-cell concurrency study that writes `concurrency:`
where the schema says `max_concurrency:` is a completely valid recipe whose
concurrency axis does not vary — both cells run the default and the study
measures nothing. Tiers 1 through 4 all pass it.

That is the failure mode this repository's docs spend the most words on, and it
costs nothing to catch, so it gets its own tier. The demo includes this case
precisely because it is the one a naive reward would pay full marks for.

Tier 6 from the plan — "does the recipe express the *asked-for* shape" — is
deliberately **not** implemented. It needs a rubric or a judge model, and
inventing an automatic proxy for it is where reward hacking starts.

### The novelty gate

Every tier above is a property of the artifact, not of the model's work, so a
policy that emits `recipes/tokenspeed/tokenspeed-serve-load.yaml`
character-for-character earns tier 5 for retrieval. That is not a caveat to
document — it is a reward the policy will find — so it is scored:

```
reward = (tier / 5) * novelty_multiplier
```

The multiplier is 1.0 below 0.80 similarity to the nearest committed recipe,
tapers linearly to 0.0 at 0.95, and is 0.0 at or above it. Two thresholds
because each alone is gameable: a pure cliff lets a policy park just underneath
it, and a pure taper never actually refuses to pay for a verbatim copy.

Similarity is measured on a **canonical** form — YAML re-parsed and re-emitted
with sorted keys, comments gone, the arbitrary `ticket` label dropped — so the
obvious evasions do not work. What the demo shows, on the committed corpus:

| Candidate | Tier | Similarity | Reward |
|---|---|---|---|
| verbatim copy of a committed recipe | 5/5 | 1.000 | **0.00** |
| same recipe, ticket renamed and keys reordered | 5/5 | 1.000 | **0.00** |
| a genuinely novel valid recipe | 5/5 | 0.729 | **1.00** |

The middle row is the one that matters: reindenting, restyling or renaming a
copied recipe does not move its canonical form, so it does not buy any reward.
Only changing what the recipe *does* moves it. The third row is what keeps this
a novelty term rather than a difficulty penalty.

Pass `--no-novelty-gate` to score tiers alone. What the gate still cannot do:
it measures distance from the *committed* corpus, which is a proxy for the
training corpus and not the same set, and it cannot tell a novel recipe from a
novel useless one. A real run still needs held-out prompts and tier-6 grading;
the gate removes the single largest way to score well without working, not all
of them.

### One gap this exposes

The deepest check a reward can reach today is `Workload._validated_config()` — a
private name. `setup()` is the public entry point, but it goes on to require
Docker and a readable `/dev/kfd`, so calling it would make the reward depend on
the grader holding a GPU. A public `Workload.validate_config()` would let a
reward function commit to a supported surface instead of to an underscore, and
is the one upstream change a productionised version of this file would want.

## `triage_reward.py`

The second reward: hand the model an archived probe run and ask what happened —
did the bug reproduce (`fail`), did the trial never validly run (`error`), or
was it clean (`pass`) — and which detectors justify that. The label comes from
`aorta.probe.classifier.verdict`'s own resolver, so the same seam applies.

```bash
python examples/rl/triage_reward.py                    # synthetic fixtures
python examples/rl/triage_reward.py --runs path/to/runs  # real archived runs
python examples/rl/triage_reward.py --runs recipes/sanitizers/survey
```

### Two label sources

`result.json` from probe cells, and `sanitizer_report.json` from Waitcheck and
ConSan runs. Both answer the same question, so `--runs` collects both from one
tree.

The sanitizer source works on real data today — the six committed reports under
`recipes/sanitizers/survey/reports/` are the only archived labelled failure
evidence in the repository, and the last command above scores them. The probe
source has no corpus yet; labelling, scoring and the degenerate-policy check are
implemented and tested against synthetic fixtures, and pointing `--runs` at
archived probe runs is the remaining step.

The seam is stronger on the sanitizer side. `SanitizerReport.from_dict`
recomputes `overall_verdict` from the check results and **raises** if the stored
value contradicts it, so a rotted report cannot be trained on — it fails to load
and is named on stderr. On the probe side the same disagreement is only flagged
`stale`.

Two deliberate choices there: the verdict vocabulary stays the sanitizer's own
(`pass`/`warn`/`fail`/`not_checked`/`error`) rather than being mapped onto the
probe's three-way split, because `warn` has no probe equivalent; and cited
evidence is namespaced as `waitcheck:wait_hazard` rather than `wait_hazard`, so
attribution reads like a detector ID and cannot be mistaken for one.

Two scoring terms, because either alone is trivially hackable:

| Term | Weight | Why |
|---|---|---|
| verdict exact match | 0.6 | the `fail` vs `error` split is the operational question: "the bug reproduced" vs "the trial never ran" |
| attribution F1 over cited detector IDs | 0.4 | without it, guessing the verdict and inventing a reason scores full marks |

**Labels are recomputed, never read.** Each fixture carries a `verdict` field
and this module ignores it, routing the recorded detector IDs back through
`partition_detectors` and `verdict_from_detectors`. That keeps the seam intact —
if the precedence rules change, every label changes in the same commit — and it
flags corpus rot: an archived run whose stored verdict disagrees with today's
rules is reported as `stale` rather than silently trained on.

### The degenerate baseline

A real probe corpus is skewed towards `pass`, so a policy that always answers
`pass` scores well on accuracy alone — the classifier analogue of the
memorisation problem. The demo therefore scores the degenerate policies
alongside the real ones, over 18 fixtures:

| Policy | Verdict accuracy | Attribution F1 | Reward |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| right verdict, invented detectors | 1.000 | 0.222 | 0.689 |
| degenerate: always `pass`, no detectors | 0.222 | 0.222 | **0.222** |
| degenerate: always `fail`, correct detectors | 0.667 | 1.000 | **0.800** |

These fixtures are deliberately failure-heavy — the failure shapes are what the
reward has to discriminate — so the floors here are the inverse of a real
corpus's: always-`pass` is cheap and always-`fail` is expensive, and both flip
once an archive arrives. The invariant is that a floor is printed at all. A
policy's reward is only meaningful stated next to it.

Row two is the term earning its place: verdict right, citation invented, and the
0.311 it loses is entirely attribution. Without that term it would be
indistinguishable from the oracle.

### Debugging-shaped fixtures

The fixtures cover two things. The first eleven exercise the verdict precedence
rules — `fail` over `error` over `pass`, the infra-only case, the synthesised
`meta:missing_pass_signal`. The rest are the failure shapes `aorta agent` is
actually pointed at, one per autopsy category the proposal contract enumerates:
a collective timeout with a hang, a page fault reported as a HIP error, a
thermal throttle, an XGMI link fault, a NaN signature with a zero exit code, and
an SDMA timeout alongside an infra timeout.

Every cited detector ID is one a classifier tier can really emit, which a test
pins by collecting the vocabulary from the tier modules' own constants rather
than hard-coding it. A fixture citing an invented ID would train attribution
against a vocabulary that does not exist.

One fixture carries `tier3:vram_growth` as a **warn** alongside a genuine
segfault. It is a red herring on purpose: advisory detectors are evidence about
a run, never justification for its verdict, so a policy that cites every
detector it can see scores lower than one that cites only the failure signals.
`tier3:vram_growth` is the only advisory detector — `tier3:thermal_throttle`
reads like a performance note but is a genuine failure signal.

## `proposal_reward.py`

The outer layer, and the cheapest reward in the plan: does a proposal satisfy
the contract `aorta agent` actually demands? Strict JSON, a `category` from the
closed autopsy set, and mitigation names that resolve in the registry *and* are
still available. Zero GPU, microseconds per sample, no labelling — the contract
is code.

```bash
python examples/rl/proposal_reward.py
python examples/rl/proposal_reward.py --json
```

### Why score it, when the consumer already validates

Because the consumer validates *quietly*, and a reward that delegated to it
would score silent failure as success.

`LiteLLMProposer.propose` drops unrecognised mitigation names before the policy
ever sees them:

```python
filtered = [m for m in step.next_mitigations if m in remaining]
```

So a proposal naming only invented mitigations — `rccl_p2p_disable`, which
sounds exactly like the 22 registered names and is not one of them — arrives at
the loop as a well-formed step with nothing to try, and `run_agent_loop` reads
an empty list as a decision to stop. The outcome is `agent_stop`, carrying the
model's own hypothesis as the operator's recommended action. Nothing raises.

`AgentStep.from_dict` repairs on the same principle: a non-bool `stop` becomes
`False`, a non-list `next_mitigations` becomes `[]`, an unparseable
`confidence` becomes `0.0`, a null `category` becomes `"unknown"`. Correct for a
serving path, useless as a training oracle.

| Tier | Check | Reward |
|---|---|---|
| 1 | Parses as a JSON object | 0.2 |
| 2 | The five demanded keys, with the demanded types | 0.4 |
| 3 | `category` is in the closed set | 0.6 |
| 4 | A non-empty mitigation list, every name in the registry | 0.8 |
| 5 | Every name still available, `confidence` in [0, 1] | 1.0 |

Each tier calls `AgentStep.from_dict`, `AgentPolicy.validate_step`,
`AUTOPSY_CATEGORIES` and `get_mitigation` rather than restating them — the same
seam the other two scorers use, so a new category or mitigation changes the
reward in the same commit.

Every scored proposal also records `consumer_outcome`: what `run_agent_loop`
would actually do with it, one of `accepted`, `silent_stop` or `policy_stop`.
That is what makes the reward readable against its real consequence. Note that
`accepted` and a low reward co-occur often — a mistyped `confidence` is
accepted after being silently zeroed — and that is the whole point.

### The ceiling, which is the reason this is a gate and not the reward

| Policy | Mean reward | Accepted by the loop |
|---|---|---|
| always the same valid proposal | **1.00** | 1.00 |
| always an empty object | 0.20 | 0.00 |
| always prose | 0.00 | 0.00 |

A fixed valid proposal scores full marks while diagnosing nothing, and the loop
accepts it every time. This is asserted as a test, so if it ever changes the
docstring's claim that this measures form only has become false. Pair it with
`triage_reward.py`, which scores whether the read is right.

Row two is the term that earns its place: a policy with the verdict right and
the citation invented keeps 0.689, and the 0.311 it loses is entirely
attribution. Without that term it would be indistinguishable from the oracle.

## `build_corpus.py`

Turns a tree of real `sanitizer_report.json` artifacts into JSONL that both
scorers read directly:

```bash
python examples/rl/build_corpus.py \
  --results <tree of sanitizer runs> \
  --baselines recipes/sanitizers/fixtures/expected/verdict_baselines.json \
  --out examples/rl/corpus --run-meta run_meta.json

python examples/rl/triage_reward.py   --corpus examples/rl/corpus/triage.jsonl
python examples/rl/proposal_reward.py --corpus examples/rl/corpus/proposal.jsonl
```

The committed corpus under `corpus/` is the first one generated from real runs;
`corpus/README.md` records what is in it, its provenance, and the tool defects
the runs surfaced.

### One example is one scenario, not one finding

The two-wave LDS race reproducer emits 64 findings that are 64 lanes of a
single race: identical instruction pair, differing only in lane mask and LDS
byte range. `distinct_evidence` dedupes on the tuple that identifies a race
*site*, and `finding_counts` reports `raw` next to `distinct_sites` so the
difference stays visible rather than being absorbed into a corpus size.

ConSan also writes the same finding objects into both `check.findings` and
`kernel_results[].findings`, which doubles a naive count — 64 findings read as
128. Identical records are collapsed before the site dedup, so `raw` means what
the tool emitted.

### Ground truth is the committed baseline, not the run

A scenario named in `verdict_baselines.json` carries its expected verdict next
to the observed one and an explicit `agrees` flag. A disagreement is emitted,
not dropped: it is evidence of a tool defect, and dropping it would hide the
one thing worth reporting. `manifest.json` lists any disagreements by name.

### `workload_family` on every example

There are three families today (`synthetic_hip_lds`, `synthetic_hip_vecadd`,
`tensile_gemm_object`) and the field looks redundant at that size. It is not:
the detector and finding vocabulary is workload-independent, but which failures
co-occur is not, so a corpus that cannot be split by family cannot be checked
for balance. Recording it during generation is free; backfilling it onto
artifacts that no longer say which workload produced them is not. A test
asserts no example carries `unknown`.

## The NCCL weight-transfer harness

`probe_weight_transfer.py` exercises the control plane; these two exercise the
*data* plane, which is the part that decides whether an RL loop can replace a
322 s cold start with a weight update.

```bash
# trainer side: joins the engine's group as rank 0 and broadcasts
python examples/rl/nccl_weight_peer.py \
  --master-port 29511 --world-size 2 --rank 0 \
  --model-path /path/to/hf/snapshot \
  --tensors model.embed_tokens.weight \
  --rounds perturb,restore --plan-out /shared/plan.json

# engine side: proves the weights actually moved
python examples/rl/nccl_roundtrip_check.py \
  --engine-url http://127.0.0.1:8000 --control-url http://127.0.0.1:8001 \
  --model Qwen/Qwen3-0.6B --plan /shared/plan.json \
  --master-port 29511 --world-size 2 --out roundtrip.json
```

`nccl_weight_peer.py` deliberately does **not** call
`torch.distributed.init_process_group`. The engine joins a standalone group over
a `PrefixStore(group_name, ...)`; torch's own initialiser prefixes the store
with `"default_pg"`, so a trainer using it rendezvouses against different keys
and both sides hang. The peer mirrors the engine's construction exactly.

### Why the check is a round trip

The verdict is not "did `/update_weights` return 200". It is three greedy
generations of one fixed prompt: baseline, after pushing perturbed weights, and
after pushing the originals back. The first difference must appear and the
second must not. Both halves are load-bearing — perturb-only would pass a path
that corrupts memory as readily as one that transfers correctly.

That is not hypothetical. On gfx950 this check found that TokenSpeed's `nccl`
receive returns `200 {"message": "Weights updated"}` while **transferring
nothing**, loading uninitialised device memory into the model instead; a
perturb-only test would have reported it as working. The findings are written up
in [`docs/tokenspeed-rl-post-training.md`](../../docs/tokenspeed-rl-post-training.md)
under Phase 2.
