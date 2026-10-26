# RL post-training seams

Demonstrations of the interfaces an RL post-training effort would need from
aorta. Nothing here is wired into CI as a job and nothing here is a workload.
Each file exists to show that one seam is real and to make its cost and its
limits concrete. The one trainer, `train_grpo_step.py`, is a deliberately small
GRPO loop that exists to exercise the multi-step episode path end to end; it is
not the framework decision.

Background and the plan these support: [`docs/tokenspeed-rl-post-training.md`](../../docs/tokenspeed-rl-post-training.md).
The multi-step episode path, its predictions and one paired evaluation:
[`docs/rl-multi-step-episodes.md`](../../docs/rl-multi-step-episodes.md).

| | What it demonstrates | Needs |
|---|---|---|
| [`proposal_reward.py`](proposal_reward.py) | A reward for the shape of an `aorta agent` proposal, scored through the agent's own contract code | nothing — no GPU, no container |
| [`triage_reward.py`](triage_reward.py) | A reward for triage classification, labelled by aorta's own verdict resolver | nothing — no GPU, no container |
| [`recipe_reward.py`](recipe_reward.py) | A graded reward for recipe synthesis, computed by calling aorta's own validators, with a novelty gate that refuses to pay for corpus copies | nothing — no GPU, no container |
| [`probe_weight_transfer.py`](probe_weight_transfer.py) | Whether an RL iteration costs a weight update or a cold restart: drives TokenSpeed's weight-sync control plane and times it | a running `tokenspeed serve` |
| [`nccl_weight_peer.py`](nccl_weight_peer.py) | The trainer half of the `nccl` weight-transfer protocol: joins the engine's group and broadcasts tensors | two GPUs, the TokenSpeed image |
| [`nccl_roundtrip_check.py`](nccl_roundtrip_check.py) | Whether a weight update *actually changes the weights*: perturb, then restore, comparing greedy completions | the above plus a running engine |
| [`event_reward.py`](event_reward.py) | A discrete per-step event reward for a probe episode, every event decidable from the reply, the log and an archived matrix | nothing — no GPU, no container |
| [`episode_env.py`](episode_env.py) | Multi-step probe episodes run offline against archived matrices, through the agent loop's own decision functions | the archived matrices (`--corpus-root`) |
| [`rescore_episodes.py`](rescore_episodes.py) | Constant policies against the corpus, a run's wire reduced, and a replay of recorded episodes | the archived matrices |
| [`train_grpo_step.py`](train_grpo_step.py) | GRPO over episodes with in-process `transformers`, every weight update checked before a checkpoint is written; `--env rocjitsu` trains on `rocjitsu_env` episodes | one GPU (two with `--kl-beta`), `transformers` |
| [`eval_episodes.py`](eval_episodes.py) | A fixed checkpoint against a control, paired per scenario, no optimiser step | one GPU, `transformers` |
| [`verify_checkpoint_delta.py`](verify_checkpoint_delta.py) | A checkpoint pair on disk: every trained tensor moved, the control did not, nothing past Adam's ceiling | `numpy` only |
| [`rocjitsu_env.py`](rocjitsu_env.py) | Multi-turn GPU-debugging episodes: find the one synchronization fault injected into a workload's machine code, with every tool output replayed from a recorded archive | a recorded corpus (`--corpus-root` or `$AORTA_RJ_CORPUS_ROOT`) |
| [`rocjitsu_baselines.py`](rocjitsu_baselines.py) | Scripted and constant policies for those episodes, scored by the same parser and reward as a model | a recorded corpus |
| [`prepare_rocjitsu_corpus.py`](prepare_rocjitsu_corpus.py) | A v1-layout corpus directory and its by-kernel split, from a recorded campaign | a recorded campaign |
| [`blas_backend.py`](blas_backend.py) | Opt-in choice of PyTorch's BLAS library before the first GEMM, recorded in the run's config | `torch` |
| [`eval_rocjitsu.py`](eval_rocjitsu.py) | A fixed checkpoint over a rocjitsu split, paired episode for episode with a control and with the CPU baselines | one GPU, `transformers` |
| [`merge_columns.py`](merge_columns.py) | The shards of one evaluation column put back into the column an unsharded run writes | the shards and the corpus |
| [`rocjitsu_demos.py`](rocjitsu_demos.py) | Solved episodes harvested as demonstrations, each replayed and re-scored first | a recorded corpus |
| [`mix_demos.py`](mix_demos.py) | Whole-episode replay mixing of demonstration files for a continual cycle | nothing |
| [`sft_rocjitsu.py`](sft_rocjitsu.py) | Supervised fine-tuning on demonstrations with the GRPO trainer's model, numerics and loss | one GPU, `transformers` |

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

Grade real files. The novelty gate — which checks a candidate against the
committed recipes so a memorised copy cannot score as a new one — is on by
default; `--no-novelty-gate` turns it off. With the gate off the grade makes no
novelty claim: `novelty_multiplier` and `memorised` are `null` in `--json`
output, and the reward is the tier reward.

```bash
python examples/rl/recipe_reward.py recipes/tokenspeed/tokenspeed-serve-load.yaml
python examples/rl/recipe_reward.py --json candidate.yaml
```

**It exits non-zero on a recipe that earned nothing**, so it can gate rather
than only report: any input that fell short of tier 5, any path that could not
be read, and any candidate the novelty gate marked `memorised`. That last one
is the case the exit code is easiest to get wrong — a verbatim copy of a
committed recipe passes every tier, because it *is* a valid recipe, so it
reaches tier 5 with a reward of 0.00. A near-copy in the soft zone still exits
0: there the gate scales the reward rather than zeroing it, and turning that
taper into a second threshold would defeat the point of having one.

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

The sanitizer source works on real data today — the two committed reports under
`recipes/sanitizers/survey/reports/` (`gemm_f32_waitcheck` is `warn`,
`gemm_f32_consan` is `error`) are the only archived labelled failure evidence
in the repository, and the last command above scores them. That set has no
`pass` row and only one workload family (`tensile_gemm_object`); the synthetic
`tiny_vecadd` and `lds_reduce` controls are local recipes, not committed
reports. The probe source has no corpus yet; labelling, scoring and the
degenerate-policy check are implemented and tested against synthetic fixtures,
and pointing `--runs` at archived probe runs is the remaining step.

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

Row two is the one to read next. `{}` parses as a JSON object, so it clears
tier 1 and collects that 0.2 and nothing further: tier 2 wants the five
demanded keys and it has none. The loop accepts it 0.00 of the time. That gap
between a non-zero reward and zero acceptance is the honest shape of a
form-only scorer — it can say the output was well-formed, and it cannot say
the output was useful.

For whether the *read* is right rather than the shape, see `triage_reward.py`.
Attribution is scored there, against real detector IDs, and it is not scored
here at all.

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

`--run-meta` is optional: a JSON object whose keys — image digest, commit,
runner — are added to every example's `provenance`. Omit it rather than
pointing it at a file you have not written. A path that is not a file, JSON
that is not an object, and an object setting `report` (each example's own
source report) are refused before the build; the first of those used to be
read as "no provenance" and exit 0.

**`corpus/` holds no generated rows** — `*.jsonl` is gitignored repo-wide and
these are run outputs, not fixtures, so the two commands above need a corpus
built first. Without a GPU, point `--results` at
`recipes/sanitizers/survey/reports`, the two survey reports this repo does
commit (`warn` and `error` on the hipBLASLt GEMM); that exercises every scorer
path on a smaller, less balanced set, with no clean (`pass`) row.

The two files the directory *does* commit are input and provenance, not rows:
`scenario_labels.json`, the hand-written ground-truth autopsy category per
scenario — kept beside the corpus rather than as a column in a row, because
`build_corpus.py` rewrites the rows and a label written into one is lost on
the next rebuild; `tests/agent/test_autopsy_categories.py` holds it to the
`aorta.agent.llm` vocabulary — and `corpus/README.md`, the provenance record
for the nine-scenario sweep the measurements in the docs were taken against:
what was in it, and the tool defects those runs surfaced.

### One example is one scenario, not one finding

The two-wave LDS race reproducer emits 64 findings that are 64 lanes of a
single race: identical instruction pair, differing only in lane mask and LDS
byte range. `distinct_evidence` dedupes on the tuple that identifies a race
*site*, and `finding_counts` reports `raw` next to `distinct_sites` so the
difference stays visible rather than being absorbed into a corpus size.

That tuple has to cover both sanitizers, and originally it did not: it read the
ConSan instruction-pair fields, which a Waitcheck finding does not have, so
Waitcheck checks collapsed to one site each and discarded the rest of their
evidence. It now also reads `metadata.context_1`/`context_2`, the producer and
consumer offsets Waitcheck locates a hazard by. The ConSan behaviour above is
unchanged — the two families' keys are disjoint — which is what makes
collapsing lanes and keeping distinct hazards the same rule rather than two.

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
artifacts that no longer say which workload produced them is not. The committed
survey reports only reach `tensile_gemm_object`; the other two families stay in
the map so a local control run (`daily-*-tiny`, `daily-*-lds-dispatch`) still
classifies. A test asserts no example carries `unknown`.

## The NCCL weight-transfer harness

`probe_weight_transfer.py` exercises the control plane; these two exercise the
*data* plane, which is the part that decides whether an RL loop can replace a
322 s cold start with a weight update.

```bash
RUN_ID="$(date +%s)-$$"   # any token the two sides agree on

# trainer side: joins the engine's group as rank 0 and broadcasts
python examples/rl/nccl_weight_peer.py \
  --master-port 29511 --world-size 2 --rank 0 \
  --model-path /path/to/hf/snapshot \
  --tensors model.embed_tokens.weight \
  --rounds perturb,restore --plan-out /shared/plan.json \
  --run-id "${RUN_ID}"

# engine side: proves the weights actually moved
python examples/rl/nccl_roundtrip_check.py \
  --engine-url http://127.0.0.1:8000 --control-url http://127.0.0.1:8001 \
  --model Qwen/Qwen3-0.6B --plan /shared/plan.json \
  --plan-run-id "${RUN_ID}" \
  --master-port 29511 --world-size 2 --out roundtrip.json
```

`--run-id` / `--plan-run-id` are required on purpose. `--plan` is a fixed
shared path, so a plan from a previous run is the ordinary state of that
directory, and the driver used to accept it: it would read stale tensor names
and shapes while this run's peer was writing its own, then mismatch the HTTP
update against the collective the peer actually broadcasts — which hangs rather
than failing. The driver now waits for a plan whose `run_id` matches, and
reports `PEER_PLAN_STALE` rather than `PEER_PLAN_NEVER_APPEARED` when it finds
one from another run, since those send you to different places.

`nccl_weight_peer.py` deliberately does **not** call
`torch.distributed.init_process_group`. The engine joins a standalone group over
a `PrefixStore(group_name, ...)`; torch's own initialiser prefixes the store
with `"default_pg"`, so a trainer using it rendezvouses against different keys
and both sides hang. The peer mirrors the engine's construction exactly.

### Why the check is a round trip

The verdict is not "did `/update_weights` return 200". It is greedy generations
of one fixed prompt in three phases: baseline, after pushing perturbed weights,
and after pushing the originals back. The first difference must appear and the
second must not. Both halves are load-bearing — perturb-only would pass a path
that corrupts memory as readily as one that transfers correctly.

### Why each phase is drawn more than once

A temperature-0 completion is **not** a deterministic function of the weights on
this stack. The sampler is out of the picture, but the argmax still moves with
batch composition and reduction order: §5.4 of
[`docs/tokenspeed-rl-e2e-sanitizer-routing.md`](../../docs/tokenspeed-rl-e2e-sanitizer-routing.md)
measures **2 of 8 distinct completions across separate temperature-0 requests**.

Compared as single strings, that noise supplies `B != A` on its own. Simulated
at the rate that section measures, a one-draw-per-phase verdict called a *no-op*
transfer `PROVEN` **11% of the time** — the strongest verdict this tool gives,
from decoder jitter. So `--replicates` (default 3) draws each phase several
times: the baseline must agree with itself, every perturb draw must differ from
it, and every restore draw must match. At three draws the same simulation gives
0.089%, and on an engine that is genuinely deterministic the replicates cost six
extra 32-token generations and change no verdict.

If the baseline disagrees with itself the run reports
`BASELINE_NONDETERMINISTIC` rather than comparing anyway. That is the honest
answer: if the observable is not a function of the weights, no later difference
in it is evidence about weights.

`--peer-grace` (default 120s) adds evidence that does not come from the model at
all. The peer writes `<plan-out>.roundN.done` once round N's broadcasts have
drained, and `dist.broadcast` on the sending rank returns only when the other
rank posts its matching receive — so a missing marker after the engine has
already answered 200 is direct proof the engine posted no collective. It
outranks the completions, and it still works when the decoder is too unstable
for the round trip to say anything.

### `--src` and what a round is for

A round's kind fixes which end of the broadcast the peer is on. `perturb` and
`restore` push tensors at the engine, so the peer must be the root and `--src`
defaults to `--rank` accordingly. `recv` is the diagnostic that swaps the roles
— it poisons a buffer and checks whether the poison survives — so it needs an
*engine* rank as the root, and inherits a default that makes the peer broadcast
its own poisoned buffer straight back. The buffer then returns untouched,
because the root's buffer is the source, and the diagnostic reports that nothing
arrived from a collective that worked. Both directions are now rejected before
the rendezvous, so a role mismatch costs nothing instead of blocking the driver
on an `/update_weights` whose broadcast is never posted.

That is not hypothetical. On gfx950 this check found that TokenSpeed's `nccl`
receive returns `200 {"message": "Weights updated"}` while **transferring
nothing**, loading uninitialised device memory into the model instead; a
perturb-only test would have reported it as working. The findings are written up
in [`docs/tokenspeed-rl-post-training.md`](../../docs/tokenspeed-rl-post-training.md)
under Phase 2.

## `rocjitsu_env.py`

An episode is a GPU workload in which at most one kernel carries one
synchronization fault, injected as a same-size edit to its machine code (or none:
a clean control). The policy calls tools — `waitcheck`, the rocjitsu simulator,
a native run, a disassembly — each at a fixed cost, then answers
`{class, kernel, offset}` within 8 turns. Ground truth is exact because the fault
was injected at a known instruction. Every tool output is replayed from an
archive recorded once per scenario, so an episode needs no GPU and runs in
milliseconds. The module's docstring records each design decision; the reward
versions are in [`docs/rl-rocjitsu-reward-v2.md`](../../docs/rl-rocjitsu-reward-v2.md).

The corpus is not part of this repository: its listings are disassembly of
kernels from third-party libraries, and its held-out test splits stay useful
only while they are not published. The environment reads two layouts, detected
per directory:

- **v1** (one workload): `sites.json`, `results.jsonl` (one record per mutant
  plus a `BASELINE:<kernel>` record per kernel), the workload's disassembly,
  `analysis.json`, and the `split.json` that `prepare_rocjitsu_corpus.py` writes.
  Scored under reward v1 only: its system prompt states v1's points, so loading
  it with reward v2 is refused.
- **v2** (many workloads): `scenarios.jsonl` (one record per scenario: workload,
  kernel id, operator, class, accepted offsets, the byte edit, and every tool's
  recorded result), `kernels.json`, `workloads.json`, `listings/<kernel id>.json`
  (the original listing; the edit is applied when it is served) and
  `splits/<name>.txt`.

The tests build small synthetic archives in both layouts and need nothing
outside the repository. Two replay suites compare the environment with the
reference implementations the corpora were recorded with, and run only when
pointed at them: `AORTA_RJ_TEST_PROTOTYPE` (a v1 campaign directory) and
`AORTA_RJ_TEST_CORPUS_V2` (a corpus-v2 build directory with `archive/` and
`scripts/rjenv2.py`).

### Known limitations

- **`run_sim all` folds quiet kernels into one line.** A kernel with no race or
  memory-wait report is listed as "no findings" even when its simulated output
  check is `FAIL` or `CRASH`; the line says "simulated output PASS" only when
  every folded kernel passed, but it does not say which one did not. A
  per-kernel `run_sim` shows the status. It is kept because the released model
  was trained on exactly this observation, and changing the text would change
  what that model sees; it will be fixed in the next environment version.
- **History pruning does not put `disasm` listings first.** When the history
  outgrows its budget, older tool results are replaced by a placeholder, and
  listings were meant to go first because they are the largest and the cheapest
  to fetch again. The test looks for a header starting with `disasm`, but every
  tool header is the JSON of the call (`{"tool": "disasm", ...}`), so it never
  matches and pruning is oldest-first for every tool. It is kept because the
  released model was trained on exactly this history; it will be fixed in the
  next environment version.
- **Near credit counts instructions in the kernel as built.** An answer is near
  when it is within 2 instructions of an accepted offset, counted in the
  original listing. A v2 edit that moves a wait past a longer instruction can
  leave an accepted offset that is an instruction only of the served listing;
  that offset earns exact credit but is skipped for near credit, so an answer 2
  instructions after it can get kernel credit only. The corpus's reference
  environment scores this way, and so does every recorded evaluation; it will be
  fixed with a new reward version.

## `train_grpo_step.py --env rocjitsu`

`--env rocjitsu` trains on `rocjitsu_env` episodes instead of probe episodes,
from a corpus split (`--corpus-root`, `--split`). Each turn of an episode is one
`(prompt, completion)` sample; generation and both loss forwards template it
under the environment's system prompt, and the loss forward keeps only the
completion's logits. What a port to another trainer has to keep for its numbers
to compare with this one's:

- **One advantage per episode**, shared by all of its turns and normalised within
  the group of episodes on one scenario: `A = (r - mean) / max(sd, 1.0)`
  (`--advantage-sd-floor 1.0`). Without the floor, a group whose episodes all
  failed differs only in tool costs, and dividing by that small spread scales
  the noise up to unit size.
- **KL to a frozen bf16 reference**, `--kl-beta 1e-3` with the k3 estimator, on
  its own device. `--reference-from` anchors it to a warm start (`--init-from`)
  rather than to the base model.
- **Adam** at `--lr 1e-6` with moments kept across iterations, no gradient clip,
  fp32 parameters.
- **One numerics for sampling and both loss forwards** (`policy_numerics()`, a
  single bf16 autocast).
- **The off-policy gate.** Each sample's loss forward is compared token by token
  with the log-probabilities generation recorded. A sample whose largest gap is
  over 2 nats is left out of the gradient; the step is refused if more than 2%
  of its samples are, or if a sample that takes a gradient has a token its loss
  forward scores below exp(-30), whose KL term would own the gradient. `--diag`
  records the comparison without stopping and writes no checkpoint.
- **Sampling:** 8 scenarios × 8 episodes per iteration
  (`--scenarios-per-iteration 8 --group 8`) at temperature 1.0, top-p 0.95 and
  top-k 20, 8 turns of at most 192 tokens. `--top-k` is passed to `generate`, so
  a model directory's own generation config cannot change it.

```bash
python examples/rl/train_grpo_step.py --env rocjitsu --corpus-root <corpus> --split train \
    --out <run> --model Qwen/Qwen3-8B --iterations 40 --scenarios-per-iteration 8 --group 8 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --max-episode-steps 8 --max-new-tokens 192 \
    --lr 1e-6 --kl-beta 1e-3 --param-dtype float32 --advantage-sd-floor 1.0 --snapshot-every 10
```

`--snapshot-every N` also writes `snapshot-itNNN` every N verified iterations:
the sampler's weights (linear layers in bf16, everything else as trained) with
the tokenizer and the generation config, so an evaluation loads exactly what
generated the rollouts. `blas_backend.py` is opt-in: with
`AORTA_PREFERRED_BLAS` set (`cublas` also needs `ROCBLAS_USE_HIPBLASLT=0`) it
sets PyTorch's preferred BLAS library before the first GEMM, and the run's
config records the choice; unset, it changes nothing.

## `eval_rocjitsu.py` and `merge_columns.py`

One checkpoint rolled out over a split, with no optimiser step, is a *column*:
`--before` (the base `--model` by default) and `--after` give one each. Every
reply is seeded from `(--seed, scenario, episode, step)` and generated on its
own, so two columns on the same scenarios are paired episode for episode and
their difference is the weights. A column records the system prompt's sha256
and every sampler setting it was drawn with.

- `--compare` writes `comparison.json`: each column's metrics over all, faulty
  and clean scenarios, the paired score difference by scenario with a bootstrap
  95% interval, McNemar counts for solved episodes and for exact locations, and
  the CPU baselines (the scripted policies and the best constant answer) on the
  same scenarios.
- It refuses to pair columns that ran different scenarios or corpus digests,
  that differ in seed, episodes, temperature, top-p, token limit or turn budget,
  or that were drawn under different or unrecorded system prompts, or under two
  recorded `top_k` values. `--reuse` resumes a partial column only under the same
  source, seed, system prompt and `top_k`.
- `--shard K/N` runs scenarios K, K+N, ... of the split, so one column can be
  spread over N devices. Each reply's seed depends only on its scenario, episode
  and step, so `merge_columns.py` can put the shards back into exactly the
  column an unsharded run writes; it refuses shards that disagree on a setting,
  overlap, are unfinished, or do not cover the split.

```bash
for k in 0 1 2 3; do
  python examples/rl/eval_rocjitsu.py --corpus-root <corpus> --split val --out <dir>/shard-$k \
      --after <checkpoint> --column after --shard $k/4 --device cuda:$k &
done; wait
python examples/rl/merge_columns.py --corpus-root <corpus> --split val --out <dir>/after.json \
    <dir>/shard-*/after.json
python examples/rl/eval_rocjitsu.py --corpus-root <corpus> --split val --out <dir> --column none --compare
```

`before.json` is computed the same way with `--column before`. Computing a
column needs a GPU and `transformers`, as `eval_episodes.py` does;
the tests cover the pairing rules, the resume refusals, `--shard` and the merge.
The merge tests run on a synthetic column and, where `$AORTA_RJ_TEST_COLUMN`
names a recorded one (with `$AORTA_RJ_TEST_CORPUS_V2` for its corpus), on that
too.

## Expert iteration: `rocjitsu_demos.py`, `mix_demos.py`, `sft_rocjitsu.py`

One cycle rolls a policy out on the train split, keeps the episodes the
environment scores as solved, mixes in replay from earlier cycles, and
fine-tunes on the result.

- **`rocjitsu_demos.py`** harvests demonstrations. Every episode is replayed reply
  by reply through `rocjitsu_env.Episode`, so each step's prompt is exactly the
  one the trainer renders, and only solved episodes are kept: the exact location
  on a faulty scenario, "none" on a clean control. The sources are an evaluation
  column (`--column`, accepted only if it was generated under the system prompt
  the demonstrations are built under, or under the profile `--generated-under`
  names), a trainer's `wire.jsonl` (`--wire`, refused unless the `train-log.json`
  beside it records that system prompt, or if a replayed reward differs from the
  recorded one), the scripted evidence-window recipe
  (`--recipe`, which reads only what the agent was shown), and `--quirk-demos N`
  copies of verified episodes on clean controls whose only waitcheck diagnostics
  are its known `s_load` write-after-write report. It keeps at most
  `--max-wire-per-scenario` distinct episodes per scenario, the highest-scoring
  first with `--best-first`. Each input is replayed under the turn budget it
  recorded, and an episode is named by its input's resolved path, its trainer
  iteration and its index.
- **`mix_demos.py`** mixes demonstration files for a continual cycle: `PATH:N`
  takes N whole episodes from a file, so an episode is kept or dropped with all
  of its turns. Every input must have been rendered under one system prompt.
- **`sft_rocjitsu.py`** fine-tunes with the GRPO trainer's model, numerics and
  loss: each step is a `(prompt, completion)` sample with advantage +1 and no
  KL term, so the objective is the completion's mean log-likelihood, end-of-turn
  token included. It refuses demonstrations rendered under another system
  prompt, and `model.embed_tokens` stays frozen, so `verify_checkpoint_delta.py`
  audits the step as it does a GRPO one.

```bash
python examples/rl/eval_rocjitsu.py --corpus-root <corpus> --split train --out <cycle>/gen \
    --before <policy> --column before --episodes-per-scenario 8 --temperature 1.0
python examples/rl/rocjitsu_demos.py --corpus-root <corpus> --split train --out <cycle>/new.jsonl \
    --column <cycle>/gen/before.json --best-first
python examples/rl/mix_demos.py --out <cycle>/mix.jsonl <cycle>/new.jsonl <previous>/mix.jsonl:300
python examples/rl/sft_rocjitsu.py --demos <cycle>/mix.jsonl --corpus-root <corpus> --out <cycle>/sft \
    --init-from <policy> --epochs 1 --lr 1e-5 --batch 16 --grad-checkpointing
```

The harvest tests run on a small synthetic corpus, and again on the recorded
corpus-v2 archive where `$AORTA_RJ_TEST_CORPUS_V2` names one; the tests of the
recipe and the quirk demonstrations need that archive. Fine-tuning itself needs
a GPU and `transformers`.
