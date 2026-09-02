# RL post-training seams

Demonstrations of the interfaces an RL post-training effort would need from
aorta. **Nothing here trains anything**, nothing here is wired into CI, and
nothing here is a workload. Each file exists to show that one seam is real and
to make its cost and its limits concrete.

Background and the plan these support: [`docs/tokenspeed-rl-post-training.md`](../../docs/tokenspeed-rl-post-training.md).

| | What it demonstrates | Needs |
|---|---|---|
| [`recipe_reward.py`](recipe_reward.py) | A graded reward for recipe synthesis, computed by calling aorta's own validators, with a novelty gate that refuses to pay for corpus copies | nothing — no GPU, no container |
| [`triage_reward.py`](triage_reward.py) | A reward for triage classification, labelled by aorta's own verdict resolver | nothing — no GPU, no container |
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
python examples/rl/triage_reward.py --runs path/to/runs  # real result.json files
```

**The corpus does not exist yet.** That is the *only* thing missing: labelling,
scoring and the degenerate-policy check are implemented and tested against
synthetic `result.json` fixtures shaped like what `SubprocessWorkload` writes.
Pointing `--runs` at a directory of archived runs is the remaining step.

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

Probe corpora are skewed towards `pass`, so a policy that always answers `pass`
scores well on accuracy alone — the classifier analogue of the memorisation
problem. The demo therefore scores that policy alongside the real ones:

| Policy | Verdict accuracy | Attribution F1 | Reward |
|---|---|---|---|
| oracle | 1.000 | 1.000 | 1.000 |
| right verdict, invented detectors | 1.000 | 0.364 | 0.746 |
| degenerate: always `pass`, no detectors | 0.364 | 0.364 | **0.364** |
| degenerate: always `fail`, correct detectors | 0.455 | 1.000 | **0.673** |

Reading nothing is worth 0.364 on these fixtures. A policy's reward is only
meaningful stated next to that floor, which is why the demo prints them
together.

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
