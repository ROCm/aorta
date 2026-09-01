# RL post-training seams

Demonstrations of the interfaces an RL post-training effort would need from
aorta. **Nothing here trains anything**, nothing here is wired into CI, and
nothing here is a workload. Each file exists to show that one seam is real and
to make its cost and its limits concrete.

Background and the plan these support: [`docs/tokenspeed-rl-post-training.md`](../../docs/tokenspeed-rl-post-training.md).

| | What it demonstrates | Needs |
|---|---|---|
| [`recipe_reward.py`](recipe_reward.py) | A graded reward for recipe synthesis, computed by calling aorta's own validators | nothing — no GPU, no container |
| [`probe_weight_transfer.py`](probe_weight_transfer.py) | Whether an RL iteration costs a weight update or a cold restart: drives TokenSpeed's weight-sync control plane and times it | a running `tokenspeed serve` |

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

### The memorisation caveat

**Verbatim reproduction of a committed recipe scores 5/5.**

Every tier above is a property of the artifact, not of the model's work. A
policy that learns to emit `recipes/tokenspeed/tokenspeed-serve-load.yaml`
character-for-character earns full automatic marks for retrieval. This reward is
a floor on syntactic and semantic well-formedness; it is not evidence of
synthesis, and it cannot be the only term in a training objective.

`--check-memorisation` reports the nearest committed recipe by similarity, which
is the cheap version of the mitigation. A real run needs at least:

- held-out prompts whose target recipes are not in the training corpus;
- a novelty term, or an explicit penalty against near-duplicates of the corpus;
- tier-6 grading by something that can distinguish "answers the question" from
  "is a valid recipe".

A rising tier-5 rate on its own is a memorisation alarm, not progress.

### One gap this exposes

The deepest check a reward can reach today is `Workload._validated_config()` — a
private name. `setup()` is the public entry point, but it goes on to require
Docker and a readable `/dev/kfd`, so calling it would make the reward depend on
the grader holding a GPU. A public `Workload.validate_config()` would let a
reward function commit to a supported surface instead of to an underscore, and
is the one upstream change a productionised version of this file would want.
