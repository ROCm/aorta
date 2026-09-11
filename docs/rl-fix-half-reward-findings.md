# The execution-checked fix half: built, measured, and it does not flip the criteria

> **Superseded in part. Read `rl-fix-half-reward-f1.md` next.** Everything below is
> the measurement of fix credit as *containment*, and it stands as recorded. But the
> diagnosis in "Why it does not flip the criteria" -- that containment is monotone in
> list length -- turned out to be a property of the formulation rather than of
> execution-checking, and reformulating fix credit as F1 over the resolvers removes
> the shotgun from contention entirely. In particular this document's closing claim
> that "breadth is underpriced" is too coarse after that change; the sequel locates
> the remaining obstacle in the form half instead.

**Result, up front.** The fix half is implemented, it recovers its ground truth
offline from a real archived probe matrix, and it does **not** rescue the reward on
the recorded rollouts. Under *every* hypothesis about which mitigation resolves the
failure -- all 2 on the narrow set, all 20 on the wide set, plus the null hypothesis
-- criterion 1 still fails and the criteria count is unchanged at **1 of 4 narrow and
2 of 4 wide**. That is the finding. It is a negative one and it is worth having.

## What was built

`examples/rl/fix_reward.py`. Given an archived probe run it asks
`aorta.agent.state.winning_mitigation` which `{mitigation}-none` cells passed, and
`fix_credit` is membership of the proposal's names in that set -- 1.0 or 0.0, no
partial credit, no length scaling. The one weight is `FIX_WEIGHT = 0.5`, set on the
"the contract has two halves" principle and fixed **before** any of the numbers below
were measured.

Three properties are load-bearing and are pinned by tests:

- **The ground truth is recovered, not asserted.** Cell verdicts are recomputed
  through the probe's own resolver, so an artifact whose stored verdict contradicts
  its detectors cannot promote itself into a resolver. A pass on `none-xnack` or on
  `tf32_off-xnack` is not attributable to the mitigation alone and does not count.
- **GPU is paid once per run, never per sample.** The ground truth is a directory
  listing; scoring a rollout is a set-membership test. This is the property that makes
  execution-checking affordable for GRPO at all.
- **A scenario with no resolver is withheld, not scored zero.** Otherwise the term is
  a constant added to every policy -- the same degeneracy it exists to escape, one
  level up.

## The prerequisite, reproduced

`recipes/probe/example-probe-smoke.yaml` with a synthetic command keyed on
`DISABLE_TF32`, on CPU, in 20 s: `none-none` failed 2/2 with
`failure_detectors_fired: ["tier1:exit_nonzero"]` against a clean `tf32_off-none`.
`build_corpus.py` ingested it as 2 scenarios / 12 examples /
`artifacts: {probe_result: 2}` / `probe_trials: 4`. `fix_reward.py --matrix` then
recovered `resolvers = {tf32_off}` from it.

> ⚠ **The probe scenario is synthetic.** It is a bash exit code keyed on an
> environment variable, not a GPU failure. It proves the *mechanism* -- that a matrix
> can be archived, that the winner can be read back out of it offline, that the scorer
> works end to end -- and it says nothing whatever about real reproducers.

## Why it does not flip the criteria

Not weight, and not a bug. **Fix credit is containment, and containment is monotone in
list length.** A policy that shotguns the offered set names every possible resolver by
construction, so its fix credit is 1.0 unconditionally, for any ground truth. Pricing
breadth is the form half's job -- `precision_credit` scales the tier 4-5 block by
`min(1, 2/names)` -- and at the measured numbers it does not price it enough:

| Set | Best constant | Model, best hypothesis | Why the constant wins |
|---|---|---|---|
| narrow (2 offered) | 0.9500 | 0.8122 | at 2 names the shotgun is *free* (`FREE_MITIGATIONS = 2`) and contains the resolver |
| wide (20 offered) | 0.7700 | 0.7316 | the shotgun pays form 0.54 but banks fix 1.00 |

The narrow case is the cleaner statement, and it is decided by arithmetic rather than
by the data: with two names offered and two free, naming both costs nothing and
guarantees the answer, so the best constant is `0.5 × 0.9 + 0.5 × 1.0 = 0.9500`, while
the model's *ceiling* -- a perfect fix rate on top of its measured form score -- is
`0.5 × 0.8689 + 0.5 × 1.0 = 0.9345`. No hypothesis and no weight can close that.

The obvious repair -- score the *position* of the resolver in the proposed list -- does
not match the loop as written: `run_agent_loop` appends every proposed name to the
mitigation axis and runs the whole matrix, so it spends all k cells regardless of
order. Making breadth genuinely costly is therefore a change to the loop, or a
different term, not a reweighting.

## ⚠ There is a favourable weight window. Do not take it.

At `fix_weight` between roughly 0.2 and 0.4, under 3 of the 20 wide hypotheses, all
four criteria pass. It is an artifact, on two independent grounds:

1. **It is 3 hypotheses out of 20**, and the three are exactly the names the model
   emits most often (`debug_hip_dynamic_queues_1`, `_2`, `debug_clr_no_batch_cpu_sync`,
   each 33/45).
2. **Those names are a habit, not a diagnosis.** The model proposes them at 0.80 on
   `consan-clean` -- a scenario with no failure at all -- and 1.00 on `consan-racy`. A
   high fix rate under a fixed hypothesis is itself constant behaviour; it means the
   invented ground truth happened to coincide with what the model always says.

Tuning the weight into that window and reporting 4 of 4 would be fitting the reward to
the answer, which is the exact failure the fix half was built to escape. The number
reported above is at the pre-registered 0.5.

## What would actually settle it

A probe-mode run on a **real** reproducer, across a mitigation axis where at least one
mitigation genuinely flips the verdict and most do not, with rollouts generated against
*that* scenario's evidence. Then the fix credit measures diagnosis rather than
vocabulary. The cheapest committed public candidate remains
`recipes/llm-determinism/example-llm-determinism.yaml` (four cells, three
`mitigations: [none]` against one `tf32_off`), and it needs a GPU.

Until then the honest summary is unchanged from before this term existed: **the harness
works and the reward does not discriminate.** The fix half narrows *why* -- it is not
that the reward never looks at content, it is that breadth is underpriced -- which is a
more actionable statement than the one we had.
