# Minimum-cost root-cause localisation: the joint action, and the objective

Two changes, one demo. The first is architectural: the agent can now act on
**both** probe axes instead of one. The second is the reward: correctness
**minus the probe cells spent**, with form demoted from a score to a gate.

Everything here is measured on CPU against artifacts already on disk, except
the one live episode, which is a real Qwen3-8B on Slurm.

---

## 1. The gap that was closed

`aorta/agent/loop.py` has always built a probe recipe with two axes and forced
the baseline diagnostic into the second one:

```
diagnostic_axis = list(recipe_template.get("diagnostic_axis", [_BASELINE_DIAGNOSTIC]))
```

and `probe/recipe_builder.py` synthesises the cells as the cartesian product of
the two. But the loop only ever appended to `mitigation_axis`, and
`AgentStep` had `next_mitigations` and no diagnostic field. So the diagnostic
was fixed by whoever wrote the recipe, and the policy had **no action** for it:
it could buy a causal test and never buy information.

Confirmed in both files before anything was changed.

### What was added

| layer | change |
|---|---|
| `agent/llm.py` | `AgentStep.next_diagnostics`, defaulting to `[]`; the same defensive parse as `next_mitigations` (only a genuine JSON list — never `list("xnack")`); a second-axis prompt, emitted only when the axis is armed; `_step_from_content` filters proposed diagnostics against what was offered |
| `agent/policy.py` | `validate_step` cleans both axes through one shared `_clean_axis_names` — registry, shell-shape and cell-name safety cannot drift between them; new `max_probe_cells` budget with `cells_affordable` / `check_cell_budget` |
| `agent/loop.py` | `plan_axis_growth` grows both axes and prices the result in cells; `diagnostics_allowlist`; an `axis_growth` log event; a diagnostic-only step is a legal action |
| `agent/state.py` | `tried_diagnostics`, `probe_cells_spent`, and the `wake()` replay for both |
| `cli/agent_mitigate.py` | `--diagnostic` (repeatable) and `--max-cells` |

### Backward compatibility, and what it rests on

A model reply with no `next_diagnostics` parses to `[]` and behaves exactly as
before. Three further guarantees, each with a test:

* **the prompt is byte-identical** when no diagnostic is offered, so a
  single-axis run pays nothing for a feature it is not using;
* **the diagnostic axis is empty unless the operator arms it.**
  `_list_candidate_diagnostics` is deliberately *not* symmetrical with
  `_list_candidate_mitigations`, which falls back to the recipe's axis order
  and then to the whole registry. A recipe's own `diagnostic_axis` is not a
  candidate list — it is diagnostics the operator wants run from the first
  cycle — and falling back to the registry would arm a multiplicative action
  on every existing recipe at once;
* **a proposed diagnostic on an unarmed axis is refused loudly** (`policy_stop`),
  not silently dropped.

All 81 pre-existing `tests/agent/` tests pass unchanged. No existing test
needed to change.

---

## 2. The cell-cost decision

**A joint proposal does not cost the number of names in it.** The cells are the
cross product and the loop cannot ask for a single cell — it can only widen an
axis. Admitting `a` mitigations and `b` diagnostics onto axes sized `M x D`
runs

```
(M + a) * (D + b) - M * D  ==  a*D + b*M + a*b     new cells
```

One mitigation plus one diagnostic onto a 4 x 1 grid is **9** new cells, not 2.
The new diagnostic has to be paired with every mitigation already on the axis.

Three decisions follow, and they are the ones that matter for the demo.

**(a) The charged cost is the enumerated grid, not a formula.**
`plan_axis_growth` calls `probe_cell_names`, which builds the names through
`recipe_builder._safe_cell_segment` — the builder's own slug function. So the
number charged is the count of cell directories that do not exist yet and will
therefore run. A test asserts `state.probe_cells_spent` equals the cell
directories on disk after a real (engine-free) loop run.

**(b) The budget is denominated in cells, and it is a second budget.**
`--max-iterations` bounds *proposal cycles* and always has. The defect already
characterised in this repo is that the loop appends every proposed name and
charges that budget exactly once, so a wide-candidate run can spend 160 cells
against a budget of 8. Adding a second axis to a cost model that already
under-counts would make it under-count *multiplicatively*. Rather than change
what `--max-iterations` means — a CLI product decision, not a reward
experiment — `max_probe_cells` is a new, opt-in budget (`None` = off, so the
shipped behaviour and the existing characterization test are untouched).

**(c) A proposal that would breach the budget is trimmed, not overrun.**
Names are admitted one at a time **in the order the policy proposed them**,
mitigations then diagnostics as the schema lists them, and the first name that
would breach the budget — and every name after it — is rejected. The total for
a fully admitted proposal does not depend on the order; which subset survives a
trim does. Cheapest-first was rejected deliberately: on a cross product the
cheapest name is whichever axis is currently shorter, which is an artefact of
history rather than a judgement about the failure. If nothing can be admitted,
the run stops (`policy_stop`) instead of re-running a grid it has already paid
for.

**A consequence worth stating to anyone planning on this.** The diagnostic
action gets *more* expensive as the search proceeds, because `M` grows. Buying
information early is cheap and buying it late is not — which is the right
incentive for minimum-cost localisation, and it was not designed in, it falls
out of the cross product.

---

## 3. The objective

`examples/rl/cost_aware_reward.py`, beside the existing scorers rather than
replacing them, so both can be run on one set of rollouts in one command.
`proposal_reward.py`, `triage_reward.py` and `fix_reward.py` are imported, not
reimplemented.

### Pre-registered weights

Fixed before anything was measured, and not moved since:

```
W_TRIAGE = 0.3     verdict correctness + attribution F1 over cited detector IDs
W_FIX    = 0.5     F1 of proposed mitigations against the true resolver set
W_COST   = 0.2     probe cells spent, normalised against a budget of 10
```

They are in the module docstring and in the commit message. This project has
twice found weight regions that flattered a result, so a weight change is a
finding to report, not a coefficient to search. **No weight was changed.**

### The four rules

1. **Form is a gate.** Valid JSON, the required keys with the right types, a
   category in the closed autopsy set, and every proposed name on both axes
   registered *and* currently offered. Fail any → total 0.0, nothing else
   computed. Every condition is one the shipped path already enforces; what
   changes is that meeting them is worth **zero** instead of up to 1.0. That
   ladder is the whole reason a two-line constant that reads nothing scores
   0.9000 against Qwen3-8B's 0.7569.
2. **Triage** — `triage_reward.score_answer`, unchanged: `0.6 * verdict-correct
   + 0.4 * attribution-F1`.
3. **Fix** — `fix_reward.fix_credit`, the F1 over resolvers recovered from the
   archived matrix by listing cell directories. GPU is paid once per scenario,
   never per rollout sample.
4. **Cost** — cells spent over the budget, clipped to `[0, 1]`, and the cells
   are counted by **`loop.plan_axis_growth` itself**. The reward cannot price
   an action differently from the code that runs it.

### The special case: an empty resolver set

Some real failures are resolved by nothing in the registry — the archived
fp16-overflow scenario is one, measured against 21 mitigations, all of which
fail. Without a special case the fix term is 0.0 for every policy there, which
teaches *always guess, guessing is free*. So **when the true resolver set is
empty and the baseline did fail, correctly reporting that nothing resolves it
earns full fix credit.**

"Correctly reporting" needs no new schema field: propose no mitigations *and*
set `stop`. An empty list without `stop` is a policy with nothing to say, which
is a different claim and earns nothing. And the same claim on a *resolvable*
scenario earns nothing, so a policy that always gives up cannot win both.

### Two things about the scale, stated rather than smoothed over

**The ceiling is 0.8, not 1.0** — `W_TRIAGE + W_FIX`, reached only by a policy
that spends zero cells, which is impossible because the baseline cell always
runs. Today's objective tops out at 1.0. **The two columns are not on the same
scale. Compare the ranking, which is all GRPO reads.**

**The total is floored at 0.0.** Without the floor a gate failure (0.0 by rule
1) would outscore a well-formed answer that earned little and spent a lot,
which teaches the model to emit garbage rather than try. The floor buys that at
the price of a **flat spot**: several distinct policies that earn nothing all
tie at 0.0, so GRPO sees no advantage between them. That is a defect of this
formulation. It is pinned by a test whose docstring says it is a finding, and
it is *not* fixed by moving a weight — fixing it means deciding what a gate
failure is worth, which is a design question.

---

## 4. The measurement

64 recorded Qwen3-8B completions (Slurm job 35729, t=0.7, whole registry
offered), re-scored from their raw wire text under both objectives. The
fix-half ground truth is the archived 8-cell x 4-trial matrix from Slurm job
35724, resolver `pytorch_no_cuda_memory_caching`.

| policy | reads evidence | cells | today | proposed |
|---|---|---|---|---|
| oracle | yes | 3 | 1.0000 | **0.7400** |
| **qwen3-8b** | yes | 5 | **0.4543** | **0.0788** |
| abstain_and_pick_first | no | 3 | 0.4500 | 0.0075 |
| abstain_and_shotgun | no | 22 | 0.3176 | 0.0000 |
| always_prose | no | 0 | 0.0000 | 0.0000 |

**Today the model beats the best constant by 0.0043 — a 0.95% margin, thinner
than the 0.0111 margin this project already learned to distrust.** Under the
proposed objective it beats it by 0.0713, and the shotgun that spends 22 cells
against a budget of 10 is zeroed by the cost term.

Per-term, under the proposed objective:

| policy | triage | fix | cost penalty | total |
|---|---|---|---|---|
| oracle | 1.0000 | 1.0000 | 0.3000 | 0.7400 |
| qwen3-8b | 0.1250 | 0.1518 | 0.5016 | 0.0788 |
| abstain_and_pick_first | 0.0500 | 0.0000 | 0.3000 | 0.0075 |
| abstain_and_shotgun | 0.0500 | 0.0952 | 1.0000 | 0.0000 |
| always_prose | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### The part that flatters the proposal

Some of that widening is **not** the objective working. The model is the only
policy with a `verdict` at all — the three constants emit none — so it alone
earns on the 0.3 triage term. Setting the triage **weight** to zero for every
policy alike (not withholding the label, which would let the constants keep the
passing cell's empty-empty attribution credit and would not be like-for-like):

| | model | best constant | margin |
|---|---|---|---|
| today | 0.4543 | 0.4500 | 0.0043 |
| proposed | 0.0788 | 0.0075 | 0.0713 |
| proposed, triage weight zeroed for all | **0.0441** | **0.0000** | **0.0441** |

So about 38% of the proposed margin comes from a term the constants do not
compete on. **The conclusion survives, and in a sharper form:** with that term
removed entirely, *every* constant scores exactly 0.0 and the model does not.
`abstain_and_pick_first` earns 0.0000 of fix credit against the model's 0.1518,
and the shotgun's 0.0952 of fix credit is wiped out by spending 22 cells
against a budget of 10. But 0.0441 is the figure to quote, not 0.0713.

Two more things that were checked rather than assumed:

* **An absent verdict is scored as a wrong verdict, not defaulted to the
  label's.** The first draft defaulted it, which handed every policy that reads
  nothing a free 0.6. Caught before any number was reported; there is now a
  test.
* **On the one cell that passes, citing no detectors is perfect attribution by
  convention**, so a policy that reads nothing collects 0.4 of the triage term
  there (0.05 of the mean, over 8 of 64 completions). That convention is
  correct for triage in isolation and is left unchanged per rule 2. Worth
  knowing when the term is one of three.

### The durable number

The model's **fix rate is 0.1518 against a chance rate of ~0.077** — about
twice chance, unchanged from the earlier measurement, and the honest headline.
Real signal, not enough to train on yet.

### Effective sample size is one

The eight `scenario_id`s are eight cells of a **single** reproducer sharing a
single ground truth: 64 completions on one problem, not eight problems. A
second unrelated resolving scenario is worth more than more samples on this
one, and is the prerequisite before any claim that the objective is right.

---

## 5. The live episode

`examples/probe/numerics/run_joint_axis_episode.sbatch`, `meta64` only, with an
explicit refusal if the allocation contains an `smci350` node. Serves Qwen3-8B
through `serve_for_rollouts.sh` — which exists for exactly this consumer and
defaults `--grammar-backend xgrammar` on, because `LiteLLMProposer` sends
`response_format={"type": "json_object"}` on every call and a default engine
answers that with a 500 — then drives `aorta agent mitigate` live across both
axes with `--diagnostic amd_log_level_4 --diagnostic hip_launch_blocking` and
`--max-cells 10`.

It is **live rather than a replay**, and the reason is specific: the archived
matrix has a diagnostic axis of exactly `[none]`, so it contains no cell a
diagnostic action could be observed at. Replaying the joint axis against it
would mean inventing the observation.

A `fake`-backend control runs the same recipe and budget. That is not
decoration: `FakeLLMProposer` walks both axes in registry order, so a live
episode that looks like the control has demonstrated the plumbing and nothing
about the policy.

`provenance.episode_backend` records what actually produced the steps. A fake
episode is never labelled real, and the payload adds a caveat saying so.

### What actually happened — Slurm job 38192, `cv350-rck-g03-e23-08`

**The episode is real.** Qwen3-8B, live, budget 10 cells:

| step | category | mitigations | diagnostics | cells | cumulative |
|---|---|---|---|---|---|
| 1 | `illegal_mem` (0.65) | `tf32_off`, `hsa_no_sdma` | `amd_log_level_4` | **+5** | 6 |
| 2 | `illegal_mem` (0.85) | `pytorch_alloc_expandable_segments` | `hip_launch_blocking` *(refused)* | +2 | 8 |
| 3 | `unknown` (0.60) | `pytorch_no_cuda_memory_caching` | `hip_launch_blocking` *(refused)* | +2 | 10 |

Outcome `resolved`, `winning_mitigation: pytorch_no_cuda_memory_caching` — the
real resolver, found on the third cycle, at exactly 10 cells against a budget
of 10. Ten cell directories on disk, matching `cells_total: 10` exactly.

Three things this demonstrates that a synthetic run could not:

* **The model uses the joint action.** Its first move was two mitigations *and*
  a diagnostic in one step — an action that did not exist before this change.
* **The cross-product cost is real and was charged.** Three names cost **5
  cells**, not 3: `M=1, D=1, a=2, b=1` gives `2*1 + 1*1 + 2*1 = 5`.
* **The budget trim fired on live traffic, twice.** At step 2 with 6 cells
  spent and 4 affordable, `pytorch_alloc_expandable_segments` took 2 and
  `hip_launch_blocking` would have taken 4 more — so it was refused rather than
  run, and refused again at step 3. Without the trim those two steps would have
  spent 12 and 14 cells against a budget of 10.

**The control is the contrast, and it is sharper than expected.** The
registry-order walk bought *both* diagnostics early, reaching a 3 x 3 grid and
9 cells by its second cycle, and then stopped: `policy_stop — No axis could be
widened: every proposed name was already on an axis or would exceed the probe
cell budget (10 max, 9 spent)`. **It never reached the resolver.** So the same
budget either finds the cause or is spent on evidence, depending on the policy
— which is the argument for training one.

One honest note on the model's read: it labelled the failure `illegal_mem`
twice and then `unknown`, and neither is right (the cause is an uninitialised
read, which is legal and silent). It found the right *fix* without ever naming
the right *cause* — which is the labelling gap, showing through on live
traffic.

### `terminal_observed`: why the per-step evidence is not enough

`steps[].observed` is the evidence the policy had **when it chose**, so a
cell's result lands on the *following* step and the last step's results land
nowhere. On a converging episode that hides the punchline: the run reports
`outcome: resolved` and a winner while every one of its 8 per-step observed
entries is `fail`. A page rendered from the steps alone shows a search finding
a fix with nothing on screen showing a fix working.

`episode.terminal_observed` is the final state of the grid — every cell the
episode ran, with the verdict it ended with, same element shape as
`steps[].observed`. It is enumerated from the **run directory's own cell
directories**, because a directory holding `trial_*/result.json` exists if and
only if the workload ran there. Deliberately not the archived matrix and not
the axes: the grid's meaning is "squares this policy chose to spend on", and
padding it with unbought cells would make the search look more thorough than it
was. The archived matrix has 8 cells on a one-diagnostic axis and this episode
ran 10 across two, so neither is a subset of the other and a padded grid would
be visible — a test asserts exactly that.

Both counts reconcile against two independent sources: `terminal_observed`
comes from the cell directories, `cells_cumulative` is replayed from the log's
`axis_growth` events. Live 10 = 10, control 9 = 9. An explanatory caveat is
emitted only on a mismatch, and does not fire.

**Two cells pass in the live episode, not one:**
`pytorch_no_cuda_memory_caching-none` *and*
`pytorch_no_cuda_memory_caching-amd_log_level_4`. Only the first is the
attributed win — `winning_mitigation` requires the baseline diagnostic,
because a pass with a diagnostic switched on is not attributable to the
mitigation alone. But the pair is itself a result: the mitigation resolved the
failure with the diagnostic both off and on, which is the evidence that
`amd_log_level_4` is inert with respect to the outcome. That is what a
diagnostic is supposed to be, and it had not been checked before.

The control's terminal grid is 9 cells and **every one fails.** That is the
cost-term argument as a picture rather than a number.

### Two things that cost time and are worth recording

**The compute nodes have python3.9 and python3.11; the head node has 3.10.**
A venv built on the head node produces a `bin/python` whose site-packages is
not present on the node, and the symptom is `ModuleNotFoundError: No module
named 'aorta'` from a script that ran — which reads like a broken install
rather than a wrong interpreter. Build the venv on the node with an explicit
`python3.11`, as `run_nan_probe.sbatch` already did.

**Engine bring-up was 430s on a node without the image cached** (the pull is
most of it), against a ~4s-per-trial workload. The GPU cost of this demo is
almost entirely the model server, not the reproducer.

---

## 6. What is not done

* **The triage term is not exercised by the recorded rollouts on equal
  footing.** The constants would need verdict-emitting variants to compete on
  it; the payload contract's policy enum has no slot for one.
* **One scenario, one ground truth.** See "effective sample size is one".
* **The flat spot at 0.0** is recorded, not fixed.
* **`--max-iterations` still charges once per cycle.** Deliberately unchanged;
  the new budget is additive. The eventual fix is cheap — the log already
  emits one `mitigation_tried` (and now `diagnostic_tried` and `axis_growth`)
  event per name, so `wake()` can reconstruct an exact cell count with no
  schema change — but it changes what a shipped CLI flag means.
