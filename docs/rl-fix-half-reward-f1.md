# Fix credit as F1: the shotgun is gone, and the obstacle has moved

Sequel to `rl-fix-half-reward-findings.md`, which measured the fix half with **fix
credit as containment** -- 1.0 if the proposal held at least one mitigation that
actually resolved the reproducer -- and reported that it flipped nothing under any
of the 22 hypotheses. The diagnosis there was that containment is *monotone in list
length*: a policy naming the whole candidate set names every possible resolver by
construction, so it banks 1.0 for free. That is a property of the **formulation**,
not of execution-checking, which is why it was worth exactly one reformulation.

**Result, up front, and it is mixed rather than negative.** Scoring fix credit as
F1 over the resolvers does what it was supposed to do: it removes the shotgun from
contention entirely, and it turns on GRPO advantage in every group of the narrow
set. It does **not** produce a trustworthy flip of criterion 1. Criterion 1 now
holds under 3 of 20 wide hypotheses where containment held under 0 of 20 -- but
those three are the same habitual names that produced the weight-window artifact
last time, and they are discounted on the same evidence. **The binding constant has
changed identity**, and that is the real finding: the opponent is no longer the
shotgun, it is the one-name abstainer, whose score comes almost entirely from the
*form* half.

## Why F1, stated before the measurement

The argument is correctness, not the number. It is recorded in commit `e766765`,
which contains the formulation and no measurement of it, so that the reasoning
cannot have been retrofitted.

`triage_reward.py` already scores a set against a set this way, one layer up:
`0.4 × attribution-F1` over the cited detector IDs, chosen as F1 rather than "cited
at least one" precisely so that **citing everything cannot win**. A single reward
that scores attribution by F1 and the fix by containment holds two different
opinions about the same question. Removing that inconsistency is right whichever
direction the resulting number moves -- which is what separates this from tuning,
and is why the formulation was committed before it was scored.

The consistency is now structural rather than asserted. The F1 was extracted from
`score_answer` into `triage_reward.set_f1` and **both terms call that one
function**, so they cannot drift and cannot disagree about the empty cases. The
arithmetic is unchanged, and no recorded triage number moves (re-scored: verdict
accuracy 1.000 / attribution F1 1.000 / reward 1.000 on the contract-perfect
policy, exactly as before).

One consequence is worth naming because the earlier document argued against it.
Breadth is now priced on **both** sides -- precision falls as `1/k` here, and
`precision_credit` scales the tier 4-5 block by `min(1, 2/k)` on the form side. The
previous decision to price it only once rested on containment being the fix
formulation, so it does not survive the change. F1 introduces no coefficient, so
this adds no free parameter; `FIX_WEIGHT` is still the only one and still 0.5.

## The criteria, before and after

Every hypothesis about which mitigation resolves these scenarios is scored and
**none is chosen** -- the recorded rollouts are all `mode: sanitizer` and archive no
mitigation matrix, so no resolver ground truth exists for them. Nothing is selected,
so nothing can be selected favourably. Both columns are at the pre-registered
`FIX_WEIGHT = 0.5`.

| | containment | F1 |
|---|---|---|
| narrow: criterion 1 holds | 0 of 2 hypotheses | 0 of 2 |
| narrow: criteria passing | 1 of 4 | **2 of 4**, under both hypotheses |
| narrow: best constant | 0.9500 | 0.9500 / 0.7833 by hypothesis |
| wide: criterion 1 holds | 0 of 20 hypotheses | 3 of 20 (discounted below) |
| wide: criteria passing | 2 of 4 | 2 of 4, or 4 of 4 under those 3 |
| wide: best constant | **0.7700** (shotgun, in 19 of 20) | **0.4500** (pick-first, in 20 of 20) |

### What is real, and does not depend on which hypothesis is true

**1. The shotgun is out of contention.** On the wide set the best constant falls
from 0.7700 to 0.4500, and the binding constant changes identity from
`abstain_and_shotgun` to `abstain_and_pick_first` under every one of the 20
hypotheses. The shotgun's own composite falls to 0.3176: it still pays form 0.54,
but its fix credit is now `2/21 = 0.095` instead of 1.000. This is the exploit the
first measurement ran into, closed by construction rather than by reweighting.

**2. The narrow set became trainable.** Criterion 3 -- non-zero within-group spread,
which is what GRPO needs to produce any gradient at all -- goes from 5 of 9 groups
(form half alone) to 7 of 9 (containment) to **9 of 9 (F1), under both hypotheses**.
A finer-grained term discriminates between samples the form ladder scored
identically. This is the most useful thing in this document and it is independent of
any ground-truth guess.

### What is not real: the 3-of-20 flip

Criterion 1 holds under `debug_hip_dynamic_queues_1`, `debug_hip_dynamic_queues_2`
and `debug_clr_no_batch_cpu_sync`. These are the **same three names** that produced
the 0.2-0.4 weight window last time, and they are discounted on the same evidence:
they are a naming habit, not a diagnosis. Under the F1 term, per scenario:

| scenario | names it | mean F1 |
|---|---|---|
| `consan-clean` (**no failure at all**) | 0.80 | 0.311 |
| `consan-racy` (a real race) | 1.00 | 0.417 |
| `consan-gemm` (a real failure) | 0.40 | 0.086 |

A policy that proposes a mitigation in 80% of samples for a scenario where nothing
is wrong is not selecting it because of the evidence. Under a hypothesis chosen for
diagnostic plausibility instead -- `tf32_off`, the mitigation the synthetic probe
actually resolves -- criterion 1 fails: 0.4401 against 0.4500. Closely, which is
worth recording, but on the wrong side.

**The weight behaviour is a different shape this time, and it still does not rescue
the flip.** Under containment the favourable region was a *window*, 0.2-0.4, failing
on both sides -- the signature of a coincidence. Under F1 it is a monotone
threshold: 0 of 20 hypotheses flip at weights 0.2 and 0.3, 3 of 20 at 0.4 and 0.5, 5
of 20 at 0.6, 9 of 20 at 0.8. The pre-registered 0.5 sits inside it without having
been chosen to. A threshold is weaker evidence of tuning than a window is, and the
count rising with weight is what you would expect if the fix half carried real
signal. But the *identity* of the flipping hypotheses is unchanged, and that is
sufficient to discount them: at higher weights the reward simply pays more for the
model's vocabulary.

### Why narrow still fails, and it is now a different reason

`abstain_and_pick_first` names exactly one mitigation. Under F1 a single name that
happens to be the resolver scores a **perfect 1.0** -- precision 1, recall 1 -- so
with two mitigations offered, a blind one-shot guess is right half the time and
unbeatable when it is. That is not a defect in F1; it is the residual fact that the
criteria are evaluated per-hypothesis rather than in expectation over hypotheses,
and a two-element candidate set is too small for that distinction to wash out. It is
the same reason the narrow set was rejected for acceptance in the first place.

## The obstacle has moved out of the fix half

The previous document's closing sentence was "breadth is underpriced". That was
right about containment and is now **too coarse**. Breadth is priced in the fix half
(precision, `1/k`) and on the form side (`precision_credit`, `min(1, 2/k)`). What
remains is that the surviving opponent, `abstain_and_pick_first`, earns **0.9000 on
the form half for reading nothing** -- it abstains on category, names one
mitigation, and collects the honest-abstainer score. Its entire wide composite of
0.4500 is `0.5 × 0.9000 + 0.5 × 0`.

So the accurate statement is not "execution-checking does not work". It is:

> Execution-checking removed the degeneracy it was built to remove. What now
> separates the model from a constant is the **form** half paying 0.9000 to a
> two-line policy that reads no evidence -- which is the finding the one-pager
> already carries, arrived at from the other direction.

That is a sharper and more actionable place to be than either previous position, and
it points at the form ladder's abstention credit rather than at a third fix-half
formulation. **No third formulation was tried, and one should not be** until there is
a real reproducer to measure against.

## Breadth in its third location: the iteration budget

Separate from the reward, and **not folded into any number above**.
`loop.py:490-499` appends every proposed name to the mitigation axis and the probe
runs one cell per entry; `loop.py:501` then charges the budget exactly once for the
whole cycle. So breadth is free in the production loop too.

Measured on the recorded proposals, after filtering to names the loop would actually
run:

| set | mean names/proposal | max | cells a `--max-iterations 8` run may spend |
|---|---|---|---|
| narrow | 1.27 | 2 | 16 |
| wide | 4.69 | 20 | **160**, while the budget reads 8 |

**Not repaired here, deliberately.** Charging cells changes what `--max-iterations`
means on the CLI, which is a product decision and not a reward experiment; and
conflating it with the F1 measurement is exactly what the two-variant separation
exists to prevent. Pinned as a characterization test
(`test_the_iteration_budget_charges_one_unit_for_a_whole_proposal`) with a docstring
saying it is a finding rather than a desideratum.

One detail makes the repair cheaper than it looks: the agent log **already records
one `mitigation_tried` event per name**, so the resume path in `state.py` could
reconstruct an exact cell count with no schema change. Only the CLI semantics need
deciding.

> ⚠ **The probe scenario is still synthetic** -- a bash exit code keyed on
> `DISABLE_TF32`, not a GPU failure. Every statement above about the *mechanism*
> stands on it; no statement about real reproducers does. What would settle that is
> unchanged: a probe-mode run on a real reproducer across a mitigation axis where at
> least one mitigation genuinely flips the verdict and most do not, with rollouts
> generated against that scenario's evidence. Cheapest committed public candidate
> remains `recipes/llm-determinism/example-llm-determinism.yaml`, and it needs a GPU.
