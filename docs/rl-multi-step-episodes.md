# Multi-step episodes for GRPO: design, predictions, and one paired evaluation

This is the design of the multi-step episode path under `examples/rl/`, the
predictions that were written down before any training run, and one fixed-weight
evaluation of a trained checkpoint. Its conclusion is narrow and is stated in
the results section. In short: training measurably changes behaviour on the
training scenarios, but this corpus cannot tell learning to debug apart from
memorising the answer key.

| file | what it is |
|---|---|
| `examples/rl/episode_env.py` | offline environment: the policy proposes, an archived probe matrix answers, repeat |
| `examples/rl/event_reward.py` | the discrete per-step event reward an episode is scored with |
| `examples/rl/rescore_episodes.py` | CPU reports: constant policies, a run's wire, and a replay of recorded episodes |
| `examples/rl/train_grpo_step.py` | GRPO over episodes, in-process `transformers`, one GPU, every update checked |
| `examples/rl/eval_episodes.py` | a fixed checkpoint against a control, paired, no optimiser step |
| `examples/rl/verify_checkpoint_delta.py` | the checkpoint pair on disk, on CPU: floor, control and an Adam ceiling |

## Why episodes

A single-step task (one scenario, one archived matrix, one completion scored
against it) is an input/output pair. When the answer is one name from a 21-name
menu it can be memorised by construction, because some fixed string scores
well. A sequence of decisions, each conditioned on what the previous one
revealed, has no such string: after the first reply the menu has changed.

Three measured properties of the single-step regime have the same cause:

1. Several step events can never fire in one completion. `name_already_tried`
   needs a history. `name_not_offered` and `malformed_reply` are invisible to
   a scorer that reads the loop's log, because the loop filters or refuses them
   before logging.
2. `terminal_unresolvable_correct` is unreachable. Making the claim needs a
   stop with nothing proposed. Earning it needs every candidate already tried.
   One reply cannot do both.
3. A constant policy counts as a complete policy when an episode is one step
   long.

## Design

**1. The loop's decisions, with the executor replaced.** `run_agent_loop` is
not called, because GRPO has to advance a whole group of episodes per scenario
in one batched `generate`, and the model call sits inside that loop. The
environment calls the loop's *decision functions* instead:
`_read_cell_summaries`, `_baseline_passed`, `_find_winning_mitigation`,
`_step_from_content`, `AgentPolicy.validate_step`, `check_iteration_budget`,
`pending_approvals` and `_resolve_stop_outcome`. It also writes an
`agent_log.jsonl` through `append_log_event`. A test drives the real
`run_agent_loop` against the same archive with the same scripted replies, then
requires the two logs to match event for event. It covers seven cases:
converge, re-proposing a tried name, an explicit stop, the iteration budget,
an unparseable reply, a refused category, and a partial drop.

There is one deliberate divergence from the shipped real-LLM proposers. They
turn an empty remaining menu into an `exhausted_candidates` stop without
calling the model. The environment asks the policy anyway, because that stop
belongs to the proposer rather than the policy.

**2. One advantage per episode; the group is the episodes on one scenario.**
Steps are not exchangeable, since step 3 faces a state step 1 never saw.
Episodes on one scenario are exchangeable. The episode's advantage is repeated
on each of its steps, which is REINFORCE's trajectory gradient. The cost is that
a good step inside a bad episode gets pushed down along with the rest; without a
critic that cost does not go away.

**3. The shipped budget.** `max_iterations = 8`, the `AgentPolicy` default. An
episode ends on convergence, an explicit stop, the budget, or the menu running
out: when every name is dropped by the filter, the loop stops with
`proposal_unresolved`.

**4. The observation.** The policy sees `_read_cell_summaries`: categorical
fields and no numbers. So a mitigation that changed the run without changing the
verdict is invisible to it. This costs nothing on a corpus whose every
non-resolving cell is a plain `fail`.

**The reward** (`event_reward.py`) is a sum of events, each decidable from the
reply, the log and the archived matrix with no judgement call. The point table
is pinned by a test. Two preconditions matter most:

- `terminal_unresolvable_correct` pays only when every offered candidate was
  tried first. Otherwise the event is `terminal_unresolvable_unearned` and
  scores 0.0.
- It pays only when the stop was the policy's own. If the candidate filter
  emptied the list, the ending is `terminal_proposal_unresolved`, valued the
  same as giving up. Without this rule, the award would be paid for a terminal
  the loop manufactured out of a name-resolution failure (aorta#449).

That second rule relies on the `proposal_unresolved` stop reason, which this
change adds to `src/aorta/agent` as a separate commit.

**The corpus** is seven archived probe matrices. Each has 22 cells × 4 trials,
all 21 registered mitigations measured, a failing baseline, and no cell whose
trials disagree; `load_scenario` checks all of that. The corpus has six
resolvable scenarios and one unresolvable one. They cover five mechanism
families, but only four distinct resolving mitigations between them. The
archives are not in the repository: point `--corpus-root` (or
`AORTA_RL_CORPUS_ROOT`) at a directory holding them by name.

## Pre-registered predictions

These were written down with the code and before any training run:

- **P1**: `xnack_page_fault`'s step-1 resolver rate rises and does not decline.
  Falsified if the step-1 rate stays flat while the episode rate rises. That
  outcome would mean episodes bought extra draws rather than learning.
- **P2**: `name_already_tried` fires, and its rate falls as the policy learns to
  read the shrinking menu.
- **P3**: `malformed_reply` and `name_not_offered` become reachable, but not
  necessarily frequent.
- **P4**: the diagnostic events stay at zero. Every archive has a single
  column, and the loop does not treat choosing a diagnostic as an action.
- **P5**: `terminal_unresolvable_correct` becomes reachable but is rarely
  taken. The system prompt never says what `stop` does.
- **P6**: constant policies get worse inside an episode. Their names are
  filtered out after the first reply, so they end on a negative terminal.

Two rates are reported, and they must not be confused. The **step-1 resolver
rate** measures the same question on every episode. The **episode resolver
rate** is close to 1.0 by arithmetic, because an episode gets up to eight draws
from a shrinking menu. Quoting only the second would overstate what training
did.

## Results: one paired, fixed-weight evaluation

**Setup.**

- Checkpoint: Qwen3-8B after 17 GRPO iterations. Settings: Adam at lr 1e-6,
  KL to the frozen base at β = 1e-3, 8 episodes per scenario per iteration,
  all seven scenarios, fp32 weights.
- The trainer used was the exploratory version this one is derived from. It
  had the same optimiser, sampling and checks, plus modes this version drops.
  There are two differences:
  - This version refuses to write a checkpoint from an iteration whose checks
    failed. The checks passed on all 17 iterations, so that difference was
    never exercised.
  - The exploratory version skipped zero-advantage samples entirely, which
    left groups with a flat reward without the KL term. This version applies
    KL to them.
- The 17 iterations were accumulated across three chained runs, so the Adam
  moments restarted twice.
- Every iteration passed the trainer's checks. The pair passes
  `verify_checkpoint_delta.py --lr 1e-6 --steps 17`: every trained tensor
  moved, the frozen control is bit-identical, and nothing moved further than
  the Adam ceiling.
- Evaluation: `eval_episodes.py` against the base model, 64 episodes per
  scenario per column. Weights were fixed, sampling was at the training
  settings (t = 0.7, top_p = 0.95), and each scenario was seeded from its own
  name.
- The recorded episodes of both columns replay through the code in this change
  with identical steps, terminals and rewards, 896 of 896
  (`rescore_episodes.py --replay`).

| scenario | step-1 rate, base → trained | z | reward, base → trained |
|---|---|---|---|
| `nan_uninit_workspace` | 0.328 → 0.953 | +7.37 | +4.42 → +7.75 |
| `reference_mismatch` | 0.203 → 0.672 | +5.35 | +4.26 → +6.77 |
| `queue_stale_read` | 0.656 → 0.828 | +2.22 | +6.75 → +7.21 |
| `scratch_exhaustion` | 0.688 → 0.781 | +1.20 | +3.38 → +4.73 |
| `stream_stale_read` | 0.625 → 0.688 | +0.74 | +7.02 → +6.98 |
| `xnack_page_fault` | 0.203 → **0.016** | **−3.40** | +4.97 → +3.83 |
| `cancellation_nan` (unresolvable) | – | – | −4.34 → −0.83 |
| **pooled / mean over the six resolvable** | **0.451 → 0.656** | **+5.73** | **+5.13 → +6.21** |

On the unresolvable scenario the earned claim `terminal_unresolvable_correct`
went from 41 to 54 of 64 episodes. `name_not_offered` fell from 70 to 17 events
across the corpus, `name_already_tried` from 18 to 14, and `malformed_reply`
stayed at 17 → 16.

**What this shows.** Training measurably changes behaviour on the scenarios it
trained on. The pooled step-1 rate rises at z = +5.73, and the mean reward on
the six resolvable scenarios rises by about one point.

**What it does not show.**

- **P1 is falsified.** `xnack_page_fault`, the one scenario whose resolver is
  not also a resolver somewhere else in the corpus, gets significantly
  *worse*.
- **A constant policy does as well as the trained model.** The constant that
  proposes the three names `gpu_max_hw_queues_2`,
  `pytorch_no_cuda_memory_caching` and `xnack` on step 1 names a resolver on
  every resolvable scenario, and scores +6.20 on each. That is a step-1 rate of
  1.000 and a reward of +6.20, against the trained model's 0.656 and +6.21. The
  constant reads nothing, and it ties.
- The four distinct resolvers fit in one reply, so this corpus cannot
  distinguish a policy that learned to debug from one that learned the answer
  key.
- **There is no held-out set.** The policy trained on all seven scenarios.
  What the evaluation controls is that the weights are fixed and the two
  columns are paired. It does not turn the training corpus into a test set.

The other constants mostly behave as P6 predicted. Seven of the ten score lower
as an episode than as a single reply. The exceptions are the cover itself, the
constant that names the whole menu and stops (which earns the unresolvable
claim on one scenario), and the bare unresolvability claim, which is
unchanged. Reproduce with `rescore_episodes.py --constants`, which needs only
the archives.

## What this cannot fix

- **The corpus.** The binding constraint is four distinct resolvers across
  seven scenarios. More scenarios whose resolver is already covered make the
  cover better, not worse. A held-out evaluation needs scenarios with
  resolvers the training set does not contain.
- **The action space.** Several registered mitigations do nothing on the
  measured stack (aorta#500, aorta#511), and episodes give the policy more
  turns inside the same menu without adding a resolver.
- **Stopping once the answer is visible.** `run_agent_loop` breaks on a
  winning cell before the proposer is called, so no reward can score that.
- **Credit within an episode.** One advantage per episode needs a critic to
  improve on.
- **The framework.** This is a hand-written GRPO step on `transformers`, not
  slime (`docs/rl-post-training-decisions.md`).
