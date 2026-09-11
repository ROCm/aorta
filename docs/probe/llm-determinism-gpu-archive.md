# The first archived probe-mode GPU run — and why it is a control

**2026-09-11. Slurm job 35706, partition `meta64`, node `cv350-rck-g03-c07-18`,
8× AMD Instinct MI350X (`gfx950:sramecc+:xnack-`), host ROCm 7.0.2.2,
container `rocm/primus:v26.3` (torch `2.10.0+git94c6e04`, HIP `7.2.53211`).**

Headline: **no mitigation flipped a verdict, because nothing failed.** Every
cell of every configuration passed. This is a genuine negative result and it
is written up as one — the archive is real, the resolve signal is not there.

Artifacts live under `/apps/vikhande/probe-gpu/`.

## What was run

`aorta`'s `llm_determinism` workload — snapshot params+RNG, run fwd+bwd, restore,
run it again, compare bit-exact checksums of every per-block activation, the
loss, the logits and every grad — as the opaque command of a probe-mode recipe,
under `torchrun --standalone --nproc_per_node=8`.

```
aorta agent mitigate \
  --recipe recipes/probe/probe-llm-determinism-replay.yaml \
  --output /apps/vikhande/probe-gpu/agent_results \
  --ticket PROBE-LLMDET-GPU \
  --symptom "bit-exact replay of a single transformer block; checking for \
run-to-run nondeterminism in matmul / collective reduction ordering on gfx950" \
  --llm-backend fake --max-iterations 4 -vv \
  -- /apps/vikhande/probe-gpu/scripts/launch_det.sh
```

Workload config: `hidden_size=2048 ffn_size=5632 num_heads=16 seq_len=512
batch_size=1 dtype=bf16 num_layers=24 seed=1234 steps=5 checksum_mode=per_rank`.
Note `steps` here is the *workload's* replay count; `matrix.md` separately
reports "Steps per trial: 1", which is the dispatcher's own field.

### Why not `recipes/llm-determinism/example-llm-determinism.yaml`

That recipe was the obvious candidate — it already carries three `none` cells
against one `tf32_off`. It cannot be used: it is **matrix-mode**, with
`workload: llm_determinism` and `cells:` and no `mode: probe`, so
`recipe.probe_extras is None` and `run_agent_loop` rejects it at
`agent/loop.py:104`. No committed probe-mode recipe ships its own workload —
probe wraps an opaque command by design, and all five in `recipes/probe/` are
axis-and-classifier templates. So the axis came from a new probe recipe
(`recipes/probe/probe-llm-determinism-replay.yaml`, `[none, tf32_off]`,
`trials: 2`) and the workload came from the launch command.

## Verdicts

| Tree | Cell | Trials | Verdict | Cell `env` |
|---|---|---|---|---|
| `agent_results/` | `none-none` | 2 | **pass** | `{}` |
| `sweep_results/` | `none-none` | 2 | **pass** | `{}` |
| `sweep_results/` | `tf32_off-none` | 2 | **pass** | `{"DISABLE_TF32": "1"}` |

**No mitigation flipped a verdict. There is nothing to flip.** The baseline
passes, so the only honest reading of `tf32_off-none` passing is that it also
passes — not that it fixed anything.

The `DISABLE_TF32` value recorded on the `tf32_off` trials is the evidence that
the mitigation actually crossed the container boundary. Without
`env_passthrough_mode: file` and `docker run --env-file` it would not have, and
the cell would have been a second baseline wearing the label of a mitigation.

### Why there are two trees

`run_agent_loop` short-circuits a passing baseline at `loop.py:399`, *before*
winner detection and before the proposer is consulted. So the agent executed
exactly one cell and stopped with `baseline_pass`; the `tf32_off` arm was
never run. That is correct behaviour — searching for a fix to a thing that
works is meaningless — but it leaves the mitigation axis unmeasured. The
`sweep_results/` tree is the same recipe and the same launch command run as a
full matrix so the axis is on record too.

## The negative result, quantified

Scouting (job 35699, node `cv350-rck-g03-c16-18`, 3m12s) ran seven
configurations at `steps: 3`, chosen to cover the mechanisms that could
plausibly diverge rather than to repeat one shape:

| Cell | Result |
|---|---|
| `baseline-bf16-24L` (the committed baseline cell) | pass |
| `tf32off-bf16-24L` | pass |
| `fp32-24L` | pass |
| `tf32off-fp32-24L` | pass |
| `moe4-bf16-12L` (the committed MoE cell) | pass |
| `moe4-fp32-12L` | pass |
| `global-fp32-24L` (`checksum_mode: global`) | pass |

fp32 is in there because bf16 never takes the TF32 path at all, so a bf16-only
scout could not distinguish "`tf32_off` does nothing here" from "`tf32_off` is
inapplicable here". MoE is in there because top-1 routing is the most
plausible in-tree scatter/atomic nondeterminism. `checksum_mode: global` is in
there because it is the only mode that additionally compares an all-reduced
fingerprint, i.e. the only one that can see collective-ordering drift.

Scout plus archive run is **51 replay comparisons** (7×3 scouting, 10 in the
agent tree, 20 in the sweep tree), each OR'd across 8 ranks — 408 rank-local
compares. Zero divergence anywhere. By the rule of three, 51 clean independent
observations put a 95% upper bound of roughly **6% on the per-replay
divergence rate**; that is an upper bound, not a demonstration of determinism,
and it is nowhere near tight enough to certify a stack.

**Deterministic or rate-based is not a question this run gets to answer**,
because there is no divergence to characterise. If one shows up later, the
distinction matters enormously and `trials × steps` is the cheap way to buy
samples: a step is a self-contained snapshot→r1→restore→r2→compare, so it adds
comparisons without paying FSDP/RCCL startup again.

Two reasons the workload is a hard place to find nondeterminism, both by
design: `enable_deterministic()` sets
`torch.use_deterministic_algorithms(True, warn_only=True)` and pins
`CUBLAS_WORKSPACE_CONFIG=:4096:8` at setup, and the replay is two runs of one
step in one process with no optimizer step between them.

## The detector was validated, so the zero means something

A clean run proves nothing if the detector never fires.
`docs/llm-determinism.md` ships a perturbation for exactly this — bump one
input token between r1 and r2, which snapshot/restore does not undo — and it
was run through the same probe path (`harness_validation/`, ticket
`HARNESS-VALIDATION-INJECTED`):

| Cell | Trials | Verdict | Detectors |
|---|---|---|---|
| `none-none` | 2 | **fail** | `tier1:exit_nonzero`, `tier4:python_traceback` |
| `tf32_off-none` | 2 | **fail** | `tier1:exit_nonzero`, `tier4:python_traceback` |

Outcome `exhausted_candidates` after two proposer steps.

**This is not a scenario and must never be ingested as one.** Its verdict is a
property of the injection, not of the hardware or the stack. It is recorded
here because it converts "we saw no divergence" from an unfalsifiable claim
into a measured one: a real divergence on this node does reach the probe's
Tier-1 exit-code detector, does become a `fail` verdict, and does drive the
agent loop. It is excluded from the corpus, and its trajectory is written to a
separate file.

## Corpus

`examples/rl/build_corpus.py` (branch `feat/corpus-probe-result-artifacts`,
`e2b7489`) over `sweep_results/` only:

```
scenarios 2 · examples 12 (2 triage + 10 proposal)
artifacts {"probe_result": 2} · probe_trials 4
verdicts  {"pass": 2} · workload_families {"llm_determinism_replay": 2}
ground_truth_disagreements []
```

`--families` was needed: a probe cell name encodes the mitigation and
diagnostic axes and says nothing about the workload, so the family had to be
supplied by the run that knows it.

Only the sweep tree was ingested. The agent tree's cell is byte-for-byte the
same configuration and carries the same `cell_name`, so ingesting both would
mean two rows with the same `example_id` — one scenario counted twice.

**What these rows are worth is limited and worth saying plainly.** Both carry
`verdict: pass`, so a fix-half reward scored over them is a constant. They
prove the *pipeline* end-to-end on real GPU artifacts — probe → archived
`trial_*/result.json` → corpus row — which was previously demonstrated only on
a synthetic bash exit code. They are not training signal.

## Trajectories

`examples/rl/harvest_trajectories.py` reshapes `agent_log.jsonl` into rows
carrying the decision steps, the mitigations tried, the terminal outcome and
the final per-cell verdicts.

| Tree | Outcome | `llm_step`s | Trainable |
|---|---|---|---|
| `agent_results/` | `baseline_pass` | **0** | no |
| `harness_validation/` (injected) | `exhausted_candidates` | 2 | yes, but excluded |

**A clean control yields no trajectory at all**, and this is structural rather
than incidental: the baseline-pass short-circuit at `loop.py:399` returns
before the proposer is consulted, so the log holds exactly `session_start` and
`baseline_pass`. Trajectory harvesting therefore does *not* come free with any
agent run — it comes free with an agent run whose **baseline fails**. Rows
carry `trainable` and a reason so a zero-step row cannot be silently counted
alongside a real search.

### aorta#449 did not bite, and could not have

[aorta#449](https://github.com/ROCm/aorta/issues/449): `LiteLLMProposer`
filters unregistered mitigation names out of `next_mitigations` *before*
`validate_step` sees them, so a hallucinated name leaves an empty list;
`loop.py:451` treats an empty list as a stop regardless of `step.stop`, and
`_resolve_stop_outcome` falls through to `agent_requested`. The result is
`outcome=agent_stop` for what was really a name-resolution failure.

Neither trajectory is affected, and the reason is not that the loop behaved:
**both runs used `--llm-backend fake`, and `FakeLLMProposer` selects from the
candidate set, so it cannot emit an unregistered name and the filter has
nothing to drop.** The issue says so itself. A clean fake-backend run is not
evidence that a litellm/openai/vllm-backed run would be clean, and the
harvester records `backend_can_reach_449` per row so the distinction survives
into the data.

The check separates what the log can and cannot resolve:

* **proven** — an `llm_step` with `stop: false` and empty `next_mitigations`
  followed by `agent_stop`. The log contradicts itself; this is #449's own
  case B.
* **indeterminate** — `stop: true` with no `stop_reason`, then `agent_stop`.
  #449's case D shows a dropped name and a genuine conclusion agreeing on
  every recorded field, so these rows are flagged, never cleared.

Verified against synthetic logs in both shapes: case B classifies as proven,
case D as indeterminate.

## What this does and does not prove

**Does:**

* A real multi-GPU workload can be driven end-to-end through `aorta agent
  mitigate` as a probe-mode opaque command, on Slurm, in a container, with
  mitigation env vars provably crossing the container boundary.
* The archived `trial_*/result.json` cells that come out are ingestible by
  `build_corpus.py` — the first time on real GPU artifacts rather than a
  synthetic bash exit code.
* `agent_log.jsonl` harvests into trajectory rows, and the #449 check
  discriminates the issue's own reproduction shapes.
* A real divergence on this node reaches the probe's Tier-1 detector and
  drives the agent loop (from the injected validation).

**Does not:**

* Does not provide a resolve signal. Nothing failed, so no mitigation resolved
  anything, and a fix-half reward scored over these rows is a constant — the
  same degeneracy in a new place.
* Does not show the stack is deterministic. It bounds the per-replay
  divergence rate at roughly 6% (95%, rule of three over 51 comparisons) for
  *these* configurations on *this* node. Nothing more.
* Does not exercise the agent's search. The passing baseline short-circuits
  it, so category, hypothesis and mitigation selection went unmeasured on the
  clean path.
* Does not clear #449 for a real LLM backend.
* Does not say anything about the two use cases that are actually blocked:
  the corpus still contains no nondeterminism scenario and no NaN/numerics
  scenario, and this run did not create one.

## What a divergent configuration would need

The gap is not tooling. It is a public, committed reproducer that genuinely
fails, with a registered mitigation that genuinely fixes it. Candidates, in
rough order of cost:

1. **A numerics reproducer.** TF32 reduction ordering is the known axis and
   `tf32_off` is already a registered mitigation, so the fix half would be
   real the moment a configuration NaNs or diverges on a public image. This
   needs a configuration that actually misbehaves with synthetic inputs — the
   open ask.
2. **Widen the replay past one step in one process.** The current shape rules
   out most nondeterminism by construction. Divergence across *processes*, or
   with `use_deterministic_algorithms` off, is a different and much weaker
   determinism claim — and a legitimate one to probe, since production
   training does not run under it.
3. **A collective-ordering reproducer at scale.** `checksum_mode: global` can
   see it and saw nothing on one node; the AINIC/RCCL SDC reproducers in
   `recipes/race/` target this but need hardware this cluster does not have.

What must not happen is the shortcut: injecting a fault and archiving the
result as a scenario. It would produce a corpus row that looks exactly like
the real thing and trains a reward on a bash-level artifact of the harness.
That is the specific failure this exercise exists to avoid, which is why the
injected run is quarantined in its own tree with its own trajectory file.
