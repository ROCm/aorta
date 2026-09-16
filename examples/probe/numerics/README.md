# A public NaN scenario with a real resolver

Findings, measurements and the honest caveats: `docs/probe/numerics-nan-reproducer.md`.
Read that first — in particular section 1, which is why this directory does not
contain the TF32 scenario it was originally meant to contain.

## What is here

| file | what it is |
|---|---|
| `nan_workspace.py` | the workload. An unscaled fp16 overflow, then a `torch.empty` workspace with an off-by-one that reads the freed block back. `NAN_WS_MODE=overflow_only` runs phase 1 alone. |
| `launch_nan_ws.sh` | the opaque command a probe cell wraps. The `--env-file "$AORTA_ENV_FILE"` line is what carries the mitigation across the container edge. |
| `run_nan_probe.sbatch` | the archiving run: `aorta agent mitigate`, then the full matrix, then the companion scenario. |
| `rate_nan_ws.sbatch` | deterministic-or-rate-based characterisation, both sides of the mitigation. |
| `scout_numerics.sbatch`, `scout/numerics_probe.py` | round 1 — is `DISABLE_TF32` wired at all, and is there a TF32 precision axis to mitigate. |
| `scout_numerics2.sbatch`, `scout/numerics_probe2.py` | round 2 — three real non-finite mechanisms against 21 of the 22 registered mitigations. |
| `run_nan_rl.sbatch` | corpus, rollouts and the fix-half rescore against the real resolver. |
| `scenario_labels.json` | honest categories, which are all `unknown`, and why. |

## Recipes

- `recipes/probe/probe-numerics-uninit-workspace.yaml` — 8 cells, baseline plus
  six decoys plus the resolver, 4 trials each.
- `recipes/probe/probe-numerics-fp16-overflow.yaml` — the companion nothing
  resolves, for `fix_reward`'s withhold path.

## The two things a reader should not get wrong

**The reproducer is authored, and one step inside it pins a lottery.** Both
mechanisms are real, but their composition is not something found in the wild,
and `_plant_residue` deliberately arranges *which* freed block gets recycled.
Without that arrangement the scenario passed 24 times out of 24. The module
docstring says this too; it is repeated here because a directory listing is
where someone will meet the scenario first.

**A NaN symptom is not a numerics root cause.** The resolving mitigation is an
allocator flag, and every numeric lever in the registry was measured not to
move this. `scenario_labels.json` refuses `numeric_instability` for that
reason.

## Running it

Slurm `meta64` only — the MI350 CI runner is under a perf-baseline window and
none of these jobs may touch it. Each sbatch re-checks its own allocation and
exits 90 on an `smci350` node.

```
export TMPDIR=/apps/vikhande/tmp HF_HOME=/apps/vikhande/cache/hf PIP_CACHE_DIR=/apps/vikhande/cache/pip
sbatch examples/probe/numerics/scout_numerics2.sbatch   # is there a lever at all
sbatch examples/probe/numerics/rate_nan_ws.sbatch       # deterministic?
sbatch examples/probe/numerics/run_nan_probe.sbatch     # archive the matrix
sbatch examples/probe/numerics/run_nan_rl.sbatch        # corpus + rollouts + rescore
```

The scripts read the workload from a host path (`/apps/vikhande/probe-gpu`)
rather than the worktree: the compute nodes' Docker daemon cannot mount the
autofs NFS home, so the tree is rsynced to the scratch mount and mounted from
there. The worktree stays the source of truth; the copy is execution-only.
