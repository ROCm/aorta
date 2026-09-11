# A public NaN scenario with a real resolver, and why it is not a TF32 one

Measured 2026-09-11 on `meta64` (Slurm), nodes `cv350-rck-g03-c07-18` and
`-c06-08`, 8x MI350X `gfx950:sramecc+:xnack-`, container `rocm/primus:v26.3`
(torch `2.10.0+git94c6e04`, HIP `7.2.53211`), host ROCm 7.0.2.2. No job touched
the CI runner `smci350-rck-g03-f16-12`; every submission re-checked its
allocation for `smci350` and would have exited 90.

The goal was a public reproducer that genuinely goes NaN, resolved by a
registered mitigation, on the premise that **TF32 reduction ordering is the
axis and `tf32_off` is the mitigation**. The first half of that premise does
not hold on this stack, and the measurement that kills it is the most
important thing in this document. The second half of the goal was reached by a
different route, and that scenario is archived.

## 1. `DISABLE_TF32` is not read by anything here, and neither is TF32

`registry/mitigations.py` registers `tf32_off` as `DISABLE_TF32=1` with the
comment *"consumed by hipBLASLt itself"*. Against a float64 reference, a
4096x4096 fp32 matmul:

| condition | `fp32_precision` readback | Frobenius rel. error |
|---|---|---|
| `allow_tf32=False` | `ieee` | 1.15e-06 |
| `allow_tf32=True` | `tf32` | 4.49e-06 |
| `allow_tf32=True`, **`DISABLE_TF32=1`** | `tf32` | **4.49e-06** |
| `allow_tf32=True`, `HIPBLASLT_ALLOW_TF32=0` | `tf32` | 4.49e-06 |
| `allow_tf32=True`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` | `tf32` | 4.44e-06 |
| `allow_tf32=False`, `DISABLE_TF32=1` | `ieee` | 1.15e-06 |

`DISABLE_TF32` moves nothing. Neither does the PyTorch-ROCm variable, included
as a control precisely so "we set the wrong variable" could be ruled out. Only
the in-process `torch.backends.cuda.matmul` knob changes anything at all.

**And the knob is barely a precision axis either.** Calibrating the same
measurement against references computed with the mantissa actually truncated:

| path | Frobenius rel. error |
|---|---|
| `ieee` | 8.1e-07 |
| `tf32` (knob says it is on) | 4.4e-06 |
| software 10-bit-mantissa truncation, i.e. what TF32 *means* | **7.7e-04** |
| bf16 cast | 2.9e-03 |

A real 10-bit mantissa costs three orders of magnitude. The `tf32` setting
costs a factor of five. Whatever gfx950 does when asked for `tf32` in this
build, it is not a 10-bit mantissa, so there is **no TF32 precision axis to
mitigate** on this hardware and stack — with or without the env var.

Two consequences, both worth acting on separately from this document:

- **`tf32_off` is misfiled by aorta's own criterion.** The comment at the top
  of `BUILTIN_MITIGATIONS` says env vars that only work because a training
  script reads them belong in a plugin via the `aorta.mitigations`
  entry-point group, not in the builtins. `tf32_off` is in the builtins on the
  strength of a claim about hipBLASLt that does not hold here. Any probe cell
  labelled `tf32_off` on this stack is a second baseline under a different
  name, which is the failure mode the whole exercise is trying to avoid.
- The transferable insight from the private customer work may still be true
  *there*. It is not reproducible *here*, and no amount of searching for the
  right reproducer would have changed that, because the knob is upstream of
  the workload.

## 2. Then does the registry contain any live lever over a NaN?

Three genuine non-finite mechanisms, each run against 21 of the 22 registered
mitigations (`amd_log_level_4` omitted: it changes log volume only, and costs
minutes of output). All three go non-finite unmitigated.

| mechanism | what it is | mitigations that resolve it |
|---|---|---|
| `fp16_overflow` | unscaled fp16 forward, activations pass 65504 at layer 11 | **0 of 21** |
| `onepass_var` | `E[x^2]-E[x]^2` on near-constant data, reductions via GEMM; `rsqrt` of a negative variance | **0 of 21** |
| `uninit_read` | a `torch.empty` buffer nobody wrote, landing on a freed non-finite block | **1 of 21** — `pytorch_no_cuda_memory_caching` |

That is the whole answer to "is there a numerics mitigation here": one lever,
and it is an allocator lever rather than a numerics one. `onepass_var` is worth
a second look because it is the closest thing to the intended scenario — a
precision-sensitive cancellation — and precision does not decide it: it goes
non-finite at `mu/sigma >= 1e4` identically with the TF32 knob on and off,
which is what section 1 predicts.

## 3. The scenario that is archived

`recipes/probe/probe-numerics-uninit-workspace.yaml`, workload
`examples/probe/numerics/nan_workspace.py`. Two phases:

1. An unscaled fp16 stack with a hot init overflows. Peaks per layer, measured:
   `14.7, 31.7, 84.6, 204, 509, 1272, 3452, 8736, 19424, 52576, inf`. Nothing
   raises — a silent fp16 overflow is exactly why loss scaling exists.
2. A later step preallocates a microbatch-accumulation workspace with
   `torch.empty` and an off-by-one leaves the last of eight slots unwritten.
   The caching allocator hands back the phase-1 block, so 3266 elements of the
   workspace are the overflowed activation and the loss is NaN — reported as
   `loss=nan`, which the built-in `tier4:nan_signature` detector matches.

**What is authored, stated plainly.** Neither half is injected: phase 1 is how
fp16 training dies, and the uninitialised workspace is common enough to be a
genre. What is authored is their composition, and one step inside it. In the
wild, which freed block a `torch.empty` lands on is a lottery, and that lottery
was measured here: with the phase-1 weights still cached, the request is served
from a coalesced ex-weight region whose fp16 bit patterns read as fp32
denormals, the workspace looks clean, and the reproducer silently passes — 24
times out of 24 across two attempts before the pinning existed. `_plant_residue`
pins it by parking the overflowed tensor on the host, emptying the device
allocator, and materialising the residue into an empty pool. Pinning a lottery
is most of what makes something a reproducer; doing it quietly would not be.

**Rate: deterministic, not rate-based.** Baseline 0 pass / 3 fail, resolver
3 pass / 0 fail on the characterisation job, and every trial reported the same
`nonfinite_workspace_elems: 3266` and the same resolver-side loss to all
digits. Combined with the archived matrix (below) the counts are large enough
that the rule-of-three reasoning that governed the previous run does not need
to be invoked: this is not a rare event being sampled, it is a fixed one.

**What it teaches, and it is true:** a NaN that disappears when you disable the
caching allocator is not a numerics bug, it is a read of memory nobody wrote.

`recipes/probe/probe-numerics-fp16-overflow.yaml` archives the companion:
phase 1 alone, a real NaN that **nothing in the registry resolves**. That is
the withhold path in `fix_reward` exercised against a real scenario rather than
assumed.

### The archived matrix

Slurm job **35724**, partition `meta64`, node `cv350-rck-g03-c17-08` — no
`smci350` in the allocation. Artifacts under
`/apps/vikhande/probe-gpu/{nan_agent,nan_matrix,nan_fp16}`.

`PROBE-NAN-WS-MATRIX`, 8 cells x 4 trials. Every cell's `env` is the value the
trial actually recorded, read back out of `result.json` rather than assumed:

| cell | recorded `env` | verdict | detectors |
|---|---|---|---|
| `none-none` | `{}` | fail 4/4 | `tier1:exit_nonzero`, `tier4:nan_signature` |
| `tf32_off-none` | `DISABLE_TF32=1` | fail 4/4 | both |
| `hsa_no_sdma-none` | `HSA_ENABLE_SDMA=0` | fail 4/4 | both |
| `hip_launch_blocking-none` | `HIP_LAUNCH_BLOCKING=1` | fail 4/4 | both |
| `xnack-none` | `HSA_XNACK=1` | fail 4/4 | both |
| `pytorch_alloc_expandable_segments-none` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | fail 4/4 | both |
| `gpu_max_hw_queues_2-none` | `GPU_MAX_HW_QUEUES=2` | fail 4/4 | both |
| **`pytorch_no_cuda_memory_caching-none`** | `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | **pass 4/4** | none |

`tier4:nan_signature` fired on every failing trial, so the NaN is detected as a
NaN and not merely as a non-zero exit.

`fix_reward.resolution_from_matrix` reads this back as
`cells=8, baseline_failed=True, resolvers=['pytorch_no_cuda_memory_caching']`
— the fix half's ground truth recovered from an archived run, which is the
input it has never had.

`PROBE-NAN-FP16`, 4 cells x 2 trials: all four fail, including the resolver of
the other scenario. Correctly withheld.

**Trial counts backing the rate claim:** 28 baseline-or-decoy failing trials
and 4 resolver passing trials in this matrix, plus 6 agent-loop cells x 4, plus
3+3 in characterisation, plus one each in the 21-mitigation sweep. No trial
ever disagreed with its cell.

### The agent loop closed on it

`aorta agent mitigate` with the `fake` backend walked the axis and converged:
six `llm_step` events, six `mitigation_tried`, and
`{"type": "converged", "winning_mitigation": "pytorch_no_cuda_memory_caching"}`.
That is the first trajectory from this project with a real search in it — the
previous run's baseline passed, so the loop short-circuited and recorded zero
decision steps. `harvest_trajectories.py` reports 1 row, trainable, 6 steps,
7 cells.

**aorta#449 did not bite, and could not have.** The backend was `fake`, and
`FakeLLMProposer` picks from the registered candidate list, so it cannot emit
the unregistered name that triggers the issue. The row carries
`backend_can_reach_449: false`; treat the clean result as inapplicable rather
than as evidence.

## 4. The fix half, scored against a real resolver

Slurm job **35729**, node `cv350-rck-g03-c16-18`. Qwen3-8B served through
`serve_for_rollouts.sh`, 8 samples per scenario at t=0.7, `--full-candidates`
so the whole 20-name registry is offered and naming the resolver is a choice
rather than a shortlist. 64 recorded proposals, all 8 groups with non-zero
spread.

`a247b27` could only report this term under *hypothesised* resolution sets.
This is it against the real one, `resolvers=['pytorch_no_cuda_memory_caching']`
recovered from the archived matrix by `resolution_from_matrix`.

| policy | form half only | with the fix half (weight 0.5) |
|---|---|---|
| `oracle_contract_perfect` | 1.0000 | 1.0000 |
| `honest_abstainer` | 0.9000 | 0.4500 |
| `abstain_and_pick_first` | 0.9000 | 0.4500 |
| **Qwen3-8B** | **0.7569** | **0.5347** |
| `abstain_and_shotgun` (names all 20) | 0.5400 | **0.7700** |
| `always_prose` | 0.0000 | 0.0000 |

Criteria: **2 of 4 pass** in both regimes — 2 (reference at top) and 3
(within-group spread), with 1 and 4 failing on both.

**The fix half does what it was built to do, and then loses to a different
constant.** It kills the humble-constant degeneracy exactly as intended:
`abstain_and_pick_first`, the two-line policy that beat the model 0.9000 to
0.8689 in the first end-to-end run and 0.9000 to 0.7569 here, drops to 0.4500
because its one name is almost never a resolver. But naming *all twenty*
guarantees the resolver is in the set, so shotgun collects full fix credit on
every sample and rises from 0.5400 to 0.7700 — past the model. The winning
strategy that reads nothing has changed identity, not disappeared.

That is a direct measurement against an argument `fix_reward.py` makes in its
own docstring: breadth is not priced in the fix half because
`proposal_reward.precision_credit` already scales the tier 4–5 block by
`min(1, 2/names)`. At `FIX_WEIGHT = 0.5` that scaling does not offset a
guaranteed fix credit. The argument was reasonable and is now falsified; the
weight and the no-length-scaling choice were both fixed before anything was
measured, which is why this is evidence rather than an embarrassment.

**The model does carry real signal on this term.** Its fix rate is 0.3125 —
the share of samples naming a resolver — against a chance rate of about 0.164
for uniformly picking its observed mean of 3.27 names out of 20. Roughly twice
chance, and not close to enough.

**One caveat that matters for how much weight to put on this.** The eight
"scenarios" are eight cells of one reproducer, so they share a single ground
truth and are not eight independent problems. The effective sample is one
scenario with 64 completions, not eight. `pytorch_no_cuda_memory_caching-none`
also scores highest (0.8254) simply because its evidence is the cell that
passed. A second, unrelated resolving scenario would be worth more here than
more samples on this one.

## 5. What this proves, and what it does not

Proven:

- A public, committed, synthetic-input reproducer that genuinely goes NaN on a
  public image, where a **registered** mitigation genuinely resolves it, with
  20 registered decoys measured not to. The fix half now has a real resolver.
- The fix half removes the humble-constant degeneracy and introduces a
  shotgun one. Measured, on real rollouts, against real ground truth.
- `DISABLE_TF32` is inert on ROCm 7.2 / torch 2.10 / gfx950, and the `tf32`
  precision setting is not a 10-bit mantissa there either.
- The registry has exactly one live lever over a non-finite outcome, and it is
  the allocator.

Not proven, and not claimed:

- **This is not the `numeric_instability` scenario the corpus is missing.** The
  symptom is a NaN; the root cause is uninitialised memory. Labelling it
  `numeric_instability` would teach the model something false, and the label
  file says so.
- Nothing here reproduces the customer TF32 finding, or shows it is wrong. It
  shows the *stack* offers no handle on it.
- The composition is authored. Its two halves are real; a reader who wants a
  discovered-in-the-wild scenario does not have one yet.
- Nothing here shows the fix half is the wrong term. It shows that at
  `FIX_WEIGHT = 0.5` with no length scaling it is beaten by shotgunning, on
  one scenario.

## 6. What to do next, in order

1. **Price breadth on the fix side, or raise the form-side precision penalty,
   and re-measure.** This is one line and the measurement to justify it now
   exists. The honest version is to pick the change on the same
   "before anything was measured" principle the original weight was picked on,
   rather than fitting it until criterion 1 flips — a number tuned into
   passing would be worth nothing.
2. **A second, unrelated resolving scenario.** The effective sample here is
   one. Two scenarios with different resolvers would separate "the model knows
   allocator flags fix stale-memory NaNs" from "the model got one right".
3. **Give `tf32_off` a body or move it.** It is registered on a claim that does
   not hold on ROCm 7.2 / gfx950, and a mitigation that sets an unread variable
   is indistinguishable from a second baseline in every matrix it appears in.
   Either move it to a workload plugin per the registry's own criterion, or
   have it set something the stack reads.
4. **A category for uninitialised-memory reuse.** Neither the 8-name set nor
   #484's 11-name set has one, and this scenario has to be labelled `unknown`
   as a result. See `examples/probe/numerics/scenario_labels.json`.
5. **Re-run with a real LLM backend on the agent loop**, not just on rollouts,
   so aorta#449 becomes reachable. Everything here used `fake` for the loop,
   which is structurally immune.
