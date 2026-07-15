# LLM Determinism Recipe

Catch kernel-level nondeterminism / silent data corruption (SDC) in a
transformer training step. Runs the **same** forward+backward twice on
the same inputs, with parameters and RNG restored between runs, and
compares bit-exact checksums of every per-block boundary activation,
the loss, the logits, every grad, and every param.

If any checksum drifts → some kernel (matmul, attention, reduction,
collective) is producing different bits for the same inputs.

---

## Quick Start

### Prerequisites

- A node with ≥1 AMD GPU (smoke verified on 8× MI350X, gfx950).
- PyTorch built with ROCm/HIP (`torch.version.hip` non-empty,
  `torch.cuda.is_available()` true). Verified on
  `torch 2.10.0.dev+rocm7.0`.
- `pip install -e .` from the repo root.

### Smoke (≈5 s on 8 GPUs)

Create `/tmp/run_llm_det.py`:

```python
import os, sys
from aorta.workloads.llm_determinism import LlmDeterminismWorkload

cfg = {
    "num_layers": 24, "hidden_size": 2048, "ffn_size": 5632,
    "num_heads": 16, "seq_len": 512, "batch_size": 1,
    "dtype": "bf16", "seed": 1234, "steps": 1,
    "checksum_mode": "per_rank",
    "capture_dir": "/tmp/llm_det_capture",
}
w = LlmDeterminismWorkload(cfg)
w.setup(); r = w.run(); w.cleanup()
print(f"[rank {os.environ.get('RANK')}] passed={r.passed} "
      f"failures={r.failure_count} elapsed={r.elapsed_sec:.2f}s "
      f"ranks_with_divergence={r.metrics['ranks_with_divergence']}")
sys.exit(0 if r.passed else 1)
```

Run it:

```bash
rm -rf /tmp/llm_det_capture
torchrun --standalone --nproc_per_node=8 /tmp/run_llm_det.py
```

**Expected on a healthy stack:**
- Exit 0; every rank prints `passed=True failures=0 ranks_with_divergence=0`
- `/tmp/llm_det_capture/rank000.jsonl` … `rank007.jsonl` exist, 2 lines each (`run=r1`, `run=r2`)
- `loss_bits` is non-zero and **equal between r1 and r2 on each rank** (with the defaults this lands around ±1.1e9 — the loss is a single fp32 scalar viewed bit-for-bit as int32, so the value is an int near `loss.item()`'s fp32 bit pattern, not a numeric magnitude)
- `block_pre` / `block_post` lists are length `num_layers`

### Validate the detector actually flags real divergence

A green smoke alone doesn't prove the detector works in your environment.
Inject a known difference between r1 and r2 by perturbing `_input_ids`
(not snapshotted/restored, so survives only into r2):

Save as `/tmp/run_llm_det_fail.py`:

```python
import os, sys, torch
from aorta.workloads.llm_determinism import LlmDeterminismWorkload

cfg = {"num_layers": 24, "hidden_size": 2048, "ffn_size": 5632,
       "num_heads": 16, "seq_len": 512, "batch_size": 1,
       "dtype": "bf16", "seed": 1234, "steps": 1,
       "checksum_mode": "per_rank",
       "capture_dir": "/tmp/llm_det_capture_fail"}
w = LlmDeterminismWorkload(cfg); w.setup()
_orig, n = w._run_once, [0]
def patched():
    n[0] += 1
    if n[0] == 2:                              # perturb after r1, before r2
        w._input_ids[0, 0] = (int(w._input_ids[0, 0]) + 1) % w._cfg.vocab_size
    return _orig()
w._run_once = patched
r = w.run(); w.cleanup()
print(f"[rank {os.environ.get('RANK')}] passed={r.passed} "
      f"failures={r.failure_count}")
for reason in r.metrics.get("divergence_reasons", [])[:5]:
    print(f"[rank {os.environ.get('RANK')}] {reason}")
sys.exit(0 if r.passed else 1)
```

```bash
rm -rf /tmp/llm_det_capture_fail
torchrun --standalone --nproc_per_node=8 /tmp/run_llm_det_fail.py
echo "exit=$?"
```

**Expected:** non-zero exit; every rank prints `passed=False`; `divergence_reasons` lists `block[0..N].pre/post`, `loss_bits`, `output_bits`, and many `grad[*]` entries. Params should **not** appear in reasons (snapshot/restore unchanged).

### Reading the capture

```bash
# r1 vs r2 equality, per rank — expect "true" on every line.
for f in /tmp/llm_det_capture/rank*.jsonl; do
  jq -s --arg f "$f" '$f + ": " + (.[0].block_post == .[1].block_post | tostring)' "$f"
done

# loss_bits across ranks (sanity-check non-zero, near each other):
jq -c '{rank, run, loss_bits}' /tmp/llm_det_capture/rank*.jsonl

# First divergent block index on rank 0 (after a failing run):
jq -s '[.[0].block_post, .[1].block_post] | transpose
        | map(.[0]==.[1]) | index(false)' \
   /tmp/llm_det_capture_fail/rank000.jsonl
```

### Running via recipe (multi-cell sweep)

The launcher script above runs one configuration. To sweep multiple
configurations in one invocation, use the recipe at
[`recipes/llm-determinism/example-llm-determinism.yaml`](../recipes/llm-determinism/example-llm-determinism.yaml).
Pass workload knobs through `workload_config`:

```yaml
# Recipe-scope defaults applied to every cell:
workload_config:
  hidden_size: 2048
  dtype: bf16
  checksum_mode: per_rank

cells:
  - name: baseline-bf16-24L
    mitigations: [none]
    environment: local
    workload_config:       # cell wins on key collision
      num_layers: 24
      capture_dir: ./llm_det_capture/baseline-bf16-24L
  - name: moe4-bf16-12L
    mitigations: [none]
    environment: local
    workload_config:
      num_layers: 12
      num_experts: 4
      capture_dir: ./llm_det_capture/moe4-bf16-12L
```

Launch (use the `aorta` console script — `python -m aorta.triage.cli` is **not** a runnable module):

```bash
torchrun --standalone --nproc_per_node=8 $(which aorta) triage run \
  --recipe recipes/llm-determinism/example-llm-determinism.yaml
```

Two schema gotchas the loader enforces:
- `steps` is a first-class recipe field; putting it in `workload_config` is rejected (would be silently clobbered).
- Keys starting with `_aorta_` are reserved and rejected.

**Per-cell step times include warmup.** Cells run sequentially in one
process; the first cell absorbs FSDP2 / RCCL / hipBLAS warmup, so its
step time is typically much higher than later cells (in one 4-cell
smoke we saw 2453 ms on the first cell vs 78–912 ms on later ones).
That is not a determinism signal — the per-rank checksum compare is.

**Where the checksum content lives.** The triage `matrix.md` table only
reports pass/fail per cell (failure rate, failures count, mean step time,
confound tag). It does **not** include checksum bits or
`divergence_reasons`. To inspect the actual checksum stream:

```bash
# Per-trial WorkloadResult (loss_bits per step, ranks_with_divergence,
# divergence_reasons when failing, model_label, etc.):
find triage_results -name result.json -exec \
  jq '{passed, rwd: .metrics.ranks_with_divergence,
       loss_r1: .metrics.steps[0].loss_bits_r1,
       reasons: .metrics.divergence_reasons}' {} \;

# Per-block / per-rank fingerprints (the big artifact for offline
# analysis) live wherever each cell's `workload_config.capture_dir`
# pointed — NOT inside the triage output tree.
```

### If you see divergence — what to send back

Tar up:
- `/tmp/llm_det_capture/` (or your `capture_dir`) — every `rank*.jsonl`
- Full torchrun stdout/stderr (all `divergence_reasons` lines)
- `aorta env probe -o env.json` output (PyTorch / ROCm / HIP / RCCL versions, env vars, GPU arch)
- The exact launcher script you used, the GPU count, and which knobs you changed from defaults

### Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `RuntimeError: Expected all tensors to be on the same device` | Stale checkout — `git pull` `main` and reinstall (`pip install -e .`). |
| Every rank hangs at `init_process_group` | Set `NCCL_DEBUG=INFO`; try `NCCL_P2P_DISABLE=1`. RCCL honours `NCCL_*` env vars. |
| `loss_bits` reported as 0 | Stale checkout — `git pull` `main` and reinstall; labels are shifted by one (`torch.roll`) to avoid the degenerate loss-collapse. |
| OOM | Drop `num_layers: 12` first, then `hidden_size: 1024, ffn_size: 2816`, then `seq_len: 256`. One knob at a time. |
| Determinism flagged but suspect false positive | Bump `checksum_mode: global` to also compare an all-reduced loss⊕output fingerprint; rerun. |

---

## Knobs

| Key | Default | Notes |
|---|---|---|
| `num_layers` | 24 | Block repetitions = comm/compute stages. Scale down with rank count. |
| `hidden_size` | 2048 | Per-block hidden width. |
| `ffn_size` | 5632 | GLU FFN width. |
| `num_heads` | 16 | Must divide `hidden_size`. |
| `vocab_size` | 32000 | Random embedding table. |
| `seq_len` | 512 | Per-rank batch sequence length. |
| `batch_size` | 1 | Per-rank batch. |
| `num_experts` | 1 | 1 = dense FFN; ≥2 = top-1 MoE router across N experts. |
| `dtype` | `bf16` | `bf16` / `fp16` / `fp32`. |
| `seed` | 1234 | Synthetic batch + RNG seed. |
| `steps` | 1 | Number of replay steps. Each step is a fresh snapshot → r1 → restore → r2 → compare. There is no optimizer step between steps, so on a healthy stack every step's checksums are bit-identical to step 0's — `steps > 1` is useful as a flake-detector / soak, not as training progress. |
| `checksum_mode` | `per_rank` | `global` also all-reduces a fingerprint across ranks. |
| `capture_dir` | unset | When set, every step writes `rank<NNN>.jsonl` for offline inspection. |
| `model_label` | `generic-repeated-block` | Advisory only; recorded in metrics. |

## Capture-mode output

Each rank writes one JSON line per (run, step):

```json
{"rank": 7, "step": 0, "run": "r1", "loss_bits": 12345, "output_bits": 67890,
 "block_pre": [<int per block>], "block_post": [<int per block>]}
```

## Metrics surface (stable for detector parsing)

`WorkloadResult.metrics` includes:

- `ranks_with_divergence` — count of ranks whose local compare flagged
- `steps[*].loss_bits_r1`, `loss_bits_r2`, `num_blocks_checked`
- `divergence_reasons` — first 32 reasons (when present)
- `model_label`, `num_layers`, `num_experts`, `dtype`, `checksum_mode`, `capture_dir`

---

## Design notes

### Why a single repeated block

The execution model is a chain of repeating
`comm_kernel_i → compute_kernel_i` stages. One repetition of the same
transformer block is one such stage; under FSDP2 `fully_shard` the
block boundary is also the all-gather / reduce-scatter boundary, so
checksumming the activation at every block boundary checksums the
tensor that actually crosses GPUs at every repetition.

**This is not a faithful LLM.** Weights are random init, the architecture
(LayerNorm + MHA + GLU FFN, optional top-1 MoE) is generic on purpose.
The model and its size are knobs to control how much data crosses GPU
boundaries — nothing more.

### What it doesn't do

- Not a quality benchmark.
- Not a perf benchmark — manual attention, no `torch.compile`, no graphs.
- Not a multi-step trainer by default; `steps` is configurable but each
  step is a self-contained replay.
- Does not intercept kernels directly. Module forward pre/post hooks are
  the practical proxy for the comm boundary; documented limitation.

### Checksum contract

The checksum is a **bit-pattern** sum, not a numeric sum: tensor storage
is re-interpreted (`view`, not `to`) as the signed integer of matching
element size, then accumulated into `int64`.

- bf16, fp16 (2-byte) → `int16` view
- fp32 (4-byte) → `int32` view
- fp64 (8-byte) → `int64` view

Two tensors with identical bits → identical checksums. Two tensors that
differ in a single bit (including NaN payload bits and the +0 vs -0
distinction that a numeric sum would erase) → different checksums.
Accumulator wraps mod 2^64 — fine, because we only ever compare two
checksums for equality, never reason about magnitude.

Note for cross-tool comparison: these numbers are **not** directly
comparable to other tools' checksums unless they use the identical
view-dtype mapping, the same hook points, and the same reduction.

### Single-rank-only compare (default)

`checksum_mode: per_rank` (default) compares each rank's run-1 vs run-2
shards / block boundaries only. Pass/fail is OR'd across ranks via a
single all-reduced integer flag, no checksum values cross ranks.
`checksum_mode: global` opts in to additionally comparing an
all-reduced loss⊕output fingerprint — catches collective ordering
drift, but no longer satisfies the strict "single-rank only" guard.

### Determinism caveats

`torch.use_deterministic_algorithms(True, warn_only=True)` is enabled
at setup. Some ops have no deterministic implementation; we tolerate
that because the checksum compare is what actually proves replayability.
`CUBLAS_WORKSPACE_CONFIG=:4096:8` is set if not already present —
hipBLAS / cuBLAS require this for deterministic matmul.
