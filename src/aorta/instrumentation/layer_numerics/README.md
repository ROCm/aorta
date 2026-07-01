# NaN Logger v2 — Diagnostic Runs for ef[10] INPUT NaN

## Background

`emb_proj.projections.10.layers.0` (a `Linear(2048, 128)` layer) intermittently
receives an input tensor that already contains NaN values. The leading hypothesis
is cross-stream allocator reuse combined with the AMD `s_waitcnt` hardware defect:
the GPU reports a kernel as "complete" before its global memory writes have
actually landed, causing `record_stream`'s protection to be insufficient — the
allocator hands a freed buffer to this layer's input while a prior kernel's
writes to that buffer are still in-flight.

These diagnostic runs collect the evidence needed to:
1. Confirm ef[10] INPUT NaN under instrumentation
2. Capture the corrupted tensor's GPU memory address
3. Identify the **donor** (which other layer's freed buffer was reused)
4. Prove cross-stream allocator reuse via the allocator event trace

---

## What's Included

| File | Purpose |
|------|---------|
| `instrument_nan_logger.py` | The logger (single file, no extra dependencies beyond PyTorch) |
| `README.md` | This file — run instructions |
| `LOGGER_REFERENCE.md` | Full technical reference (all env vars, output format, analysis recipes) |

---

## Prerequisites

- **PyTorch** with ROCm (the logger uses standard PyTorch APIs)
- **torchrec** (if using the standalone reproducer with `TrainPipelineSparseDist`)
- **No pip install needed** — `instrument_nan_logger.py` has zero dependencies beyond PyTorch
- **Disk space:** ~500 MB per run (Run 1 with `SAMPLE_EVERY=1` and allocator snapshot)
- **GPU memory:** same as your normal training run (logger adds <10 MB host memory)
- **Run duration:** recommend **5000+ steps** per trial (NaN typically appears within
  100–2000 steps, but can be intermittent)

---

## Setup

### Step 1: Place the logger

Copy `instrument_nan_logger.py` to a directory accessible from the training host:

```bash
mkdir -p /opt/nan_logger_v2
cp instrument_nan_logger.py /opt/nan_logger_v2/
```

### Step 2: Choose invocation method

**Option A — wrapper invocation (preferred, tested):**

```bash
NANLOG_DIR=/output/nanlog \
NANLOG_CHANNELS=... \
  python3 /opt/nan_logger_v2/instrument_nan_logger.py /path/to/your_training_script.py [args...]
```

The wrapper arms the hooks, then runs your script via `runpy`. Hooks bind to the
model the moment `DistributedModelParallel` is constructed.

**Option B — import at script top (if you can't change the launch command):**

Add these 3 lines at the very top of your training script, **before** the model
is built:

```python
import sys
sys.path.insert(0, "/opt/nan_logger_v2")
import instrument_nan_logger  # noqa: F401
```

Then set `NANLOG_*` env vars in your shell as usual.

---

## Sanity Check (verify at startup — MUST pass before counting the run)

In the startup log (stderr), look for:

```
nanlog: attached layer hooks to N module(s) (types=[...], names=[...]); channels=[...]; capture_addr=True; ...
```

**Verification checklist:**

| Check | Expected |
|-------|----------|
| `N > 0` | Hooks attached. If N = 0, hooks didn't bind — see Troubleshooting. |
| N ≈ 156 | Full watch (Run 1). |
| N ≈ 99 | Linear-only (Run 2 fallback). |
| `capture_addr=True` | Address recording is on. |
| `locate=True` | Row-locate enabled. |
| `bad_values=True` | Bad-value position enabled. |
| `alloc_snapshot=True` | Allocator event trace on (Run 1). |

If N = 0, the model isn't going through `DistributedModelParallel`, or the
logger was imported after the model was built. See Troubleshooting below.

---

## Run 1: Full Diagnostic (primary)

**Purpose:** Capture the complete evidence chain — GPU memory address overlap,
corruption pattern, corrupted tensor dump, and allocator event trace — in a
single run.

```bash
NANLOG_DIR=/output/run1_full \
NANLOG_CHANNELS=act,input,igrad,weight,bias,wgrad,bgrad \
NANLOG_WATCH_TYPES=Linear,LinearProjection,AttentionBlock,InteractionLayer,MLP,EmbeddingGate \
NANLOG_ADDR=1 \
NANLOG_LOCATE=1 \
NANLOG_BAD_VALUES=1 \
NANLOG_DUMP_TENSOR=1 \
NANLOG_ALLOC_SNAPSHOT=1 \
NANLOG_PRE_CONTEXT=20 \
NANLOG_SAMPLE_EVERY=1 \
  python3 /opt/nan_logger_v2/instrument_nan_logger.py /path/to/your_training_script.py
```

**What each option does:**

| Option | Purpose |
|--------|---------|
| `CHANNELS=act,input,igrad,weight,bias,wgrad,bgrad` | Record all channels: forward output (`act`), forward **input**, backward grad (`igrad`), parameter values (`weight`,`bias`), and parameter gradients (`wgrad`,`bgrad`). The `input` channel is **required** to capture ef[10]'s input address. |
| `WATCH_TYPES=Linear,...` | Watch ~156 modules (all matmul-bearing layers). Needed to find the donor whose output buffer was reused as ef[10]'s input. |
| `ADDR=1` | Record `storage_ptr` + `storage_nbytes` for every watched tensor. These are absolute per-process GPU virtual addresses — directly comparable within one rank's JSONL. |
| `LOCATE=1` | Record `bad_rows` (how many dim-0 rows have NaN). `bad_rows==1` on `[1024,2048]` = one-tile late write (aliasing signature). |
| `BAD_VALUES=1` | Record exact position (`first_bad_row`, `first_bad_col`) and value of first corrupted element. |
| `DUMP_TENSOR=1` | Save the full corrupted tensor to disk (`.pt` file) on first NaN detection. One-shot (~8 MB). |
| `ALLOC_SNAPSHOT=1` | Enable PyTorch allocator event recording. On first NaN, dump full trace (pickle) with every alloc/free: address, size, **stream ID**, timestamp, call stack. |
| `PRE_CONTEXT=20` | Buffer last 20 steps in memory; dump all on first NaN (full-resolution address history leading up to the corruption). |
| `SAMPLE_EVERY=1` | Log every step (maximum resolution for address tracking). |

**Overhead:** ~10% per-step from allocator stack capture. GPU kernel timing and
launch order are NOT affected (the overhead is host-side bookkeeping only).

**Recommended:** Run **3 independent trials** (different output dirs:
`run1_full_t1`, `run1_full_t2`, `run1_full_t3`). The NaN is intermittent —
multiple trials increase capture probability and provide cross-validation.

**Step count:** 5000 steps minimum. If no NaN by 5000 steps, continue to 10000.

**Expected output:**
```
/output/run1_full/
  layers_rank0.jsonl                    # per-layer, per-step records with GPU addresses
  summary_rank0.json                    # first_bad fingerprint + run metadata
  alloc_snapshot_step*_rank0.pickle     # allocator event trace (only if NaN detected)
  bad_tensor_step*_rank0.pt            # full corrupted tensor (only if NaN detected)
```

**Success criteria:**
- `summary_rank0.json` → `first_bad` is NOT null
- `first_bad` layer contains `projections.10`
- `first_bad` role = `input`
- `alloc_snapshot_step*_rank0.pickle` exists

---

## Run 2: Lightweight Fallback (only if Run 1 does NOT reproduce)

**Purpose:** If Run 1's 10% allocator-snapshot overhead masks the bug (changes
timing enough to prevent the race), this lighter configuration preserves original
timing as closely as possible while still capturing addresses and NaN position.

```bash
NANLOG_DIR=/output/run2_light \
NANLOG_CHANNELS=act,input,igrad,weight,bias,wgrad,bgrad \
NANLOG_WATCH_TYPES=Linear \
NANLOG_ADDR=1 \
NANLOG_LOCATE=1 \
NANLOG_BAD_VALUES=1 \
NANLOG_DUMP_TENSOR=1 \
NANLOG_ALLOC_SNAPSHOT=0 \
NANLOG_PRE_CONTEXT=20 \
NANLOG_SAMPLE_EVERY=50 \
  python3 /opt/nan_logger_v2/instrument_nan_logger.py /path/to/your_training_script.py
```

**Differences from Run 1:**

| Setting | Run 1 | Run 2 | Why |
|---------|-------|-------|-----|
| `WATCH_TYPES` | 6 types (~156 modules) | `Linear` only (~99 modules) | Less perturbation |
| `ALLOC_SNAPSHOT` | 1 | **0** | No allocator overhead |
| `SAMPLE_EVERY` | 1 | **50** | Smaller JSONL, less host work |

**Overhead:** <1%. This is nearly invisible to GPU timing.

**Trade-off:** Without `ALLOC_SNAPSHOT`, we lose the definitive allocator event
trace (the `free→alloc` sequence with stream IDs). We still get addresses and
can identify the donor from address overlap, but cannot prove the cross-stream
mechanism from the allocator log alone.

---

## Summary: Test Matrix

| Run | When to use | Watch scope | Allocator snapshot | Sample rate | Overhead |
|-----|-------------|-------------|--------------------|-------------|----------|
| 1 | **Always start here** | Full (156 modules) | **On** | Every step | ~10% |
| 2 | Only if Run 1 doesn't reproduce | Linear only (99) | Off | 1-in-50 | <1% |

Recommend 3 trials per run, 5000 steps each.

---

## What to Send Back

For **each completed trial**, please send:

### Required:

1. **The entire `NANLOG_DIR` folder** — contains:
   - `layers_rank*.jsonl` (per-layer per-step records with GPU addresses)
   - `summary_rank*.json` (first_bad fingerprint + metadata)
   - `alloc_snapshot_step*_rank*.pickle` (Run 1 only, if NaN detected)
   - `bad_tensor_step*_rank*.pt` (if NaN detected)

2. **Console/stderr log** from the training run — should contain:
   - The `nanlog: attached layer hooks to N module(s) ...` startup line
   - The `nanlog: FIRST BAD: ...` line (if NaN was detected)

### Also helpful:

3. **`summary_rank0.json` pasted inline** (small file, <1 KB) — for quick triage

4. **Which run config** (Run 1 or Run 2, trial number)

5. **Number of steps completed** before stopping

### Data organization suggestion:

```
results/
  run1_full_t1/
    layers_rank0.jsonl
    summary_rank0.json
    alloc_snapshot_step84_rank0.pickle
    bad_tensor_step84_rank0.pt
    console.log
  run1_full_t2/
    ...
```

---

## How We'll Analyze the Data

### Step 1: Identify the victim

Find the first record where `projections.10.layers.0` with `role=="input"` has
`bad==true`. Verify `bad_rows == 1` (single-tile aliasing signature).

### Step 2: Find the donor (address overlap)

Take the victim's `[storage_ptr, storage_ptr + storage_nbytes)` range. Search
all other records at the same step and step-1 for any layer whose storage range
overlaps. A match names the donor — the layer whose freed buffer was recycled as
ef[10]'s input.

```python
import json
recs = [json.loads(l) for l in open("layers_rank0.jsonl")]
bad = next(r for r in recs if r["bad"] and "projections.10" in r["layer_name"]
           and r["role"] == "input")
lo = int(bad["storage_ptr"], 16); hi = lo + bad["storage_nbytes"]; S = bad["step"]

for r in recs:
    if r is bad or r["step"] not in (S, S - 1) or "storage_ptr" not in r:
        continue
    a = int(r["storage_ptr"], 16); b = a + r.get("storage_nbytes", 0)
    if a < hi and lo < b:
        print(f"DONOR: step={r['step']} {r['layer_name']}|{r['role']} "
              f"shape={r['shape']} ptr={r['storage_ptr']}")
```

### Step 3: Allocator event trace (Run 1)

In `alloc_snapshot_step*_rank*.pickle`, search for:
```
(free addr=X on stream A) → (alloc addr=X on stream B)
```
at the victim's `storage_ptr` address. This proves the allocator handed out a
block still in-use on a different stream.

### Step 4: Examine the corrupted tensor

Load `bad_tensor_step*_rank0.pt` and inspect the NaN pattern:
- Single contiguous row of NaN = tile-sized late write (aliasing)
- Scattered NaN = numeric blowup (different root cause)

---

## Technical Notes

### Address Comparability (apple-to-apple)

The `storage_ptr` values are **absolute GPU virtual addresses** within the
process (via `tensor.untyped_storage().data_ptr()`). All records in a single
rank's JSONL share the same virtual address space. Comparing `storage_ptr`
ranges between any two records in the same file is a direct apples-to-apples
comparison — overlapping ranges = same physical GPU memory block.

### What "input" captures for ef[10]

The logger's forward hook records the first tensor argument passed to
`module.forward()`. For `projections.10.layers.0` (`nn.Linear(2048, 128)`),
this is the 8 MiB `[1024, 2048]` buffer from the embedding lookup output.

### No GPU sync from address capture

`data_ptr()` and `untyped_storage().data_ptr()` are pure host-side metadata
queries — NO GPU synchronization, NO kernel launch order change.

---

## Troubleshooting

### N = 0 (no modules hooked)

The logger patches `DistributedModelParallel.__init__`. If N = 0:
- The model was built before the logger was imported → move the import earlier
- The model doesn't use `DistributedModelParallel` → see `LOGGER_REFERENCE.md`

### No NaN reproduced after 5000 steps

- Run more trials (5–10) or longer (10000–20000 steps)
- The bug is timing-sensitive and intermittent
- If Run 1 never reproduces but you know the NaN happens in production, try Run 2
  (lower overhead preserves original timing better)

### JSONL is empty

- Check `NANLOG_CHANNELS` includes `input` (default is `act,igrad` only)
- Check N > 0 in startup log

### Allocator snapshot missing despite NaN

- Verify `NANLOG_ALLOC_SNAPSHOT=1` is set
- Check startup log for `alloc_snapshot=True`
- On some PyTorch builds, `torch.cuda.memory._record_memory_history` may not be
  available — the logger prints a warning if so

---

## Quick Reference: Copy-Paste Commands

### Run 1 (full diagnostic):
```bash
NANLOG_DIR=/output/run1_full_t1 \
NANLOG_CHANNELS=act,input,igrad,weight,bias,wgrad,bgrad \
NANLOG_WATCH_TYPES=Linear,LinearProjection,AttentionBlock,InteractionLayer,MLP,EmbeddingGate \
NANLOG_ADDR=1 NANLOG_LOCATE=1 NANLOG_BAD_VALUES=1 \
NANLOG_DUMP_TENSOR=1 NANLOG_ALLOC_SNAPSHOT=1 \
NANLOG_PRE_CONTEXT=20 NANLOG_SAMPLE_EVERY=1 \
  python3 /opt/nan_logger_v2/instrument_nan_logger.py \
  /path/to/your_training_script.py 2>&1 | tee /output/run1_full_t1/console.log
```

### Run 2 (lightweight, only if Run 1 doesn't reproduce):
```bash
NANLOG_DIR=/output/run2_light_t1 \
NANLOG_CHANNELS=act,input,igrad,weight,bias,wgrad,bgrad \
NANLOG_WATCH_TYPES=Linear \
NANLOG_ADDR=1 NANLOG_LOCATE=1 NANLOG_BAD_VALUES=1 \
NANLOG_DUMP_TENSOR=1 NANLOG_ALLOC_SNAPSHOT=0 \
NANLOG_PRE_CONTEXT=20 NANLOG_SAMPLE_EVERY=50 \
  python3 /opt/nan_logger_v2/instrument_nan_logger.py \
  /path/to/your_training_script.py 2>&1 | tee /output/run2_light_t1/console.log
```

---

## Version

- **Logger:** June 29, 2026 (NaN Logger v2)
- **Capabilities:** address capture, bad-value locate, allocator snapshot,
  tensor dump, pre-context buffer, input channel
- **Tested on:** PyTorch 2.x with ROCm, single-node multi-GPU, torchrec 0.5+
