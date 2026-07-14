"""Per-layer NaN / magnitude logger for the training run.

Watches the model layers (forward activations and backward gradients) so a
NaN/Inf can be traced back to the layer and step where it first appears.

What it records, per watched layer, per step:
  - nan_count / inf_count / huge_count   (huge = |x| > threshold)
  - finite_abs_max / finite_max / finite_min, logged EVERY step even when there
    are no NaNs yet. This is the key feature: it captures a value growing
    large -> huge -> NaN over steps, which a NaN-only check cannot see.
  - tf32_path: whether this layer ran on the fp32 + allow_tf32 (TF32) path
  - matmul_calls_so_far: a running counter (no GPU sync) that can be used to
    cross-reference a matmul-level "call #N" from a separate GEMM-output tool
  - identity: layer name, direction (fwd/bwd), rank/gpu/host/pid

How it stays out of the way (important):
  The per-layer hooks compute their NaN/magnitude reductions ON THE GPU and keep
  the results as GPU values WITHOUT reading them back (no .item()). Once per step
  all of those values are copied to the host in ONE batched transfer. So there is
  no per-GEMM host sync, the GPU kernel launch order is not changed, and any
  timing-sensitive behavior is preserved. (Reading a value back after every GEMM
  would force a sync per GEMM, serialize the stream, and change the timing.)

Known limitations:
  - This sees layer-level state, not individual GPU kernel ordering. It locates
    the first layer/step that goes bad; it cannot, by itself, prove a kernel-
    level timing issue (that would need per-GEMM ordering, i.e. a per-GEMM sync,
    which would change the timing).
  - It does not prove "clean inputs -> NaN output" for a specific GEMM. That
    would require capturing the GEMM operands and replaying them, which this
    logger does not do.

How to run:
  HIP_VISIBLE_DEVICES=0 NUM_STEPS=1000 \
    NANLOG_DIR=/output/nan_logger \
    python instrument_nan_logger.py /path/to/standalone_single_file.py

  The logger arms its hooks, then runs your script. The hooks attach to the real
  model the moment it is built, so no edit to the training script is needed.

Settings (environment variables):
  NANLOG_DIR             output dir for the JSONL + summary (default ./nan_logger_out)
  NANLOG_HUGE_THRESHOLD  "huge" cutoff, |x| > T             (default 1e10)
  NANLOG_SAMPLE_EVERY    write a record for a CLEAN layer 1 step in N (default 50;
                         layers with NaN/Inf/huge are ALWAYS written). Keeps the
                         output small while still sampling the magnitude trend.
  NANLOG_PRE_CONTEXT     keep the last K steps of ALL records in memory and dump
                         them to the JSONL the moment the first bad layer is seen,
                         so you get full-resolution run-up to the NaN without
                         logging every step on a clean run (default 0 = off). When
                         >0 this overrides sampling for those buffered steps.
  NANLOG_MAX_RECORDS     hard cap on records written        (default 200000)
  NANLOG_WATCH_TYPES     comma list of module CLASS names to watch
                         (default "Linear" -> torch.nn.Linear layers). Combined
                         with NANLOG_WATCH_NAMES by UNION (a module is watched if
                         it matches either). The "Linear" default is suppressed
                         automatically when NANLOG_WATCH_NAMES is set, so naming a
                         layer does not also pull in every Linear.
  NANLOG_WATCH_NAMES     comma list of substrings matched against each module's
                         fully-qualified path; a module is watched if its path
                         CONTAINS any of them (default empty = match nothing by
                         name). Use to target a specific layer or block without
                         editing the model, e.g.
                           NANLOG_WATCH_NAMES=encoder.blocks.3.mlp.fc1
                         watches only that layer (no NANLOG_WATCH_TYPES needed);
                           NANLOG_WATCH_NAMES=encoder.blocks.3
                         watches the whole block (all sub-layers). To watch named
                         layers AND a class, set both (e.g. add
                         NANLOG_WATCH_TYPES=MoELayer for that block + all MoE).
  NANLOG_CHANNELS        comma list of capture channels (default "act,igrad" =
                         the original activation + grad_input behavior, byte-equal
                         timing). Each channel is an independent observation that
                         adds its own on-GPU reductions; new channels are OFF by
                         default so repro runs keep the original timing profile.
                         All reductions feed the SAME single per-step drain, so no
                         channel adds a host sync. Valid channels:
                           act    forward output activation        (fwd hook)
                           input  forward input activation         (fwd hook)
                           igrad  grad w.r.t. inputs (grad_input)  (bwd hook)
                           weight param values, ndim>=2            (root pre-hook*)
                           bias   param values, ndim==1            (root pre-hook*)
                           wgrad  param.grad, ndim>=2              (opt step-hook+)
                           bgrad  param.grad, ndim==1              (opt step-hook+)
                         *weight/bias read at the START of the step (root pre-hook),
                         so their values are the PREVIOUS step's (the only sync-free
                         read point for persistent params); each such record carries
                         value_is_from_prev_step=true.
                         +wgrad/bgrad read param.grad at the optimizer step_pre_hook
                         (after backward, before the update), so they are the CURRENT
                         step's grad (value_is_from_prev_step=false) and sync-free /
                         outside the autograd graph. Requires a torch.optim.Optimizer
                         (auto-discovered, any subclass); if grads are freed in
                         backward (optimizer-in-backward/fused) or there is no
                         optimizer, a one-time WARNING is logged and these channels
                         produce nothing — use weight/bias instead.
                         weight/bias split is by SHAPE (ndim), so custom param
                         names (.w, .scale, .b) are handled; the exact name is in
                         param_name. Replaces the old NANLOG_BWD.
  NANLOG_MAX_PARAM_NUMEL param channels skip params above this many elements, as a
                         backstop for pathologically large dense params (default
                         50000000). Embedding tables are skipped by TYPE, not this
                         guard, so it can stay high. Skipped params are catalogued
                         in the summary (never silently dropped). Embedding-table
                         scanning options are a separate deferred follow-up.
  NANLOG_STOP_ON_FIRST   "1" -> stop writing clean records after the first bad
                         layer is seen (default 0)
  NANLOG_VERBOSE         "1" -> print one line per step       (default 0)
  NANLOG_ADDR            "1" -> record each watched tensor's GPU address (data_ptr)
                         + backing-storage extent (storage_ptr, storage_offset_bytes,
                         storage_nbytes). Host-side metadata, no GPU sync. Default 0
                         (off) so the base record schema stays minimal; enable to
                         trace memory aliasing. For a sparse (KJT) record the address
                         is that of the KJT's .values() tensor, not the KJT object.
  NANLOG_LOCATE          "1" -> also record bad_rows: how many rows (dim 0) hold a
                         NaN/Inf/huge element. Extra on-GPU reduction, same drain,
                         no host sync (default 0).
  NANLOG_BAD_VALUES      "1" -> for each BAD tensor, record first_bad_flat_idx,
                         first_bad_row, first_bad_col, first_bad_value (first
                         NaN/Inf/huge element). GPU reductions, same drain (default 0).
  NANLOG_DUMP_TENSOR     "1" -> on first NaN/Inf/huge detection, save the full bad
                         tensor to a .pt file (one-shot; the input is saved before
                         the output when both are bad). May delay allocator reuse
                         (default 0).
  NANLOG_ALLOC_SNAPSHOT  "1" -> enable the caching-allocator event recorder and dump
                         a snapshot pickle on first NaN/Inf/huge (alloc/free events
                         with address, size, stream, Python stack). ~10% step
                         overhead; GPU kernel timing unaffected (default 0).
  NANLOG_PIPELINE        "1" -> monkeypatch TrainPipelineSparseDist stage methods
                         (copy_batch_to_gpu / start_sparse_data_dist /
                         wait_sparse_data_dist) to tag records with a `phase` and
                         re-scan the tracked flow objects at each stage, before the
                         forward reads them. Warns once and stays inactive if the
                         stage methods are absent. Same per-step drain (default 0).
  NANLOG_TRACK_ATTR      comma list of batch attribute/key names to follow as flow
                         tensors (default "embedding_features"); a name may resolve
                         to a Tensor, list/tuple/dict of Tensors, or a KJT (its
                         values() is tracked). Falls back to a bounded auto-discovery
                         walk if none are found. Only with NANLOG_PIPELINE=1.
  NANLOG_TRACK_MAX       hard cap on how many tensor objects the tracker follows
                         (default 64).
  NANLOG_SPARSE          "1" -> at the pipeline stages, capture cheap host-side
                         KJT/JaggedTensor metadata (num keys, lengths/offsets shape,
                         total_lengths, values shape/dtype/device) and route the
                         index values through the normal sync-free reduction. Requires
                         NANLOG_PIPELINE=1 (default 0).
  NANLOG_SPARSE_HEAVY    "1" -> add sparse stats needing a host readback (empty_bags,
                         max_bag_len, index value min/max). The one sparse path that
                         syncs; gated separately. Requires NANLOG_SPARSE=1 (default 0).
  NANLOG_TRACK_EVERY_LAYER "1" -> also re-scan the tracked flow objects at EACH watched
                         layer's forward hook (record carries checkpoint=<layer_name>),
                         bracketing the corruption to a layer interval within forward.
                         Multiplies the per-step reduction count; can perturb a
                         timing-sensitive bug. Requires NANLOG_PIPELINE=1 (default 0).
  NANLOG_TRACK_LAYER_STRIDE  re-scan every Kth watched layer (default 1). Only with
                         NANLOG_TRACK_EVERY_LAYER=1.
  NANLOG_BOUNDS          per-tensor in-range check "substr:lo:hi;substr:lo:hi": a
                         watched tensor takes the first entry whose substring is in
                         its layer_name; elements outside [lo,hi] count as oob and
                         mark the record bad (kind="oob"). Two-sided, unlike the
                         one-sided huge threshold. One reduction, same drain, no host
                         sync. Unmatched tensors are unbounded (default off).
  NANLOG_BOUND_LO / NANLOG_BOUND_HI  match-all fallback range applied to EVERY tensor
                         (degenerate single-range form of NANLOG_BOUNDS; sensible only
                         when watching one well-bounded target).

  Every record carries a `batch_id`, assigned when a batch enters at
  copy_batch_to_gpu and re-read at each later checkpoint, so one batch's
  copy/sparse/forward records can be grouped across steps (null before its copy).
"""
from __future__ import annotations

import json
import math
import os
import runpy
import socket
import sys
import time
from collections import deque
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DIR = Path(os.environ.get("NANLOG_DIR", "nan_logger_out"))
_HUGE = float(os.environ.get("NANLOG_HUGE_THRESHOLD", "1e10"))
_SAMPLE_EVERY = int(os.environ.get("NANLOG_SAMPLE_EVERY", "50"))
_PRE_CONTEXT = int(os.environ.get("NANLOG_PRE_CONTEXT", "0"))
_MAX_RECORDS = int(os.environ.get("NANLOG_MAX_RECORDS", "200000"))
# A module is watched if its CLASS NAME is in _WATCH_TYPES, OR its fully-qualified
# module path CONTAINS any substring in _WATCH_NAMES (union, not intersection — so
# you can target one specific layer AND a whole type at once). An unset axis
# contributes no matches.
#
# The type filter defaults to "Linear", BUT only when no name filter is given.
# This makes the common targeted case intuitive: setting NANLOG_WATCH_NAMES alone
# watches exactly those named layers — the "Linear" default does NOT silently
# union in every Linear, so you never have to write NANLOG_WATCH_TYPES= to clear
# it. To watch named layers AND a type, set both explicitly:
#   NANLOG_WATCH_NAMES=encoder.blocks.3                   -> only that block
#   NANLOG_WATCH_NAMES=...  NANLOG_WATCH_TYPES=MoELayer   -> that block + all MoE
# The bare default (neither var set) stays types=Linear -> original behavior.
_WATCH_NAMES = tuple(
    s.strip() for s in os.environ.get("NANLOG_WATCH_NAMES", "").split(",") if s.strip()
)
_types_default = "" if _WATCH_NAMES else "Linear"
_WATCH_TYPES = tuple(
    s.strip() for s in os.environ.get("NANLOG_WATCH_TYPES", _types_default).split(",") if s.strip()
)
# Capture channels (comma list). Each channel is one independently switchable
# observation; each adds its own on-GPU reductions, so default to the cheap pair
# that reproduces the original (activation + grad_input) timing profile exactly.
#   act    forward output activation       (fwd hook)
#   input  forward input activation        (fwd hook)
#   igrad  grad w.r.t. inputs (grad_input) (bwd hook)   -- old NANLOG_BWD
#   weight param values,  ndim >= 2        (root pre-hook, PREV step, sync-free)
#   bias   param values,  ndim == 1        (root pre-hook, PREV step, sync-free)
#   wgrad  param.grad,    ndim >= 2        (opt step_pre_hook, CURRENT step, sync-free)
#   bgrad  param.grad,    ndim == 1        (opt step_pre_hook, CURRENT step, sync-free)
_ALL_CHANNELS = ("act", "input", "igrad", "weight", "bias", "wgrad", "bgrad")
_CHANNELS = frozenset(
    s.strip() for s in os.environ.get("NANLOG_CHANNELS", "act,igrad").split(",") if s.strip()
)
_unknown_channels = _CHANNELS - set(_ALL_CHANNELS)
# Param channels split by WHERE they read (both sync-free, both feed the one drain):
#   value channels (weight/bias) -> persistent params, read in the root pre-hook
#     at the START of the step => PREVIOUS step's value (from_prev_step=True).
#   grad channels (wgrad/bgrad)  -> param.grad, read at the OPTIMIZER step_pre_hook
#     (after backward, before the update) => CURRENT step's grad, and OUTSIDE the
#     autograd graph so it cannot perturb timing-sensitive behavior. Recovers grads
#     that the root pre-hook misses (the optimizer's zero_grad has already run by then).
_VALUE_CHANNELS = _CHANNELS & {"weight", "bias"}
_GRAD_CHANNELS = _CHANNELS & {"wgrad", "bgrad"}
_PARAM_CHANNELS = _VALUE_CHANNELS | _GRAD_CHANNELS
# Number of steps to wait before warning that enabled grad channels produced no
# records, so the warning reflects steady state rather than a cold-start step.
_GRAD_WARN_AFTER = 3
# Size backstop for param scanning: skip params above this many elements (logged,
# never silently dropped). Set high so dense weights pass; embedding tables are
# gated separately by module TYPE (see _is_embedding_module), not by this guard.
_MAX_PARAM_NUMEL = int(os.environ.get("NANLOG_MAX_PARAM_NUMEL", "50000000"))
# Module class names treated as embeddings -> param channels SKIP them (deferred
# to the embedding-options follow-up). Tracked here so the param walk is correct.
_EMBEDDING_TYPES = ("Embedding", "EmbeddingBag", "EmbeddingBagCollection",
                    "EmbeddingCollection", "ShardedEmbeddingBagCollection",
                    "ShardedEmbeddingCollection")
_STOP_ON_FIRST = os.environ.get("NANLOG_STOP_ON_FIRST", "0") == "1"
_VERBOSE = os.environ.get("NANLOG_VERBOSE", "0") == "1"
# Address capture (default OFF, sync-free). Records each watched tensor's GPU
# virtual address + backing storage extent. This is the bridge that lets a
# memory-ALIASING corruption be traced to its PRODUCER: when projections.10's
# INPUT goes bad at step S, its data_ptr/storage names the exact 8 MiB block; any
# OTHER watched module whose output/param shares that storage at a nearby step is
# the donor/writer. data_ptr()/storage queries are pure host-side metadata — no
# GPU sync, no kernel-order change — so it is safe to enable on repro runs, but it
# is OFF by default so the base record schema stays minimal.
_CAPTURE_ADDR = os.environ.get("NANLOG_ADDR", "0") == "1"
# Locate (default OFF). For each watched tensor also reduce how many ROWS (dim 0)
# contain a NaN/Inf/huge element -> `bad_rows`. The acceptance test for the
# aliasing hypothesis is bad_rows==1 on the proj.10 input (a single ~8 KiB row =
# one tile-sized late write, vs a spread-out numeric blowup). This is an extra
# on-GPU reduction that feeds the SAME per-step drain, so it adds no host sync; it
# is off by default to keep the base repro's GPU kernel mix unchanged.
_LOCATE = os.environ.get("NANLOG_LOCATE", "0") == "1"
# Bad-value capture (default OFF). For each watched tensor also find the FIRST bad
# element's flat index, row, col, and actual value. All are GPU scalar reductions
# (argmax + indexing) that feed the same per-step drain -> no host sync.
_BAD_VALUES = os.environ.get("NANLOG_BAD_VALUES", "0") == "1"
# Allocator snapshot (default OFF). Records full caching-allocator alloc/free event
# history with stream IDs and Python call stacks. On first NaN detection, dumps the
# snapshot to a pickle file for post-hoc aliasing analysis. Adds ~10% step overhead
# from per-allocation stack capture; GPU kernel timing unaffected.
_ALLOC_SNAPSHOT = os.environ.get("NANLOG_ALLOC_SNAPSHOT", "0") == "1"
# Dump full bad tensor(s) on first detection (default OFF — see docstring for risk).
_DUMP_TENSOR = os.environ.get("NANLOG_DUMP_TENSOR", "0") == "1"
# Pipeline-stage tracking (default OFF). Monkeypatch TrainPipelineSparseDist stage
# methods to checkpoint tracked flow objects (EF by default) BEFORE forward reads
# them, tagging every record with its `phase`. This closes the copy->forward gap
# that the layer hooks cannot see. General tensor-flow tracker: EF is the default
# target via NANLOG_TRACK_ATTR, not a hard-coded case.
_PIPELINE = os.environ.get("NANLOG_PIPELINE", "0") == "1"
_TRACK_ATTR = tuple(
    s.strip() for s in os.environ.get("NANLOG_TRACK_ATTR", "embedding_features").split(",")
    if s.strip()
)
_TRACK_MAX = int(os.environ.get("NANLOG_TRACK_MAX", "64"))
# Sparse (TorchRec/FBGEMM) metadata capture at the pipeline stages (default OFF).
# Cheap, host-side only. _SPARSE_HEAVY adds the readback-requiring stats (the only
# sparse path that syncs); gated separately so it can never fire on a repro run.
_SPARSE = os.environ.get("NANLOG_SPARSE", "0") == "1"
_SPARSE_HEAVY = os.environ.get("NANLOG_SPARSE_HEAVY", "0") == "1"
# Per-layer re-scan of the tracked flow objects (default OFF). When on, EACH watched
# layer's forward hook ALSO re-scans the SAME tracked ef objects (not just the tensor
# that layer consumes), tagged checkpoint=<layer_name>. This turns "corrupt somewhere
# in forward-N" into "clean at layer B, corrupt at layer C" — the diagram-3 step B
# within-forward bracket. COST: it multiplies the per-step reduction count by
# (tracked_objects x scanned_layers); with 31 ef x 156 layers that is ~15x the base
# GPU reduction kernels, launched inline on the compute stream, and CAN suppress the
# timing-sensitive bug. So it is OFF by default and STRIDED: with stride K only every
# Kth watched layer re-scans, trading resolution for perturbation. Start at a coarse
# stride, tighten only once the stage-level bracket is stable.
_TRACK_EVERY_LAYER = os.environ.get("NANLOG_TRACK_EVERY_LAYER", "0") == "1"
_TRACK_LAYER_STRIDE = max(1, int(os.environ.get("NANLOG_TRACK_LAYER_STRIDE", "1")))
# In-range bound check (default OFF). Flags elements OUTSIDE [lo, hi] as bad — a
# two-sided test the one-sided huge threshold (|x| > T) cannot express (it cannot
# catch a small out-of-range value, e.g. a negative in a [0,60] embedding input).
# Bounds are PER-TENSOR by nature (a Linear activation has no [0,60] rule), so the
# range is chosen per watched tensor by matching its record name against patterns:
#   NANLOG_BOUNDS="emb_proj.projections:0:60;dense_arch:-10:10"
# Each entry is "substring:lo:hi"; a tensor takes the FIRST entry whose substring is
# in its layer_name (so this composes with NANLOG_WATCH_NAMES). Tensors matching no
# pattern are unbounded (oob_count=0). The bare match-all form is the degenerate
# single-range case: NANLOG_BOUND_LO / NANLOG_BOUND_HI apply to EVERY tensor and are
# only sensible when you watch a single well-bounded target. When any bound is
# active, every record carries oob_count (0 when no bound matched) so the fixed
# per-step drain stack stays valid; an out-of-range tensor is `bad` with kind="oob".
# One extra on-GPU reduction, same drain, no host sync.
def _parse_bounds(spec: str):
    """Parse "substr:lo:hi;substr:lo:hi" into [(substr, lo, hi), ...]. Malformed
    entries are skipped with a warning rather than crashing the sidecar."""
    out = []
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.rsplit(":", 2)
        if len(parts) != 3:
            sys.stderr.write(f"nanlog: WARNING: ignoring malformed NANLOG_BOUNDS entry {entry!r} "
                             "(expected substring:lo:hi)\n")
            continue
        substr, lo_s, hi_s = parts
        try:
            out.append((substr, float(lo_s), float(hi_s)))
        except ValueError:
            sys.stderr.write(f"nanlog: WARNING: ignoring NANLOG_BOUNDS entry {entry!r} "
                             "(lo/hi not numeric)\n")
    return out


_BOUNDS = _parse_bounds(os.environ.get("NANLOG_BOUNDS", ""))
_BOUND_LO = os.environ.get("NANLOG_BOUND_LO")
_BOUND_HI = os.environ.get("NANLOG_BOUND_HI")
# Match-all fallback (applies to every tensor) when the bare LO/HI form is used.
_BOUND_GLOBAL = (
    (float(_BOUND_LO) if _BOUND_LO is not None else float("-inf"),
     float(_BOUND_HI) if _BOUND_HI is not None else float("inf"))
    if (_BOUND_LO is not None or _BOUND_HI is not None) else None
)
_BOUNDS_ACTIVE = bool(_BOUNDS) or _BOUND_GLOBAL is not None


def _bound_for(layer_name: str):
    """Return (lo, hi) for this tensor: first matching NANLOG_BOUNDS pattern, else
    the global LO/HI fallback, else None (unbounded)."""
    for substr, lo, hi in _BOUNDS:
        if substr in layer_name:
            return (lo, hi)
    return _BOUND_GLOBAL

# Create the output dir. If it is not writable (e.g. the default lands in a
# read-only working dir), fall back to a temp dir rather than crashing the
# training run — the logger is a sidecar and must never take the job down.
try:
    _DIR.mkdir(parents=True, exist_ok=True)
except OSError as _e:
    import tempfile
    _fallback = Path(tempfile.gettempdir()) / "nan_logger_out"
    _fallback.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(
        f"nanlog: WARNING: cannot write NANLOG_DIR={_DIR} ({_e}); "
        f"falling back to {_fallback}. Set NANLOG_DIR to a writable path.\n")
    sys.stderr.flush()
    _DIR = _fallback
_RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
_HOST = socket.gethostname()
_PID = os.getpid()
_JSONL = _DIR / f"layers_rank{_RANK}.jsonl"
_SUMMARY = _DIR / f"summary_rank{_RANK}.json"


def _log(msg: str) -> None:
    sys.stderr.write(f"{time.strftime('%H:%M:%S')} [PID={_PID} rank={_RANK}] nanlog: {msg}\n")
    sys.stderr.flush()


if _unknown_channels:
    _log(f"WARNING: unknown NANLOG_CHANNELS {sorted(_unknown_channels)} ignored; "
         f"valid: {list(_ALL_CHANNELS)}")

if not _WATCH_TYPES and not _WATCH_NAMES:
    _log("WARNING: both NANLOG_WATCH_TYPES and NANLOG_WATCH_NAMES are empty; "
         "no modules will be watched")

# Scope-knob dependency warnings: these features silently no-op unless their
# prerequisite is set, which reads as "the flag did nothing" with no explanation.
# Warn once at import so a misconfigured scope is visible, mirroring the
# unknown-channels warning above.
if _TRACK_EVERY_LAYER and not _PIPELINE:
    _log("WARNING: NANLOG_TRACK_EVERY_LAYER=1 has no effect without NANLOG_PIPELINE=1 "
         "(the per-layer re-scan targets are captured by the pipeline hook); set "
         "NANLOG_PIPELINE=1 to enable per-layer tracking.")
if os.environ.get("NANLOG_TRACK_LAYER_STRIDE") is not None and not _TRACK_EVERY_LAYER:
    _log("WARNING: NANLOG_TRACK_LAYER_STRIDE is set but NANLOG_TRACK_EVERY_LAYER is "
         "off, so the stride is ignored; set NANLOG_TRACK_EVERY_LAYER=1 to use it.")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# Per-step pending reductions: list of (record_dict, gpu_scalar_tensor_dict).
# The gpu scalars are stashed WITHOUT .item(); drained once per step.
_pending: list = []
_step = 0
_records_written = 0
# Current pipeline phase, set by the stage-boundary checkpoints when NANLOG_PIPELINE
# is on (copy / sparse_start / sparse_wait / forward). Every record stamps this so
# a corruption can be bracketed to the interval between two consecutive phases.
# "forward" is the default because the layer hooks all fire during the root forward.
_phase = "forward"
# Current batch identity, the cross-STEP / cross-PHASE correlation key. Because the
# 3-in-flight pipeline scans copy(N+2), sparse(N+1), forward(N) in one tick, the
# step alone cannot line up one batch's copy/sparse/forward observations (they land
# in DIFFERENT steps and the data_ptr is recycled, so neither is a reliable join
# key). batch_id is: it is assigned once at copy_batch_to_gpu, carried on the batch
# object, and re-read at every later checkpoint so all records of the same batch
# share it. Grouping the JSONL by batch_id reconstructs one batch's clean->corrupt
# timeline with no pipeline-depth arithmetic. None until the first copy checkpoint.
_batch_id = None
_batch_counter = 0           # monotonic source for batch_id (host-side, no sync)
# Fallback map id(batch_obj) -> batch_id, used only when the batch object cannot be
# tagged with an attribute (some torchrec versions re-wrap the batch). Bounded ring.
_batch_id_by_obj: dict = {}
_BATCH_ID_ATTR = "_nanlog_batch_id"   # attribute stamped on the batch object
# Held references to the forward batch's tracked ef tensor OBJECTS for this step, so
# each watched layer's forward hook can re-scan the SAME objects (per-layer re-scan,
# NANLOG_TRACK_EVERY_LAYER). Rebuilt every step in the root forward pre-hook; empty
# when per-layer re-scan is off. Holding the ref is free (the tensor is already alive
# for the forward pass); the reductions happen in the layer hooks.
_layer_scan_targets: list = []
_layer_scan_idx = 0          # counts watched-layer fwd hooks this step, for striding
_matmul_calls = 0          # running count of watched layer tensors (no GPU sync)
_first_bad = None          # first layer to go bad: {"step","layer","direction","kind"}
# OOB aggregates for the summary, so a run's out-of-bound status is visible without
# grepping the JSONL. Updated in the drain from oob_count/finite range. _first_oob is
# the first bound violation seen (may equal _first_bad if oob is the first badness);
# _oob_records counts records with oob_count>0; _peak_finite_* track the widest range
# observed on any BOUND tensor (so "how close did it get to the bound" is answerable
# on a clean run too).
_first_oob = None
_oob_records = 0
_peak_finite_max = None
_peak_finite_min = None
_root_installed = False
# Ring buffer of the last _PRE_CONTEXT steps of records that were NOT otherwise
# written (clean + unsampled). Dumped to the JSONL when the first bad is seen so
# we get the full run-up to the NaN. Each entry is (step, [record, ...]).
_pre_buffer = deque(maxlen=_PRE_CONTEXT) if _PRE_CONTEXT > 0 else None
_pre_flushed = False       # once the buffer is dumped, write every step thereafter
# Precomputed param-scan plan: one entry per parameter the param channels WILL
# read each step, with its channels already resolved. Built once at attach so the
# per-step pre-hook does no named_parameters()/type/numel work on the hot path.
# Each entry: {"layer_name", "param_name", "param", "value_role", "grad_role"}.
_scan_plan: list = []
# Params the param channels deliberately do NOT scan (embedding/oversized), with
# reason. Built once at attach; surfaced in the summary so "not scanned" is never
# mistaken for "scanned and clean".
_skipped_params: list = []
# Fully-qualified names of every watched module (for the summary + a startup
# eyeball check that the name/type filter targeted what you intended).
_watched_names: list = []
# Grad-channel plan: id(param) -> {"layer_name","param_name","grad_role"} for every
# watched param a grad channel wants. Built at attach (same eligibility as the value
# plan); used by the optimizer step_pre_hook to label grads by id without any
# named_parameters() walk on the hot path.
_grad_plan: dict = {}
# True once at least one optimizer has had a step_pre_hook registered. If grad
# channels are requested but this stays False, no Optimizer subclass was ever built
# (regime #3 / inline update) -> warn.
_opt_registered = False
# Running count of grad records the optimizer hook has stashed, and a once-only flag
# for the "grads requested but not readable here" warning (regime #1).
_opt_grad_stashed = 0
_grad_warned = False


# ---------------------------------------------------------------------------
# On-device reductions (NO .item() here — that is the whole point)
# ---------------------------------------------------------------------------
def _device_stats(t: torch.Tensor, bound=None) -> dict:
    """Return a dict of GPU scalar tensors. No host sync; read back later.

    finite_abs_max masks non-finite values to 0 first, so it stays meaningful
    even when some elements are already NaN/Inf (lets us see the magnitude
    trend leading up to a NaN).

    bound: optional (lo, hi) for this tensor. When bound tracking is active
    (_BOUNDS_ACTIVE), oob_count is ALWAYS emitted (0 when bound is None) so the
    fixed per-step drain stack stays uniform across records."""
    fin = torch.isfinite(t)
    # Mask non-finite values out per reduction with an identity that can never
    # win: 0 for abs-max (no real magnitude loses to 0), -inf for max, +inf for
    # min. Using 0 for max/min would be a BUG -- e.g. all-negative finite values
    # plus one NaN would report finite_max=0 instead of the true (negative) max.
    abs_safe = torch.where(fin, t.abs(), torch.zeros((), dtype=t.dtype, device=t.device))
    neg_inf = torch.full((), float("-inf"), dtype=t.dtype, device=t.device)
    pos_inf = torch.full((), float("inf"), dtype=t.dtype, device=t.device)
    safe_max = torch.where(fin, t, neg_inf)
    safe_min = torch.where(fin, t, pos_inf)
    stats = {
        "nan_count": torch.isnan(t).sum(),
        "inf_count": torch.isinf(t).sum(),
        "finite_count": fin.sum(),
        "huge_count": (fin & (t.abs() > _HUGE)).sum(),
        "finite_abs_max": abs_safe.max(),
        "finite_max": safe_max.max(),
        "finite_min": safe_min.min(),
        "numel": torch.tensor(t.numel(), device=t.device),
    }
    # Out-of-range count: finite elements outside [lo, hi]. Emitted for EVERY record
    # once bound tracking is active (0 when this tensor has no bound) so the drain's
    # fixed key stack is uniform. One extra reduction, same drain, no host sync.
    if _BOUNDS_ACTIVE:
        if bound is not None:
            lo, hi = bound
            stats["oob_count"] = (fin & ((t < lo) | (t > hi))).sum()
        else:
            stats["oob_count"] = torch.zeros((), dtype=torch.int64, device=t.device)
    # Bad mask includes out-of-range so NANLOG_LOCATE/BAD_VALUES point at the OOB
    # element too, not only NaN/Inf/huge.
    if _LOCATE or _BAD_VALUES:
        bad = (~fin) | (fin & (t.abs() > _HUGE))
        if _BOUNDS_ACTIVE and bound is not None:
            lo, hi = bound
            bad = bad | (fin & ((t < lo) | (t > hi)))
    if _LOCATE:
        rows = t.shape[0] if t.dim() >= 1 else 1
        stats["bad_rows"] = bad.reshape(rows, -1).any(dim=1).sum()
    if _BAD_VALUES:
        flat_bad = bad.reshape(-1).to(torch.int64)
        first_idx = flat_bad.argmax()          # GPU scalar: flat index of first bad
        stats["first_bad_flat"] = first_idx
        stats["first_bad_val"] = t.reshape(-1)[first_idx]  # GPU scalar via GPU-scalar indexing
        if t.dim() >= 2:
            # Elements per dim-0 slice — consistent with bad_rows (which reshapes
            # to (shape[0], -1)). For a [256,8192] tensor this is 8192; for a
            # [B,C,H,W] tensor this is C*H*W.
            elems_per_row = t.numel() // t.shape[0]
            stats["first_bad_row"] = first_idx // elems_per_row
            stats["first_bad_col"] = first_idx % elems_per_row
        else:
            stats["first_bad_row"] = torch.zeros((), dtype=torch.int64, device=t.device)
            stats["first_bad_col"] = first_idx
    return stats


def _is_tf32_path(t: torch.Tensor) -> bool:
    """A matmul uses the TF32 path when operands are fp32 AND allow_tf32 is on."""
    return bool(t.dtype is torch.float32 and torch.backends.cuda.matmul.allow_tf32)


# Channels that represent compute-flow tensors (forward/backward activations).
# Only these advance matmul_calls_so_far, so the counter keeps cross-referencing a
# GEMM-output scanner's "call #N" even when param channels are enabled.
_FLOW_ROLES = frozenset({"act", "igrad"})


def _stash(layer_name: str, direction: str, t: torch.Tensor, role: str = "act",
           param_name: str = "", from_prev_step: bool = False,
           extra: dict = None) -> None:
    """Queue one tensor's on-GPU reduction for this step. No host sync.

    Args:
        layer_name:     fully-qualified module name the tensor belongs to.
        direction:      "fwd" | "bwd" | "param" | "grad" | "pipeline" — coarse
                        origin of the tensor (see ``role`` for the precise channel).
        t:              the tensor to reduce (skipped if empty / non-tensor).
        role:           the channel that produced this record
                        (act/input/igrad/weight/bias/wgrad/bgrad/track/sparse).
        param_name:     exact owned-parameter name (e.g. "w", "scale") for the
                        param/grad channels; "" for activation channels.
        from_prev_step: True for the weight/bias value channels, which are read in
                        the root pre-hook and therefore hold the PREVIOUS step's
                        value (the only sync-free read point for persistent params).
                        False for activations and for the wgrad/bgrad channels,
                        which are read at the optimizer step boundary (current
                        step). Flagged so the JSONL is never misread.
        extra:          optional dict of host-side metadata merged into the record
                        (e.g. sparse KJT stats). No GPU sync — caller's job.
    """
    global _matmul_calls
    if not torch.is_tensor(t) or t.numel() == 0:
        return
    if role in _FLOW_ROLES:
        _matmul_calls += 1
    rec = {
        "type": "layer_step",
        "step": _step,
        "phase": _phase,
        "batch_id": _batch_id,
        "rank": _RANK, "gpu": _RANK, "host": _HOST, "pid": _PID,
        "layer_name": layer_name,
        "direction": direction,
        "role": role,
        "param_name": param_name,
        "value_is_from_prev_step": from_prev_step,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "tf32_path": _is_tf32_path(t),
        "matmul_calls_so_far": _matmul_calls,
    }
    if extra:
        rec.update(extra)
    if _CAPTURE_ADDR:
        # Pure host-side metadata (no GPU sync): the tensor's address and the
        # extent of its backing block. Correlate ACROSS records of the SAME step
        # (or step-1) to find which producer's buffer aliases a corrupted input:
        # equal/overlapping [storage_ptr, storage_ptr+storage_nbytes) ranges name
        # the donor/writer pair that delivered the garbage to proj.10's input.
        try:
            st = t.untyped_storage()
            rec["data_ptr"] = hex(t.data_ptr())
            rec["storage_ptr"] = hex(st.data_ptr())
            rec["storage_offset_bytes"] = int(t.storage_offset() * t.element_size())
            rec["storage_nbytes"] = int(st.nbytes())
        except Exception:
            rec["data_ptr"] = hex(t.data_ptr())
    bound = _bound_for(layer_name) if _BOUNDS_ACTIVE else None
    _pending.append((rec, _device_stats(t, bound),
                     t if _DUMP_TENSOR and not _tensor_dumped else None))


# ---------------------------------------------------------------------------
# Per-step drain (ONE sync for ALL watched tensors this step) — the only host sync
# ---------------------------------------------------------------------------
def _write_rec(rec: dict) -> None:
    """Append one record to the JSONL, respecting the hard record cap."""
    global _records_written
    if _records_written >= _MAX_RECORDS:
        return
    with _JSONL.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    _records_written += 1


def _drain_step() -> None:
    """Flush last step's pending reductions with a single batched sync."""
    global _records_written, _first_bad, _pre_flushed, _tensor_dumped
    global _first_oob, _oob_records, _peak_finite_max, _peak_finite_min
    if not _pending:
        return
    pending, _pending[:] = list(_pending), []

    # Batch every scalar into one stacked tensor -> ONE .tolist() host transfer.
    keys = ["nan_count", "inf_count", "finite_count", "huge_count",
            "finite_abs_max", "finite_max", "finite_min", "numel"]
    if _BOUNDS_ACTIVE:
        keys = keys + ["oob_count"]
    if _LOCATE:
        keys = keys + ["bad_rows"]
    if _BAD_VALUES:
        keys = keys + ["first_bad_flat", "first_bad_val", "first_bad_row", "first_bad_col"]
    flat = []
    for _rec, stats, _held_t in pending:
        for k in keys:
            flat.append(stats[k].to(torch.float64).reshape(()))
    try:
        vals = torch.stack(flat).cpu().tolist() if flat else []
    except Exception as e:
        _log(f"drain failed: {e!r}")
        return

    worst_abs = -1.0
    worst = None
    bad_this_step = False        # did first_bad trigger during THIS drain?
    step_records = []            # all records this step, in order, for buffering
    for i, (rec, _stats, held_t) in enumerate(pending):
        base = i * len(keys)
        d = dict(zip(keys, vals[base:base + len(keys)]))
        rec["nan_count"] = int(d["nan_count"])
        rec["inf_count"] = int(d["inf_count"])
        rec["finite_count"] = int(d["finite_count"])
        rec["huge_count"] = int(d["huge_count"])
        # When a tensor is ENTIRELY non-finite (finite_count == 0), the masked
        # max/min reductions return -inf / +inf, which json.dumps emits as the
        # non-standard tokens -Infinity / Infinity (rejected by strict JSON
        # readers, jq, pandas). Emit JSON null instead. This is exactly the
        # first-bad case (an all-NaN gradient), so it must stay standards-clean.
        has_finite = rec["finite_count"] > 0
        rec["finite_abs_max"] = d["finite_abs_max"] if has_finite else None
        rec["finite_max"] = d["finite_max"] if has_finite else None
        rec["finite_min"] = d["finite_min"] if has_finite else None
        rec["numel"] = int(d["numel"])
        if _BOUNDS_ACTIVE:
            rec["oob_count"] = int(d["oob_count"])
            # Summary aggregates. Peak range is tracked only for tensors that
            # actually carry a bound, so "how close to the limit" reflects the
            # bounded targets, not unrelated activations.
            if _bound_for(rec["layer_name"]) is not None and has_finite:
                if _peak_finite_max is None or rec["finite_max"] > _peak_finite_max:
                    _peak_finite_max = rec["finite_max"]
                if _peak_finite_min is None or rec["finite_min"] < _peak_finite_min:
                    _peak_finite_min = rec["finite_min"]
            if rec["oob_count"] > 0:
                _oob_records += 1
                if _first_oob is None:
                    _first_oob = {"step": rec["step"], "layer": rec["layer_name"],
                                  "phase": rec.get("phase"), "oob_count": rec["oob_count"]}
        if _LOCATE:
            rec["bad_rows"] = int(d["bad_rows"])
        bad = (rec["nan_count"] or rec["inf_count"] or rec["huge_count"]
               or rec.get("oob_count", 0))
        rec["bad"] = bool(bad)
        if _BAD_VALUES and bad:
            rec["first_bad_flat_idx"] = int(d["first_bad_flat"])
            rec["first_bad_row"] = int(d["first_bad_row"])
            rec["first_bad_col"] = int(d["first_bad_col"])
            val = d["first_bad_val"]
            if math.isnan(val):
                rec["first_bad_value"] = "NaN"
            elif math.isinf(val):
                rec["first_bad_value"] = "Inf" if val > 0 else "-Inf"
            else:
                rec["first_bad_value"] = val
        step_records.append(rec)

        if bad and _first_bad is None:
            kind = ("nan" if rec["nan_count"] else
                    "inf" if rec["inf_count"] else
                    "huge" if rec["huge_count"] else "oob")
            _first_bad = {"step": rec["step"], "layer": rec["layer_name"],
                          "direction": rec["direction"], "kind": kind,
                          "matmul_calls_so_far": rec["matmul_calls_so_far"]}
            bad_this_step = True
            _log(f"FIRST BAD: step={rec['step']} layer={rec['layer_name']} "
                 f"dir={rec['direction']} kind={kind} "
                 f"call#~{rec['matmul_calls_so_far']} tf32={rec['tf32_path']}")
            _dump_alloc_snapshot(rec["step"])

        if bad and not _tensor_dumped and held_t is not None:
            _dump_bad_tensor(rec, held_t)

        if bad and rec["finite_abs_max"] is not None and rec["finite_abs_max"] > worst_abs:
            worst_abs, worst = rec["finite_abs_max"], rec

    # Pre-context mode: dump the buffered run-up the first time we see a bad
    # layer, then write every record from here on. Buffer holds prior steps'
    # clean records that were never written; flush them in chronological order
    # BEFORE this step's records so the JSONL stays ordered.
    if _pre_buffer is not None and bad_this_step and not _pre_flushed:
        for _bstep, brecs in _pre_buffer:
            for brec in brecs:
                _write_rec(brec)
        _pre_buffer.clear()
        _pre_flushed = True
        _log(f"pre-context: dumped {_PRE_CONTEXT}-step run-up before first bad")

    # Decide what to persist for this step.
    if _pre_buffer is not None and not _pre_flushed:
        # Pre-context active, no bad yet: write bad/sampled now, buffer the rest
        # so they can be dumped if a bad appears within the next K steps.
        held = []
        for rec in step_records:
            if rec["bad"] or (_step % _SAMPLE_EVERY == 0):
                _write_rec(rec)
            else:
                held.append(rec)
        if held:
            _pre_buffer.append((_step, held))
    else:
        # Normal path (no pre-context, or already flushed -> write everything).
        for rec in step_records:
            write = rec["bad"] or _pre_flushed or (_step % _SAMPLE_EVERY == 0)
            if write and not (_STOP_ON_FIRST and _first_bad and not rec["bad"]):
                _write_rec(rec)

    if _VERBOSE and worst is not None:
        _log(f"step={_step} worst_bad={worst['layer_name']} "
             f"abs_max={worst_abs:.3e} dir={worst['direction']}")


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def _first_tensor(x):
    """Pull the first tensor out of a hook arg (tensor, or tuple/list of them)."""
    if isinstance(x, (tuple, list)):
        return x[0] if x else None
    return x


def _rescan_tracked_at_layer(layer_name: str) -> None:
    """Re-scan the forward batch's tracked ef objects AT this layer boundary, so the
    JSONL shows each ef object clean->corrupt across the forward layer sequence
    (diagram-3 step B). Strided by NANLOG_TRACK_LAYER_STRIDE. Each record carries
    role='track', direction='pipeline', phase='forward', checkpoint=<layer_name> so
    it is distinct from the layer's own input/act. Reductions join the per-step
    drain (no extra host sync); the GPU-kernel cost is the striding tradeoff."""
    global _layer_scan_idx
    if not _layer_scan_targets:
        return
    do_scan = (_layer_scan_idx % _TRACK_LAYER_STRIDE == 0)
    _layer_scan_idx += 1
    if not do_scan:
        return
    for tname, t in _layer_scan_targets:
        _stash(tname, "pipeline", t, role="track", extra={"checkpoint": layer_name})


def _fwd_hook(name):
    def hook(_module, inp, out):
        # Stash the INPUT before the output (act). When an aliasing corruption
        # makes BOTH a layer's input and its resulting output bad in the same
        # step, the one-shot NANLOG_DUMP_TENSOR then captures the INPUT (the
        # corrupt origin / 8 MiB block) rather than the output (a downstream
        # consequence of feeding that garbage through the GEMM). This is the
        # whole point of the emb_proj aliasing workflow; it also matches the
        # natural input->output order. Layers where only the output is bad still
        # dump the output (the input isn't a bad candidate), so nothing is lost.
        if "input" in _CHANNELS:
            t = _first_tensor(inp)
            if torch.is_tensor(t):
                _stash(name, "fwd", t, role="input")
        if "act" in _CHANNELS:
            t = _first_tensor(out)
            if torch.is_tensor(t):
                _stash(name, "fwd", t, role="act")
        # Per-layer re-scan of the tracked ef objects (gated). Runs AFTER this
        # layer's own capture so the JSONL order reads: layer input/act, then the
        # ef snapshot as of this layer boundary.
        if _TRACK_EVERY_LAYER:
            _rescan_tracked_at_layer(name)
    return hook


def _bwd_hook(name):
    def hook(_module, grad_input, _grad_output):
        g = _first_tensor(grad_input)
        if torch.is_tensor(g):
            _stash(name, "bwd", g, role="igrad")
    return hook


def _is_embedding_module(module) -> bool:
    """True if any class in the MRO is a known embedding type (gated separately;
    param channels skip these — see embedding-options follow-up)."""
    return any(c.__name__ in _EMBEDDING_TYPES for c in type(module).__mro__)


def _plan_param_scan(layer_name: str, module: torch.nn.Module) -> None:
    """Resolve, once at attach, which of a module's own parameters the param
    channels will read each step. Eligibility (channel match, embedding type,
    size guard) is static, so deciding it here keeps the per-step pre-hook off the
    named_parameters()/type/numel path entirely. Skipped params are recorded in
    _skipped_params for the summary so 'not scanned' is never read as 'clean'."""
    embedding = _is_embedding_module(module)
    for pname, p in module.named_parameters(recurse=False):
        is_w = p.ndim >= 2
        value_role = "weight" if is_w else "bias"
        grad_role = "wgrad" if is_w else "bgrad"
        want_value = value_role in _CHANNELS
        want_grad = grad_role in _CHANNELS
        if not (want_value or want_grad):
            continue
        if embedding or p.numel() > _MAX_PARAM_NUMEL:
            _skipped_params.append({
                "layer_name": layer_name, "param_name": pname,
                "reason": "embedding_type" if embedding else "numel_over_guard",
                "numel": int(p.numel()), "ndim": int(p.ndim),
            })
            continue
        # Value channels read the persistent param in the root pre-hook.
        if want_value:
            _scan_plan.append({
                "layer_name": layer_name, "param_name": pname, "param": p,
                "value_role": value_role,
            })
        # Grad channels read param.grad at the optimizer step_pre_hook; key by
        # id(param) so the hook can label grads without a named_parameters() walk.
        if want_grad:
            _grad_plan[id(p)] = {
                "layer_name": layer_name, "param_name": pname,
                "grad_role": grad_role,
            }


def _scan_params() -> None:
    """Queue the planned VALUE (weight/bias) reductions for this step. Runs in the
    root pre-hook (start of step), so values are the PREVIOUS step's (the only
    sync-free read point for persistent params). Reductions join the same per-step
    drain -> no new host sync. Grads are NOT read here (zero_grad has run by now);
    they are read at the optimizer step_pre_hook instead — see _opt_step_pre_hook."""
    for entry in _scan_plan:
        _stash(entry["layer_name"], "param", entry["param"], role=entry["value_role"],
               param_name=entry["param_name"], from_prev_step=True)


def _opt_step_pre_hook(optimizer, *_args, **_kwargs) -> None:
    """Optimizer step_pre_hook: read param.grad AFTER backward, BEFORE the update
    (and its zero_grad). This is sync-free and OUTSIDE the autograd graph, so it
    does not perturb timing-sensitive behavior — unlike a per-param
    post-accumulate-grad hook. Only grads for watched params (in _grad_plan) are
    stashed; reductions join the same per-step drain. Grads are CURRENT-step
    (from_prev_step=False)."""
    global _opt_grad_stashed
    if not _grad_plan:
        return
    for group in optimizer.param_groups:
        for p in group["params"]:
            entry = _grad_plan.get(id(p))
            if entry is None:
                continue
            g = getattr(p, "grad", None)
            if torch.is_tensor(g):
                _stash(entry["layer_name"], "grad", g, role=entry["grad_role"],
                       param_name=entry["param_name"], from_prev_step=False)
                _opt_grad_stashed += 1


def _maybe_warn_grad() -> None:
    """Warn ONCE if grad channels were requested but produced nothing readable, so
    a quiet 'no gradients' is never mistaken for 'gradients are clean'. Two causes:
      - no Optimizer subclass was ever built (regime #3 / inline update), or
      - grads are freed during backward before the step_pre_hook (regime #1,
        optimizer-in-backward). Either way: use the weight/bias channels instead."""
    global _grad_warned
    if _grad_warned or not _GRAD_CHANNELS or _step < _GRAD_WARN_AFTER or _opt_grad_stashed > 0:
        return
    _grad_warned = True
    if not _opt_registered:
        _log(f"WARNING: grad channels {sorted(_GRAD_CHANNELS)} are enabled but no "
             "torch.optim.Optimizer was constructed, so param.grad cannot be read "
             "(custom or in-line optimizer update?). Use the weight/bias channels "
             "instead. No gradient records will be written.")
    else:
        _log(f"WARNING: grad channels {sorted(_GRAD_CHANNELS)} are enabled but no "
             f"gradients were readable at the optimizer step boundary in the first "
             f"{_GRAD_WARN_AFTER} steps. This happens when the optimizer update is "
             "fused into the backward pass (e.g. apply_optimizer_in_backward), which "
             "frees each gradient before the step boundary. Use the weight/bias "
             "channels instead. No gradient records will be written.")


def _root_pre_hook(_module, _inp):
    """Fires once per step at the START of the root forward. Drains the PREVIOUS
    step's pending reductions (all GPU work from last step is already queued),
    advances the step counter, then queues this step's param scan (prev-step
    values). The single host sync per step lives in the drain."""
    global _step, _phase, _layer_scan_idx
    _drain_step()
    _step += 1
    _phase = "forward"   # stage wrappers set copy/sparse phases outside forward
    _layer_scan_idx = 0
    # Scan the batch forward is about to read (batch N) at forward entry: emits the
    # phase='forward' embedding checkpoint (making forward a stage like copy/sparse)
    # and tags forward records with batch N's id. Also arms the per-layer re-scan
    # targets when NANLOG_TRACK_EVERY_LAYER is on. No-op unless NANLOG_PIPELINE.
    _capture_forward_batch(_inp)
    _scan_params()
    _maybe_warn_grad()


def _is_watched(layer_name: str, module: torch.nn.Module) -> bool:
    """A module is watched if its class name matches NANLOG_WATCH_TYPES OR its
    fully-qualified path contains any NANLOG_WATCH_NAMES substring (union)."""
    if type(module).__name__ in _WATCH_TYPES:
        return True
    return any(pat in layer_name for pat in _WATCH_NAMES)


def _attach(root: torch.nn.Module) -> int:
    """Register the enabled channels' hooks on every watched module (see
    _is_watched), and build the static param-scan plan. Returns the number of
    watched modules; also records their names in _watched_names for the summary."""
    n = 0
    want_fwd = bool(_CHANNELS & {"act", "input"})
    want_bwd = "igrad" in _CHANNELS
    for layer_name, module in root.named_modules():
        if not _is_watched(layer_name, module):
            continue
        if want_fwd:
            module.register_forward_hook(_fwd_hook(layer_name))
        if want_bwd:
            module.register_full_backward_hook(_bwd_hook(layer_name))
        if _PARAM_CHANNELS:
            _plan_param_scan(layer_name, module)
        _watched_names.append(layer_name)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Auto-attach: wrap DistributedModelParallel so the hooks bind to the real
# sharded model the moment it is built (no edit to the training script needed).
# ---------------------------------------------------------------------------
def _install_autohook() -> None:
    global _root_installed
    try:
        import torchrec.distributed.model_parallel as _mp
    except Exception as e:
        _log(f"could not import DistributedModelParallel to auto-hook: {e!r}")
        return
    _orig_dmp_init = _mp.DistributedModelParallel.__init__

    def _patched_init(self, *a, **k):
        _orig_dmp_init(self, *a, **k)
        global _root_installed
        if _root_installed:
            return
        _start_alloc_recording()
        n = _attach(self)
        # Drive one drain+step per training iteration off the root forward.
        self.register_forward_pre_hook(_root_pre_hook)
        _root_installed = True
        _log(f"attached layer hooks to {n} module(s) (types={list(_WATCH_TYPES)}, "
             f"names={list(_WATCH_NAMES)}); channels={sorted(_CHANNELS)}; "
             f"capture_addr={_CAPTURE_ADDR}; locate={_LOCATE}; "
             f"bad_values={_BAD_VALUES}; dump_tensor={_DUMP_TENSOR}; "
             f"alloc_snapshot={_ALLOC_SNAPSHOT}; "
             f"params_scanned={len(_scan_plan)}; params_skipped={len(_skipped_params)}; "
             f"per-step drain installed on DMP root")
        if 0 < n <= 10:
            _log(f"watched modules: {_watched_names}")

    _mp.DistributedModelParallel.__init__ = _patched_init
    _log("auto-hook armed (will attach when DistributedModelParallel is built)")


# ---------------------------------------------------------------------------
# Auto-attach (grad channels): wrap torch.optim.Optimizer.__init__ so EVERY
# optimizer (library or custom subclass) self-registers a step_pre_hook the moment
# it is built — no training-script edit, no concrete optimizer class name. Only
# armed when a grad channel is requested.
# ---------------------------------------------------------------------------
def _install_optimizer_autohook() -> None:
    if not _GRAD_CHANNELS:
        return
    _orig_opt_init = torch.optim.Optimizer.__init__

    def _patched_opt_init(self, *a, **k):
        _orig_opt_init(self, *a, **k)
        global _opt_registered
        try:
            if hasattr(self, "register_step_pre_hook"):
                self.register_step_pre_hook(_opt_step_pre_hook)
                _opt_registered = True
                _log(f"grad channels {sorted(_GRAD_CHANNELS)}: hooked optimizer "
                     f"{type(self).__name__} to read gradients")
            else:
                _log(f"WARNING: this PyTorch ({torch.__version__}) has no "
                     "Optimizer.register_step_pre_hook (added in 2.1); grad "
                     f"channels {sorted(_GRAD_CHANNELS)} are unavailable. Use the "
                     "weight/bias channels instead.")
        except Exception as e:  # a sidecar must never take the training run down
            _log(f"WARNING: could not hook optimizer {type(self).__name__} for "
                 f"grad channels: {e!r}")

    torch.optim.Optimizer.__init__ = _patched_opt_init
    _log(f"grad channels {sorted(_GRAD_CHANNELS)} enabled; will read gradients at "
         "the optimizer step boundary")


# ---------------------------------------------------------------------------
# Pipeline-stage tracking (NANLOG_PIPELINE): checkpoint tracked flow objects at
# the TorchRec pipeline stage boundaries, BEFORE forward reads them. General
# tensor-flow tracker — EF is only the default target (NANLOG_TRACK_ATTR).
# ---------------------------------------------------------------------------
_pipeline_installed = False
_pipeline_checkpoints = 0     # count of stage checkpoints that stashed something
_pipeline_warned = False


def _kjt_types():
    """Lazily resolve (KeyedJaggedTensor, JaggedTensor) classes; () if unavailable.
    Kept out of module import so a torchrec-less environment (e.g. the smoke test)
    still imports this logger cleanly."""
    try:
        from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
        return (KeyedJaggedTensor, JaggedTensor)
    except Exception:
        return ()


def _sparse_extra(obj) -> dict:
    """Cheap, host-side-only sparse distribution metadata for a KJT/JaggedTensor.
    No GPU sync unless NANLOG_SPARSE_HEAVY (empty_bags/max_bag_len/index min-max).
    Returns {} for non-sparse objects."""
    kjt_types = _kjt_types()
    if not kjt_types or not isinstance(obj, kjt_types):
        return {}
    extra = {"sparse": True}
    try:
        keys = obj.keys()
        extra["sparse_num_keys"] = len(keys)
    except Exception:
        pass
    for meth, tag in (("lengths", "lengths"), ("offsets", "offsets"), ("values", "values")):
        try:
            t = getattr(obj, meth)()
        except Exception:
            continue
        if not torch.is_tensor(t):
            continue
        extra[f"sparse_{tag}_shape"] = list(t.shape)
        extra[f"sparse_{tag}_dtype"] = str(t.dtype)
        if tag == "lengths":
            extra["sparse_total_lengths"] = int(t.numel())  # host-side, no sync
        if tag == "values":
            extra["sparse_values_device"] = str(t.device)
    if _SPARSE_HEAVY:
        # The ONE sparse path that reads back from device (explicit .item() sync).
        # Gated so it can never fire on a timing-sensitive repro run.
        try:
            lengths = obj.lengths()
            if torch.is_tensor(lengths) and lengths.numel():
                extra["sparse_empty_bags"] = int((lengths == 0).sum().item())
                extra["sparse_max_bag_len"] = int(lengths.max().item())
        except Exception:
            pass
        try:
            vals = obj.values()
            if torch.is_tensor(vals) and vals.numel():
                extra["sparse_index_min"] = int(vals.min().item())
                extra["sparse_index_max"] = int(vals.max().item())
        except Exception:
            pass
    return extra


def _emit_tracked(name: str, obj, seen: set) -> int:
    """Stash one tracked object (or its tensor members) with role track/sparse and
    the current phase. Recurses one level into list/tuple/dict. De-dupes by id so a
    tensor shared across containers is scanned once per checkpoint. Returns count."""
    if len(seen) >= _TRACK_MAX:
        return 0
    kjt_types = _kjt_types()
    n = 0
    if kjt_types and isinstance(obj, kjt_types):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        try:
            vals = obj.values()
        except Exception:
            vals = None
        role = "sparse" if _SPARSE else "track"
        extra = _sparse_extra(obj) if _SPARSE else None
        if torch.is_tensor(vals):
            _stash(name, "pipeline", vals, role=role, extra=extra)
            n += 1
        return n
    if torch.is_tensor(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        _stash(name, "pipeline", obj, role="track")
        return 1
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            if len(seen) >= _TRACK_MAX:
                break
            n += _emit_tracked(f"{name}[{i}]", item, seen)
        return n
    if isinstance(obj, dict):
        for k, item in obj.items():
            if len(seen) >= _TRACK_MAX:
                break
            n += _emit_tracked(f"{name}[{k}]", item, seen)
        return n
    return n


def _auto_discover(batch, seen: set) -> int:
    """Fallback when no NANLOG_TRACK_ATTR name is found on the batch: bounded walk
    (depth<=2) over attributes / container members collecting Tensors and KJTs."""
    n = 0
    kjt_types = _kjt_types()

    def visit(name, obj, depth):
        nonlocal n
        if len(seen) >= _TRACK_MAX:
            return
        if torch.is_tensor(obj) or (kjt_types and isinstance(obj, kjt_types)):
            n += _emit_tracked(name, obj, seen)
            return
        if depth <= 0:
            return
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                visit(f"{name}[{i}]", item, depth - 1)
        elif isinstance(obj, dict):
            for k, item in obj.items():
                visit(f"{name}[{k}]", item, depth - 1)
        else:
            for attr in getattr(obj, "__dict__", {}):
                if attr.startswith("_"):
                    continue
                try:
                    visit(f"{name}.{attr}", getattr(obj, attr), depth - 1)
                except Exception:
                    continue

    visit("batch", batch, 2)
    return n


def _resolve_batch_id(batch, phase: str):
    """Return the batch_id for this batch, host-side and sync-free.

    At the COPY phase a batch is entering the pipeline for the first time -> assign
    a fresh monotonic id and stamp it on the batch object (best effort) plus an
    id()-keyed fallback map for objects that reject attributes. At later phases the
    SAME batch is seen again -> read the id back so copy/sparse/forward of one batch
    share a key. Returns None if the batch can neither be tagged nor found (records
    then carry batch_id=null rather than a wrong guess)."""
    global _batch_counter
    # Prefer an id already carried on the object (set at its copy checkpoint).
    existing = getattr(batch, _BATCH_ID_ATTR, None)
    if existing is not None:
        return existing
    existing = _batch_id_by_obj.get(id(batch))
    if existing is not None:
        return existing
    if phase != "copy":
        # A non-copy phase with no id means we started mid-pipeline (missed this
        # batch's copy). Don't fabricate an id; leave it null.
        return None
    _batch_counter += 1
    bid = _batch_counter
    try:
        object.__setattr__(batch, _BATCH_ID_ATTR, bid)
    except Exception:
        # Object rejects attributes (e.g. __slots__, frozen). Use the id() map,
        # bounded so a long run cannot grow it without limit.
        if len(_batch_id_by_obj) > 4096:
            _batch_id_by_obj.clear()
        _batch_id_by_obj[id(batch)] = bid
    return bid


def _collect_tracked(batch, seen: set, out: list) -> None:
    """Collect the tracked flow objects on `batch` as (name, tensor) pairs into
    `out`, resolving a KJT to its .values() tensor. Same discovery order as
    _checkpoint (named NANLOG_TRACK_ATTR first, else bounded auto-discovery) but it
    RETURNS the tensors instead of stashing them, so the per-layer re-scan can hold
    the object references and re-emit them at each layer. Host-side, no reduction."""
    kjt_types = _kjt_types()

    def take(name, obj):
        if len(seen) >= _TRACK_MAX:
            return
        if kjt_types and isinstance(obj, kjt_types):
            if id(obj) in seen:
                return
            seen.add(id(obj))
            try:
                v = obj.values()
            except Exception:
                v = None
            if torch.is_tensor(v):
                out.append((name, v))
            return
        if torch.is_tensor(obj):
            if id(obj) in seen:
                return
            seen.add(id(obj))
            out.append((name, obj))
            return
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                take(f"{name}[{i}]", item)
        elif isinstance(obj, dict):
            for k, item in obj.items():
                take(f"{name}[{k}]", item)

    found_named = False
    for name in _TRACK_ATTR:
        obj = None
        if hasattr(batch, name):
            obj = getattr(batch, name)
        elif isinstance(batch, dict) and name in batch:
            obj = batch[name]
        if obj is not None:
            found_named = True
            take(name, obj)
    if not found_named:
        # Reuse the checkpoint auto-discovery, but collect into `out`. Cheapest to
        # just walk attributes here rather than thread a callback through.
        def visit(name, obj, depth):
            if len(seen) >= _TRACK_MAX:
                return
            if torch.is_tensor(obj) or (kjt_types and isinstance(obj, kjt_types)):
                take(name, obj)
                return
            if depth <= 0:
                return
            if isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    visit(f"{name}[{i}]", item, depth - 1)
            elif isinstance(obj, dict):
                for k, item in obj.items():
                    visit(f"{name}[{k}]", item, depth - 1)
            else:
                for attr in getattr(obj, "__dict__", {}):
                    if attr.startswith("_"):
                        continue
                    try:
                        visit(f"{name}.{attr}", getattr(obj, attr), depth - 1)
                    except Exception:
                        continue
        visit("batch", batch, 2)


def _capture_forward_batch(inp) -> None:
    """From the root forward's positional args, find the batch being processed
    (batch N, the one forward actually reads — distinct from the copy(N+2)/
    sparse(N+1) batches the stage wrappers just saw) and scan its tracked ef objects
    at forward ENTRY (phase='forward'), so forward is a first-class stage alongside
    copy/sparse — no watched module or per-layer re-scan required. When per-layer
    re-scan IS on, also hold the tracked objects in _layer_scan_targets so each layer
    hook can re-scan the SAME objects. Host-side only; the reductions join the drain.

    The forward-entry _checkpoint resolves _batch_id to batch N. This is REQUIRED
    whenever pipeline tracking is on: otherwise the forward/layer records inherit the
    stale _batch_id the LAST stage wrapper left set (start_sparse_data_dist for batch
    N+1 in the standard prefetch order), tagging the emb_proj input record where
    first_bad fires with the WRONG batch. _batch_id ends up batch N (or None on a
    mid-pipeline start / re-wrapped batch), never a stale stage value."""
    global _layer_scan_targets
    _layer_scan_targets = []
    if not _PIPELINE:
        return
    # The batch is usually the first positional arg to the root forward.
    batch = inp[0] if isinstance(inp, (tuple, list)) and inp else inp
    if batch is None:
        return
    try:
        # Forward-entry checkpoint: scan the 31 ef objects once here (sets _batch_id
        # to batch N and _phase='forward'). Makes 'forward' a stage like copy/sparse
        # without needing NANLOG_WATCH_NAMES or NANLOG_TRACK_EVERY_LAYER.
        _checkpoint(batch, "forward")
        if _TRACK_EVERY_LAYER:
            seen: set = set()
            targets: list = []
            _collect_tracked(batch, seen, targets)
            _layer_scan_targets = targets
    except Exception as e:
        _log(f"WARNING: forward-batch capture failed: {e!r}")


def _checkpoint(batch, phase: str) -> None:
    """Set the phase + batch_id, discover the tracked flow objects on `batch`, and
    stash a re-scan of each. Sync-free (reductions join the per-step drain). Called
    from the wrapped pipeline stage methods. Never raises into the training run."""
    global _phase, _batch_id, _pipeline_checkpoints
    if batch is None:
        return
    _phase = phase
    try:
        _batch_id = _resolve_batch_id(batch, phase)
    except Exception:
        _batch_id = None
    seen: set = set()
    n = 0
    try:
        found_named = False
        for name in _TRACK_ATTR:
            obj = None
            if hasattr(batch, name):
                obj = getattr(batch, name)
            elif isinstance(batch, dict) and name in batch:
                obj = batch[name]
            if obj is not None:
                found_named = True
                n += _emit_tracked(name, obj, seen)
        if not found_named:
            n += _auto_discover(batch, seen)
    except Exception as e:
        _log(f"WARNING: pipeline checkpoint ({phase}) failed: {e!r}")
        return
    if n:
        _pipeline_checkpoints += 1


def _install_pipeline_hook() -> None:
    """Monkeypatch TrainPipelineSparseDist stage methods so each fires a sync-free
    checkpoint of the tracked flow objects with the right `phase`. Defensive: if the
    installed torchrec exposes none of the known stage methods, warn once and leave
    the layer hooks untouched. Same auto-attach philosophy as the DMP __init__ patch."""
    global _pipeline_installed, _pipeline_warned
    if not _PIPELINE or _pipeline_installed:
        return
    try:
        from torchrec.distributed.train_pipeline import TrainPipelineSparseDist as _TP
    except Exception as e:
        _pipeline_warned = True
        _log(f"WARNING: NANLOG_PIPELINE=1 but TrainPipelineSparseDist could not be "
             f"imported ({e!r}); pipeline checkpoints inactive. Layer hooks unaffected.")
        return

    # (method name on the pipeline, phase, index of the batch arg in *a, whether
    # the batch is the RETURN value instead of an argument).
    #   copy_batch_to_gpu(self, dataloader_iter) -> batch      (batch = return)
    #   start_sparse_data_dist(self, batch, ...)               (batch = a[0])
    #   wait_sparse_data_dist(self, batch, ...) OR (self)       (a[0] if present)
    specs = [
        ("copy_batch_to_gpu", "copy", True),
        ("start_sparse_data_dist", "sparse_start", False),
        ("wait_sparse_data_dist", "sparse_wait", False),
    ]
    patched = []
    for meth_name, phase, from_return in specs:
        orig = getattr(_TP, meth_name, None)
        if orig is None or not callable(orig):
            continue

        def make(orig, phase, from_return):
            def wrapper(self, *a, **k):
                result = orig(self, *a, **k)
                try:
                    if from_return:
                        # copy_batch_to_gpu returns the batch, or a (batch, context)
                        # tuple across torchrec versions; unwrap the first element
                        # only when the named track attr is not on the tuple itself.
                        batch = result
                        if (isinstance(batch, tuple) and batch
                                and not any(hasattr(batch, nm) for nm in _TRACK_ATTR)):
                            batch = batch[0]
                    else:
                        batch = a[0] if a else None
                    _checkpoint(batch, phase)
                except Exception as e:
                    _log(f"WARNING: pipeline wrapper ({phase}) error: {e!r}")
                return result
            return wrapper

        setattr(_TP, meth_name, make(orig, phase, from_return))
        patched.append(meth_name)

    if not patched:
        _pipeline_warned = True
        _log(f"WARNING: NANLOG_PIPELINE=1 but TrainPipelineSparseDist has none of the "
             f"known stage methods (copy_batch_to_gpu/start_sparse_data_dist/"
             f"wait_sparse_data_dist); checkpoints inactive. torchrec API changed?")
        return
    _pipeline_installed = True
    _log(f"pipeline tracking armed: patched {patched}; track_attr={list(_TRACK_ATTR)}; "
         f"track_max={_TRACK_MAX}; sparse={_SPARSE}; sparse_heavy={_SPARSE_HEAVY}; "
         f"track_every_layer={_TRACK_EVERY_LAYER}; track_layer_stride={_TRACK_LAYER_STRIDE}")


def _write_summary() -> None:
    """Write the end-of-run summary (first bad layer + totals)."""
    _drain_step()  # flush the final step
    summary = {
        "rank": _RANK, "host": _HOST, "pid": _PID,
        "steps_seen": _step,
        "records_written": _records_written,
        "matmul_calls_total": _matmul_calls,
        "first_bad": _first_bad,
        "huge_threshold": _HUGE,
        "pre_context": _PRE_CONTEXT,
        "pre_context_flushed": _pre_flushed,
        "watch_types": list(_WATCH_TYPES),
        "watch_names": list(_WATCH_NAMES),
        "watched_count": len(_watched_names),
        "watched_names": list(_watched_names),
        "channels": sorted(_CHANNELS),
        "capture_addr": _CAPTURE_ADDR,
        "locate": _LOCATE,
        "bad_values": _BAD_VALUES,
        "dump_tensor": _DUMP_TENSOR,
        "dump_tensor_dumped": _tensor_dumped,
        "alloc_snapshot": _ALLOC_SNAPSHOT,
        "alloc_snapshot_dumped": _snapshot_dumped,
        "pipeline": _PIPELINE,
        "pipeline_installed": _pipeline_installed,
        "pipeline_checkpoints": _pipeline_checkpoints,
        "track_attr": list(_TRACK_ATTR),
        "track_max": _TRACK_MAX,
        "sparse": _SPARSE,
        "sparse_heavy": _SPARSE_HEAVY,
        "track_every_layer": _TRACK_EVERY_LAYER,
        "track_layer_stride": _TRACK_LAYER_STRIDE,
        "bounds": [{"pattern": s, "lo": lo, "hi": hi} for s, lo, hi in _BOUNDS],
        "bound_global": (list(_BOUND_GLOBAL) if _BOUND_GLOBAL is not None else None),
        "first_oob": _first_oob,
        "oob_records": _oob_records,
        "peak_finite_max": _peak_finite_max,
        "peak_finite_min": _peak_finite_min,
        "batches_seen": _batch_counter,
        "max_param_numel": _MAX_PARAM_NUMEL,
        "params_scanned": len(_scan_plan),
        "grad_params_planned": len(_grad_plan),
        "grad_records_stashed": _opt_grad_stashed,
        "optimizer_hook_registered": _opt_registered,
        "skipped_params": _skipped_params,
        "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "jsonl": str(_JSONL),
        "notes": (
            "first_bad is the first layer/step that went NaN/Inf/huge, or null if "
            "none was seen. matmul_calls_so_far counts watched layer tensors (not "
            "raw GEMMs); it can be lined up with a separate GEMM-output tool's "
            "'call #N' if one is also running."
        ),
    }
    try:
        with _SUMMARY.open("w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        _log(f"summary -> {_SUMMARY} (first_bad={_first_bad})")
    except Exception as e:
        _log(f"summary write failed: {e!r}")


# ---------------------------------------------------------------------------
# Allocator snapshot (NANLOG_ALLOC_SNAPSHOT)
# ---------------------------------------------------------------------------
_snapshot_dumped = False
_tensor_dumped = False


_alloc_recording_started = False


def _start_alloc_recording() -> None:
    """Enable allocator event recording. Safe to call multiple times (idempotent).

    Deferred to the DMP __init__ hook so CUDA is guaranteed to be initialized.
    Calling at import time would fail with 'No CUDA GPUs available' because
    torch.cuda is not yet set up."""
    global _alloc_recording_started
    if not _ALLOC_SNAPSHOT or _alloc_recording_started:
        return
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", stacks="python", max_entries=500000)
        _alloc_recording_started = True
        _log("allocator snapshot: recording enabled (stacks=python, "
             "max_entries=500000). Overhead: ~10% per step from stack capture; "
             "GPU kernel timing unchanged.")
    except Exception as e:
        _log(f"WARNING: allocator snapshot: could not enable recording: {e!r}. "
             "NANLOG_ALLOC_SNAPSHOT will be inactive.")


def _dump_bad_tensor(rec: dict, t: torch.Tensor) -> None:
    """Save the full bad tensor to disk on first detection. One-shot."""
    global _tensor_dumped
    if _tensor_dumped or not _DUMP_TENSOR:
        return
    _tensor_dumped = True
    layer_safe = rec["layer_name"].replace(".", "_").replace("/", "_")
    pt_path = _DIR / f"bad_tensor_step{rec['step']}_{layer_safe}_{rec['role']}_rank{_RANK}.pt"
    try:
        torch.save(t.detach().cpu(), pt_path)
        _log(f"dump_tensor -> {pt_path} (shape={list(t.shape)}, dtype={t.dtype}, "
             f"{t.numel() * t.element_size() / 1024:.0f} KB)")
    except Exception as e:
        _log(f"WARNING: dump_tensor failed: {e!r}")


def _dump_alloc_snapshot(step: int) -> None:
    """Dump the allocator snapshot on first NaN detection, then stop recording."""
    global _snapshot_dumped
    if _snapshot_dumped or not _ALLOC_SNAPSHOT:
        return
    _snapshot_dumped = True
    snap_path = _DIR / f"alloc_snapshot_step{step}_rank{_RANK}.pickle"
    try:
        torch.cuda.memory._dump_snapshot(str(snap_path))
        _log(f"allocator snapshot -> {snap_path}")
    except Exception as e:
        _log(f"WARNING: allocator snapshot dump failed: {e!r}")
    try:
        torch.cuda.memory._record_memory_history(enabled=None)
        _log("allocator snapshot: recording stopped after dump")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Install + run target
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import atexit
    atexit.register(_write_summary)

    if len(sys.argv) < 2:
        _log("usage: python instrument_nan_logger.py <target_script.py> [args...]")
        sys.exit(2)
    _install_autohook()
    _install_optimizer_autohook()
    _install_pipeline_hook()
    target = Path(sys.argv[1]).resolve()
    sys.argv = [str(target)] + sys.argv[2:]
    _log(f"running target {target}")
    runpy.run_path(str(target), run_name="__main__")
