"""A NaN reproducer whose two halves are both mechanisms measured on this box.

The story is a compound bug, and a real one rather than a demonstration of an
API:

  Phase 1 -- an fp16 stack with a hot initialisation overflows. Activations
    grow ~2.5x per layer and pass fp16's 65504 ceiling at layer 11, so the
    tensor left at the end of the phase is full of +/-Inf. Nothing raises: an
    unscaled fp16 forward that overflows is silent, which is why loss scaling
    exists. The per-layer peaks are printed, so this is measured rather than
    asserted.

  Phase 2 -- a later step preallocates a workspace with ``torch.empty`` to
    accumulate microbatch gradients, and an off-by-one in the slice bounds
    leaves the last slot unwritten. The caching allocator satisfies that
    same-sized allocation with the phase-1 block, contents intact, so the
    unwritten slot is the overflowed activation and the loss goes NaN a long
    way from anything numeric.

What is authored, stated plainly so nobody has to infer it
----------------------------------------------------------
Neither half is a fault injected to make a number look good: phase 1 is how
fp16 training actually dies, and the uninitialised workspace is common enough
to be a genre. What is authored is their *composition*, and specifically the
step that makes the block reuse deterministic. In the wild, which freed block
a ``torch.empty`` lands on is a lottery over whatever the allocator is holding
-- and that lottery was measured here: with the phase-1 weights still cached,
the request is served from a coalesced ex-weight region whose fp16 bit
patterns read as fp32 denormals, so the workspace looks clean and the
reproducer silently passes. That happened 24 times out of 24 across two
attempts before this file pinned it.

The pinning is ``_plant_residue``: the overflowed activation is parked on the
host, the device allocator is emptied, and the residue is then materialised
into an otherwise-empty pool. Phase 2's request is the same size, so it gets
that block and nothing else. A reproducer is allowed to pin a lottery -- that
is most of what makes it a reproducer -- but it is not allowed to do so
quietly, hence this paragraph.

The mitigation
--------------
``pytorch_no_cuda_memory_caching`` (PYTORCH_NO_CUDA_MEMORY_CACHING=1) resolves
this, causally: with caching off the allocation goes to hipMalloc and does not
carry the previous block's contents. Twenty other registered mitigations were
measured against the same mechanism and none of them change the outcome, so
naming the resolver is a real discrimination rather than a coin flip.

The lesson it encodes is true and transferable: a NaN that disappears when you
disable the caching allocator is not a numerics bug, it is a read of memory
nobody wrote.
"""

from __future__ import annotations

import json
import os
import sys

import torch

DEV = "cuda"
WIDTH = 4096
ROWS = 1024          # phase-1 activation is ROWS x WIDTH
MICRO = 8            # phase-2 workspace is MICRO slots covering ROWS*WIDTH
DEPTH = 12
GAIN = 2.5           # per-layer growth factor; 2.5 reaches the fp16 ceiling at 11


def phase1_overflow():
    """Unscaled fp16 forward with a hot init. Returns peaks and the Inf block."""
    torch.manual_seed(0)
    layers = []
    for _ in range(DEPTH):
        lin = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        torch.nn.init.normal_(lin.weight, std=GAIN / (WIDTH**0.5))
        layers.append(lin)
    model = torch.nn.Sequential(*layers).to(DEV).half()

    peaks: list[float] = []
    with torch.no_grad():
        h = torch.randn(ROWS, WIDTH, device=DEV, dtype=torch.float16)
        for layer in model:
            h = layer(h)
            peaks.append(float(h.abs().max().item()))
            if not torch.isfinite(h).all():
                break
        overflow_host = h.float().reshape(-1).cpu()

    del model, layers, h
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return peaks, overflow_host


def _plant_residue(overflow_host):
    """Put the Inf block into an otherwise-empty pool, then free it.

    See the module docstring: this is the pinning step, and it is the reason
    the scenario has a stable verdict instead of a rate.
    """
    residue = overflow_host.to(DEV)
    nonfinite = int((~torch.isfinite(residue)).sum().item())
    reserved = int(torch.cuda.memory_reserved())
    del residue
    torch.cuda.synchronize()
    return nonfinite, reserved


def phase2_workspace():
    """Accumulate microbatch grads into a torch.empty workspace, one slot short."""
    torch.manual_seed(1)
    total = ROWS * WIDTH
    slot = total // MICRO
    # The bug: torch.empty, then a loop bound that stops one slot early.
    workspace = torch.empty(total, device=DEV, dtype=torch.float32)
    for i in range(MICRO - 1):
        workspace[i * slot:(i + 1) * slot] = torch.randn(slot, device=DEV) * 0.01
    loss = workspace.pow(2).mean()
    nonfinite = int((~torch.isfinite(workspace)).sum().item())
    return float(loss.item()), nonfinite, total


def main() -> int:
    trial = os.environ.get("AORTA_TRIAL", "?")
    print(f"[nan-workspace] trial={trial} torch={torch.__version__} "
          f"hip={torch.version.hip} "
          f"gpu={torch.cuda.get_device_properties(0).gcnArchName}", flush=True)
    print(f"[nan-workspace] PYTORCH_NO_CUDA_MEMORY_CACHING="
          f"{os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')!r} "
          f"DISABLE_TF32={os.environ.get('DISABLE_TF32')!r}", flush=True)

    peaks, overflow_host = phase1_overflow()
    shown = [round(p, 1) if p == p and p != float("inf") else p for p in peaks]
    print(f"[nan-workspace] phase 1: fp16 peaks per layer = {shown}", flush=True)

    if os.environ.get("NAN_WS_MODE") == "overflow_only":
        # Phase 1 on its own, treated as the failure. This is the honest
        # negative companion to the main scenario: the fp16 overflow is just
        # as real, and no registered mitigation touches it, because overflow
        # is an exponent problem and the registry's levers are queues,
        # transports and allocators. A scenario nothing resolves is what
        # fix_reward withholds rather than scores, and having a real one is
        # worth more than assuming the withhold path works.
        overflowed = any(p != p or p == float("inf") for p in peaks)
        if overflowed:
            print(f"[nan-workspace] loss=nan  (unscaled fp16 forward overflowed at "
                  f"layer {len(peaks)}; peak passed 65504)", flush=True)
        print("NAN_WS_JSON " + json.dumps({
            "finite": not overflowed, "mode": "overflow_only",
            "phase1_peaks": [p if p == p and p != float("inf") else None for p in peaks],
            "phase1_overflowed": overflowed, "trial": trial,
            "no_caching": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
        }), flush=True)
        return 1 if overflowed else 0

    planted, reserved = _plant_residue(overflow_host)
    print(f"[nan-workspace] planted a {overflow_host.numel()}-element fp32 block, "
          f"{planted} non-finite, reserved={reserved} bytes; freed it", flush=True)

    loss, nonfinite, total = phase2_workspace()
    finite = nonfinite == 0 and loss == loss and abs(loss) != float("inf")

    if finite:
        print(f"[nan-workspace] phase 2: loss={loss:.6g}  workspace clean", flush=True)
    else:
        # Wording chosen to match the built-in tier4:nan_signature detector,
        # which looks for `loss(?: is)? NaN|loss=nan`.
        print(f"[nan-workspace] phase 2: loss=nan  ({nonfinite} of {total} workspace "
              f"elements non-finite; the unwritten slot came back holding the "
              f"phase-1 residue)", flush=True)

    print("NAN_WS_JSON " + json.dumps({
        "finite": finite,
        "loss": None if (loss != loss or abs(loss) == float("inf")) else loss,
        "nonfinite_workspace_elems": nonfinite,
        "workspace_elems": total,
        "unwritten_slot_elems": total // MICRO,
        "phase1_peaks": [p if p == p and p != float("inf") else None for p in peaks],
        "phase1_overflowed": any(p != p or p == float("inf") for p in peaks),
        "planted_nonfinite": planted,
        "reserved_at_plant": reserved,
        "no_caching": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING"),
        "trial": trial,
    }), flush=True)
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
