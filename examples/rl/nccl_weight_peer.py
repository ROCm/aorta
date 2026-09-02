#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Trainer-side peer for TokenSpeed's ``nccl`` weight-transfer backend.

TokenSpeed's engine is only the *receiving* half of the weight-update path. The
sending half is the trainer's, and it is not shipped: the engine's control plane
exposes ``/init_weight_transfer_engine`` + ``/update_weights`` and then blocks on
a ``dist.broadcast(..., src=0)`` that something else has to post. This script is
that something else, reduced to the smallest thing that is protocol-correct, so
the receive path can be exercised without standing up a real trainer.

Why not ``torch.distributed.init_process_group``
------------------------------------------------
The engine does *not* join the default process group. It builds a standalone one
via ``_new_process_group_helper`` over a ``PrefixStore(group_name, ...)`` (see
``runtime/execution/model_runner.py::init_weights_update_group``). Torch's own
``init_process_group`` prefixes the store with ``"default_pg"`` instead, so a
trainer that uses it rendezvouses against different store keys, never exchanges
the NCCL unique id, and both sides hang until the 30-minute timeout. The group
construction here mirrors the engine's line for line, including the
``backend_options``/``pg_options`` rename at torch 2.6, because that is the
contract -- not because a private API is a nice thing to depend on.

Rank layout
-----------
The trainer is rank 0 and owns the TCP store, so it must be running before the
engine's ``/init_weight_transfer_engine`` call. Engine worker ``i`` takes rank
``rank_offset + i``, so with ``rank_offset=1`` a TP=N engine occupies ranks
``1..N`` and ``world_size`` is ``N + 1``.

Tensor shapes are the *unsharded* checkpoint shapes at every TP degree: the
receive side feeds the broadcast tensors to the model's own ``load_weights``,
which applies the same sharding it applies to an initial load. The sender does
not shard.

Broadcasts are an ordered collective, so the tensors must be sent in exactly the
order the ``names`` in ``update_info`` list them. Each round here posts its
broadcasts and blocks; they complete when the engine posts the matching receives
in response to its own ``/update_weights`` call, which is what lets the driver
sequence HTTP and NCCL without a side channel.

Runs inside the TokenSpeed image (it needs that image's torch + ROCm), on a GPU
the engine does not have, which is the disaggregated topology ``nccl`` implies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

_DTYPE_NAMES = {
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.int8": "int8",
    "torch.uint8": "uint8",
    "torch.int32": "int32",
    "torch.int64": "int64",
}


def log(msg: str) -> None:
    print(f"[peer] {msg}", flush=True)


def load_checkpoint_tensors(model_path: str, wanted: list[str]) -> dict[str, Any]:
    """Read named tensors out of a HF safetensors checkpoint on the host."""
    from safetensors.torch import load_file

    shards = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors under {model_path}")

    found: dict[str, Any] = {}
    for shard in shards:
        blob = load_file(os.path.join(model_path, shard))
        for name in wanted:
            if name in blob and name not in found:
                found[name] = blob[name]
        if len(found) == len(wanted):
            break

    missing = [n for n in wanted if n not in found]
    if missing:
        raise SystemExit(f"tensors absent from checkpoint: {missing}")
    return found


def join_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    group_name: str,
    device_index: int,
):
    """Form the engine's weight-update group as rank 0.

    Mirrors ``ModelRunner.init_weights_update_group`` exactly; see the module
    docstring for why this cannot be ``init_process_group``.
    """
    import torch
    from packaging.version import parse as parse_version
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    init_method = f"tcp://{master_address}:{master_port}"
    log(f"rendezvous {init_method} rank={rank} world_size={world_size}")
    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=default_pg_timeout)
    )
    store.set_timeout(default_pg_timeout)
    store = PrefixStore(group_name, store)

    opt = (
        "backend_options"
        if parse_version(torch.__version__) >= parse_version("2.6")
        else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        Backend("nccl"),
        store,
        group_name=group_name,
        **{opt: None},
        timeout=default_pg_timeout,
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    log(f"group '{group_name}' formed on {device}")
    return pg, device


def build_round(kind: str, originals: dict[str, Any], names: list[str], device):
    """Materialise one round's payload on the sender's device.

    ``perturb`` zeroes every tensor, which is the most detectable possible
    change: a receive path that silently drops the payload cannot produce the
    same completions as one that applies it. ``restore`` sends the checkpoint
    values back, which is the stronger half of the test -- it distinguishes a
    faithful transfer from one that merely corrupts something.
    """
    payload = []
    for name in names:
        original = originals[name].to(device)
        if kind == "perturb":
            payload.append(original.clone().zero_())
        elif kind == "restore":
            payload.append(original)
        elif kind == "recv":
            # Diagnostic: swap roles and make the peer the receiver, to tell
            # "the engine posts no collective" from "the engine posts one but
            # as the root". Poison the buffer with a value the engine cannot
            # plausibly send, so an unchanged buffer means nothing arrived.
            poisoned = original.clone()
            poisoned.fill_(-1.0)
            payload.append(poisoned)
        else:
            raise SystemExit(f"unknown round kind: {kind}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master-address", default="127.0.0.1")
    ap.add_argument("--master-port", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument(
        "--src",
        type=int,
        default=None,
        help="broadcast root (defaults to --rank, i.e. this peer sends)",
    )
    ap.add_argument("--group-name", default="weight_update_group")
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--model-path", required=True)
    ap.add_argument(
        "--tensors",
        required=True,
        help="comma-separated checkpoint tensor names to broadcast",
    )
    ap.add_argument(
        "--rounds",
        default="perturb,restore",
        help="comma-separated round kinds, each consuming one /update_weights",
    )
    ap.add_argument(
        "--plan-out",
        required=True,
        help="where to write the update_info metadata the driver must POST",
    )
    args = ap.parse_args()
    if args.src is None:
        args.src = args.rank

    import torch
    import torch.distributed as dist

    names = [n for n in args.tensors.split(",") if n]
    rounds = [r for r in args.rounds.split(",") if r]

    originals = load_checkpoint_tensors(args.model_path, names)

    # The driver needs names/dtypes/shapes before it can call /update_weights,
    # and it cannot call that until the group exists -- so publish the plan
    # first, then block in rendezvous waiting for the engine to join.
    plan = {
        "names": names,
        "dtype_names": [_DTYPE_NAMES[str(originals[n].dtype)] for n in names],
        "shapes": [list(originals[n].shape) for n in names],
    }
    with open(args.plan_out, "w") as fh:
        json.dump(plan, fh, indent=2)
    log(f"plan written to {args.plan_out}: {len(names)} tensor(s)")
    for name in names:
        log(f"  {name} {tuple(originals[name].shape)} {originals[name].dtype}")

    pg, device = join_group(
        args.master_address,
        args.master_port,
        args.rank,
        args.world_size,
        args.group_name,
        args.device_index,
    )

    for index, kind in enumerate(rounds, start=1):
        payload = build_round(kind, originals, names, device)
        log(f"round {index}/{len(rounds)} ({kind}): posting {len(payload)} broadcast(s)")
        started = time.time()
        # Per-tensor sync, so a receiver that consumes fewer tensors than the
        # plan lists shows up here as the exact broadcast that never matched
        # rather than as one opaque hang. Costs a sync per tensor; this is a
        # diagnostic sender, and knowing *which* tensor desynchronised is worth
        # more than the throughput.
        for name, tensor in zip(names, payload, strict=True):
            posted = time.time()
            before = tensor.flatten()[:4].clone() if kind == "recv" else None
            dist.broadcast(tensor, src=args.src, group=pg)
            torch.cuda.synchronize(device)
            verb = "recv" if kind == "recv" else "sent"
            log(f"  {verb} {name} in {time.time() - posted:.3f}s")
            if before is not None:
                after = tensor.flatten()[:4]
                arrived = not torch.equal(before, after)
                log(f"    buffer changed: {arrived} "
                    f"(poison {before.tolist()} -> {after.tolist()})")
        log(f"round {index} ({kind}) drained in {time.time() - started:.3f}s")
        # A marker per round lets the driver tell "the sender finished this
        # round" from "the receiver claimed it did".
        with open(f"{args.plan_out}.round{index}.done", "w") as fh:
            fh.write(kind)

    dist.destroy_process_group(pg)
    log("group destroyed; peer exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
