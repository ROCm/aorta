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


# The round kinds `build_round` knows how to materialise. Named here so the
# argument check and the dispatch below read the same list: a kind added to one
# and not the other is the drift this constant exists to prevent.
_ROUND_KINDS = ("perturb", "restore", "recv")

# Which end of the broadcast each kind puts this peer on. A round's kind fixes
# its role -- `perturb` and `restore` exist to push tensors at the engine, and
# `recv` exists to make the engine push one at us -- so `--src` is not free to
# contradict it. It silently could: `--src` defaults to `--rank`, which makes
# this peer the broadcast root, and for `recv` that means it *sends* the
# poisoned buffer it was supposed to have received into. The buffer then comes
# back unchanged, because the root's buffer is the source, and the diagnostic
# reports "nothing arrived" for a collective that worked perfectly.
#
# Both directions are checked rather than just the reported one. The mirror --
# `--src` pointing away from a sending round -- makes this peer a receiver on a
# round it logs as "sent", which is the same defect with the roles swapped and
# just as quiet.
_SENDING_ROUND_KINDS = frozenset({"perturb", "restore"})
_RECEIVING_ROUND_KINDS = frozenset({"recv"})


def check_round_roles(rounds: list[str], rank: int, src: int) -> str | None:
    """Why this ``--src`` cannot serve these rounds, or ``None`` if it can.

    Returns a message rather than raising so the caller decides the exit
    route, and so a test can reach every branch without a process group.
    """
    sending = [k for k in rounds if k in _SENDING_ROUND_KINDS]
    receiving = [k for k in rounds if k in _RECEIVING_ROUND_KINDS]
    if sending and receiving:
        # One `--src` cannot be both this peer and not this peer, so this is a
        # contradiction in the request rather than a wrong value to correct.
        return (
            f"--rounds mixes sending rounds {sorted(set(sending))} with "
            f"receiving rounds {sorted(set(receiving))}; one --src cannot be "
            "this peer for some rounds and the engine for others. Run them as "
            "separate invocations."
        )
    if receiving and src == rank:
        return (
            f"--rounds {receiving} receives, so the broadcast root must be an "
            f"engine rank, but --src resolved to {src}, which is this peer's "
            "own --rank. This peer would broadcast the poisoned buffer instead "
            "of receiving into it, and report that nothing arrived even when "
            "the collective works. Pass --src <engine rank>, e.g. the "
            "--rank-offset the driver uses."
        )
    if sending and src != rank:
        return (
            f"--rounds {sorted(set(sending))} sends, so this peer must be the "
            f"broadcast root, but --src is {src} and --rank is {rank}. This "
            "peer would receive into the payload it believes it is sending."
        )
    return None


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
            # Unreachable from the CLI, which validates against `_ROUND_KINDS`
            # before joining the group. Kept as a backstop for a programmatic
            # caller, and deliberately not the only check: reaching it means a
            # peer that has already rendezvoused exits without posting, which is
            # what made a typo look like a transport hang.
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
        help="broadcast root (defaults to --rank, i.e. this peer sends). "
             "Required for a `recv` round, where the root must be an engine "
             "rank instead",
    )
    # Stamped into the plan so the driver can tell this run's plan from a
    # previous run's. The two processes are launched by hand against a fixed
    # shared path (`--plan-out` here, `--plan` there), so a leftover file is
    # the normal state of that directory rather than an unusual one.
    ap.add_argument("--run-id", required=True,
                    help="token shared with the driver's --plan-run-id")
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

    # Validated here, before the torch import and before the rendezvous, rather
    # than inside `build_round` where it used to be. `build_round` is called
    # only after `join_group` has returned, so a typo in `--rounds` was rejected
    # *after* this peer had joined the process group -- and then exited without
    # posting the broadcast the driver's `/update_weights` call is blocking on,
    # so the driver hung for its full timeout on what is really a command-line
    # error, with the engine left mid-update. A misspelled argument should cost
    # nothing, and it now costs nothing: no import, no store, no group.
    rounds = [r for r in args.rounds.split(",") if r]
    if not rounds:
        raise SystemExit("--rounds is empty; expected " + ",".join(_ROUND_KINDS))
    unknown = [r for r in rounds if r not in _ROUND_KINDS]
    if unknown:
        raise SystemExit(
            f"unknown round kind(s) {unknown}; expected any of "
            + ", ".join(_ROUND_KINDS)
        )
    # Checked here too, and for the same reason the kind check moved here: a
    # role mismatch is a command-line error, and paying for it after the
    # rendezvous means the driver blocks on an /update_weights whose broadcast
    # is never posted. The difference is that an unknown kind at least exits;
    # this one used to run to completion and publish a wrong answer.
    role_error = check_round_roles(rounds, args.rank, args.src)
    if role_error:
        raise SystemExit(role_error)

    # An empty tensor list is not a small plan, it is a forged result. The peer
    # would join the group, broadcast nothing, write its round markers and exit
    # 0 -- and those markers are what the driver now reads as proof a collective
    # was matched, so a no-op plan manufactures exactly the evidence that is
    # meant to be unfakeable. Harmless while the driver ignored the markers; not
    # harmless now that it reads them. Checked up here with the other argument
    # errors, so it costs no import and no rendezvous.
    names = [n for n in args.tensors.split(",") if n]
    if not names:
        raise SystemExit(
            "--tensors is empty; a plan with no tensors broadcasts nothing "
            "while still reporting its rounds complete"
        )

    import torch
    import torch.distributed as dist

    originals = load_checkpoint_tensors(args.model_path, names)

    # The driver needs names/dtypes/shapes before it can call /update_weights,
    # and it cannot call that until the group exists -- so publish the plan
    # first, then block in rendezvous waiting for the engine to join.
    plan = {
        "run_id": args.run_id,
        "names": names,
        "dtype_names": [_DTYPE_NAMES[str(originals[n].dtype)] for n in names],
        "shapes": [list(originals[n].shape) for n in names],
    }
    # Published by rename, because the driver treats the file *existing* as the
    # signal that the plan is complete and parses it on the next line. Writing
    # in place creates the path before `json.dump` has put anything in it, so
    # the driver can win that race and die on a `JSONDecodeError` that looks
    # like a corrupt plan. `os.replace` is atomic within a filesystem, and the
    # temp file is a sibling so the rename never crosses one -- which matters
    # here, since the two processes are documented as coordinating through a
    # shared mount.
    plan_tmp = f"{args.plan_out}.partial"
    with open(plan_tmp, "w") as fh:
        json.dump(plan, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(plan_tmp, args.plan_out)
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
        #
        # Stamped with `run_id` for exactly the reason the plan is, and it
        # matters more here. `--plan-out` is a fixed shared path, so a marker
        # from a previous run is the ordinary state of that directory -- and a
        # stale one read as this run's says a collective was matched when none
        # was, which is the single observation standing between a silently
        # dead transport and a `PROVEN` verdict. Written by rename so a marker
        # that exists is a marker that is complete.
        marker = f"{args.plan_out}.round{index}.done"
        marker_tmp = f"{marker}.partial"
        with open(marker_tmp, "w") as fh:
            json.dump({"run_id": args.run_id, "kind": kind}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(marker_tmp, marker)

    dist.destroy_process_group(pg)
    log("group destroyed; peer exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
