#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Prove TokenSpeed's ``nccl`` weight transfer actually moves weights.

An HTTP 200 from ``/update_weights`` proves the control plane parsed some
metadata. It does not prove a tensor landed in the model, and a partially wired
receive path returns exactly that 200 with unchanged outputs -- which is the
worst thing to report as working. This driver closes that gap by observing the
model's behaviour instead of the status code.

The test is a round trip, in three greedy generations of one fixed prompt:

    baseline   -> completion A
    perturb    -> completion B, which must differ from A
    restore    -> completion C, which must equal A

Both halves are load-bearing. B != A rules out a receive path that drops the
payload; C == A rules out one that corrupts memory or lands tensors in the wrong
place, and shows the transfer is faithful rather than merely destructive. A
perturbation-only test passes in both of those cases.

Pair with ``nccl_weight_peer.py``, which posts the matching broadcasts. Start
the peer first: it is rank 0 and owns the TCP store, and the engine's
``/init_weight_transfer_engine`` blocks until the group forms.

Stdlib only, so it runs on the login node against a remote engine or beside the
peer inside the image.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TIMEOUT_S = 1800


def log(msg: str) -> None:
    print(f"[check] {msg}", flush=True)


def call(
    base: str, method: str, path: str, body: dict[str, Any] | None = None, timeout: int = TIMEOUT_S
) -> tuple[int, Any, float]:
    """One HTTP call, returning (status, parsed-or-raw body, wall seconds)."""
    url = f"{base}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        status = e.code
    except Exception as e:  # noqa: BLE001 - report transport failures as data
        return 0, {"transport_error": str(e)}, time.time() - started
    elapsed = time.time() - started
    try:
        return status, json.loads(raw), elapsed
    except json.JSONDecodeError:
        return status, raw, elapsed


def wait_healthy(base: str, deadline_s: int) -> bool:
    limit = time.time() + deadline_s
    while time.time() < limit:
        status, _, _ = call(base, "GET", "/health", timeout=10)
        if status == 200:
            return True
        time.sleep(3)
    return False


def generate(base: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    """Greedy, fixed-length completion -- the observable that must move.

    temperature 0 with a fixed prompt and ``ignore_eos`` makes the completion a
    deterministic function of the weights, so any difference between two calls
    is a difference in the weights and not in the sampler.
    """
    status, body, elapsed = call(
        base,
        "POST",
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
        },
        timeout=300,
    )
    text = None
    if status == 200 and isinstance(body, dict):
        choices = body.get("choices") or []
        if choices:
            text = choices[0].get("text")
    return {"status": status, "text": text, "seconds": round(elapsed, 3), "raw": body if text is None else None}


def lifecycle_update(control: str, plan: dict[str, Any], label: str) -> dict[str, Any]:
    """One trainer step: start -> update -> finish, each timed separately.

    ``/update_weights`` is the interesting number. It blocks while the workers
    receive every broadcast in the plan, so its wall time is the actual cost of
    moving these tensors, which is what the per-iteration budget is spent on.
    """
    out: dict[str, Any] = {"label": label}
    started = time.time()

    status, body, elapsed = call(control, "POST", "/start_weight_update", {})
    out["start"] = {"status": status, "seconds": round(elapsed, 3), "body": body}

    status, body, elapsed = call(control, "POST", "/update_weights", {"update_info": plan})
    out["update"] = {"status": status, "seconds": round(elapsed, 3), "body": body}

    status, body, elapsed = call(control, "POST", "/finish_weight_update", {})
    out["finish"] = {"status": status, "seconds": round(elapsed, 3), "body": body}

    out["total_seconds"] = round(time.time() - started, 3)
    return out


def decide_verdict(
    *,
    baseline: dict[str, Any],
    perturbed: dict[str, Any],
    restored: dict[str, Any],
    perturb_status: int,
    restore_status: int,
) -> tuple[str, bool | None, bool | None]:
    """The round trip's verdict, plus the two observations behind it.

    Split out of ``main`` because the ordering of these branches is the whole
    check, and getting it wrong is silent: a failed generation carries
    ``text=None``, and ``None`` compares unequal to the baseline, which is
    exactly what a successful weight change looks like. Ordered naively, a 500
    from the perturbed engine followed by a healthy restore reports ``PROVEN``
    against a completion B that never existed -- the strongest possible verdict
    from the weakest possible evidence, which is the one failure this driver
    was written to prevent.

    So both post-update generations must have happened before ``changed`` and
    ``recovered`` mean anything, and both updates must have been accepted --
    the restore for the same reason as the perturb. ``changed`` and
    ``recovered`` come back as ``None`` when there was nothing to compare,
    rather than as a default that reads like an observation.
    """
    both_generated = perturbed["text"] is not None and restored["text"] is not None
    changed = perturbed["text"] != baseline["text"] if both_generated else None
    recovered = restored["text"] == baseline["text"] if both_generated else None

    if perturb_status != 200:
        verdict = "UPDATE_REJECTED"
    elif restore_status != 200:
        verdict = "RESTORE_UPDATE_REJECTED"
    elif not both_generated:
        verdict = "POST_UPDATE_GENERATION_FAILED"
    elif changed and recovered:
        verdict = "PROVEN"
    elif changed:
        verdict = "CHANGED_BUT_NOT_FAITHFUL"
    else:
        verdict = "HTTP_OK_BUT_WEIGHTS_UNCHANGED"
    return verdict, changed, recovered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-url", default="http://127.0.0.1:30000")
    ap.add_argument("--control-url", default="http://127.0.0.1:30010")
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True, help="plan file written by the peer")
    ap.add_argument("--master-address", default="127.0.0.1")
    ap.add_argument("--master-port", type=int, required=True)
    ap.add_argument("--rank-offset", type=int, default=1)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--group-name", default="weight_update_group")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", required=True, help="where to write the JSON verdict")
    args = ap.parse_args()

    report: dict[str, Any] = {
        "engine_url": args.engine_url,
        "control_url": args.control_url,
        "model": args.model,
        "world_size": args.world_size,
        "rank_offset": args.rank_offset,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
    }

    def flush() -> None:
        Path(args.out).write_text(json.dumps(report, indent=2))

    # The control URL, not the generation one. `/health` lives on the control
    # endpoint -- `serve_for_rollouts.sh` polls it there, `probe_weight_transfer.py`
    # reads it there, and `/get_world_size` two lines below is already addressed
    # there. Probing the gateway instead means the documented
    # `--engine-url :8000 --control-url :8001` invocation waits the full fifteen
    # minutes and then reports ENGINE_UNHEALTHY against a server that came up.
    if not wait_healthy(args.control_url, 900):
        report["verdict"] = "ENGINE_UNHEALTHY"
        flush()
        return 2
    log("engine healthy")

    status, world, _ = call(args.control_url, "GET", "/get_world_size", timeout=30)
    report["engine_get_world_size"] = {"status": status, "body": world}
    log(f"engine reports world size: {world}")

    baseline = generate(args.engine_url, args.model, args.prompt, args.max_tokens)
    report["baseline"] = baseline
    if baseline["text"] is None:
        report["verdict"] = "BASELINE_GENERATION_FAILED"
        flush()
        return 2
    log(f"baseline completion: {baseline['text']!r}")
    flush()

    # The peer publishes the plan before it blocks in rendezvous, so this
    # arriving means the sender is up and it is safe to make the engine join.
    plan_path = Path(args.plan)
    limit = time.time() + 600
    while not plan_path.exists() and time.time() < limit:
        time.sleep(2)
    if not plan_path.exists():
        report["verdict"] = "PEER_PLAN_NEVER_APPEARED"
        flush()
        return 2
    plan = json.loads(plan_path.read_text())
    report["plan"] = plan
    log(f"plan: {len(plan['names'])} tensor(s) {plan['names']}")

    status, body, elapsed = call(
        args.control_url,
        "POST",
        "/init_weight_transfer_engine",
        {
            "init_info": {
                "master_address": args.master_address,
                "master_port": args.master_port,
                "rank_offset": args.rank_offset,
                "world_size": args.world_size,
                "group_name": args.group_name,
            }
        },
    )
    report["init"] = {"status": status, "seconds": round(elapsed, 3), "body": body}
    log(f"init_weight_transfer_engine -> {status} in {elapsed:.3f}s")
    flush()
    if status != 200:
        report["verdict"] = "GROUP_INIT_FAILED"
        flush()
        return 2

    report["perturb"] = lifecycle_update(args.control_url, plan, "perturb")
    log(f"perturb update -> {report['perturb']['update']['status']} in {report['perturb']['update']['seconds']}s")
    flush()

    perturbed = generate(args.engine_url, args.model, args.prompt, args.max_tokens)
    report["after_perturb"] = perturbed
    log(f"after perturb: {perturbed['text']!r}")
    flush()

    report["restore"] = lifecycle_update(args.control_url, plan, "restore")
    log(f"restore update -> {report['restore']['update']['status']} in {report['restore']['update']['seconds']}s")
    flush()

    restored = generate(args.engine_url, args.model, args.prompt, args.max_tokens)
    report["after_restore"] = restored
    log(f"after restore: {restored['text']!r}")

    verdict, changed, recovered = decide_verdict(
        baseline=baseline,
        perturbed=perturbed,
        restored=restored,
        perturb_status=report["perturb"]["update"]["status"],
        restore_status=report["restore"]["update"]["status"],
    )
    report["weights_changed_under_perturb"] = changed
    report["weights_recovered_under_restore"] = recovered
    report["verdict"] = verdict

    report["update_seconds"] = {
        "perturb": report["perturb"]["update"]["seconds"],
        "restore": report["restore"]["update"]["seconds"],
    }

    flush()
    log(f"VERDICT: {report['verdict']}")
    return 0 if report["verdict"] == "PROVEN" else 1


if __name__ == "__main__":
    sys.exit(main())
