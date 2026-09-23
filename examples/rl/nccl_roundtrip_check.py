#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Prove TokenSpeed's ``nccl`` weight transfer actually moves weights.

An HTTP 200 from ``/update_weights`` proves the control plane parsed some
metadata. It does not prove a tensor landed in the model, and a partially wired
receive path returns exactly that 200 with unchanged outputs -- which is the
worst thing to report as working. This driver closes that gap by observing the
model's behaviour instead of the status code.

The test is a round trip, in greedy generations of one fixed prompt:

    baseline   -> completion A, generated ``--replicates`` times and required
                  to agree with itself
    perturb    -> completion B, which must differ from A on every replicate
    restore    -> completion C, which must equal A on every replicate

Both halves are load-bearing. B != A rules out a receive path that drops the
payload; C == A rules out one that corrupts memory or lands tensors in the wrong
place, and shows the transfer is faithful rather than merely destructive. A
perturbation-only test passes in both of those cases.

Why the replicates, and why the peer evidence
---------------------------------------------
One completion per phase cannot carry this verdict, and the reason is measured
rather than theoretical: on this stack a temperature-0 completion is *not* a
function of the weights alone. Separate greedy requests differ occasionally --
1-2 of 8 -- because the argmax moves with batch composition and floating-point
reduction order (``docs/tokenspeed-rl-e2e-sanitizer-routing.md`` §5.4 records
2/8 distinct completions at temperature 0). With a no-op transfer, a single
jittering B and a matching C therefore report ``PROVEN`` on decoder noise. At
the rate that section measures, a single-draw verdict does that 11% of the time.

So the completions are now compared as phases rather than as strings. The
baseline is drawn ``--replicates`` times and must agree with itself, which is
what makes the observable's stability an observation instead of an assumption;
if it does not agree, the verdict says so and no later difference is read as
evidence. B must differ on *every* replicate and C must match on *every* one,
so a jitter-driven false ``PROVEN`` needs the coincidence to repeat.

Beyond the model's behaviour, ``nccl_weight_peer.py`` writes
``<plan>.roundN.done`` once round N's broadcasts have drained, and a
``dist.broadcast`` from the sending rank returns only when the other rank posts
its matching receive. A missing marker after the engine has already answered
200 is therefore direct evidence that the engine posted no collective --
independent of anything the model emits, and available even when the decoder is
too unstable for the round trip to say anything. The peer has always written
these; this driver used to discard them.

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

    temperature 0 with a fixed prompt and ``ignore_eos`` removes the *sampler*
    as a source of variation. It does not make the completion a deterministic
    function of the weights, and this docstring used to claim that it did: the
    argmax still moves with batch composition and reduction order, which the
    measurement in §5.4 of the sanitizer-routing doc records as 2 of 8 distinct
    completions across separate temperature-0 requests. Callers must therefore
    treat a single completion as one draw from a narrow distribution, not as a
    reading of the weights -- see ``generate_phase``.
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


def generate_phase(
    base: str, model: str, prompt: str, max_tokens: int, replicates: int
) -> dict[str, Any]:
    """One phase of the round trip: ``replicates`` greedy draws, kept together.

    A phase rather than a completion because the comparison the verdict makes
    is between phases. ``texts`` is every draw in order, ``distinct`` is how
    many different answers came back, and ``stable`` says whether the phase
    agreed with itself -- which is the observation that decides whether any
    comparison involving it means anything.

    ``text`` is kept as the first draw so a report stays readable next to the
    older ones, and so a reader who looks only at that field sees a completion
    rather than a summary. It is deliberately *not* what the verdict reads.
    """
    draws = [generate(base, model, prompt, max_tokens) for _ in range(replicates)]
    texts = [d["text"] for d in draws]
    generated = [t for t in texts if t is not None]
    return {
        "status": draws[0]["status"],
        "text": texts[0],
        "texts": texts,
        "seconds": round(sum(d["seconds"] for d in draws), 3),
        "raw": draws[0]["raw"],
        "replicates": replicates,
        # Counted over the draws that produced text: a failed generation is a
        # missing observation, not a distinct answer.
        "distinct": len(set(generated)),
        "ok": len(generated) == replicates and replicates > 0,
        "stable": len(set(generated)) == 1 and len(generated) == replicates,
    }


def phase_texts(phase: dict[str, Any]) -> list[Any]:
    """Every draw in a phase, tolerating a single-completion dict.

    ``decide_verdict`` is reachable from callers that predate phases, and a
    bare generation is exactly a one-replicate phase. Normalising here rather
    than at each comparison keeps one definition of what a phase contains.
    """
    texts = phase.get("texts")
    if texts is None:
        return [phase.get("text")]
    return list(texts)


# Keys the peer puts on the plan for the driver's benefit, which the engine has
# never heard of. Kept out of `update_info`: an unknown key there is a 500.
_PLAN_COORDINATION_KEYS = frozenset({"run_id"})


def peer_round_marker(plan: str, index: int) -> Path:
    """Where ``nccl_weight_peer.py`` records that round ``index`` drained.

    The peer writes ``<plan-out>.roundN.done`` after round N's broadcasts have
    all returned, numbering rounds from 1 in the order of its own ``--rounds``.
    With the documented ``--rounds perturb,restore`` that makes the perturb
    round 1 and the restore round 2, which is the coupling the two indices in
    ``main`` encode.
    """
    return Path(f"{plan}.round{index}.done")


def wait_for_peer_round(
    plan: str, index: int, grace_s: float, run_id: str, kind: str
) -> dict[str, Any]:
    """Did the sender's broadcasts for this round actually complete?

    ``dist.broadcast`` from the sending rank returns only once the other rank
    posts its matching receive, so the peer reaching its round marker is proof
    a collective was matched -- and *not* reaching it, after the engine has
    already answered 200, is proof one was not. That is a direct observation of
    the transport, where the completions are an inference from model behaviour.

    Matched on ``run_id``, not on the path existing, for the same reason
    ``wait_for_plan`` is -- and the stakes are higher here. ``--plan`` is a
    fixed shared path, so a marker left by a previous run is the ordinary state
    of that directory, and accepting one means reporting that a collective was
    matched when none was. That is the single observation standing between a
    silently dead transport and ``PROVEN``, so a stale marker would quietly
    undo the reason this check exists. ``run_id_seen`` is reported rather than
    swallowed, because "the peer never got here" and "the directory needs
    clearing" send an operator to different places.

    The round's ``kind`` is matched too, for a separate reason. This driver
    numbers rounds by position -- perturb is 1, restore is 2 -- because that is
    what the documented ``--rounds perturb,restore`` produces. The peer accepts
    them in either order, and under ``--rounds restore,perturb`` round 1 *is*
    the restore: the marker appears on time and the evidence that a collective
    was matched is attached to the wrong phase. Position is this driver's
    assumption; the kind is the peer's own record of what it sent, so comparing
    them turns the assumption into a check.

    Polled with a grace period rather than read once, because the two sides
    finish microseconds apart across a shared filesystem that is not
    synchronous, so a single read races the rename rather than measuring it.
    """
    marker = peer_round_marker(plan, index)
    started = time.time()
    deadline = started + grace_s
    interval = min(1.0, max(0.01, grace_s / 20)) if grace_s > 0 else 0.0
    stale_seen: str | None = None
    mismatched_kind: str | None = None
    while True:
        try:
            record = json.loads(marker.read_text())
            found, found_kind = record.get("run_id"), record.get("kind")
        except (OSError, json.JSONDecodeError, AttributeError):
            # Absent, mid-rename, or written by a peer that predates the
            # stamp. None of those is this run's marker.
            found = found_kind = None
        if found is not None:
            if found != run_id:
                stale_seen = str(found)
            elif found_kind != kind:
                # This run's marker, but for a different round than the one
                # the driver believes it is reading.
                mismatched_kind = str(found_kind)
            else:
                return {
                    "marker": str(marker),
                    "appeared": True,
                    "run_id_seen": found,
                    "kind_seen": found_kind,
                    "waited_seconds": round(time.time() - started, 3),
                }
        if time.time() >= deadline:
            return {
                "marker": str(marker),
                "appeared": False,
                "run_id_seen": stale_seen,
                "kind_expected": kind,
                "kind_seen": mismatched_kind,
                "waited_seconds": round(time.time() - started, 3),
            }
        time.sleep(interval)


def lifecycle_update(control: str, plan: dict[str, Any], label: str) -> dict[str, Any]:
    """One trainer step: start -> update -> finish, each timed separately.

    ``/update_weights`` is the interesting number. It blocks while the workers
    receive every broadcast in the plan, so its wall time is the actual cost of
    moving these tensors, which is what the per-iteration budget is spent on.

    It is also why a rejected ``/start_weight_update`` stops the step here. The
    engine never entered the update state, so there is no receive to match, and
    posting the update anyway meant blocking on ``TIMEOUT_S`` -- **thirty
    minutes** -- before ``_lifecycle_failure`` got to say that the *first* leg
    had failed. The verdict was already decided and the driver spent half an
    hour not saying so, on precisely the failure path it exists to detect.

    A rejected ``/update_weights`` is deliberately *not* treated the same way:
    start succeeded, so the engine is mid-update, and ``/finish_weight_update``
    is what takes it back out. Skipping that would leave the engine in the
    update state for whatever runs next.
    """
    out: dict[str, Any] = {"label": label}
    started = time.time()

    status, body, elapsed = call(control, "POST", "/start_weight_update", {})
    out["start"] = {"status": status, "seconds": round(elapsed, 3), "body": body}
    if status != 200:
        # Named rather than simply absent. `_lifecycle_failure` walks the legs
        # in order and reports `start` whether or not the later keys exist, so
        # the verdict is unaffected either way -- but a reader of the JSON
        # would otherwise have to infer why two legs are missing, and "the
        # driver crashed here" and "the driver declined to post these" are
        # very different things to conclude from a truncated record.
        out["skipped"] = ["update", "finish"]
        out["skipped_reason"] = (
            "start was not accepted, so the engine never entered the update "
            "state; posting /update_weights would block on a receive that "
            "cannot be matched"
        )
        out["total_seconds"] = round(time.time() - started, 3)
        return out

    # `run_id` is coordination, not part of the engine's contract, and it must
    # not travel in the payload. `update_info` is
    # `{names, dtype_names, shapes, packed?, ...}` and the engine rejects an
    # unknown key with a 500 -- `probe_weight_transfer.py` has an
    # "update_weights unknown key" step establishing exactly that. So stamping
    # `run_id` onto the plan to defeat a stale file, which is what the previous
    # commit did, made every round fail before a single broadcast was
    # attempted: the fix for one silent failure created a loud one.
    #
    # Stripped by name from a named set rather than inline, so the next
    # coordination key added to the plan is a one-line change in an obvious
    # place instead of this bug again.
    update_info = {k: v for k, v in plan.items() if k not in _PLAN_COORDINATION_KEYS}
    status, body, elapsed = call(
        control, "POST", "/update_weights", {"update_info": update_info}
    )
    out["update"] = {"status": status, "seconds": round(elapsed, 3), "body": body}

    status, body, elapsed = call(control, "POST", "/finish_weight_update", {})
    out["finish"] = {"status": status, "seconds": round(elapsed, 3), "body": body}

    out["total_seconds"] = round(time.time() - started, 3)
    return out


def wait_for_plan(
    plan_path: Path, run_id: str, timeout_s: int
) -> tuple[dict[str, Any] | None, str | None]:
    """This run's plan, or ``(None, <the run_id seen instead>)``.

    Matched on ``run_id`` rather than on the path existing. The peer and the
    driver are launched by hand against a fixed shared path -- ``/shared/plan.json``
    in the README -- so a plan left by a previous run is the ordinary state of
    that directory, not an unusual one. Accepting it meant reading stale tensor
    names and shapes while this run's peer was writing its own, and then
    mismatching the HTTP update against the collective the peer actually
    broadcasts: that hangs rather than failing, so it would not even have
    reported a verdict.

    ``os.replace`` on the peer's side publishes each plan atomically, so a read
    whose id matches is a complete plan for this run.

    Extracted from ``main`` for the same reason ``decide_verdict`` was: the
    branch ordering is the whole check, and reaching it through ``main`` means
    first waiting on an engine that a test does not have.
    """
    limit = time.time() + timeout_s
    stale_seen: str | None = None
    while True:
        if plan_path.exists():
            try:
                candidate = json.loads(plan_path.read_text())
            except (OSError, json.JSONDecodeError):
                # Mid-rename or unreadable: not this run's plan yet.
                candidate = None
            if isinstance(candidate, dict):
                found = candidate.get("run_id")
                if found == run_id:
                    return candidate, None
                stale_seen = str(found)
        if time.time() >= limit:
            return None, stale_seen
        time.sleep(2)


def _log_step(step: dict[str, Any]) -> bool:
    """Report one lifecycle step, and say whether it is worth asking the peer.

    Returns ``False`` when the step stopped before posting ``/update_weights``.
    The round marker cannot appear for a round the engine never joined, so
    waiting out ``--peer-grace`` there buys nothing but delay on a path that is
    already slow for the wrong reasons, and ``decide_verdict`` ranks the
    lifecycle failure above the peer evidence regardless.

    ``peer_sent`` is then left ``None`` rather than ``False``, which is the
    distinction that module has kept throughout: ``False`` is "the peer did not
    send" and ``None`` is "nobody asked".
    """
    update = step.get("update")
    if update is None:
        log(
            f"{step['label']} step stopped at start -> "
            f"{step['start']['status']}: {step.get('skipped_reason', 'not posted')}"
        )
        return False
    log(f"{step['label']} update -> {update['status']} in {update['seconds']}s")
    return True


def _update_seconds(step: dict[str, Any]) -> float | None:
    """How long this step's update leg took, or ``None`` if it never ran.

    ``None`` rather than ``0.0``: a leg that was never posted has no duration,
    and a zero here would read as an update that returned instantly -- which
    is a real and interesting observation this driver has recorded before.
    """
    update = step.get("update")
    return None if update is None else update["seconds"]


def _lifecycle_failure(lifecycle: dict[str, Any]) -> str | None:
    """The first leg of a start -> update -> finish step that was not accepted.

    All three legs, not just `/update_weights`. A rejected `/start_weight_update`
    means the engine never entered the update state, and a rejected
    `/finish_weight_update` means it may still be in it -- so neither is a step
    whose completions can be read as evidence about weights, and the engine can
    be left mid-update for whatever runs next. Judging on the update status
    alone let both come back `PROVEN` whenever the two generations happened to
    round-trip.
    """
    for leg in ("start", "update", "finish"):
        status = (lifecycle.get(leg) or {}).get("status")
        if status != 200:
            return leg
    return None


def round_one_verdict(
    perturb_lifecycle: dict[str, Any], peer_sent_perturb: bool | None
) -> str | None:
    """The verdict the perturb round decides on its own, or ``None`` to go on.

    The top two rungs of ``decide_verdict``'s ladder, lifted out so ``main`` can
    read them *before* it posts the restore update instead of only after the
    whole run. Both already outrank everything the restore round could
    contribute, so consulting them early costs no evidence -- but continuing
    past them costs correctness on real hardware.

    Broadcasts are ordered. A rejected perturb lifecycle or a missing round-1
    marker both mean the peer may still be blocked in its round-1 send, and the
    two rounds carry identical tensor shapes, so the restore update matches
    that *pending perturb* payload: the engine receives the zeroed weights
    while this driver labels the step "restore", round 2 is the one left
    blocked, and the JSON records a restore that was accepted. Nothing in the
    report would say which round's payload actually landed, and the engine is
    left holding weights nobody asked for. Stopping is the only remedy that
    does not require tearing down and recreating both group participants first,
    which this driver cannot do -- it does not own the peer.
    """
    perturb_bad = _lifecycle_failure(perturb_lifecycle)
    if perturb_bad is not None:
        # Named by leg, because "the update was rejected" and "the engine never
        # entered the update state" are different failures to chase.
        return (
            "UPDATE_REJECTED"
            if perturb_bad == "update"
            else f"LIFECYCLE_REJECTED_{perturb_bad.upper()}"
        )
    if peer_sent_perturb is False:
        return "HTTP_OK_BUT_PEER_NEVER_SENT"
    return None


def decide_verdict(
    *,
    baseline: dict[str, Any],
    perturbed: dict[str, Any],
    restored: dict[str, Any],
    perturb_lifecycle: dict[str, Any],
    restore_lifecycle: dict[str, Any],
    peer_sent_perturb: bool | None = None,
    peer_sent_restore: bool | None = None,
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
    ``recovered`` mean anything, and both update *steps* must have been accepted
    in full -- start, update and finish, the restore for the same reason as the
    perturb. ``changed`` and ``recovered`` come back as ``None`` when there was
    nothing to compare, rather than as a default that reads like an observation.

    Two further things gate the comparison, because status codes and a single
    completion each turned out to be weaker evidence than they look.

    *Peer evidence outranks the completions, deliberately.* A missing round
    marker says the engine posted no collective at all; that is a direct
    observation of the transport, where the completions are an inference from
    model behaviour. When the peer never sent, whatever the model emitted
    afterwards is not about weights that were pushed, whichever way it reads.
    ``None`` means the check was not performed and is evidence either way.

    *The baseline must agree with itself.* A greedy completion is not a
    deterministic function of the weights on this stack -- the argmax moves
    with batch composition -- so comparing single strings lets decoder noise
    supply ``B != A``. Measured against the 2-of-8 jitter rate this repository
    records, single-draw comparison returns a false ``PROVEN`` for a *no-op*
    transfer 11% of the time. An unstable baseline therefore yields its own
    verdict rather than a quiet comparison, and ``changed``/``recovered`` are
    ``None`` because there was nothing trustworthy to compare. The phases are
    then compared as wholes: every perturb draw must differ and every restore
    draw must match, so the coincidence has to repeat to survive.
    """
    perturb_bad = _lifecycle_failure(perturb_lifecycle)
    restore_bad = _lifecycle_failure(restore_lifecycle)

    baseline_seen = phase_texts(baseline)
    perturb_seen = phase_texts(perturbed)
    restore_seen = phase_texts(restored)

    # A phase counts as generated only if *every* replicate came back. A phase
    # that produced two completions and one 500 is a partial observation, and
    # scoring it on the two that survived is the same "absence read as
    # evidence" this function already refuses for the single-completion case.
    perturb_generated = bool(perturb_seen) and all(t is not None for t in perturb_seen)
    restore_generated = bool(restore_seen) and all(t is not None for t in restore_seen)
    both_generated = perturb_generated and restore_generated

    baseline_generated = bool(baseline_seen) and all(
        t is not None for t in baseline_seen
    )
    baseline_stable = baseline_generated and len(set(baseline_seen)) == 1

    # Both booleans require the whole round trip to have happened -- both update
    # steps accepted in full *and* both generations returned text. Computing
    # them from the completions alone published
    # `weights_changed_under_perturb: true` into the report while the verdict
    # said the lifecycle was rejected, so the JSON asserted an observation the
    # verdict had just disowned. `None` is what the docstring promises when
    # there was nothing to compare, and a rejected start or finish means there
    # was nothing to compare: the engine either never entered the update state
    # or may still be in it, so those completions are not evidence about
    # weights.
    #
    # The peer evidence belongs in the same conjunction, and leaving it out was
    # the same defect one rank down. The verdict ladder below already puts
    # `peer_sent_* is False` *above* the generation checks -- the docstring says
    # peer evidence outranks the completions, because a missing round marker is
    # a direct observation that no collective was posted, where the completions
    # are an inference from model behaviour. But `comparable` did not read it,
    # so a run that ended `HTTP_OK_BUT_PEER_NEVER_SENT` still published
    # `weights_changed_under_perturb: true` beside it: the JSON asserting an
    # observation about a transfer the verdict had just said never happened.
    #
    # `is not False`, not truthiness. `None` means the marker check was not
    # performed, which the docstring is explicit is evidence either way, and
    # `not peer_sent_perturb` would read an unperformed check as a failed one
    # -- refusing to compare on every run without a peer, which is most of
    # them.
    comparable = (
        perturb_bad is None
        and restore_bad is None
        and peer_sent_perturb is not False
        and peer_sent_restore is not False
        and both_generated
        and baseline_stable
    )
    # `all`, not `!=` on one pair. One draw differing is what jitter looks
    # like; every draw differing is what a weight change looks like.
    changed = (
        all(t not in set(baseline_seen) for t in perturb_seen) if comparable else None
    )
    recovered = (
        all(t in set(baseline_seen) for t in restore_seen) if comparable else None
    )

    # The first two rungs live in `round_one_verdict` so that `main` can apply
    # exactly this naming when it stops after the perturb round, rather than a
    # second copy of it that can drift.
    round_one = round_one_verdict(perturb_lifecycle, peer_sent_perturb)
    if round_one is not None:
        verdict = round_one
    elif restore_bad is not None:
        verdict = (
            "RESTORE_UPDATE_REJECTED"
            if restore_bad == "update"
            else f"RESTORE_LIFECYCLE_REJECTED_{restore_bad.upper()}"
        )
    elif peer_sent_restore is False:
        verdict = "HTTP_OK_BUT_PEER_NEVER_SENT"
    elif not both_generated:
        verdict = "POST_UPDATE_GENERATION_FAILED"
    elif not baseline_stable:
        # Ranked below the peer check on purpose: an unstable decoder costs us
        # the round trip, but it does not touch the marker evidence, so the
        # strongest finding this tool can make is still available above.
        verdict = (
            "BASELINE_GENERATION_FAILED"
            if not baseline_generated
            else "BASELINE_NONDETERMINISTIC"
        )
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
    # Required rather than optional: the whole point is that a plan is only
    # trusted when it is this run's, and an optional check is one an
    # invocation can leave off precisely when it matters.
    ap.add_argument("--plan-run-id", required=True,
                    help="must match the peer's --run-id")
    # Bounded here rather than hard-coded in the wait, so the stale-plan and
    # never-appeared paths are reachable in a test without waiting ten
    # minutes for each.
    ap.add_argument("--plan-timeout", type=int, default=600,
                    help="seconds to wait for this run's plan (default 600)")
    ap.add_argument("--master-address", default="127.0.0.1")
    ap.add_argument("--master-port", type=int, required=True)
    ap.add_argument("--rank-offset", type=int, default=1)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--group-name", default="weight_update_group")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=32)
    # Three rather than one, because one cannot separate a weight change from
    # argmax jitter, and three drives the measured false-PROVEN rate for a
    # no-op transfer from 11% to under 0.1% at the jitter rate this repository
    # records. 1 restores the old single-draw behaviour for a caller who wants
    # it and knows what it costs.
    ap.add_argument("--replicates", type=int, default=3,
                    help="greedy draws per phase; >1 separates a weight "
                         "change from decoder jitter (default 3)")
    # The peer writes its round markers unconditionally, so this costs nothing
    # when the transport works -- the marker is already there and the first
    # poll returns. It only spends the grace period when there is a real
    # finding to report.
    ap.add_argument("--peer-grace", type=float, default=120.0,
                    help="seconds to wait for the peer's <plan>.roundN.done "
                         "marker after each update; 0 disables the check")
    ap.add_argument("--out", required=True, help="where to write the JSON verdict")
    args = ap.parse_args()
    if args.replicates < 1:
        ap.error("--replicates must be at least 1")
    # `> 0` is what gates the marker check, so every negative value is a
    # second, undocumented opt-out -- and the one a typo produces. `0` is the
    # opt-out the help names, and it reads as one: a caller who types it has
    # decided to accept the verdict without the markers. `-1` reads as a
    # duration, and a caller who types it is asking to wait, not to stop
    # asking. Refused rather than clamped for that reason: clamping to 0 would
    # honour the reading nobody meant, and clamping to the default would wait
    # two minutes on an argument that asked for less than none. The markers are
    # the only direct evidence a collective happened at all, so silently
    # dropping them weakens every verdict this script goes on to print.
    if args.peer_grace < 0:
        ap.error(
            f"--peer-grace is {args.peer_grace}; it is a number of seconds to "
            "wait, and 0 is how the check is disabled. A negative value "
            "disables it too, which is not what it looks like it does."
        )

    report: dict[str, Any] = {
        "engine_url": args.engine_url,
        "control_url": args.control_url,
        "model": args.model,
        "world_size": args.world_size,
        "rank_offset": args.rank_offset,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "replicates": args.replicates,
        "peer_grace_seconds": args.peer_grace,
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

    baseline = generate_phase(
        args.engine_url, args.model, args.prompt, args.max_tokens, args.replicates
    )
    report["baseline"] = baseline
    if not baseline["ok"]:
        report["verdict"] = "BASELINE_GENERATION_FAILED"
        flush()
        return 2
    log(f"baseline completion: {baseline['text']!r}")
    # Reported now rather than only in the verdict, because an operator who
    # sees this line knows immediately that the round trip half of the run is
    # going to be unreadable, while the peer evidence below is not. The run
    # continues either way: the marker check is the stronger finding and it
    # does not depend on the decoder.
    if not baseline["stable"]:
        log(f"baseline is NOT deterministic: {baseline['distinct']} distinct "
            f"completions in {args.replicates} greedy draws -- no later "
            f"difference in the completions is evidence about weights")
    else:
        log(f"baseline is deterministic across {args.replicates} draws")
    flush()

    # The peer publishes the plan before it blocks in rendezvous, so this
    # arriving means the sender is up and it is safe to make the engine join.
    #
    # Matched on `run_id`, not just on the path existing. The two processes are
    # launched by hand against a fixed shared path -- `/shared/plan.json` in the
    # README -- so a plan left by a previous run is the normal state of that
    # directory. Accepting it meant reading stale tensor names and shapes while
    # this run's peer was still writing its own, and then mismatching the HTTP
    # update against the collective the peer actually broadcasts, which hangs
    # rather than failing. `os.replace` on the peer's side makes each plan
    # appear atomically, so a read that matches the id is a complete plan for
    # this run.
    plan, stale_seen = wait_for_plan(
        Path(args.plan), args.plan_run_id, args.plan_timeout
    )
    if plan is None:
        # Distinguished, because the two send an operator to different places:
        # nothing appeared means the peer never got that far, while a plan from
        # another run means the path needs clearing or the ids do not match.
        report["verdict"] = (
            "PEER_PLAN_STALE" if stale_seen is not None else "PEER_PLAN_NEVER_APPEARED"
        )
        report["plan_run_id_expected"] = args.plan_run_id
        report["plan_run_id_seen"] = stale_seen
        flush()
        return 2
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

    # Round indices match the peer's own numbering of its `--rounds`, which
    # defaults to `perturb,restore` and is enumerated from 1. Named here rather
    # than inline so the coupling between the two scripts is visible in one
    # place if a third round is ever added.
    perturb_round, restore_round = 1, 2

    report["perturb"] = lifecycle_update(args.control_url, plan, "perturb")
    posted_perturb = _log_step(report["perturb"])
    peer_sent_perturb: bool | None = None
    if posted_perturb and args.peer_grace > 0:
        peer = wait_for_peer_round(
            args.plan, perturb_round, args.peer_grace, args.plan_run_id, "perturb"
        )
        report["perturb"]["peer"] = peer
        peer_sent_perturb = peer["appeared"]
        log(f"peer round {perturb_round} marker appeared: {peer['appeared']} "
            f"after {peer['waited_seconds']}s")
    flush()

    perturbed = generate_phase(
        args.engine_url, args.model, args.prompt, args.max_tokens, args.replicates
    )
    report["after_perturb"] = perturbed
    log(f"after perturb: {perturbed['text']!r} ({perturbed['distinct']} distinct)")
    flush()

    # Consulted here, before the second collective, rather than left to
    # `decide_verdict` at the end of the run. The verdict is the same either
    # way; what differs is whether the restore update was posted into a group
    # whose first round may never have completed. See `round_one_verdict` for
    # what that does to the engine's weights.
    stopped = round_one_verdict(report["perturb"], peer_sent_perturb)
    if stopped is not None:
        report["verdict"] = stopped
        # `None`, the same withdrawal `decide_verdict` makes for these two
        # verdicts. The perturb completions exist and are in the report, but
        # nothing was established about the weights behind them.
        report["weights_changed_under_perturb"] = None
        report["weights_recovered_under_restore"] = None
        report["stopped_after_round"] = perturb_round
        # Recorded as a declined step rather than left absent, the same way
        # `lifecycle_update` records the legs it declines to post: "the driver
        # crashed here" and "the driver refused to post these" are very
        # different things for a reader to have to infer from a truncated
        # record.
        report["restore"] = {
            "label": "restore",
            "skipped": ["start", "update", "finish"],
            "skipped_reason": (
                "the perturb round did not complete, so the peer may still be "
                "blocked in its round-1 broadcast; this update has the same "
                "shapes and would have been matched against that pending "
                "payload instead of a restore"
            ),
        }
        report["update_seconds"] = {
            "perturb": _update_seconds(report["perturb"]),
            "restore": None,
        }
        flush()
        log(f"stopped after round {perturb_round}: restore not attempted")
        log(f"VERDICT: {report['verdict']}")
        # 1, not 2: the exit code follows the verdict, and these are the same
        # verdicts the full path already returns 1 for. 2 is for a run that
        # never reached a verdict about the transfer at all.
        return 1

    report["restore"] = lifecycle_update(args.control_url, plan, "restore")
    posted_restore = _log_step(report["restore"])
    peer_sent_restore: bool | None = None
    if posted_restore and args.peer_grace > 0:
        peer = wait_for_peer_round(
            args.plan, restore_round, args.peer_grace, args.plan_run_id, "restore"
        )
        report["restore"]["peer"] = peer
        peer_sent_restore = peer["appeared"]
        log(f"peer round {restore_round} marker appeared: {peer['appeared']} "
            f"after {peer['waited_seconds']}s")
    flush()

    restored = generate_phase(
        args.engine_url, args.model, args.prompt, args.max_tokens, args.replicates
    )
    report["after_restore"] = restored
    log(f"after restore: {restored['text']!r} ({restored['distinct']} distinct)")

    verdict, changed, recovered = decide_verdict(
        baseline=baseline,
        perturbed=perturbed,
        restored=restored,
        perturb_lifecycle=report["perturb"],
        restore_lifecycle=report["restore"],
        peer_sent_perturb=peer_sent_perturb,
        peer_sent_restore=peer_sent_restore,
    )
    report["weights_changed_under_perturb"] = changed
    report["weights_recovered_under_restore"] = recovered
    report["verdict"] = verdict

    report["update_seconds"] = {
        "perturb": _update_seconds(report["perturb"]),
        "restore": _update_seconds(report["restore"]),
    }

    flush()
    log(f"VERDICT: {report['verdict']}")
    return 0 if report["verdict"] == "PROVEN" else 1


if __name__ == "__main__":
    sys.exit(main())
