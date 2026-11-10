"""C5 step 6: the perturb-then-restore round trip (design Section 6.5).

On fresh engines at run start, and after any engine restart: take a baseline, reload a perturbed slot (every
probe draw's prompt log-probs must change), reload the restore slot (every draw must be identical to the
baseline), each phase judged on `reps` replicates as in an earlier prototype. A draw is one probe scored alone: its
prompt log-probs (a fresh prefill, so the prefix cache cannot supply the baseline, Section 6.5 consequence 1)
and a short greedy continuation.

Verdicts: RELOAD_REJECTED (a reload failed), UNSTABLE_BASELINE (the baseline does not agree with itself),
PROVEN, NOT_PROVEN.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from aorlta.sync.landing import LandingError, ProbeSet, parse_input_logprobs
from aorlta.sync.merge import merge_all
from aorlta.sync.reload import ReloadError, reload_all

Draw = tuple[tuple[float, ...], tuple[int, ...]]


def draw_request(ids: Sequence[int], greedy_tokens: int) -> dict[str, Any]:
    return dict(input_ids=list(ids), sampling_params={"temperature": 0.0, "max_new_tokens": greedy_tokens},
                return_logprob=True, logprob_start_len=0)


def draw(engine, ids: Sequence[int], greedy_tokens: int = 16) -> Draw:
    out = engine.generate(**draw_request(ids, greedy_tokens))
    lps = parse_input_logprobs(out, ids)
    if not all(math.isfinite(v) for v in lps):
        raise LandingError(f"draw has a prompt log-prob that is not finite: "
                           f"{[v for v in lps if not math.isfinite(v)][:3]}")
    gen = out.get("output_ids")
    if not isinstance(gen, (list, tuple)) or not gen:
        raise LandingError(f"draw has no output ids: {str(out)[:200]}")
    return tuple(lps), tuple(int(t) for t in gen)


def probe_draws(engines: Sequence[Any], probes: ProbeSet, reps: int, greedy_tokens: int = 16) -> list[list[Draw]]:
    """[rep][engine-major, probe-minor] draws, every probe scored alone and in sequence."""
    return [[draw(e, ids, greedy_tokens) for e in engines for ids in probes.ids] for _ in range(reps)]


@dataclass(frozen=True)
class RoundTripResult:
    verdict: str
    reps: int
    engines: int
    probes: int
    draws_per_phase: int
    baseline_self_consistent: bool
    perturbed_draws_differing: int
    restore_draws_matching: int
    seconds: float
    reloads: list = field(default_factory=list)
    detail: str = ""

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("verdict", "reps", "engines", "probes", "draws_per_phase",
                                               "baseline_self_consistent", "perturbed_draws_differing",
                                               "restore_draws_matching", "seconds", "reloads", "detail")}


def judge(base: list[list[Draw]], pert: list[list[Draw]], rest: list[list[Draw]]) -> tuple[str, bool, int, int]:
    """Judged by phase: the baseline agrees with itself, every perturbed draw's prompt log-probs differ from it (a
    changed continuation alone is not a landed reload), every restore draw equals it."""
    ref = base[0]
    consistent = all(b == ref for b in base)
    differing = sum(p[j][0] != ref[j][0] for p in pert for j in range(len(ref)))
    matching = sum(r[j] == ref[j] for r in rest for j in range(len(ref)))
    per_phase = len(ref) * len(pert)
    if not consistent:
        verdict = "UNSTABLE_BASELINE"
    elif differing == per_phase and matching == len(ref) * len(rest):
        verdict = "PROVEN"
    else:
        verdict = "NOT_PROVEN"
    return verdict, consistent, differing, matching


def round_trip(engines: Sequence[Any], probes: ProbeSet, *, perturbed_slot: str | Path, restore_slot: str | Path,
               expected: int, in_flight: Callable[[], int], reps: int = 3, greedy_tokens: int = 16) -> RoundTripResult:
    t0 = time.monotonic()
    base = probe_draws(engines, probes, reps, greedy_tokens)
    reloads = []
    try:
        reloads.append([r.__dict__ for r in reload_all(engines, perturbed_slot, expected, in_flight=in_flight)])
        pert = probe_draws(engines, probes, reps, greedy_tokens)
        reloads.append([r.__dict__ for r in reload_all(engines, restore_slot, expected, in_flight=in_flight)])
        rest = probe_draws(engines, probes, reps, greedy_tokens)
    except ReloadError as exc:
        return RoundTripResult("RELOAD_REJECTED", reps, len(engines), len(probes.ids), reps * len(engines) *
                               len(probes.ids), all(b == base[0] for b in base), 0, 0, time.monotonic() - t0,
                               reloads, str(exc)[:500])
    verdict, consistent, differing, matching = judge(base, pert, rest)
    return RoundTripResult(verdict, reps, len(engines), len(probes.ids), len(base[0]) * reps, consistent, differing,
                           matching, time.monotonic() - t0, reloads)


def synthetic_perturbation(base: dict[str, torch.Tensor], *, seed: int = 7, rank: int = 16, alpha: float = 32.0,
                           b_std: float = 0.1) -> dict[str, torch.Tensor]:
    """A large random r16/alpha32 LoRA merged into every target weight: it changes
    every probe. Returns W_eff keyed like `base`."""
    gen = torch.Generator(device=next(iter(base.values())).device).manual_seed(seed)
    factors = {}
    for key in sorted(base):
        w = base[key]
        out_f, in_f = w.shape
        a = (torch.rand(rank, in_f, generator=gen, device=w.device) * 2 - 1) / in_f ** 0.5
        b = torch.randn(out_f, rank, generator=gen, device=w.device) * b_std
        factors[key] = (a, b, alpha / rank)
    return merge_all(base, factors, base, version=0).weights
