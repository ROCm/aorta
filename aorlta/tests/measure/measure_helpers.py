"""Synthetic spans shared by the measurement tests."""

from aorlta.measure.timeline import GAP
from aorlta.spans import Span

T0 = 1_793_600_000.0  # UTC seconds; every synthetic time is an offset from it


def sp(name, start, end, it=1, **attrs):
    return Span(name, it, T0 + start, T0 + end, attrs)


def v1_iteration(it=1):
    """One v1 iteration with three chains. Chain b needs a repair turn, so it ends generation last;
    checks stream as chains end, training streams while checks run, then the sync tail. The critical
    path is known by construction (C9_T1_PATH)."""
    a, b, c = (f"run/{it}/standin:p/{k}" for k in range(3))
    return [
        sp("select", 0, 1, it),
        sp("gen.turn0", 1, 50, it, chain_id=a), sp("gen.turn0", 1, 30, it, chain_id=b),
        sp("gen.turn0", 1, 45, it, chain_id=c),
        sp("compile", 50, 51, it, chain_id=a), sp("compile", 30, 31, it, chain_id=b),
        sp("compile", 45, 46, it, chain_id=c),
        sp("gen.repair", 31, 80, it, chain_id=b), sp("compile", 80, 81, it, chain_id=b),
        sp("gate.parse", 51, 52, it, candidate_id=a), sp("ref.native", 52, 70, it, candidate_id=a),
        sp("san.consan", 52, 90, it, candidate_id=a),
        sp("gate.parse", 46, 47, it, candidate_id=c), sp("asan.queue", 47, 60, it, candidate_id=c, request="r1"),
        sp("asan.run", 60, 150, it, request="r1", packages=1),
        sp("gate.parse", 81, 82, it, candidate_id=b), sp("ref.native", 82, 100, it, candidate_id=b),
        sp("san.consan", 82, 120, it, candidate_id=b),
        sp("score.wait", 81, 150.5, it),
        sp("train.stream", 91, 110, it, group="p0"), sp("train.stream", 121, 135, it, group="p1"),
        sp("train.final", 150.5, 170, it),
        sp("merge", 171, 172, it), sp("write", 172, 173.5, it), sp("reload", 173.5, 175, it),
        sp("landing", 175, 176, it, verdict="LANDED"),
        sp("record", 176, 177, it), sp("ckpt", 177, 178, it),
    ]


C9_T1_PATH = [
    (0, 1, "select"), (1, 30, "gen.turn0"), (30, 31, "compile"), (31, 80, "gen.repair"), (80, 81, "compile"),
    (81, 150.5, "score.wait"), (150.5, 170, "train.final"), (170, 171, GAP), (171, 172, "merge"),
    (172, 173.5, "write"), (173.5, 175, "reload"), (175, 176, "landing"),
]
