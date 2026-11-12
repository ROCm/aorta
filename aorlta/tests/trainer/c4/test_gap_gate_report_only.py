"""The gap gate in report-only mode (`trainer.gap_gate: report_only`): the gap is measured and the gate's verdict
logged, but no sample is excluded and no step refused for it, until T10 sets the statistic and limits. Enforced, the
same samples are excluded and the step refused, as before."""

from __future__ import annotations

import pytest
import torch

from c4_helpers import engine_logprobs_from, merged_forward, synthetic_iteration, tiny_peft
from aorlta.trainer import checks as C
from aorlta.trainer.loss import LossSettings, build_groups
from aorlta.trainer.ranks import LocalWorld, OptimSettings, RankTrainer


def _setup(*, shift=None, gate=True, **loss):
    model = tiny_peft(torch.bfloat16, seed=0)
    merged = merged_forward(model)
    samples, results = synthetic_iteration(groups=6, k=4, seed=100, unscored=0.0, quarantined=0.0,
                                           one_scored_group=False, scored_without_sample=0)
    samples = engine_logprobs_from(model, samples, amp=True, shift=shift)
    plans = build_groups(samples, results)
    t = RankTrainer(model, merged, LossSettings(loss_temperature=0.7, **loss), OptimSettings(lr=1e-3, grad_clip=1.0),
                    world=LocalWorld(), micro_batch_tokens=256, amp=True, gap_gate=gate)
    return plans, t


# T10's gate: the per-sample mean |trainer - engine| over the trained reply, 0.05 nats, 2% at most
T10_GATE = {"gap_statistic": "mean_abs", "max_generation_gap": 0.05, "max_off_policy_fraction": 0.02}


def _two_over_the_bound():
    plans, _ = _setup()
    ids = [s.chain_id for g in plans for s in g.samples]
    return ids, {ids[0]: (0, 5.0), ids[5]: (1, 5.0)}  # 2 of 24 samples, 8% > 2%


def test_report_only_excludes_nothing_refuses_nothing_and_logs_what_enforcing_would_have_done():
    ids, shift = _two_over_the_bound()
    plans, t = _setup(shift=shift, gate=False)
    t.stream_begin()
    for g in plans:
        t.stream_add(g)
    rep = t.stream_finish()  # the step is taken
    assert rep["excluded"] == [] and rep["counts"]["excluded"] == 0 and rep["totals"]["samples"] == 24
    gg = rep["gap_gate"]
    assert gg["mode"] == "report_only" and gg["statistic"] == "max_abs"
    assert (gg["compared"], gg["over_bound"], gg["over_bound_ids"]) == (24, 2, sorted([ids[0], ids[5]]))
    assert gg["fraction"] == pytest.approx(2 / 24) and gg["would_refuse"] is True
    assert rep["gap"]["samples"] == 24 and rep["gap"]["max_abs"]["max"] >= 5.0
    assert rep["checks"]["off_policy_fraction"]["gate"] == "off"


def test_enforced_the_same_samples_are_excluded_and_the_step_refused():
    ids, shift = _two_over_the_bound()
    plans, t = _setup(shift=shift, gate=True)
    with pytest.raises(C.StepRefused) as exc:
        t.train_iteration(plans)
    rep = exc.value.report
    assert sorted(rep["excluded"]) == sorted([ids[0], ids[5]])
    assert rep["gap_gate"]["mode"] == "enforce" and rep["gap_gate"]["over_bound_ids"] == sorted([ids[0], ids[5]])
    assert rep["gap_gate"]["would_refuse"] is True


@pytest.mark.parametrize("gate", [True, False])
def test_the_adopted_gate_judges_and_names_its_own_statistic(gate):
    """One planted 5-nat token is a mean of at least 0.25 nats over a reply of at most 20 tokens: over 0.05."""
    ids, shift = _two_over_the_bound()
    plans, t = _setup(shift=shift, gate=gate, **T10_GATE)
    if gate:
        with pytest.raises(C.StepRefused) as exc:
            t.train_iteration(plans)
        rep = exc.value.report
        assert sorted(rep["excluded"]) == sorted([ids[0], ids[5]])
    else:
        (rep,) = t.train_iteration(plans)
        assert rep["excluded"] == []
    gg = rep["gap_gate"]
    assert gg["statistic"] == "mean_abs" and gg["max_generation_gap"] == 0.05
    assert gg["over_bound_ids"] == sorted([ids[0], ids[5]]) and gg["would_refuse"] is True


def test_within_the_bound_report_only_would_not_refuse():
    plans, t = _setup(gate=False)
    (rep,) = t.train_iteration(plans)
    assert rep["gap_gate"]["over_bound"] == 0 and rep["gap_gate"]["would_refuse"] is False


def test_an_open_step_reports_its_gap_without_stepping():
    ids, shift = _two_over_the_bound()
    plans, t = _setup(shift=shift, gate=False)
    before = [p.detach().clone() for p in t.params]
    t.stream_begin()
    for g in plans:
        t.stream_add(g)
    rep = t.stream_report()
    assert rep["gap"]["samples"] == 24 and rep["gap_gate"]["over_bound"] == 2 and rep["counts"]["compared"] == 24
    t.stream_discard()
    assert t.steps_done == 0 and all(C.bit_identical(p, q) for p, q in zip(t.params, before))
