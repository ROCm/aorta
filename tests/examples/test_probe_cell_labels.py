"""Labelling a probe *cell* -- several trials of one configuration -- as one run.

CPU-only. ``label_trials`` is what an archived matrix's ground truth is built
from, so the claims here are the ones a wrong label would silently break: the
documented fail > error > pass precedence across trials, the producer's
``meta:`` placement surviving the aggregation, and a cell with nothing in it
refusing to become a ``pass``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "rl"))

from triage_reward import (  # noqa: E402
    Answer,
    Label,
    find_probe_cells,
    label_run,
    label_trials,
    score_answer,
    set_f1,
)


def trial(verdict=None, failures=(), errors=()):
    doc = {"failure_detectors_fired": list(failures), "error_detectors_fired": list(errors)}
    if verdict is not None:
        doc["verdict"] = verdict
    return doc


def test_one_reproducing_trial_makes_the_cell_a_reproduction():
    label = label_trials([trial("pass"), trial("fail", ["tier4:nan_signature"]), trial("pass")])
    assert label.verdict == "fail"
    assert label.failure_detectors == ["tier4:nan_signature"]


def test_infra_noise_alone_leaves_the_cell_an_error():
    label = label_trials([trial("error", errors=["tier1:timeout"]), trial("pass")])
    assert label.verdict == "error"


def test_a_clean_cell_is_a_pass_with_nothing_cited():
    label = label_trials([trial("pass"), trial("pass")])
    assert label.verdict == "pass"
    assert label.cited_detectors == set()


def test_the_producers_meta_placement_survives_aggregation():
    """``label_run`` keeps a ``meta:`` error ID where the producer put it,
    because the resolver does not own it. Re-splitting the union through the
    resolver would move it to the failure side and score a trial that never
    ran as a reproduction."""
    doc = trial("error", errors=["meta:env_file_validation_failed"])
    assert label_run(doc).verdict == "error"
    label = label_trials([doc, trial("pass")])
    assert label.verdict == "error"
    assert label.error_detectors == ["meta:env_file_validation_failed"]
    assert label.failure_detectors == []


def test_a_meta_failure_signal_is_still_a_failure():
    """Narrowness: the carve-out is positional, not by prefix."""
    label = label_trials([trial("fail", failures=["meta:missing_pass_signal"])])
    assert label.verdict == "fail"


def test_detectors_are_unioned_in_first_seen_order_without_repeats():
    label = label_trials([
        trial("fail", ["b", "a"]),
        trial("fail", ["a", "c"]),
    ])
    assert label.failure_detectors == ["b", "a", "c"]


def test_an_empty_cell_refuses_rather_than_labelling_itself_a_pass():
    with pytest.raises(ValueError, match="no trial results"):
        label_trials([], source="cell")


def test_a_non_mapping_trial_is_refused():
    with pytest.raises(ValueError, match="must be a mapping"):
        label_trials([trial("pass"), ["not", "a", "doc"]])


def test_stale_is_reported_when_the_stored_verdicts_disagree():
    label = label_trials([trial("pass", ["tier4:nan_signature"]), trial("pass")])
    assert label.verdict == "fail"
    assert label.stored_verdict == "pass"
    assert label.stale


def test_agreeing_stored_verdicts_are_not_stale():
    label = label_trials([trial("fail", ["x"]), trial("pass")])
    assert label.stored_verdict == "fail"
    assert not label.stale


def test_a_rotted_stored_verdict_makes_the_cell_stale_and_is_named():
    label = label_trials([trial("fal", ["x"]), trial("fail", ["x"])])
    assert label.stale
    assert label.stored_verdict == "fal"


def test_a_partially_stamped_cell_has_no_aggregate_to_disagree_with():
    label = label_trials([trial(None, ["x"]), trial("fail", ["x"])])
    assert label.stored_verdict is None
    assert not label.stale


def test_find_probe_cells_reads_only_the_trial_layout(tmp_path):
    for rel in ("A-none/trial_0", "A-none/trial_1", "B-none/trial_0", "stray"):
        (tmp_path / rel).mkdir(parents=True)
        (tmp_path / rel / "result.json").write_text(json.dumps(trial("pass")))
    assert [c.name for c in find_probe_cells(tmp_path)] == ["A-none", "B-none"]


@pytest.mark.parametrize(
    "predicted, actual, expected",
    [
        ([], [], 1.0),
        (["a"], [], 0.0),
        ([], ["a"], 0.0),
        (["a"], ["b"], 0.0),
        (["a"], ["a"], 1.0),
        (["a", "b"], ["a"], 2 * 0.5 * 1.0 / 1.5),
    ],
)
def test_set_f1_settles_the_empty_cases(predicted, actual, expected):
    assert set_f1(predicted, actual) == pytest.approx(expected)


def test_score_answer_uses_the_shared_f1():
    """The extraction moved no number: attribution is ``set_f1`` exactly."""
    label = Label(verdict="fail", failure_detectors=["a", "b"])
    for cited in ([], ["a"], ["a", "b"], ["a", "c"], ["c"]):
        score = score_answer(Answer(verdict="fail", detectors=cited), label)
        assert score.attribution_f1 == pytest.approx(set_f1(cited, ["a", "b"]))
