"""C5 step 6: how the round trip judges its draws (design Section 6.5), on a stand-in engine whose draws depend on
the slot it last reloaded."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from aorlta.sync import roundtrip as RT
from aorlta.sync.landing import LandingError, ProbeSet
from aorlta.sync.writer import INDEX_NAME, MANIFEST_NAME

PROBES = ProbeSet(((11, 12, 13), (21, 22, 23)))
BASE = {"lps": [[-1.0, -2.0], [-0.5, -3.0]], "gen": [[5, 6], [7, 8]]}
MOVED = {"lps": [[-1.25, -2.5], [-0.75, -3.5]], "gen": BASE["gen"]}


class PhaseEngine:
    """Answers a probe with the draw of the slot it last reloaded, or of `base` before any reload."""

    engine_id = 0

    def __init__(self, phases: dict):
        self.phases = phases
        self.phase = "base"

    def update_weights_from_disk(self, model_path, flush_cache=True):
        self.phase = Path(model_path).name
        return (True, f"Succeeded to update model weights from {model_path} (31 checkpoint tensors read)", 0)

    def generate(self, input_ids, sampling_params, return_logprob=True, logprob_start_len=-1):
        j = PROBES.ids.index(tuple(input_ids))
        lps, gen = self.phases[self.phase]["lps"][j], self.phases[self.phase]["gen"][j]
        rows = [(None, input_ids[0], None)] + [(v, t, None) for v, t in zip(lps, input_ids[1:])]
        return {"output_ids": list(gen), "meta_info": {"cached_tokens": 0, "input_token_logprobs": rows}}


def round_trip(tmp_path, pert, rest=BASE, probes=PROBES, engine=None):
    slots = []
    for name in ("pert", "rest"):
        (tmp_path / name).mkdir()
        (tmp_path / name / INDEX_NAME).write_text("{}")
        (tmp_path / name / MANIFEST_NAME).write_text("{}")
        slots.append(tmp_path / name)
    e = engine or PhaseEngine({"base": BASE, "pert": pert, "rest": rest})
    return RT.round_trip([e], probes, perturbed_slot=slots[0], restore_slot=slots[1], expected=31,
                         in_flight=lambda: 0, reps=2, greedy_tokens=2)


def test_a_perturbation_that_moves_every_log_prob_is_proven(tmp_path):
    res = round_trip(tmp_path, MOVED)
    assert res.verdict == "PROVEN"
    assert res.perturbed_draws_differing == res.restore_draws_matching == res.draws_per_phase == 4


def test_a_changed_continuation_alone_does_not_prove_the_reload(tmp_path):
    res = round_trip(tmp_path, {"lps": BASE["lps"], "gen": [[9, 9], [9, 9]]})
    assert res.verdict == "NOT_PROVEN" and res.perturbed_draws_differing == 0


def test_the_restore_must_bring_back_the_continuation_too(tmp_path):
    res = round_trip(tmp_path, MOVED, rest={"lps": BASE["lps"], "gen": [[5, 6], [7, 9]]})
    assert res.verdict == "NOT_PROVEN" and res.restore_draws_matching == 2


def test_an_empty_probe_set_is_refused_before_the_baseline_and_the_reloads(tmp_path):
    e = PhaseEngine({"base": BASE, "pert": MOVED, "rest": BASE})
    with pytest.raises(LandingError, match="no probes"):
        round_trip(tmp_path, MOVED, probes=ProbeSet(()), engine=e)
    assert e.phase == "base"


@pytest.mark.parametrize("bad", [-math.inf, math.inf, math.nan])
def test_a_non_finite_prompt_log_prob_stops_the_round_trip(tmp_path, bad):
    with pytest.raises(LandingError, match="finite"):
        round_trip(tmp_path, {"lps": [[-1.25, bad], MOVED["lps"][1]], "gen": BASE["gen"]})
