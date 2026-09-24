"""The fixed-checkpoint paired evaluation: pooling, pairing, and its refusals.

CPU-only. Everything here is the pure half of ``eval_episodes`` -- the half
that decides whether two columns may be compared and what the comparison says.
The model-loading half is replaced by a stub wherever a test needs to show a
path was or was not reached, so these tests mean the same thing on a machine
with or without torch.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))

import eval_episodes  # noqa: E402


def group(scenario_id: str, *, n: int = 8, step1_hits: int | None = 2,
          reward: float = 1.0, converged: float = 1.0) -> dict:
    """A group in the shape ``episode_rollouts`` returns, trimmed to what is read.

    ``step1_hits=None`` is the unresolvable scenario: the rollout writes
    ``step1_resolver_rate: None`` and no step-1 event counter.
    """
    return {
        "scenario_id": scenario_id,
        "n": n,
        "reward_mean": reward,
        "converged_rate": converged,
        "step1_resolver_rate": None if step1_hits is None else step1_hits / n,
        "events": {} if step1_hits is None else {"resolver_named_on_step_1": step1_hits},
    }


def run(groups: list[dict], **config) -> dict:
    base = {
        "init_from": "somewhere",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 320,
        "max_episode_steps": 8,
        "seed": 20260923,
        "scenarios": [g["scenario_id"] for g in groups],
    }
    base.update(config)
    return {"config": base, "groups": groups}


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------


def test_a_resolvable_group_reports_hits_over_episodes():
    assert eval_episodes.step1_counts(group("s", n=8, step1_hits=3)) == (3, 8)


def test_an_unresolvable_group_contributes_no_trials_at_all():
    """Not (0, 8): a scenario with no resolver cannot fail to name one."""
    assert eval_episodes.step1_counts(group("s", step1_hits=None)) == (0, 0)


def test_a_group_that_resolved_nothing_is_still_a_group_of_trials():
    """Narrowness: 0 of 8 is a measurement; no resolver to name is not."""
    assert eval_episodes.step1_counts(group("s", n=8, step1_hits=0)) == (0, 8)


def test_a_real_difference_gets_a_real_z_and_the_sign_points_forward():
    assert eval_episodes.two_proportion_z(16, 64, 32, 64) > 2.0
    assert eval_episodes.two_proportion_z(32, 64, 16, 64) < 0


@pytest.mark.parametrize("counts, why", [
    ((0, 0, 0, 0), "no trials on either side"),
    ((3, 8, 0, 0), "no trials on one side"),
    ((8, 8, 8, 8), "everybody solves it, so pooled variance is zero"),
    ((0, 8, 0, 8), "nobody solves it, so pooled variance is zero"),
])
def test_an_undefined_comparison_reads_as_no_evidence(counts, why):
    assert eval_episodes.two_proportion_z(*counts) == 0.0, why


def test_a_matched_pair_compares_and_pools_only_resolvable_scenarios():
    result = eval_episodes.compare(
        run([group("a", step1_hits=2), group("b", step1_hits=None)]),
        run([group("a", step1_hits=6), group("b", step1_hits=None)]),
    )
    assert result["pooled_step1_before"] == pytest.approx(0.25)
    assert result["pooled_step1_after"] == pytest.approx(0.75)
    assert [r["resolvable"] for r in result["scenarios"]] == [True, False]


def test_reward_is_reported_twice_because_the_unresolvable_group_moves_one():
    result = eval_episodes.compare(
        run([group("a", reward=4.0), group("b", step1_hits=None, reward=-8.0)]),
        run([group("a", reward=4.0), group("b", step1_hits=None, reward=-2.0)]),
    )
    assert result["resolvable_reward_before"] == result["resolvable_reward_after"] == 4.0
    assert result["reward_before"] == pytest.approx(-2.0)
    assert result["reward_after"] == pytest.approx(1.0)


def test_a_corpus_with_no_resolvable_scenario_prints_rather_than_crashing(capsys):
    result = eval_episodes.compare(run([group("b", step1_hits=None)]),
                                   run([group("b", step1_hits=None)]))
    assert result["pooled_step1_before"] is None
    eval_episodes.print_comparison(result)
    assert "does not distinguish" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# compare: refusals, each with its narrowness case
# ---------------------------------------------------------------------------


def test_a_column_that_did_not_finish_its_scenarios_is_refused():
    """Both columns partial in the same way satisfy the pairing check alone."""
    partial = run([group("a")], scenarios=["a", "b", "c"])
    with pytest.raises(ValueError, match="before column is incomplete"):
        eval_episodes.compare(partial, partial)


def test_the_after_column_is_checked_for_completeness_too():
    with pytest.raises(ValueError, match="after column is incomplete"):
        eval_episodes.compare(run([group("a")]), run([group("a")], scenarios=["a", "b"]))


def test_a_column_that_covers_what_it_asked_for_is_not_refused():
    whole = run([group("a"), group("b")], scenarios=["a"])
    assert eval_episodes.compare(whole, whole)


def test_two_columns_that_saw_different_scenarios_are_refused():
    with pytest.raises(ValueError, match="did not see the same scenarios"):
        eval_episodes.compare(run([group("a"), group("b")]), run([group("a"), group("c")]))


def test_the_same_scenarios_in_a_different_order_are_not_refused():
    result = eval_episodes.compare(run([group("a"), group("b")]),
                                   run([group("b"), group("a")]))
    assert [r["scenario_id"] for r in result["scenarios"]] == ["a", "b"]


def test_different_episode_counts_are_refused():
    with pytest.raises(ValueError, match="different episode counts"):
        eval_episodes.compare(run([group("a", n=8)]), run([group("a", n=64, step1_hits=16)]))


@pytest.mark.parametrize("field, value", [
    ("temperature", 1.0), ("top_p", 0.8), ("max_new_tokens", 512),
    ("max_episode_steps", 4), ("seed", 1),
])
def test_two_columns_sampled_differently_are_refused(field, value):
    with pytest.raises(ValueError, match=f"differ in '{field}'"):
        eval_episodes.compare(run([group("a")]), run([group("a")], **{field: value}))


def test_two_columns_from_different_checkpoints_are_the_whole_point():
    """Narrowness: ``init_from`` sits in the same dict and MUST differ."""
    result = eval_episodes.compare(run([group("a")], init_from="base"),
                                   run([group("a")], init_from="checkpoint-last"))
    assert (result["before"], result["after"]) == ("base", "checkpoint-last")


# ---------------------------------------------------------------------------
# scenario_seed
# ---------------------------------------------------------------------------


def test_each_scenario_gets_its_own_stream_and_it_is_process_independent():
    """A literal expected value pins it, because ``hash()`` would be salted."""
    assert eval_episodes.scenario_seed(7, "a") != eval_episodes.scenario_seed(7, "b")
    assert eval_episodes.scenario_seed(20260923, "xnack_page_fault") == (
        20260923 + zlib.crc32(b"xnack_page_fault")) % (2**31 - 1)
    assert eval_episodes.scenario_seed(1, "s") != eval_episodes.scenario_seed(2, "s")


def test_the_seed_stays_in_range_for_manual_seed():
    for name in ("a", "zzzzzzzzzzzz", "xnack_page_fault"):
        assert 0 <= eval_episodes.scenario_seed(2**31 - 2, name) < 2**31 - 1


# ---------------------------------------------------------------------------
# reuse
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, **kw):
        self.__dict__.update({
            "model": "Qwen/Qwen3-8B", "param_dtype": "float32", "episodes_per_scenario": 64,
            "max_episode_steps": 8, "temperature": 0.7, "top_p": 0.95,
            "max_new_tokens": 320, "seed": 20260923, "scenarios": "",
        })
        self.__dict__.update(kw)


def column_file(tmp_path: Path, **config) -> Path:
    base = {
        "init_from": "/ckpt/last", "model": "Qwen/Qwen3-8B", "param_dtype": "float32",
        "episodes_per_scenario": 64, "max_episode_steps": 8, "temperature": 0.7,
        "top_p": 0.95, "max_new_tokens": 320, "seed": 20260923, "scenarios": ["a"],
    }
    base.update(config)
    path = tmp_path / "before.json"
    path.write_text(json.dumps({"config": base, "groups": [group("a")]}))
    return path


def test_a_column_written_under_the_same_settings_is_reused(tmp_path):
    reused = eval_episodes.reusable_column(column_file(tmp_path), "/ckpt/last", _Args())
    assert reused["config"]["init_from"] == "/ckpt/last"


@pytest.mark.parametrize("field, value", [
    ("model", "Qwen/Qwen3-4B"), ("param_dtype", "bfloat16"), ("episodes_per_scenario", 32),
    ("max_episode_steps", 4), ("temperature", 1.0), ("top_p", 0.8),
    ("max_new_tokens", 512), ("seed", 1), ("init_from", "/ckpt/earlier"),
])
def test_a_column_written_under_different_settings_is_refused(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        eval_episodes.reusable_column(column_file(tmp_path, **{field: value}), "/ckpt/last",
                                      _Args())


def test_the_base_model_column_matches_by_model_name(tmp_path):
    args = _Args()
    assert eval_episodes.column_source(None, args) == "Qwen/Qwen3-8B"
    assert eval_episodes.reusable_column(column_file(tmp_path, init_from="Qwen/Qwen3-8B"),
                                         "Qwen/Qwen3-8B", args)


def test_a_narrowed_scenario_set_is_checked_against_the_flag(tmp_path):
    path = column_file(tmp_path, scenarios=["a", "b"])
    with pytest.raises(ValueError, match="--scenarios asked for"):
        eval_episodes.reusable_column(path, "/ckpt/last", _Args(scenarios="a"))
    assert eval_episodes.reusable_column(path, "/ckpt/last", _Args(scenarios="b, a"))


class ReachedError(Exception):
    pass


def _stub_evaluate(monkeypatch):
    calls = []

    def fake(checkpoint, args, done=None, on_progress=None):
        calls.append((checkpoint, done))
        raise ReachedError

    monkeypatch.setattr(eval_episodes, "evaluate", fake)
    return calls


def test_a_finished_column_is_reused_without_evaluating(tmp_path, capsys, monkeypatch):
    calls = _stub_evaluate(monkeypatch)
    column = eval_episodes._column(column_file(tmp_path), Path("/ckpt/last"), "trained", _Args())
    assert column["config"]["init_from"] == "/ckpt/last"
    assert calls == []
    assert "[reuse]" in capsys.readouterr().out


def test_a_partial_column_is_resumed_rather_than_reused(tmp_path, capsys, monkeypatch):
    calls = _stub_evaluate(monkeypatch)
    path = column_file(tmp_path, scenarios=["a", "b"])
    with pytest.raises(ReachedError):
        eval_episodes._column(path, Path("/ckpt/last"), "trained", _Args())
    assert calls[0][1]["groups"][0]["scenario_id"] == "a", "the done part is carried in"
    assert "1 of 2 scenarios already done" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main's pre-flight refusals
# ---------------------------------------------------------------------------


def test_an_existing_column_is_refused_without_reuse(tmp_path, capsys, monkeypatch):
    calls = _stub_evaluate(monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    (out / "before.json").write_text("{}")
    assert eval_episodes.main(["--after", str(tmp_path / "ckpt"), "--out", str(out)]) \
        == eval_episodes.EXIT_REFUSED
    assert "--reuse" in capsys.readouterr().err
    assert (out / "before.json").read_text() == "{}"
    assert calls == []


def test_comparing_a_checkpoint_with_itself_is_refused_before_any_work(
    tmp_path, capsys, monkeypatch
):
    calls = _stub_evaluate(monkeypatch)
    code = eval_episodes.main(["--before", str(tmp_path / "ckpt"),
                               "--after", str(tmp_path / "ckpt/."),
                               "--out", str(tmp_path / "out")])
    assert code == eval_episodes.EXIT_REFUSED
    assert "comparing a checkpoint with itself" in capsys.readouterr().err
    assert calls == [] and not (tmp_path / "out").exists()


def test_the_base_model_control_is_not_caught_by_that_refusal(tmp_path, monkeypatch):
    """``--before`` omitted means the base model: the normal case, and it must
    get past the refusal to the evaluation."""
    calls = _stub_evaluate(monkeypatch)
    with pytest.raises(ReachedError):
        eval_episodes.main(["--after", str(tmp_path / "ckpt"), "--out", str(tmp_path / "out")])
    assert calls[0][0] is None


def test_a_zero_episode_count_is_refused(tmp_path, monkeypatch):
    calls = _stub_evaluate(monkeypatch)
    assert eval_episodes.main(["--after", str(tmp_path / "c"), "--out", str(tmp_path / "o"),
                               "--episodes-per-scenario", "0"]) == eval_episodes.EXIT_REFUSED
    assert calls == []


def test_the_sampling_defaults_are_the_trainers():
    """A comparison at different sampling settings than training measures two
    things at once; the constants here are only defaults if they match."""
    pytest.importorskip("torch")
    import train_grpo_step

    trainer = train_grpo_step.build_parser().parse_args(["--out", "x"])
    evaluator = eval_episodes.build_parser().parse_args(["--after", "a", "--out", "o"])
    for field in ("temperature", "top_p", "max_new_tokens", "gen_batch", "max_episode_steps"):
        assert getattr(evaluator, field) == getattr(trainer, field), field
    assert evaluator.seed != trainer.seed
