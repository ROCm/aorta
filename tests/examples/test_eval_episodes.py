"""The fixed-checkpoint paired evaluation: pooling, pairing, and its refusals.

CPU-only. Everything here is the pure half of ``eval_episodes`` -- the half
that decides whether two columns may be compared and what the comparison says.
The model-loading half is replaced by a stub wherever a test needs to show a
path was or was not reached, so these tests mean the same thing on a machine
with or without torch.
"""

from __future__ import annotations

import json
import re
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
        "gen_batch": 4,
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
    ("max_episode_steps", 4), ("seed", 1), ("gen_batch", 8),
])
def test_two_columns_sampled_differently_are_refused(field, value):
    with pytest.raises(ValueError, match=f"differ in '{field}'"):
        eval_episodes.compare(run([group("a")]), run([group("a")], **{field: value}))


def test_a_column_that_predates_gen_batch_does_not_pair_with_one_that_has_it():
    old = run([group("a")])
    del old["config"]["gen_batch"]
    with pytest.raises(ValueError, match="differ in 'gen_batch'"):
        eval_episodes.compare(old, run([group("a")]))
    assert eval_episodes.compare(old, old)


def test_columns_scored_against_different_archives_are_refused():
    with pytest.raises(ValueError, match=r"different archives for \['a'\]"):
        eval_episodes.compare(run([group("a")], corpus={"a": "x"}),
                              run([group("a")], corpus={"a": "y"}))


def test_a_column_with_a_corpus_never_pairs_with_one_without():
    with pytest.raises(ValueError, match="different archives"):
        eval_episodes.compare(run([group("a")]), run([group("a")], corpus={"a": "x"}))


def test_matching_corpora_are_verified_and_missing_ones_are_flagged(capsys):
    same = eval_episodes.compare(run([group("a")], corpus={"a": "x"}),
                                 run([group("a")], corpus={"a": "x"}))
    assert same["corpus_verified"] is True
    legacy = eval_episodes.compare(run([group("a")]), run([group("a")]))
    assert legacy["corpus_verified"] is False
    eval_episodes.print_comparison(legacy)
    assert "not verified to share a ground truth" in capsys.readouterr().out


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
            "max_new_tokens": 320, "gen_batch": 4, "seed": 20260923, "scenarios": "",
            "corpus_root": None,
        })
        self.__dict__.update(kw)


#: What the corpus on disk digests to, in these tests: one scenario, "a".
DIGESTS = {"a": "digest-a"}
#: A two-scenario corpus, for the tests that need one.
TWO = {"a": "digest-a", "b": "digest-b"}
#: The BLAS backend this process reports, in these tests.
BACKEND = {"torch": "2.x", "preferred_blas_library": "backend-a", "env": {}}
#: The scorer identity of this checkout, in these tests.
SCORER = {"sha256": "scorer-a", "files": 3}
#: The checkpoint identity of what this run would load, in these tests.
#: The package versions this run would generate with, in these tests.
RUNTIME = {"python": "3.x", "transformers": "4.x", "tokenizers": "0.x",
           "safetensors": "0.x", "huggingface_hub": "0.x"}
WEIGHTS = {"weights": {"sha256": "w-a", "files": 2, "bytes": 8},
           "tokenizer": {"sha256": "t-a", "files": 1, "bytes": 4}}


def column_file(tmp_path: Path, **config) -> Path:
    base = {
        "init_from": "/ckpt/last", "model": "Qwen/Qwen3-8B", "param_dtype": "float32",
        "episodes_per_scenario": 64, "max_episode_steps": 8, "temperature": 0.7,
        "top_p": 0.95, "max_new_tokens": 320, "gen_batch": 4, "seed": 20260923,
        "scenarios": ["a"], "corpus": dict(DIGESTS), "blas_backend": dict(BACKEND), "scorer": dict(SCORER),
        "checkpoint": dict(WEIGHTS), "runtime": dict(RUNTIME),
    }
    base.update(config)
    path = tmp_path / "before.json"
    path.write_text(json.dumps({"config": base, "groups": [group("a")]}))
    return path


def test_a_column_written_under_the_same_settings_is_reused(tmp_path):
    reused = eval_episodes.reusable_column(column_file(tmp_path), "/ckpt/last", _Args(),
                                           DIGESTS, BACKEND, SCORER, WEIGHTS, RUNTIME)
    assert reused["config"]["init_from"] == "/ckpt/last"


@pytest.mark.parametrize("field, value", [
    ("model", "Qwen/Qwen3-4B"), ("param_dtype", "bfloat16"), ("episodes_per_scenario", 32),
    ("max_episode_steps", 4), ("temperature", 1.0), ("top_p", 0.8),
    ("max_new_tokens", 512), ("gen_batch", 8), ("seed", 1), ("init_from", "/ckpt/earlier"),
])
def test_a_column_written_under_different_settings_is_refused(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        eval_episodes.reusable_column(column_file(tmp_path, **{field: value}), "/ckpt/last",
                                      _Args(), DIGESTS, BACKEND, SCORER, WEIGHTS, RUNTIME)


def test_the_base_model_column_matches_by_model_name(tmp_path):
    args = _Args()
    assert eval_episodes.column_source(None, args) == "Qwen/Qwen3-8B"
    assert eval_episodes.reusable_column(column_file(tmp_path, init_from="Qwen/Qwen3-8B"),
                                         "Qwen/Qwen3-8B", args, DIGESTS, BACKEND, SCORER, WEIGHTS, RUNTIME)


def test_a_narrowed_scenario_set_is_checked_against_the_flag(tmp_path):
    path = column_file(tmp_path, scenarios=["a", "b"], corpus=dict(TWO))
    with pytest.raises(ValueError, match="--scenarios asked for"):
        eval_episodes.reusable_column(path, "/ckpt/last", _Args(scenarios="a"), TWO, BACKEND, SCORER, WEIGHTS, RUNTIME)
    assert eval_episodes.reusable_column(path, "/ckpt/last", _Args(scenarios="b, a"), TWO, BACKEND, SCORER, WEIGHTS, RUNTIME)


def test_a_written_column_records_the_digest_of_every_scenario_it_asked_for():
    """The other half of the reuse check: a column has to carry what it is
    checked against, or every future reuse of it is a refusal."""
    from types import SimpleNamespace

    scenarios = [SimpleNamespace(scenario_id="a", digest="digest-a"),
                 SimpleNamespace(scenario_id="b", digest="digest-b")]
    payload = eval_episodes._column_payload(_Args(), scenarios, [], [], Path("/ckpt/last"),
                                            BACKEND, SCORER, WEIGHTS)
    assert payload["config"]["scorer"] == SCORER
    assert payload["config"]["blas_backend"] == BACKEND
    assert payload["config"]["corpus"] == {"a": "digest-a", "b": "digest-b"}
    assert payload["config"]["scenarios"] == ["a", "b"]


def test_a_column_computed_with_another_blas_backend_is_refused(tmp_path):
    other = dict(BACKEND, preferred_blas_library="backend-b")
    with pytest.raises(ValueError, match="BLAS backend"):
        eval_episodes.reusable_column(column_file(tmp_path), "/ckpt/last", _Args(), DIGESTS,
                                      other, SCORER, WEIGHTS, RUNTIME)


def test_a_column_that_records_no_blas_backend_is_refused(tmp_path):
    path = column_file(tmp_path)
    doc = json.loads(path.read_text())
    del doc["config"]["blas_backend"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="records no BLAS backend"):
        eval_episodes.reusable_column(path, "/ckpt/last", _Args(), DIGESTS, BACKEND, SCORER, WEIGHTS, RUNTIME)


def test_a_blas_environment_variable_is_part_of_the_backend(monkeypatch):
    monkeypatch.setenv("SOME_BLAS_SETTING", "1")
    assert eval_episodes.blas_backend()["env"].get("SOME_BLAS_SETTING") == "1"
    monkeypatch.setenv("UNRELATED_SETTING", "1")
    assert "UNRELATED_SETTING" not in eval_episodes.blas_backend()["env"]


def test_columns_computed_with_different_backends_are_refused():
    with pytest.raises(ValueError, match="different BLAS backends"):
        eval_episodes.compare(run([group("a")], blas_backend={"x": 1}),
                              run([group("a")], blas_backend={"x": 2}))
    with pytest.raises(ValueError, match="different BLAS backends"):
        eval_episodes.compare(run([group("a")]), run([group("a")], blas_backend={"x": 1}))
    same = eval_episodes.compare(run([group("a")], blas_backend={"x": 1}),
                                 run([group("a")], blas_backend={"x": 1}))
    assert same["backend_verified"] is True


def test_a_column_that_records_no_corpus_is_refused(tmp_path):
    """No way to tell which answer key it was scored against."""
    path = column_file(tmp_path)
    doc = json.loads(path.read_text())
    del doc["config"]["corpus"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="records no corpus digest"):
        eval_episodes.reusable_column(path, "/ckpt/last", _Args(), DIGESTS, BACKEND, SCORER, WEIGHTS, RUNTIME)


def test_a_column_scored_against_other_archives_is_refused(tmp_path):
    """Every setting matches; the ground truth does not."""
    with pytest.raises(ValueError, match=r"different archives for \['a'\]"):
        eval_episodes.reusable_column(column_file(tmp_path), "/ckpt/last", _Args(),
                                      {"a": "another-digest"}, BACKEND, SCORER, WEIGHTS, RUNTIME)


def test_only_the_scenarios_the_column_covers_are_checked(tmp_path):
    """Narrowness: a changed archive the column never scored is not its business."""
    assert eval_episodes.reusable_column(column_file(tmp_path), "/ckpt/last", _Args(scenarios="a"),
                                         {"a": "digest-a", "b": "changed"}, BACKEND, SCORER, WEIGHTS, RUNTIME)


class ReachedError(Exception):
    pass


def _stub_evaluate(monkeypatch):
    import episode_env

    monkeypatch.setattr(episode_env, "corpus_digests", lambda *a, **k: dict(DIGESTS))
    monkeypatch.setattr(eval_episodes, "blas_backend", lambda: dict(BACKEND))
    monkeypatch.setattr(eval_episodes, "scorer_identity", lambda: dict(SCORER))
    monkeypatch.setattr(eval_episodes, "checkpoint_identity", lambda *_a: dict(WEIGHTS))
    monkeypatch.setattr(eval_episodes, "runtime_versions", lambda: dict(RUNTIME))
    calls = []

    def fake(checkpoint, args, done=None, on_progress=None, weights=None):
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
    import episode_env

    calls = _stub_evaluate(monkeypatch)
    monkeypatch.setattr(episode_env, "corpus_digests", lambda *a, **k: dict(TWO))
    path = column_file(tmp_path, scenarios=["a", "b"], corpus=dict(TWO))
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


@pytest.mark.parametrize("flag", ["--max-episode-steps", "--max-new-tokens", "--gen-batch"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_non_positive_sampling_size_is_refused_before_any_work(tmp_path, monkeypatch, capsys,
                                                                  flag, value):
    calls = _stub_evaluate(monkeypatch)
    assert eval_episodes.main(["--after", str(tmp_path / "c"), "--out", str(tmp_path / "o"),
                               flag, value]) == eval_episodes.EXIT_REFUSED
    assert calls == [] and f"{flag} must be >= 1" in capsys.readouterr().err


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


# ---------------------------------------------------------------------------
# the scorer identity: a column is reused only under the code that scored it
# ---------------------------------------------------------------------------

def test_a_column_scored_by_other_code_is_refused(tmp_path):
    """Settings, corpus and backend all match; the reward does not."""
    other = dict(SCORER, sha256="scorer-b")
    with pytest.raises(ValueError, match="scored by different code"):
        eval_episodes.reusable_column(column_file(tmp_path), "/ckpt/last", _Args(), DIGESTS,
                                      BACKEND, other, WEIGHTS, RUNTIME)


def test_a_column_that_records_no_scorer_is_refused(tmp_path):
    path = column_file(tmp_path)
    doc = json.loads(path.read_text())
    del doc["config"]["scorer"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="records no scorer identity"):
        eval_episodes.reusable_column(path, "/ckpt/last", _Args(), DIGESTS, BACKEND, SCORER, WEIGHTS, RUNTIME)


def test_columns_scored_by_different_code_are_not_compared():
    with pytest.raises(ValueError, match="scored by different code"):
        eval_episodes.compare(run([group("a")], scorer={"sha256": "x"}),
                              run([group("a")], scorer={"sha256": "y"}))
    with pytest.raises(ValueError, match="scored by different code"):
        eval_episodes.compare(run([group("a")]), run([group("a")], scorer={"sha256": "x"}))
    same = eval_episodes.compare(run([group("a")], scorer={"sha256": "x"}),
                                 run([group("a")], scorer={"sha256": "x"}))
    assert same["scorer_verified"] is True
    assert eval_episodes.compare(run([group("a")]), run([group("a")]))["scorer_verified"] is False


def _scorer_tree(root: Path) -> tuple[Path, Path]:
    """A miniature checkout: sibling modules plus an ``aorta`` package."""
    here, package = root / "rl", root / "src" / "aorta"
    (package / "agent").mkdir(parents=True)
    here.mkdir()
    (here / "episode_env.py").write_text(
        "import event_reward\nimport aorta.deep.leaf\nfrom aorta.agent import loop\n\n"
        "def lazy():\n    from aorta import lazily_used\n")
    (here / "event_reward.py").write_text("POINTS = {'resolver_named': 4.0}\n")
    (here / "not_a_scorer.py").write_text("X = 1\n")
    (package / "__init__.py").write_text("")
    (package / "agent" / "__init__.py").write_text("")
    (package / "agent" / "loop.py").write_text("MAX = 8\n")
    (package / "deep").mkdir()
    (package / "deep" / "__init__.py").write_text("")
    (package / "deep" / "leaf.py").write_text("W = 1\n")
    (package / "lazily_used.py").write_text("Y = 1\n")
    (package / "unrelated.py").write_text("Z = 1\n")
    return here, package


@pytest.mark.parametrize("edited", [
    "rl/event_reward.py",           # reached through a sibling import
    "src/aorta/agent/loop.py",      # reached through ``from aorta.agent import loop``
    "src/aorta/lazily_used.py",     # an import inside a function
    "src/aorta/agent/__init__.py",  # executed by importing its submodule
    "src/aorta/deep/__init__.py",   # the same, for ``import aorta.deep.leaf``
])
def test_any_edit_to_code_the_scorer_imports_changes_the_identity(tmp_path, edited):
    """The reason for a digest over a version constant: nothing has to remember
    to bump it."""
    here, package = _scorer_tree(tmp_path)
    before = eval_episodes.scorer_identity(here, package)
    target = tmp_path / edited
    target.write_text(target.read_text() + "# edited\n")
    assert eval_episodes.scorer_identity(here, package) != before


@pytest.mark.parametrize("edited", ["rl/not_a_scorer.py", "src/aorta/unrelated.py"])
def test_an_edit_to_code_the_scorer_never_imports_does_not(tmp_path, edited):
    """Narrowness: an unrelated file does not refuse every reuse."""
    here, package = _scorer_tree(tmp_path)
    before = eval_episodes.scorer_identity(here, package)
    target = tmp_path / edited
    target.write_text(target.read_text() + "# edited\n")
    assert eval_episodes.scorer_identity(here, package) == before


def test_the_identity_does_not_depend_on_where_the_checkout_is(tmp_path):
    a = eval_episodes.scorer_identity(*_scorer_tree(tmp_path / "one"))
    b = eval_episodes.scorer_identity(*_scorer_tree(tmp_path / "elsewhere" / "two"))
    assert a == b and a["files"] == 8


def test_the_real_scorer_covers_the_reward_the_loop_and_the_prompts():
    keys = set(eval_episodes.scorer_files())
    assert {"rl:episode_env", "rl:event_reward", "rl:triage_reward", "rl:proposal_reward",
            "rl:train_grpo_step", "rl:eval_episodes",
            "aorta.agent.loop", "aorta.agent.llm", "aorta.agent.policy"} <= keys
    assert "rl:verify_checkpoint_delta" not in keys


# ---------------------------------------------------------------------------
# the checkpoint identity: new weights at the same path are not the same column
# ---------------------------------------------------------------------------

def _tree(root: Path, weights: bytes = b"0123") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(weights)
    (root / "config.json").write_text("{}")
    return root


def test_an_overwrite_with_the_same_size_and_mtime_changes_the_identity(tmp_path):
    """The case a (path, size, mtime) cache would miss: `cp -p` of other weights."""
    import os

    tree = _tree(tmp_path / "ckpt")
    before = eval_episodes.tree_identity(tree)
    stat = (tree / "model.safetensors").stat()
    (tree / "model.safetensors").write_bytes(b"9876")
    os.utime(tree / "model.safetensors", ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert (tree / "model.safetensors").stat().st_size == stat.st_size
    assert eval_episodes.tree_identity(tree) != before


def test_an_untouched_tree_has_the_same_identity_wherever_it_is(tmp_path):
    """Narrowness: reading is not a change, and neither is the directory's name."""
    a = eval_episodes.tree_identity(_tree(tmp_path / "one"))
    assert eval_episodes.tree_identity(tmp_path / "one") == a
    assert eval_episodes.tree_identity(_tree(tmp_path / "elsewhere" / "two")) == a
    assert a["files"] == 2 and a["bytes"] == 6


def test_a_renamed_or_added_file_changes_the_identity(tmp_path):
    tree = _tree(tmp_path / "ckpt")
    before = eval_episodes.tree_identity(tree)
    (tree / "config.json").rename(tree / "generation_config.json")
    assert eval_episodes.tree_identity(tree) != before


def test_the_identity_reads_every_chunk(tmp_path):
    tree = _tree(tmp_path / "ckpt", weights=bytes(100))
    before = eval_episodes.tree_identity(tree, chunk=7)
    (tree / "model.safetensors").write_bytes(bytes(99) + b"\x01")
    assert eval_episodes.tree_identity(tree, chunk=7) != before


def test_a_column_computed_from_other_weights_is_refused(tmp_path):
    other = {**WEIGHTS, "weights": dict(WEIGHTS["weights"], sha256="w-b")}
    with pytest.raises(ValueError, match="different weights"):
        eval_episodes.reusable_column(column_file(tmp_path), "/ckpt/last", _Args(), DIGESTS,
                                      BACKEND, SCORER, other, RUNTIME)


def test_a_column_that_records_no_checkpoint_identity_is_refused(tmp_path):
    path = column_file(tmp_path)
    doc = json.loads(path.read_text())
    del doc["config"]["checkpoint"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="records no checkpoint identity"):
        eval_episodes.reusable_column(path, "/ckpt/last", _Args(), DIGESTS, BACKEND, SCORER,
                                      WEIGHTS, RUNTIME)


def test_same_path_new_weights_is_refused_end_to_end(tmp_path, monkeypatch):
    """The review's case, through the real identity: a finished column is reused
    while the checkpoint is untouched, and refused once the path holds new weights."""
    import episode_env

    ckpt, model = _tree(tmp_path / "ckpt"), _tree(tmp_path / "base", weights=b"base")
    args = _Args(model=str(model))
    monkeypatch.setattr(episode_env, "corpus_digests", lambda *a, **k: dict(DIGESTS))
    monkeypatch.setattr(eval_episodes, "blas_backend", lambda: dict(BACKEND))
    monkeypatch.setattr(eval_episodes, "scorer_identity", lambda: dict(SCORER))
    monkeypatch.setattr(eval_episodes, "runtime_versions", lambda: dict(RUNTIME))
    monkeypatch.setattr(eval_episodes, "evaluate",
                        lambda *a, **k: pytest.fail("evaluated a finished column"))
    path = column_file(tmp_path, init_from=str(ckpt), model=str(model),
                       checkpoint=eval_episodes.checkpoint_identity(ckpt, args))
    assert eval_episodes._column(path, ckpt, "after", args)["config"]["init_from"] == str(ckpt)
    (ckpt / "model.safetensors").write_bytes(b"4567")
    with pytest.raises(ValueError, match="different weights"):
        eval_episodes._column(path, ckpt, "after", args)


def test_the_tokenizer_is_part_of_the_identity(tmp_path):
    ckpt, model = _tree(tmp_path / "ckpt"), _tree(tmp_path / "base")
    args = _Args(model=str(model))
    before = eval_episodes.checkpoint_identity(ckpt, args)
    (model / "config.json").write_text('{"changed": true}')
    after = eval_episodes.checkpoint_identity(ckpt, args)
    assert after["weights"] == before["weights"] and after["tokenizer"] != before["tokenizer"]


def test_a_written_column_records_its_checkpoint_identity():
    from types import SimpleNamespace

    payload = eval_episodes._column_payload(
        _Args(), [SimpleNamespace(scenario_id="a", digest="digest-a")], [], [],
        Path("/ckpt/last"), BACKEND, SCORER, WEIGHTS)
    assert payload["config"]["checkpoint"] == WEIGHTS


# ---------------------------------------------------------------------------
# the paired statistic, beside the unpaired one
# ---------------------------------------------------------------------------

def paired_group(scenario_id, hits):
    g = group(scenario_id, n=len(hits), step1_hits=sum(hits))
    g["step1_hits"] = list(hits)
    return g


def test_mcnemar_uses_only_the_discordant_pairs():
    assert eval_episodes.discordant_pairs([0, 0, 1, 1], [1, 0, 1, 0]) == (1, 1)
    assert eval_episodes.mcnemar_z(9, 1) == pytest.approx(8 / 10**0.5)
    assert eval_episodes.mcnemar_z(0, 0) == 0.0
    with pytest.raises(ValueError, match="cannot pair"):
        eval_episodes.discordant_pairs([0, 1], [0])


def test_the_paired_z_is_reported_beside_the_unpaired_one():
    """Same marginals, different pairing: the unpaired z cannot tell these apart."""
    before = [1] * 4 + [0] * 4
    shifted = eval_episodes.compare(run([paired_group("a", before)]),
                                    run([paired_group("a", [1] * 6 + [0] * 2)]))
    crossed = eval_episodes.compare(run([paired_group("a", before)]),
                                    run([paired_group("a", [0] * 2 + [1] * 6)]))
    assert shifted["pooled_step1_z"] == crossed["pooled_step1_z"]
    assert (shifted["pooled_step1_gained"], shifted["pooled_step1_lost"]) == (2, 0)
    assert (crossed["pooled_step1_gained"], crossed["pooled_step1_lost"]) == (4, 2)
    assert shifted["pooled_step1_paired_z"] == pytest.approx(2 / 2**0.5)
    assert crossed["pooled_step1_paired_z"] == pytest.approx(2 / 6**0.5)


def test_pairs_are_summed_within_scenarios_never_across_them():
    result = eval_episodes.compare(
        run([paired_group("a", [0, 0]), paired_group("b", [1, 1])]),
        run([paired_group("a", [1, 1]), paired_group("b", [1, 0])]))
    rows = {r["scenario_id"]: r for r in result["scenarios"]}
    assert (rows["a"]["step1_gained"], rows["a"]["step1_lost"]) == (2, 0)
    assert (rows["b"]["step1_gained"], rows["b"]["step1_lost"]) == (0, 1)
    assert (result["pooled_step1_gained"], result["pooled_step1_lost"]) == (2, 1)


def test_an_unresolvable_scenario_adds_no_pairs():
    result = eval_episodes.compare(
        run([paired_group("a", [0, 1]), {**group("u", n=2, step1_hits=None), "step1_hits": [0, 1]}]),
        run([paired_group("a", [1, 1]), {**group("u", n=2, step1_hits=None), "step1_hits": [1, 0]}]))
    assert (result["pooled_step1_gained"], result["pooled_step1_lost"]) == (1, 0)


def test_a_column_without_per_episode_outcomes_has_no_paired_z(capsys):
    """A legacy column reports the paired statistic as unavailable, not as zero."""
    result = eval_episodes.compare(run([group("a")]), run([paired_group("a", [1] * 8)]))
    assert result["pooled_step1_paired_z"] is None
    assert result["scenarios"][0]["step1_paired_z"] is None
    eval_episodes.print_comparison(result)
    assert "paired (McNemar) z            unavailable" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the whole-corpus column, and the progress file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corpus, fragment", [
    ({"a": "digest-a", "b": "digest-b"}, "adds ['b'] and no longer has []"),
    ({}, "adds [] and no longer has ['a']"),
])
def test_a_whole_corpus_column_is_refused_once_the_corpus_gains_or_loses_a_scenario(
    tmp_path, corpus, fragment
):
    """Without --scenarios the column stands for the whole corpus; after the
    corpus changes it would otherwise read as complete against its own list."""
    with pytest.raises(ValueError, match=re.escape(fragment)):
        eval_episodes.reusable_column(column_file(tmp_path), "/ckpt/last", _Args(), corpus,
                                      BACKEND, SCORER, WEIGHTS, RUNTIME)


def test_a_whole_corpus_column_over_the_same_corpus_is_reused(tmp_path):
    """Narrowness: same scenario set, same digests."""
    path = column_file(tmp_path, scenarios=["a", "b"], corpus={"a": "digest-a", "b": "digest-b"})
    assert eval_episodes.reusable_column(path, "/ckpt/last", _Args(),
                                         {"b": "digest-b", "a": "digest-a"}, BACKEND, SCORER, WEIGHTS, RUNTIME)


def test_the_progress_file_is_replaced_atomically(tmp_path, monkeypatch):
    """A write that dies midway leaves the previous column readable."""
    import os

    path = tmp_path / "after.json"
    eval_episodes.write_atomically(path, json.dumps({"groups": [1]}))
    real_fsync = os.fsync

    def die(fd):
        real_fsync(fd)
        raise OSError("preempted")

    monkeypatch.setattr(os, "fsync", die)
    with pytest.raises(OSError):
        eval_episodes.write_atomically(path, json.dumps({"groups": [1, 2]}))
    assert json.loads(path.read_text()) == {"groups": [1]}
    monkeypatch.setattr(os, "fsync", real_fsync)
    eval_episodes.write_atomically(path, json.dumps({"groups": [1, 2]}))
    assert json.loads(path.read_text()) == {"groups": [1, 2]}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["after.json"]


def test_the_column_writes_go_through_the_atomic_writer():
    """Both the progress callback and the final write: either one truncating in
    place is the defect."""
    source = Path(eval_episodes.__file__).read_text()
    body = source[source.index("def _column("):source.index("def build_parser(")]
    assert body.count("write_atomically(path,") == 2 and "path.write_text" not in body


# ---------------------------------------------------------------------------
# the generation stack is part of what a column is
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("package", ["transformers", "tokenizers", "safetensors"])
def test_a_column_generated_with_another_package_version_is_refused(tmp_path, package):
    other = dict(RUNTIME, **{package: "9.9"})
    with pytest.raises(ValueError, match="generated with other packages"):
        eval_episodes.reusable_column(column_file(tmp_path), "/ckpt/last", _Args(), DIGESTS,
                                      BACKEND, SCORER, WEIGHTS, other)


def test_a_column_that_records_no_package_versions_is_refused(tmp_path):
    path = column_file(tmp_path)
    doc = json.loads(path.read_text())
    del doc["config"]["runtime"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="records no package versions"):
        eval_episodes.reusable_column(path, "/ckpt/last", _Args(), DIGESTS, BACKEND, SCORER,
                                      WEIGHTS, RUNTIME)


def test_columns_generated_with_different_stacks_are_not_compared():
    with pytest.raises(ValueError, match="different packages"):
        eval_episodes.compare(run([group("a")], runtime={"transformers": "4.1"}),
                              run([group("a")], runtime={"transformers": "4.2"}))
    with pytest.raises(ValueError, match="different packages"):
        eval_episodes.compare(run([group("a")]), run([group("a")], runtime={"transformers": "4.2"}))
    assert eval_episodes.compare(run([group("a")], runtime={"transformers": "4.1"}),
                                 run([group("a")], runtime={"transformers": "4.1"}))


def test_the_recorded_versions_are_the_installed_ones_and_absence_is_explicit(monkeypatch):
    from importlib import metadata

    versions = eval_episodes.runtime_versions()
    assert set(versions) == {"python", *eval_episodes.RUNTIME_PACKAGES}
    for name in eval_episodes.RUNTIME_PACKAGES:
        try:
            assert versions[name] == metadata.version(name)
        except metadata.PackageNotFoundError:
            assert versions[name] is None
    monkeypatch.setattr(eval_episodes, "RUNTIME_PACKAGES", ("no-such-package-xyz",))
    assert eval_episodes.runtime_versions()["no-such-package-xyz"] is None


def test_a_written_column_records_its_package_versions():
    from types import SimpleNamespace

    payload = eval_episodes._column_payload(
        _Args(), [SimpleNamespace(scenario_id="a", digest="digest-a")], [], [],
        Path("/ckpt/last"), BACKEND, SCORER, WEIGHTS, RUNTIME)
    assert payload["config"]["runtime"] == RUNTIME
