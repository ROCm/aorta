"""tools/eval_compare.py over two score runs made with the fake components: the headline numbers are the runs' own
totals, every prompt gets a row, samples are written, and runs that scored different prompts or sampled differently
are refused."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


X = _load("eval_compare", ROOT / "tools" / "eval_compare.py")


def score_run(tmp_path: Path, name: str, **over) -> Path:
    from aorlta.orchestrator.run import main

    c = yaml.safe_load((ROOT / "configs" / "score-standin.yaml").read_text())
    c["run"]["root"] = str(tmp_path / name / "runs")
    c["model"]["base_dir"] = str(tmp_path / name / "base")
    c["data"]["prompt_pools"][0]["path"] = str(tmp_path / "pool")
    c["sync"] = {"dir": str(tmp_path / name / "shm")}
    for k, v in over.items():
        c["rollout"][k] = v
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(c))
    assert main(["--config", str(path), "--components", "fake"]) == 0
    return tmp_path / name / "runs" / "standin" / c["run"]["run_id"]


def test_two_score_runs_compare_side_by_side(tmp_path, capsys):
    base, trained = score_run(tmp_path, "base"), score_run(tmp_path, "trained")
    assert X.main(["--base", str(base), "--trained", str(trained), "--out", str(tmp_path / "out"), "--samples", "3"]) == 0
    result = json.loads((tmp_path / "out" / "eval-compare.json").read_text())
    total = next(json.loads(x) for x in (base / "score" / "report.jsonl").read_text().splitlines()
                 if json.loads(x)["kind"] == "total")
    assert result["base"]["pass_rate"] == total["clean"] / total["samples"]
    assert result["base"]["mean_reward"] == total["reward"]["mean"]
    assert len(result["prompts"]) == 16 and result["delta"]["pass_rate"] == 0.0  # the fakes are seeded alike
    md = (tmp_path / "out" / "eval-compare.md").read_text()
    assert md.count("**base**") == 3 and md.count("**trained**") == 3 and "pass rate" in md
    assert "pass rate" in capsys.readouterr().out


def test_runs_that_sampled_differently_are_refused(tmp_path):
    base, other = score_run(tmp_path, "base"), score_run(tmp_path, "other", temperature=1.0)
    assert X.main(["--base", str(base), "--trained", str(other), "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out" / "eval-compare.json").exists()


def test_a_directory_that_is_not_a_score_run_is_refused(tmp_path):
    base = score_run(tmp_path, "base")
    assert X.main(["--base", str(base), "--trained", str(tmp_path), "--out", str(tmp_path / "out")]) == 2


def test_runs_with_another_batch_layout_or_reward_are_refused(tmp_path, capsys):
    """Seeds follow the iteration number, which a different P changes; a different reward scores another thing."""
    base = score_run(tmp_path, "base")
    other = score_run(tmp_path, "p8")
    _rewrite(other / "score" / "report.jsonl",
             lambda r: dict(r, options=dict(r["options"], prompts_per_batch=8)) if r["kind"] == "run" else r)
    assert X.main(["--base", str(base), "--trained", str(other), "--out", str(tmp_path / "out")]) == 2
    assert "differ in options" in capsys.readouterr().err
    other = score_run(tmp_path, "rfn")
    _rewrite(other / "score" / "report.jsonl",
             lambda r: dict(r, reward=dict(r["reward"], reward_fn="other.v1")) if r["kind"] == "run" else r)
    assert X.main(["--base", str(base), "--trained", str(other), "--out", str(tmp_path / "out")]) == 2
    assert "differ in reward" in capsys.readouterr().err


def _rewrite(report: Path, edit) -> None:
    rows = [json.loads(x) for x in report.read_text().splitlines()]
    report.write_text("".join(json.dumps(edit(r)) + "\n" for r in rows))


def test_runs_whose_tasks_order_or_sample_counts_differ_are_refused(tmp_path, capsys):
    base = score_run(tmp_path, "base")
    first = next(json.loads(x)["prompt_id"] for x in (base / "score" / "report.jsonl").read_text().splitlines()
                 if json.loads(x)["kind"] == "prompt")
    for name, edit, what in (
            ("task", lambda r: dict(r, task_sha256="0" * 64) if r.get("prompt_id") == first else r, "task_sha256"),
            ("count", lambda r: dict(r, samples=r["samples"] - 1) if r.get("prompt_id") == first else r, "samples")):
        other = score_run(tmp_path, name)
        _rewrite(other / "score" / "report.jsonl", edit)
        assert X.main(["--base", str(base), "--trained", str(other), "--out", str(tmp_path / "out")]) == 2
        assert what in capsys.readouterr().err
    other = score_run(tmp_path, "order")
    rep = other / "score" / "report.jsonl"
    rows = [json.loads(x) for x in rep.read_text().splitlines()]
    prompts = [r for r in rows if r["kind"] == "prompt"]
    rows = [rows[0], *reversed(prompts), rows[-1]]
    rep.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert X.main(["--base", str(base), "--trained", str(other), "--out", str(tmp_path / "out")]) == 2
    assert "same order" in capsys.readouterr().err


def test_a_sample_with_no_reply_is_shown_as_such(tmp_path):
    """A chain that died before its first reply has no turns; the report shows it rather than crashing."""
    base, trained = score_run(tmp_path, "base"), score_run(tmp_path, "trained")
    for f in trained.glob("iterations/iter-*/episodes.jsonl"):
        eps = [json.loads(x) for x in f.read_text().splitlines()]
        f.write_text("".join(json.dumps(dict(e, turns=[], final_turn=-1, end_reason="engine_error")) + "\n"
                             for e in eps))
    assert X.main(["--base", str(base), "--trained", str(trained), "--out", str(tmp_path / "out")]) == 0
    md = (tmp_path / "out" / "eval-compare.md").read_text()
    assert "(no reply: the chain ended engine_error before its first turn)" in md


def _edit_episodes(run_dir: Path, edit) -> None:
    for f in run_dir.glob("iterations/iter-*/episodes.jsonl"):
        eps = [json.loads(x) for x in f.read_text().splitlines()]
        f.write_text("".join(json.dumps(edit(e)) + "\n" for e in eps))


def test_runs_whose_prompts_seeds_reward_code_or_plugins_differ_are_refused(tmp_path, capsys):
    """Names can agree while what they name changed: a Q&A template (the first prompt's token hash), the seeds, the
    reward function's code, or a plugin's file."""
    base = score_run(tmp_path, "base")

    def first_turn(**kw):
        return lambda e: dict(e, turns=[dict(e["turns"][0], **kw), *e["turns"][1:]]) if e["turns"] else e

    for name, edit, what in (("prompt", first_turn(prompt_ids_sha256="0" * 64), "first prompts or seeds"),
                             ("seed", first_turn(seed=7), "first prompts or seeds"),
                             ("code", lambda e: dict(e, reward_code_sha256="0" * 64) if e["status"] == "scored"
                              else e, "different reward code")):
        other = score_run(tmp_path, name)
        _edit_episodes(other, edit)
        assert X.main(["--base", str(base), "--trained", str(other), "--out", str(tmp_path / "out")]) == 2, name
        assert what in capsys.readouterr().err
    other = score_run(tmp_path, "plugin")
    plan = json.loads((other / "run_plan.json").read_text())
    plan["plugins"] = {"my_plugin.py": {"module": "m", "file": "f", "sha256": "1" * 64}}
    (other / "run_plan.json").write_text(json.dumps(plan))
    assert X.main(["--base", str(base), "--trained", str(other), "--out", str(tmp_path / "out")]) == 2
    assert "different plugin files" in capsys.readouterr().err
