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
