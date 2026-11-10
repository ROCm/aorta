"""The run entry point, with the fake components, from a YAML config (the job script's CPU smoke path)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"


def _config(tmp_path: Path) -> Path:
    cfg = f"""
run: {{run_id: cli-smoke, source: standin, root: {tmp_path}/runs}}
model: {{base_dir: {tmp_path}/base, source: standin}}
data: {{prompt_pools: [{{path: {tmp_path}/pool, source: standin}}]}}
rollout: {{engines: [0, 1], K: 8, P: 4, compile_adapter: fake.compile}}
reward: {{profile: fake, reward_fn: standin.v11}}
orchestrator: {{iterations: 2, run_seed: 7, eval_every: 2, min_scored_fraction: 0.75}}
"""
    path = tmp_path / "run.yaml"
    path.write_text(cfg)
    return path


def _main(*args, src=SRC):
    env = dict(os.environ, PYTHONPATH=str(src), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "aorlta.orchestrator.run", *args], capture_output=True, text=True,
                          env=env, timeout=120)


def test_the_entry_point_runs_the_fake_loop_from_a_config_and_resumes(tmp_path):
    cfg = _config(tmp_path)
    proc = _main("--config", str(cfg), "--components", "fake")
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"status": "completed", "iterations_completed": 2, "policy_version": "cli-smoke:v2", "stop": None}
    plan = json.loads((tmp_path / "runs" / "standin" / "cli-smoke" / "run_plan.json").read_text())
    assert plan["config_file"]["path"] == str(cfg.resolve()) and any(k.startswith("base/") for k in plan["extra_files"])
    result = tmp_path / "runs" / "standin" / "cli-smoke" / "run_result.json"
    before = result.read_bytes()
    again = _main("--config", str(cfg), "--components", "fake")
    assert again.returncode == 1 and "already set" in again.stderr  # a run id is set once
    assert result.read_bytes() == before  # and the refused process left the run's records alone
    resumed = _main("--config", str(cfg), "--components", "fake", "--resume")
    assert resumed.returncode == 0 and json.loads(resumed.stdout.strip().splitlines()[-1])["iterations_completed"] == 2


def test_real_components_refuse_what_the_wiring_cannot_run(tmp_path):
    proc = _main("--config", str(_config(tmp_path)))
    assert proc.returncode == 2, proc.stderr[-1500:]
    assert "trainer.ranks is 4" in proc.stderr and "eval_every is set" in proc.stderr
    assert not (tmp_path / "runs").exists()  # refused before the run directory was made


def test_the_fake_components_record_the_commit_given_for_a_snapshot_without_git(tmp_path):
    snapshot = tmp_path / "snapshot" / "src"  # an exported tree with no .git, as the run job executes
    shutil.copytree(SRC, snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    commit = "c0ffee" * 6 + "c0ff"
    proc = _main("--config", str(_config(tmp_path)), "--components", "fake", "--commit", commit, src=snapshot)
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    code = json.loads((tmp_path / "runs" / "standin" / "cli-smoke" / "run_plan.json").read_text())["code"]
    assert (code["commit"], code["dirty"], code["package_dir"]) == (commit, None, str(snapshot / "aorlta"))


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


@pytest.mark.parametrize("make, message", [
    (lambda d: d / "absent.yaml", "No such file or directory"),
    (lambda d: d, "Is a directory"),
    (lambda d: _write(d / "torn.yaml", "run: {run_id: x, source: standin\nmodel: [\n"), "invalid YAML"),
    (lambda d: _write(d / "k1.yaml", _config(d).read_text().replace("K: 8", "K: 1")), "rollout.K: must be >= 2"),
])
def test_a_config_that_does_not_load_is_a_usage_error_with_a_message(tmp_path, make, message):
    path = make(tmp_path)
    for components in ("fake", "real"):
        proc = _main("--config", str(path), "--components", components)
        assert proc.returncode == 2, proc.stderr[-1500:]
        assert proc.stderr.startswith("cannot load the config: ") and message in proc.stderr
        assert "Traceback" not in proc.stderr and proc.stdout == ""
    assert not (tmp_path / "runs").exists()
