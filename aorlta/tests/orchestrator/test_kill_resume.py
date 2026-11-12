"""C6.T1: kill -9 the orchestrator at 10 random points across an iteration (fake components); every resume yields the
same trained state as an uninterrupted run with the same fake results.

Each run is a real process (`python -m aorlta.orchestrator.fakes`). A killed run dies by SIGKILL at the K-th time the
loop reaches a named point, so nothing in it can clean up; a second process resumes the run directory and finishes.
The resumed run is then compared with an uninterrupted one: the trained state (the fake trainer's latent, Adam
moments, step count and RNG, as saved in every checkpoint), every iteration's episodes, summary and TensorBoard
scalars, and the eval requests and results. Timings are left out: the virtual clock restarts with each process.
"""

import json
import os
import random
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from aorlta.episodes.tensorboard import read_scalars, tb_files
from aorlta.episodes.writer import complete_iterations, read_episodes, read_summary
from aorlta.eval.hook import read_request
from aorlta.orchestrator.fakes import FakeRunSpec
from aorlta.orchestrator.state import complete_checkpoints

SRC = Path(__file__).resolve().parents[2] / "src"
PHASES = ("select", "gen.", "score.", "policy.", "train.", "sync.", "record.", "ckpt.", "eval.")
SPEC = {"iterations": 3, "eval_every": 2, "P": 4, "K": 8, "pool_size": 8}
SUMMARY_KEYS = ("samples", "scored_fraction", "status_counts", "unscored_by_reason", "end_reasons", "reward", "gates",
                "compile", "truncation", "length", "cache", "groups", "trainer", "sync", "policy", "deadlines",
                "noop_streak", "warnings")


def harness(spec_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "aorlta.orchestrator.fakes", "--spec", str(spec_path), *args],
                          capture_output=True, text=True, env=env, timeout=300)


def write_spec(root: Path) -> tuple[Path, Path]:
    spec = FakeRunSpec(root=str(root / "runs"), **SPEC)
    path = root / "spec.json"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_json()))
    return path, root / "runs" / "standin" / spec.run_id


def _strip_turn_times(e: dict) -> dict:
    return dict(e, turns=[{k: v for k, v in t.items() if k not in ("t_submit", "t_done")} for t in e["turns"]])


def _summary(run_dir: Path, n: int) -> dict:
    s = read_summary(run_dir, n)
    out = {k: s[k] for k in SUMMARY_KEYS}
    # measured durations are timings: differences of absolute clock values, which a new process offsets
    out["deadlines"] = {k: v for k, v in s["deadlines"].items() if not k.endswith("_dur_s")}
    return out


def _state(path: Path) -> dict:
    state = json.loads(path.read_text())
    state.pop("history")  # the measured phase durations (timings); the deadlines they gave are compared above
    state.pop("config_sha256")  # the config names the run's root directory, which differs between the two runs
    return state


def _request(path: Path, run_dir: Path) -> dict:
    req = vars(read_request(path))
    req = {k: v for k, v in req.items() if k != "requested_at"}
    req["checkpoint"] = dict(req["checkpoint"], path=req["checkpoint"]["path"].replace(str(run_dir), "<run>"))
    return req


def fingerprint(run_dir: Path) -> dict:
    iters = complete_iterations(run_dir)
    ckpts = {n: (_state(run_dir / "ckpt" / f"iter-{n:06d}" / "state.json"),
                 json.loads((run_dir / "ckpt" / f"iter-{n:06d}" / "trainer" / "trainer_state.json").read_text()))
             for n in complete_checkpoints(run_dir / "ckpt")}
    ledger = [json.loads(line) for line in (run_dir / "eval" / "ledger.jsonl").read_text().splitlines()]
    requests = {p.name: _request(p, run_dir) for p in sorted((run_dir / "eval" / "inbox").glob("*.json"))}
    result = json.loads((run_dir / "run_result.json").read_text())
    return {
        "iterations": iters,
        "episodes": {n: [_strip_turn_times(e) for e in read_episodes(run_dir, n)] for n in iters},
        "summaries": {n: _summary(run_dir, n) for n in iters},
        "tb": {n: {k: v for k, v in sc.items() if not k.startswith("time/")}
               for n, sc in read_scalars(run_dir / "tb").items()},
        "tb_files": [n for n, _ in tb_files(run_dir / "tb")],
        "checkpoints": ckpts,
        "eval_requests": requests,
        "eval_events": sorted((e["event"], e.get("request_id") or e.get("export_id"), e["iteration"], e.get("status"))
                              for e in ledger),
        "eval_outbox": sorted(p.name for p in (run_dir / "eval" / "outbox").glob("*.json")),
        "eval_checkpoints_held": sorted(p.name for p in (run_dir / "eval" / "checkpoints").iterdir()),
        "partials": sorted(str(p) for p in run_dir.rglob("*.partial")),
        "result": (result["status"], result["iterations_completed"], result["policy_version"]),
    }


def assert_same_run(got: dict, want: dict) -> None:
    assert got["partials"] == [], f"leftover .partial files: {got['partials']}"
    for n, eps in got["episodes"].items():
        ids = [e["chain_id"] for e in eps]
        assert len(ids) == len(set(ids)) == 32, f"iteration {n}: duplicated or missing episodes"
    for key in want:
        assert got[key] == want[key], f"{key} differs from the uninterrupted run"


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference")
    spec, run_dir = write_spec(root)
    points = root / "points.json"
    proc = harness(spec, "--points-out", str(points))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    fp = fingerprint(run_dir)
    assert fp["iterations"] == [1, 2, 3] and fp["result"] == ("completed", 3, "fake-run:v3")
    return fp, json.loads(points.read_text())


def kill_and_resume(root: Path, *kill_ats: str) -> dict:
    spec, run_dir = write_spec(root)
    for kill_at in kill_ats:
        proc = harness(spec, "--kill-at", kill_at, *(["--resume"] if run_dir.joinpath("run_plan.json").exists() else []))
        assert proc.returncode == -signal.SIGKILL, (
            f"{kill_at}: expected death by SIGKILL, got rc {proc.returncode}\n{proc.stdout[-1500:]}{proc.stderr[-1500:]}")
    proc = harness(spec, "--resume")
    assert proc.returncode == 0, f"resume after {kill_ats}: rc {proc.returncode}\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}"
    return fingerprint(run_dir)


def ten_points(points: list) -> list[str]:
    """One random point in each phase of iteration 2, and one more anywhere in it (seeded)."""
    rng = random.Random(20261102)
    in_two = [(name, it, k) for name, it, k in points if it == 2]
    chosen = []
    for phase in PHASES:
        options = [p for p in in_two if p[0].startswith(phase)]
        assert options, f"no point in phase {phase}"
        chosen.append(rng.choice(options))
    chosen.append(rng.choice([p for p in in_two if p not in chosen]))
    return [f"{n}:{i}:{k}" for n, i, k in chosen]


def test_c6_t1_ten_points_cover_every_phase_of_an_iteration(reference):
    _, points = reference
    chosen = ten_points(points)
    assert len(chosen) == len(set(chosen)) == 10
    assert all(any(c.startswith(p) for c in chosen) for p in PHASES)


@pytest.mark.parametrize("index", range(10))
def test_c6_t1_kill_9_at_a_random_point_then_resume_gives_the_uninterrupted_result(reference, tmp_path, index):
    want, points = reference
    kill_at = ten_points(points)[index]
    got = kill_and_resume(tmp_path, kill_at)
    assert_same_run(got, want)


@pytest.mark.parametrize("kill_at", [
    "gen.chain_end:1:3",  # before anything was committed: the resume starts from scratch
    "ckpt.partial_written:2:1",  # the checkpoint is written but not renamed
    "ckpt.renamed:2:1",  # the checkpoint is in place, its iteration not committed: it must be discarded
    "record.committed:2:1",  # committed, TensorBoard file not yet written: resume writes it from the summary
    "ckpt.pruned:3:1",
    "eval.export_done:2:1",  # the eval checkpoint exists, its request does not
    "eval.request_written:2:1",
])
def test_c6_t1_kill_9_at_each_commit_boundary_then_resume(reference, tmp_path, kill_at):
    want, _ = reference
    assert_same_run(kill_and_resume(tmp_path, kill_at), want)


def test_c6_t1_a_resume_that_is_itself_killed_resumes_again(reference, tmp_path):
    want, _ = reference
    assert_same_run(kill_and_resume(tmp_path, "train.group:2:3", "score.result:3:9"), want)
