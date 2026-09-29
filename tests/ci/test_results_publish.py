"""nightly-eval.yml's publisher, executed against a local ci-results branch.

The step reads its token and repository from the environment rather than
through `${{ }}`, so its `run` body is ordinary bash. These tests run the real
body with git's `insteadOf` pointing the GitHub URL at a local bare repository
and `date` pinned, then read back what landed on the branch (issue #457).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATE = "2026-09-10"
_RUN_ID = "4242"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / "ci" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _publish_body() -> str:
    doc = yaml.safe_load((_REPO_ROOT / ".github/workflows/nightly-eval.yml").read_text("utf-8"))
    for step in doc["jobs"]["publish"]["steps"]:
        if step.get("name") == "Update history branch (ci-results)":
            return step["run"]
    raise AssertionError("nightly-eval.yml's publish job has no ci-results update step")


def _record(generated_at: str, upstream_run_id) -> dict:
    return {
        "generated_at": generated_at,
        "build": {"amd_aorta_version": "0.3.1", "lane": "gate", "upstream_run_id": upstream_run_id},
        "summary": {"total": 1, "pass": 1, "fail": 0, "record": 0, "skip": 0},
        "entries": [],
    }


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout


class _Branch:
    """A bare repository standing in for origin, seeded with a ci-results branch."""

    def __init__(self, root: Path, existing: dict[str, dict]):
        self.root = root
        self.bare = root / "remote" / "o" / "r.git"
        self.bare.mkdir(parents=True)
        _git("init", "-q", "--bare", cwd=self.bare)
        seed = root / "seed"
        seed.mkdir()
        _git("init", "-q", "-b", "ci-results", cwd=seed)
        for rel, doc in existing.items():
            path = seed / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc), encoding="utf-8")
        _git("add", "-A", cwd=seed)
        _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty",
             "-m", "seed", cwd=seed)
        _git("push", "-q", str(self.bare), "ci-results", cwd=seed)

    def publish(self, incoming: dict) -> subprocess.CompletedProcess:
        work = self.root / "job"
        (work / "incoming").mkdir(parents=True, exist_ok=True)
        (work / "incoming" / "gpu-nightly-results.json").write_text(
            json.dumps(incoming), encoding="utf-8")
        bin_dir = self.root / "bin"
        bin_dir.mkdir(exist_ok=True)
        (bin_dir / "date").write_text(f"#!/bin/sh\necho {_DATE}\n", encoding="utf-8")
        (bin_dir / "date").chmod(0o755)
        env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "GH_TOKEN": "tok",
            "GITHUB_REPOSITORY": "o/r",
            "GITHUB_RUN_ID": _RUN_ID,
            "TMPDIR": str(self.root),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"url.file://{self.root / 'remote'}/.insteadOf",
            "GIT_CONFIG_VALUE_0": "https://x-access-token:tok@github.com/",
        }
        return subprocess.run(
            ["bash", "-c", _publish_body()], cwd=work, env=env,
            capture_output=True, text=True, timeout=120,
        )

    def checkout(self) -> Path:
        out = self.root / "readback"
        if out.exists():
            _git("fetch", "-q", "origin", "ci-results", cwd=out)
            _git("reset", "-q", "--hard", "FETCH_HEAD", cwd=out)
        else:
            _git("clone", "-q", "--branch", "ci-results", str(self.bare), str(out), cwd=self.root)
        return out

    def subject(self) -> str:
        return _git("log", "-1", "--format=%s", "ci-results", cwd=self.bare).strip()


def _scheduled_path() -> str:
    return f"results/{_DATE}.json"


def test_a_dispatch_leaves_that_dates_scheduled_record_byte_for_byte_intact(tmp_path):
    scheduled = _record(f"{_DATE}T11:19:55Z", "34115289502")
    branch = _Branch(tmp_path, {_scheduled_path(): scheduled})
    before = (tmp_path / "seed" / _scheduled_path()).read_bytes()

    proc = branch.publish(_record(f"{_DATE}T13:08:55Z", ""))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    tree = branch.checkout()
    assert (tree / _scheduled_path()).read_bytes() == before
    dispatched = tree / "results" / "dispatch" / f"{_DATE}-{_RUN_ID}.json"
    assert json.loads(dispatched.read_text("utf-8"))["generated_at"] == f"{_DATE}T13:08:55Z"
    assert "dispatch" in branch.subject() and _RUN_ID in branch.subject()


@pytest.mark.parametrize("upstream", ["", None, "missing"])
def test_every_spelling_of_no_upstream_run_is_routed_as_a_dispatch(tmp_path, upstream):
    """nightly_eval.py records `os.environ.get("UPSTREAM_RUN_ID")`, so an unset
    variable gives null rather than "". Neither may reach the date key."""
    incoming = _record(f"{_DATE}T13:08:55Z", upstream)
    if upstream == "missing":
        del incoming["build"]["upstream_run_id"]
    branch = _Branch(tmp_path, {})

    proc = branch.publish(incoming)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    tree = branch.checkout()
    assert not (tree / _scheduled_path()).exists()
    assert (tree / "results" / "dispatch" / f"{_DATE}-{_RUN_ID}.json").is_file()


def test_a_wheel_triggered_run_still_writes_the_date_key(tmp_path):
    """The narrowness half: the scheduled lane is unchanged, including the
    last-write-wins re-publish of its own date (a re-run of that night)."""
    branch = _Branch(tmp_path, {_scheduled_path(): _record(f"{_DATE}T11:00:00Z", "111")})

    proc = branch.publish(_record(f"{_DATE}T11:19:55Z", "222"))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    tree = branch.checkout()
    doc = json.loads((tree / _scheduled_path()).read_text("utf-8"))
    assert doc["build"]["upstream_run_id"] == "222"
    assert not (tree / "results" / "dispatch").exists()
    assert branch.subject() == f"nightly results {_DATE}"


def test_dispatches_are_invisible_to_the_gated_series_and_its_retention(tmp_path):
    """180 scheduled nights plus a dispatch: all 180 nights survive the prune,
    and the dashboard reader sees exactly one point per scheduled night."""
    existing = {
        f"results/2026-{m:02d}-{d:02d}.json": _record(f"2026-{m:02d}-{d:02d}T11:19:55Z", str(n))
        for n, (m, d) in enumerate(
            [(m, d) for m in range(3, 10) for d in range(1, 29)][:180])
    }
    existing["results/dispatch/2026-01-01-1.json"] = _record("2026-01-01T13:00:00Z", "")
    branch = _Branch(tmp_path, existing)

    proc = branch.publish(_record(f"{_DATE}T13:08:55Z", ""))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    tree = branch.checkout()
    assert len(list((tree / "results").glob("*.json"))) == 180
    assert len(list((tree / "results" / "dispatch").glob("*.json"))) == 2
    series = _load("gen_dashboard").load_results(tree / "results")
    assert len(series) == 180
    assert all(doc["build"]["upstream_run_id"] for doc in series)


def test_the_dispatch_directory_keeps_its_own_180_file_window(tmp_path):
    existing = {
        f"results/dispatch/2026-{m:02d}-{d:02d}-{n}.json": _record(f"2026-{m:02d}-{d:02d}T13:00:00Z", "")
        for n, (m, d) in enumerate(
            [(m, d) for m in range(1, 9) for d in range(1, 29)][:180])
    }
    existing[_scheduled_path()] = _record(f"{_DATE}T11:19:55Z", "34115289502")
    branch = _Branch(tmp_path, existing)

    proc = branch.publish(_record(f"{_DATE}T13:08:55Z", ""))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    tree = branch.checkout()
    kept = sorted(p.name for p in (tree / "results" / "dispatch").glob("*.json"))
    assert len(kept) == 180
    assert "2026-01-01-0.json" not in kept
    assert f"{_DATE}-{_RUN_ID}.json" in kept
    assert (tree / _scheduled_path()).is_file()


def test_a_dispatch_onto_a_branch_with_no_scheduled_record_yet_publishes(tmp_path):
    """results/ holds no top-level file after a dispatch-only publish, which
    is the case the retention prune has to tolerate under pipefail."""
    branch = _Branch(tmp_path, {})

    proc = branch.publish(_record(f"{_DATE}T13:08:55Z", ""))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not list((branch.checkout() / "results").glob("*.json"))
