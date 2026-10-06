"""When the nightly perf measurement runs (issue #486).

nightly-eval.yml fires on the completion of the whole "Nightly wheels" run, so
every job in nightly.yml sits on the measurement's critical path. The chat
index used to be one of them, and its runtime (seconds on an unchanged corpus,
most of an hour on a rebuild) moved the perf slot by ~45 minutes between
nights. These pin the decoupling.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text("utf-8"))


def _upstreams(doc: dict) -> list[str]:
    on = doc.get(True)
    if not isinstance(on, dict):
        return []
    return list((on.get("workflow_run") or {}).get("workflows") or [])


def test_the_wheel_workflow_holds_only_the_wheel_job():
    """The eval consumes nothing but the wheel, so nothing else may gate it."""
    nightly = _load("nightly.yml")
    assert nightly["name"] in _upstreams(_load("nightly-eval.yml"))
    assert list(nightly["jobs"]) == ["nightly"], (
        f"nightly.yml has jobs {list(nightly['jobs'])}; nightly-eval.yml waits for all "
        "of them, so each one delays the perf measurement by its runtime. Put "
        "post-wheel work in its own workflow_run workflow."
    )


def test_the_chat_index_follows_a_successful_wheel_run_from_its_own_workflow():
    """Same trigger and publish conditions as when it was `needs: nightly`."""
    doc = _load("chat-index-nightly.yml")
    assert _upstreams(doc) == [_load("nightly.yml")["name"]]
    assert doc[True]["workflow_run"]["types"] == ["completed"]
    assert set(doc[True]) == {"workflow_run"}

    job = doc["jobs"]["chat-index"]
    # Whole, so that loosening an `&&` or dropping an arm cannot pass.
    assert " ".join(job["if"].split()) == (
        "github.repository == 'ROCm/aorta' && "
        "github.event.workflow_run.conclusion == 'success' && "
        "((github.event.workflow_run.event == 'schedule') || "
        "(github.event.workflow_run.event == 'workflow_dispatch' && "
        "github.event.workflow_run.head_branch == 'main'))"
    )

    # Both write the dev-wheels release; the wheel job force-moves its tag.
    assert doc["concurrency"] == _load("nightly.yml")["concurrency"]
    assert doc["concurrency"]["cancel-in-progress"] is False

    checkout = next(s for s in job["steps"] if "actions/checkout" in str(s.get("uses", "")))
    assert checkout["with"]["ref"] == "${{ github.event.workflow_run.head_sha }}", (
        "the index must be built from the commit the wheels were built from, not "
        "whatever main is when the workflow_run fires"
    )


def test_nothing_else_is_chained_behind_the_chat_index():
    """A workflow_run consumer of the index would reintroduce the wait one hop on."""
    index_name = _load("chat-index-nightly.yml")["name"]
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        assert index_name not in _upstreams(_load(path.name)), path.name


def _index_steps() -> list[dict]:
    return _load("chat-index-nightly.yml")["jobs"]["chat-index"]["steps"]


def _position(steps: list[dict], match) -> int:
    hits = [i for i, step in enumerate(steps) if match(step)]
    assert len(hits) == 1, hits
    return hits[0]


def test_the_index_is_published_only_while_the_tag_names_its_commit():
    """The wheel run releases the group before this run exists.

    So a wheel run already waiting on dev-wheels-release moves the tag first,
    and an upload by tag name would attach this commit's index to the newer one.
    """
    steps = _index_steps()
    digest = _position(steps, lambda s: s.get("id") == "digest")
    check = _position(steps, lambda s: s.get("id") == "tag")
    publish = _position(steps, lambda s: "action-gh-release" in str(s.get("uses", "")))
    # Before the digest step, its `changed` output is unset and the check never
    # runs; after the upload, it guards nothing.
    assert digest < check < publish
    assert steps[check]["env"]["BUILT_SHA"] == "${{ github.event.workflow_run.head_sha }}"
    assert steps[publish]["if"] == (
        "steps.digest.outputs.changed == 'true' && steps.tag.outputs.current == 'true'"
    )


_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
}


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, **_GIT_ENV}
    done = subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


@pytest.fixture
def clone(tmp_path: Path) -> tuple[Path, str, str]:
    """A checkout whose origin has two commits on main and no dev-wheels tag."""
    _git(tmp_path, "init", "-q", "--bare", "origin.git")
    work = tmp_path / "work"
    _git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(work))
    shas = []
    for message in ("built", "later"):
        _git(work, "commit", "-q", "--allow-empty", "-m", message)
        shas.append(_git(work, "rev-parse", "HEAD"))
    _git(work, "push", "-q", "origin", "HEAD:refs/heads/main")
    return work, shas[0], shas[1]


def _tag_origin(work: Path, sha: str) -> None:
    _git(work, "tag", "-f", "dev-wheels", sha)
    _git(work, "push", "-q", "--force", "origin", "refs/tags/dev-wheels")


def _run_check(work: Path, built_sha: str) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
    """Run the step as the runner does: bash -e, BUILT_SHA from env:."""
    step = next(s for s in _index_steps() if s.get("id") == "tag")
    script = work.parent / "check.sh"
    script.write_text(step["run"], encoding="utf-8")
    output = work.parent / "github_output"
    output.write_text("", encoding="utf-8")
    env = {**os.environ, **_GIT_ENV, "BUILT_SHA": built_sha, "GITHUB_OUTPUT": str(output)}
    done = subprocess.run(
        ["bash", "-e", str(script)], cwd=work, env=env, capture_output=True, text=True
    )
    pairs = (line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    return done, dict(pairs)


def test_the_tag_check_passes_when_the_tag_names_the_built_commit(clone):
    work, built, _ = clone
    _tag_origin(work, built)
    done, outputs = _run_check(work, built)
    assert done.returncode == 0, done.stderr
    assert outputs == {"current": "true"}


def test_the_tag_check_skips_once_a_later_wheel_run_has_moved_the_tag(clone):
    work, built, later = clone
    _tag_origin(work, later)
    done, outputs = _run_check(work, built)
    assert done.returncode == 0, done.stderr
    assert outputs == {"current": "false"}
    assert f"::warning::dev-wheels points at {later}, not {built}" in done.stdout


@pytest.mark.parametrize("with_sha", [True, False], ids=["built-sha", "empty-sha"])
def test_the_tag_check_skips_when_there_is_no_tag(clone, with_sha):
    work, built, _ = clone
    done, outputs = _run_check(work, built if with_sha else "")
    assert done.returncode == 0, done.stderr
    assert outputs == {"current": "false"}


def test_the_tag_check_fails_rather_than_guess_when_origin_is_unreadable(clone):
    work, built, _ = clone
    _tag_origin(work, built)
    _git(work, "remote", "set-url", "origin", str(work.parent / "missing.git"))
    done, outputs = _run_check(work, built)
    assert done.returncode != 0
    assert outputs == {}
    assert "::error::Could not read the dev-wheels tag" in done.stdout
