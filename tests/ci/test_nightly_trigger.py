"""When the nightly perf measurement runs (issue #486).

nightly-eval.yml fires on the completion of the whole "Nightly wheels" run, so
every job in nightly.yml sits on the measurement's critical path. The chat
index used to be one of them, and its runtime (seconds on an unchanged corpus,
most of an hour on a rebuild) moved the perf slot by ~45 minutes between
nights. These pin the decoupling.
"""

from __future__ import annotations

from pathlib import Path

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
    """Same ordering and publish conditions as when it was `needs: nightly`."""
    doc = _load("chat-index-nightly.yml")
    assert _upstreams(doc) == [_load("nightly.yml")["name"]]
    assert doc[True]["workflow_run"]["types"] == ["completed"]
    assert set(doc[True]) == {"workflow_run"}

    job = doc["jobs"]["chat-index"]
    cond = " ".join(job["if"].split())
    assert "github.repository == 'ROCm/aorta'" in cond
    assert "github.event.workflow_run.conclusion == 'success'" in cond
    assert "github.event.workflow_run.event == 'schedule'" in cond
    assert "github.event.workflow_run.head_branch == 'main'" in cond

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
