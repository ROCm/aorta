"""A guard that could not check must not answer as if it checked and passed.

aorta#506. A scorer that folds "could not check" into its pass value produces
a number that reads exactly like a measurement, and nothing downstream can
recover the difference. Each guard below has a could-not-check outcome of its
own -- an exception, ``None``, a non-zero exit or a recorded skip, whichever
fits how the guard is called -- and this holds each one to it.

The contract is that the could-not-check outcome is never the pass outcome. It
is not that it differs from every *fail* outcome: ``serve_for_rollouts.sh
backends`` answers an unreadable ``/get_server_info`` with 57, the code a
greedy engine also gets, which fails closed and is documented as such.

The list is maintained by hand; nothing in the tree marks a function as a
guard, so there is no registry to derive it from. A new guard belongs here.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "rl"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import nccl_roundtrip_check  # noqa: E402
import recipe_reward  # noqa: E402
import triage_reward  # noqa: E402


def _outcome(call):
    """What a caller observes: the value returned, or the type of what was raised."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - the raise is the outcome under test
        return type(exc).__name__


def _novelty_gate(tmp_path):
    """The reward a candidate is paid, against no corpus and against an unlike one."""
    unlike = {
        "recipes/other.yaml": yaml.safe_dump(
            {"schema_version": 1, "workload": "some_other_workload", "trials": 9}
        )
    }
    candidate = recipe_reward._GOOD
    return (
        _outcome(lambda: recipe_reward.grade_recipe_text(candidate, corpus={}).reward),
        _outcome(lambda: recipe_reward.grade_recipe_text(candidate, corpus=unlike).reward),
    )


def _recipe_corpus(tmp_path):
    """The corpus the gate compares against, whole and with one file unreadable."""
    root = tmp_path / "recipes"
    root.mkdir()
    (root / "a.yaml").write_text("schema_version: 1\n")
    passed = _outcome(lambda: sorted(recipe_reward.load_corpus(root)))
    # A directory, so it is unreadable whatever uid the suite runs as.
    (root / "b.yaml").mkdir()
    return _outcome(lambda: sorted(recipe_reward.load_corpus(root))), passed


def _round_trip(tmp_path):
    """The two published observations, with the perturb step rejected and accepted."""
    same = {"texts": ["A", "A", "A"]}
    moved = {"texts": ["B", "B", "B"]}
    accepted = {"start": {"status": 200}, "update": {"status": 200}, "finish": {"status": 200}}
    rejected = {**accepted, "start": {"status": 500}}

    def observed(perturb):
        _, changed, recovered = nccl_roundtrip_check.decide_verdict(
            baseline=same, perturbed=moved, restored=same,
            perturb_lifecycle=perturb, restore_lifecycle=accepted,
        )
        return changed, recovered

    return observed(rejected), observed(accepted)


def _backends(tmp_path):
    """`serve_for_rollouts.sh backends`'s exit, with the control port dead and healthy."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def exit_status(curl_body):
        curl = bin_dir / "curl"
        curl.write_text(f"#!/usr/bin/env bash\n{curl_body}\n")
        curl.chmod(0o755)
        return subprocess.run(
            ["bash", str(EXAMPLES / "serve_for_rollouts.sh"), "backends"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        ).returncode

    return (
        exit_status("exit 7"),
        exit_status(
            "echo '{\"sampling_backend\": \"triton\", \"grammar_backend\": \"xgrammar\"}'"
        ),
    )


def _probe_runs(tmp_path):
    """How many files `--runs --json` reports skipping, with one unreadable and none."""
    root = tmp_path / "runs"
    (root / "good").mkdir(parents=True)
    (root / "good" / "result.json").write_text(
        json.dumps(triage_reward._run("ok", "fail", ["tier1:exit_nonzero"], []))
    )

    def skips():
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            triage_reward.main(["--runs", str(root), "--json"])
        return len(json.loads(out.getvalue())["skipped"])

    passed = _outcome(skips)
    (root / "rotted").mkdir()
    (root / "rotted" / "result.json").write_text("{ not json")
    return _outcome(skips), passed


# guard: (probe returning (could-not-check, passed), passed, could-not-check)
GUARDS = {
    "recipe_reward novelty gate": (_novelty_gate, 1.0, "EmptyCorpus"),
    "recipe_reward.load_corpus": (_recipe_corpus, ["recipes/a.yaml"], "UnreadableCorpus"),
    "nccl_roundtrip_check.decide_verdict": (_round_trip, (True, True), (None, None)),
    "serve_for_rollouts.sh backends": (_backends, 0, 57),
    "triage_reward.load_runs": (_probe_runs, 0, 1),
}


@pytest.mark.parametrize("guard", sorted(GUARDS))
def test_could_not_check_never_reads_as_passed(guard, tmp_path):
    probe, passed, could_not_check = GUARDS[guard]
    assert could_not_check != passed, "the table must not declare the collapse legal"

    unchecked, checked = probe(tmp_path)

    assert checked == passed, f"{guard}: the passing case no longer passes"
    assert unchecked != passed, f"{guard}: could not check, and answered as if it passed"
    assert unchecked == could_not_check, f"{guard}: could-not-check is now {unchecked!r}"
