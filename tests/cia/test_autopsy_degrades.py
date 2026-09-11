"""One unreadable artifact does not take the whole verdict with it.

Autopsy runs on a bundle assembled from a job that has just failed, so a sweep
interrupted mid-write leaves a ``matrix.json`` that is half a document. Reading
it was expected to succeed: the orchestrator called ``json.loads`` uncaught, and
so did the adapter — two parses of the same file, neither guarded.

A malformed matrix therefore raised out of ``run_autopsy`` and threw away the
sanitizer and watchdog evidence already collected, over an artifact that may
simply have arrived late. The sanitizer path had always degraded to a tooling
gap instead; this makes the two agree.
"""

from __future__ import annotations

import json
import logging

import pytest

from aorta.cia.autopsy.adapters.aorta_matrix import AortaMatrixAdapter, load_matrix
from aorta.cia.autopsy.adapters.base import BundleContext
from aorta.cia.autopsy.orchestrator import run_autopsy

#: The shapes a bundle actually produces when something went wrong mid-write.
BROKEN = {
    "half written": '{"cells": [{"name": "a"',
    "empty file": "",
    "not json at all": "<html>gateway timeout</html>",
    "json but not an object": "[1, 2, 3]",
    "truncated mid-string": '{"cells": [{"name": "abc',
}


def _bundle(tmp_path, matrix_text: str | None):
    root = tmp_path / "bundle"
    (root / "aorta").mkdir(parents=True)
    (root / "logs").mkdir()
    if matrix_text is not None:
        (root / "aorta" / "matrix.json").write_text(matrix_text)
    (root / "logs" / "watch.stderr.log").write_text("[train] step=5 loss=nan\n")
    (root / "manifest.yaml").write_text(
        "job_id: cia-test\n"
        "paths:\n"
        "  aorta_matrix: aorta/matrix.json\n"
        "  stderr: logs/watch.stderr.log\n"
    )
    return root


class TestAutopsyStillFinishes:
    @pytest.mark.parametrize("label", sorted(BROKEN))
    def test_a_matrix_that_cannot_be_read(self, label, tmp_path):
        result = run_autopsy(_bundle(tmp_path, BROKEN[label]), use_llm=False)
        assert result["category"], f"{label} produced no verdict"

    def test_the_other_evidence_survives(self, tmp_path):
        """The point: the watchdog already had something to say."""
        result = run_autopsy(_bundle(tmp_path, BROKEN["half written"]), use_llm=False)
        assert result["evidence"], "evidence collected before the bad artifact was discarded"
        assert result["category"] == "numeric_silent"

    def test_a_missing_matrix_and_an_unreadable_one_agree(self, tmp_path):
        absent = run_autopsy(_bundle(tmp_path, None), use_llm=False)
        broken = run_autopsy(_bundle(tmp_path / "b", BROKEN["not json at all"]), use_llm=False)
        assert absent["category"] == broken["category"]

    def test_a_readable_matrix_is_still_used(self, tmp_path):
        matrix = {"cells": [{"name": "none-none", "status": "pass"}]}
        result = run_autopsy(_bundle(tmp_path, json.dumps(matrix)), use_llm=False)
        assert result["category"]


class TestTheLoader:
    @pytest.mark.parametrize("label", sorted(BROKEN))
    def test_returns_none_rather_than_raising(self, label, tmp_path):
        path = tmp_path / "matrix.json"
        path.write_text(BROKEN[label])
        assert load_matrix(path) is None

    def test_returns_the_matrix_when_it_is_one(self, tmp_path):
        path = tmp_path / "matrix.json"
        path.write_text('{"cells": []}')
        assert load_matrix(path) == {"cells": []}

    def test_a_missing_path_is_none(self, tmp_path):
        assert load_matrix(tmp_path / "absent.json") is None
        assert load_matrix(None) is None

    def test_it_says_so(self, tmp_path, caplog):
        """Degrading quietly is how the earlier findings in this package began."""
        path = tmp_path / "matrix.json"
        path.write_text(BROKEN["half written"])
        with caplog.at_level(logging.WARNING, logger="aorta.cia.autopsy.adapters.aorta_matrix"):
            load_matrix(path)
        assert caplog.records


class TestTheAdapterDegradesToo:
    """Both readers of this file, not just the one the traceback named."""

    def test_an_unreadable_matrix_is_a_tooling_gap(self, tmp_path):
        root = _bundle(tmp_path, BROKEN["half written"])
        ctx = BundleContext(
            root=root,
            manifest={"paths": {"aorta_matrix": "aorta/matrix.json"}},
            job_id="cia-test",
        )
        artifact = AortaMatrixAdapter().collect(ctx)

        assert artifact.tooling_gaps
        assert not artifact.signals

    def test_it_matches_the_shape_used_for_an_absent_matrix(self, tmp_path):
        root = _bundle(tmp_path, None)
        ctx = BundleContext(
            root=root,
            manifest={"paths": {"aorta_matrix": "aorta/matrix.json"}},
            job_id="cia-test",
        )
        absent = AortaMatrixAdapter().collect(ctx)
        assert absent.tooling_gaps
        assert absent.tooling_gaps[0]["missing_signal"] == "aorta.matrix.json"
