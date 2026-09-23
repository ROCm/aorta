"""``aorta laya``: that its hard-coded lists have not drifted, and that it refuses.

Two of the three things tested here are about drift. The Click choices are
hard-coded in ``aorta/cli/laya.py`` rather than read from the registries, for the
reason ``aorta/cli/chat.py`` hard-codes its provider list: a decorator runs at
import time, so enumerating the builders or the checkpoints there would import
them on every ``aorta --help``. That duplication is only safe if something fails
when the two disagree, which is this file.

The third is the refusal. ``eval`` against the fake predictor must stop rather
than print a table of numbers derived from a hash, and it must exit non-zero when
the kill criterion is not met so it works as a gate in a script.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from aorta.cli.laya import _BACKENDS, _CORPUS_KINDS, laya
from aorta.laya.corpus import BUILDERS, write_corpus
from aorta.laya.corpus.schema import LabelledExample
from aorta.laya.predictor import CHECKPOINTS, Noul


def test_the_corpus_kinds_match_the_builders():
    assert sorted(_CORPUS_KINDS) == sorted(BUILDERS)


def test_the_backends_match_the_checkpoints_plus_fake():
    assert sorted(_BACKENDS) == sorted({"fake", *CHECKPOINTS})


def test_the_registry_entry_resolves_and_is_named_after_its_key():
    """``aorta laya`` has to reach this group through the lazy registry."""
    from aorta.cli import _COMMANDS

    assert _COMMANDS["laya"].load() is laya


class TestCorpusCommand:
    def test_an_empty_build_exits_non_zero(self, tmp_path: Path):
        """Building nothing is the Phase 0 gate failing, not a success with zero rows.

        A script that reads it as success goes on to eval a corpus that is not
        there, three steps from the command that produced nothing.
        """
        result = CliRunner().invoke(
            laya, ["corpus", "watch", "--root", str(tmp_path / "absent")]
        )
        assert result.exit_code == 1

    def test_an_empty_build_writes_no_file_to_be_found_later(self, tmp_path: Path):
        target = tmp_path / "corpus.jsonl"
        CliRunner().invoke(
            laya,
            ["corpus", "watch", "--root", str(tmp_path / "absent"), "--output", str(target)],
        )
        assert not target.exists()

    def test_the_proposer_corpus_refuses_to_guess_a_root(self):
        """An agent run's output directory is chosen per run; guessing walks a home dir."""
        result = CliRunner().invoke(laya, ["corpus", "proposer"])
        assert result.exit_code != 0
        assert "--root is required" in result.output

    def test_the_json_form_carries_the_skips_and_the_warnings(self, tmp_path: Path):
        result = CliRunner().invoke(
            laya,
            ["corpus", "watch", "--root", str(tmp_path / "absent"), "--json"],
        )
        payload = json.loads(result.output)
        assert payload["examples"] == 0
        assert payload["warnings"]

    def test_an_unknown_kind_is_rejected_by_the_choice(self):
        result = CliRunner().invoke(laya, ["corpus", "autopsy"])
        assert result.exit_code != 0


def _corpus(path: Path, count: int = 20) -> Path:
    write_corpus(
        path,
        [
            LabelledExample(
                decision="watch_healthy",
                state=f"log {index}",
                question=Noul(question="is this log healthy?"),
                label="true" if index % 2 else "false",
                join_key=f"job-{index}",
            )
            for index in range(count)
        ],
    )
    return path


class TestEvalCommand:
    def test_it_refuses_the_fake_predictor(self, tmp_path: Path):
        """The refusal that keeps a hash out of a results file."""
        result = CliRunner().invoke(
            laya,
            [
                "eval",
                "--corpus",
                str(_corpus(tmp_path / "c.jsonl")),
                "--backend",
                "fake",
            ],
        )
        assert result.exit_code != 0
        assert "hash" in result.output

    def test_a_missing_corpus_says_how_to_build_one(self, tmp_path: Path):
        result = CliRunner().invoke(
            laya, ["eval", "--corpus", str(tmp_path / "absent.jsonl"), "--backend", "fake"]
        )
        assert result.exit_code != 0
        assert "aorta laya corpus" in result.output

    def test_a_missing_laya_install_reads_as_advice_not_a_traceback(self, tmp_path: Path):
        """The distinction ``_load`` draws in ``cli/chat.py``, applied here.

        Skipped where the extra happens to be installed, because then the command
        would go on to try to fetch weights.
        """
        import importlib.util

        if importlib.util.find_spec("laya") is not None:
            pytest.skip("the laya extra is installed, so there is no missing extra to report")
        result = CliRunner().invoke(
            laya,
            [
                "eval",
                "--corpus",
                str(_corpus(tmp_path / "c.jsonl")),
                "--backend",
                "laya-typed-decisions",
            ],
        )
        assert result.exit_code != 0
        assert "amd-aorta[laya]" in result.output
        assert "Traceback" not in result.output
