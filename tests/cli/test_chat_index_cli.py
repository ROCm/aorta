"""The ``aorta chat index`` and ``aorta chat doctor`` Click surface.

Same spirit as ``test_chat.py``: the commands are registered, their flags parse,
and the failures users will actually hit come out as sentences rather than
tracebacks. The logic behind them is tested in ``tests/chat/``.

One thing is asserted here and nowhere else: the refusal and pre-seed messages
survive the trip through Click. They are the deliverable of Decisions 20a and
21b, and a ``ClickException`` that swallowed them would look like a pass in
every other test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from aorta.cli.chat import chat

_CHAT_AVAILABLE = importlib.util.find_spec("langchain_core") is not None

pytestmark = pytest.mark.skipif(not _CHAT_AVAILABLE, reason="amd-aorta[chat-cli] not installed")


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestRegistration:
    def test_the_index_group_lists_its_subcommands(self, runner: CliRunner):
        result = runner.invoke(chat, ["index", "--help"])
        assert result.exit_code == 0, result.output
        for subcommand in ("build", "fetch", "digest", "eval"):
            assert subcommand in result.output

    def test_doctor_is_registered(self, runner: CliRunner):
        result = runner.invoke(chat, ["doctor", "--help"])
        assert result.exit_code == 0, result.output

    def test_the_chat_help_lists_index_and_doctor(self, runner: CliRunner):
        result = runner.invoke(chat, ["--help"])
        assert "index" in result.output
        assert "doctor" in result.output

    def test_fetch_documents_both_resolution_and_side_loading(self, runner: CliRunner):
        result = runner.invoke(chat, ["index", "fetch", "--help"])
        assert "--version" in result.output
        assert "--from" in result.output


class TestFlagValidation:
    def test_version_and_from_are_mutually_exclusive(self, runner: CliRunner):
        """One says "resolve for me", the other says "use this file"."""
        result = runner.invoke(chat, ["index", "fetch", "--version", "0.2.1", "--from", "x"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output


class TestBuild:
    def test_a_missing_corpus_path_is_a_sentence_not_a_traceback(
        self, runner: CliRunner, tmp_path: Path
    ):
        result = runner.invoke(chat, ["index", "build", "--path", str(tmp_path / "absent")])
        assert result.exit_code != 0
        assert "corpus path does not exist" in result.output
        assert "Traceback" not in result.output

    def test_public_only_refuses_a_non_public_tree(self, runner: CliRunner, tmp_path: Path):
        """The guard the workflow relies on, exercised through the flag CI passes."""
        result = runner.invoke(chat, ["index", "build", "--public-only", "--path", str(tmp_path)])
        assert result.exit_code != 0
        assert "Traceback" not in result.output


class TestFetchErrors:
    def test_a_mismatch_refusal_reaches_the_user_intact(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ):
        """Decision 20a's whole value is this message being read."""
        from aorta.chat.config import settings
        from aorta.chat.rag import manifest as manifest_mod
        from aorta.chat.rag.index_ops import ASSET_NAME

        monkeypatch.setattr(settings, "embedding_provider", "local")
        monkeypatch.setattr(settings, "embedding_model", "BAAI/bge-small-en-v1.5")

        staged = tmp_path / ASSET_NAME
        staged.write_bytes(b"index bytes")
        manifest_mod.write_manifest(
            staged,
            manifest_mod.Manifest(
                aorta_version="0.2.1",
                aorta_sha="a" * 40,
                embedding_provider="local",
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                dimensions=384,
                collection="aorta",
                chunk_size=512,
                chunk_overlap=50,
                index_sha256=manifest_mod.sha256_file(staged),
            ),
        )

        result = runner.invoke(
            chat,
            ["index", "fetch", "--from", str(staged), "--output", str(tmp_path / "c/i.sqlite")],
        )
        assert result.exit_code != 0
        assert "REFUSING" in result.output
        assert "all-MiniLM-L6-v2" in result.output
        assert "aorta chat index build" in result.output
        assert "Traceback" not in result.output

    def test_a_staged_index_without_a_manifest_is_refused_cleanly(
        self, runner: CliRunner, tmp_path: Path
    ):
        lone = tmp_path / "index.sqlite"
        lone.write_bytes(b"x")
        result = runner.invoke(
            chat, ["index", "fetch", "--from", str(lone), "--output", str(tmp_path / "c/i.sqlite")]
        )
        assert result.exit_code != 0
        assert "no manifest beside" in result.output


class TestDoctorOutput:
    def test_it_exits_non_zero_when_something_is_broken(self, runner: CliRunner, monkeypatch):
        """So it works as a setup gate, not just as a printout."""
        from aorta.chat import doctor as doctor_mod

        report = doctor_mod.Report()
        report.add("chat index", doctor_mod.FAIL, "absent", hint="aorta chat index fetch")
        monkeypatch.setattr(doctor_mod, "run_checks", lambda **kwargs: report)

        result = runner.invoke(chat, ["doctor", "--no-backend"])
        assert result.exit_code == 1
        assert "[FAIL]" in result.output
        assert "aorta chat index fetch" in result.output

    def test_it_exits_zero_when_only_warnings_are_present(self, runner: CliRunner, monkeypatch):
        from aorta.chat import doctor as doctor_mod

        report = doctor_mod.Report()
        report.add("index manifest", doctor_mod.WARN, "source drift")
        monkeypatch.setattr(doctor_mod, "run_checks", lambda **kwargs: report)

        result = runner.invoke(chat, ["doctor", "--no-backend"])
        assert result.exit_code == 0
        assert "[warn]" in result.output

    def test_the_pre_seed_procedure_is_printed_in_full(self, runner: CliRunner, monkeypatch):
        """Decision 21b's mitigation, at the point the user is stuck."""
        from aorta.chat import doctor as doctor_mod
        from aorta.chat.rag.embeddings.fastembed_bge import PRE_SEED_PROCEDURE

        report = doctor_mod.Report()
        report.add(
            "embedding model cache",
            doctor_mod.FAIL,
            "not cached and huggingface.co is unreachable",
            procedure=PRE_SEED_PROCEDURE.format(model="BAAI/bge-small-en-v1.5", cache="/tmp/hf"),
        )
        monkeypatch.setattr(doctor_mod, "run_checks", lambda **kwargs: report)

        result = runner.invoke(chat, ["doctor", "--no-backend"])
        assert "HF_HOME" in result.output
        assert "HF_HUB_OFFLINE=1" in result.output
        assert "/tmp/hf" in result.output

    def test_the_json_form_is_machine_readable(self, runner: CliRunner, monkeypatch):
        from aorta.chat import doctor as doctor_mod

        report = doctor_mod.Report()
        report.add("python", doctor_mod.OK, "3.11.13")
        monkeypatch.setattr(doctor_mod, "run_checks", lambda **kwargs: report)

        result = runner.invoke(chat, ["doctor", "--no-backend", "--json"])
        payload = json.loads(result.output)
        assert payload["ok"] is True
        assert payload["checks"][0]["name"] == "python"


class TestDigest:
    def test_it_prints_json_a_workflow_can_parse(self, runner: CliRunner, tmp_path: Path):
        """nightly.yml reads this to decide whether to rebuild at all."""
        tree = tmp_path / "corpus"
        (tree / "pkg").mkdir(parents=True)
        (tree / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")

        result = runner.invoke(chat, ["index", "digest", "--path", str(tree)])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert len(payload["corpus_digest"]) == 64
        assert payload["files"] == 1

    def test_it_needs_no_embedding_model(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """It has to stay cheap: CI runs it to decide whether to build at all.

        Asserted by making the provider unusable -- if the digest reached for
        an embedder, this would fail rather than pass.
        """
        from aorta.chat.rag import index_ops

        def _explode():
            raise AssertionError("index digest must not build an embedding model")

        monkeypatch.setattr(index_ops, "get_provider", _explode)
        tree = tmp_path / "corpus"
        tree.mkdir()
        (tree / "mod.py").write_text("x = 1\n", encoding="utf-8")
        result = runner.invoke(chat, ["index", "digest", "--path", str(tree)])
        assert result.exit_code == 0, result.output
