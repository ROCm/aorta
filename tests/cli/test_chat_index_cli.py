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
        for subcommand in ("build", "fetch", "digest", "eval", "runs"):
            assert subcommand in result.output

    def test_the_run_collection_has_a_command_of_its_own(self, runner: CliRunner):
        """``index_run_artifacts()`` had no caller outside its own error text.

        So the run-artifact collection that ``search_run_artifacts`` and the
        tool prompts both advertise could not be built by any documented route.
        """
        result = runner.invoke(chat, ["index", "runs", "--help"])
        assert result.exit_code == 0, result.output
        # Whitespace-normalised: Click rewraps the docstring to the terminal
        # width, so a phrase can land across two lines.
        help_text = " ".join(result.output.split())
        assert "run artifacts" in help_text.lower()
        assert "never part of a published index" in help_text

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


class TestFetchSaysWhatItIsDoingFirst:
    """The reported symptom: `index fetch` printed nothing for minutes.

    Every ``click.echo`` in the command ran after ``fetch_index`` returned, so
    the resolved tag, the URL and the destination -- all known before any
    network I/O -- appeared on success only.
    """

    @staticmethod
    def _unreachable(monkeypatch) -> None:
        import urllib.error
        import urllib.request

        def _refuse(url, timeout=None):  # noqa: ARG001 - signature match
            raise urllib.error.URLError("Network is unreachable")

        monkeypatch.setattr(urllib.request, "urlopen", _refuse)

    def test_the_target_is_printed_even_when_the_fetch_fails(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ):
        self._unreachable(monkeypatch)

        result = runner.invoke(
            chat,
            ["index", "fetch", "--version", "0.2.1", "--output", str(tmp_path / "i.sqlite")],
        )

        assert result.exit_code != 0
        assert "v0.2.1" in result.output
        assert "aorta-chat-index.sqlite" in result.output
        assert str(tmp_path / "i.sqlite") in result.output

    def test_it_goes_to_stderr_so_json_mode_stays_parseable(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ):
        """`--json` is advertised for scripting, so stdout must hold only the object."""
        self._unreachable(monkeypatch)

        result = runner.invoke(
            chat,
            [
                "index",
                "fetch",
                "--version",
                "0.2.1",
                "--json",
                "--output",
                str(tmp_path / "i.sqlite"),
            ],
        )

        assert "Fetching the published index" in result.stderr
        assert "Fetching the published index" not in result.stdout


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


class TestStatus:
    """The read-only comparison, which CI had in bash and the user did not."""

    @staticmethod
    def _serve(monkeypatch, *, manifest_overrides=None, reachable=True) -> None:
        """Publish a manifest the local provider accepts, or nothing at all."""
        import io
        import urllib.error
        import urllib.request

        from aorta.chat.config import settings
        from aorta.chat.rag import manifest as manifest_mod
        from aorta.chat.rag.embeddings.base import build_collection_name
        from aorta.chat.rag.embeddings.fastembed_bge import LOCAL_COLLECTION_PREFIX

        model = "BAAI/bge-small-en-v1.5"
        monkeypatch.setattr(settings, "embedding_provider", "local")
        monkeypatch.setattr(settings, "embedding_model", model)

        values = {
            "aorta_version": "0.2.1",
            "aorta_sha": "b" * 40,
            "embedding_provider": "local",
            "embedding_model": model,
            "dimensions": 384,
            # Derived, so group E's rename flows through with no edit here.
            "collection": build_collection_name(LOCAL_COLLECTION_PREFIX, model),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "index_sha256": "c" * 64,
            "corpus_roots": ["src/aorta", "docs", "README.md"],
            "corpus_digest": "published123",
            "built_at": "2026-09-01T00:00:00+00:00",
        }
        values.update(manifest_overrides or {})
        body = manifest_mod.Manifest(**values).to_json().encode()

        class _Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()
                return False

        def _urlopen(url, timeout=None):  # noqa: ARG001 - signature match
            if not reachable:
                raise urllib.error.URLError("Network is unreachable")
            return _Response(body)

        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    def test_it_is_registered_alongside_the_other_index_commands(self, runner: CliRunner):
        result = runner.invoke(chat, ["index", "--help"])
        assert "status" in result.output

    def test_it_prints_both_sides_and_a_verdict(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ):
        self._serve(monkeypatch)

        result = runner.invoke(
            chat, ["index", "status", "--version", "0.2.1", "--index", str(tmp_path / "absent")]
        )

        assert result.exit_code == 0, result.output
        assert "verdict:" in result.output
        assert "local" in result.output and "published" in result.output
        # Which asset was compared against, since a dev install resolves to
        # the rolling tag and the verdict would otherwise be ambiguous.
        assert "v0.2.1" in result.output

    def test_the_table_names_the_embedding_identity_not_only_the_model(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ):
        """The model name alone is not what has to match for two vectors to compare.

        For a remote provider the identity carries the endpoint too, so two
        rows reading the same ``model`` can still be two vector spaces that
        share a name -- and the *incompatible* verdict could be printed over a
        table showing no visible difference at all.
        """
        self._serve(
            monkeypatch,
            manifest_overrides={"embedding_identity": "https://gateway.internal/v1\nbge-small"},
        )

        result = runner.invoke(
            chat, ["index", "status", "--version", "0.2.1", "--index", str(tmp_path / "absent")]
        )

        assert result.exit_code == 0, result.output
        assert "identity" in result.output
        assert "gateway.internal" in result.output

    def test_a_multi_line_identity_does_not_break_the_columns(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ):
        """``vector_identity`` is newline-joined, and this is a fixed-width table.

        Printed as recorded it would put the model on its own unlabelled row
        and misalign every row after it.
        """
        self._serve(
            monkeypatch,
            manifest_overrides={"embedding_identity": "https://gateway.internal/v1\nbge-small"},
        )

        result = runner.invoke(
            chat, ["index", "status", "--version", "0.2.1", "--index", str(tmp_path / "absent")]
        )

        identity_rows = [
            line for line in result.output.splitlines() if line.strip().startswith("identity")
        ]
        assert len(identity_rows) == 1, result.output
        # Both halves on the one row, so nothing below it is pushed out of column.
        assert "gateway.internal" in identity_rows[0] and "bge-small" in identity_rows[0]

    def test_the_json_form_is_machine_readable(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ):
        self._serve(monkeypatch)

        result = runner.invoke(
            chat,
            [
                "index",
                "status",
                "--version",
                "0.2.1",
                "--index",
                str(tmp_path / "absent"),
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload.get("verdict") == "no_local_index"
        assert payload.get("up_to_date") is False
        assert payload.get("published", {}).get("corpus_digest") == "published123"

    def test_both_sides_carry_the_same_keys_with_no_baseline(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ):
        """``--json`` is a contract, so a key must not vanish in a branch.

        ``published`` was an empty object whenever the baseline could not be
        read, which is the run a consumer most needs to inspect -- and the
        happy-path test above could not see it, because every key is there.
        """
        self._serve(monkeypatch, reachable=False)

        result = runner.invoke(
            chat,
            [
                "index",
                "status",
                "--version",
                "0.2.1",
                "--index",
                str(tmp_path / "absent"),
                "--json",
            ],
        )

        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        from aorta.chat.rag.index_ops import _SIDE_FIELDS

        for key in _SIDE_FIELDS:
            assert key in payload.get("published", {}), f"published lost {key}"
            assert key in payload.get("local", {}), f"local lost {key}"
            assert payload["published"][key] is None
        assert payload.get("baseline_error")

    def test_no_baseline_exits_non_zero_and_is_not_called_up_to_date(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ):
        """The one rendering that would be actively misleading."""
        self._serve(monkeypatch, reachable=False)

        result = runner.invoke(
            chat, ["index", "status", "--version", "0.2.1", "--index", str(tmp_path / "absent")]
        )

        assert result.exit_code == 1
        assert "no published baseline" in result.output
        assert "up to date" not in result.output

    def test_an_unreachable_host_is_a_sentence_not_a_traceback(
        self, runner: CliRunner, tmp_path: Path, monkeypatch
    ):
        self._serve(monkeypatch, reachable=False)

        result = runner.invoke(
            chat, ["index", "status", "--version", "0.2.1", "--index", str(tmp_path / "absent")]
        )

        assert "Traceback" not in result.output


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

        class _NoModelProvider:
            """Answers its identity, refuses to build weights.

            The digest has to name the model whose vectors it describes, or the
            nightly comparison against a remote-built manifest never matches.
            Naming one is cheap; loading one is the 65 MB download this command
            exists to avoid.
            """

            name = "local"

            def model_id(self):
                return "bge-small"

            def get_embeddings(self):
                raise AssertionError("index digest must not build an embedding model")

        monkeypatch.setattr(index_ops, "get_provider", _NoModelProvider)
        tree = tmp_path / "corpus"
        tree.mkdir()
        (tree / "mod.py").write_text("x = 1\n", encoding="utf-8")
        result = runner.invoke(chat, ["index", "digest", "--path", str(tree)])
        assert result.exit_code == 0, result.output
