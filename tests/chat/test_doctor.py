"""``aorta chat doctor``: report everything, raise nothing, and name the fix.

Two properties matter more than the individual checks.

First, **every check runs even when an earlier one failed.** A user whose chat
session just broke wants the whole list, not the first item on it, and a doctor
that dies partway through has failed at its only job.

Second, **the missing-model check carries the pre-seed procedure.** Decision 21b
publishes the index but not the embedding weights, so an air-gapped user is
blocked twice and discovers the second blocker as a HuggingFace connection
error. The procedure has to appear where they are stuck, because nobody reads
docs at that moment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.chat import doctor
from aorta.chat.config import settings
from aorta.chat.doctor import FAIL, OK, SKIP, WARN, run_checks

MODEL = "BAAI/bge-small-en-v1.5"


def _by_name(report, name: str):
    found = [check for check in report.checks if check.name == name]
    assert found, f"no check named {name!r}; got {[c.name for c in report.checks]}"
    return found[0]


@pytest.fixture(autouse=True)
def offline_but_quiet(monkeypatch, tmp_path: Path):
    """No network probes and no LLM preflight; each test opts into what it needs."""
    monkeypatch.setattr(doctor, "_probe_huggingface", lambda: True)
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "embedding_model", MODEL)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))


class TestStructure:
    def test_it_reports_rather_than_raises(self):
        report = run_checks(backend=False)
        assert report.checks

    def test_python_and_aorta_are_always_reported(self):
        report = run_checks(backend=False)
        assert _by_name(report, "python").status == OK
        assert _by_name(report, "aorta").detail

    def test_the_backend_check_can_be_skipped(self):
        """It needs the network, so it is optional rather than mandatory."""
        assert _by_name(run_checks(backend=False), "llm backend").status == SKIP

    def test_chat_cli_is_reported_as_a_required_extra(self):
        """It is installed in this environment, so it must read as present."""
        assert _by_name(run_checks(backend=False), "extra chat-cli").status == OK

    def test_an_uninstalled_optional_extra_is_a_fact_not_a_finding(self, monkeypatch):
        monkeypatch.setitem(doctor._EXTRA_MODULES, "chat-imaginary", (("no_such_module", "nope"),))
        report = run_checks(backend=False)
        assert _by_name(report, "extra chat-imaginary").status == SKIP
        assert not [c for c in report.checks if c.name.startswith("extra ") and c.status == FAIL]

    def test_a_missing_required_extra_fails_with_an_install_command(self, monkeypatch):
        monkeypatch.setitem(doctor._EXTRA_MODULES, "chat-cli", (("no_such_module", "nope"),))
        check = _by_name(run_checks(backend=False), "extra chat-cli")
        assert check.status == FAIL
        assert "pip install 'amd-aorta[chat-cli]'" in check.hint

    def test_a_check_that_explodes_becomes_a_finding_not_a_traceback(self, monkeypatch):
        """A doctor must not die on its own diagnostics."""

        def _explode(report):
            raise RuntimeError("the probe itself is broken")

        monkeypatch.setattr(doctor, "_check_sqlite", _explode)
        report = run_checks(backend=False)
        assert _by_name(report, "sqlite").status == FAIL
        # And the checks after it still ran.
        assert _by_name(report, "embedding provider")


class TestEmbeddingModelCache:
    def _seed(self, tmp_path: Path) -> None:
        weights = (
            tmp_path
            / "hf"
            / "hub"
            / "models--qdrant--bge-small-en-v1.5-onnx-q"
            / "model_optimized.onnx"
        )
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"\x00")

    def test_a_warm_cache_is_ok(self, tmp_path: Path):
        self._seed(tmp_path)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == OK
        assert MODEL in check.detail

    def test_a_cold_cache_with_egress_is_only_a_warning(self, monkeypatch):
        """It will download itself on first use, so this is information."""
        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: True)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == WARN
        assert "aorta chat index build" in check.hint

    def test_a_cold_cache_with_no_egress_fails_and_prints_the_procedure(self, monkeypatch):
        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: False)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == FAIL
        assert "HF_HOME" in check.procedure
        assert "HF_HUB_OFFLINE=1" in check.procedure

    def test_the_procedure_names_the_directory_to_populate(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: False)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert str(tmp_path / "hf") in check.procedure

    def test_the_procedure_also_covers_the_index_which_is_the_other_blocker(self, monkeypatch):
        """The two are separate artifacts, so one procedure would be half an answer."""
        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: False)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert "index fetch --from" in check.procedure

    def test_a_remote_provider_needs_no_local_weights(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "remote")
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == SKIP

    def test_an_unknown_provider_is_reported_rather_than_raised(self, monkeypatch):
        monkeypatch.setattr(settings, "embedding_provider", "sbert")
        check = _by_name(run_checks(backend=False), "embedding provider")
        assert check.status == FAIL
        assert "unknown embedding provider" in check.detail


class TestIndexChecks:
    def test_an_absent_index_fails_with_both_ways_to_get_one(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(settings, "index_path", str(tmp_path / "absent.sqlite"))
        check = _by_name(run_checks(backend=False), "chat index")
        assert check.status == FAIL
        assert "aorta chat index fetch" in check.hint
        assert "aorta chat index build" in check.hint

    def test_an_index_without_a_manifest_warns(self, monkeypatch, tmp_path: Path):
        index = tmp_path / "index.sqlite"
        index.write_bytes(b"x" * 1024)
        monkeypatch.setattr(settings, "index_path", str(index))
        report = run_checks(backend=False)
        assert _by_name(report, "chat index").status == OK
        assert _by_name(report, "index manifest").status == WARN

    def test_a_matching_manifest_is_ok(self, monkeypatch, tmp_path: Path):
        index = self._write_index(monkeypatch, tmp_path)
        check = _by_name(run_checks(backend=False), "index manifest")
        assert check.status == OK
        assert MODEL in check.detail
        assert index.exists()

    def test_a_mismatched_manifest_fails_and_says_why_it_matters(self, monkeypatch, tmp_path: Path):
        """The report has to convey that this is not cosmetic."""
        self._write_index(monkeypatch, tmp_path, embedding_model="other/model")
        check = _by_name(run_checks(backend=False), "index manifest")
        assert check.status == FAIL
        assert "queries are refused" in check.detail
        assert "not comparable" in check.procedure

    def test_version_drift_warns_rather_than_failing(self, monkeypatch, tmp_path: Path):
        self._write_index(monkeypatch, tmp_path, aorta_version="0.0.1")
        check = _by_name(run_checks(backend=False), "index manifest")
        assert check.status == WARN
        assert any("source drift" in line for line in check.hint.splitlines())

    def _write_index(self, monkeypatch, tmp_path: Path, **overrides) -> Path:
        from aorta.chat.rag import manifest as manifest_mod
        from aorta.chat.rag.embeddings.factory import get_provider

        index = tmp_path / "index.sqlite"
        index.write_bytes(b"x" * 2048)
        monkeypatch.setattr(settings, "index_path", str(index))
        values = {
            "aorta_version": doctor._dist_version("amd-aorta"),
            "aorta_sha": "a" * 40,
            "embedding_provider": "local",
            "embedding_model": MODEL,
            "dimensions": 384,
            "collection": get_provider().collection_name(),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "index_sha256": manifest_mod.sha256_file(index),
        }
        values.update(overrides)
        manifest_mod.write_manifest(index, manifest_mod.Manifest(**values))
        return index


class TestBackendCheck:
    def test_an_unreachable_backend_fails_with_the_underlying_error(self, monkeypatch):
        from aorta.chat.inference.providers import factory

        class _Dead:
            name = "vllm"

            async def preflight(self):
                raise ConnectionError("connection refused")

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _Dead())
        check = _by_name(run_checks(backend=True), "llm backend")
        assert check.status == FAIL
        assert "connection refused" in check.hint

    def test_a_healthy_backend_is_ok(self, monkeypatch):
        from aorta.chat.inference.providers import factory

        class _Alive:
            name = "vllm"

            async def preflight(self):
                return None

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _Alive())
        assert _by_name(run_checks(backend=True), "llm backend").status == OK
