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


def _write_index(monkeypatch, tmp_path: Path, **overrides) -> Path:
    """Install an index and a matching manifest, and point the settings at it."""
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


@pytest.fixture(autouse=True)
def offline_but_quiet(monkeypatch, tmp_path: Path):
    """No network probes and no LLM preflight; each test opts into what it needs.

    ``index_path`` is pinned at an absent file rather than left alone because
    the embedding-cache hint is now conditioned on index health, so a developer
    who happens to have a real index in ``~/.cache`` would otherwise get a
    different report from CI.
    """
    monkeypatch.setattr(doctor, "_probe_huggingface", lambda: True)
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "embedding_model", MODEL)
    monkeypatch.setattr(settings, "index_path", str(tmp_path / "absent.sqlite"))
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

    def test_a_cold_cache_with_egress_and_no_index_is_only_a_warning(self, monkeypatch):
        """It will download itself on first use, so this is information."""
        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: True)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == WARN
        assert "65 MB" in check.hint

    def test_a_cold_cache_beside_a_healthy_index_is_not_a_warning(
        self, monkeypatch, tmp_path: Path
    ):
        """``index fetch`` never warms the cache, so the happy path landed here.

        A warning that fires on every correctly completed setup teaches people
        to skim the one command that also reports the fatal mismatches.
        """
        _write_index(monkeypatch, tmp_path)
        check = _by_name(run_checks(backend=False), "embedding model cache")
        assert check.status == SKIP
        assert "Nothing to do" in check.hint

    def test_the_pre_warm_advice_never_says_index_build(self, monkeypatch, tmp_path: Path):
        """It would overwrite a fetched index with a code-only one, silently.

        ``index build`` defaults ``--output`` to the index already installed and
        its corpus to ``src/aorta`` alone, so following it drops ``docs/`` and
        ``README.md`` out of retrieval to download a file that downloads itself.
        """
        without_index = _by_name(run_checks(backend=False), "embedding model cache").hint
        _write_index(monkeypatch, tmp_path)
        with_index = _by_name(run_checks(backend=False), "embedding model cache").hint

        for hint in (without_index, with_index):
            assert "index build" not in hint
            assert "TextEmbedding" in hint

    def test_the_pre_warm_command_is_the_one_the_procedure_uses(self, monkeypatch):
        """Two copies of an invocation drift; this is what stops them."""
        from aorta.chat.rag.embeddings.fastembed_bge import (
            PRE_SEED_PROCEDURE,
            describe_model_state,
        )

        monkeypatch.setattr(doctor, "_probe_huggingface", lambda: True)
        hint = _by_name(run_checks(backend=False), "embedding model cache").hint
        command = next(line.strip() for line in hint.splitlines() if "TextEmbedding" in line)

        # Same invocation, different cache_dir: the procedure seeds a machine
        # that has egress, this one seeds the machine being diagnosed. Both
        # halves are pinned whole rather than by a shared prefix, because the
        # prefix stops before ``cache_dir`` -- the argument the two differ on
        # and the one a drift would drop.
        state = describe_model_state()
        assert command == doctor._WARM_COMMAND.format(
            model=state["model"], cache=state["cache_dir"]
        )
        assert MODEL in command
        seeded = doctor._WARM_COMMAND.format(model=MODEL, cache="/tmp/aorta-model-cache")
        assert seeded in PRE_SEED_PROCEDURE.format(model=MODEL, cache="/tmp/cache")

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
        index = _write_index(monkeypatch, tmp_path)
        check = _by_name(run_checks(backend=False), "index manifest")
        assert check.status == OK
        assert MODEL in check.detail
        assert index.exists()

    def test_a_mismatched_manifest_fails_and_says_why_it_matters(self, monkeypatch, tmp_path: Path):
        """The report has to convey that this is not cosmetic."""
        _write_index(monkeypatch, tmp_path, embedding_model="other/model")
        check = _by_name(run_checks(backend=False), "index manifest")
        assert check.status == FAIL
        assert "queries are refused" in check.detail
        assert "not comparable" in check.procedure

    def test_the_mismatch_remedy_offers_fetch_on_a_local_provider(
        self, monkeypatch, tmp_path: Path
    ):
        _write_index(monkeypatch, tmp_path, embedding_model="other/model")
        check = _by_name(run_checks(backend=False), "index manifest")
        assert "aorta chat index fetch" in check.procedure
        # It is already running; suggesting it back is noise.
        assert "aorta chat doctor" not in check.procedure

    def test_the_mismatch_remedy_does_not_lead_with_fetch_on_a_remote_provider(
        self, monkeypatch, tmp_path: Path
    ):
        """CI publishes one asset, built locally, so a fetch would refuse in turn."""
        from aorta.chat.rag import manifest as manifest_mod

        _write_index(monkeypatch, tmp_path, embedding_model="other/model")
        # Only the remedy's view of the provider, not the factory's: building a
        # real remote provider needs an endpoint and a key, which is a different
        # check's problem.
        monkeypatch.setattr(manifest_mod, "_configured_embedding_provider", lambda: "remote")
        check = _by_name(run_checks(backend=False), "index manifest")
        commands = [line for line in check.procedure.splitlines() if line.startswith("  aorta")]
        assert commands == ["  aorta chat index build     re-embed the corpus with the configured"]
        assert "AORTA_CHAT_EMBEDDING_PROVIDER=local" in check.procedure

    def test_version_drift_warns_rather_than_failing(self, monkeypatch, tmp_path: Path):
        _write_index(monkeypatch, tmp_path, aorta_version="0.0.1")
        check = _by_name(run_checks(backend=False), "index manifest")
        assert check.status == WARN
        assert any("source drift" in line for line in check.hint.splitlines())


class TestToolMode:
    """The setup-time signal for a failure that otherwise only shows as a dead query.

    ``text`` is the shipped default and a reasoning model cannot drive it: it
    writes its working to a separate channel and returns empty content where the
    ``ACTION:`` line should be, so the act loop re-prompts until it gives up.
    The user-facing give-up message points here, so this check has to name both
    the problem and the setting that fixes it.
    """

    def test_the_resolved_mode_is_always_reported(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_tool_mode", "native")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == OK
        assert "native" in check.detail

    def test_text_on_a_local_vllm_is_ok_but_still_costs_the_native_flags(self, monkeypatch):
        """A stock vLLM drives text mode, and needs two server flags for native."""
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "vllm")
        monkeypatch.setattr(settings, "vllm_model", "Qwen/Qwen2.5-Coder-7B-Instruct")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == OK
        assert "Qwen/Qwen2.5-Coder-7B-Instruct" in check.detail
        assert "--enable-auto-tool-choice" in check.hint

    def test_a_locally_served_reasoning_model_is_warned_about_too(self, monkeypatch):
        """The reasoning channel belongs to the model, not to the endpoint.

        Reading only ``remote_llm_model`` left gpt-oss on a local vLLM -- the
        configuration ``docs/chat/providers.md`` measures at 0 parseable
        actions in 8 rounds -- with no signal at all, on the provider that is
        the shipped default.
        """
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "vllm")
        monkeypatch.setattr(settings, "vllm_model", "openai/gpt-oss-20b")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == WARN
        assert 'llm_tool_mode = "native"' in check.hint
        # Not the remote remedy: a stock vLLM has to be restarted for native.
        assert "--enable-auto-tool-choice" in check.hint
        assert "gateway" not in check.hint

    def test_a_provider_with_no_model_setting_reads_no_name(self, monkeypatch):
        """Guessing which setting holds it would invent one; the backend check reports it."""
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "not-a-provider")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == OK
        assert not check.hint

    def test_text_with_a_reasoning_model_warns_and_names_the_setting(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "remote_llm_model", "GPT-oss-20B")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == WARN
        assert "GPT-oss-20B" in check.detail
        assert 'llm_tool_mode = "native"' in check.hint
        assert "empty content" in check.hint

    @pytest.mark.parametrize("model", ["gpt-oss-120b", "o3-mini", "deepseek-r1", "Qwen/QwQ-32B"])
    def test_the_reasoning_models_it_recognises(self, monkeypatch, model):
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "litellm")
        monkeypatch.setattr(settings, "remote_llm_model", model)
        assert _by_name(run_checks(backend=False), "llm tool mode").status == WARN

    @pytest.mark.parametrize("model", ["gpt-4o-mini", "claude-sonnet-4", "llama-3.3-70b"])
    def test_a_model_that_can_drive_text_mode_is_not_warned_about(self, monkeypatch, model):
        """A warning on every correct setup is worth less than no warning."""
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "remote_llm_model", model)
        assert _by_name(run_checks(backend=False), "llm tool mode").status == OK

    def test_it_still_points_at_native_for_a_deployment_name_it_cannot_read(self, monkeypatch):
        """A gateway can call a deployment anything, so the OK line has to carry the fix too."""
        monkeypatch.setattr(settings, "llm_tool_mode", "text")
        monkeypatch.setattr(settings, "llm_provider", "openai")
        monkeypatch.setattr(settings, "remote_llm_model", "prod-chat-deployment")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == OK
        assert 'llm_tool_mode = "native"' in check.hint

    def test_an_unknown_mode_fails_before_it_raises_mid_query(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_tool_mode", "function_calling")
        check = _by_name(run_checks(backend=False), "llm tool mode")
        assert check.status == FAIL
        assert "function_calling" in check.detail


class TestBackendCheck:
    """The check calls ``probe``, not ``preflight``.

    Which is the whole point: ``preflight`` is allowed to be permissive, so a
    diagnostic built on it reported ``ok`` for the most likely failure. The
    reachability behaviour itself lives in test_backend_reachability.py; these
    pin the report shape against a stand-in backend.
    """

    def test_an_unreachable_backend_fails_with_the_underlying_error(self, monkeypatch):
        from aorta.chat.inference.providers import factory

        class _Dead:
            name = "vllm"

            async def probe(self, timeout=None):
                raise ConnectionError("connection refused")

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _Dead())
        check = _by_name(run_checks(backend=True), "llm backend")
        assert check.status == FAIL
        assert "connection refused" in check.hint

    def test_an_unexpected_exception_type_is_named_in_the_hint(self, monkeypatch):
        """Only BackendUnreachableError's message stands on its own."""
        from aorta.chat.inference.providers import factory

        class _Weird:
            name = "vllm"

            async def probe(self, timeout=None):
                raise ConnectionError("connection refused")

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _Weird())
        assert _by_name(run_checks(backend=True), "llm backend").hint.startswith("ConnectionError:")

    def test_a_permissive_preflight_is_not_what_gets_called(self, monkeypatch):
        """The false positive, pinned: a backend that starts anyway is still a FAIL."""
        from aorta.chat.inference.providers import factory

        class _StartsAnyway:
            name = "vllm"

            async def preflight(self):
                return None  # what the local backend does after waiting 300s

            async def probe(self, timeout=None):
                raise ConnectionError("connection refused")

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _StartsAnyway())
        assert _by_name(run_checks(backend=True), "llm backend").status == FAIL

    def test_a_healthy_backend_is_ok(self, monkeypatch):
        from aorta.chat.inference.providers import factory

        class _Alive:
            name = "vllm"

            async def probe(self, timeout=None):
                return None

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _Alive())
        assert _by_name(run_checks(backend=True), "llm backend").status == OK

    def test_the_probe_is_given_a_diagnostics_budget(self, monkeypatch):
        """Five minutes is a session's patience, not a waiting operator's."""
        from aorta.chat.inference.providers import factory

        budgets: list[float | None] = []

        class _Recording:
            name = "vllm"

            async def probe(self, timeout=None):
                budgets.append(timeout)

            def describe(self):
                return "vllm at http://localhost:8000/v1"

        monkeypatch.setattr(factory, "get_backend", lambda *a, **k: _Recording())
        run_checks(backend=True)
        assert budgets and budgets[0] is not None and budgets[0] <= 10
