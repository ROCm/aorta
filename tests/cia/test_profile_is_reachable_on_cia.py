"""The shared profile the agents advertise is one a [cia] install can read.

``chat_provider`` reads ``aorta.chat.config`` so that configuring chat
configures the agents. That module needs pydantic and pydantic-settings, and
``[cia]`` declared neither: they were arriving two hops out, through dspy-ai to
litellm. It worked, and would have gone on working until litellm changed its
own dependencies -- which is the same shape as the certifi note already in that
extra, "not a promise litellm makes".

The failure it would have caused is quiet rather than loud. Every setting has a
default, so falling back to environment-only does not raise; it answers, with
a different endpoint than the operator configured.

And the fallback used to explain itself wrongly. It stated the Python 3.10
reason whatever had actually happened, so an install failing on a missing
package was told it needed 3.11 -- on 3.13. A diagnostic naming the wrong cause
is worse than none, because it gets followed.
"""

from __future__ import annotations

import builtins
import logging
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pytest.importorskip("dspy", reason="the agents need the [cia] extra")

import aorta.cia.llm as llm_mod


def _cia_requirements() -> list[str]:
    spec = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    return spec["project"]["optional-dependencies"]["cia"]


class TestTheExtraDeclaresWhatItReadsTheProfileWith:
    @pytest.mark.parametrize("package", ["pydantic", "pydantic-settings"])
    def test_it_is_declared(self, package):
        declared = " ".join(_cia_requirements())

        assert package in declared, f"[cia] does not declare {package}"

    def test_they_are_what_the_config_module_imports(self):
        """Pinned against the module, so a new import is noticed here."""
        config = (
            Path(llm_mod.__file__).resolve().parents[1] / "chat" / "config.py"
        ).read_text(encoding="utf-8")

        assert "from pydantic import" in config
        assert "from pydantic_settings import" in config

    def test_dspy_is_still_declared(self):
        """The fix must not have displaced what was already there."""
        assert any(r.startswith("dspy-ai") for r in _cia_requirements())

    def test_python_310_gets_the_tomllib_backport(self):
        declared = " ".join(_cia_requirements())

        assert "tomli>=2.0" in declared
        assert "python_version < '3.11'" in declared


class TestTheProfileIsActuallyReadable:
    def test_the_config_module_imports(self):
        """The thing the extra exists to make possible."""
        from aorta.chat.config import settings

        assert settings is not None

    def test_chat_provider_reads_values_from_the_toml_file(
        self, tmp_path, monkeypatch
    ):
        """The 3.10 lane proves tomli reads the profile, not just imports."""
        profile = tmp_path / "aorta" / "chat.toml"
        profile.parent.mkdir(parents=True)
        profile.write_text(
            "\n".join(
                [
                    'llm_provider = "vllm"',
                    'vllm_base_url = "http://profile:4000/v1"',
                    'vllm_api_key = "profile-key"',
                    'vllm_model = "profile-model"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        for name in (
            "AORTA_CHAT_LLM_PROVIDER",
            "AORTA_CHAT_VLLM_BASE_URL",
            "AORTA_CHAT_VLLM_API_KEY",
            "AORTA_CHAT_VLLM_MODEL",
        ):
            monkeypatch.delenv(name, raising=False)

        from aorta.chat.config import reset_settings

        reset_settings()
        try:
            assert llm_mod.chat_provider() == (
                "http://profile:4000/v1",
                "profile-key",
                "profile-model",
                "vllm",
            )
        finally:
            reset_settings()

    def test_chat_provider_reaches_it(self, monkeypatch):
        monkeypatch.setenv("AORTA_CHAT_VLLM_BASE_URL", "http://configured:4000/v1")
        from aorta.chat.config import reset_settings

        reset_settings()
        try:
            resolved = llm_mod.chat_provider()
            assert resolved is not None
            assert resolved[0] == "http://configured:4000/v1"
        finally:
            reset_settings()


class TestTheFallbackReportsWhatHappened:
    @staticmethod
    def _fail_the_import(monkeypatch, message: str):
        real = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("aorta.chat.config"):
                raise ImportError(message)
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        monkeypatch.setattr(llm_mod, "_warned_no_settings", False)

    def test_it_names_the_actual_error(self, monkeypatch, caplog):
        self._fail_the_import(monkeypatch, "No module named 'pydantic_settings'")

        with caplog.at_level(logging.WARNING):
            llm_mod.chat_provider()

        assert "pydantic_settings" in caplog.text

    def test_it_does_not_blame_python_310_on_a_newer_python(self, monkeypatch, caplog):
        """The misdiagnosis: told to upgrade to 3.11, while on 3.13."""
        self._fail_the_import(monkeypatch, "No module named 'pydantic_settings'")

        with caplog.at_level(logging.WARNING):
            llm_mod.chat_provider()

        assert "which is 3.11" not in caplog.text, caplog.text

    def test_it_points_at_the_extra(self, monkeypatch, caplog):
        self._fail_the_import(monkeypatch, "No module named 'pydantic_settings'")

        with caplog.at_level(logging.WARNING):
            llm_mod.chat_provider()

        assert "amd-aorta[cia]" in caplog.text

    def test_it_still_says_the_environment_is_honoured(self, monkeypatch, caplog):
        """The user's next question is whether anything still works."""
        self._fail_the_import(monkeypatch, "No module named 'pydantic_settings'")

        with caplog.at_level(logging.WARNING):
            llm_mod.chat_provider()

        assert "AORTA_CHAT_* is still honoured" in caplog.text

    def test_it_warns_once(self, monkeypatch, caplog):
        """Per round, per job, per module -- this would be a lot of lines."""
        self._fail_the_import(monkeypatch, "No module named 'pydantic_settings'")

        with caplog.at_level(logging.WARNING):
            llm_mod.chat_provider()
            llm_mod.chat_provider()
            llm_mod.chat_provider()

        assert caplog.text.count("Reading the chat settings") == 1

    def test_the_environment_still_answers(self, monkeypatch):
        """Degraded, not broken: that is why the failure is quiet."""
        monkeypatch.setenv("AORTA_CHAT_VLLM_BASE_URL", "http://fallback:4000/v1")
        self._fail_the_import(monkeypatch, "No module named 'pydantic_settings'")

        resolved = llm_mod.chat_provider()

        assert resolved is not None and resolved[0] == "http://fallback:4000/v1"
