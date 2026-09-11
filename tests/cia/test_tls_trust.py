"""Importing a module must not change TLS verification for the process.

``aorta.cia.llm`` set SSL_CERT_FILE and REQUESTS_CA_BUNDLE to the certifi
bundle at import. Those variables are process-global, so everything imported
afterwards -- unrelated aorta code, the chat provider layer, anything the user
imported alongside -- verified against certifi instead of the system store,
without asking and without saying so.

Certifi is right for a site behind a TLS interception proxy whose CA certifi
knows. It is wrong for the opposite site, whose corporate CA is in the system
store and not in certifi, and there it breaks TLS that worked before the import.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from aorta.cia import llm as llm_mod

CA_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")


@pytest.fixture
def no_ca_env(monkeypatch):
    for var in (*CA_VARS, "CIA_SSL_USE_CERTIFI"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def lm_built(monkeypatch):
    """Build an LM without reaching a network."""
    monkeypatch.setattr(llm_mod.dspy, "LM", lambda **kw: object())
    monkeypatch.setattr(llm_mod.dspy, "configure", lambda **kw: None)
    monkeypatch.setattr(llm_mod, "_configured", False)


class TestImportingIsInert:
    def test_importing_the_module_sets_no_ca_variable(self):
        """In a fresh interpreter, so an earlier import cannot mask it."""
        code = (
            "import os, sys;"
            "import aorta.cia.llm;"
            "print(bool(os.environ.get('SSL_CERT_FILE') or "
            "os.environ.get('REQUESTS_CA_BUNDLE')))"
        )
        env = {
            k: v
            for k, v in __import__("os").environ.items()
            if k not in CA_VARS
        }
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        assert out.stdout.strip().endswith("False"), out.stdout + out.stderr

    def test_the_module_has_no_import_time_side_effect_in_its_source(self):
        import inspect

        header = inspect.getsource(llm_mod).split("def ")[0]
        assert "os.environ.setdefault" not in header
        assert "certifi.where()" not in header


class TestConfiguringMay:
    def test_building_an_lm_points_at_certifi(self, no_ca_env, lm_built, monkeypatch):
        import os

        llm_mod.build_lm()
        assert "certifi" in os.environ.get("SSL_CERT_FILE", "")
        assert "certifi" in os.environ.get("REQUESTS_CA_BUNDLE", "")

    def test_an_existing_setting_is_somebody_having_decided(
        self, no_ca_env, lm_built, monkeypatch
    ):
        import os

        monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/corporate.pem")
        llm_mod.build_lm()
        assert os.environ["SSL_CERT_FILE"] == "/etc/ssl/corporate.pem"

    def test_the_other_variable_alone_also_counts_as_decided(
        self, no_ca_env, lm_built, monkeypatch
    ):
        import os

        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/corporate.pem")
        llm_mod.build_lm()
        assert os.environ.get("SSL_CERT_FILE") is None

    def test_it_can_be_turned_off_for_the_site_it_would_break(
        self, no_ca_env, lm_built, monkeypatch
    ):
        import os

        monkeypatch.setenv("CIA_SSL_USE_CERTIFI", "0")
        llm_mod.build_lm()
        assert os.environ.get("SSL_CERT_FILE") is None

    def test_a_missing_certifi_is_not_a_crash(self, no_ca_env, lm_built, monkeypatch):
        """certifi arrived transitively; it may not be there at all."""
        real_import = __import__

        def no_certifi(name, *args, **kwargs):
            if name == "certifi":
                raise ImportError("no certifi here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", no_certifi)
        llm_mod.build_lm()  # must not raise


def test_certifi_is_declared_rather_than_inherited():
    """It worked only because litellm happened to pull it in."""
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    cia = data["project"]["optional-dependencies"]["cia"]
    assert any(dep.split(";")[0].strip().startswith("certifi") for dep in cia), cia
