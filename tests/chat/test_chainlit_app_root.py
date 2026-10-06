"""The shipped Chainlit settings have to reach the child process.

Chainlit reads ``.chainlit/config.toml`` under ``CHAINLIT_APP_ROOT``, falling
back to the working directory and generating a default file when it finds
none. The launcher used to set neither, so the settings that applied were
whatever directory the operator was standing in -- and on a wheel, where the
repository file does not exist, that meant Chainlit's own defaults:
``allow_origins = ["*"]`` on a UI whose tools run pasted code on GPU nodes.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from aorta.cli.chat import _chainlit_app_root


@pytest.fixture()
def app_root(tmp_path, monkeypatch):
    """An app root resolved from somewhere that is not the repository."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    return _chainlit_app_root()


def test_settings_exist_away_from_the_checkout(app_root):
    """The config is provisioned rather than left to the working directory."""
    assert (app_root / ".chainlit" / "config.toml").is_file()


def test_origins_are_not_a_wildcard(app_root):
    """The browsers allowed to reach the UI stay an explicit list."""
    settings = tomllib.loads(
        (app_root / ".chainlit" / "config.toml").read_text(encoding="utf-8")
    )
    origins = settings["project"]["allow_origins"]
    assert "*" not in origins, f"any page could call the UI: {origins}"


def test_user_environment_is_masked(app_root):
    """API keys in the environment are not echoed into the transcript."""
    settings = tomllib.loads(
        (app_root / ".chainlit" / "config.toml").read_text(encoding="utf-8")
    )
    assert settings["project"]["mask_user_env"] is True


def test_the_app_root_is_writable(app_root):
    """Chainlit writes uploads and translations here, so it cannot be read-only."""
    probe = app_root / ".chainlit" / ".probe"
    probe.write_text("", encoding="utf-8")
    assert probe.is_file()


def test_operator_edits_survive(app_root, monkeypatch, tmp_path):
    """Seeding happens once; a later launch does not revert local changes."""
    settings = app_root / ".chainlit" / "config.toml"
    settings.write_text(settings.read_text(encoding="utf-8") + "\n# mine\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    assert "# mine" in (_chainlit_app_root() / ".chainlit" / "config.toml").read_text(
        encoding="utf-8"
    )


def test_the_shipped_file_is_package_data():
    """A wheel has no repository root to fall back on, so the file must ship."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    package_data = declared["tool"]["setuptools"]["package-data"]["aorta"]
    assert "chat/ui/chainlit_config.toml" in package_data


def test_the_launcher_points_the_child_at_it():
    """Setting it in the parent would not carry: the child gets an explicit env."""
    source = (
        Path(__file__).resolve().parents[2] / "src" / "aorta" / "cli" / "chat.py"
    ).read_text(encoding="utf-8")
    assert 'child_env["CHAINLIT_APP_ROOT"]' in source


def test_the_checkout_copy_does_not_drift():
    """Developing from the checkout must exercise the settings we ship.

    Chainlit reads the repository's own ``.chainlit/config.toml`` when the
    working directory is the checkout, so a divergence here would mean local
    runs quietly testing something other than what installs get.
    """
    repo = Path(__file__).resolve().parents[2]
    shipped = repo / "src" / "aorta" / "chat" / "ui" / "chainlit_config.toml"
    checkout = repo / ".chainlit" / "config.toml"
    if not checkout.is_file():
        pytest.skip("no checkout-local Chainlit config")
    assert tomllib.loads(shipped.read_text(encoding="utf-8")) == tomllib.loads(
        checkout.read_text(encoding="utf-8")
    ), "the shipped Chainlit config and the checkout's copy have diverged"
