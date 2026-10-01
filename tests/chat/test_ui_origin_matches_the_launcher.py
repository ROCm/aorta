"""What the command serves on, the origin policy has to admit.

Chainlit reads ``allow_origins`` from a config file and from nowhere else --
there is no environment override -- while the bind address arrives as a command
argument. So the two can disagree and nothing notices until a browser tries to
connect: the page loads, the websocket is refused, and it reads as the UI being
broken rather than as a setting being one line short.

It did disagree. The shipped policy listed port 8010 while ``aorta chat ui``
defaulted to 8000, which is every default install. The test guarding it asserted
the same literal the file contained, so it agreed with the file rather than with
the command.

Two things hold it now. The defaults are asserted against the command's own
declared default rather than a number written out again, and the command checks
the effective host and port against the policy in force at startup, naming the
file and the line to add.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytest.importorskip("click", reason="the CLI needs the base install")

from aorta.cli.chat import (
    _configured_origins,
    _warn_if_origin_not_allowed,
    origins_for,
    ui,
)

_REPO = Path(__file__).resolve().parents[2]
_SHIPPED = _REPO / "src" / "aorta" / "chat" / "ui" / "chainlit_config.toml"
_CHECKOUT = _REPO / ".chainlit" / "config.toml"


def _defaults() -> tuple[str, int]:
    params = {p.name: p.default for p in ui.params}
    return params["host"], params["port"]


def _origins(path: Path) -> list[str]:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]["allow_origins"]


class TestTheShippedPolicyAdmitsTheDefaultUI:
    def test_the_command_defaults_to_port_8080(self):
        assert _defaults() == ("127.0.0.1", 8080)

    def test_the_packaged_config_covers_the_default_bind(self):
        """The copy a wheel installs, which is most installs."""
        if not _SHIPPED.is_file():
            pytest.skip("packaged config only present in a source checkout")
        host, port = _defaults()

        assert any(o in _origins(_SHIPPED) for o in origins_for(host, port))

    def test_the_checkout_config_covers_it_too(self):
        """The copy a source checkout runs against."""
        if not _CHECKOUT.is_file():
            pytest.skip("checkout config only present in a source checkout")
        host, port = _defaults()

        assert any(o in _origins(_CHECKOUT) for o in origins_for(host, port))

    def test_the_two_copies_agree(self):
        """Seeding one from the other is only safe while they say the same."""
        if not (_SHIPPED.is_file() and _CHECKOUT.is_file()):
            pytest.skip("both configs only present in a source checkout")

        assert _origins(_SHIPPED) == _origins(_CHECKOUT)


class TestWhichOriginsABindImplies:
    def test_loopback_covers_both_spellings(self):
        """A browser sent to 127.0.0.1 does not send localhost as its origin."""
        assert origins_for("127.0.0.1", 8000) == [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]

    def test_localhost_covers_both_as_well(self):
        assert set(origins_for("localhost", 8000)) == {
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        }

    def test_all_interfaces_is_not_itself_an_origin(self):
        """0.0.0.0 means every interface; the browser sends a real name."""
        assert "http://0.0.0.0:8000" not in origins_for("0.0.0.0", 8000)

    def test_a_named_host_is_taken_at_its_word(self):
        assert origins_for("gpu-01", 9000) == ["http://gpu-01:9000"]

    def test_the_port_travels(self):
        assert all(":9999" in origin for origin in origins_for("127.0.0.1", 9999))


@pytest.fixture()
def rooted(tmp_path):
    """An app root holding a config with the given origins."""

    def make(origins: list[str] | None) -> Path:
        settings = tmp_path / ".chainlit" / "config.toml"
        settings.parent.mkdir(parents=True, exist_ok=True)
        if origins is not None:
            listed = ", ".join(f'"{o}"' for o in origins)
            settings.write_text(f"[project]\nallow_origins = [{listed}]\n")
        return tmp_path

    return make


class TestReadingThePolicyInForce:
    def test_it_reads_the_configured_origins(self, rooted):
        root = rooted(["http://localhost:8000"])

        assert _configured_origins(root) == ["http://localhost:8000"]

    def test_a_missing_file_is_not_an_answer(self, rooted):
        """None, not empty: absent and "admits nothing" are different."""
        assert _configured_origins(rooted(None)) is None

    def test_unparseable_toml_is_not_an_answer_either(self, tmp_path):
        settings = tmp_path / ".chainlit" / "config.toml"
        settings.parent.mkdir(parents=True)
        settings.write_text("this is not toml = = =\n")

        assert _configured_origins(tmp_path) is None


class TestSayingSoBeforeTheBrowserDoes:
    @staticmethod
    def _warning(capsys, root: Path, host: str, port: int) -> str:
        _warn_if_origin_not_allowed(root, host, port)
        return capsys.readouterr().err

    def test_a_matching_origin_says_nothing(self, rooted, capsys):
        root = rooted(["http://localhost:8000", "http://127.0.0.1:8000"])

        assert self._warning(capsys, root, "127.0.0.1", 8000) == ""

    def test_a_port_the_policy_does_not_list_is_called_out(self, rooted, capsys):
        root = rooted(["http://localhost:8000"])

        warning = self._warning(capsys, root, "127.0.0.1", 9000)

        assert "9000" in warning
        assert "refused" in warning

    def test_it_names_the_file_to_edit(self, rooted, capsys):
        root = rooted(["http://localhost:8000"])

        assert "config.toml" in self._warning(capsys, root, "127.0.0.1", 9000)

    def test_it_shows_what_is_currently_allowed(self, rooted, capsys):
        root = rooted(["http://localhost:8000"])

        assert "http://localhost:8000" in self._warning(capsys, root, "127.0.0.1", 9000)

    def test_a_wildcard_is_somebody_elses_problem(self, rooted, capsys):
        """Permissiveness is asserted elsewhere; it does not refuse anything."""
        root = rooted(["*"])

        assert self._warning(capsys, root, "127.0.0.1", 9000) == ""

    def test_an_unreadable_config_does_not_cry_wolf(self, rooted, capsys):
        """Chainlit will write its own; guessing about it helps nobody."""
        assert self._warning(capsys, rooted(None), "127.0.0.1", 9000) == ""

    def test_a_named_host_is_checked_too(self, rooted, capsys):
        root = rooted(["http://localhost:8000", "http://127.0.0.1:8000"])

        assert "gpu-01" in self._warning(capsys, root, "gpu-01", 8000)
