"""Submitting work to a cluster is not something an extra should bring with it.

Every other tool keeps a bound the integration docs state plainly: paths
resolve under a configured root and anything that escapes is refused. The three
triage tools do the opposite. They write under ``jobs_root`` rather than the
source root, reach a scheduler over SSH, and run source the user pasted on a
GPU node -- so one chat turn can occupy one for minutes.
``run_terminal_command`` would refuse every one of those operations, and it is
off by default for less.

They were registered as ordinary built-ins, which under
``llm_tool_mode = "native"`` means their schemas went to the provider's
function-calling API alongside ``list_files``. So they are behind
``allow_cluster_jobs`` now, and while it is off they are absent from the
registry rather than refused at call time -- a tool the model is never told
about is one no prompt-injected text can talk it into reaching for.

Reading what a past job already produced stays available. It is the same shape
as the file tools on a different root.
"""

from __future__ import annotations

import pytest

from aorta.chat.config import reset_settings, settings
from aorta.chat.plugins import (
    _DIAGNOSTIC_JOB_TOOLS,
    _DIAGNOSTIC_READ_TOOLS,
    diagnostic_tools,
    enabled_builtins,
)

pytestmark = pytest.mark.usefixtures("_clean_setting")


@pytest.fixture()
def _clean_setting(monkeypatch):
    monkeypatch.delenv("AORTA_CHAT_ALLOW_CLUSTER_JOBS", raising=False)
    reset_settings()
    yield
    reset_settings()


def _enable(monkeypatch) -> None:
    monkeypatch.setenv("AORTA_CHAT_ALLOW_CLUSTER_JOBS", "true")
    reset_settings()


class TestOffByDefault:
    def test_the_setting_defaults_to_false(self):
        assert settings.allow_cluster_jobs is False

    @pytest.mark.parametrize("name", _DIAGNOSTIC_JOB_TOOLS)
    def test_a_submitting_tool_is_not_registered(self, name):
        pytest.importorskip("dspy", reason="needs the [cia] extra")

        assert name not in diagnostic_tools()

    @pytest.mark.parametrize("name", _DIAGNOSTIC_JOB_TOOLS)
    def test_nor_does_it_reach_the_merged_registry(self, name):
        """enabled_builtins is what the graph and the prompts are built from."""
        pytest.importorskip("dspy", reason="needs the [cia] extra")

        assert name not in enabled_builtins()

    def test_it_is_absent_rather_than_refused(self):
        """Under native tool mode every registered schema goes to the provider.

        Refusing at call time would still have told the model the tool exists.
        """
        pytest.importorskip("dspy", reason="needs the [cia] extra")
        from aorta.chat.tools.capabilities import catalogue

        listed = catalogue(enabled_builtins())
        assert not any(name in listed for name in _DIAGNOSTIC_JOB_TOOLS)


class TestReadingPastJobsStillWorks:
    """Bounded to jobs_root, nothing submitted: the file tools' shape."""

    @pytest.mark.parametrize("name", _DIAGNOSTIC_READ_TOOLS)
    def test_a_read_tool_is_registered_without_the_setting(self, name):
        pytest.importorskip("dspy", reason="needs the [cia] extra")

        assert name in diagnostic_tools()

    def test_the_assistant_is_still_useful_about_past_failures(self):
        pytest.importorskip("dspy", reason="needs the [cia] extra")

        assert "list_cluster_jobs" in enabled_builtins()


class TestTurningItOn:
    @pytest.mark.parametrize("name", _DIAGNOSTIC_JOB_TOOLS)
    def test_the_submitting_tools_appear(self, monkeypatch, name):
        pytest.importorskip("dspy", reason="needs the [cia] extra")
        _enable(monkeypatch)

        assert name in diagnostic_tools()

    def test_and_reach_the_merged_registry(self, monkeypatch):
        pytest.importorskip("dspy", reason="needs the [cia] extra")
        _enable(monkeypatch)
        registry = enabled_builtins()

        assert all(name in registry for name in _DIAGNOSTIC_JOB_TOOLS)

    def test_the_welcome_message_says_what_the_session_can_do(self, monkeypatch):
        """A user should not have to read the profile to learn this is on."""
        import importlib

        _enable(monkeypatch)
        welcome = importlib.reload(importlib.import_module("aorta.chat.ui.welcome"))
        said = welcome.capabilities()

        assert "occupy a node for minutes" in said

    def test_and_does_not_when_it_is_off(self, monkeypatch):
        import importlib

        welcome = importlib.reload(importlib.import_module("aorta.chat.ui.welcome"))

        assert "occupy a node" not in welcome.capabilities()


class TestReadingIsStillBounded:
    """A read tool outside its root is the same breach, quieter.

    It took an absolute path unchecked, so a model-supplied one read any
    ``<path>/bundle/report.json`` on the machine. It goes through the rule the
    other file tools share now, rather than a fourth spelling of it -- which is
    what ``_sandbox`` exists to prevent, having already been written three
    times and drifted three ways.
    """

    @staticmethod
    def _read(arg: str) -> str:
        pytest.importorskip("dspy", reason="needs the [cia] extra")
        from aorta.chat.tools.cluster import read_autopsy_report

        return read_autopsy_report.func(arg)

    def test_an_absolute_path_outside_the_jobs_root_is_refused(self, tmp_path):
        outside = tmp_path / "elsewhere"
        (outside / "bundle").mkdir(parents=True)
        (outside / "bundle" / "report.json").write_text(
            '{"category":"leaked"}', encoding="utf-8"
        )

        answer = self._read(str(outside))

        assert "escapes the jobs root" in answer
        assert "leaked" not in answer

    @pytest.mark.parametrize(
        "arg",
        ["../../etc", "/etc", "../cia-jobs-old/x"],
        ids=["traversal", "absolute", "sibling-prefix"],
    )
    def test_an_escape_is_refused(self, arg):
        """The last one is why the rule resolves rather than compares prefixes:
        ``cia-jobs-old`` starts with the characters of ``cia-jobs``."""
        assert "escapes the jobs root" in self._read(arg)

    def test_a_job_id_still_resolves_under_the_root(self):
        answer = self._read("cia-does-not-exist")

        assert "escapes" not in answer
        assert "no report at" in answer

    def test_it_uses_the_shared_rule(self):
        """A fifth copy would drift the same way the first three did."""
        import inspect

        from aorta.chat.tools import cluster

        assert "resolve_within" in inspect.getsource(cluster.read_autopsy_report.func)

    def test_and_no_longer_advertises_absolute_paths(self):
        """The docstring is the tool description the model reads."""
        from aorta.chat.tools import cluster

        assert "absolute" not in (cluster.read_autopsy_report.func.__doc__ or "")


class TestTheBoundIsWrittenDown:
    """The finding's other half: an undocumented exception is not one."""

    @staticmethod
    def _doc(name: str) -> str:
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        return (root / "docs" / "chat" / name).read_text(encoding="utf-8")

    def test_extending_says_these_tools_are_the_exception(self):
        doc = self._doc("extending.md")

        assert "allow_cluster_jobs" in doc
        assert "seccomp=unconfined" in doc, "the container's own bound should be stated"

    def test_it_says_a_turn_can_occupy_a_node(self):
        assert "occupies one for minutes" in self._doc(
            "extending.md"
        ) or "occupy a node" in self._doc("extending.md")

    def test_configuration_documents_the_switch(self):
        doc = self._doc("configuration.md")

        assert "`allow_cluster_jobs`" in doc
        assert "`false`" in doc
