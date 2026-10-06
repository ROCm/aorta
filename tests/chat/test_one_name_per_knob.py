"""One setting, whichever dialect you spell it in.

The integration docs lock a single prefix: every chat setting is
``AORTA_CHAT_<NAME>`` or a key in the profile. The cluster work added settings
for facts the agents already had their own names for -- where job records go,
which node to pin, which GPU to build for -- so the same fact had two spellings
and nothing made them agree.

Where that bit was not where it looked. ``jobs_path`` was fine by accident: the
property falls through to the agents' own default, which reads ``CIA_JOBS_ROOT``,
and the chat tools pass their resolved value back down as ``--jobs-root``. The
two that were actually broken were ``CIA_DEMO_NODE``, which the chat side did
not read at all, and ``CIA_GPU_ARCH``, which only the chat side read -- and read
once, at import, so a later change to the setting did nothing.

Each is one field now, answering to both names. Not two fields kept in step,
because two fields is the thing that drifts.
"""

from __future__ import annotations

import pytest

from aorta.chat.config import ENV_PREFIX, Settings, get_settings, reset_settings

#: The facts both halves need, and the agents' own spelling of each.
_SHARED = [
    ("jobs_path", "CIA_JOBS_ROOT", "/tmp/somewhere-shared"),
    ("cia_demo_node", "CIA_DEMO_NODE", "node42"),
    ("gpu_arch", "CIA_GPU_ARCH", "gfx942"),
]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for field, agent_name, _ in _SHARED:
        monkeypatch.delenv(agent_name, raising=False)
        monkeypatch.delenv(f"{ENV_PREFIX}{field.upper()}", raising=False)
    reset_settings()
    yield
    reset_settings()


class TestEitherSpellingConfiguresTheSetting:
    @pytest.mark.parametrize("field,agent_name,value", _SHARED, ids=[f[0] for f in _SHARED])
    def test_the_agents_own_name_is_read(self, monkeypatch, field, agent_name, value):
        """A user following the agents' docs sets this one."""
        monkeypatch.setenv(agent_name, value)
        reset_settings()

        assert getattr(get_settings(), field) == value

    @pytest.mark.parametrize("field,agent_name,value", _SHARED, ids=[f[0] for f in _SHARED])
    def test_the_chat_prefix_is_read(self, monkeypatch, field, agent_name, value):
        """A user following docs/chat/configuration.md sets this one."""
        monkeypatch.setenv(f"{ENV_PREFIX}{field.upper()}", value)
        reset_settings()

        assert getattr(get_settings(), field) == value

    @pytest.mark.parametrize("field,agent_name,value", _SHARED, ids=[f[0] for f in _SHARED])
    def test_the_chat_name_wins_when_both_are_set(self, monkeypatch, field, agent_name, value):
        """The more specific of the two, and the documented precedence."""
        monkeypatch.setenv(agent_name, "from-the-agents")
        monkeypatch.setenv(f"{ENV_PREFIX}{field.upper()}", value)
        reset_settings()

        assert getattr(get_settings(), field) == value


class TestTheProfileStillWorks:
    """An aliased field is matched by its alias alone unless told otherwise.

    Without ``populate_by_name`` the profile key and the constructor argument --
    both of which use the field's own name -- are ignored for every aliased
    field, and the value comes back as the default with no error raised. That
    is a silent failure in the file users are told to edit.
    """

    @pytest.mark.parametrize("field,_agent,value", _SHARED, ids=[f[0] for f in _SHARED])
    def test_the_field_name_is_accepted(self, field, _agent, value):
        assert getattr(Settings(**{field: value}), field) == value

    def test_an_unaliased_field_is_unaffected(self, monkeypatch):
        monkeypatch.setenv(f"{ENV_PREFIX}LLM_TIMEOUT", "42")
        reset_settings()

        assert get_settings().llm_timeout == 42


class TestBothHalvesGetTheSameValue:
    """The point of the shared name: the tools and the agents agree."""

    def _argv(self, monkeypatch) -> list[str]:
        pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
        from unittest.mock import patch

        import aorta.chat.tools.cluster as cluster

        seen: dict = {}

        def fake(argv, *, stop=None):
            seen["argv"] = argv
            return {"ok": True, "job_id": "j", "job_dir": "/tmp"}

        with patch.object(cluster, "run_triage", fake):
            cluster._run_triage(["--source", "k.hip"], "label")
        return seen["argv"]

    def test_the_arch_the_tools_use_is_the_arch_the_agents_get(self, monkeypatch):
        """They were read from two places and could name different chips."""
        pytest.importorskip("dspy", reason="needs the [cia] extra")
        import aorta.chat.tools.cluster as cluster

        monkeypatch.setenv("CIA_GPU_ARCH", "gfx90a")
        reset_settings()
        argv = self._argv(monkeypatch)

        assert cluster._arch() == "gfx90a"
        assert argv[argv.index("--arch") + 1] == "gfx90a"

    def test_the_jobs_root_reaches_the_agents_both_ways(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(tmp_path))
        reset_settings()
        argv = self._argv(monkeypatch)
        forwarded = [argv[i + 1] for i, a in enumerate(argv) if a == "--env"]

        assert argv[argv.index("--jobs-root") + 1] == str(tmp_path)
        assert f"CIA_JOBS_ROOT={tmp_path}" in forwarded

    def test_the_pinned_node_reaches_the_agents(self, monkeypatch):
        monkeypatch.setenv("CIA_DEMO_NODE", "node42")
        reset_settings()
        argv = self._argv(monkeypatch)

        assert argv[argv.index("--node") + 1] == "node42"


class TestTheArchIsNotFrozenAtImport:
    """It was a module constant, so the first import decided it for the process."""

    def test_a_later_change_takes_effect(self, monkeypatch):
        pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
        import aorta.chat.tools.cluster as cluster

        monkeypatch.setenv("CIA_GPU_ARCH", "gfx942")
        reset_settings()
        first = cluster._arch()

        monkeypatch.setenv("CIA_GPU_ARCH", "gfx90a")
        reset_settings()

        assert (first, cluster._arch()) == ("gfx942", "gfx90a")


class TestTheDocsListThem:
    """The finding's other half: configuration.md described a no-cluster product."""

    @staticmethod
    def _doc() -> str:
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        return (root / "docs" / "chat" / "configuration.md").read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "field",
        [
            "jobs_path",
            "gpu_arch",
            "cia_demo_node",
            "rocjitsu_build",
            "rocjitsu_preload",
            "triage_timeout",
            "waitcheck_timeout",
        ],
    )
    def test_each_cluster_setting_is_documented(self, field):
        assert f"`{field}`" in self._doc()

    @pytest.mark.parametrize("_field,agent_name,_v", _SHARED, ids=[f[0] for f in _SHARED])
    def test_the_second_spelling_is_documented_too(self, _field, agent_name, _v):
        """A reader who has the agents' name needs to find it here."""
        assert f"`{agent_name}`" in self._doc()
