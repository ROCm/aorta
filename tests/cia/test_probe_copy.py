"""The result comes back the way everything else here reaches the node.

Every other operation in ``probe.py`` goes through the head node --
``_ssh(head_node, "ssh <compute-node> ...")`` -- because many sites reach
compute nodes only through a login host. The copy that fetched the result did
not: it scp'd straight to ``job.node`` from wherever the agent happened to run.
At those sites the sweep succeeded and then the copy failed, which is the
expensive order to fail in.

It also built that command as a string for ``shell=True``, interpolating the
node and the remote path. Both come from the job record, and the node is chosen
by the planner from scheduler output.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from aorta.cia.autopsy import probe
from aorta.cia.launch.job import JobRecord


def _job(**overrides) -> JobRecord:
    fields = {
        "job_id": "cia-test",
        "node": "node-01",
        "recipe": "a recipe",
        "launched_at": "2026-01-01T00:00:00Z",
        "log_path": "/tmp/cia-test/watch.log",
        "aorta_output": "/jobs/cia-test/aorta",
        "head_node": "head-01",
        "recipe_path": "/jobs/cia-test/recipe.yaml",
    }
    return JobRecord(**{**fields, **overrides})


@pytest.fixture
def copied(monkeypatch, tmp_path):
    """Run through to the copy and capture what scp would be given."""
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["shell"] = kwargs.get("shell", False)
        return subprocess.CompletedProcess(cmd, 0)

    # The sweep launch and the wait loop both go through _ssh; make the file
    # appear immediately so the test is about the copy.
    monkeypatch.setattr(
        probe, "_ssh", lambda node, cmd, **kw: subprocess.CompletedProcess([], 0, "EXISTS", "")
    )
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    seen["dest_root"] = tmp_path
    return seen


class TestItGoesThroughTheHeadNode:
    def test_the_copy_jumps_via_the_configured_head_node(self, copied):
        probe.run_aorta_probe(copied["dest_root"], _job())
        cmd = copied["cmd"]
        assert "-J" in cmd, cmd
        assert cmd[cmd.index("-J") + 1].endswith("@head-01")

    def test_it_still_fetches_from_the_compute_node(self, copied):
        probe.run_aorta_probe(copied["dest_root"], _job())
        assert any("@node-01:" in str(a) for a in copied["cmd"])

    def test_an_explicit_head_node_is_the_one_used(self, copied):
        probe.run_aorta_probe(copied["dest_root"], _job(), head_node="jump-99")
        cmd = copied["cmd"]
        assert cmd[cmd.index("-J") + 1].endswith("@jump-99")


class TestNothingReachesALocalShell:
    def test_the_copy_is_an_argv_list(self, copied):
        probe.run_aorta_probe(copied["dest_root"], _job())
        assert isinstance(copied["cmd"], list)
        assert copied["shell"] is False

    def test_the_module_no_longer_uses_shell_true(self):
        """Read the code, not the comment that explains its removal."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(probe))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body:
                first = body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    body.pop(0)
        assert "shell=True" not in ast.unparse(tree)

    def test_the_remote_path_is_quoted_for_the_far_side(self, copied):
        """scp hands the remote half to a shell on the other machine."""
        probe.run_aorta_probe(copied["dest_root"], _job(aorta_output="/jobs/a b"))
        remote = next(a for a in copied["cmd"] if "@node-01:" in str(a))
        assert "'/jobs/a b/matrix.json'" in remote


class TestAHostnameOrNothing:
    @pytest.mark.parametrize(
        "node",
        ["n1; curl attacker.example", "n1 && rm -rf /", "n1 $(whoami)", "a node with spaces", ""],
    )
    def test_a_node_that_is_not_a_hostname_stops_the_probe(self, node, copied, capsys):
        assert probe.run_aorta_probe(copied["dest_root"], _job(node=node)) is None
        assert "cmd" not in copied, "nothing may run for a node we will not accept"

    @pytest.mark.parametrize("node", ["node-01", "cv350-rck-g03-e15-18", "n1.cluster.local", "n1"])
    def test_an_ordinary_hostname_is_accepted(self, node):
        assert probe.valid_host(node)

    @pytest.mark.parametrize("node", ["n1;x", "n1 x", "n1|x", "n1`x`", "", "n1\nx"])
    def test_anything_that_also_means_something_to_a_shell_is_not(self, node):
        assert not probe.valid_host(node)
