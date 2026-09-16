"""Two turns staging at once must not write the same file.

Tool calls used to be serialised by the event loop they blocked. They are not
any more -- four run at once on their own executor -- so process-global state
that was previously safe by accident is reachable.

Staging is where that bites. Each triage writes the user's source under
``jobs_root`` before submitting it. When the filename was the kernel name and a
timestamp, two turns staging a kernel of the same name produced the same path,
and one overwrote the other's source while the first was still being read on
the node. The user gets a verdict about somebody else's code, with nothing
anywhere saying so.

A finer timestamp narrowed that window without closing it, because the write
stayed last-one-wins. Each call now stages into a directory of its own, created
with ``O_EXCL``, so a clash is impossible rather than unlikely -- and the writes
inside it are exclusive too, so if that reasoning is ever wrong it raises
instead of silently replacing a sibling call's source.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Importing the cluster tools pulls in aorta.cia.triage, which needs dspy from
# the cia extra. The classes that read cluster.py as text do not, and they are
# the ones worth keeping on a minimal install, so the guard sits on the class
# that imports rather than on the module.
_needs_cluster_tools = pytest.mark.skipif(
    importlib.util.find_spec("dspy") is None,
    reason="staging behaviour needs the cluster tools, which need the cia extra",
)

_CLUSTER = (
    Path(__file__).resolve().parents[2] / "src" / "aorta" / "chat" / "tools" / "cluster.py"
)


class TestEveryStagedPathIsUniquePerCall:
    """Read off the source, because the alternative is a real cluster job."""

    def test_all_three_sites_stage_into_their_own_directory(self):
        """Kernel, assembly and workload each write somewhere only they own."""
        source = _CLUSTER.read_text(encoding="utf-8")

        assert source.count("_stage_dir(") == 4, "three call sites and the helper"

    def test_the_directory_is_created_exclusively(self):
        """mkdtemp retries on a clash rather than handing back a shared path."""
        source = _CLUSTER.read_text(encoding="utf-8")

        assert "tempfile.mkdtemp(" in source

    def test_nothing_stages_by_timestamp_alone(self):
        """A stamp narrows the window; it does not close it."""
        source = _CLUSTER.read_text(encoding="utf-8")

        assert ".write_text(prepared.program" not in source
        assert "script.write_text(source" not in source

    def test_the_writes_are_exclusive(self):
        """So a latent collision is loud rather than silent."""
        source = _CLUSTER.read_text(encoding="utf-8")

        assert 'open(path, "x"' in source


@_needs_cluster_tools
class TestTheCollisionItself:
    def test_two_calls_never_share_a_directory(self, tmp_path):
        from aorta.chat.tools.cluster import _stage_dir

        made = {_stage_dir(tmp_path, "reduce_sum") for _ in range(200)}

        assert len(made) == 200

    def test_a_burst_under_threads_stays_distinct(self, tmp_path):
        """The case that matters: the same name, at the same moment."""
        from concurrent.futures import ThreadPoolExecutor

        from aorta.chat.tools.cluster import _stage_dir

        with ThreadPoolExecutor(max_workers=8) as pool:
            made = list(pool.map(lambda _: _stage_dir(tmp_path, "same"), range(200)))

        assert len(set(made)) == 200, "two calls were handed the same directory"

    def test_an_overwrite_raises_rather_than_replacing(self, tmp_path):
        """What last-one-wins did silently."""
        from aorta.chat.tools.cluster import _write_new

        target = tmp_path / "kernel.hip"
        _write_new(target, "first")

        with pytest.raises(FileExistsError):
            _write_new(target, "second")
        assert target.read_text(encoding="utf-8") == "first"


class TestTheOtherGlobalStateIsGone:
    """The two the review names were fixed on the tools PR; assert they stay so."""

    def test_run_triage_does_not_mutate_the_environment(self):
        source = (
            Path(__file__).resolve().parents[2] / "src" / "aorta" / "cia" / "triage.py"
        ).read_text(encoding="utf-8")
        writes = [
            line
            for line in source.splitlines()
            if "os.environ[" in line or "os.environ.pop" in line
        ]

        assert writes == [], writes

    def test_the_workload_name_is_not_a_constant(self):
        """It was staged/workload.py for every default-labelled request."""
        source = _CLUSTER.read_text(encoding="utf-8")

        assert 'staged / "workload.py"' not in source
        assert '"workload"' in source, "the fallback stem should still exist"

    def test_the_caches_are_per_conversation(self):
        """Process-global dicts were the other thing concurrency exposed."""
        source = _CLUSTER.read_text(encoding="utf-8")

        assert "_TRIAGE_CACHE" not in source
        assert "_ASM_CACHE" not in source
        assert "current_tool_cache()" in source


class TestConcurrentStagingWritesDistinctFiles:
    """Driven through the tool, with the cluster call stubbed out."""

    def test_four_turns_stage_four_files(self, tmp_path, monkeypatch):
        pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
        from concurrent.futures import ThreadPoolExecutor

        import aorta.chat.tools.cluster as cluster
        from aorta.chat.config import reset_settings

        monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(tmp_path))
        reset_settings()

        staged: list[str] = []

        def fake(extra_args, label):
            staged.append(extra_args[extra_args.index("--source") + 1])
            return "Autopsy verdict:\n  category: gpu_race"

        monkeypatch.setattr(cluster, "_run_triage", fake)

        # Four different pastes that wrap to the same kernel name, which is the
        # case that collided. Identical sources would be deduplicated by the
        # per-conversation cache before reaching the staging at all.
        sources = [
            "__global__ void reduce_sum(float* o) "
            f"{{ __shared__ float s[256]; s[{i}] = 1.0f; }}"
            for i in range(4)
        ]

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(
                lambda s: cluster.triage_kernel_source.func(s, label="race"), sources
            ))
        reset_settings()

        assert len(staged) == 4, f"only {len(staged)} reached the cluster"
        assert len(set(staged)) == 4, f"two turns shared a path: {staged}"
        # The kernel names the directory now; the file inside it is plain.
        assert all(Path(p).name == "reduce_sum.hip" for p in staged), staged
        assert all("reduce_sum-" in Path(p).parent.name for p in staged), staged
        assert len({Path(p).parent for p in staged}) == 4, "a directory was shared"
