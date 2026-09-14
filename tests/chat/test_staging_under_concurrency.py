"""Two turns staging at once must not write the same file.

Tool calls used to be serialised by the event loop they blocked. They are not
any more -- four run at once on their own executor -- so process-global state
that was previously safe by accident is reachable.

Staging is where that bites. Each triage writes the user's source under
``jobs_root`` before submitting it, and the filename is the kernel name and a
timestamp. At second resolution two turns staging a kernel of the same name in
the same second produce the same path, and one overwrites the other's source
while the first is still being read on the node. The user gets a verdict about
somebody else's code, with nothing anywhere saying so.

The workload path was already stamped to the microsecond. The kernel and
assembly paths were not.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

_CLUSTER = (
    Path(__file__).resolve().parents[2] / "src" / "aorta" / "chat" / "tools" / "cluster.py"
)


class TestEveryStagedPathIsUniquePerCall:
    """Read off the source, because the alternative is a real cluster job."""

    def test_every_stamp_carries_microseconds(self):
        stamps = re.findall(r'strftime\("([^"]+)"\)|strftime\(\'([^\']+)\'\)',
                            _CLUSTER.read_text(encoding="utf-8"))
        formats = [a or b for a, b in stamps]

        assert formats, "no timestamps found; has the staging moved?"
        assert all("%f" in f for f in formats), formats

    def test_there_are_three_of_them(self):
        """Kernel, assembly and workload each stage a file."""
        formats = re.findall(r"strftime\(", _CLUSTER.read_text(encoding="utf-8"))

        assert len(formats) == 3


class TestTheCollisionItself:
    @staticmethod
    def _stamp(fmt: str) -> str:
        return datetime.now().strftime(fmt)

    def test_second_resolution_collides(self):
        """What the kernel and assembly paths did."""
        fmt = "%Y%m%d-%H%M%S"

        assert self._stamp(fmt) == self._stamp(fmt), "the premise of the fix"

    def test_microsecond_resolution_does_not(self):
        fmt = "%Y%m%d-%H%M%S-%f"

        assert self._stamp(fmt) != self._stamp(fmt)

    def test_a_burst_of_them_stays_distinct(self):
        fmt = "%Y%m%d-%H%M%S-%f"
        stamps = {self._stamp(fmt) for _ in range(200)}

        assert len(stamps) == 200


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
        assert all("reduce_sum-" in Path(p).name for p in staged), staged
