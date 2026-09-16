"""Two sessions staging the same kernel must not write the same file.

Each triage writes the user's source under ``jobs_root`` before submitting it.
The filename was the kernel name and a timestamp at second resolution, so two
sessions triaging a kernel of the same name in the same second produced the
same path -- and ``write_text`` is last-one-wins. The loser's source was
replaced between staging it and the node reading it, and the job compiled
whichever arrived second with nothing anywhere saying so.

A verdict about somebody else's code is worse than no verdict, because the
answer looks exactly like one about yours.

A finer timestamp narrows that window without closing it. Each call now stages
into a directory of its own, created with ``O_EXCL``, and the writes inside it
are exclusive too -- so if that reasoning is ever wrong it raises rather than
silently replacing a sibling's source.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")

from aorta.chat.tools.cluster import _stage_dir, _write_new

_CLUSTER = (
    Path(__file__).resolve().parents[2] / "src" / "aorta" / "chat" / "tools" / "cluster.py"
)


class TestEachCallGetsItsOwnDirectory:
    def test_two_calls_with_one_name_do_not_share(self, tmp_path):
        first = _stage_dir(tmp_path, "reduce_sum")
        second = _stage_dir(tmp_path, "reduce_sum")

        assert first != second

    def test_a_burst_stays_distinct(self, tmp_path):
        made = {_stage_dir(tmp_path, "reduce_sum") for _ in range(200)}

        assert len(made) == 200

    def test_concurrent_calls_stay_distinct(self, tmp_path):
        """The case the finding names: same name, same instant, two sessions."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            made = list(pool.map(lambda _: _stage_dir(tmp_path, "same"), range(200)))

        assert len(set(made)) == 200, "two calls were handed the same directory"

    def test_the_directory_is_real_and_empty(self, tmp_path):
        work = _stage_dir(tmp_path, "k")

        assert work.is_dir()
        assert list(work.iterdir()) == []

    def test_the_name_is_still_recognisable(self, tmp_path):
        """A staged path is reported to the user, so it should read as theirs."""
        work = _stage_dir(tmp_path, "reduce_sum")

        assert work.name.startswith("reduce_sum-")


class TestTheWritesAreExclusive:
    def test_a_second_write_to_one_path_raises(self, tmp_path):
        target = _stage_dir(tmp_path, "k") / "k.hip"
        _write_new(target, "first")

        with pytest.raises(FileExistsError):
            _write_new(target, "second")

    def test_the_first_source_survives(self, tmp_path):
        """What last-one-wins did, and did silently."""
        target = _stage_dir(tmp_path, "k") / "k.hip"
        _write_new(target, "first")
        with pytest.raises(FileExistsError):
            _write_new(target, "second")

        assert target.read_text(encoding="utf-8") == "first"

    def test_an_ordinary_write_works(self, tmp_path):
        target = _stage_dir(tmp_path, "k") / "k.hip"
        _write_new(target, "__global__ void k() {}")

        assert "__global__" in target.read_text(encoding="utf-8")


class TestAllThreePathsUseIt:
    def test_the_source_stages_three_times(self):
        source = _CLUSTER.read_text(encoding="utf-8")

        assert source.count("_stage_dir(") == 4, "three call sites and the helper"

    def test_nothing_writes_by_timestamped_name_any_more(self):
        source = _CLUSTER.read_text(encoding="utf-8")

        assert ".write_text(prepared.program" not in source
        assert "script.write_text(source" not in source

    def test_the_kernel_path_no_longer_builds_a_shared_name(self):
        source = _CLUSTER.read_text(encoding="utf-8")

        assert 'src_path = staging /' not in source

    def test_the_assembly_path_no_longer_does_either(self):
        source = _CLUSTER.read_text(encoding="utf-8")

        assert 'asm_path = staging /' not in source
