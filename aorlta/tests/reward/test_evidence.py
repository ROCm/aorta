"""The GPU legs' evidence is kept: before the orchestrator removes its node-local scratch, the sanitizer legs' logs,
harness sources and JSON, and each parsed source, go into the run's evidence store, one tarball per attempt. Copying
the scratch from the job script afterwards gives an empty archive, because the scratch is already gone by then."""

from __future__ import annotations

import tarfile
from pathlib import Path

from aorlta.config import config_from_dict
from aorlta.orchestrator import wiring
from aorlta.reward.clock import Clock
from aorlta.reward.evidence import archive_legs, kept, next_archive
from aorlta.spans import SpanRecorder

PKG_FILES = {  # path under pkg/ -> kept?
    "ab/abcd/gate.parse-0123456789abcdef/source.hip": True,
    "ab/abcd/gate.compile-0123456789abcdef/object.o": False,
    "ab/abcd/san.consan-0123456789abcdef/consan/consan.log": True,
    "ab/abcd/san.consan-0123456789abcdef/k0/aorlta_harness.hip": True,
    "ab/abcd/san.consan-0123456789abcdef/k0/harness": False,  # the built program
    "ab/abcd/san.consan-0123456789abcdef/kernels.json": True,
    "ab/abcd/san.asan-0123456789abcdef/asan/asan.log": True,
    "ab/abcd/san.waitcheck-0123456789abcdef/waitcheck-0.log": True,
    "ab/abcd/response.txt": False,
    "ab/abcd/ref.native-0123456789abcdef/r1/run.log": True,  # a task's own check, as the worked HIP example's
    "ab/abcd/ref.native-0123456789abcdef/r1/harness.hip": True,
    "ab/abcd/ref.native-0123456789abcdef/r1/record.json": True,
    "ab/abcd/ref.native-0123456789abcdef/r1/candidate.co": False,
}


def _plant(root: Path) -> None:
    for rel in PKG_FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"evidence of {rel}\n")


def _members(tgz: Path) -> set[str]:
    with tarfile.open(tgz) as t:
        return set(t.getnames())


def test_only_the_legs_evidence_is_kept(tmp_path):
    _plant(tmp_path / "pkg")
    big = tmp_path / "pkg" / "ab" / "abcd" / "san.asan-0123456789abcdef" / "asan" / "huge.log"
    big.write_bytes(b"x" * 4096)
    n = archive_legs(tmp_path / "pkg", tmp_path / "out" / "e.tgz", max_file_bytes=1024)
    want = {rel for rel, keep in PKG_FILES.items() if keep}
    assert _members(tmp_path / "out" / "e.tgz") == want and n == len(want)  # a file over the cap is left out
    assert all(kept(Path(rel)) == keep for rel, keep in PKG_FILES.items())
    assert archive_legs(tmp_path / "missing", tmp_path / "out" / "none.tgz") == 0
    assert not (tmp_path / "out" / "none.tgz").exists()


def test_archives_are_numbered_so_a_resumed_run_keeps_the_earlier_one(tmp_path):
    assert next_archive(tmp_path) == tmp_path / "gpu-legs-1.tgz"
    (tmp_path / "gpu-legs-1.tgz").write_bytes(b"")
    assert next_archive(tmp_path) == tmp_path / "gpu-legs-2.tgz"


def _cfg(tmp_path):
    return config_from_dict({
        "run": {"run_id": "w", "source": "standin", "root": str(tmp_path / "runs")},
        "model": {"base_dir": str(tmp_path / "base"), "source": "standin"},
        "data": {"prompt_pools": [{"path": str(tmp_path / "ds" / "listings"), "source": "standin"}]},
        "rollout": {"engines": [0, 1], "K": 4, "P": 4, "compile_adapter": "fake.compile"},
        "reward": {"profile": "fake", "reward_fn": "standin.v11"},
        "orchestrator": {"iterations": 1, "run_seed": 7},
        "trainer": {"ranks": 1},
        "sync": {"dir": str(tmp_path / "shm")},
    })


def test_the_reward_adapter_keeps_the_evidence_before_it_removes_the_scratch(tmp_path):
    cfg = _cfg(tmp_path)
    for attempt in (1, 2):  # a run, then its resume: each attempt's close keeps its own archive
        clock = Clock()
        r = wiring.RewardAdapter(cfg, clock=clock, spans=SpanRecorder(monotonic=clock.now, wall=clock.wall))
        r.reopen()
        _plant(r.work_dir / "pkg")
        r.close()
        assert not r.work_dir.exists()
        tgz = cfg.run_dir / "evidence" / "reward" / f"gpu-legs-{attempt}.tgz"
        assert _members(tgz) == {rel for rel, keep in PKG_FILES.items() if keep}
    assert sorted(p.name for p in (cfg.run_dir / "evidence" / "reward").iterdir()) == ["gpu-legs-1.tgz",
                                                                                        "gpu-legs-2.tgz"]
