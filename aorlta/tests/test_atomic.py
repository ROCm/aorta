"""Atomic writes (Section 5 item 6): a temporary file per writer, renamed into place."""

import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from aorlta.atomic import PartialFileError, atomic_write_bytes

SRC = Path(__file__).resolve().parents[1] / "src"
PAYLOADS = [bytes([ord("A") + i]) * (64 << 10) for i in range(4)]

WRITER = """
import sys
from aorlta.atomic import atomic_write_bytes
dest, data = sys.argv[1], bytes([int(sys.argv[2])]) * (64 << 10)
print("ready", flush=True)
sys.stdin.readline()
for _ in range(50):
    atomic_write_bytes(dest, data)
"""


def test_threads_writing_one_file_never_fail_and_readers_only_see_whole_files(tmp_path: Path):
    dest = atomic_write_bytes(tmp_path / "x.json", PAYLOADS[0])
    errors: list[BaseException] = []
    torn: list[int] = []
    done = threading.Event()

    def read() -> None:
        while not done.is_set():
            data = dest.read_bytes()
            if data not in PAYLOADS:
                torn.append(len(data))

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for _ in range(25):
            barrier = threading.Barrier(len(PAYLOADS))

            def write(data: bytes) -> None:
                barrier.wait()
                try:
                    atomic_write_bytes(dest, data)
                except BaseException as exc:
                    errors.append(exc)

            writers = [threading.Thread(target=write, args=(p,)) for p in PAYLOADS]
            for t in writers:
                t.start()
            for t in writers:
                t.join()
            assert dest.read_bytes() in PAYLOADS
    finally:
        done.set()
        reader.join()
    assert errors == [] and torn == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json"]


def test_processes_writing_one_file_never_fail(tmp_path: Path):
    dest = tmp_path / "y.json"
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, str(dest), str(p[0])], env=env, text=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             for p in PAYLOADS]
    try:
        for p in procs:
            assert p.stdout.readline() == "ready\n"
        for p in procs:  # all four start writing together
            p.stdin.write("\n")
            p.stdin.flush()
        stderr = [p.communicate(timeout=120)[1] for p in procs]
    finally:
        for p in procs:
            p.kill()
    assert [p.returncode for p in procs] == [0] * len(procs), stderr
    assert dest.read_bytes() in PAYLOADS
    assert sorted(p.name for p in tmp_path.iterdir()) == ["y.json"]


def test_a_failed_write_leaves_no_temporary_file(tmp_path: Path):
    (tmp_path / "d").mkdir()
    with pytest.raises(IsADirectoryError):  # the rename fails
        atomic_write_bytes(tmp_path / "d", b"x")
    with pytest.raises(TypeError):  # the write fails
        atomic_write_bytes(tmp_path / "e", "not bytes")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d"]


def test_replace_false_never_replaces_an_existing_file(tmp_path: Path):
    path = atomic_write_bytes(tmp_path / "plan.json", b"first", replace=False)
    with pytest.raises(FileExistsError):
        atomic_write_bytes(path, b"second", replace=False)
    assert path.read_bytes() == b"first"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plan.json"]


def test_permissions_follow_the_umask(tmp_path: Path):
    old = os.umask(0o027)
    try:
        path = atomic_write_bytes(tmp_path / "f", b"x")
    finally:
        os.umask(old)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_a_partial_name_is_never_a_destination(tmp_path: Path):
    with pytest.raises(PartialFileError):
        atomic_write_bytes(tmp_path / "x.partial", b"")
    assert list(tmp_path.iterdir()) == []
