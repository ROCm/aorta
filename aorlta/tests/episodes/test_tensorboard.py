"""TensorBoard scalars: the file format, written without TensorBoard or protobuf installed."""

import errno
import os
import stat
import struct

import pytest

from aorlta.episodes import tensorboard as tb


def test_crc32c_matches_the_standard_check_value():
    assert tb.crc32c(b"123456789") == 0xE3069283  # CRC-32C (Castagnoli) check value
    assert tb.crc32c(b"") == 0


def test_the_masked_crc_is_tfrecord_s():
    crc = tb.crc32c(b"abc")
    assert tb.masked_crc32c(b"abc") == ((((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF)


def test_an_event_encodes_as_the_protobuf_wire_format():
    got = tb.event_bytes(1.5, 7, scalars={"a": 2.0})
    want = (bytes([0x09]) + struct.pack("<d", 1.5)  # 1: double wall_time
            + bytes([0x10, 0x07])  # 2: int64 step
            + bytes([0x2A, 0x0A,  # 5: Summary, 10 bytes
                     0x0A, 0x08,  # Summary.value (1), 8 bytes
                     0x0A, 0x01, ord("a"),  # Value.tag (1)
                     0x15]) + struct.pack("<f", 2.0))  # Value.simple_value (2, fixed32)
    assert got == want
    assert tb.event_bytes(0.0, 0, file_version="brain.Event:2").endswith(bytes([0x1A, 13]) + b"brain.Event:2")


def test_a_record_frames_its_data_with_two_masked_crcs():
    data = b"\x01\x02\x03"
    rec = tb.record_bytes(data)
    assert rec[:8] == struct.pack("<Q", 3)
    assert rec[8:12] == struct.pack("<I", tb.masked_crc32c(rec[:8]))
    assert rec[12:15] == data and rec[15:] == struct.pack("<I", tb.masked_crc32c(data))


def test_scalars_round_trip_one_file_per_iteration(tmp_path):
    tb.write_iteration(tmp_path, 1, {"reward/mean": -0.25, "train/loss": 1.5, "skip": None}, 1.8e9)
    tb.write_iteration(tmp_path, 2, {"reward/mean": 0.5}, 1.8e9 + 60)
    assert [p.name for _, p in tb.tb_files(tmp_path)] == ["events.out.tfevents.0000000001.aorlta",
                                                         "events.out.tfevents.0000000002.aorlta"]
    assert tb.read_scalars(tmp_path) == {1: {"reward/mean": -0.25, "train/loss": 1.5}, 2: {"reward/mean": 0.5}}
    events = tb.read_events(tb.tb_files(tmp_path)[0][1])
    assert events[0]["file_version"] == "brain.Event:2" and events[1]["step"] == 1
    with pytest.raises(FileExistsError):
        tb.write_iteration(tmp_path, 1, {"reward/mean": 0.0}, 1.8e9)


def test_a_damaged_file_fails_its_crc(tmp_path):
    path = tb.write_iteration(tmp_path, 1, {"reward/mean": -0.25}, 1.8e9)
    data = bytearray(path.read_bytes())
    data[-6] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(tb.TensorBoardError):
        tb.read_events(path)


@pytest.mark.parametrize("bad", [{"reward/mean": "text"}, {"reward/mean": float("nan")}, {"reward mean": 1.0},
                                 {"reward/mean": True}])
def test_scalars_only_never_text(tmp_path, bad):
    with pytest.raises(tb.TensorBoardError):
        tb.write_iteration(tmp_path, 1, bad, 1.8e9)


def test_the_temporary_name_is_never_one_tensorboard_would_open(tmp_path, monkeypatch):
    names = []
    real_replace = tb.os.replace

    def spy(src, dst):
        names.append(str(src).rsplit("/", 1)[-1])
        return real_replace(src, dst)

    monkeypatch.setattr(tb.os, "replace", spy)
    tb.write_iteration(tmp_path, 3, {"reward/mean": 0.1}, 1.8e9)
    assert names and all("tfevents" not in n and n.endswith(".partial") for n in names)


def test_remove_after_drops_the_files_of_later_iterations(tmp_path):
    for n in (1, 2, 3):
        tb.write_iteration(tmp_path, n, {"reward/mean": float(n)}, 1.8e9)
    assert tb.remove_after(tmp_path, 1) == ["events.out.tfevents.0000000002.aorlta",
                                            "events.out.tfevents.0000000003.aorlta"]
    assert list(tb.read_scalars(tmp_path)) == [1]


@pytest.mark.parametrize("code, written", [(errno.EINVAL, True), (errno.ENOTSUP, True), (errno.EIO, False)])
def test_a_directory_that_refuses_fsync_is_tolerated_only_when_unsupported(tmp_path, monkeypatch, code, written):
    """As for the checkpoint and the episodes: a filesystem without directory fsync still takes the file, and a storage
    error stops the write."""
    real = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(code, os.strerror(code))
        return real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    if written:
        tb.write_iteration(tmp_path, 1, {"reward/mean": 0.5}, 1.8e9)
        assert tb.read_scalars(tmp_path)[1]["reward/mean"] == pytest.approx(0.5)
    else:
        with pytest.raises(OSError) as exc:
            tb.write_iteration(tmp_path, 1, {"reward/mean": 0.5}, 1.8e9)
        assert exc.value.errno == code
