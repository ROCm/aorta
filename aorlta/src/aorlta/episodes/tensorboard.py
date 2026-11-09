"""TensorBoard scalars in place of W&B (design Section 6.8).

Scalars only, never text. The writer needs no TensorBoard or protobuf package (neither is in the dev venv or the
TokenSpeed image): an event file is TFRecord framing around serialized `Event` protobufs,

    record  = uint64 len | uint32 masked_crc32c(len) | data | uint32 masked_crc32c(data)
    Event   = 1: double wall_time, 2: int64 step, 3: string file_version, 5: Summary summary
    Summary = 1: repeated Value value;  Value = 1: string tag, 2: float simple_value

which is what TensorBoard reads. Each iteration gets its own file, `events.out.tfevents.<n, 10 digits>.aorlta`,
holding a `brain.Event:2` header and one event with that iteration's scalars at step n. It is written under a name
without "tfevents" (TensorBoard opens any file whose name contains it) ending in `.partial`, and renamed into
place, so a reader never sees a partial iteration and a resume can drop the files of uncommitted iterations.
"""

from __future__ import annotations

import math
import os
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..atomic import PARTIAL_SUFFIX, read_bytes

FILE_VERSION = "brain.Event:2"
FILE_RE = re.compile(r"events\.out\.tfevents\.([0-9]{10})\.aorlta")
TAG_RE = re.compile(r"[A-Za-z0-9_./-]+")
LANDING_VALUE = {"LANDED": 1.0, "NOOP": 0.0}


class TensorBoardError(ValueError):
    pass


# --- CRC32C (Castagnoli), as TFRecord uses it ----------------------------------------------------------------


def _crc_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
        table.append(crc)
    return table


_TABLE = _crc_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = _TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


# --- protobuf encoding --------------------------------------------------------------------------------------


def _varint(n: int) -> bytes:
    if n < 0:
        n &= (1 << 64) - 1  # int64 two's complement
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _len_field(field: int, data: bytes) -> bytes:
    return _key(field, 2) + _varint(len(data)) + data


def event_bytes(wall_time: float, step: int, *, file_version: str | None = None,
                scalars: Mapping[str, float] | None = None) -> bytes:
    out = _key(1, 1) + struct.pack("<d", float(wall_time)) + _key(2, 0) + _varint(int(step))
    if file_version is not None:
        out += _len_field(3, file_version.encode("utf-8"))
    if scalars:
        values = b"".join(
            _len_field(1, _len_field(1, tag.encode("utf-8")) + _key(2, 5) + struct.pack("<f", float(v)))
            for tag, v in scalars.items()
        )
        out += _len_field(5, values)
    return out


def record_bytes(data: bytes) -> bytes:
    header = struct.pack("<Q", len(data))
    return header + struct.pack("<I", masked_crc32c(header)) + data + struct.pack("<I", masked_crc32c(data))


# --- decoding (tests and the leak scan read the files back) -------------------------------------------------------


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        if i >= len(buf):
            raise TensorBoardError("truncated varint")
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes) -> list[tuple[int, int, Any]]:
    out, i = [], 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _read_varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 2:
            n, i = _read_varint(buf, i)
            value, i = buf[i:i + n], i + n
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise TensorBoardError(f"unsupported wire type {wire}")
        if i > len(buf):
            raise TensorBoardError("truncated field")
        out.append((field, wire, value))
    return out


def parse_event(data: bytes) -> dict:
    ev: dict[str, Any] = {"scalars": {}}
    for field, wire, value in _fields(data):
        if field == 1 and wire == 1:
            ev["wall_time"] = struct.unpack("<d", value)[0]
        elif field == 2 and wire == 0:
            ev["step"] = value - (1 << 64) if value >= 1 << 63 else value
        elif field == 3 and wire == 2:
            ev["file_version"] = value.decode("utf-8")
        elif field == 5 and wire == 2:
            for vf, vw, vbuf in _fields(value):
                if vf != 1 or vw != 2:
                    raise TensorBoardError("a Summary holds only Values")
                tag, simple = None, None
                for f2, w2, v2 in _fields(vbuf):
                    if f2 == 1 and w2 == 2:
                        tag = v2.decode("utf-8")
                    elif f2 == 2 and w2 == 5:
                        simple = struct.unpack("<f", v2)[0]
                    else:
                        raise TensorBoardError(f"Value field {f2} is not a scalar")
                if tag is None or simple is None:
                    raise TensorBoardError("a Value without a tag or a simple_value")
                ev["scalars"][tag] = simple
        else:
            raise TensorBoardError(f"unexpected Event field {field}")
    return ev


def read_events(path: Path) -> list[dict]:
    buf = read_bytes(path)
    out, i = [], 0
    while i < len(buf):
        if i + 12 > len(buf):
            raise TensorBoardError(f"{path}: truncated record header")
        header = buf[i:i + 8]
        (n,) = struct.unpack("<Q", header)
        (hcrc,) = struct.unpack("<I", buf[i + 8:i + 12])
        if hcrc != masked_crc32c(header):
            raise TensorBoardError(f"{path}: bad length CRC at byte {i}")
        data = buf[i + 12:i + 12 + n]
        if len(data) != n or i + 16 + n > len(buf):
            raise TensorBoardError(f"{path}: truncated record")
        (dcrc,) = struct.unpack("<I", buf[i + 12 + n:i + 16 + n])
        if dcrc != masked_crc32c(data):
            raise TensorBoardError(f"{path}: bad data CRC at byte {i}")
        out.append(parse_event(data))
        i += 16 + n
    return out


# --- the per-iteration files ------------------------------------------------------------------------------------


def event_file_name(n: int) -> str:
    return f"events.out.tfevents.{n:010d}.aorlta"


def tb_files(tb_dir: Path) -> list[tuple[int, Path]]:
    if not Path(tb_dir).is_dir():
        return []
    return sorted((int(m.group(1)), p) for p in Path(tb_dir).iterdir() if (m := FILE_RE.fullmatch(p.name)))


def clean_scalars(scalars: Mapping[str, Any]) -> dict[str, float]:
    """Finite numbers under valid tags; None values are dropped, anything else is refused (scalars only)."""
    out = {}
    for tag, v in scalars.items():
        if not (isinstance(tag, str) and TAG_RE.fullmatch(tag)):
            raise TensorBoardError(f"tag {tag!r:.80} is not [A-Za-z0-9_./-]+")
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise TensorBoardError(f"{tag}: TensorBoard gets finite numbers only, got {type(v).__name__}")
        out[tag] = float(v)
    return out


def write_iteration(tb_dir: Path, n: int, scalars: Mapping[str, Any], wall_time: float) -> Path:
    tb_dir = Path(tb_dir)
    tb_dir.mkdir(parents=True, exist_ok=True)
    final = tb_dir / event_file_name(n)
    if final.exists():
        raise FileExistsError(f"{final} exists: iteration {n}'s scalars are already written")
    data = record_bytes(event_bytes(wall_time, 0, file_version=FILE_VERSION))
    data += record_bytes(event_bytes(wall_time, n, scalars=clean_scalars(scalars)))
    tmp = tb_dir / f".tb-{n:06d}{PARTIAL_SUFFIX}"  # no "tfevents" in the name, so TensorBoard never opens it
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    fd = os.open(tb_dir, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return final


def read_scalars(tb_dir: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for n, path in tb_files(tb_dir):
        events = read_events(path)
        if not events or events[0].get("file_version") != FILE_VERSION:
            raise TensorBoardError(f"{path}: no {FILE_VERSION} header")
        for ev in events[1:]:
            if ev.get("step") != n:
                raise TensorBoardError(f"{path}: an event at step {ev.get('step')} in iteration {n}'s file")
            out.setdefault(n, {}).update(ev["scalars"])
    return out


def remove_after(tb_dir: Path, n: int) -> list[str]:
    removed = []
    for m, path in tb_files(tb_dir):
        if m > n:
            path.unlink()
            removed.append(path.name)
    return removed


# --- the tags (Section 6.8) ------------------------------------------------------------------------------------------


def _tag(*parts: str) -> str:
    return "/".join(re.sub(r"[^A-Za-z0-9_.-]", "_", p) for p in parts)


def scalars_from_summary(s: Mapping[str, Any]) -> dict[str, float]:
    """The iteration summary's numbers under the doc's tags."""
    out: dict[str, Any] = {}
    reward = s.get("reward") or {}
    out["reward/mean"] = reward.get("mean")
    out["reward/std"] = reward.get("std")
    out["reward/scored_frac"] = s.get("scored_fraction")
    for reason, n in (s.get("unscored_by_reason") or {}).items():
        out[_tag("unscored", reason)] = n
    for gate, n in (s.get("gates") or {}).items():
        out[_tag("gate", gate.removeprefix("gate."))] = n
    for t, row in ((s.get("compile") or {}).get("per_turn") or {}).items():
        out[_tag("turn", t, "compile_ok")] = row.get("ok_frac")
        out[_tag("turn", t, "repair_rescue")] = row.get("rescue_frac")
    for phase, row in ((s.get("timing") or {}).get("phases") or {}).items():
        out[_tag("time", phase)] = row.get("union_s")
    out["time/per_256_scored"] = (s.get("timing") or {}).get("per_256_scored_s")
    for leg, row in (s.get("cache") or {}).items():
        out[_tag("cache", leg)] = row.get("hit_rate")
    asan = (s.get("reward_service") or {}).get("asan") or {}
    out["asan/batch_size"] = asan.get("batch_size_mean")
    out["asan/requests"] = asan.get("requests")
    out["asan/wait_s"] = asan.get("max_wait_s")
    tr = s.get("trainer") or {}
    out["train/loss"] = tr.get("loss")
    out["train/grad_norm"] = tr.get("grad_norm")
    out["train/gap_median"] = tr.get("gap_median")
    out["train/gap_p99"] = tr.get("gap_p99")
    out["train/excluded"] = tr.get("excluded")
    sy = s.get("sync") or {}
    out["sync/coverage"] = sy.get("coverage")
    out["sync/changed_frac"] = sy.get("changed_frac")
    out["sync/reload_s"] = sy.get("reload_s")
    out["sync/landing"] = LANDING_VALUE.get(sy.get("landing")) if sy.get("landing") is not None else None
    ln = s.get("length") or {}
    out["len/response_mean"] = ln.get("response_mean")
    out["len/truncation_rate"] = (s.get("truncation") or {}).get("rate")
    return clean_scalars(out)
