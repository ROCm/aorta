"""Kernel descriptors of a device code object, read on CPU: each kernel's static LDS (`group_segment_fixed_size`).

Two readings of one code object, which must agree (`read_group_segments`):

  - the AMDGPU metadata note, as `llvm-readelf --notes` prints it: each kernel's `.symbol` and
    `.group_segment_fixed_size`;
  - the kernel descriptor itself: the 64-byte object `<kernel>.kd`, whose first field, a little-endian u32, is the
    group segment size (`kd_group_segments`).

Either reading failing, finding no kernel, or the two disagreeing gives no sizes, never zeros. Nothing here reads the
candidate's text.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

EM_AMDGPU = 224
ET_REL = 1
SHT_SYMTAB, SHT_NOBITS, SHT_DYNSYM = 2, 8, 11
SHN_LORESERVE = 0xFF00
KD_SIZE = 64  # sizeof(kernel_descriptor_t)
Runner = Callable[[Sequence[str], float], tuple[int | None, str]]


class DescriptorError(ValueError):
    """The code object, or its metadata, is not what a kernel's LDS can be read from."""


def _cstring(data: bytes, start: int) -> str:
    end = data.find(b"\0", start)
    if start < 0 or end < 0:
        raise DescriptorError("a symbol name runs past its string table")
    return data[start:end].decode("latin-1")


def kd_group_segments(data: bytes) -> dict[str, int]:
    """{kernel symbol: the u32 at byte 0 of its `<kernel>.kd` descriptor}, for every descriptor in an AMDGPU ELF64
    code object (a linked one, or a relocatable object). Raises DescriptorError on anything unexpected."""
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise DescriptorError("not a little-endian ELF64 object")
    e_type, e_machine = struct.unpack_from("<HH", data, 16)
    if e_machine != EM_AMDGPU:
        raise DescriptorError(f"not an AMDGPU code object (machine {e_machine})")
    (e_shoff,) = struct.unpack_from("<Q", data, 40)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 58)
    if not e_shoff or e_shentsize != 64 or not e_shnum or e_shoff + e_shnum * 64 > len(data):
        raise DescriptorError("no readable section table")
    # (name, type, flags, addr, offset, size, link, info, addralign, entsize)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * 64) for i in range(e_shnum)]
    tables = [s for s in sections if s[1] == SHT_SYMTAB] or [s for s in sections if s[1] == SHT_DYNSYM]
    if len(tables) != 1:
        raise DescriptorError(f"{len(tables)} symbol tables")
    table = tables[0]
    if table[6] >= e_shnum or table[4] + table[5] > len(data) or table[5] % 24:
        raise DescriptorError("a malformed symbol table")
    strings = sections[table[6]]
    names = data[strings[4]:strings[4] + strings[5]]
    out: dict[str, int] = {}
    for at in range(table[4], table[4] + table[5], 24):
        st_name, _info, _other, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", data, at)
        name = _cstring(names, st_name)
        if not name.endswith(".kd"):
            continue
        if st_size != KD_SIZE or not 0 < st_shndx < min(e_shnum, SHN_LORESERVE):
            raise DescriptorError(f"{name}: not a {KD_SIZE}-byte descriptor in a section")
        sec = sections[st_shndx]
        if sec[1] == SHT_NOBITS or (e_type != ET_REL and st_value < sec[3]):
            raise DescriptorError(f"{name}: not in its section's bytes")
        start = sec[4] + (st_value if e_type == ET_REL else st_value - sec[3])
        if start + KD_SIZE > min(sec[4] + sec[5], len(data)):
            raise DescriptorError(f"{name}: runs past its section")
        (size,) = struct.unpack_from("<I", data, start)
        kernel = name[:-len(".kd")]
        if out.get(kernel, size) != size:
            raise DescriptorError(f"{name}: two descriptors disagree")
        out[kernel] = size
    if not out:
        raise DescriptorError("no kernel descriptor")
    return out


def note_group_segments(readelf_notes: str) -> dict[str, int]:
    """{kernel symbol: `.group_segment_fixed_size`} from `llvm-readelf --notes` output: its one AMDGPU metadata
    document, every kernel in `amdhsa.kernels`. Raises DescriptorError on anything unexpected."""
    lines = readelf_notes.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(starts) != 1:
        raise DescriptorError(f"{len(starts)} AMDGPU metadata documents")
    end = next((i for i in range(starts[0] + 1, len(lines)) if lines[i].strip() == "..."), None)
    if end is None:
        raise DescriptorError("the AMDGPU metadata document does not end")
    try:
        doc = yaml.safe_load("\n".join(lines[starts[0] + 1:end]))
    except yaml.YAMLError as exc:
        raise DescriptorError(f"the AMDGPU metadata is not YAML: {type(exc).__name__}") from None
    kernels = doc.get("amdhsa.kernels") if isinstance(doc, dict) else None
    if not isinstance(kernels, list) or not kernels:
        raise DescriptorError("no kernel in the AMDGPU metadata")
    out: dict[str, int] = {}
    for k in kernels:
        symbol = k.get(".symbol") if isinstance(k, dict) else None
        size = k.get(".group_segment_fixed_size") if isinstance(k, dict) else None
        if not (isinstance(symbol, str) and symbol.endswith(".kd") and type(size) is int and size >= 0):
            raise DescriptorError("a kernel without its symbol or group segment size")
        if symbol[:-len(".kd")] in out:
            raise DescriptorError(f"{symbol} is listed twice")
        out[symbol[:-len(".kd")]] = size
    return out


def read_group_segments(path: str | Path, *, readelf: str, run: Runner, timeout_s: float = 60.0
                        ) -> tuple[dict[str, int] | None, str | None]:
    """Every kernel's static LDS in the code object at `path`, from its descriptors and its metadata note, which must
    agree: (sizes, None), or (None, why) when either cannot be read or they differ."""
    try:
        from_kd = kd_group_segments(Path(path).read_bytes())
    except (OSError, DescriptorError) as exc:
        return None, f"descriptor_unreadable: {exc}"[:300]
    rc, notes = run([readelf, "--notes", str(path)], timeout_s)
    if rc != 0:
        return None, f"metadata_unreadable: {readelf} exit {rc}"[:300]
    try:
        from_note = note_group_segments(notes)
    except DescriptorError as exc:
        return None, f"metadata_unreadable: {exc}"[:300]
    if from_note != from_kd:
        return None, "descriptor_mismatch: the metadata note and the kernel descriptors disagree"
    return from_kd, None
