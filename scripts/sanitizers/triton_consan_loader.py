#!/usr/bin/env python3
"""Generic ConSan target for JIT-compiled Triton/Gluon kernels (ROCm/aorta#399).

``source.consan_command`` needs an executable that puts exactly the selected code
object on the device, so the RocJITsu hook instruments that object and nothing
else. The committed loaders (``consan_load.hip``, ``lds_dispatch.hip``) bake the
object path in at *compile* time via ``-DOBJECT``, which cannot work for a Triton
kernel: the ``.hsaco`` does not exist until the kernel runs, it is keyed by a
content hash under ``TRITON_CACHE_DIR``, and it is specific to the image, the GPU
target, and the compiled shapes. There is nothing to pin in a fixture directory
and no source to compile.

This loader binds ``libamdhip64`` through ctypes instead, so the object path is
resolved at *run* time and no compiler is in the loop. Point it at a Triton cache
entry (a ``.hsaco`` plus the adjacent ``.json`` Triton writes next to it) and it
loads that object under whatever hook the parent set in ``HSA_TOOLS_LIB``.

Two modes, mirroring the two committed HIP loaders:

``load`` (default)
    ``hipModuleLoad`` the object and resolve the kernel symbol. Needs no
    knowledge of the launch ABI, so it works for *any* Triton kernel. This is the
    Triton analogue of ``consan_load.hip``, and is what makes ConSan reachable at
    all -- ConSan instruments a code object when it is loaded.

``dispatch``
    Additionally launch the kernel once, for record/replay's dynamic coverage.
    This needs the argument signature, which Triton does **not** always write to
    the metadata JSON (it is absent in 3.7.1), so pass ``--launch-spec`` when the
    metadata has no ``signature``. Note that dispatching a caller-supplied object
    currently fails closed on the shipping RocJITsu build for the same reason
    ``daily-consan-lds-dispatch.yaml`` does (zero captured records -> exit 86,
    ROCm/rocm-systems#9972).

``run_consan`` executes ``source.consan_command`` as a bare argv with no
arguments, so a parameterised loader cannot be named directly. ``emit-command``
bridges that: it writes a tiny executable shim with the resolved arguments baked
in, which is the run-time analogue of the ``-DOBJECT`` build step. It also bakes
in a SHA-256 for every input whose bytes decide what runs, and ``run`` re-checks
each one before touching the device. That is what makes the ``command_sha256``
``run_consan`` records meaningful: a cache entry can be evicted and repopulated,
so pinning the paths alone would let a rebuilt cache feed different code to the
same recorded command and selected identity.

Usage:

    # One shim per code object, named as source.consan_command in a recipe.
    ./triton_consan_loader.py emit-command \\
        --cache-entry "$TRITON_CACHE_DIR/M3AQ...SCA" \\
        --output fixtures/bin/consan_add_kernel

    # Or run it directly to check an object loads before wiring up a recipe.
    ./triton_consan_loader.py run --cache-entry "$TRITON_CACHE_DIR/M3AQ...SCA"
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import shlex
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

# Scalar Triton signature types -> (struct size, alignment) in the launch buffer.
# Pointers are handled separately: every ``*T`` is one device address.
_POINTER_SIZE = 8
_SCALAR_TYPES: dict[str, int] = {
    "i1": 1,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "u1": 1,
    "u8": 1,
    "u16": 2,
    "u32": 4,
    "u64": 8,
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
}
_FLOAT_TYPES = frozenset({"fp16", "bf16", "fp32", "fp64"})

# hip/hip_runtime.h launch-config sentinels.
_HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
_HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
_HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)

_DEFAULT_BUFFER_BYTES = 8 * 1024 * 1024

# Widths of the C types every launch parameter is narrowed into. ctypes wraps
# silently on overflow -- c_uint(2**32 + 1) is 1 -- which would turn an oversized
# grid or buffer into a quietly tiny launch, so each value is range-checked
# against the real platform width before conversion.
_UINT_MAX = 2 ** (8 * ctypes.sizeof(ctypes.c_uint)) - 1
_SIZE_MAX = 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1


class LoaderError(Exception):
    """A fail-closed loader error: never degrade into a vacuous clean run."""


def require_in_range(value: int, *, name: str, minimum: int, maximum: int) -> int:
    """Reject a value that would wrap when narrowed into its C type."""

    if value < minimum or value > maximum:
        raise LoaderError(f"{name} must be in {minimum}..{maximum}, got {value}")
    return value


# --------------------------------------------------------------------------
# Triton cache entry resolution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheEntry:
    """One Triton cache entry: a code object plus its sidecar metadata."""

    hsaco: Path
    metadata_path: Path
    metadata: dict[str, object]

    @property
    def kernel_name(self) -> str:
        name = self.metadata.get("name")
        if not isinstance(name, str) or not name:
            raise LoaderError(f"{self.metadata_path}: metadata has no 'name'")
        return name

    @property
    def arch(self) -> str | None:
        target = self.metadata.get("target")
        if isinstance(target, dict):
            arch = target.get("arch")
            if isinstance(arch, str):
                return arch
        arch = self.metadata.get("arch")
        return arch if isinstance(arch, str) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_digest(path: Path, expected: str | None, *, what: str) -> None:
    """Fail closed when a pinned input's bytes have changed since emission.

    A Triton cache entry is keyed by a content hash, but the entry can be evicted
    and repopulated, so a path alone does not pin the bytes. Baking the digest
    into the emitted shim and re-checking it here means the ``command_sha256``
    ``run_consan`` records covers *what* is loaded, not just where it lives --
    otherwise a rebuilt cache could feed different code to the same recorded
    command and selected identity.
    """

    if expected is None:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise LoaderError(
            f"{what} digest mismatch for {path}: expected {expected}, found {actual}. "
            "The Triton cache entry changed since this command was emitted; "
            "re-run emit-command against the current cache."
        )


def read_json_object(path: Path, *, what: str = "Triton metadata") -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LoaderError(f"{path}: cannot read {what}: {exc}") from exc
    if not isinstance(data, dict):
        raise LoaderError(f"{path}: {what} must be a JSON object")
    return data


def entry_from_hsaco(hsaco: Path, metadata_path: Path | None = None) -> CacheEntry:
    """Pair a ``.hsaco`` with its sidecar JSON, defaulting to the adjacent one."""

    if not hsaco.is_file():
        raise LoaderError(f"code object not found: {hsaco}")
    resolved_metadata = metadata_path or hsaco.with_suffix(".json")
    if not resolved_metadata.is_file():
        raise LoaderError(
            f"no Triton metadata beside {hsaco}: expected {resolved_metadata}. "
            "Pass --metadata to name it explicitly."
        )
    return CacheEntry(
        hsaco=hsaco.resolve(),
        metadata_path=resolved_metadata.resolve(),
        metadata=read_json_object(resolved_metadata),
    )


def discover_entries(root: Path) -> list[CacheEntry]:
    """Find every ``.hsaco``/``.json`` pair at or under a Triton cache directory.

    Accepts either a single cache entry directory or a whole cache root, so the
    several distinct objects one logical kernel compiles to (shape-selected
    variants such as ``_mfma_lds_mediumm_kernel`` / ``_mfma_lds_largem_kernel``)
    are all discoverable from one path.
    """

    if not root.is_dir():
        raise LoaderError(f"Triton cache directory not found: {root}")
    entries: list[CacheEntry] = []
    # Driving off the code objects rather than the JSON files keeps Triton's
    # ``__grp__`` sibling index out of the results: it has no matching .hsaco.
    for hsaco in sorted(root.rglob("*.hsaco")):
        sidecar = hsaco.with_suffix(".json")
        if not sidecar.is_file():
            continue
        entries.append(entry_from_hsaco(hsaco, sidecar))
    if not entries:
        raise LoaderError(
            f"no Triton code objects under {root}: expected a '<kernel>.hsaco' "
            "with an adjacent '<kernel>.json'. Has the kernel been run yet?"
        )
    return entries


def select_entry(
    entries: list[CacheEntry],
    *,
    kernel_name: str | None = None,
    cache_hash: str | None = None,
) -> CacheEntry:
    """Narrow a candidate list to exactly one entry, or fail closed.

    ConSan is only meaningful for a single selected identity (``run_consan``
    rejects a multi-kernel worklist), so an ambiguous selection must not silently
    pick one. The error lists the candidates and how to disambiguate them.
    """

    candidates = entries
    if kernel_name is not None:
        candidates = [entry for entry in candidates if entry.kernel_name == kernel_name]
    if cache_hash is not None:
        candidates = [
            entry for entry in candidates if str(entry.metadata.get("hash", "")).startswith(cache_hash)
        ]
    if not candidates:
        raise LoaderError(
            "no Triton cache entry matches "
            f"kernel_name={kernel_name!r} hash={cache_hash!r}; "
            f"available: {', '.join(sorted(entry.kernel_name for entry in entries))}"
        )
    if len(candidates) > 1:
        listing = "\n".join(
            f"  {entry.kernel_name}  hash={entry.metadata.get('hash', '?')}  {entry.hsaco}"
            for entry in candidates
        )
        raise LoaderError(
            f"{len(candidates)} Triton cache entries match; ConSan needs exactly one "
            f"code object per run. Narrow it with --kernel-name / --cache-hash:\n{listing}"
        )
    return candidates[0]


# --------------------------------------------------------------------------
# Launch plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgSpec:
    """One kernel argument as it appears in the packed Triton launch buffer."""

    name: str
    ttype: str

    @property
    def is_pointer(self) -> bool:
        return self.ttype.startswith("*")

    @property
    def size(self) -> int:
        if self.is_pointer:
            return _POINTER_SIZE
        size = _SCALAR_TYPES.get(self.ttype)
        if size is None:
            raise LoaderError(
                f"argument {self.name!r} has unsupported Triton type {self.ttype!r}; "
                f"supported: *T pointers and {', '.join(sorted(_SCALAR_TYPES))}"
            )
        return size


def parse_signature(signature: object) -> tuple[ArgSpec, ...]:
    """Turn a Triton ``signature`` mapping into ordered, packable arg specs.

    ``constexpr`` parameters are compiled into the kernel rather than passed, so
    they are dropped from the launch buffer.
    """

    if not isinstance(signature, dict):
        raise LoaderError("signature must be a JSON object of {arg_name: triton_type}")
    specs = []
    for name, ttype in signature.items():
        if not isinstance(ttype, str):
            raise LoaderError(f"signature entry {name!r} must map to a type string")
        if ttype == "constexpr":
            continue
        specs.append(ArgSpec(name=str(name), ttype=ttype))
    return tuple(specs)


def block_dim(metadata: dict[str, object]) -> int:
    """Derive the Triton launch block size (``num_warps`` lanes of ``warp_size``)."""

    target = metadata.get("target")
    warp_size = None
    if isinstance(target, dict):
        warp_size = target.get("warp_size")
    if not isinstance(warp_size, int):
        warp_size = metadata.get("warp_size")
    if not isinstance(warp_size, int) or warp_size <= 0:
        raise LoaderError("metadata has no positive 'warp_size'")
    num_warps = metadata.get("num_warps")
    if not isinstance(num_warps, int) or num_warps <= 0:
        raise LoaderError("metadata has no positive 'num_warps'")
    return require_in_range(
        num_warps * warp_size, name="block size (num_warps * warp_size)", minimum=1, maximum=_UINT_MAX
    )


def shared_bytes(metadata: dict[str, object]) -> int:
    value = metadata.get("shared", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LoaderError("metadata 'shared' must be a non-negative integer")
    return require_in_range(value, name="metadata 'shared'", minimum=0, maximum=_UINT_MAX)


def pack_arguments(
    specs: tuple[ArgSpec, ...],
    *,
    pointers: dict[str, int],
    scalars: dict[str, str],
) -> bytes:
    """Pack one Triton launch buffer, honouring each argument's natural alignment.

    Scalars default to zero. That is deliberate: a synthesized non-zero size or
    stride would index buffers whose real extents we cannot know, turning a
    sanitizer run into a memory fault. Callers who want the kernel to touch
    memory supply both the extents (``--buffer-bytes``) and the scalars
    (``--arg``).
    """

    buffer = bytearray()
    for spec in specs:
        size = spec.size
        padding = (-len(buffer)) % size
        buffer.extend(b"\x00" * padding)
        if spec.is_pointer:
            buffer.extend(pointers[spec.name].to_bytes(_POINTER_SIZE, "little"))
            continue
        raw = scalars.get(spec.name, "0")
        if spec.ttype in _FLOAT_TYPES:
            buffer.extend(_pack_float(spec, raw))
        else:
            buffer.extend(_pack_int(spec, raw, size))
    # Trailing pad to the widest member, matching C struct sizing.
    if specs:
        widest = max(spec.size for spec in specs)
        buffer.extend(b"\x00" * ((-len(buffer)) % widest))
    return bytes(buffer)


def _pack_float(spec: ArgSpec, raw: str) -> bytes:
    try:
        value = float(raw)
    except ValueError as exc:
        raise LoaderError(f"argument {spec.name!r} expects a float, got {raw!r}") from exc
    # struct.pack raises OverflowError for a finite value too wide for the target
    # format (fp16 above 65504, fp32 above ~3.4e38). Surface it as a loader error
    # so an out-of-range --arg fails closed instead of raising a traceback.
    try:
        if spec.ttype == "fp32":
            return struct.pack("<f", value)
        if spec.ttype == "fp64":
            return struct.pack("<d", value)
        if spec.ttype == "fp16":
            return struct.pack("<e", value)
        # bf16 is the upper half of the fp32 bit pattern (round-to-nearest-even).
        bits = int.from_bytes(struct.pack("<f", value), "little")
    except OverflowError as exc:
        raise LoaderError(
            f"argument {spec.name!r} value {value} does not fit in {spec.ttype}"
        ) from exc
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (rounded & 0xFFFF).to_bytes(2, "little")


def _pack_int(spec: ArgSpec, raw: str, size: int) -> bytes:
    try:
        value = int(raw, 0)
    except ValueError as exc:
        raise LoaderError(f"argument {spec.name!r} expects an integer, got {raw!r}") from exc
    signed = not spec.ttype.startswith("u")
    try:
        return value.to_bytes(size, "little", signed=signed)
    except OverflowError as exc:
        raise LoaderError(
            f"argument {spec.name!r} value {value} does not fit in {spec.ttype}"
        ) from exc


_KERNARG_SIZE = re.compile(r"^\s*\.kernarg_segment_size:\s*(\d+)\s*$", re.MULTILINE)


def kernarg_segment_size(entry: CacheEntry) -> int | None:
    """Read the kernel's kernarg segment size from the adjacent ``.amdgcn`` listing.

    Triton writes the generated assembly beside the code object, and its metadata
    block records the segment size the kernel was compiled for. Returns ``None``
    when the listing is absent, since it is not required to exist.
    """

    listing = entry.hsaco.with_suffix(".amdgcn")
    if not listing.is_file():
        return None
    match = _KERNARG_SIZE.search(listing.read_text(encoding="utf-8", errors="replace"))
    return int(match.group(1)) if match else None


def check_kernarg_fit(packed: int, segment: int | None, *, kernel: str) -> None:
    """Reject a launch buffer larger than the kernel's kernarg segment.

    HIP does not validate ``HIP_LAUNCH_PARAM_BUFFER_SIZE`` against the kernel
    descriptor, so a wrong ``--launch-spec`` signature otherwise dispatches
    silently and can scribble past the segment. The segment also covers the
    hidden arguments the compiler appends, so a *smaller* buffer is normal and
    only the over-run direction is provably wrong.
    """

    if segment is not None and packed > segment:
        raise LoaderError(
            f"packed launch buffer for {kernel!r} is {packed} bytes but the kernel's "
            f"kernarg segment is {segment} bytes: the signature does not match this "
            "code object"
        )


def parse_grid(raw: str) -> tuple[int, int, int]:
    parts = raw.split(",")
    if len(parts) != 3:
        raise LoaderError(f"--grid must be 'X,Y,Z', got {raw!r}")
    try:
        dims = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise LoaderError(f"--grid must be three integers, got {raw!r}") from exc
    for axis, dim in zip("xyz", dims, strict=True):
        require_in_range(dim, name=f"--grid {axis}", minimum=1, maximum=_UINT_MAX)
    return dims  # type: ignore[return-value]


def parse_arg_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name.strip():
            raise LoaderError(f"--arg must be 'name=value', got {pair!r}")
        overrides[name.strip()] = value.strip()
    return overrides


def resolve_signature(entry: CacheEntry, launch_spec: dict[str, object] | None) -> object:
    """Prefer an explicit launch spec's signature, else the metadata's."""

    if launch_spec is not None and "signature" in launch_spec:
        return launch_spec["signature"]
    if "signature" in entry.metadata:
        return entry.metadata["signature"]
    raise LoaderError(
        f"{entry.metadata_path} has no 'signature' and no --launch-spec was given. "
        "Triton does not emit the argument signature in every version (it is absent "
        "in 3.7.1), so dispatch mode needs it supplied explicitly. Use --mode load "
        "to instrument the object without launching it."
    )


# --------------------------------------------------------------------------
# HIP runtime binding
# --------------------------------------------------------------------------


class Hip:
    """The thin slice of the HIP runtime this loader needs, via ctypes."""

    def __init__(self, library: str = "libamdhip64.so") -> None:
        try:
            self._lib = ctypes.CDLL(library)
        except OSError as exc:
            raise LoaderError(f"cannot load {library}: {exc}") from exc
        self._lib.hipGetErrorString.restype = ctypes.c_char_p
        self._lib.hipGetErrorString.argtypes = [ctypes.c_int]

    def check(self, code: int, what: str) -> None:
        if code != 0:
            message = self._lib.hipGetErrorString(code) or b"unknown error"
            raise LoaderError(f"{what} failed: {message.decode(errors='replace')} ({code})")

    def module_load(self, path: Path) -> ctypes.c_void_p:
        module = ctypes.c_void_p()
        self.check(
            self._lib.hipModuleLoad(ctypes.byref(module), str(path).encode()),
            f"hipModuleLoad({path})",
        )
        return module

    def module_unload(self, module: ctypes.c_void_p) -> None:
        self.check(self._lib.hipModuleUnload(module), "hipModuleUnload")

    def module_get_function(self, module: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
        function = ctypes.c_void_p()
        self.check(
            self._lib.hipModuleGetFunction(ctypes.byref(function), module, name.encode()),
            f"hipModuleGetFunction({name})",
        )
        return function

    def malloc(self, size: int) -> ctypes.c_void_p:
        require_in_range(size, name="buffer bytes", minimum=1, maximum=_SIZE_MAX)
        pointer = ctypes.c_void_p()
        self.check(self._lib.hipMalloc(ctypes.byref(pointer), ctypes.c_size_t(size)), "hipMalloc")
        try:
            self.check(self._lib.hipMemset(pointer, 0, ctypes.c_size_t(size)), "hipMemset")
        except LoaderError:
            # The allocation succeeded, so release it before propagating rather
            # than leaking device memory the caller never learns about.
            self._lib.hipFree(pointer)
            raise
        return pointer

    def free(self, pointer: ctypes.c_void_p) -> None:
        self.check(self._lib.hipFree(pointer), "hipFree")

    def launch(
        self,
        function: ctypes.c_void_p,
        *,
        grid: tuple[int, int, int],
        block: int,
        shared: int,
        arguments: bytes,
    ) -> None:
        buffer = ctypes.create_string_buffer(arguments, len(arguments))
        size = ctypes.c_size_t(len(arguments))
        config = (ctypes.c_void_p * 5)(
            _HIP_LAUNCH_PARAM_BUFFER_POINTER,
            ctypes.cast(buffer, ctypes.c_void_p),
            _HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.c_void_p(ctypes.addressof(size)),
            _HIP_LAUNCH_PARAM_END,
        )
        self.check(
            self._lib.hipModuleLaunchKernel(
                function,
                ctypes.c_uint(grid[0]),
                ctypes.c_uint(grid[1]),
                ctypes.c_uint(grid[2]),
                ctypes.c_uint(block),
                ctypes.c_uint(1),
                ctypes.c_uint(1),
                ctypes.c_uint(shared),
                ctypes.c_void_p(None),
                ctypes.c_void_p(None),
                config,
            ),
            "hipModuleLaunchKernel",
        )
        self.check(self._lib.hipDeviceSynchronize(), "hipDeviceSynchronize")


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def run_load(hip: Hip, entry: CacheEntry) -> None:
    module = hip.module_load(entry.hsaco)
    try:
        hip.module_get_function(module, entry.kernel_name)
        print(
            f"[triton-consan-loader] loaded+instrumented {entry.kernel_name} "
            f"arch={entry.arch} from {entry.hsaco} (no dispatch)"
        )
    finally:
        hip.module_unload(module)


def run_dispatch(
    hip: Hip,
    entry: CacheEntry,
    *,
    specs: tuple[ArgSpec, ...],
    grid: tuple[int, int, int],
    buffer_bytes: int,
    scalars: dict[str, str],
) -> None:
    unknown = sorted(set(scalars) - {spec.name for spec in specs})
    if unknown:
        raise LoaderError(f"--arg names not in the kernel signature: {', '.join(unknown)}")

    module = hip.module_load(entry.hsaco)
    allocations: list[ctypes.c_void_p] = []
    try:
        function = hip.module_get_function(module, entry.kernel_name)
        pointers: dict[str, int] = {}
        for spec in specs:
            if not spec.is_pointer:
                continue
            allocation = hip.malloc(buffer_bytes)
            allocations.append(allocation)
            pointers[spec.name] = allocation.value or 0
        arguments = pack_arguments(specs, pointers=pointers, scalars=scalars)
        check_kernarg_fit(len(arguments), kernarg_segment_size(entry), kernel=entry.kernel_name)
        block = block_dim(entry.metadata)
        shared = shared_bytes(entry.metadata)
        hip.launch(
            function,
            grid=grid,
            block=block,
            shared=shared,
            arguments=arguments,
        )
        print(
            f"[triton-consan-loader] dispatched {entry.kernel_name} "
            f"grid={grid} block={block} shared={shared} "
            f"args={len(arguments)}B buffers={len(allocations)}x{buffer_bytes}B"
        )
    finally:
        for allocation in allocations:
            hip.free(allocation)
        hip.module_unload(module)


def render_shim(loader: Path, argv: list[str], *, entry: CacheEntry) -> str:
    """Render the zero-argument shim that a recipe names as ``consan_command``."""

    quoted = " \\\n    ".join(shlex.quote(item) for item in argv)
    return (
        "#!/bin/sh\n"
        "# Generated by scripts/sanitizers/triton_consan_loader.py -- do not edit.\n"
        f"# ConSan target for Triton kernel '{entry.kernel_name}' (arch {entry.arch}).\n"
        f"# Code object: {entry.hsaco}\n"
        f"# Metadata:    {entry.metadata_path}\n"
        "#\n"
        "# sanitizer_plan.source.consan_command must be a bare executable taking no\n"
        "# arguments, so the loader's arguments are baked in here instead. The\n"
        "# --expect-*-sha256 digests pin the bytes this was generated against: a\n"
        "# Triton cache entry can be evicted and repopulated, so the paths alone\n"
        "# would not stop a rebuilt cache from feeding different code to the same\n"
        "# recorded command. Re-run emit-command after any recompile.\n"
        "set -e\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(loader))} \\\n    {quoted}\n"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# Inputs whose bytes decide what actually runs, so the emitted shim pins each by
# digest: the code object is loaded, the metadata sets the kernel name / block /
# LDS, and the launch spec sets the argument signature.
_PINNED_INPUTS = {
    "object": "code object",
    "metadata": "Triton metadata",
    "launch-spec": "launch spec",
}


def _buffer_bytes(raw: str) -> int:
    """Validate ``--buffer-bytes`` at parse time, before it reaches a shim or HIP."""

    try:
        value = int(raw, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer, got {raw!r}") from exc
    if value < 1 or value > _SIZE_MAX:
        raise argparse.ArgumentTypeError(f"must be in 1..{_SIZE_MAX}, got {value}")
    return value


def _grid(raw: str) -> tuple[int, int, int]:
    """Validate ``--grid`` at parse time, so emit-command cannot bake a bad one in."""

    try:
        return parse_grid(raw)
    except LoaderError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cache-entry",
        type=Path,
        help="Triton cache directory (one entry, or a cache root to search)",
    )
    source.add_argument("--hsaco", type=Path, help="Triton code object path")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="sidecar metadata JSON (default: the .json beside --hsaco)",
    )
    parser.add_argument("--kernel-name", default=None, help="disambiguate by kernel name")
    parser.add_argument("--cache-hash", default=None, help="disambiguate by Triton hash prefix")
    parser.add_argument(
        "--mode",
        choices=("load", "dispatch"),
        default="load",
        help="load the object only (default), or also launch it once",
    )
    parser.add_argument(
        "--launch-spec",
        type=Path,
        default=None,
        help="JSON supplying 'signature' when the Triton metadata omits it",
    )
    parser.add_argument(
        "--grid",
        type=_grid,
        default=(1, 1, 1),
        help="dispatch grid 'X,Y,Z' (default 1,1,1)",
    )
    parser.add_argument(
        "--buffer-bytes",
        type=_buffer_bytes,
        default=_DEFAULT_BUFFER_BYTES,
        help="bytes to allocate per pointer argument in dispatch mode",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="scalar argument override; repeatable (default: every scalar is 0)",
    )


def _resolve_entry(args: argparse.Namespace) -> CacheEntry:
    if args.hsaco is not None:
        return entry_from_hsaco(args.hsaco, args.metadata)
    entries = discover_entries(args.cache_entry)
    return select_entry(entries, kernel_name=args.kernel_name, cache_hash=args.cache_hash)


def _loader_argv(args: argparse.Namespace, entry: CacheEntry) -> list[str]:
    """Build the fully-resolved ``run`` argv a shim should exec.

    Everything is resolved to an absolute path and a concrete object here, so the
    shim never depends on the cache still being searchable or on a relative
    working directory at ConSan time.
    """

    argv = [
        "run",
        "--hsaco",
        str(entry.hsaco),
        "--metadata",
        str(entry.metadata_path),
        "--mode",
        args.mode,
        "--expect-object-sha256",
        sha256_file(entry.hsaco),
        "--expect-metadata-sha256",
        sha256_file(entry.metadata_path),
    ]
    if args.mode == "dispatch":
        grid = ",".join(str(dim) for dim in args.grid)
        argv += ["--grid", grid, "--buffer-bytes", str(args.buffer_bytes)]
        if args.launch_spec is not None:
            launch_spec = args.launch_spec.resolve()
            argv += [
                "--launch-spec",
                str(launch_spec),
                "--expect-launch-spec-sha256",
                sha256_file(launch_spec),
            ]
        for override in args.arg:
            argv += ["--arg", override]
    return argv


def _command_run(args: argparse.Namespace) -> int:
    entry = _resolve_entry(args)
    # Check every pinned input before touching the device, so a cache that was
    # repopulated since emission fails closed instead of loading other bytes.
    verify_digest(entry.hsaco, args.expect_object_sha256, what="code object")
    verify_digest(entry.metadata_path, args.expect_metadata_sha256, what="Triton metadata")
    if args.expect_launch_spec_sha256 is not None and not args.launch_spec:
        # A requested check that silently does not run is worse than no check.
        raise LoaderError("--expect-launch-spec-sha256 given without --launch-spec")
    hip = Hip()
    if args.mode == "load":
        run_load(hip, entry)
        return 0
    launch_spec = None
    if args.launch_spec:
        verify_digest(args.launch_spec, args.expect_launch_spec_sha256, what="launch spec")
        launch_spec = read_json_object(args.launch_spec, what="launch spec")
    specs = parse_signature(resolve_signature(entry, launch_spec))
    run_dispatch(
        hip,
        entry,
        specs=specs,
        grid=args.grid,
        buffer_bytes=args.buffer_bytes,
        scalars=parse_arg_overrides(args.arg),
    )
    return 0


def _command_emit(args: argparse.Namespace) -> int:
    entry = _resolve_entry(args)
    loader = Path(__file__).resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_shim(loader, _loader_argv(args, entry), entry=entry), encoding="utf-8"
    )
    args.output.chmod(args.output.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"[triton-consan-loader] wrote {args.output} -> {entry.kernel_name} ({args.mode})")
    return 0


def _command_list(args: argparse.Namespace) -> int:
    for entry in discover_entries(args.cache_entry):
        print(f"{entry.kernel_name}\t{entry.metadata.get('hash', '?')}\t{entry.arch}\t{entry.hsaco}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="load (and optionally dispatch) the code object")
    _add_selection_arguments(run)
    # Set by emit-command so the shim pins the bytes it was generated against, not
    # just their paths. Optional when running by hand against a live cache.
    for name, what in _PINNED_INPUTS.items():
        run.add_argument(
            f"--expect-{name}-sha256",
            default=None,
            help=f"fail unless the {what} matches this SHA-256 (set by emit-command)",
        )
    run.set_defaults(handler=_command_run)

    emit = subparsers.add_parser(
        "emit-command", help="write a zero-argument shim for source.consan_command"
    )
    _add_selection_arguments(emit)
    emit.add_argument("--output", type=Path, required=True, help="shim path to write")
    emit.set_defaults(handler=_command_emit)

    listing = subparsers.add_parser("list", help="list the Triton cache entries under a directory")
    listing.add_argument("--cache-entry", type=Path, required=True)
    listing.set_defaults(handler=_command_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except LoaderError as exc:
        print(f"triton_consan_loader: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
