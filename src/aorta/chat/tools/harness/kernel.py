"""Turn a pasted HIP kernel into a runnable program.

Engineers paste the kernel they are worried about, not a whole benchmark, but
ConSan analyses a *process*: it needs something to launch. This module builds the
missing main() by reading the kernel signature and allocating one device buffer
per pointer parameter, so the paste-a-kernel path works without asking the user
to write boilerplate they do not have.

The generated harness only needs to make the kernel execute; it does not need to
compute anything meaningful, since the race is a property of the memory access
pattern rather than of the values involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aorta.chat.tools.harness._fences import extract_fenced

DEFAULT_BLOCK = 256
DEFAULT_GRID = 1
# gfx950 executes 64 lanes per wavefront; conflicts are only visible between waves.
WAVEFRONT = 64
# Buffers are padded well past the launch geometry: an out-of-bounds read in the
# harness would fault and mask the race we are trying to observe.
MIN_ELEMENTS = 4096

_KERNEL_SIG = re.compile(r"__global__\s+(?:[\w:]+\s+)*?void\s+(\w+)\s*\(", re.MULTILINE)
_SHARED_DECL = re.compile(r"__shared__\s+[\w:]+\s+\w+\s*\[\s*([^\]]+?)\s*\]")
_DEFINE = re.compile(r"^\s*#\s*define\s+(\w+)\s+(\w+)\s*$", re.MULTILINE)
#: Block and grid axes are separate launch contracts. Supplying ``block_y`` for
#: ``blockIdx.y`` used to satisfy one combined check while the scalar grid still
#: pinned that index at zero.
_USES_BLOCK_DIM = {
    "y": re.compile(r"\b(?:threadIdx|blockDim)\.y\b"),
    "z": re.compile(r"\b(?:threadIdx|blockDim)\.z\b"),
}
_USES_GRID_DIM = {
    "y": re.compile(r"\b(?:blockIdx|gridDim)\.y\b"),
    "z": re.compile(r"\b(?:blockIdx|gridDim)\.z\b"),
}


def block_dims_used(source: str) -> list[str]:
    """Which y/z block axes the kernel uses, in order."""
    return [
        axis for axis, pattern in _USES_BLOCK_DIM.items() if pattern.search(source)
    ]


def grid_dims_used(source: str) -> list[str]:
    """Which y/z grid axes the kernel uses, in order."""
    return [
        axis for axis, pattern in _USES_GRID_DIM.items() if pattern.search(source)
    ]


def dims_used(source: str) -> list[str]:
    """Which y/z axes appear in either launch domain, for compatibility."""
    block = set(block_dims_used(source))
    grid = set(grid_dims_used(source))
    return [axis for axis in ("y", "z") if axis in block or axis in grid]
_MAIN = re.compile(r"\bint\s+main\s*\(", re.MULTILINE)
_IDENT = re.compile(r"[A-Za-z_]\w*$")


class HarnessError(ValueError):
    """The pasted source cannot be turned into a runnable program."""


@dataclass(frozen=True)
class Param:
    base_type: str
    name: str
    is_pointer: bool


def looks_like_hip(block: str) -> bool:
    """Whether a fenced block is the kernel rather than a log or a diff.

    A message often carries several: the kernel, the error it produced, the
    launch line someone tried. What identifies this one is a device function or
    a program entry point -- the two things this harness knows how to build.
    """
    return bool(_KERNEL_SIG.search(block) or _MAIN.search(block))


def extract_source(text: str) -> str:
    """The HIP out of a message that also explains the problem.

    The assembly harness has done this since it was written; this one had the
    same messages pasted into it and none of the extraction, so a fenced paste
    reached hipcc with the backticks and the prose still attached and the
    compile failed on the English.
    """
    code, _fenced = extract_fenced(text, looks_like_hip)
    return code


def has_main(source: str) -> bool:
    return bool(_MAIN.search(source))


def kernel_name(source: str) -> str:
    match = _KERNEL_SIG.search(source)
    if not match:
        raise HarnessError(
            "no __global__ kernel found in the pasted source. Paste the kernel "
            "definition, including its __global__ signature."
        )
    return match.group(1)


def _split_params(text: str) -> list[str]:
    """Split a parameter list on commas that are not nested in <> or ()."""
    parts, depth, current = [], 0, []
    for char in text:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _parse_param(text: str, index: int) -> Param:
    cleaned = text.replace("__restrict__", " ").replace("restrict", " ").strip()
    is_pointer = "*" in cleaned
    if "[" in cleaned:  # array parameter decays to a pointer
        is_pointer = True
        cleaned = cleaned[: cleaned.index("[")]
    cleaned = cleaned.replace("*", " ").replace("&", " ")

    tokens = [t for t in cleaned.split() if t not in {"const", "volatile", "struct"}]
    if not tokens:
        raise HarnessError(f"could not parse parameter {index + 1}: {text!r}")

    # A trailing identifier is the parameter name; without one (e.g. "float*")
    # synthesise a name so the harness still compiles.
    if len(tokens) > 1 and _IDENT.match(tokens[-1]):
        name, base_tokens = tokens[-1], tokens[:-1]
    else:
        name, base_tokens = f"arg{index}", tokens
    return Param(base_type=" ".join(base_tokens), name=name, is_pointer=is_pointer)


def parse_params(source: str) -> list[Param]:
    match = _KERNEL_SIG.search(source)
    if not match:
        raise HarnessError("no __global__ kernel found in the pasted source.")

    start = match.end()  # just past '('
    depth, end = 1, None
    for i in range(start, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise HarnessError("unbalanced parentheses in the kernel signature.")

    inner = source[start:end].strip()
    if not inner or inner == "void":
        return []
    return [_parse_param(p, i) for i, p in enumerate(_split_params(inner))]


def infer_block_size(source: str) -> int:
    """Guess the intended block size from the largest __shared__ array.

    A shared array is almost always sized to the block, so this recovers the
    geometry the author had in mind; guessing too small would leave part of the
    array untouched and could hide the race.
    """
    defines = {m.group(1): m.group(2) for m in _DEFINE.finditer(source)}
    best = 0
    for match in _SHARED_DECL.finditer(source):
        raw = match.group(1).strip()
        value = defines.get(raw, raw)
        try:
            extent = int(value, 0)
        except ValueError:
            continue
        if 1 <= extent <= 1024:
            best = max(best, extent)
    return best or DEFAULT_BLOCK


def _block_arg(block: int, block_y: int, block_z: int) -> str:
    """The block argument, as dim3 only when more than one dimension is in use.

    A scalar where a scalar will do: <<<1, 256>>> is what someone reading the
    generated harness expects, and dim3(256, 1, 1) invites the question of what
    the other two are for.
    """
    if block_z:
        return f"dim3({block}, {max(block_y, 1)}, {block_z})"
    if block_y:
        return f"dim3({block}, {block_y})"
    return str(block)


def _grid_arg(grid: int, grid_y: int, grid_z: int) -> str:
    """The grid argument, preserving a scalar for a one-dimensional launch."""
    if grid_z:
        return f"dim3({grid}, {max(grid_y, 1)}, {grid_z})"
    if grid_y:
        return f"dim3({grid}, {grid_y})"
    return str(grid)


def build_harness(
    source: str,
    *,
    block: int = 0,
    grid: int = 0,
    elements: int = 0,
    block_y: int = 0,
    block_z: int = 0,
    grid_y: int = 0,
    grid_z: int = 0,
    fill_byte: int = 0,
) -> str:
    """Wrap a bare kernel in a main() that launches it once.

    *fill_byte* is the byte every input buffer is filled with. Changing it can
    explore a different path, but one repeated byte is not representative
    input and never makes a clean result conclusive.
    """
    name = kernel_name(source)
    params = parse_params(source)
    block = block or infer_block_size(source)
    grid = grid or DEFAULT_GRID
    if not 1 <= block <= 1024:
        raise HarnessError(f"block size {block} is outside the valid range 1..1024.")
    if grid < 1:
        raise HarnessError(f"grid size {grid} must be at least 1.")
    for dimension_name, value in (
        ("block_y", block_y),
        ("block_z", block_z),
        ("grid_y", grid_y),
        ("grid_z", grid_z),
    ):
        if value < 0:
            raise HarnessError(
                f"{dimension_name} must be zero (unused) or a positive extent."
            )
    threads = block * max(block_y, 1) * max(block_z, 1)
    if threads > 1024:
        raise HarnessError(
            f"a block of {block}x{max(block_y, 1)}x{max(block_z, 1)} is "
            f"{threads} threads, past the 1024 a block may have."
        )
    blocks = grid * max(grid_y, 1) * max(grid_z, 1)
    count = elements or max(threads * blocks * 4, MIN_ELEMENTS)

    prologue = "" if "hip_runtime.h" in source else "#include <hip/hip_runtime.h>\n"
    prologue += "" if "cstdio" in source or "stdio.h" in source else "#include <cstdio>\n"

    setup, args, cleanup = [], [], []
    for param in params:
        if param.is_pointer:
            var = f"h_{param.name}"
            setup.append(f"  {param.base_type} *{var} = nullptr;")
            setup.append(
                f"  if (hipMalloc(&{var}, kElements * sizeof({param.base_type})) != hipSuccess)\n"
                f"    return 1;"
            )
            # Zero is the safe default for a buffer a kernel writes, and the
            # wrong one for a buffer it branches on: `if (input[i] > 0)` never
            # runs, so a race behind it cannot be reported. fill lets the
            # caller say so; the caveat in the result says when it matters.
            setup.append(
                f"  if (hipMemset({var}, {fill_byte}, kElements * sizeof({param.base_type})) != hipSuccess)\n"
                f"    return 1;"
            )
            args.append(var)
            cleanup.append(f"  if (hipFree({var}) != hipSuccess)\n    ok = false;")
        elif any(t in param.base_type for t in ("float", "double")):
            args.append("1.0f")
        else:
            # Integral scalars are nearly always a length or stride, so the
            # element count is the value least likely to walk off a buffer.
            args.append("static_cast<int>(kElements)")

    body = "\n".join(
        [
            "// Harness generated by AORTA so the pasted kernel can be launched under ConSan.",
            "int main() {",
            f"  constexpr size_t kElements = {count};",
            *setup,
            "",
            f"  {name}<<<{_grid_arg(grid, grid_y, grid_z)}, "
            f"{_block_arg(block, block_y, block_z)}>>>"
            f"({', '.join(args)});",
            "  const hipError_t launch = hipGetLastError();",
            "  const hipError_t sync = hipDeviceSynchronize();",
            "",
            "  bool ok = launch == hipSuccess && sync == hipSuccess;",
            *cleanup,
            "  if (!ok) {",
            '    fprintf(stderr, "harness: launch=%s sync=%s\\n",',
            "            hipGetErrorString(launch), hipGetErrorString(sync));",
            "  }",
            "  return ok ? 0 : 1;",
            "}",
        ]
    )
    # Includes must precede the kernel: threadIdx and friends are declared by the
    # HIP runtime header, so appending it would leave the pasted kernel unable to
    # compile against the very symbols it uses.
    return f"{prologue}\n{source.rstrip()}\n\n{body}\n"


@dataclass(frozen=True)
class Prepared:
    program: str
    kernel: str
    wrapped: bool
    block: int
    grid: int
    block_y: int = 0
    block_z: int = 0
    grid_y: int = 0
    grid_z: int = 0
    #: Pointer parameters the kernel reads inside a branch condition. Empty
    #: when it has none, or when the caller supplied their own main().
    input_guards: tuple[str, ...] = ()

    @property
    def threads(self) -> int:
        """Threads per block across every dimension the launch uses."""
        return self.block * max(self.block_y, 1) * max(self.block_z, 1)

    @property
    def generated_inputs(self) -> bool:
        """Whether AORTA, rather than the caller, chose every argument value.

        This is the clean-result boundary. Synthetic inputs exercise one path;
        no source regex can prove that path represents scalar guards, values
        derived from pointers, ternaries, switches, or conditions hidden in a
        helper. A caller-supplied main is the only case where this harness did
        not choose the data.
        """
        return self.wrapped

    @property
    def data_dependent(self) -> bool:
        """Whether the lightweight scan found an obvious pointer guard.

        Informational only. It enriches the generated-input caveat but never
        decides whether that caveat is shown.
        """
        return self.wrapped and bool(self.input_guards)

    @property
    def single_wave(self) -> bool:
        """A one-wavefront launch cannot exhibit a cross-wave race.

        ConSan finds conflicts between waves, so this geometry can only ever
        return a clean result and must not be read as proof the kernel is safe.

        Counted across the block rather than its first dimension: 32x32 is a
        thousand threads and sixteen waves, and reading the 32 alone would have
        warned about a launch that is nothing like single-wave.
        """
        return self.wrapped and self.threads <= WAVEFRONT


#: A direct pointer read in an if/while condition. Deliberately only a hint for
#: the message: it cannot see aliases, scalar parameters, ternaries, switches,
#: helper functions, or a value computed before the condition, so it must never
#: be used to declare a generated run representative.
_GUARD = re.compile(r"\b(?:if|while)\s*\(([^)]*)\)", re.S)


def branches_on_input(source: str, params: list[Param]) -> list[str]:
    """Obvious pointer parameters to name in the generated-input caveat.

    The generated harness fills every buffer with zeros, so a branch guarded
    by one of them never runs. ConSan then reports a clean sweep of code it
    never reached -- and the report says "pass", not "the guarded path was not
    executed". A kernel with a race behind ``if (input[i] > 0)`` comes back
    clean and looks diagnosed. Missing a name here does not remove the caveat.
    """
    pointers = [p.name for p in params if p.is_pointer]
    if not pointers:
        return []
    found: list[str] = []
    for condition in _GUARD.findall(source):
        for name in pointers:
            if name not in found and re.search(rf"\b{re.escape(name)}\s*\[", condition):
                found.append(name)
    return found


def prepare_source(
    source: str,
    *,
    block: int = 0,
    grid: int = 0,
    elements: int = 0,
    block_y: int = 0,
    block_z: int = 0,
    grid_y: int = 0,
    grid_z: int = 0,
    fill_byte: int = 0,
) -> Prepared:
    """Turn a pasted kernel or program into something ConSan can run."""
    source = extract_source(source).strip()
    if not source:
        raise HarnessError("no source was provided.")
    if "template" in source and "__global__" in source:
        raise HarnessError(
            "templated kernels are not supported yet: paste a concrete "
            "instantiation, or include your own main()."
        )
    name = kernel_name(source)
    if has_main(source):
        needs_include = "hip_runtime.h" not in source
        prefix = "#include <hip/hip_runtime.h>\n" if needs_include else ""
        return Prepared(prefix + source + "\n", name, False, 0, 0)

    # Missing block and grid dimensions are independent. Neither fails at
    # launch: the corresponding index is pinned at zero and ConSan observes a
    # different execution that can look clean.
    missing_block = [
        axis
        for axis in block_dims_used(source)
        if not (block_y if axis == "y" else block_z)
    ]
    missing_grid = [
        axis
        for axis in grid_dims_used(source)
        if not (grid_y if axis == "y" else grid_z)
    ]
    if missing_block or missing_grid:
        requirements = []
        examples = []
        if missing_block:
            axes = " and ".join(
                f"threadIdx.{axis}/blockDim.{axis}" for axis in missing_block
            )
            arguments = ", ".join(f"block_{axis}" for axis in missing_block)
            requirements.append(f"{axes} requires {arguments}")
            examples.append("block_size=32")
            examples.extend(f"block_{axis}=32" for axis in missing_block)
        if missing_grid:
            axes = " and ".join(
                f"blockIdx.{axis}/gridDim.{axis}" for axis in missing_grid
            )
            arguments = ", ".join(f"grid_{axis}" for axis in missing_grid)
            requirements.append(f"{axes} requires {arguments}")
            examples.append("grid_size=4")
            examples.extend(f"grid_{axis}=4" for axis in missing_grid)
        raise HarnessError(
            "this kernel uses multidimensional launch geometry: "
            + "; ".join(requirements)
            + f". Pass each extent explicitly (for example {', '.join(examples)}). "
            "With a missing dimension its "
            "index remains zero and the run does not exercise what you pasted."
        )

    resolved_block = block or infer_block_size(source)
    resolved_grid = grid or DEFAULT_GRID
    program = build_harness(
        source,
        block=resolved_block,
        grid=resolved_grid,
        elements=elements,
        block_y=block_y,
        block_z=block_z,
        grid_y=grid_y,
        grid_z=grid_z,
        fill_byte=fill_byte,
    )
    return Prepared(
        program=program,
        kernel=name,
        wrapped=True,
        block=resolved_block,
        grid=resolved_grid,
        block_y=block_y,
        block_z=block_z,
        grid_y=grid_y,
        grid_z=grid_z,
        input_guards=tuple(branches_on_input(source, parse_params(source))),
    )
