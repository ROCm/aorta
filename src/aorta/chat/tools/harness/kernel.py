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
_MAIN = re.compile(r"\bint\s+main\s*\(", re.MULTILINE)
_IDENT = re.compile(r"[A-Za-z_]\w*$")


class HarnessError(ValueError):
    """The pasted source cannot be turned into a runnable program."""


@dataclass(frozen=True)
class Param:
    base_type: str
    name: str
    is_pointer: bool


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


def build_harness(
    source: str, *, block: int = 0, grid: int = 0, elements: int = 0
) -> str:
    """Wrap a bare kernel in a main() that launches it once."""
    name = kernel_name(source)
    params = parse_params(source)
    block = block or infer_block_size(source)
    grid = grid or DEFAULT_GRID
    if not 1 <= block <= 1024:
        raise HarnessError(f"block size {block} is outside the valid range 1..1024.")
    count = elements or max(block * grid * 4, MIN_ELEMENTS)

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
            setup.append(
                f"  if (hipMemset({var}, 0, kElements * sizeof({param.base_type})) != hipSuccess)\n"
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
            f"  {name}<<<{grid}, {block}>>>({', '.join(args)});",
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

    @property
    def single_wave(self) -> bool:
        """A one-wavefront launch cannot exhibit a cross-wave race.

        ConSan finds conflicts between waves, so this geometry can only ever
        return a clean result and must not be read as proof the kernel is safe.
        """
        return self.wrapped and self.block <= WAVEFRONT


def prepare_source(
    source: str, *, block: int = 0, grid: int = 0, elements: int = 0
) -> Prepared:
    """Turn a pasted kernel or program into something ConSan can run."""
    source = source.strip()
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

    resolved_block = block or infer_block_size(source)
    resolved_grid = grid or DEFAULT_GRID
    program = build_harness(
        source, block=resolved_block, grid=resolved_grid, elements=elements
    )
    return Prepared(program, name, True, resolved_block, resolved_grid)
