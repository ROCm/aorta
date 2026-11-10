"""A launch harness for a kernel-only stand-in candidate (design Section 6.10 item 7; the standin-gpu
profile's ConSan, ASan and waitcheck legs).

The stand-in prompt asks for a HIP source that defines its `__global__` kernel and no `main` (`templates.STANDIN_USER0`),
but aorta's ConSan and ASan run a program, and a template kernel has no code until something instantiates it. For a
source without `main`, `harness_source` writes a translation unit that includes the candidate unchanged and adds, for
each kernel instance it is given, a launcher in the kernel's own namespace, so every name the source uses resolves as
it does there, and a `main` that launches one instance once.

The kernels come from clang's own AST of the source (`find_kernels`: `-ast-dump-filter` on each `__global__` name the
source text declares, host side), never from parsing C++ here:

  - every kernel the source defines is one instance, and several kernels in a source are checked one at a time;
  - a kernel in a namespace, named or anonymous, is launched from inside that namespace;
  - an overloaded kernel is taken by its exact function type, as clang prints it;
  - a template kernel is instantiated with representative arguments (`template_attempts`): a type parameter is float,
    or int when its name says it is an index, a size or a count (then every type parameter int); an integral value
    parameter is the launch's block size when its name says block or threads, the wavefront when it says warp or
    wave, otherwise NTTP_INT; a bool is false; a trailing parameter with a default keeps it; a trailing pack is empty.

The launch arguments come from the instantiated kernel's own signature (a function template over
`void (*)(Args...)`), by one rule applied to parameters and, recursively, to the members of a struct:

  - a pointer gets its own zero-filled device buffer of BUFFER_BYTES (a pointer to an aggregate struct also gets one
    value of it, built by this rule, in its first element); a function pointer is null, since there is no device
    function to give it, and a kernel that calls it faults, which settles nothing;
  - a reference refers to such a buffer, holding one value built by this rule;
  - an integer is INT_ARG, a bool true, an enum zero, a floating-point (or float-constructible, such as `__half`)
    value 1;
  - an aggregate struct (`std::array<char*, N>`, a functor, a metadata struct) is brace-initialised member by member
    by this rule, up to FILL_MAX members (brace elision flattens arrays), the rest zero; another struct is built from
    1.0f if it can be and is trivially copyable, else default-constructed;
  - the launch is GRID blocks of the block size (BLOCK, or fewer if `__launch_bounds__` asks for fewer), with
    DYNAMIC_LDS_BYTES of dynamic shared memory when the source declares `extern __shared__`;
  - `main` returns 0 only if the launch and `hipDeviceSynchronize` both succeed.

Anything else fails closed and leaves the leg unsettled, never clean: a struct the rule cannot build, a template
template parameter, a value parameter of another type, a kernel that is a class member, an instantiation that does not
compile with any attempt (`harness_build_failed`), or a source in which clang finds no kernel definition
(`NoKernel`, a `HarnessUnsupported`).

This is a stand-in. Its launch is not the kernel's intended one, so a clean leg means "no memory error or race in this
one launch", and a finding may come from a launch the kernel was not written for. Runs on it are mechanics only.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

HARNESS_VERSION = "2"
BUFFER_BYTES = 16 << 20
INT_ARG = 64
GRID = 4
BLOCK = 256
WAVEFRONT = 64  # gfx950
NTTP_INT = 4
FILL_MAX = 16
DYNAMIC_LDS_BYTES = 16 << 10
HARNESS_NAME = "aorlta_harness.hip"
AST_TIMEOUT_S = 120

HAS_MAIN = re.compile(r"\bint\s+main\s*\(")
DYNAMIC_SHARED = re.compile(r"\bextern\s+__shared__\b")
LAUNCH_BOUNDS = re.compile(r"__launch_bounds__\s*\(\s*(\d+)")
_ATTR = r"(?:__launch_bounds__\s*\([^()]*\)|__attribute__\s*\(\([^()]*(?:\([^()]*\)[^()]*)*\)\))"
GLOBAL_FN = re.compile(rf"__global__\s+(?:(?:[A-Za-z_]\w*|{_ATTR})\s+)*?void\s+(?:{_ATTR}\s+)*([A-Za-z_]\w*)\s*\(")
IDENT = re.compile(r"[A-Za-z_]\w*")
INDEX_LIKE = re.compile(r"(?i)index|idx|offset|count|size|^i$|^int|_int$|^n$")
CALLABLE_LIKE = re.compile(r"(?i)func|^op|op$|^f$|lambda|combine|reduc|pred|callable")
FUNCTOR = "::aorlta_harness::functor"
BLOCK_LIKE = re.compile(r"(?i)block|thread")
WAVE_LIKE = re.compile(r"(?i)warp|wave")
INTEGRAL_TYPES = frozenset({
    "int", "unsigned", "unsigned int", "signed", "signed int", "long", "unsigned long", "long long",
    "unsigned long long", "short", "unsigned short", "char", "signed char", "unsigned char", "size_t", "std::size_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "std::int32_t",
    "std::int64_t", "std::uint32_t", "std::uint64_t", "index_t", "ptrdiff_t", "std::ptrdiff_t"})


class HarnessUnsupported(ValueError):
    """The harness cannot launch this source's kernels; the leg stays unsettled."""


class NoKernel(HarnessUnsupported):
    """Clang read the source and found no kernel defined in it: nothing to launch, and nothing to instantiate."""


def has_main(source_text: str) -> bool:
    return HAS_MAIN.search(strip_comments(source_text)) is not None


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def dynamic_lds_bytes(source_text: str) -> int:
    """The dynamic LDS the harness launches every kernel of this source with: DYNAMIC_LDS_BYTES when the source
    declares `extern __shared__`, else 0."""
    return DYNAMIC_LDS_BYTES if DYNAMIC_SHARED.search(strip_comments(source_text)) else 0


def launch_block(source_text: str) -> int:
    """BLOCK, or the smallest `__launch_bounds__` maximum in the source if that is lower."""
    bounds = [int(b) for b in LAUNCH_BOUNDS.findall(strip_comments(source_text)) if int(b) > 0]
    return min([BLOCK, *bounds])


def declared_names(source_text: str) -> list[str]:
    """The names the source text declares `__global__` (comments are not code); clang decides what they are."""
    return sorted(set(GLOBAL_FN.findall(strip_comments(source_text))))


# --- the kernels, from clang's AST ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateParam:
    kind: str  # "type" | "value" | "template"
    name: str
    type: str | None  # a value parameter's type, as written
    has_default: bool
    pack: bool


@dataclass(frozen=True)
class Kernel:
    name: str
    scope: tuple[str, ...]  # enclosing namespaces, outermost first; "" is an anonymous namespace
    template: tuple[TemplateParam, ...] | None  # None: not a template
    function_type: str  # as clang prints it, e.g. "void (float *, int)"
    overloaded: bool = False  # another kernel of this name in this scope
    specialization: tuple[str, ...] | None = None  # an explicit specialization's own template arguments

    @property
    def qualified(self) -> str:
        return "::".join([*(s or "(anonymous namespace)" for s in self.scope), self.name])


def _objects(text: str) -> list[dict]:
    dec, i, out = json.JSONDecoder(), 0, []
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text):
            obj, i = dec.raw_decode(text, i)
            out.append(obj)
    return out


def _loc_files(node, cur: str | None, out: list) -> str | None:
    """Walk a JSON AST dump in its own order: clang names a location's file only when it changes from the last
    location it printed, so the file in force at each top-level node is carried across them."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("loc", "begin", "end", "spellingLoc", "expansionLoc") and isinstance(value, dict):
                if isinstance(value.get("file"), str):
                    cur = value["file"]
                cur = _loc_files(value, cur, out)
            elif key == "file" and isinstance(value, str):
                cur = value
            elif isinstance(value, (dict, list)):
                cur = _loc_files(value, cur, out)
    elif isinstance(node, list):
        for value in node:
            cur = _loc_files(value, cur, out)
    return cur


def _top_level_files(objs: list[dict]) -> list[str | None]:
    """The file of each top-level node's own location (`loc`, or its expansion for a name made by a macro)."""
    files, cur = [], None
    for obj in objs:
        loc = obj.get("loc") or {}
        own = loc.get("file") or (loc.get("expansionLoc") or {}).get("file") or (loc.get("spellingLoc") or {}).get(
            "file")
        files.append(own or cur)
        cur = _loc_files(obj, cur, [])
    return files


def _template_param(node: dict) -> TemplateParam | None:
    kind = {"TemplateTypeParmDecl": "type", "NonTypeTemplateParmDecl": "value",
            "TemplateTemplateParmDecl": "template"}.get(node.get("kind"))
    if kind is None:
        return None
    default = any(isinstance(c, dict) and c.get("kind") == "TemplateArgument" for c in node.get("inner") or [])
    return TemplateParam(kind, str(node.get("name") or ""), (node.get("type") or {}).get("qualType"),
                         bool(default or node.get("defaultArg")), bool(node.get("isParameterPack")))


def _kernel_function(node: dict) -> tuple[dict | None, tuple[TemplateParam, ...] | None]:
    if node.get("kind") == "FunctionDecl":
        return node, None
    if node.get("kind") == "FunctionTemplateDecl":
        inner = node.get("inner") or []
        params = tuple(p for p in (_template_param(c) for c in inner if isinstance(c, dict)) if p is not None)
        fn = next((c for c in inner if isinstance(c, dict) and c.get("kind") == "FunctionDecl"), None)
        return fn, params
    return None, None


def _specialization(fn: dict) -> tuple[str, ...] | None:
    """An explicit specialization's template arguments, as clang prints them; None for any other function."""
    args = [c for c in fn.get("inner") or [] if isinstance(c, dict) and c.get("kind") == "TemplateArgument"]
    if not args:
        return None
    out = []
    for a in args:
        if isinstance((a.get("type") or {}).get("qualType"), str):
            out.append(a["type"]["qualType"])
        elif isinstance(a.get("value"), (str, int)):
            out.append(str(a["value"]))
        else:
            raise HarnessUnsupported("an explicit specialization with an argument the harness cannot spell")
    return tuple(out)


def _is_kernel_definition(fn: dict) -> bool:
    inner = [c for c in fn.get("inner") or [] if isinstance(c, dict)]
    return any(c.get("kind") == "CUDAGlobalAttr" for c in inner) and any(c.get("kind") == "CompoundStmt" for c in inner)


def _scope(qualified: str, name: str) -> tuple[str, ...]:
    parts = qualified.split("::")
    if parts[-1] != name:
        raise HarnessUnsupported(f"clang names {qualified!r}, not a kernel {name!r}")
    scope = tuple("" if p == "(anonymous namespace)" else p for p in parts[:-1])
    if not all(s == "" or IDENT.fullmatch(s) for s in scope):
        raise HarnessUnsupported(f"{qualified!r} is not in a namespace the harness can reopen")
    return scope


def execute(argv: Sequence[str], timeout_s: float) -> tuple[int | None, str]:
    """Run a command: (exit code, stdout); None when it did not finish. Adapters pass their own killable runner."""
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True, errors="replace", timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError):
        return None, ""
    return proc.returncode, proc.stdout


def _clang_dump(run, hipcc: str, target: str, source: Path, name: str, json_mode: bool) -> str:
    argv = [hipcc, "-x", "hip", f"--offload-arch={target}", "--cuda-host-only", "-fsyntax-only", "-w", "-Xclang",
            "-ast-dump=json" if json_mode else "-ast-dump", "-Xclang", f"-ast-dump-filter={name}", str(source)]
    rc, out = run(argv, AST_TIMEOUT_S)
    if rc != 0:
        raise HarnessUnsupported(f"clang could not read the source (exit {rc})")
    return out


def find_kernels(source_path: str | Path, *, hipcc: str = "/opt/rocm/bin/hipcc", target: str = "gfx950",
                 run=execute) -> list[Kernel]:
    """Every `__global__` function the source itself defines, as clang reads it. Raises HarnessUnsupported when there
    is none, or when one is a shape the harness cannot launch (a class member, a declaration clang and the dump's
    headers disagree on)."""
    source = Path(source_path).resolve()
    names = declared_names(source.read_text(encoding="utf-8", errors="replace"))
    found: list[Kernel] = []
    for name in names:
        heads = [line[len("Dumping "):-1] for line in _clang_dump(run, hipcc, target, source, name, False).splitlines()
                 if line.startswith("Dumping ") and line.endswith(":")]
        objs = _objects(_clang_dump(run, hipcc, target, source, name, True))
        if len(heads) != len(objs):
            raise HarnessUnsupported(f"{name}: clang's two dumps disagree ({len(heads)} against {len(objs)})")
        for qualified, obj, file in zip(heads, objs, _top_level_files(objs)):
            if obj.get("name") != name or file is None or Path(file).resolve() != source:
                continue  # another declaration the filter matched, or one in a header
            if obj.get("kind") == "CXXMethodDecl" or (obj.get("kind") == "FunctionTemplateDecl" and any(
                    isinstance(c, dict) and c.get("kind") == "CXXMethodDecl" for c in obj.get("inner") or [])):
                raise HarnessUnsupported(f"{qualified} is a class member")
            fn, tparams = _kernel_function(obj)
            if fn is None or not _is_kernel_definition(fn):
                continue  # a declaration without a body, or not a kernel
            ftype = (fn.get("type") or {}).get("qualType")
            if not isinstance(ftype, str) or not ftype.startswith("void ("):
                raise HarnessUnsupported(f"{qualified}: an unexpected kernel type {ftype!r}")
            spec = _specialization(fn) if tparams is None else None
            found.append(Kernel(name, _scope(qualified, name), tparams, ftype, specialization=spec))
    if not found:
        raise NoKernel("no kernel definition in the source")
    keys = [(k.scope, k.name) for k in found]  # a plain kernel shares its name with a template: it is taken by type
    return [dataclasses.replace(k, overloaded=keys.count((k.scope, k.name)) > 1) for k in found]


# --- template arguments ---------------------------------------------------------------------------------------


def _value_arg(p: TemplateParam, block: int) -> str:
    t = re.sub(r"\b(const|volatile)\b", "", p.type or "").strip()
    if t == "bool":
        return "false"
    if t not in INTEGRAL_TYPES:
        raise HarnessUnsupported(f"a template value parameter of type {p.type!r}")
    if BLOCK_LIKE.search(p.name):
        return str(block)
    if WAVE_LIKE.search(p.name):
        return str(WAVEFRONT)
    return str(NTTP_INT)


def template_attempts(kernel: Kernel, block: int) -> list[list[str]]:
    """The explicit template argument lists to try, in order: index-like type parameters int and the rest float, then
    every type parameter int. A trailing run of defaulted parameters, and a trailing pack, are left to the template."""
    params = list(kernel.template or ())
    if any(p.kind == "template" for p in params):
        raise HarnessUnsupported("a template template parameter")
    if any(p.pack for p in params[:-1]):
        raise HarnessUnsupported("a template parameter pack that is not last")
    while params and (params[-1].pack or params[-1].has_default):
        params.pop()
    fixed = {i: _value_arg(p, block) for i, p in enumerate(params) if p.kind == "value"}
    fixed.update({i: FUNCTOR for i, p in enumerate(params) if p.kind == "type" and CALLABLE_LIKE.search(p.name)})
    named = [fixed.get(i) or ("int" if INDEX_LIKE.search(p.name) else "float") for i, p in enumerate(params)]
    ints = [fixed.get(i) or "int" for i, p in enumerate(params)]
    return [named] if named == ints else [named, ints]


@dataclass(frozen=True)
class Instance:
    """One kernel to launch: the kernel, its explicit template arguments (None if not a template) and the block."""
    kernel: Kernel
    template_args: tuple[str, ...] | None
    block: int

    @property
    def label(self) -> str:
        args = "" if self.template_args is None else "<" + ", ".join(self.template_args) + ">"
        typed = f" [{self.kernel.function_type}]" if self.template_args is None and self.kernel.overloaded else ""
        return self.kernel.qualified + args + typed

    def reference(self) -> str:
        """How the launcher, inside the kernel's namespace, names this instance."""
        k = self.kernel
        if self.template_args is not None:
            return f"&{k.name}<{', '.join(self.template_args)}>"
        if k.overloaded:
            return f"static_cast<{k.function_type.replace('void (', 'void (*)(', 1)}>(&{k.name})"
        return f"&{k.name}"


ARG_TEXT = re.compile(r"[^;{}\n\\\"'`#]+")


# --- the translation unit -----------------------------------------------------------------------------------------

_PRELUDE = """// AORLTA stand-in launch harness (aorlta.reward.adapters.standin_harness); generated, not the candidate's.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "{include}"

namespace aorlta_harness {{
constexpr size_t kBufferBytes = {buffer};
constexpr long long kInt = {int_arg};
constexpr size_t kFillMax = {fill_max};
constexpr size_t kNoFill = static_cast<size_t>(-1);
template <typename T> struct unsupported : std::false_type {{}};

// The representative type of a template parameter that is called (its name says functor, op, func...): it gives back
// its first argument.
struct functor {{
  template <typename A, typename... R>
  __host__ __device__ constexpr A operator()(A a, R...) const {{ return a; }}
}};

inline std::vector<void*>& buffers() {{ static std::vector<void*> b; return b; }}
inline int& depth() {{ static int d = 0; return d; }}

inline void* device_zeros(size_t bytes) {{
  void* p = nullptr;
  if (hipMalloc(&p, bytes) != hipSuccess || hipMemset(p, 0, bytes) != hipSuccess) {{
    std::fprintf(stderr, "aorlta harness: device allocation failed\\n");
    std::exit(3);
  }}
  buffers().push_back(p);
  return p;
}}

template <typename T> T make();

struct filler {{
  template <typename V> operator V() const {{ return make<V>(); }}
}};

template <typename T, size_t... I>
constexpr auto braces(std::index_sequence<I...>) -> decltype(T{{(void(I), filler{{}})...}}, true) {{ return true; }}
template <typename T>
constexpr bool braces(...) {{ return false; }}

template <typename T, size_t N = kFillMax>
constexpr size_t fill_count() {{
  if constexpr (braces<T>(std::make_index_sequence<N>{{}})) return N;
  else if constexpr (N == 0) return kNoFill;
  else return fill_count<T, N - 1>();
}}

template <typename T, size_t... I>
T aggregate(std::index_sequence<I...>) {{ return T{{(void(I), filler{{}})...}}; }}

template <typename E>
void put(void* p, const E& v) {{
  if (hipMemcpy(p, &v, sizeof(E), hipMemcpyHostToDevice) != hipSuccess) {{
    std::fprintf(stderr, "aorlta harness: device copy failed\\n");
    std::exit(3);
  }}
}}

template <typename E, typename = void> struct complete : std::false_type {{}};
template <typename E> struct complete<E, std::void_t<decltype(sizeof(E))>> : std::true_type {{}};

template <typename E>
constexpr bool fills_pointee() {{  // an aggregate struct behind a pointer gets one value in its first element
  if constexpr (!complete<E>::value || !std::is_class_v<E>) return false;
  else if constexpr (!std::is_aggregate_v<E> || !std::is_trivially_copyable_v<E> || sizeof(E) > kBufferBytes) return false;
  else return fill_count<E>() != kNoFill;
}}

template <typename T>
T make() {{
  using U = std::remove_cv_t<T>;
  if constexpr (std::is_pointer_v<U>) {{
    using E = std::remove_cv_t<std::remove_pointer_t<U>>;
    if constexpr (std::is_function_v<E>) {{
      return nullptr;  // there is no device function to point at; a call through it faults, which settles nothing
    }} else {{
      void* p = device_zeros(kBufferBytes);
      if constexpr (fills_pointee<E>()) {{
        if (depth() < 4) {{
          ++depth();
          put(p, make<E>());
          --depth();
        }}
      }}
      return reinterpret_cast<U>(p);
    }}
  }} else if constexpr (std::is_same_v<U, bool>) {{
    return true;
  }} else if constexpr (std::is_integral_v<U>) {{
    return static_cast<U>(kInt);
  }} else if constexpr (std::is_enum_v<U>) {{
    return U{{}};
  }} else if constexpr (std::is_floating_point_v<U>) {{
    return static_cast<U>(1);
  }} else if constexpr (std::is_class_v<U> && std::is_aggregate_v<U>) {{
    constexpr size_t n = fill_count<U>();
    if constexpr (n == kNoFill) {{
      static_assert(unsupported<U>::value, "aorlta harness: a struct it cannot initialise");
    }} else {{
      return aggregate<U>(std::make_index_sequence<n>{{}});
    }}
  }} else if constexpr (std::is_constructible_v<U, float> && std::is_trivially_copyable_v<U>) {{
    return U(1.0f);
  }} else if constexpr (std::is_default_constructible_v<U>) {{
    return U{{}};
  }} else {{
    static_assert(unsupported<U>::value, "aorlta harness: a kernel parameter type it cannot supply");
  }}
}}

template <typename A>
using slot_t = std::conditional_t<std::is_reference_v<A>, std::remove_cv_t<std::remove_reference_t<A>>*,
                                  std::remove_cv_t<A>>;

template <typename A>
slot_t<A> arg() {{
  if constexpr (std::is_reference_v<A>) {{
    using E = std::remove_cv_t<std::remove_reference_t<A>>;
    void* p = device_zeros(sizeof(E) > kBufferBytes ? sizeof(E) : kBufferBytes);
    put(p, make<E>());
    return reinterpret_cast<E*>(p);
  }} else {{
    return make<std::remove_cv_t<A>>();
  }}
}}

template <typename... Args, size_t... I>
int launch(void (*kernel)(Args...), unsigned block, std::index_sequence<I...>) {{
  std::tuple<slot_t<Args>...> values{{arg<Args>()...}};
  void* argv[] = {{const_cast<void*>(static_cast<const void*>(&std::get<I>(values)))..., nullptr}};
  hipError_t e = hipLaunchKernel(reinterpret_cast<const void*>(kernel), dim3({grid}), dim3(block), argv, {lds},
                                 nullptr);
  if (e == hipSuccess) e = hipDeviceSynchronize();
  for (void* p : buffers()) (void)hipFree(p);
  buffers().clear();
  if (e != hipSuccess) {{
    std::fprintf(stderr, "aorlta harness: %s\\n", hipGetErrorString(e));
    return 1;
  }}
  return 0;
}}

template <typename... Args>
int run(void (*kernel)(Args...), unsigned block) {{
  return launch(kernel, block, std::index_sequence_for<Args...>{{}});
}}
}}  // namespace aorlta_harness
"""


def _launcher(index: int, inst: Instance) -> str:
    scope = inst.kernel.scope
    opens = "".join(f"namespace {s} {{ " if s else "namespace { " for s in scope)
    closes = "} " * len(scope)
    body = f"int aorlta_launch_{index}() {{ return ::aorlta_harness::run({inst.reference()}, {inst.block}u); }}"
    return f"// {inst.label}\n{opens}\n{body}\n{closes}\n".replace("\n\n", "\n")


def _call(index: int, inst: Instance) -> str:
    named = [s for s in inst.kernel.scope if s]
    return "::" + "::".join([*named, f"aorlta_launch_{index}"])


def harness_source(source_path: str | Path, source_text: str, instances: Sequence[Instance], launch: int = 0) -> str:
    """The harness translation unit: a launcher for each instance, and a `main` that launches `instances[launch]`."""
    if not instances or not 0 <= launch < len(instances):
        raise ValueError("a harness needs its instances and the one it launches")
    for inst in instances:
        if not IDENT.fullmatch(inst.kernel.name) or not all(s == "" or IDENT.fullmatch(s) for s in inst.kernel.scope):
            raise ValueError(f"not a kernel name: {inst.kernel.qualified!r}")
        for a in inst.template_args or ():
            if not ARG_TEXT.fullmatch(a) or "//" in a or "/*" in a:
                raise ValueError(f"not a template argument: {a!r}")
        if not (isinstance(inst.block, int) and 0 < inst.block <= 1024):
            raise ValueError(f"not a block size: {inst.block!r}")
    lds = dynamic_lds_bytes(source_text)
    include = str(Path(source_path).resolve()).replace("\\", "\\\\").replace('"', '\\"')
    out = [_PRELUDE.format(include=include, buffer=BUFFER_BYTES, int_arg=INT_ARG, fill_max=FILL_MAX, grid=GRID,
                           lds=lds)]
    out += [_launcher(i, inst) for i, inst in enumerate(instances)]
    out.append(f"int main() {{ return {_call(launch, instances[launch])}(); }}\n")
    return "\n".join(out)


def write_harness(source_path: str | Path, instances: Sequence[Instance], out_dir: str | Path, launch: int = 0,
                  name: str = HARNESS_NAME) -> Path:
    """Write the harness for `source_path` into `out_dir`; returns its path."""
    text = Path(source_path).read_text(encoding="utf-8")
    out = Path(out_dir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(harness_source(source_path, text, instances, launch), encoding="utf-8")
    return out


# --- the plan: which instance of each kernel ------------------------------------------------------------------


@dataclass(frozen=True)
class Planned:
    kernel: Kernel
    instance: Instance | None
    why: str | None  # why the kernel has no instance the harness can launch


def _compiles(run, source_path: Path, inst: Instance, hipcc: str, target: str, workdir: Path, timeout_s: float) -> bool:
    probe = write_harness(source_path, [inst], workdir, name="aorlta_probe.hip")
    rc, _ = run([hipcc, f"--offload-arch={target}", "-fsyntax-only", "-w", str(probe)], timeout_s)
    return rc == 0


def plan(source_path: str | Path, *, hipcc: str = "/opt/rocm/bin/hipcc", target: str = "gfx950",
         workdir: str | Path, timeout_s: float = 120.0, run=execute) -> list[Planned]:
    """One instance for each kernel the source defines. A template kernel takes the first of `template_attempts` whose
    harness compiles (host and device, `-fsyntax-only`); one with none has no instance, and says why. Raises
    HarnessUnsupported when the source has no kernel the harness can see (`find_kernels`)."""
    source = Path(source_path)
    text = source.read_text(encoding="utf-8", errors="replace")
    block = launch_block(text)
    out, labels = [], set()
    for k in find_kernels(source, hipcc=hipcc, target=target, run=run):
        if k.specialization is not None or k.template is None:
            inst = Instance(k, k.specialization, block)
            if inst.label not in labels:
                labels.add(inst.label)
                out.append(Planned(k, inst, None))
            continue
        try:
            attempts = template_attempts(k, block)
        except HarnessUnsupported as exc:
            out.append(Planned(k, None, str(exc)))
            continue
        chosen = next((inst for inst in (Instance(k, tuple(a), block) for a in attempts)
                       if _compiles(run, source, inst, hipcc, target, Path(workdir), timeout_s)), None)
        if chosen is not None and chosen.label in labels:
            continue  # an explicit specialization of the same arguments is already planned
        if chosen is not None:
            labels.add(chosen.label)
        out.append(Planned(k, chosen, None if chosen else "no instantiation compiles"))
    return out
