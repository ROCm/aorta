"""``condition:`` expression sandbox for ``custom_patterns`` (issue #188 Phase 2).

The ``condition`` field on a ``custom_patterns[*].match`` block lets a
recipe author write a small boolean expression that is evaluated AFTER
the regex matches a per-trial log line. The expression has access to a
fixed set of variables (the named regex captures, the trial's exit
code, walltime, and peak VRAM) and a fixed set of callables (``float``,
``int``, ``len``, ``math.isnan``, ``math.isinf``). It is the only place
where user-supplied YAML can drive arbitrary computation against
trial-time data, so the security posture is **default-deny at
parse time**:

1. ``validate_and_compile`` walks the parsed AST and rejects anything
   that is not in the explicit allow-list (``_ALLOWED_NODES``,
   ``_ALLOWED_NAMES``, ``_ALLOWED_CALLS``).
2. ``eval`` is invoked with ``__builtins__={}`` so even if a hostile
   expression slipped past the whitelist it cannot reach the import
   machinery.
3. Expressions longer than :data:`MAX_EXPR_LEN` characters (after
   stripping) are rejected before ``ast.parse`` to prevent resource
   exhaustion at compile time.

The whitelist is intentionally small (~80 lines of walker code). A
third-party sandbox library (e.g. ``RestrictedPython``) was considered
and rejected: heavyweight dep, more permissive defaults than the issue
demands, and an auditable hand-rolled walker is the response a security
reviewer can sign off on by reading.

See :mod:`docs/probe-188/sandbox.md` for the worked-example corpus and
``tests/probe/fixtures/conditions/hostile.txt`` for the parameterised
rejection suite (`tests/probe/test_sandbox.py::test_hostile_inputs_rejected`).
"""

from __future__ import annotations

import ast
import math
import sys
from types import CodeType
from typing import Any

from aorta.triage.recipe import RecipeSchemaError

# CPython 3.11+ caps ``int()``-from-string parsing at 4300 digits by
# default (`PEP 651 / GH-95778`) to defend against the O(n^2) digit
# parser. 3.10 does not. The sandbox allows ``int()`` calls (it's a
# legitimate move on a numeric regex capture), and ``capture[...]``
# can return a multi-megabyte string when a hostile pattern matches
# a huge log window — so ``int(capture['x'])`` would burn the CPU
# for minutes on 3.10. Apply the same 4300-digit cap explicitly at
# import time so the sandbox has matching behavior across the
# project's supported Python versions (pyproject ``requires-python
# = ">=3.10"``). No-op on 3.11+ where the same cap is already the
# default. Per Sonbol's PR #197 review.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(4300)

# Hard cap on per-trial log bytes scanned for regex matches. Hardens
# against catastrophic backtracking on operator-supplied regex without
# requiring a different regex engine (Phase 2 explicitly forbids the
# ``regex`` / ``re2`` swap). Per-scan window; logs larger than this
# are scanned in successive windows that overlap by a small
# ``_WINDOW_OVERLAP_BYTES`` slice (defined in
# :mod:`aorta.probe.classifier.tier4_patterns` /
# :mod:`aorta.probe.classifier.tier5_custom`) so a match straddling
# the seam still fires.
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MiB

# Per-expression character cap (post-strip). Prevents an attacker from
# burning ``ast.parse`` time on a multi-MB expression. 256 chars is
# enough for every legitimate ``condition`` (a half-dozen comparisons
# at most) and well under any pathological-backtracking input the
# parser could be coerced into.
MAX_EXPR_LEN = 256

# Magnitude cap on integer literals. Catches OOM-bait constants like
# ``exit_code == 1000000000`` from ever reaching ``eval`` -- a literal
# that survives parse-time validation can still be evaluated, and there
# is no runtime numeric guard once the expression is compiled. ``10**9``
# is comfortably above every legitimate value a ``condition`` would
# name (exit codes, walltime in seconds, VRAM in MiB, log-line counts)
# and orders of magnitude below the size that turns ``int.__mul__``
# into a memory bomb.
#
# The resource-exhaustion operators (``**``, ``*``, ``%``) are also
# NOT in the allow-list (see :data:`_ALLOWED_NODES`), so the
# magnitude cap is not the last line of defence against
# ``2 ** capture['n']``, ``'a' * 999999998``, or
# ``'%999999998s' % capture['x']``-style attacks: those expressions
# reject at parse time before either operand is considered.
MAX_INT_CONSTANT = 10**9

# AST node types that the walker accepts. Anything else (Lambda,
# ListComp, DictComp, SetComp, GeneratorExp, FormattedValue,
# JoinedStr, ClassDef, FunctionDef, Import, ...) rejects.
#
# ``ast.Pow``, ``ast.Mult``, and ``ast.Mod`` are deliberately EXCLUDED:
#
# * ``Pow`` -- :data:`MAX_INT_CONSTANT` only restricts literal
#   constants. ``2 ** int(capture['exp'])`` would still allocate a
#   giant Python int from untrusted regex-capture text. Dropping
#   ``**`` closes that path at parse time before either operand is
#   considered.
# * ``Mult`` -- Python's ``*`` operator overloads on strings and
#   tuples (``'a' * 999999998`` allocates a ~1 GiB string). The
#   integer-literal cap is at 10**9 (just below the
#   string-repetition pressure point), but a literal like
#   ``999999998`` slips under the cap and detonates at eval time;
#   the walker can't distinguish "numeric multiply" from "string
#   repeat" because operand types are only known at runtime. The
#   simplest correct defence is to forbid ``*`` entirely.
# * ``Mod`` -- Python's ``%`` overloads on strings as printf-style
#   formatting (``'%999999998s' % capture['x']`` allocates the same
#   billion-byte buffer). Same reasoning: no walk-time type
#   discrimination, drop the operator.
#
# Legitimate ``condition`` expressions are simple boolean checks
# (``exit_code == 137``, ``walltime_sec > 60``, ``peak_vram_mib > 100``);
# none of them need ``**``, ``*``, or ``%``, so dropping all three
# is a free safety win.
_ALLOWED_NODES = frozenset(
    {
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.Subscript,
        ast.Name,
        ast.Constant,
        ast.IfExp,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Add,
        ast.Sub,
        ast.Div,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.Load,
    }
)

# The four trial-time variables plus the ``math`` module reference
# plus the names of every callable in :data:`_ALLOWED_CALLS` (bare
# ``Name`` lookups for ``float`` / ``int`` / ``len`` arrive at the
# walker before the ``Call`` check fires; allowing them here lets the
# walker pass them so the per-Call whitelist still gets to decide).
# Any other ``Name`` lookup (``os``, ``__import__``, ``True`` is fine
# because it parses as ``Constant``, ``capture.items`` is fine because
# that's an attribute walk -- rejected separately).
_ALLOWED_NAMES = frozenset(
    {
        "capture",
        "exit_code",
        "walltime_sec",
        "peak_vram_mib",
        "math",
        "float",
        "int",
        "len",
    }
)

# Callables a condition may invoke. ``ast.unparse`` of the call's
# ``func`` must be exactly one of these strings. ``getattr``,
# ``hasattr``, ``type``, ``isinstance``, ``len`` on a non-capture
# value, etc., all fail this check.
_ALLOWED_CALLS = frozenset({"float", "int", "len", "math.isnan", "math.isinf"})


class SandboxError(RecipeSchemaError):
    """Raised when a ``condition`` expression fails sandbox validation.

    Subclasses :class:`RecipeSchemaError` so the existing recipe-loader
    error path (CLI handler catches ``RecipeSchemaError``, surfaces as
    a ``ClickException``) catches it without code changes. The
    distinct subclass lets tests / callers assert "this rejection
    came from the sandbox" specifically.
    """


def validate_and_compile(expr: str) -> CodeType:
    """Parse, whitelist-validate, and compile a ``condition`` expression.

    Returns a :class:`CodeType` object ready for
    ``eval(code, globals, locals)`` with ``__builtins__={}`` and
    ``math`` in ``globals`` (see :func:`evaluate`).

    Raises :class:`SandboxError` if any of the following hold:

    * The expression is empty or longer than :data:`MAX_EXPR_LEN`
      characters after ``str.strip()``.
    * ``ast.parse(expr, mode="eval")`` raises ``SyntaxError``.
    * The AST contains a node type not in :data:`_ALLOWED_NODES`,
      an :class:`ast.Attribute` access that isn't ``math.isnan`` or
      ``math.isinf``, a :class:`ast.Name` lookup outside
      :data:`_ALLOWED_NAMES`, a :class:`ast.Subscript` against
      anything other than ``capture[...]``, or a :class:`ast.Call`
      whose target isn't in :data:`_ALLOWED_CALLS`.

    The check is at PARSE TIME — a hostile expression that walks the
    object graph or imports something can never reach :func:`evaluate`.
    """
    if not isinstance(expr, str):
        raise SandboxError(f"condition: must be a string, got {type(expr).__name__}")
    stripped = expr.strip()
    if not stripped:
        raise SandboxError("condition: expression must be non-empty")
    if len(stripped) > MAX_EXPR_LEN:
        raise SandboxError(
            f"condition: expression length {len(stripped)} exceeds the "
            f"{MAX_EXPR_LEN}-character cap (rejected before ast.parse to "
            "bound parse-time work)"
        )

    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError as exc:
        raise SandboxError(
            f"condition: syntax error: {exc.msg} (line {exc.lineno}, col {exc.offset})"
        ) from exc

    for node in ast.walk(tree):
        _check_node(node)

    return compile(tree, "<condition>", "eval")


def _check_node(node: ast.AST) -> None:
    """Whitelist-check a single AST node.

    Split out so the per-node logic stays small enough to audit. The
    rules deliberately mirror the rubric §2.E.3 sketch with one
    addition: a ``Compare`` node's comparator chain is implicitly
    valid because each comparator is itself walked by ``ast.walk``,
    which means the per-node check runs against each one too. The
    same applies to ``BoolOp`` operands and ``BinOp`` left/right.
    """
    if isinstance(node, ast.Attribute):
        # The only attribute access we allow is ``math.<isnan|isinf>``.
        # Everything else (``capture.update``, ``(0).__class__``,
        # ``math.__loader__``) rejects.
        value = node.value
        if not (
            isinstance(value, ast.Name) and value.id == "math" and node.attr in ("isnan", "isinf")
        ):
            raise SandboxError(f"condition: forbidden attribute access: {ast.unparse(node)}")
        return

    if type(node) not in _ALLOWED_NODES:
        raise SandboxError(f"condition: forbidden AST node: {type(node).__name__}")

    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise SandboxError(f"condition: forbidden name: {node.id}")
        return

    if isinstance(node, ast.Subscript):
        # ``capture[...]`` only. ``foo[0]`` rejects because the
        # subscripted name must be ``capture``.
        if not (isinstance(node.value, ast.Name) and node.value.id == "capture"):
            raise SandboxError("condition: subscript only allowed on capture[...]")
        # The slice must be a *string literal* — anything else (a
        # ``Name`` like ``capture[exit_code]``, a slice like
        # ``capture['x':]``, a tuple/star/call) gets through the
        # whitelist of nested node types because ``ast.walk``
        # validates them in isolation, but lands as a runtime
        # ``KeyError`` / ``TypeError`` on eval. The Tier-5 runner
        # then swallows that with ``except Exception: fired =
        # False`` (tier5_custom.py post-eval), so a typoed recipe
        # ships to prod and the detector silently never fires.
        # Closing this at parse time means the sandbox contract
        # ("you can only look up named string keys in capture") is
        # actually enforced, and the recipe author gets a
        # ``SandboxError`` at recipe-load time naming the bad index
        # rather than a silent no-fire. Per Sonbol's PR #197
        # review.
        if not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            raise SandboxError(
                f"condition: capture[...] index must be a string literal, "
                f"got {ast.unparse(node.slice)!r}"
            )
        return

    if isinstance(node, ast.Call):
        # The full dotted path for the call's target. ``ast.unparse``
        # handles attribute chains (`math.isnan`) and bare names
        # (`float`) uniformly. The whitelist is exact-match.
        try:
            target = ast.unparse(node.func)
        except (AttributeError, ValueError) as exc:
            raise SandboxError(
                f"condition: forbidden call target: {type(node.func).__name__}"
            ) from exc
        if target not in _ALLOWED_CALLS:
            raise SandboxError(f"condition: forbidden call: {target}")
        return

    if isinstance(node, ast.Constant):
        # ``int`` and ``float`` constants are subject to a magnitude
        # cap so a literal like ``exit_code == 1000000000`` (and
        # similar resource-exhaustion inputs in the hostile-input
        # corpus, §2.E.4) refuses at parse time before ``eval`` can
        # construct a large arbitrary-precision integer at compile
        # time. ``str``/``bool``/``None`` constants pass through.
        # ``**`` is no longer in the allow-list (see
        # :data:`_ALLOWED_NODES`), so Pow-based ints from non-literal
        # exponents are stopped one layer up.
        constant_value = node.value
        if isinstance(constant_value, bool):
            return
        if isinstance(constant_value, int) and abs(constant_value) >= MAX_INT_CONSTANT:
            raise SandboxError(
                f"condition: integer constant magnitude {abs(constant_value)} >= "
                f"{MAX_INT_CONSTANT} (rejected to prevent resource exhaustion)"
            )
        return


def evaluate(
    code: CodeType,
    *,
    capture: dict[str, str],
    exit_code: int,
    walltime_sec: float,
    peak_vram_mib: int | None,
) -> bool:
    """Run a sandbox-compiled ``condition`` against trial-time data.

    ``peak_vram_mib`` is bound to ``0`` when the actual value is
    ``None`` so a condition that references it (``peak_vram_mib >
    100``) does not blow up with a ``TypeError`` mid-eval. Documented
    in rubric §2.E.1 — ``int | None`` at the source, ``int`` at
    eval time.

    ``__builtins__`` is set to ``{}`` so even a hostile expression
    that slipped past :func:`validate_and_compile` (it shouldn't —
    every callable goes through the whitelist) cannot reach
    ``__import__``. ``math`` is the only module in scope; the
    whitelist already restricts attribute access to ``math.isnan`` /
    ``math.isinf`` so the broader module surface is unreachable.

    Returns the result coerced to ``bool``. A non-boolean expression
    (``capture['x']`` returning a string) follows Python's usual
    truthiness rules — the recipe author can be more strict by
    wrapping with ``len(...)`` or comparing explicitly.
    """
    if peak_vram_mib is None:
        peak_vram_mib = 0
    # ``__builtins__={}`` neutralises ``__import__``; the whitelisted
    # callables (``float``, ``int``, ``len``) are explicitly seeded
    # into ``globals`` so they resolve without re-enabling the
    # builtins namespace. ``math`` is the only module reference in
    # scope; attribute access is parse-time-restricted to
    # ``math.isnan`` / ``math.isinf`` (see :func:`validate_and_compile`).
    globals_dict: dict[str, Any] = {
        "__builtins__": {},
        "math": math,
        "float": float,
        "int": int,
        "len": len,
    }
    locals_dict: dict[str, Any] = {
        "capture": capture,
        "exit_code": exit_code,
        "walltime_sec": walltime_sec,
        "peak_vram_mib": peak_vram_mib,
    }
    return bool(eval(code, globals_dict, locals_dict))  # noqa: S307 -- sandboxed


__all__ = [
    "MAX_EXPR_LEN",
    "MAX_LOG_BYTES",
    "SandboxError",
    "evaluate",
    "validate_and_compile",
]
