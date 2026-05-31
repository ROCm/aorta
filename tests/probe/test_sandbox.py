"""Tests for :mod:`aorta.probe.sandbox` — the ``condition:`` evaluator.

Covers Phase 2 FR 2.7 (hostile-input corpus rejection at recipe load)
and FR 2.12 (parse-time enforcement — a hostile input never reaches
``eval``).
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from aorta.probe.sandbox import (
    MAX_EXPR_LEN,
    MAX_INT_CONSTANT,
    SandboxError,
    evaluate,
    validate_and_compile,
)

FIXTURES = Path(__file__).parent / "fixtures" / "conditions"


def _hostile_lines() -> list[str]:
    """Read the hostile-input corpus, returning one expression per line."""
    text = (FIXTURES / "hostile.txt").read_text(encoding="utf-8")
    return [line.rstrip("\n") for line in text.splitlines() if line.strip()]


@pytest.mark.parametrize("expr", _hostile_lines())
def test_hostile_inputs_rejected(expr: str) -> None:
    """Every entry in ``hostile.txt`` MUST be rejected by the sandbox."""
    with pytest.raises(SandboxError):
        validate_and_compile(expr)


def test_no_eval_reach_for_rejected_input() -> None:
    """FR 2.12: a hostile expression never reaches ``builtins.eval``.

    Patches Python's ``eval`` and asserts the patched callable is
    never invoked when ``validate_and_compile`` rejects the input.
    The patch covers the *module-global* ``eval`` symbol; the
    sandbox's own ``compile``/``eval`` would route through it if
    enforcement leaked past parse time.
    """
    with patch("aorta.probe.sandbox.eval") as eval_mock:
        for expr in _hostile_lines():
            with pytest.raises(SandboxError):
                validate_and_compile(expr)
    assert eval_mock.call_count == 0


def test_length_cap_rejects_at_parse_time() -> None:
    """A 257+ char expression rejects before ``ast.parse`` can run."""
    long_expr = "1" + (" + 1" * 70)
    assert len(long_expr) > MAX_EXPR_LEN
    with pytest.raises(SandboxError, match="exceeds"):
        validate_and_compile(long_expr)


def test_empty_expression_rejected() -> None:
    with pytest.raises(SandboxError, match="non-empty"):
        validate_and_compile("")
    with pytest.raises(SandboxError, match="non-empty"):
        validate_and_compile("   ")


def test_non_string_input_rejected() -> None:
    with pytest.raises(SandboxError, match="must be a string"):
        validate_and_compile(42)  # type: ignore[arg-type]


def test_syntax_error_rejected() -> None:
    with pytest.raises(SandboxError, match="syntax error"):
        validate_and_compile("capture[")


# ---- Happy path -----------------------------------------------------------


def test_allows_named_capture_lookup() -> None:
    code = validate_and_compile("capture['loss'] == 'nan'")
    assert evaluate(
        code,
        capture={"loss": "nan"},
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert not evaluate(
        code,
        capture={"loss": "fine"},
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )


def test_allows_float_int_len() -> None:
    code = validate_and_compile("int(capture['count']) > 10 and len(capture) > 0")
    assert evaluate(
        code,
        capture={"count": "42", "other": "v"},
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )


def test_allows_math_isnan_isinf() -> None:
    code = validate_and_compile("math.isnan(float(capture['loss']))")
    assert evaluate(
        code,
        capture={"loss": "nan"},
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert not evaluate(
        code,
        capture={"loss": "1.0"},
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )


def test_allows_exit_code_walltime_peak_vram() -> None:
    code = validate_and_compile("exit_code != 0 and walltime_sec > 1.0 and peak_vram_mib > 100")
    assert evaluate(
        code,
        capture={},
        exit_code=137,
        walltime_sec=12.5,
        peak_vram_mib=512,
    )


def test_peak_vram_none_binds_to_zero() -> None:
    """FR §2.E.1: peak_vram_mib bound to 0 when actual is None.

    The expression compares ``peak_vram_mib > 0`` — must not blow up
    on None and must evaluate to False (because the bound value is 0).
    """
    code = validate_and_compile("peak_vram_mib > 0")
    assert not evaluate(
        code,
        capture={},
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )


def test_ifexp_and_boolop_allowed() -> None:
    code = validate_and_compile("(1 if exit_code != 0 else 0) or (walltime_sec > 0)")
    assert evaluate(
        code,
        capture={},
        exit_code=1,
        walltime_sec=0.0,
        peak_vram_mib=None,
    )


def test_int_constant_below_cap_allowed() -> None:
    code = validate_and_compile(f"exit_code == {MAX_INT_CONSTANT - 1}")
    assert not evaluate(
        code,
        capture={},
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )


def test_int_constant_at_cap_rejected() -> None:
    # ``**`` is no longer in the allow-list -- exercise the magnitude
    # cap via a comparison literal instead, which is the realistic
    # shape an operator would write.
    with pytest.raises(SandboxError, match="integer constant magnitude"):
        validate_and_compile(f"exit_code == {MAX_INT_CONSTANT}")


def test_pow_operator_rejected() -> None:
    """Regression for PR #197 review: dropping ``ast.Pow`` from the
    allow-list closes the ``2 ** int(capture['exp'])`` resource-
    exhaustion path -- the magnitude cap only restricts literal
    constants, so an exponent derived from regex-capture text would
    otherwise let an attacker allocate an enormous Python int at
    eval time.
    """
    for hostile in ("2 ** 31", "10 ** int(capture['exp'])", "2 ** exit_code"):
        with pytest.raises(SandboxError, match="forbidden AST node"):
            validate_and_compile(hostile)


def test_mult_operator_rejected_for_string_repeat() -> None:
    """Regression for PR #197 round-3 review: dropping ``ast.Mult``
    from the allow-list closes the string-repetition path
    (``'a' * 999999998`` allocates a ~1 GiB string without ever
    crossing :data:`MAX_INT_CONSTANT`). The walker can't tell at
    parse time whether ``*`` is a numeric multiply or a string /
    tuple repeat, so the simplest correct defence is to forbid the
    operator. Same reasoning as Pow.
    """
    for hostile in (
        "'a' * 999999998",
        "capture['x'] * 999999998",
        "exit_code * 2",
        "(1,) * 999999998",
    ):
        with pytest.raises(SandboxError, match="forbidden AST node"):
            validate_and_compile(hostile)


def test_mod_operator_rejected_for_printf_format() -> None:
    """Regression for PR #197 round-3 review: dropping ``ast.Mod``
    closes the printf-style string-formatting path
    (``'%999999998s' % capture['x']`` allocates the same billion-
    byte buffer as string-repeat). The walker can't discriminate
    numeric modulo from string formatting at parse time, so the
    operator goes entirely.
    """
    for hostile in (
        "'%999999998s' % capture['x']",
        "exit_code % 2",
        "'%s' % capture['code']",
    ):
        with pytest.raises(SandboxError, match="forbidden AST node"):
            validate_and_compile(hostile)


def test_subscript_index_must_be_string_literal() -> None:
    """Regression for PR #197 review (Sonbol): the Subscript rule
    only checks that the *base* is ``capture`` -- without an index
    check, ``capture[exit_code]`` (Name), ``capture[0]`` (numeric
    Constant), and ``capture['x':]`` (slice) all parse cleanly and
    only blow up at eval time. The Tier 5 runner then swallows
    those eval-time exceptions silently with
    ``except Exception: fired = False``, so a typoed recipe ships
    to prod and the detector never fires. Closing the check at
    parse time means recipe authors get a ``SandboxError`` at
    load time naming the bad index.
    """
    for hostile, expected_marker in (
        ("capture[exit_code]", "exit_code"),
        ("capture[0]", "0"),
        ("capture['x':]", "'x'"),
    ):
        with pytest.raises(SandboxError, match="must be a string literal"):
            validate_and_compile(hostile)
        # Sanity: the error message names the bad index so the
        # operator can fix it without reading the sandbox source.
        with pytest.raises(SandboxError) as exc_info:
            validate_and_compile(hostile)
        assert expected_marker in str(exc_info.value)


def test_int_from_string_digit_cap_active() -> None:
    """Regression for PR #197 review (Sonbol): Python 3.10 has no
    default cap on ``int()``-from-string parsing, so a hostile
    regex capture returning a multi-megabyte string would let
    ``int(capture['x'])`` burn CPU for minutes via the O(n^2)
    digit parser. The sandbox module sets
    ``sys.set_int_max_str_digits(4300)`` at import time to apply
    the CPython 3.11+ default cap explicitly on 3.10.

    Verified by attempting to parse a 5000-char digit string at
    eval time -- the cap raises ``ValueError`` synchronously
    rather than hanging.
    """
    code = validate_and_compile("int(capture['x']) > 0")
    with pytest.raises(ValueError, match="(?i)exceeds the limit"):
        evaluate(
            code,
            capture={"x": "0" * 5000},
            exit_code=0,
            walltime_sec=1.0,
            peak_vram_mib=None,
        )


def test_empty_builtins_neutralises_eval() -> None:
    """Defence-in-depth: even if ``eval`` were reached, ``__builtins__={}``
    means ``__import__`` is unreachable.

    Constructed by manually compiling an expression that LOOKS like an
    import and asserting that running it via the sandbox's
    ``evaluate`` fails with NameError (because ``__import__`` is not
    in the empty builtins). Demonstrates the belt-and-suspenders
    layer even if the AST walker were to regress.
    """
    code = compile("__import__('os')", "<test>", "eval")
    with pytest.raises(NameError):
        evaluate(
            code,
            capture={},
            exit_code=0,
            walltime_sec=1.0,
            peak_vram_mib=None,
        )


def test_math_module_is_only_globals_module() -> None:
    """``os`` is not in the eval globals — name lookup fails."""
    code = compile("os.getenv('PATH')", "<test>", "eval")
    with pytest.raises(NameError):
        evaluate(
            code,
            capture={},
            exit_code=0,
            walltime_sec=1.0,
            peak_vram_mib=None,
        )


def test_math_is_only_safe_attrs() -> None:
    """Sanity: ``math.isnan`` works post-compile; ``math.pi`` is not
    in the whitelist so ``validate_and_compile`` rejects it.
    """
    with pytest.raises(SandboxError, match="forbidden attribute"):
        validate_and_compile("math.pi")
    # And isnan/isinf are unambiguously allowed.
    code = validate_and_compile("math.isinf(float(capture['x']))")
    assert evaluate(
        code,
        capture={"x": "inf"},
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    # NB: math.inf is also accessible at eval because ``math`` is a
    # real module reference in globals; the whitelist runs at parse,
    # so the only way to NAME math.<attr> in a condition is through
    # the AST walker, which rejects everything but isnan/isinf.
    _ = math.inf  # silence "unused import" lints


__all__: list[str] = []
