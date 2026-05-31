# `condition:` Sandbox — Whitelist + Worked Examples (Issue #188, Phase 2)

The `condition:` field on a `custom_patterns[*].match` block lets a
probe-mode recipe author write a boolean expression evaluated **after**
the regex matches a per-trial log line. It is the only place where
user-supplied YAML drives arbitrary computation against trial-time
data, so the security posture is **default-deny at parse time**.

Source: `src/aorta/probe/sandbox.py`. Tests:
`tests/probe/test_sandbox.py`. Hostile-input corpus:
`tests/probe/fixtures/conditions/hostile.txt`.

---

## What Goes Through the Sandbox

For every `custom_patterns[*]` entry with a `match.condition`:

1. At **recipe load**, `validate_and_compile(condition_text)`:
   - Strips the expression.
   - Rejects empty / non-string input.
   - Rejects expressions longer than **256 characters** (`MAX_EXPR_LEN`).
   - `ast.parse(text, mode="eval")` (any parse error → `SandboxError`).
   - Walks the AST and rejects every node, name, attribute, subscript,
     and call target outside the whitelist (see below).
   - Rejects integer literals whose magnitude is `>= 10^9`
     (`MAX_INT_CONSTANT`) — prevents bare resource-exhaustion
     literals like `exit_code == 1000000000` from ever reaching
     `eval`. Three resource-exhaustion operators (`**`, `*`, `%`)
     are also rejected one layer up because they are not in the
     AST allow-list, closing the variants that operate on
     non-literal operands: `2 ** capture['n']` (huge int),
     `'a' * 999999998` (huge string), `'%999999998s' % capture['x']`
     (huge printf buffer). The magnitude cap and the operator
     allow-list together cover both the literal and the
     name-derived attack shapes.
   - Returns a compiled `CodeType` cached on the `CompiledPattern`.
2. At **trial post-exit**, if the pattern's regex matched, the runner
   calls `evaluate(code, capture=..., exit_code=..., walltime_sec=...,
   peak_vram_mib=...)`:
   - Globals are `{"__builtins__": {}, "math": math, "float": float,
     "int": int, "len": len}`.
   - Locals are the four documented variables (see "Variables in
     scope" below).
   - Result is coerced to `bool`. Any runtime exception is treated as
     "did not fire" (the trial verdict is not aborted by a hostile
     condition that survived the parse-time walker — defence in
     depth).

Rejections are reported with the offending AST node type and source
line so the recipe author can find the bad expression without
counting yaml indents. The exception is `SandboxError`, which
subclasses `RecipeSchemaError` so the existing CLI / recipe-loader
error path catches it without code changes.

---

## Variables in Scope (Read-Only)

| Name | Type | Source |
|---|---|---|
| `capture` | `dict[str, str]` | Named regex captures from the matched pattern. Indexed only as `capture['name']`. |
| `exit_code` | `int` | The trial's process exit code (negative when signalled, e.g. `-11` for SIGSEGV). |
| `walltime_sec` | `float` | Wall-clock seconds from `Popen.start()` to `wait()` return. |
| `peak_vram_mib` | `int` (0 if unavailable) | Per-trial peak VRAM from `amd-smi`. Bound to **0** when the actual value is None so `peak_vram_mib > 100` does not blow up mid-eval. |

## Allowed AST Nodes

```python
ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
ast.Call, ast.Subscript, ast.Name, ast.Constant, ast.IfExp,
ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
ast.Gt, ast.GtE, ast.Add, ast.Sub, ast.Div,
ast.FloorDiv, ast.USub, ast.UAdd, ast.Load,
```

Plus `ast.Attribute` ONLY when it spells `math.isnan` or `math.isinf`.

`ast.Pow` (`**`), `ast.Mult` (`*`), and `ast.Mod` (`%`) are
deliberately omitted. The integer-literal magnitude cap only
restricts *literal* constants, and Python's `*` / `%` / `**`
operators all overload on non-numeric operands (strings repeat,
tuples repeat, printf-formatting allocates), so the walker can't
distinguish "numeric arithmetic" from "resource-exhaustion bomb"
at parse time without knowing the operand types. The simplest
correct defence is to forbid all three operators. Legitimate
`condition:` expressions are boolean checks (`exit_code == 137`,
`walltime_sec > 60`, `peak_vram_mib > 100`) and never need
exponentiation, multiplication, or modulo.

## Allowed Names

```python
{"capture", "exit_code", "walltime_sec", "peak_vram_mib",
 "math", "float", "int", "len"}
```

## Allowed Calls

`float`, `int`, `len`, `math.isnan`, `math.isinf`. Anything else — including
`getattr`, `hasattr`, `type`, `isinstance`, `capture.update(...)`,
`open(...)`, `__import__(...)` — is rejected at parse time.

`capture[...]` subscript is permitted; `foo[...]` for any other name
is rejected.

---

## Worked Examples — Allowed

```yaml
# Trial failed when loss exploded.
condition: "math.isnan(float(capture['loss']))"

# Slow iteration only counts on a real GPU host.
condition: "walltime_sec > 60 and peak_vram_mib > 0"

# Exit code 137 (OOM-kill) plus a non-empty captured message
# (we can't substring-search the message today; ``ast.In`` is
# rejected by the walker, so use a presence check instead).
condition: "exit_code == 137 and len(capture['msg']) > 0"

# Specific value check (Tier-3 OOM-killer signature).
condition: "int(capture['code']) == 137"
```

> **`in` is currently rejected.** `ast.Compare` with the `ast.In`
> operator is **not** in the allow-list today, so a condition like
> `'oom' in capture['msg']` raises `SandboxError` at recipe-load
> time. If you need substring matching, do it in the
> `custom_patterns[*].match.regex` field (where it belongs) and
> use `condition:` for the boolean glue. Membership-test support
> is tracked as a planned future addition; until then the
> "Rejected" section below pins the current behaviour.

## Worked Examples — Rejected

Every entry below is in `tests/probe/fixtures/conditions/hostile.txt`
and is parametrised in `tests/probe/test_sandbox.py::test_hostile_inputs_rejected`:

```python
__import__('os').system('rm -rf /')
(0).__class__.__bases__[0].__subclasses__()
().__class__.__mro__[-1]
open('/etc/passwd').read()
exec("import os; os.system('id')")
eval("1+1")
lambda x: x
[x for x in range(10)]
{k: v for k, v in capture.items()}
capture.update({'pwn': '1'})
type(capture)
getattr(capture, 'pop')('eval_loss')
capture['x'].__class__
math.__loader__.load_module('os')
2 ** 1000000000  # rejected because ast.Pow is not in the allow-list
'a' * 999999998  # rejected because ast.Mult is not in the allow-list (string repeat -> ~1 GiB alloc)
'%999999998s' % capture['x']  # rejected because ast.Mod is not in the allow-list (printf-style formatting -> same billion-byte buffer)
'oom' in capture['msg']  # rejected because ast.In is not in the allow-list (no membership tests today; use the regex for substring matching)
```

Each line covers a distinct exploit class:

- Import / system-call gadgets (`__import__`, `exec`, `eval`).
- Object-graph walks (`__class__`, `__bases__`, `__subclasses__`,
  `__mro__`).
- Attribute-walking on whitelisted names (`capture.update`,
  `getattr(capture, 'pop')`, `math.__loader__`).
- Filesystem access (`open(...)`).
- Function literals (`lambda`).
- Comprehensions (`ListComp`, `DictComp`).
- Forbidden builtins (`type`, `getattr`).
- Resource-exhaustion exponentiation (`**`) — rejected at the AST
  level rather than relying on the integer-literal magnitude cap,
  because the exponent can be derived from non-literal expressions
  like `int(capture['exp'])` that the magnitude cap does not see.
- Resource-exhaustion integer literals
  (`exit_code == 1000000000`-style) — rejected by
  `MAX_INT_CONSTANT = 10**9`.

---

## Regex DoS Hardening (§2.E.5)

The sandbox doc covers `condition:` validation; the recipe loader
separately compile-validates every `custom_patterns[*].match.regex`
via `re.compile(regex)` at load (FR 2.6) so a bad pattern surfaces
immediately.

To bound catastrophic backtracking on operator-supplied regex at
runtime — without swapping the regex engine — every per-trial
regex scan runs against a **10 MiB-capped window**
(`MAX_LOG_BYTES = 10 * 1024 * 1024`). Logs longer than the cap are
scanned in successive windows that overlap by **4 KiB**
(`_WINDOW_OVERLAP_BYTES`, capped at half the window so each step
still advances) so a multi-line match straddling a seam — a
Python traceback header at the end of one window with its body at
the start of the next — still fires. The same rule applies to the
Tier 4 built-in scanner.

---

## Defence-in-Depth Layers

The sandbox is intentionally layered so a single regression cannot
unlock arbitrary code execution:

1. **Parse-time AST whitelist** (`validate_and_compile`). The
   walker is ~80 lines and entirely auditable.
2. **Empty `__builtins__`** at eval. Even if the walker were to
   regress, `__import__` is unreachable.
3. **No module references in scope** besides `math`. Attribute
   access on `math` is parse-time-restricted to `isnan` / `isinf`.
4. **Length cap before parse**, integer-magnitude cap during walk.
5. **Runtime exception → did-not-fire**, not classifier abort.
   Hostile conditions that survive (they shouldn't) cannot poison
   the verdict for an entire run.

The hostile-input corpus
(`tests/probe/fixtures/conditions/hostile.txt`) is the regression
suite for layer #1; the rest of the layers have their own dedicated
tests (`test_empty_builtins_neutralises_eval`,
`test_math_module_is_only_globals_module`,
`test_int_constant_at_cap_rejected`).

---

## Why Not RestrictedPython?

A third-party Python sandbox library (`RestrictedPython`,
`pysandbox`, etc.) was considered and rejected:

- **Heavier dependency surface**. Each library is itself a security
  artifact requiring its own review cadence; rolling our own
  ~80-line walker is auditable in a single review pass.
- **More permissive default whitelist**. `RestrictedPython` allows
  loops, list comprehensions, and explicit re-exports that this
  use case has no need for.
- **Phase-2 do-not-do list (§2.H)** explicitly forbids third-party
  sandbox libraries.

If a future need arises that genuinely requires a larger surface
(generators, list comprehensions, multi-argument calls), the right
move is to revisit the rubric and the security-reviewer sign-off
together — not to silently broaden this walker.

---

## Security-Reviewer Sign-off

Per Open Question #1 in the rubric, Phase 2 PR merge is **blocked**
on a security-reviewer's approval of this sandbox. The reviewer
should read:

1. This document.
2. `src/aorta/probe/sandbox.py` end-to-end.
3. `tests/probe/test_sandbox.py` end-to-end.
4. `tests/probe/fixtures/conditions/hostile.txt` (parametrised
   rejection corpus).

The same reviewer should also approve Phase 3's redaction module
when it lands.
