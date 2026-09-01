"""Guard the mechanism used to count kernels in an AMDGPU code object.

``llvm-readelf --symbols`` prints ``.dynsym`` *and* ``.symtab``, and a kernel is
listed in both, so counting matches across that output reports exactly twice the
kernel count. That is not a hypothetical: the figures recorded for the heavy f32
Tensile object (490 / 1554 / 682 kernels) were all 2x for this reason, and the
490 was quoted in five places before it was caught. ``--dyn-syms`` reads the one
table and is the correct source.

The flag has spellings. ``-s`` and ``--syms`` are aliases of ``--symbols``, and
``-a`` / ``--all`` expands to ``-h -l -S -s -r -d -V -A -I`` -- which includes
``-s``, so it double-counts too. All five are rejected; matching only the long
form would let the same bug back in under a shorter name.

The check is phrased as the property rather than as a diff against one known-bad
line, so a new caller anywhere in the tree inherits it.

What makes it a *property* and not a pattern match is where it looks. The unit
is a readelf invocation that asks for a symbol table at all, and the question
asked of it is only "which table". That is deliberately not "which invocation
goes on to count kernels": the count is frequently a separate statement from the
read, and chasing it would mean following a variable across statements. Reading
the flag off the argv needs no such thing, and it cannot be evaded by moving the
counting elsewhere::

    dynsyms="$(llvm-readelf --dyn-syms "${OBJECT}")"      # the invocation
    kernels="$(printf '%s\\n' "${dynsyms}" | grep -c 'FUNC')"   # the count

The trade is that a readelf ``--symbols`` used for something *other* than
counting kernels would also be flagged. There is no such caller today, and if
one appears it should have to say so out loud rather than sit next to a figure
nobody can re-derive.

Two things keep this from matching the wrong text. Scanning is restricted to
files that can execute a command, because prose that quotes an invocation is not
one and re-wrapping a paragraph must not fail this test. And continued lines are
folded first, since an invocation wraps: on ``\\`` or a trailing pipeline
operator in shell, and on an unclosed bracket in Python.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# This file necessarily spells out the pattern it forbids -- in the docstring
# above and as fixture data below -- so it is not itself a call site.
_SELF = Path(__file__).resolve()

# Only files that can run a command. Markdown and friends are excluded on
# purpose: they describe call sites rather than being them.
_COMMAND_SUFFIXES = frozenset({".sh", ".bash", ".zsh", ".py", ".yaml", ".yml"})

# Every spelling that asks llvm-readelf for a symbol table. Matched as whole
# tokens: a substring test for "-s" would fire on "--dyn-syms", which is the
# flag we want people to use.
_SYMBOL_TABLE_FLAGS = re.compile(
    r"(?<![\w-])(?:--dyn-syms|--symbols|--syms|--all|-s|-a)(?![\w-])"
)

# The subset of those that emit .symtab alongside .dynsym, and so double-count.
_DOUBLE_COUNTING_FLAGS = re.compile(r"(?<![\w-])(?:--symbols|--syms|--all|-s|-a)(?![\w-])")

# A readelf invocation ends at the first operator that starts another command.
# Flags have to be read off readelf's own argv, not off whatever consumes its
# output: `llvm-readelf --dyn-syms x | grep -a -c FUNC` is a correct call site,
# and attributing grep's -a to readelf would reject it.
_SEGMENT_END = re.compile(r"[|;&]")

# A physical line ending in one of these operators is continued by the next one,
# so the two are a single command as far as the shell is concerned. A trailing
# backslash does the same and is handled separately, since it is consumed rather
# than kept. ``||`` needs no entry of its own: it also ends in ``|``.
_CONTINUATIONS = ("|", "&&")


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [_REPO_ROOT / name for name in out.split("\0") if name]


def _runs_commands(path: Path, text: str) -> bool:
    """True for a file that can carry a command: a script, a module, a workflow."""
    if path.suffix in _COMMAND_SUFFIXES:
        return True
    return not path.suffix and text.startswith("#!")


def _has_open_bracket(text: str) -> bool:
    """True while a bracket is still open, so a call spans more lines.

    Deliberately naive -- it does not know about string literals. That is safe
    only because it is applied to Python, where brackets balance per statement,
    and only inside the handful of files that mention readelf at all. Shell is
    excluded: ``awk '{print $1}'`` would read as an open brace forever.
    """
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    return depth > 0


def _logical_lines(text: str, *, brackets: bool = False) -> list[tuple[int, str]]:
    """Fold continued physical lines into commands, as ``(first_lineno, text)``.

    Comment-only lines are dropped, so explaining the double count in a comment
    does not read as committing one. ``brackets`` additionally folds an unclosed
    call across lines, which is how a Python invocation wraps.
    """
    commands: list[tuple[int, str]] = []
    buffer = ""
    start = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not buffer:
            if not line or line.startswith("#"):
                continue
            start = lineno
            buffer = line
        else:
            buffer = f"{buffer} {line}"
        if buffer.endswith("\\"):
            buffer = buffer[:-1].rstrip()
            continue
        if buffer.endswith(_CONTINUATIONS):
            continue
        if brackets and _has_open_bracket(buffer):
            continue
        commands.append((start, buffer))
        buffer = ""
    if buffer:
        commands.append((start, buffer))
    return commands


def _readelf_argv(command: str) -> str:
    """The argv slice of every llvm-readelf invocation in a folded command.

    Everything downstream of a pipe belongs to another tool and its flags are
    not readelf's, so only these slices are searched for the bad spellings.
    """
    slices = []
    for match in re.finditer("readelf", command):
        rest = command[match.end() :]
        end = _SEGMENT_END.search(rest)
        slices.append(rest[: end.start()] if end else rest)
    return " ".join(slices)


def _symbol_table_reads() -> list[tuple[Path, int, str]]:
    """Every readelf invocation in the tree that asks for a symbol table."""
    found: list[tuple[Path, int, str]] = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: not a readelf call site
        if "readelf" not in text or not _runs_commands(path, text):
            continue
        if path.resolve() == _SELF:
            continue
        for lineno, command in _logical_lines(text, brackets=path.suffix == ".py"):
            argv = _readelf_argv(command)
            if argv and _SYMBOL_TABLE_FLAGS.search(argv):
                found.append((path.relative_to(_REPO_ROOT), lineno, command))
    return found


def test_kernel_counts_never_come_from_the_double_counting_table() -> None:
    """No symbol table is read with ``--symbols`` or an alias, which report 2x."""
    offenders = [
        (path, lineno, command)
        for path, lineno, command in _symbol_table_reads()
        if _DOUBLE_COUNTING_FLAGS.search(_readelf_argv(command))
    ]
    assert not offenders, (
        "llvm-readelf --symbols (and its aliases -s / --syms, and -a / --all, "
        "which includes -s) prints .dynsym and .symtab, so counting kernel "
        "symbols across it double-counts; use --dyn-syms:\n"
        + "\n".join(f"  {path}:{lineno}: {cmd}" for path, lineno, cmd in offenders)
    )


def test_the_repro_script_still_reads_a_symbol_table() -> None:
    """The guard above stays meaningful only while a real call site exists.

    Without this, deleting the last readelf invocation would leave the
    ``--symbols`` assertion vacuously true.
    """
    call_sites = _symbol_table_reads()
    assert call_sites, "no symbol-table readelf call site found; the guard above is vacuous"
    assert any("--dyn-syms" in command for _, _, command in call_sites), (
        "expected at least one symbol-table read to use --dyn-syms; found:\n"
        + "\n".join(f"  {path}:{lineno}: {cmd}" for path, lineno, cmd in call_sites)
    )


def test_reading_the_flag_survives_the_count_moving_to_another_statement() -> None:
    """The reason the unit is the invocation and not the count.

    The repro script reads into a variable so it can keep llvm-readelf's exit
    status, then counts on the next line. A guard that required the invocation
    and the kernel marker in one expression would see no call site at all here.
    """
    script = (
        'if dynsyms="$(llvm-readelf --symbols "${OBJECT}" 2>/dev/null)"; then\n'
        "    kernels=\"$(printf '%s\\n' \"${dynsyms}\" | grep -c 'FUNC.*GLOBAL')\"\n"
        "fi\n"
    )
    reads = [
        command
        for _, command in _logical_lines(script)
        if _SYMBOL_TABLE_FLAGS.search(_readelf_argv(command))
    ]
    assert len(reads) == 1, reads
    assert "FUNC" not in reads[0], "the count is on the other line; that must not matter"
    assert _DOUBLE_COUNTING_FLAGS.search(_readelf_argv(reads[0]))


def test_a_wrapped_pipeline_is_still_seen_as_one_command() -> None:
    """The fold is what makes the guard hold up against a split pipeline.

    A count written across two physical lines is the natural shape once the
    command grows, and it is the shape that a line-at-a-time scan misses.
    """
    script = (
        "# llvm-readelf --symbols is wrong here, per FUNC below\n"
        'kernels="$(llvm-readelf --symbols "${OBJECT}" \\\n'
        "    | grep -c 'FUNC.*GLOBAL')\"\n"
    )
    commands = _logical_lines(script)
    assert len(commands) == 1, commands
    lineno, command = commands[0]
    assert lineno == 2, "the comment above it is not part of the command"
    assert "--symbols" in command and "FUNC" in command
    assert "\\" not in command, "the continuation marker is consumed by the fold"


def test_every_spelling_of_the_double_counting_flag_is_rejected() -> None:
    """The bug can come back under a shorter name, so pin all of them.

    ``-s`` and ``--syms`` are aliases of ``--symbols``; ``-a`` / ``--all``
    expands to a set that includes ``-s``. ``--dyn-syms`` is the correct flag
    and must not be caught by the ``-s`` arm -- it contains those two characters.
    """
    for flag in ("--symbols", "--syms", "-s", "--all", "-a"):
        command = f"kernels=$(llvm-readelf {flag} \"${{OBJECT}}\" | grep -c 'FUNC.*GLOBAL')"
        argv = _readelf_argv(command)
        assert _SYMBOL_TABLE_FLAGS.search(argv), f"{flag} reads a symbol table"
        assert _DOUBLE_COUNTING_FLAGS.search(argv), f"{flag} double-counts but is accepted"

    good = "kernels=$(llvm-readelf --dyn-syms \"${OBJECT}\" | grep -c 'FUNC.*GLOBAL')"
    argv = _readelf_argv(good)
    assert _SYMBOL_TABLE_FLAGS.search(argv), "--dyn-syms is still a symbol-table read"
    assert not _DOUBLE_COUNTING_FLAGS.search(argv), "the correct flag must not be flagged"


def test_a_downstream_tools_options_are_not_read_as_readelf_flags() -> None:
    """Flags belong to the command that owns them.

    ``grep`` has its own ``-a`` and ``-s``, and they say nothing about which
    symbol table readelf printed. Searching the whole folded pipeline would
    reject these correct call sites.
    """
    for downstream in ("grep -a -c 'FUNC.*GLOBAL'", "grep -s -c 'FUNC.*GLOBAL'"):
        command = f'kernels=$(llvm-readelf --dyn-syms "${{OBJECT}}" | {downstream})'
        assert not _DOUBLE_COUNTING_FLAGS.search(_readelf_argv(command)), (
            f"{downstream} is downstream of the pipe; its options are not readelf's"
        )

    # The same pipeline with the bad flag on readelf itself is still caught, so
    # the narrowing above did not simply disarm the check.
    bad = "kernels=$(llvm-readelf --symbols \"${OBJECT}\" | grep -a -c 'FUNC.*GLOBAL')"
    assert _DOUBLE_COUNTING_FLAGS.search(_readelf_argv(bad))


def test_a_wrapped_python_invocation_is_one_expression() -> None:
    """Python wraps on brackets, not on backslashes.

    A ``subprocess`` argv list split over lines puts the tool name and the flag
    on different ones, which the shell fold alone would not join.
    """
    module = 'out = subprocess.check_output(\n    ["llvm-readelf",\n     "--symbols", obj]\n)\n'
    commands = _logical_lines(module, brackets=True)
    assert len(commands) == 1, commands
    assert _DOUBLE_COUNTING_FLAGS.search(_readelf_argv(commands[0][1]))

    # Without bracket folding the flag lands in its own record, away from the
    # tool name, and nothing matches -- which is why the shell fold is not enough.
    assert not any(
        _DOUBLE_COUNTING_FLAGS.search(_readelf_argv(command))
        for _, command in _logical_lines(module)
    )


def test_prose_is_not_a_call_site() -> None:
    """Re-wrapping a doc must never fail the guard above.

    The docs quote the wrong flag verbatim, so they would read as call sites if
    prose were scanned. The bait is asserted first: without it this would pass
    for the wrong reason.
    """
    doc = _REPO_ROOT / "docs/sanitizers/consan-4112-overlapping-anchor-patches.md"
    text = doc.read_text(encoding="utf-8")
    assert "llvm-readelf --symbols" in text, (
        f"bait is gone: {doc.name} no longer quotes the flag the scan looks for"
    )
    assert not _runs_commands(doc, text)

    # The property, not just this one file: prose is never a call site.
    assert not [path for path, _, _ in _symbol_table_reads() if path.suffix == ".md"]
