"""Guard the mechanism used to count kernels in an AMDGPU code object.

``llvm-readelf --symbols`` prints ``.dynsym`` *and* ``.symtab``, and a kernel is
listed in both, so counting matches across that output reports exactly twice the
kernel count. That is not a hypothetical: the figures recorded for the heavy f32
Tensile object (490 / 1554 / 682 kernels) were all 2x for this reason, and the
490 was quoted in four places before it was caught. ``--dyn-syms`` reads the one
table and is the correct source.

The check is phrased as the property rather than as a diff against one known-bad
line, so a new caller anywhere in the tree inherits it.

Two things keep the property from being matched on the wrong text. Scanning is
restricted to files that can execute a command, because prose that quotes a
readelf invocation is not a call site and re-wrapping a paragraph must not fail
this test. And a pipeline is folded into one logical command before matching,
because the flag and the thing that counts its output routinely sit on
different physical lines::

    kernels="$(llvm-readelf --symbols "${OBJECT}" \\
        | grep -c 'FUNC.*GLOBAL')"

That is a real double count, and matching physical lines would miss it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# This file necessarily spells out the pattern it forbids -- in the docstring
# above and as fixture data below -- so it is not itself a call site.
_SELF = Path(__file__).resolve()

# A readelf invocation is "counting kernels" if it also names something that
# identifies a kernel symbol: the FUNC symbol type or the .kd descriptor suffix.
_COUNTING_MARKERS = ("FUNC", ".kd")

# Only files that can run a command. Markdown and friends are excluded on
# purpose: they describe call sites rather than being them.
_COMMAND_SUFFIXES = frozenset({".sh", ".bash", ".zsh", ".py", ".yaml", ".yml"})

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
    """True for a file whose contents are executed rather than read."""
    if path.suffix in _COMMAND_SUFFIXES:
        return True
    return not path.suffix and text.startswith("#!")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Fold continued physical lines into commands, as ``(first_lineno, text)``.

    Comment-only lines are dropped, so explaining the double count in a comment
    does not read as committing one.
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
        commands.append((start, buffer))
        buffer = ""
    if buffer:
        commands.append((start, buffer))
    return commands


def _kernel_counting_readelf_commands() -> list[tuple[Path, int, str]]:
    """Every command in a tracked file that counts kernel symbols via llvm-readelf."""
    found: list[tuple[Path, int, str]] = []
    for path in _tracked_text_files():
        if path.resolve() == _SELF:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: not a readelf call site
        if "readelf" not in text or not _runs_commands(path, text):
            continue
        for lineno, command in _logical_lines(text):
            if "readelf" not in command:
                continue
            if not any(marker in command for marker in _COUNTING_MARKERS):
                continue
            found.append((path.relative_to(_REPO_ROOT), lineno, command))
    return found


def test_kernel_counts_never_come_from_the_double_counting_table() -> None:
    """No kernel count is derived from ``--symbols``, which reports 2x."""
    offenders = [
        (path, lineno, command)
        for path, lineno, command in _kernel_counting_readelf_commands()
        if "--symbols" in command
    ]
    assert not offenders, (
        "llvm-readelf --symbols prints .dynsym and .symtab, so counting kernel "
        "symbols across it double-counts; use --dyn-syms:\n"
        + "\n".join(f"  {path}:{lineno}: {cmd}" for path, lineno, cmd in offenders)
    )


def test_the_repro_script_still_counts_kernels() -> None:
    """The guard above stays meaningful only while a real call site exists.

    Without this, deleting the last kernel-counting line would leave the
    ``--symbols`` assertion vacuously true.
    """
    call_sites = _kernel_counting_readelf_commands()
    assert call_sites, "no kernel-counting readelf call site found; the guard above is vacuous"
    assert any("--dyn-syms" in command for _, _, command in call_sites), (
        "expected at least one kernel count to use --dyn-syms; found:\n"
        + "\n".join(f"  {path}:{lineno}: {cmd}" for path, lineno, cmd in call_sites)
    )


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


def test_prose_is_not_a_call_site() -> None:
    """Re-wrapping a doc must never fail the guard above.

    The docs quote both the wrong flag and the ``.kd`` suffix, and a reflow can
    land them on one line without changing a word of meaning.
    """
    assert not _runs_commands(
        _REPO_ROOT / "docs/sanitizers/consan-4112-overlapping-anchor-patches.md",
        "`llvm-readelf --symbols` counts each `.kd` twice",
    )
