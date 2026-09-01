"""Guard the mechanism used to count kernels in an AMDGPU code object.

``llvm-readelf --symbols`` prints ``.dynsym`` *and* ``.symtab``, and a kernel is
listed in both, so counting matches across that output reports exactly twice the
kernel count. That is not a hypothetical: the figures recorded for the heavy f32
Tensile object (490 / 1554 / 682 kernels) were all 2x for this reason, and the
490 was quoted in four places before it was caught. ``--dyn-syms`` reads the one
table and is the correct source.

The flag has spellings. ``-s`` and ``--syms`` are aliases of ``--symbols``, and
``-a`` / ``--all`` expands to ``-h -l -S -s -r -d -V -A -I`` -- which includes
``-s``, so it double-counts too. All five are rejected; matching only the long
form would let the same bug back in under a shorter name.

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

import re
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

# Every spelling that makes llvm-readelf emit .symtab alongside .dynsym. Matched
# as whole tokens: a substring test for "-s" would fire on "--dyn-syms", which is
# the flag we want people to use.
_DOUBLE_COUNTING_FLAGS = re.compile(r"(?<![\w-])(?:--symbols|--syms|--all|-s|-a)(?![\w-])")

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
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: not a readelf call site
        if "readelf" not in text or not _runs_commands(path, text):
            continue
        if path.resolve() == _SELF:
            continue
        for lineno, command in _logical_lines(text):
            if "readelf" not in command:
                continue
            if not any(marker in command for marker in _COUNTING_MARKERS):
                continue
            found.append((path.relative_to(_REPO_ROOT), lineno, command))
    return found


def test_kernel_counts_never_come_from_the_double_counting_table() -> None:
    """No kernel count is derived from ``--symbols`` or an alias, which report 2x."""
    offenders = [
        (path, lineno, command)
        for path, lineno, command in _kernel_counting_readelf_commands()
        if _DOUBLE_COUNTING_FLAGS.search(command)
    ]
    assert not offenders, (
        "llvm-readelf --symbols (and its aliases -s / --syms, and -a / --all, "
        "which includes -s) prints .dynsym and .symtab, so counting kernel "
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


def test_every_spelling_of_the_double_counting_flag_is_rejected() -> None:
    """The bug can come back under a shorter name, so pin all of them.

    ``-s`` and ``--syms`` are aliases of ``--symbols``; ``-a`` / ``--all``
    expands to a set that includes ``-s``. ``--dyn-syms`` is the correct flag
    and must not be caught by the ``-s`` arm -- it contains those two characters.
    """
    for flag in ("--symbols", "--syms", "-s", "--all", "-a"):
        command = f"kernels=$(llvm-readelf {flag} \"${{OBJECT}}\" | grep -c 'FUNC.*GLOBAL')"
        assert _DOUBLE_COUNTING_FLAGS.search(command), f"{flag} double-counts but is accepted"

    good = "kernels=$(llvm-readelf --dyn-syms \"${OBJECT}\" | grep -c 'FUNC.*GLOBAL')"
    assert not _DOUBLE_COUNTING_FLAGS.search(good), "the correct flag must not be flagged"


def test_prose_is_not_a_call_site() -> None:
    """Re-wrapping a doc must never fail the guard above.

    The doc quotes both the wrong flag and the ``.kd`` marker, so a reflow that
    landed them on one line would read as a call site if prose were scanned.
    The bait is asserted first: without it this would pass for the wrong reason.
    """
    doc = _REPO_ROOT / "docs/sanitizers/consan-4112-overlapping-anchor-patches.md"
    text = doc.read_text(encoding="utf-8")
    assert "llvm-readelf --symbols" in text and ".kd" in text, (
        "bait is gone: this doc no longer quotes the flag and the marker the scan looks for"
    )
    assert not _runs_commands(doc, text)
    flagged = {path for path, _, _ in _kernel_counting_readelf_commands()}
    assert doc.relative_to(_REPO_ROOT) not in flagged
