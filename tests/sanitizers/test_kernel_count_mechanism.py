"""Guard the mechanism used to count kernels in an AMDGPU code object.

``llvm-readelf --symbols`` prints ``.dynsym`` *and* ``.symtab``, and a kernel is
listed in both, so counting matches across that output reports exactly twice the
kernel count. That is not a hypothetical: the figures recorded for the heavy f32
Tensile object (490 / 1554 / 682 kernels) were all 2x for this reason, and the
490 was quoted in four places before it was caught. ``--dyn-syms`` reads the one
table and is the correct source.

The check is phrased as the property rather than as a diff against one known-bad
line, so a new caller anywhere in the tree inherits it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# A readelf invocation is "counting kernels" if it also names something that
# identifies a kernel symbol: the FUNC symbol type or the .kd descriptor suffix.
_COUNTING_MARKERS = ("FUNC", ".kd")


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [_REPO_ROOT / name for name in out.split("\0") if name]


def _kernel_counting_readelf_lines() -> list[tuple[Path, int, str]]:
    """Every line in a tracked file that counts kernel symbols via llvm-readelf."""
    found: list[tuple[Path, int, str]] = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: not a readelf call site
        if "readelf" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "readelf" not in line:
                continue
            if not any(marker in line for marker in _COUNTING_MARKERS):
                continue
            found.append((path.relative_to(_REPO_ROOT), lineno, line.strip()))
    return found


def test_kernel_counts_never_come_from_the_double_counting_table() -> None:
    """No kernel count is derived from ``--symbols``, which reports 2x."""
    offenders = [
        (path, lineno, line)
        for path, lineno, line in _kernel_counting_readelf_lines()
        if "--symbols" in line
    ]
    assert not offenders, (
        "llvm-readelf --symbols prints .dynsym and .symtab, so counting kernel "
        "symbols across it double-counts; use --dyn-syms:\n"
        + "\n".join(f"  {path}:{lineno}: {line}" for path, lineno, line in offenders)
    )


def test_the_repro_script_still_counts_kernels() -> None:
    """The guard above stays meaningful only while a real call site exists.

    Without this, deleting the last kernel-counting line would leave the
    ``--symbols`` assertion vacuously true.
    """
    call_sites = _kernel_counting_readelf_lines()
    assert call_sites, "no kernel-counting readelf call site found; the guard above is vacuous"
    assert any("--dyn-syms" in line for _, _, line in call_sites), (
        "expected at least one kernel count to use --dyn-syms; found:\n"
        + "\n".join(f"  {path}:{lineno}: {line}" for path, lineno, line in call_sites)
    )
