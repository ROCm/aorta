#!/usr/bin/env python3
"""Audit the env-knob registry against the GEMM libraries actually installed.

Why this exists
---------------
``ENV_KNOB_REGISTRY`` in ``aorta.instrumentation.env_knobs`` claims a library and a
provenance for every environment variable ``aorta env probe`` captures. A test that
compares the registry against a second hand-written list only detects drift between two
hand-written lists -- it cannot show that the *installed* libraries are covered. This
script closes that gap by reading the shipped shared objects and diffing their env-var
string tables against the registry, so coverage is a measurement rather than a claim.

What it reports
---------------
* ``covered``    -- registry name found in an installed library's string table.
* ``uncovered``  -- env-var-shaped string in a library that the registry never mentions
                    (the failure this script exists to catch: an upstream knob nobody
                    noticed).
* ``not_present``-- registry name absent from every installed library. Expected and fine
                    for forward-compat entries and for knobs a given ROCm drops; the probe
                    reports ``null`` for them. Never an error on its own.

Resolving the right library matters: a ROCm tree can keep a stale
``libhipblaslt.so.1.0.70002`` beside the active ``1.4.70002``, so globbing
``libhipblaslt.so.*`` gives the wrong answer. This follows the soname symlink to the file
that would actually load.

Usage
-----
    python scripts/audit_env_knobs.py                      # audit the local ROCm install
    python scripts/audit_env_knobs.py --rocm-lib /opt/rocm-7.0.2/lib
    python scripts/audit_env_knobs.py --json report.json   # machine-readable
    python scripts/audit_env_knobs.py --strict             # non-zero exit on uncovered

Exit codes: 0 clean, 1 uncovered strings found (only with ``--strict``), 2 setup problem.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Only prefixes whose upstream owners this registry actually audits. Kept narrow on
# purpose: HSA_/NCCL_/MIOPEN_ knobs live in other libraries whose string tables are not
# part of this audit, and reporting them as "uncovered" would be noise, not signal.
AUDITED_PREFIXES = ("HIPBLASLT_", "ROCBLAS_", "TENSILE_")
ENV_NAME_RE = re.compile(r"^(?:HIPBLASLT|ROCBLAS|TENSILE)_[A-Z0-9_]+$")

# Sonames rather than globs: the symlink names the library that would actually load.
DEFAULT_SONAMES = ("libhipblaslt.so", "librocblas.so")
DEFAULT_ROCM_LIB = Path("/opt/rocm/lib")


@dataclass(frozen=True)
class LibraryStrings:
    soname: str
    resolved: Path
    names: frozenset[str]


def resolve_library(rocm_lib: Path, soname: str) -> Path | None:
    """Follow ``<soname>`` (and the ``.1``-style major link) to the real file."""
    for candidate in (rocm_lib / soname, *sorted(rocm_lib.glob(f"{soname}.[0-9]"))):
        if candidate.exists():
            return candidate.resolve()
    return None


def extract_names(lib: Path) -> frozenset[str]:
    """Env-var-shaped strings in a shared object, via ``strings``.

    ``strings`` ships in binutils and is present in every ROCm image; a pure-Python
    fallback keeps the script usable where it is not.
    """
    try:
        proc = subprocess.run(
            ["strings", "-a", str(lib)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        blob = proc.stdout if proc.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        blob = ""

    if not blob:  # fallback: scan printable runs ourselves
        data = lib.read_bytes()
        blob = "\n".join(
            m.group().decode("ascii", "ignore") for m in re.finditer(rb"[\x20-\x7e]{4,}", data)
        )

    return frozenset(tok for tok in blob.split() if ENV_NAME_RE.match(tok))


def _import_registry():
    here = Path(__file__).resolve().parent.parent
    src = here / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from aorta.instrumentation.env_knobs import ENV_KNOB_REGISTRY

    return ENV_KNOB_REGISTRY


def load_registry() -> tuple[dict[str, str], str]:
    """``{name: library}`` from the registry, plus a description of its source."""
    registry = _import_registry()
    return (
        {knob.name: knob.library for knob in registry},
        f"aorta.instrumentation.env_knobs ({len(registry)} knobs)",
    )


# Markers delimiting the generated inventory inside docs/env-probe.md. The table
# between them is emitted by --emit-docs-table and checked by
# tests/instrumentation/test_env_knob_audit.py, so the docs cannot drift from the
# manifest the probe actually reads.
DOCS_TABLE_BEGIN = "<!-- BEGIN GENERATED: env-knob-inventory -->"
DOCS_TABLE_END = "<!-- END GENERATED: env-knob-inventory -->"


def render_docs_table() -> str:
    """The captured-knob inventory as a markdown table, grouped by category.

    Generated rather than hand-maintained: a hand-written inventory is a second
    copy of the manifest, and the whole point of the manifest is that there is
    only one copy. Needs no ROCm install -- it reads the manifest, not libraries.
    """
    registry = _import_registry()
    by_category: dict[tuple[str, str], list[str]] = {}
    for knob in registry:
        by_category.setdefault((knob.category, knob.library), []).append(knob.name)

    lines = [
        DOCS_TABLE_BEGIN,
        "",
        f"{len(registry)} knobs, generated from `ENV_KNOB_REGISTRY` by",
        "`python scripts/audit_env_knobs.py --emit-docs-table`. `library` is the component"
        " that reads the variable; for the GEMM prefixes it is measured from the shipped"
        " shared object's string table. A row's presence records that the variable is"
        " preserved in a snapshot -- not that the installed library supports it, that the"
        " process exported it, or that it affected a run.",
        "",
        "| Category | Library | Variables |",
        "| --- | --- | --- |",
    ]
    for (category, library), names in by_category.items():
        rendered = ", ".join(f"`{n}`" for n in names)
        lines.append(f"| `{category}` | {library} | {rendered} |")
    lines += ["", DOCS_TABLE_END]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rocm-lib", type=Path, default=DEFAULT_ROCM_LIB)
    ap.add_argument("--soname", action="append", dest="sonames", default=None)
    ap.add_argument(
        "--names-file",
        type=Path,
        help="bootstrap mode: newline-separated names instead of the registry",
    )
    ap.add_argument("--json", type=Path, help="write the report as JSON")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when a library exposes a knob the registry omits",
    )
    ap.add_argument(
        "--emit-docs-table",
        action="store_true",
        help="print the generated knob inventory for docs/env-probe.md and exit "
        "(reads the manifest only; no ROCm install needed)",
    )
    args = ap.parse_args()

    # Before any library resolution: this mode needs the manifest, nothing else.
    if args.emit_docs_table:
        print(render_docs_table())
        return 0

    if args.names_file:
        names = {
            line.strip(): "(bootstrap)"
            for line in args.names_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        source = f"{args.names_file} ({len(names)} names)"
    else:
        try:
            names, source = load_registry()
        except Exception as exc:  # noqa: BLE001 -- a bad import must not look like a clean audit
            print(f"error: cannot load the registry: {exc}", file=sys.stderr)
            return 2

    if not args.rocm_lib.is_dir():
        print(f"error: {args.rocm_lib} is not a directory", file=sys.stderr)
        return 2

    libs: list[LibraryStrings] = []
    for soname in args.sonames or DEFAULT_SONAMES:
        resolved = resolve_library(args.rocm_lib, soname)
        if resolved is None:
            print(f"warning: {soname} not found under {args.rocm_lib}", file=sys.stderr)
            continue
        libs.append(LibraryStrings(soname, resolved, extract_names(resolved)))
    if not libs:
        print("error: no GEMM libraries resolved; nothing to audit", file=sys.stderr)
        return 2

    audited = {n for n in names if n.startswith(AUDITED_PREFIXES)}
    in_libs = {n for lib in libs for n in lib.names}

    covered = sorted(audited & in_libs)
    not_present = sorted(audited - in_libs)
    uncovered = sorted(in_libs - set(names))

    where: dict[str, list[str]] = {}
    for name in covered:
        where[name] = [lib.soname for lib in libs if name in lib.names]

    print("=" * 78)
    print("ENV KNOB AUDIT")
    print("=" * 78)
    print(f"  registry source : {source}")
    print(f"  audited prefixes: {', '.join(AUDITED_PREFIXES)}")
    for lib in libs:
        print(f"  library         : {lib.resolved}  ({len(lib.names)} env strings)")
    print()
    print(f"  covered      {len(covered):4d}  registry knobs found in an installed library")
    print(f"  not_present  {len(not_present):4d}  registry knobs absent here (probe reports null)")
    print(f"  uncovered    {len(uncovered):4d}  library knobs the registry omits")

    if uncovered:
        print("\n  UNCOVERED -- present in the installed library, missing from the registry:")
        for name in uncovered:
            owners = ", ".join(lib.soname for lib in libs if name in lib.names)
            print(f"    {name}  ({owners})")
    if not_present:
        print("\n  NOT PRESENT in this install (fine: forward-compat or version-dropped):")
        for name in not_present:
            print(f"    {name}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "registry_source": source,
                    "libraries": [
                        {
                            "soname": lib.soname,
                            "resolved": str(lib.resolved),
                            "env_string_count": len(lib.names),
                        }
                        for lib in libs
                    ],
                    "covered": covered,
                    "covered_in": where,
                    "not_present": not_present,
                    "uncovered": uncovered,
                },
                indent=1,
                sort_keys=True,
            )
        )
        print(f"\n  wrote {args.json}")

    return 1 if (uncovered and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
