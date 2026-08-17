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
* ``uncovered``  -- audited env-var string in a library that the registry never mentions
                    (all three standard prefixes plus the source-verified non-prefixed
                    names in ``AUDITED_EXACT_NAMES``).
* ``not_present``-- registry name absent from every installed library. Expected and fine
                    for forward-compat entries and for knobs a given ROCm drops. Capture is
                    independent: an exported value is still preserved; ``null`` means unset.
* owner errors   -- the registry's library attribution disagrees with the exact
                    reference-build binaries.
* prov errors    -- reference-build presence disagrees with ``source_reference``.

Resolving the right library matters: a ROCm tree can keep a stale
``libhipblaslt.so.1.0.70002`` beside the active ``1.4.70002``, so globbing
``libhipblaslt.so.*`` gives the wrong answer. This follows the soname symlink to the file
that would actually load.

Usage
-----
    python scripts/audit_env_knobs.py                      # audit the local ROCm install
    python scripts/audit_env_knobs.py --rocm-lib /opt/rocm-7.0.2/lib
    python scripts/audit_env_knobs.py --json report.json   # machine-readable
    python scripts/audit_env_knobs.py --strict             # non-zero exit on audit errors

Exit codes: 0 clean; 1 uncovered/ownership/provenance error under ``--strict``;
2 setup problem such as a missing requested library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Prefixes plus exact non-prefixed names whose getenv call sites were verified in the
# reference libraries. HSA_/NCCL_/MIOPEN_ knobs live elsewhere and are out of scope.
AUDITED_PREFIXES = ("HIPBLASLT_", "ROCBLAS_", "TENSILE_")
AUDITED_EXACT_NAMES = frozenset(
    {
        "ANALYTICAL_GEMM_DEBUG",
        "ANALYTICAL_GEMM_HEURISTICS",
        "ANALYTICAL_GEMM_HEURISTICS_VARIANCE",
        "GRIDBASED_TOPSOLS",
        "ORIGAMI_LOG_FILE",
    }
)
PREFIXED_ENV_NAME_RE = re.compile(r"^(?:HIPBLASLT|ROCBLAS|TENSILE)_[A-Z0-9_]+$")
_BINARY_ENV_NAME_RE = re.compile(
    rb"(?:HIPBLASLT|ROCBLAS|TENSILE)_[A-Z0-9_]+|"
    + b"|".join(
        re.escape(name.encode("ascii"))
        for name in sorted(AUDITED_EXACT_NAMES, key=len, reverse=True)
    )
)

# Sonames rather than globs: the symlink names the library that would actually load.
DEFAULT_SONAMES = ("libhipblaslt.so", "librocblas.so")
# Fallback only. The real default is resolved at run time (see
# ``default_rocm_lib``) because a TheRock wheel install keeps the GEMM
# libraries under site-packages rather than /opt/rocm -- issue #381.
DEFAULT_ROCM_LIB = Path("/opt/rocm/lib")
REFERENCE_LIBRARY_BASENAMES = {
    "libhipblaslt.so": "libhipblaslt.so.1.4.70002",
    "librocblas.so": "librocblas.so.5.0.70002",
}
REFERENCE_LIBRARY_SHA256 = {
    "libhipblaslt.so": "77b5dfa8b0434a64fbe10d7db9643ffdd475d4f692db3c90bdc2443e83879a63",
    "librocblas.so": "e919db633b9de51a824552ecf51a33259f130b9e9af11e7cd1d240379762e1b6",
}


@dataclass(frozen=True)
class LibraryStrings:
    soname: str
    resolved: Path
    names: frozenset[str]
    sha256: str


def is_audited_name(name: str) -> bool:
    return bool(PREFIXED_ENV_NAME_RE.fullmatch(name)) or name in AUDITED_EXACT_NAMES


def resolve_library(rocm_lib: Path, soname: str) -> Path | None:
    """Follow ``<soname>``, or the highest ``.<major>`` link, to the real file.

    The bare ``<soname>`` link is the devel package's; a runtime-only tree ships
    only ``<soname>.<major>``. That fallback must pick the HIGHEST major:
    ``sorted()`` is lexicographic, so it selected the oldest co-installed
    library, and a ``[0-9]`` glob never matched a two-digit major at all --
    which resolved to nothing and reported every knob of an unread library as
    "absent from this install".
    """
    direct = rocm_lib / soname
    if direct.is_file():
        return direct.resolve()
    majors: list[tuple[int, Path]] = []
    for candidate in rocm_lib.glob(f"{soname}.*"):
        suffix = candidate.name[len(soname) + 1 :]
        if suffix.isdigit() and candidate.is_file():
            majors.append((int(suffix), candidate))
    if not majors:
        return None
    return max(majors, key=lambda item: item[0])[1].resolve()


def extract_names(lib: Path) -> frozenset[str]:
    """Return audited names that occupy a complete NUL-delimited C string.

    ``strings`` splits on any non-printable byte (including newlines), which can
    manufacture a standalone-looking name from help text. Scan the mapped ELF
    bytes directly and require NUL on both sides of the candidate instead.
    """
    with lib.open("rb") as fh:
        if fh.read(4) != b"\x7fELF":
            raise ValueError(f"{lib} is empty or is not an ELF shared object")
    names: set[str] = set()
    with lib.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for match in _BINARY_ENV_NAME_RE.finditer(data):
            start, end = match.span()
            if (start == 0 or data[start - 1] == 0) and end < len(data) and data[end] == 0:
                names.add(match.group().decode("ascii"))
    return frozenset(names)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_registry_module():
    here = Path(__file__).resolve().parent.parent
    src = here / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from aorta.instrumentation import env_knobs

    return env_knobs


def _import_registry():
    return _import_registry_module().ENV_KNOB_REGISTRY


def default_rocm_lib() -> Path:
    """The lib directory to audit when ``--rocm-lib`` is not given.

    Resolved rather than hardcoded so the audit finds libhipblaslt.so on a
    TheRock wheel install (where it lives under site-packages) as well as on
    a classic /opt/rocm one. Falls back to the classic path if the resolver
    cannot be imported -- this script is also run standalone, and a wrong
    default is better than a traceback when ``--rocm-lib`` was going to be
    passed explicitly anyway.
    """
    here = Path(__file__).resolve().parent.parent
    src = here / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from aorta.instrumentation.rocm_paths import resolve_rocm_roots
    except ImportError:
        return DEFAULT_ROCM_LIB
    return resolve_rocm_roots().lib_dir


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
        " the variable belongs to. For GEMM knobs present in the reference build, ownership"
        " is measured from the shipped shared object's exact string table -- which shows the"
        " name, not a call site, so it is ownership and not proof of consumption;"
        " forward-compatible absent entries are marked in the manifest and are declared"
        " rather than measured. A row's presence records"
        " that the variable is preserved in a snapshot -- not that the installed library"
        " supports it, that the process exported it, or that it affected a run.",
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
    ap.add_argument(
        "--rocm-lib",
        type=Path,
        default=None,
        help="lib dir to audit (default: resolved from the local ROCm install)",
    )
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
        help=(
            "exit 1 on uncovered names, library-owner mismatches, or reference-build "
            "provenance mismatches; owner/provenance checks run only on the exact "
            "reference build (setup failures exit 2 in every mode)"
        ),
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

    registry = None
    registry_module = None
    if args.names_file:
        try:
            raw_names = args.names_file.read_text()
        except OSError as exc:
            # Exit 2, not 1: an unreadable input is a setup problem, and exit 1
            # is the documented "audit found drift" class.
            print(f"error: cannot read {args.names_file}: {exc}", file=sys.stderr)
            return 2
        names = {
            line.strip(): "(bootstrap)"
            for line in raw_names.splitlines()
            if line.strip() and not line.startswith("#")
        }
        source = f"{args.names_file} ({len(names)} names)"
    else:
        try:
            registry_module = _import_registry_module()
            registry = registry_module.ENV_KNOB_REGISTRY
            names = {knob.name: knob.library for knob in registry}
            source = f"aorta.instrumentation.env_knobs ({len(registry)} knobs)"
        except Exception as exc:  # noqa: BLE001 -- a bad import must not look like a clean audit
            print(f"error: cannot load the registry: {exc}", file=sys.stderr)
            return 2

    rocm_lib = args.rocm_lib if args.rocm_lib is not None else default_rocm_lib()
    if not rocm_lib.is_dir():
        print(f"error: {rocm_lib} is not a directory", file=sys.stderr)
        return 2

    requested_sonames = tuple(args.sonames or DEFAULT_SONAMES)
    libs: list[LibraryStrings] = []
    missing_sonames: list[str] = []
    library_errors: list[dict[str, str]] = []
    for soname in requested_sonames:
        resolved = resolve_library(rocm_lib, soname)
        if resolved is None:
            print(f"warning: {soname} not found under {rocm_lib}", file=sys.stderr)
            missing_sonames.append(soname)
            continue
        try:
            libs.append(
                LibraryStrings(
                    soname,
                    resolved,
                    extract_names(resolved),
                    _sha256_file(resolved),
                )
            )
        except (OSError, ValueError) as exc:
            # Invalid/unreadable input is a setup failure, not registry drift.
            # Preserve it in the JSON report and keep scanning the other
            # requested libraries so one bad path does not erase all context.
            print(f"error: cannot audit {resolved}: {exc}", file=sys.stderr)
            library_errors.append(
                {
                    "soname": soname,
                    "resolved": str(resolved),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    audited = {n for n in names if is_audited_name(n)}
    in_libs = {n for lib in libs for n in lib.names}

    covered = sorted(audited & in_libs)
    not_present = sorted(audited - in_libs)
    uncovered = sorted(in_libs - set(names))

    where: dict[str, list[str]] = {}
    for name in covered:
        where[name] = [lib.soname for lib in libs if name in lib.names]

    library_mismatches: list[dict[str, object]] = []
    provenance_mismatches: list[dict[str, object]] = []
    is_reference_build = False
    if registry is not None and registry_module is not None:
        by_name = {knob.name: knob for knob in registry}
        # Both ownership and source_reference are scoped to the reference build.
        # Validate them only when BOTH exact reference libraries were resolved;
        # on another ROCm release, presence and library ownership can legitimately
        # differ.
        resolved_by_soname = {lib.soname: lib for lib in libs}
        is_reference_build = set(REFERENCE_LIBRARY_BASENAMES).issubset(resolved_by_soname) and all(
            resolved_by_soname[soname].resolved.name == basename
            and resolved_by_soname[soname].sha256 == REFERENCE_LIBRARY_SHA256[soname]
            for soname, basename in REFERENCE_LIBRARY_BASENAMES.items()
        )
        if is_reference_build:
            reference_where = {
                name: [
                    soname for soname in DEFAULT_SONAMES if name in resolved_by_soname[soname].names
                ]
                for name in audited
            }
            for name, present_in in reference_where.items():
                if not present_in:
                    continue
                knob = by_name[name]
                observed_library = (
                    "hipblaslt+rocblas"
                    if len(present_in) == 2
                    else "hipblaslt" if present_in == ["libhipblaslt.so"] else "rocblas"
                )
                if knob.library != observed_library:
                    library_mismatches.append(
                        {
                            "name": name,
                            "declared": knob.library,
                            "observed": observed_library,
                            "present_in": present_in,
                        }
                    )
            for name in sorted(audited):
                knob = by_name[name]
                present_in = reference_where[name]
                expected = (
                    registry_module.REF_BOTH_SO
                    if len(present_in) == 2
                    else (
                        registry_module.REF_HIPBLASLT_SO
                        if present_in == ["libhipblaslt.so"]
                        else (
                            registry_module.REF_ROCBLAS_SO
                            if present_in == ["librocblas.so"]
                            else registry_module.ABSENT_FROM_REFERENCE_BUILD
                        )
                    )
                )
                if knob.source_reference != expected:
                    provenance_mismatches.append(
                        {
                            "name": name,
                            "declared": knob.source_reference,
                            "expected": expected,
                            "present_in": present_in,
                        }
                    )

    print("=" * 78)
    print("ENV KNOB AUDIT")
    print("=" * 78)
    print(f"  registry source : {source}")
    print(f"  audited prefixes: {', '.join(AUDITED_PREFIXES)}")
    print(f"  audited exact   : {', '.join(sorted(AUDITED_EXACT_NAMES))}")
    for lib in libs:
        print(
            f"  library         : {lib.resolved}  "
            f"(sha256={lib.sha256[:12]}…, {len(lib.names)} env strings)"
        )
    for soname in missing_sonames:
        print(f"  MISSING library : {soname}")
    for item in library_errors:
        print(f"  INVALID library : {item['soname']} ({item['error']})")
    if registry is not None:
        state = "active" if is_reference_build else "skipped (different library build)"
        print(f"  reference checks: {state}")
    print()
    print(f"  covered      {len(covered):4d}  registry knobs found in an installed library")
    print(
        f"  not_present  {len(not_present):4d}  registry knobs absent here "
        "(capture still preserves any exported value)"
    )
    print(f"  uncovered    {len(uncovered):4d}  library knobs the registry omits")
    # Never print 0 for a check that did not run: owner / provenance are scoped
    # to the reference build, and a bare "0" reads as "checked, clean" to both
    # an operator and a JSON consumer.
    reference_count = (
        (lambda n: f"{n:4d}") if is_reference_build else (lambda n: "  --")  # noqa: ARG005
    )
    reference_note = "" if is_reference_build else " (not checked: not the reference build)"
    print(
        f"  owner errors {reference_count(len(library_mismatches))}  "
        f"manifest library disagrees with reference binary{reference_note}"
    )
    print(
        f"  prov errors  {reference_count(len(provenance_mismatches))}  "
        f"reference-build provenance mismatch{reference_note}"
    )

    if uncovered:
        print("\n  UNCOVERED -- present in the installed library, missing from the registry:")
        for name in uncovered:
            owners = ", ".join(lib.soname for lib in libs if name in lib.names)
            print(f"    {name}  ({owners})")
    if not_present:
        print("\n  NOT PRESENT in this install (fine: forward-compat or version-dropped):")
        for name in not_present:
            print(f"    {name}")
    if library_mismatches:
        print("\n  LIBRARY ATTRIBUTION MISMATCHES:")
        for item in library_mismatches:
            print(f"    {item['name']}: declared={item['declared']} observed={item['observed']}")
    if provenance_mismatches:
        print("\n  REFERENCE-BUILD PROVENANCE MISMATCHES:")
        for item in provenance_mismatches:
            print(f"    {item['name']}: declared={item['declared']} expected={item['expected']}")

    if args.json:
        report = json.dumps(
            {
                "registry_source": source,
                "libraries": [
                    {
                        "soname": lib.soname,
                        "resolved": str(lib.resolved),
                        "sha256": lib.sha256,
                        "env_string_count": len(lib.names),
                    }
                    for lib in libs
                ],
                "covered": covered,
                "covered_in": where,
                "not_present": not_present,
                "uncovered": uncovered,
                "missing_sonames": missing_sonames,
                "library_errors": library_errors,
                "reference_build_validation": is_reference_build,
                # ``null`` (not ``[]``) when the reference gate did not activate,
                # so a consumer cannot read "did not run" as "ran and found
                # nothing".
                "library_mismatches": (library_mismatches if is_reference_build else None),
                "provenance_mismatches": (provenance_mismatches if is_reference_build else None),
            },
            indent=1,
            sort_keys=True,
        )
        try:
            args.json.write_text(report)
        except OSError as exc:
            # Setup problem (exit 2), not audit drift (exit 1).
            print(f"error: cannot write {args.json}: {exc}", file=sys.stderr)
            return 2
        print(f"\n  wrote {args.json}")

    if missing_sonames or library_errors:
        return 2
    audit_errors = uncovered or library_mismatches or provenance_mismatches
    return 1 if (audit_errors and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
