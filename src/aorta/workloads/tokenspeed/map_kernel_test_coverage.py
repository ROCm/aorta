#!/usr/bin/env python3
"""Map TokenSpeed's registered kernels to the tests that actually exercise them.

Why this exists: ``list_harness_coverage.py`` answers "what can the *benchmark*
harness drive", and the answer is one family (``gemm.mm``). But TokenSpeed's own
pytest suites build inputs for the other families directly, so they reach kernels
the benchmark harness cannot. Before leaning on those suites we need to know
which kernels they genuinely touch -- a suite existing for a family does not mean
every solution in that family is exercised.

Static analysis cannot answer this: the tests parametrize over solutions and skip
via a ``require`` fixture at run time, so the only honest way to tell a covered
kernel from a skipped one is to watch the kernel run. This patches the registry,
runs pytest in-process so the patch holds, and records three distinct things,
narrowest first. Each of the wider two has at some point been mistaken for
coverage, which is why they are reported rather than folded in:

``covered`` means the implementation was **entered** -- called, or subscripted as
a Triton entry point (``kernel[grid](...)``). This is the only one of the three
that proves the kernel executed.

``lookup_only`` is ``KernelRegistry.get_impl(name)`` returning the callable and
the test never entering it. That is a real pattern upstream: a suite may look an
implementation up purely to assert which module it lives in. This lookup used to
*be* the definition of ``covered``, which credited kernels that never launched.

``candidate_only`` comes from ``get_for_operator``, which returns *every*
candidate matching a family/mode. The upstream ``require`` fixture calls it only
to decide whether to skip, before narrowing candidates by dtype, so a name
appearing there means "a test looked at this operator" -- not that the kernel
ran.

Read ``covered`` as "this implementation ran", not "asserted correct" -- a test
can enter a kernel and still be a weak test. Two known blind spots, both of
which under-report:

* ``tokenspeed-kernel-amd/test/ops`` imports implementations directly
  (``from tokenspeed_kernel_amd.ops.gemm.mm_a16w16_gfx950 import ...``) and
  never consults the registry, so kernels it exercises look uncovered here.
* Coverage depends on visible hardware. Expert-parallel (``_ep_``) solutions
  skip when too few GPUs are exposed, so run this with the full node visible
  before concluding a kernel has no test.

Each suite runs in its own subprocess. Both suites are rooted at ``test/ops``
with their own conftest and a top-level ``utils`` module, so running them in one
interpreter collides on module names; the results are merged afterwards.

Runs INSIDE the TokenSpeed container, where /workspace holds the source tree.
Mount the staged copy (``stage_scripts.sh`` puts this file there):

  docker run --rm --device=/dev/kfd --device=/dev/dri \
      --group-add video --group-add render \
      --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
      --shm-size=16g --security-opt seccomp=unconfined \
      -v /tmp/ts-work/scripts:/tools -v /tmp/ts-work/cov:/cov <image> \
      python3 /tools/map_kernel_test_coverage.py --out /cov/coverage.json

Exit codes:
  0   the map was produced (independently of whether the tests passed)
  64  usage / environment error -- registry not importable, suites missing, or a
      per-suite probe that produced no map, which would leave its kernels
      reported as uncovered rather than unknown
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

# The suites that drive real kernels, cheap-first.
DEFAULT_SUITES = (
    "tokenspeed-kernel-amd/test/ops",
    "tokenspeed-kernel/test/ops",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="map TokenSpeed kernels to tests")
    p.add_argument(
        "--workspace",
        default="/workspace",
        help="TokenSpeed source tree root inside the container",
    )
    p.add_argument(
        "--suite",
        action="append",
        default=None,
        help="test path relative to --workspace; repeatable (default: both op suites)",
    )
    p.add_argument("--out", default=None, help="write the JSON map here")
    p.add_argument(
        "--pytest-arg",
        action="append",
        default=None,
        help="extra arg passed through to pytest; repeatable",
    )
    p.add_argument(
        "--_single",
        default=None,
        help=argparse.SUPPRESS,  # internal: run exactly one suite in-process
    )
    return p.parse_args(argv)


def _registry_inventory() -> dict[str, dict[str, str]]:
    """Every registered kernel, keyed by name.

    Reads the registry's private ``_by_name`` because there is no public
    enumeration API -- a real coupling to TokenSpeed internals, so it fails
    loudly rather than silently reporting nothing if the attribute moves.
    """
    from tokenspeed_kernel.registry import KernelRegistry

    registry = KernelRegistry.get()
    by_name = getattr(registry, "_by_name", None)
    if by_name is None:
        raise SystemExit(
            "KernelRegistry has no _by_name; TokenSpeed's registry internals "
            "changed and this tool needs updating"
        )

    return {
        str(name): {
            "family": str(getattr(spec, "family", "?")),
            "mode": str(getattr(spec, "mode", "?")),
            "solution": str(getattr(spec, "solution", "?")),
        }
        for name, spec in by_name.items()
    }


class _LaunchProbe:
    """What ``kernel[grid]`` returns: records only once the launcher is called.

    Triton's ``JITFunction.__getitem__`` binds the grid and returns a callable;
    the dispatch happens in that call. Keeping the record here rather than in
    ``__getitem__`` is what makes ``covered`` mean "was launched" instead of
    "someone built a launcher", and it keeps the object transparent for suites
    that inspect the launcher before calling it.
    """

    __slots__ = ("_aorta_launcher", "_aorta_name", "_aorta_record")

    def __init__(self, launcher: Any, name: str, record: Any) -> None:
        object.__setattr__(self, "_aorta_launcher", launcher)
        object.__setattr__(self, "_aorta_name", name)
        object.__setattr__(self, "_aorta_record", record)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._aorta_record(self._aorta_name, "launch")
        return self._aorta_launcher(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        return getattr(object.__getattribute__(self, "_aorta_launcher"), attr)

    def __repr__(self) -> str:
        return repr(self._aorta_launcher)


class _EntryProbe:
    """Transparent proxy that records when a kernel is actually entered.

    ``get_impl`` returning a callable proves only that a test asked for one.
    TokenSpeed itself has suites that go no further --
    ``test/ops/moe/test_latent_input.py`` calls ``get_impl`` solely to assert
    each implementation's ``__module__`` -- so crediting the lookup marks kernels
    covered that were never launched, which is the same overstatement as counting
    candidates instead of dispatches, one layer down.

    Both entry paths are recorded: ``proxy(...)`` for an ordinary callable and
    ``proxy[grid](...)`` for a Triton entry point. Subscripting alone is *not*
    an entry -- ``kernel[grid]`` only binds the grid and hands back a launcher
    object, and a test is free to build one and never call it -- so
    ``__getitem__`` returns a :class:`_LaunchProbe` and the record happens when
    that launcher is invoked. Recording at the subscript would credit the
    kernel for a launch that never happened, which is the same overstatement
    this class exists to remove, one step further in.

    This is also why a plain ``functools.wraps`` wrapper would not do: it
    forwards neither ``__getitem__`` nor much else.

    Everything else must reach the wrapped object unchanged, because these suites
    are the measurement: if the proxy alters their behaviour it invalidates the
    very numbers it is collecting. Attribute access forwards, and
    ``update_wrapper`` copies ``__module__`` / ``__name__`` / ``__qualname__`` /
    ``__doc__`` onto the instance -- without that, ``proxy.__module__`` would
    resolve to *this* module via the class and fail exactly the assertion in
    ``test_latent_input.py``. Equality and hashing delegate so a suite holding
    impls in a set or comparing two lookups is unaffected, and proxies are cached
    per implementation so two lookups of one kernel stay identical objects.
    """

    def __init__(self, impl: Any, name: str, record: Any) -> None:
        object.__setattr__(self, "_aorta_impl", impl)
        object.__setattr__(self, "_aorta_name", name)
        object.__setattr__(self, "_aorta_record", record)
        # Not functools.update_wrapper: it assigns via setattr, which this class
        # forwards to the wrapped object, so it would quietly rewrite the
        # implementation's own __module__ instead of the proxy's. object's
        # __setattr__ is the one that puts these on the instance, where they
        # shadow the class attributes of the same name -- which is the point,
        # since `proxy.__module__` would otherwise resolve to this module and
        # fail an upstream assertion about where an implementation lives.
        for attr in functools.WRAPPER_ASSIGNMENTS:
            try:
                object.__setattr__(self, attr, getattr(impl, attr))
            except AttributeError:
                pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._aorta_record(self._aorta_name, "call")
        return self._aorta_impl(*args, **kwargs)

    def __getitem__(self, grid: Any) -> Any:
        return _LaunchProbe(
            self._aorta_impl[grid],
            self._aorta_name,
            self._aorta_record,
        )

    def __getattr__(self, attr: str) -> Any:
        return getattr(object.__getattribute__(self, "_aorta_impl"), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self._aorta_impl, attr, value)

    def __repr__(self) -> str:
        return repr(self._aorta_impl)

    def __eq__(self, other: Any) -> bool:
        other = getattr(other, "_aorta_impl", other)
        return bool(self._aorta_impl == other)

    def __hash__(self) -> int:
        return hash(self._aorta_impl)


def _install_probe(
    entered: dict[str, set[str]],
    dispatched: dict[str, set[str]],
    candidates: dict[str, set[str]],
    requested: dict[str, int],
) -> None:
    """Record what the suites resolve, dispatch, and actually enter.

    Three different questions, and conflating any two of them overstates
    coverage:

    ``get_for_operator`` returns *every* candidate matching a family/mode. The
    upstream ``require`` fixture calls it only to decide whether to skip, before
    narrowing candidates by dtype, so a name appearing here means "a test looked
    at this operator", not "a test ran this kernel". Recorded as ``candidates``.

    ``get_impl(name)`` is the post-selection lookup: how a caller obtains the
    callable for the kernel it intends to run. Recorded as ``dispatched``.

    Entering that callable -- ``impl(...)`` or ``impl[grid](...)`` -- is the only
    one of the three that proves the kernel ran, so it is what ``covered`` is
    computed from. Recorded as ``entered`` via :class:`_EntryProbe`. A lookup
    with no entry is reported as ``lookup_only``, which is a real state upstream:
    ``test_latent_input.py`` looks up three MoE implementations purely to assert
    their ``__module__``.

    Both accessors are patched on the class rather than the singleton, because
    tests may rebuild the registry and a per-instance patch would then stop
    recording silently.
    """
    from tokenspeed_kernel.registry import KernelRegistry

    original_for_operator = KernelRegistry.get_for_operator
    original_get_impl = KernelRegistry.get_impl
    proxies: dict[int, Any] = {}

    def record_entry(name: str, how: str) -> None:
        entered[name].add(how)

    def probed_for_operator(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_for_operator(self, *args, **kwargs)

        family = kwargs.get("family", args[0] if args else "?")
        mode = kwargs.get("mode", args[1] if len(args) > 1 else "?")
        solution = kwargs.get("solution", "?")
        key = f"{family}.{mode}::{solution}"
        requested[key] = requested.get(key, 0) + 1

        # Record names so the join is by kernel identity rather than by family,
        # which would over-credit multi-solution families.
        specs = result if isinstance(result, (list, tuple, set)) else [result]
        for spec in specs:
            name = getattr(spec, "name", None)
            if name:
                candidates[str(name)].add(key)

        return result

    def probed_get_impl(self: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        result = original_get_impl(self, name, *args, **kwargs)
        # A None result is a miss -- the kernel has no registered implementation,
        # so nothing was dispatched and recording it would credit a kernel that
        # cannot run.
        if result is None or not name:
            return result
        dispatched[str(name)].add("get_impl")
        if not callable(result) and not hasattr(type(result), "__getitem__"):
            # Nothing to enter, so nothing to instrument; the lookup is all the
            # signal there is for this one.
            return result
        try:
            return proxies.setdefault(id(result), _EntryProbe(result, str(name), record_entry))
        except Exception as exc:  # pragma: no cover - defensive
            # Never let instrumentation break the suites: an un-proxyable
            # implementation degrades this kernel to lookup_only rather than
            # failing the run, and says so instead of quietly under-reporting.
            print(
                f"map_kernel_test_coverage: cannot instrument {name} "
                f"({type(exc).__name__}: {exc}); recording lookup only",
                file=sys.stderr,
            )
            return result

    KernelRegistry.get_for_operator = probed_for_operator  # type: ignore[method-assign]
    KernelRegistry.get_impl = probed_get_impl  # type: ignore[method-assign]


def _run_one_suite(args: argparse.Namespace) -> int:
    """In-process: patch the registry, run one suite, emit a partial map."""
    import pytest

    suite = Path(args._single)
    entered: dict[str, set[str]] = defaultdict(set)
    dispatched: dict[str, set[str]] = defaultdict(set)
    candidates: dict[str, set[str]] = defaultdict(set)
    requested: dict[str, int] = {}

    inventory = _registry_inventory()
    _install_probe(entered, dispatched, candidates, requested)

    argv = [str(suite), "-q", "--no-header", "-p", "no:cacheprovider"]
    argv += args.pytest_arg or []
    code = int(pytest.main(argv))

    partial = {
        "inventory": inventory,
        "entered": {name: sorted(keys) for name, keys in entered.items()},
        "dispatched": {name: sorted(keys) for name, keys in dispatched.items()},
        "candidates": {name: sorted(keys) for name, keys in candidates.items()},
        "lookups": requested,
        "exit_code": code,
    }
    Path(args.out).write_text(json.dumps(partial) + "\n")
    return 0


def _suite_cwd(workspace: Path, suite: Path) -> Path:
    """The package root a suite must run from -- the parent of its `test` dir."""
    for parent in suite.parents:
        if parent.name == "test":
            return parent.parent
    return workspace


def main() -> int:
    args = _parse_args()

    if args._single:
        return _run_one_suite(args)

    workspace = Path(args.workspace)
    if not workspace.is_dir():
        print(f"workspace {workspace} is not a directory", file=sys.stderr)
        return 64

    suites = args.suite or list(DEFAULT_SUITES)
    resolved = [(s, workspace / s) for s in suites]
    missing = [str(p) for _, p in resolved if not p.exists()]
    if missing:
        print(f"test suites not found: {', '.join(missing)}", file=sys.stderr)
        return 64

    inventory: dict[str, dict[str, str]] = {}
    entered: dict[str, set[str]] = defaultdict(set)
    dispatched: dict[str, set[str]] = defaultdict(set)
    candidates: dict[str, set[str]] = defaultdict(set)
    lookups: dict[str, int] = {}
    exit_codes: dict[str, int] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for idx, (label, path) in enumerate(resolved):
            part = Path(tmp) / f"part{idx}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--_single",
                str(path),
                "--out",
                str(part),
                "--workspace",
                str(workspace),
            ]
            for extra in args.pytest_arg or []:
                cmd += ["--pytest-arg", extra]

            print(f"=== {label} ===", flush=True)
            proc = subprocess.run(cmd, cwd=_suite_cwd(workspace, path), check=False)

            # A non-zero rc here is the *wrapper* crashing, not tests failing:
            # `_run_one_suite` returns 0 and records pytest's own exit code
            # inside the partial map. So either of these means this suite
            # contributed no observations, and continuing would report its
            # kernels as uncovered rather than unknown -- understating coverage
            # in a number this tool exists to state precisely.
            if proc.returncode != 0:
                print(
                    f"map_kernel_test_coverage: {label} probe exited "
                    f"{proc.returncode}; coverage totals would be incomplete",
                    file=sys.stderr,
                )
                return 64
            if not part.exists():
                print(
                    f"map_kernel_test_coverage: {label} wrote no map to {part}; "
                    "coverage totals would be incomplete",
                    file=sys.stderr,
                )
                return 64

            data = json.loads(part.read_text())
            inventory.update(data["inventory"])
            for name, keys in data["entered"].items():
                entered[name].update(keys)
            for name, keys in data["dispatched"].items():
                dispatched[name].update(keys)
            for name, keys in data["candidates"].items():
                candidates[name].update(keys)
            for key, count in data["lookups"].items():
                lookups[key] = lookups.get(key, 0) + count
            exit_codes[label] = data["exit_code"]

    if not inventory:
        print("no registry inventory collected; nothing to report", file=sys.stderr)
        return 64

    report: dict[str, Any] = {
        "kernels": {
            # Three states, narrowest first, because each of the wider two was at
            # some point mistaken for coverage:
            #
            #   covered      the implementation was entered -- called, or
            #                subscripted as a Triton entry point. The only one
            #                that proves the kernel ran.
            #   lookup_only  get_impl handed the test the callable and the test
            #                never entered it (upstream does this to assert
            #                __module__).
            #   candidate_only
            #                a test looked at the operator but this
            #                implementation was never selected, usually filtered
            #                out by dtype after the skip check.
            name: {
                **meta,
                "covered": name in entered,
                "lookup_only": name in dispatched and name not in entered,
                "candidate_only": (
                    name in candidates and name not in dispatched and name not in entered
                ),
                "entered_via": sorted(entered.get(name, ())),
                "via": sorted(candidates.get(name, ())),
            }
            for name, meta in sorted(inventory.items())
        },
        "lookups": dict(sorted(lookups.items())),
        "suite_exit_codes": exit_codes,
    }
    n_cov = sum(1 for k in report["kernels"].values() if k["covered"])
    n_look = sum(1 for k in report["kernels"].values() if k["lookup_only"])
    n_cand = sum(1 for k in report["kernels"].values() if k["candidate_only"])
    report["summary"] = {
        "kernels_total": len(inventory),
        "kernels_covered": n_cov,
        "kernels_lookup_only": n_look,
        "kernels_candidate_only": n_cand,
        "kernels_uncovered": len(inventory) - n_cov - n_look - n_cand,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}")
    else:
        print(text)

    print(f"\nsummary: {report['summary']}")
    print("\ncovered kernels (implementation entered):")
    for name, meta in sorted(report["kernels"].items()):
        if meta["covered"]:
            via = ",".join(meta["entered_via"])
            print(f"  {meta['family']}.{meta['mode']:<26} {name} ({via})")
    print("\nlookup-only kernels (implementation obtained, never entered):")
    for name, meta in sorted(report["kernels"].items()):
        if meta["lookup_only"]:
            print(f"  {meta['family']}.{meta['mode']:<26} {name}")
    print("\ncandidate-only kernels (operator reached, implementation not selected):")
    for name, meta in sorted(report["kernels"].items()):
        if meta["candidate_only"]:
            print(f"  {meta['family']}.{meta['mode']:<26} {name}")
    print("\nuncovered kernels:")
    for name, meta in sorted(report["kernels"].items()):
        if not (meta["covered"] or meta["lookup_only"] or meta["candidate_only"]):
            print(f"  {meta['family']}.{meta['mode']:<26} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
