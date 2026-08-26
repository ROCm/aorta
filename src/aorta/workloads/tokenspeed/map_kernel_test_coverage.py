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
kernel from a skipped one is to watch the dispatch happen. This patches the
registry, runs pytest in-process so the patch holds, and records two distinct
things:

``covered`` comes from ``KernelRegistry.get_impl(name)`` -- the post-selection
lookup a caller makes to obtain the callable for the one kernel it is about to
run. A name here is the implementation actually selected.

``candidate_only`` comes from ``get_for_operator``, which returns *every*
candidate matching a family/mode. The upstream ``require`` fixture calls it only
to decide whether to skip, before narrowing candidates by dtype, so a name
appearing there means "a test looked at this operator" -- not that the kernel
ran. Counting those as covered would overstate what the suites exercise, so they
are reported as their own category.

Read ``covered`` as "this implementation was dispatched", not "asserted
correct" -- a test can dispatch a kernel and still be a weak test. Two known
blind spots, both of which under-report:

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
      --ipc=host --shm-size=16g --security-opt seccomp=unconfined \
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


def _install_probe(
    dispatched: dict[str, set[str]],
    candidates: dict[str, set[str]],
    requested: dict[str, int],
) -> None:
    """Record what the suites resolve, and separately what they dispatch.

    Two different questions, and conflating them overstates coverage:

    ``get_for_operator`` returns *every* candidate matching a family/mode. The
    upstream ``require`` fixture calls it only to decide whether to skip, before
    narrowing candidates by dtype, so a name appearing here means "a test looked
    at this operator", not "a test ran this kernel". Recorded as ``candidates``.

    ``get_impl(name)`` is the post-selection lookup: it is how a caller obtains
    the callable for the one kernel it is about to run, so a name appearing here
    is the implementation actually selected. Recorded as ``dispatched``, and it
    is what ``covered`` is computed from.

    The returned callable is deliberately *not* wrapped. Triton entry points are
    invoked as ``kernel[grid](...)``, and a ``functools.wraps`` function wrapper
    does not forward ``__getitem__`` -- instrumenting the call would break the
    very suites being measured. The lookup is the strongest signal available
    without changing behaviour.

    Both are patched on the class rather than the singleton, because tests may
    rebuild the registry and a per-instance patch would then stop recording
    silently.
    """
    from tokenspeed_kernel.registry import KernelRegistry

    original_for_operator = KernelRegistry.get_for_operator
    original_get_impl = KernelRegistry.get_impl

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
        if result is not None and name:
            dispatched[str(name)].add("get_impl")
        return result

    KernelRegistry.get_for_operator = probed_for_operator  # type: ignore[method-assign]
    KernelRegistry.get_impl = probed_get_impl  # type: ignore[method-assign]


def _run_one_suite(args: argparse.Namespace) -> int:
    """In-process: patch the registry, run one suite, emit a partial map."""
    import pytest

    suite = Path(args._single)
    dispatched: dict[str, set[str]] = defaultdict(set)
    candidates: dict[str, set[str]] = defaultdict(set)
    requested: dict[str, int] = {}

    inventory = _registry_inventory()
    _install_probe(dispatched, candidates, requested)

    argv = [str(suite), "-q", "--no-header", "-p", "no:cacheprovider"]
    argv += args.pytest_arg or []
    code = int(pytest.main(argv))

    partial = {
        "inventory": inventory,
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
            # `covered` is dispatch, not candidacy. `candidate_only` names the
            # difference explicitly: a test looked at the operator but the
            # implementation was never selected (usually filtered out by dtype
            # after the skip check), so counting it as covered would overstate
            # what the suites exercise.
            name: {
                **meta,
                "covered": name in dispatched,
                "candidate_only": name in candidates and name not in dispatched,
                "via": sorted(candidates.get(name, ())),
            }
            for name, meta in sorted(inventory.items())
        },
        "lookups": dict(sorted(lookups.items())),
        "suite_exit_codes": exit_codes,
    }
    n_cov = sum(1 for k in report["kernels"].values() if k["covered"])
    n_cand = sum(1 for k in report["kernels"].values() if k["candidate_only"])
    report["summary"] = {
        "kernels_total": len(inventory),
        "kernels_covered": n_cov,
        "kernels_candidate_only": n_cand,
        "kernels_uncovered": len(inventory) - n_cov - n_cand,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}")
    else:
        print(text)

    print(f"\nsummary: {report['summary']}")
    print("\ncovered kernels (implementation dispatched):")
    for name, meta in sorted(report["kernels"].items()):
        if meta["covered"]:
            print(f"  {meta['family']}.{meta['mode']:<26} {name}")
    print("\ncandidate-only kernels (operator reached, implementation not selected):")
    for name, meta in sorted(report["kernels"].items()):
        if meta["candidate_only"]:
            print(f"  {meta['family']}.{meta['mode']:<26} {name}")
    print("\nuncovered kernels:")
    for name, meta in sorted(report["kernels"].items()):
        if not meta["covered"] and not meta["candidate_only"]:
            print(f"  {meta['family']}.{meta['mode']:<26} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
