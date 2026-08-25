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
kernel from a skipped one is to watch the resolution happen. This wraps
``KernelRegistry.get_for_operator``, runs pytest in-process so the patch holds,
and records which kernel specs each lookup actually returned. An empty return is
the skip path, which is exactly the "not covered" signal we want.

Read ``covered`` as "resolved through the registry", not "untested". Two known
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
  64  usage / environment error -- registry not importable, or suites missing
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


def _install_probe(seen: dict[str, set[str]], requested: dict[str, int]) -> None:
    """Wrap get_for_operator so every resolution is recorded.

    Patched on the class rather than the singleton, because tests may rebuild
    the registry and a per-instance patch would then stop recording silently.
    """
    from tokenspeed_kernel.registry import KernelRegistry

    original = KernelRegistry.get_for_operator

    def probed(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)

        family = kwargs.get("family", args[0] if args else "?")
        mode = kwargs.get("mode", args[1] if len(args) > 1 else "?")
        solution = kwargs.get("solution", "?")
        key = f"{family}.{mode}::{solution}"
        requested[key] = requested.get(key, 0) + 1

        # A non-empty result means the lookup selected real kernels; empty is
        # the skip path. Record names so the join is by kernel identity rather
        # than by family, which would over-credit multi-solution families.
        specs = result if isinstance(result, (list, tuple, set)) else [result]
        for spec in specs:
            name = getattr(spec, "name", None)
            if name:
                seen[str(name)].add(key)

        return result

    KernelRegistry.get_for_operator = probed  # type: ignore[method-assign]


def _run_one_suite(args: argparse.Namespace) -> int:
    """In-process: patch the registry, run one suite, emit a partial map."""
    import pytest

    suite = Path(args._single)
    seen: dict[str, set[str]] = defaultdict(set)
    requested: dict[str, int] = {}

    inventory = _registry_inventory()
    _install_probe(seen, requested)

    argv = [str(suite), "-q", "--no-header", "-p", "no:cacheprovider"]
    argv += args.pytest_arg or []
    code = int(pytest.main(argv))

    partial = {
        "inventory": inventory,
        "seen": {name: sorted(keys) for name, keys in seen.items()},
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
    seen: dict[str, set[str]] = defaultdict(set)
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
            subprocess.run(cmd, cwd=_suite_cwd(workspace, path), check=False)

            if not part.exists():
                print(f"  no map produced for {label}", file=sys.stderr)
                exit_codes[label] = -1
                continue

            data = json.loads(part.read_text())
            inventory.update(data["inventory"])
            for name, keys in data["seen"].items():
                seen[name].update(keys)
            for key, count in data["lookups"].items():
                lookups[key] = lookups.get(key, 0) + count
            exit_codes[label] = data["exit_code"]

    if not inventory:
        print("no registry inventory collected; nothing to report", file=sys.stderr)
        return 64

    report: dict[str, Any] = {
        "kernels": {
            name: {**meta, "covered": name in seen, "via": sorted(seen.get(name, ()))}
            for name, meta in sorted(inventory.items())
        },
        "lookups": dict(sorted(lookups.items())),
        "suite_exit_codes": exit_codes,
    }
    n_cov = sum(1 for k in report["kernels"].values() if k["covered"])
    report["summary"] = {
        "kernels_total": len(inventory),
        "kernels_covered": n_cov,
        "kernels_uncovered": len(inventory) - n_cov,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}")
    else:
        print(text)

    print(f"\nsummary: {report['summary']}")
    print("\ncovered kernels:")
    for name, meta in sorted(report["kernels"].items()):
        if meta["covered"]:
            print(f"  {meta['family']}.{meta['mode']:<26} {name}")
    print("\nuncovered kernels:")
    for name, meta in sorted(report["kernels"].items()):
        if not meta["covered"]:
            print(f"  {meta['family']}.{meta['mode']:<26} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
