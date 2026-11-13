#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Step 1: find a planted bug on a CPU mesh with the Monarch debugger.

One machine, no Slurm, no GPU. The controller starts eight worker processes
on this host. Each worker holds a number, ``partial = rank + 1``, so the
honest total over eight workers is 36. Every worker stops at ``breakpoint()``.
Right after that line, ranks 2 and 6 set their partial to 0. That is the bug.
The controller prints each worker's partial and the total, which comes out
as 26.

While the workers wait, open a second terminal on this machine and run
``monarch debug``. From there you can list the stopped workers, attach to
one, print ``rank`` and ``partial``, step over the bad line, and continue.

Usage
-----

With the Python environment that has Monarch active (see
``docs/monarch-install.md``), from the aorta checkout:

    python examples/monarch-debug/cpu_mesh.py

Then, in a second terminal, activate the same environment and run:

    monarch debug
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from typing import Any

_NO_MONARCH = (
    f"Monarch cannot be imported with {sys.executable}. Run this with the Python "
    "of the environment that has Monarch; see docs/monarch-install.md."
)

try:
    from monarch.actor import Actor, current_rank, endpoint, this_host
except Exception as exc:  # a broken install fails with more than ImportError
    _NO_MONARCH += f" The import failed with {type(exc).__name__}: {exc}"

    class Actor:  # type: ignore[no-redef]
        """Stand-in so this module imports where Monarch is not installed."""

    def endpoint(method):  # type: ignore[no-redef]
        return method

    def _no_monarch(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(_NO_MONARCH)

    current_rank = this_host = _no_monarch
    _MONARCH_READY = False
else:
    _MONARCH_READY = True


def is_bug_rank(rank: int) -> bool:
    return rank % 4 == 2


class WorkerActor(Actor):
    @endpoint
    def partial_sum(self) -> dict[str, Any]:
        rank = current_rank().rank
        partial = rank + 1
        breakpoint()  # noqa: T100
        # One line, so a single `n` in pdb runs the bug.
        partial = 0 if is_bug_rank(rank) else partial
        return {
            "rank": rank,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "partial": partial,
        }


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="worker processes on this host; every rank with rank %% 4 == 2 gets the bug (default 8)",
    )
    return parser.parse_args(argv)


def _detach_stdin_from_terminal() -> None:
    """Point stdin at /dev/null, so the workers do not inherit this terminal.

    Each worker runs in its own background process group. On its first
    ``breakpoint()``, pdb imports ``readline``, which in a libedit-based Python
    calls ``tcsetattr`` on stdin. On a terminal, the kernel stops the worker
    with SIGTTOU before it reaches the debugger, and ``list`` stays empty.
    """
    if os.isatty(0):
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        os.close(devnull)


def _print_debug_hint(workers: int) -> None:
    port = os.environ.get("MONARCH_DEBUG_SERVER_PORT", "27000")
    print(
        f"{workers} workers will stop at breakpoint(). In a second terminal on this machine, "
        "activate the environment that has Monarch and run:\n"
        f"  monarch debug\n"
        f"It connects to localhost:{port}. Try: list, attach worker 2, p partial, n, "
        "p partial, continue.",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if not _MONARCH_READY:
        print(_NO_MONARCH, file=sys.stderr)
        return 2
    if args.workers < 1:
        print(f"--workers must be >= 1, got {args.workers}", file=sys.stderr)
        return 2

    honest = sum(rank + 1 for rank in range(args.workers))
    planted = [rank for rank in range(args.workers) if is_bug_rank(rank)]
    expected = honest - sum(rank + 1 for rank in planted)

    _detach_stdin_from_terminal()
    procs = this_host().spawn_procs(per_host={"workers": args.workers})
    workers = procs.spawn("worker", WorkerActor)
    _print_debug_hint(args.workers)
    results = workers.partial_sum.call().get()

    rows = sorted((row for _point, row in results.items()), key=lambda row: row["rank"])
    print(f"{'rank':>4}  {'host':<20} {'pid':>8}  partial")
    for row in rows:
        print(f"{row['rank']:>4}  {row['host']:<20} {row['pid']:>8}  {row['partial']}")
    total = sum(row["partial"] for row in rows)
    zeroed = [row["rank"] for row in rows if row["partial"] == 0]
    print(f"total {total}  honest total {honest}  expected with the bug {expected}")
    print(f"ranks with partial 0: {', '.join(map(str, zeroed)) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
