#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Steps 2 and 3: find a planted bug in a GPU all-reduce with the Monarch debugger.

The controller attaches to the Slurm or Spur allocation this shell already
holds and starts one Monarch worker on each of ``--nodes`` hosts. On every
host it spawns eight GPU processes, one per GPU, in an actor mesh named
``worker``. The flat rank is ``hosts * 8 + gpus``.

Each rank puts ``rank + 1`` in a one-element CUDA tensor and stops at
``breakpoint()``. Right after that line, gpu 3 on the last host sets its
tensor to 0. That is the bug. Then every rank joins one NCCL process group
and all-reduces the tensor.

``--nodes 1`` is step 2: eight GPUs, honest sum 36, sum with the bug 32,
and the bad rank is 3. ``--nodes 2`` is step 3: sixteen GPUs, honest sum
136, sum with the bug 124, and the bad rank is 11.

While the ranks wait, open a second terminal on this machine and run
``monarch debug``. The controller stays here, so the debugger talks to
localhost. Do not step one rank into the all-reduce while the others still
sit at ``breakpoint()``: it waits for them forever. Inspect the bad rank,
then ``continue`` the whole mesh.

Usage
-----

Hold an allocation that includes the GPUs, then from the aorta checkout:

    salloc -N1 --gpus-per-node=8
    bash examples/monarch-debug/run-slurm.sh 1

    salloc -N2 --gpus-per-node=8
    bash examples/monarch-debug/run-slurm.sh 2

    spur salloc -N2 --gpus-per-node=8 --exclusive
    bash examples/monarch-debug/run-spur.sh 2

The launchers run this file with ``$MONARCH_PYTHON`` (default ``python``),
the Python of the environment that has Monarch; see
``docs/monarch-install.md``. Run them from a shell on an allocated node.
``--nodes 1`` also works inside a two-node allocation; on Slurm it uses the
node this shell is on. Then, in a second terminal on the same machine,
activate that environment and run:

    monarch debug

This file does not request an allocation. ``--scheduler slurm`` (the
default) needs ``SLURM_JOB_ID`` or ``MONARCH_SLURM_JOB`` and attaches with
``srun --jobid --overlap``. ``--scheduler spur`` needs ``SPUR_JOB_ID``, which
``spur salloc`` sets, and runs ``spur srun`` from that shell. Spur cannot
cancel one step, so each worker stops when a file appears in
``DEBUG_DEMO_SHARED_DIR`` (or ``AORTA_SHARED_DIR``), a directory every node
can write. ``MONARCH_PORT`` (default 22222) is the worker port, and
``DEMO_MASTER_PORT`` (default 29531) is the NCCL rendezvous port on the
first host.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import IO, Any

_NO_MONARCH = (
    f"Monarch cannot be imported with {sys.executable}. Run this with the Python "
    "of the environment that has Monarch (set MONARCH_PYTHON for the launchers); "
    "see docs/monarch-install.md."
)

try:
    from monarch.actor import Actor, current_rank, current_size, endpoint
except Exception as exc:  # a broken install fails with more than ImportError
    _NO_MONARCH += f" The import failed with {type(exc).__name__}: {exc}"

    class Actor:  # type: ignore[no-redef]
        """Stand-in so this module imports where Monarch is not installed."""

    def endpoint(method):  # type: ignore[no-redef]
        return method

    def _no_monarch(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(_NO_MONARCH)

    current_rank = current_size = _no_monarch
    _MONARCH_READY = False
else:
    _MONARCH_READY = True

_HERE = Path(__file__).resolve().parent
_BUG_GPU = 3


def is_bug_rank(host: int, gpu: int, hosts: int) -> bool:
    return host == hosts - 1 and gpu == _BUG_GPU


class GpuWorker(Actor):
    @endpoint
    def reduce(self, master_addr: str, master_port: int) -> dict[str, Any]:
        import torch
        import torch.distributed as dist

        point = current_rank()
        sizes = current_size()
        hosts, gpus = sizes["hosts"], sizes["gpus"]
        host, gpu = point["hosts"], point["gpus"]
        rank = point.rank
        world = hosts * gpus
        if torch.cuda.device_count() <= gpu:
            raise RuntimeError(
                f"rank {rank} wants gpu {gpu} but {socket.gethostname()} shows "
                f"{torch.cuda.device_count()} GPUs. Hold the allocation with --gpus-per-node."
            )
        torch.cuda.set_device(gpu)
        partial = torch.tensor([rank + 1.0], device="cuda")
        breakpoint()  # noqa: T100
        # One line, so a single `n` in pdb runs the bug.
        partial = torch.zeros_like(partial) if is_bug_rank(host, gpu, hosts) else partial
        before = partial.item()
        # Join the group only after the breakpoint. A rank parked in pdb before
        # the rendezvous would make every other rank time out waiting for it.
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world,
            init_method=f"tcp://{master_addr}:{master_port}",
        )
        try:
            dist.all_reduce(partial)
            reduced = partial.item()
        finally:
            dist.destroy_process_group()
        return {
            "rank": rank,
            "host": host,
            "gpu": gpu,
            "hostname": socket.gethostname(),
            "device": torch.cuda.get_device_name(gpu),
            "partial": before,
            "reduced": reduced,
        }


# Spur cannot cancel one step. Each worker stops itself when this file appears.
_SPUR_WATCH = """\
set -m
"$@" &
child=$!
set +m
trap 'kill -TERM -- "-$child" 2>/dev/null || true' TERM INT HUP
cancel=${AORTA_STEP_CANCEL:-}
if [ -n "$cancel" ]; then
  (
    while [ ! -e "$cancel" ]; do
      sleep 0.2
    done
    kill -TERM -- "-$child" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$child" 2>/dev/null || true
  ) >/dev/null 2>&1 &
  watcher=$!
fi
status=0
wait "$child" || status=$?
if [ -n "${watcher:-}" ]; then
  kill "$watcher" 2>/dev/null || true
  wait "$watcher" 2>/dev/null || true
fi
exit "$status"
"""


def step_prefix(*, scheduler: str, nodes: int, spur_bin: str = "spur") -> list[str]:
    """``srun`` arguments for one task on each of ``nodes`` hosts.

    ``slurm`` attaches with ``srun --jobid --overlap``. ``spur`` runs ``spur
    srun`` inside the allocation that set ``SPUR_JOB_ID``. Neither form
    requests an allocation.
    """
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1:
        raise ValueError(f"nodes must be an int >= 1, got {nodes!r}")
    counts = [f"-N{nodes}", f"--ntasks={nodes}", "--ntasks-per-node=1"]
    if scheduler == "spur":
        if not os.environ.get("SPUR_JOB_ID", "").strip():
            raise RuntimeError(
                "scheduler 'spur' runs inside a Spur allocation. Start it with "
                "`spur salloc` so SPUR_JOB_ID is set. This script does not request one."
            )
        return [spur_bin, "srun", *counts]
    if scheduler != "slurm":
        raise ValueError(f"scheduler must be 'slurm' or 'spur', got {scheduler!r}")
    return ["srun", "--jobid", _slurm_job_id(), "--overlap", *counts]


def _slurm_job_id() -> str:
    explicit = os.environ.get("MONARCH_SLURM_JOB", "").strip()
    job_id = explicit or os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RuntimeError(
            "This attaches to an existing Slurm allocation. Set SLURM_JOB_ID "
            "or MONARCH_SLURM_JOB. This script does not call salloc or sbatch."
        )
    if not explicit and os.environ.get("SPUR_JOB_ID", "").strip() == job_id:
        raise RuntimeError(
            f"SLURM_JOB_ID {job_id} is a Spur allocation. Use --scheduler spur, "
            "or set MONARCH_SLURM_JOB to the Slurm job id."
        )
    return job_id


def _controller_ip() -> str:
    for key in ("AORTA_CONTROLLER_IP", "MONARCH_CONTROLLER_IP"):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                ipaddress.IPv4Address(raw)
            except ipaddress.AddressValueError as exc:
                raise RuntimeError(f"{key} must be an IPv4 address, got {raw!r}") from exc
            return raw
    try:
        ip = socket.gethostbyname(socket.getfqdn())
    except OSError as exc:
        raise RuntimeError(
            "Could not resolve this machine's IPv4 address. Set AORTA_CONTROLLER_IP "
            f"or MONARCH_CONTROLLER_IP. Lookup failed: {exc}"
        ) from exc
    try:
        address = ipaddress.IPv4Address(ip)
    except ipaddress.AddressValueError as exc:
        raise RuntimeError(
            f"Controller address {ip!r} is not IPv4. Set AORTA_CONTROLLER_IP or "
            "MONARCH_CONTROLLER_IP."
        ) from exc
    if address.is_loopback:
        raise RuntimeError(
            f"{socket.getfqdn()} resolves to the loopback address {ip}, which the "
            "workers cannot reach. Run this from a shell on an allocated node, or set "
            "AORTA_CONTROLLER_IP or MONARCH_CONTROLLER_IP to an address the nodes can reach."
        )
    return ip


def _port_from_env(key: str, default: str) -> int:
    raw = os.environ.get(key, default)
    if not raw.isdigit():
        raise ValueError(f"{key} must be an integer, got {raw!r}")
    port = int(raw)
    if not 1 <= port <= 65535:
        raise ValueError(f"{key} must be in 1..65535, got {port}")
    return port


def _probe_nodes(prefix: list[str], nodes: int, scheduler: str = "slurm") -> list[tuple[str, str]]:
    """``(node name, hostname)`` for each host, in ``SLURM_NODEID`` order.

    The node name is what ``-w`` takes: ``SLURMD_NODENAME`` on Slurm. A
    ``spur srun`` step sets ``SLURM_NODEID`` but passes the caller's
    ``SLURMD_NODENAME`` through unchanged, so Spur uses ``hostname -s``, which
    is how Spur names these nodes.
    """
    node_name = "$(hostname -s)" if scheduler == "spur" else "${SLURMD_NODENAME:-}"
    script = (
        f'echo "$SLURM_NODEID {node_name} '
        '$(python3 -c "import socket; print(socket.gethostname())")"'
    )
    try:
        out = subprocess.check_output([*prefix, "bash", "-c", script], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not list hosts with {' '.join(prefix)}: {exc}") from exc
    rows: list[tuple[int, str, str]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3 or not parts[0].isdigit() or "," in parts[1]:
            continue
        rows.append((int(parts[0]), parts[1], parts[2]))
    rows.sort()
    if len(rows) != nodes or len({node for _, node, _ in rows}) != nodes:
        raise RuntimeError(f"expected {nodes} nodes from the allocation, got {len(rows)}: {out!r}")
    return [(node, host) for _, node, host in rows]


def _listing(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True).stdout
    except OSError as exc:
        raise RuntimeError(f"Could not run {argv[0]}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip() or f"exit status {exc.returncode}"
        raise RuntimeError(f"`{' '.join(argv)}` failed: {detail}") from exc


def _allocation_nodes(job_id: str) -> list[str]:
    """Node names of Slurm job ``job_id``, in allocation order."""
    listed = ",".join(_listing(["squeue", "-j", job_id, "-h", "-o", "%N"]).split())
    if not listed:
        raise RuntimeError(
            f"Slurm job {job_id} has no nodes. Check that it is still running: squeue -u $USER"
        )
    names = _listing(["scontrol", "show", "hostnames", listed]).split()
    if not names:
        raise RuntimeError(f"scontrol could not expand the node list {listed!r} of job {job_id}")
    return names


def _choose_nodes(allocated: list[str], nodes: int, here: str) -> tuple[list[str], list[str]]:
    """``nodes`` of ``allocated`` to run on, this machine first, and the ones left out."""
    if nodes > len(allocated):
        raise ValueError(
            f"--nodes {nodes} needs {nodes} nodes, but the allocation has "
            f"{len(allocated)}: {', '.join(allocated)}"
        )
    order = [here] if here in allocated else []
    order += [node for node in allocated if node != here]
    chosen = order[:nodes]
    return chosen, [node for node in allocated if node not in chosen]


def _place(
    prefix: list[str], scheduler: str, nodes: int
) -> tuple[list[str], list[tuple[str, str]]]:
    """``prefix`` pinned to the hosts every step uses, and those hosts.

    Slurm 25.05 does not always honor ``-w`` alone for a step: in testing, an
    ``-N1 -w <node>`` step ran on the other node about half the time. Adding
    ``-x`` for the rest of the allocation pins it, so the probe and the
    workers get the same ``-w`` and ``-x``.
    """
    if scheduler == "spur":
        placed = _probe_nodes(prefix, nodes, scheduler)
        return [*prefix, "-w", ",".join(node for node, _host in placed)], placed
    allocated = _allocation_nodes(_slurm_job_id())
    chosen, others = _choose_nodes(allocated, nodes, socket.gethostname().split(".")[0])
    pinned = [*prefix, "-w", ",".join(chosen)]
    if others:
        pinned += ["-x", ",".join(others)]
    placed = _probe_nodes(pinned, nodes)
    seen = sorted(node for node, _host in placed)
    if seen != sorted(chosen):
        raise RuntimeError(
            f"asked Slurm for {', '.join(chosen)}, but the step ran on {', '.join(seen)}"
        )
    return pinned, placed


def _worker_argv(scheduler: str, pinned: list[str], worker: list[str]) -> list[str]:
    flags: list[str] = []
    if scheduler == "slurm":
        flags = ["--kill-on-bad-exit=1", f"--job-name=debug-demo-{os.getpid()}"]
    return [*pinned, *flags, *worker]


def _worker_env() -> dict[str, str]:
    """This directory on ``PYTHONPATH``, so a worker can import ``gpu_reduce``."""
    extra = [str(_HERE)]
    current = os.environ.get("PYTHONPATH", "")
    if current:
        extra.append(current)
    return {"PYTHONPATH": os.pathsep.join(extra)}


def _cancel_path() -> Path:
    key = "DEBUG_DEMO_SHARED_DIR"
    raw = os.environ.get(key, "").strip()
    if not raw:
        key = "AORTA_SHARED_DIR"
        raw = os.environ.get(key, "").strip()
    if not raw:
        raise RuntimeError(
            "scheduler 'spur' needs DEBUG_DEMO_SHARED_DIR (or AORTA_SHARED_DIR) set "
            "to a directory every node can read and write. Spur cannot cancel one "
            "step, so a worker stops when a file appears there. /tmp on one node "
            "is not enough."
        )
    root = Path(raw)
    if not root.is_dir():
        raise RuntimeError(f"{key} {raw!r} is not a directory")
    return root / f"debug-demo-step-cancel-{os.getpid()}"


def _signal_group(proc: subprocess.Popen[str], pgid: int | None, sig: int) -> None:
    """Send ``sig`` to the worker step's process group, or to the step alone without one."""
    if pgid is None or pgid == os.getpgrp():
        proc.send_signal(sig)
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except OSError as exc:
        print(f"could not signal process group {pgid}: {exc}", file=sys.stderr, flush=True)
        proc.send_signal(sig)


def _stop_workers(proc: subprocess.Popen[str], pgid: int | None, cancel: Path | None) -> None:
    if cancel is not None:
        cancel.parent.mkdir(parents=True, exist_ok=True)
        cancel.touch()
    _signal_group(proc, pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _signal_group(proc, pgid, signal.SIGKILL)
        proc.wait(timeout=5)


def _open_logs(stdout_path: Path, stderr_path: Path) -> tuple[IO[str], IO[str]]:
    """Open both worker logs for writing, or neither."""
    stdout_log = stdout_path.open("w", encoding="utf-8")
    try:
        return stdout_log, stderr_path.open("w", encoding="utf-8")
    except BaseException:
        stdout_log.close()
        raise


def _print_debug_hint(world: int, bug_rank: int | None) -> None:
    port = os.environ.get("MONARCH_DEBUG_SERVER_PORT", "27000")
    target = bug_rank if bug_rank is not None else 0
    print(
        f"{world} ranks will stop at breakpoint(). In a second terminal on this machine, "
        "activate the environment that has Monarch and run:\n"
        f"  monarch debug\n"
        f"It connects to localhost:{port}. Try: list, attach worker {target}, p rank, "
        "p partial, n, p partial, then continue.\n"
        "Do not step one rank into all_reduce while the others wait at breakpoint(); "
        "it hangs until they arrive. Use continue for the whole mesh.",
        flush=True,
    )


async def _reduce_on_mesh(
    hosts: Any,
    nodes: int,
    gpus_per_node: int,
    master_addr: str,
    master_port: int,
    bug_rank: int | None,
) -> list[dict[str, Any]]:
    await hosts.initialized
    procs = hosts.spawn_procs(name="gpus", per_host={"gpus": gpus_per_node})
    workers = procs.spawn("worker", GpuWorker)
    _print_debug_hint(nodes * gpus_per_node, bug_rank)
    results = await workers.reduce.call(master_addr, master_port)
    return sorted((row for _point, row in results.items()), key=lambda row: row["rank"])


def run(
    nodes: int, gpus_per_node: int, scheduler: str = "slurm", spur_bin: str = "spur"
) -> list[dict[str, Any]]:
    """Attach to the current Slurm or Spur allocation and run the all-reduce on it."""
    from monarch._src.job._slurm_batch import _worker_bootstrap
    from monarch._src.job.service_identity import (
        SERVICE_PROC_IDS_ENV,
        allocate_service_proc_ids,
        serialize_service_proc_ids,
        service_proc_addrs,
    )
    from monarch.actor import attach_to_workers, enable_transport

    if gpus_per_node < 1:
        raise ValueError(f"--gpus-per-node must be >= 1, got {gpus_per_node}")
    prefix = step_prefix(scheduler=scheduler, nodes=nodes, spur_bin=spur_bin)
    cancel = _cancel_path() if scheduler == "spur" else None
    ip = _controller_ip()
    port = _port_from_env("MONARCH_PORT", "22222")
    master_port = _port_from_env("DEMO_MASTER_PORT", "29531")
    pinned, placed = _place(prefix, scheduler, nodes)
    enable_transport(f"tcp://{ip}:0")
    hostnames = [host for _node, host in placed]
    master_addr = hostnames[0]
    bug_rank = (nodes - 1) * gpus_per_node + _BUG_GPU if gpus_per_node > _BUG_GPU else None
    print(
        f"controller {ip}  port {port}  hosts {', '.join(hostnames)}  "
        f"nccl master {master_addr}:{master_port}",
        flush=True,
    )
    proc_ids = allocate_service_proc_ids(nodes)
    env = os.environ.copy()
    env.update(_worker_env())
    env[SERVICE_PROC_IDS_ENV] = serialize_service_proc_ids(proc_ids)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(key, None)
    bootstrap = _worker_bootstrap(port, None)
    inner = [sys.executable, "-c", bootstrap]
    if cancel is not None:
        env["AORTA_STEP_CANCEL"] = str(cancel)
        worker = ["bash", "-c", _SPUR_WATCH, "debug-demo", *inner]
    else:
        worker = inner
    argv = _worker_argv(scheduler, pinned, worker)
    log_dir = Path(f"/tmp/{os.environ.get('USER', 'debug-demo')}/debug-demo")
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "gpu-workers.out"
    stderr_path = log_dir / "gpu-workers.err"
    print(f"worker logs {stdout_path} {stderr_path}", flush=True)
    stdout_log, stderr_log = _open_logs(stdout_path, stderr_path)
    workers: subprocess.Popen[str] | None = None
    pgid: int | None = None
    mesh_ok = False
    try:
        workers = subprocess.Popen(
            argv,
            env=env,
            start_new_session=True,
            stdout=stdout_log,
            stderr=stderr_log,
        )
        try:
            pgid = os.getpgid(workers.pid)
        except OSError:
            pgid = None
        step = workers

        # Cancelling a pending Monarch call does not end it, so asyncio's own
        # Ctrl-C handling never returns while ranks are stuck in NCCL setup.
        def _stop_and_exit(signum: int, _frame: Any) -> None:
            print(f"\nsignal {signum}: stopping the worker step", file=sys.stderr, flush=True)
            _stop_workers(step, pgid, cancel)
            os._exit(128 + signum)

        signal.signal(signal.SIGINT, _stop_and_exit)
        signal.signal(signal.SIGTERM, _stop_and_exit)
        hosts = attach_to_workers(
            name="debug-demo",
            ca="trust_all_connections",
            workers=service_proc_addrs([f"tcp://{host}:{port}" for host in hostnames], proc_ids),
        )

        async def _run_and_close() -> list[dict[str, Any]]:
            nonlocal mesh_ok
            try:
                return await _reduce_on_mesh(
                    hosts, nodes, gpus_per_node, master_addr, master_port, bug_rank
                )
            finally:
                try:
                    await hosts.shutdown()
                    mesh_ok = True
                except Exception as exc:
                    print(f"monarch host shutdown failed: {exc}", file=sys.stderr)

        return asyncio.run(_run_and_close())
    finally:
        # On Spur the remote workers can outlive the local `spur srun` client.
        if workers is not None and (cancel is not None or workers.poll() is None):
            _stop_workers(workers, pgid, cancel)
        stdout_log.close()
        stderr_log.close()
        if cancel is not None and mesh_ok:
            cancel.unlink(missing_ok=True)


def _print_report(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'rank':>4}  {'host':>4}  {'gpu':>3}  {'hostname':<32} {'device':<28} "
        f"{'partial':>8}  {'reduced':>8}"
    )
    for row in rows:
        print(
            f"{row['rank']:>4}  {row['host']:>4}  {row['gpu']:>3}  {row['hostname']:<32} "
            f"{row['device']:<28} {row['partial']:>8g}  {row['reduced']:>8g}"
        )
    world = len(rows)
    reduced = {row["reduced"] for row in rows}
    honest = world * (world + 1) // 2
    zeroed = [row["rank"] for row in rows if row["partial"] == 0]
    shown = ", ".join(f"{value:g}" for value in sorted(reduced))
    print(f"reduced sum {shown}  honest sum {honest}")
    print(f"ranks with partial 0: {', '.join(map(str, zeroed)) or 'none'}")


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scheduler", choices=("slurm", "spur"), default="slurm")
    parser.add_argument("--nodes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--spur-bin", default="spur")
    return parser.parse_args(argv)


def _reexec_as_module(argv: list[str]) -> None:
    """Start this file again under its real module name.

    Running ``python gpu_reduce.py`` names the file ``__main__``. The worker
    processes import it as ``gpu_reduce`` and cannot rebuild a class that was
    saved under the name ``__main__``.
    """
    env = os.environ.copy()
    env.update(_worker_env())
    code = "import gpu_reduce, sys; raise SystemExit(gpu_reduce.main(sys.argv[1:]))"
    os.execvpe(sys.executable, [sys.executable, "-c", code, *argv], env)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if not _MONARCH_READY:
        print(_NO_MONARCH, file=sys.stderr)
        return 2
    if __name__ == "__main__":
        _reexec_as_module(list(argv) if argv is not None else sys.argv[1:])
    try:
        rows = run(args.nodes, args.gpus_per_node, args.scheduler, args.spur_bin)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    _print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
