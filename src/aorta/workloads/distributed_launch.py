"""Spur and Monarch launchers for the tiny-transformer workload.

Neither launcher is an install dependency. Spur is a subprocess of
``spur_bin`` (default ``spur``). Monarch is imported only when
:func:`launch_monarch` runs.

The spur controller runs one worker, :func:`run_worker_rank`, with the
already resolved training config. It does not replay the parent ``aorta``
command: a triage matrix would otherwise train every cell inside each
launch, and a library caller would otherwise keep only ``steps``. Each
worker maps ``SLURM_PROCID`` to ``RANK`` and ``SLURM_LOCALID`` to
``LOCAL_RANK`` (``RANK % gpus`` is only the block-placement fallback).
``MASTER_ADDR`` is rank 0's IPv4, because that rank hosts the rendezvous
store. Rank 0 writes ``rank0.json`` under ``AORTA_SHARED_DIR``, a path
every node can see.
The controller's verdict passes only when the worker exit code is 0 and
that rank-0 result passed.

The monarch controller follows this sequence: TCP transport on the
controller IPv4, one worker per node, a proc mesh, and
``setup_torch_elastic_env_async``. The workers start as one step of the
current allocation: ``srun --jobid <id> --overlap`` under Slurm, or
``spur srun`` from inside a Spur allocation (``scheduler: spur``). The actor
calls :func:`run_training_rank` instead of an inlined model. A missing
``monarch`` import raises an error that says Monarch is not installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aorta.workloads._base import WorkloadResult

log = logging.getLogger(__name__)

# Grace before SIGKILL, then a short reap. SIGKILL does not need the SIGTERM
# budget; a second Ctrl-C during the grace still force-kills and re-raises.
_TERMINATE_GRACE_SEC = 10.0
_REAP_AFTER_KILL_SEC = 1.0

_IDENTITY_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
)

# Worker-side mapping. spur sets SLURM_PROCID per task; RANK has to be set
# before the training process starts because the loop reads it to pick a device
# and to decide who publishes rank0.json.
SPUR_WORKER_SCRIPT = """\
set -euo pipefail
if [ -z "${SLURM_PROCID:-}" ]; then
  echo "spur worker: SLURM_PROCID is required to set RANK" >&2
  exit 2
fi
export RANK="$SLURM_PROCID"
gpus="${AORTA_GPUS_PER_NODE:?AORTA_GPUS_PER_NODE is required}"
# SLURM_LOCALID is this task's index on the node. RANK % gpus matches that
# only for block placement. Cyclic placement puts global ranks 0 and 2 on
# one node, and both would select GPU 0.
if [ -n "${SLURM_LOCALID:-}" ]; then
  case "$SLURM_LOCALID" in
    *[!0-9]*)
      echo "spur worker: SLURM_LOCALID must be an integer, got ${SLURM_LOCALID}" >&2
      exit 2
      ;;
  esac
  export LOCAL_RANK="$SLURM_LOCALID"
else
  export LOCAL_RANK=$(( RANK % gpus ))
fi
: "${WORLD_SIZE:?WORLD_SIZE is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${AORTA_INNER:?AORTA_INNER is required}"
exec "$@"
"""

_MONARCH_MISSING = (
    "Monarch is not installed. launcher 'monarch' needs a pytorch_monarch "
    "build that imports on this host (published torchmonarch wheels are "
    "CUDA-only); import failed: {exc!r}"
)

# Launcher knobs. They never reach the training loop.
_CONTROLLER_CONFIG_KEYS = frozenset(
    {"launcher", "nodes", "gpus_per_node", "spur_bin", "scheduler"}
)

# One rank, one config. Importing the function keeps the worker from
# rebuilding ``sys.argv`` (a triage parent would rerun every cell).
_WORKER_BOOTSTRAP = """\
import json
import os

from aorta.workloads.distributed_launch import run_worker_rank

with open(os.environ["AORTA_WORKER_CONFIG"], encoding="utf-8") as handle:
    loaded = json.load(handle)
if not isinstance(loaded, dict):
    raise SystemExit("AORTA_WORKER_CONFIG must contain a JSON object")
run_worker_rank(loaded)
"""

_LOCAL_RESULT_PREFIXES = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))


def is_ipv4(value: object) -> bool:
    """True for a dotted decimal IPv4 address without leading zeros."""
    if not isinstance(value, str):
        return False
    parts = value.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        if part != "0" and part.startswith("0"):
            return False
        if int(part) > 255:
            return False
    return True


def controller_ipv4() -> str:
    """IPv4 address workers should dial.

    ``AORTA_CONTROLLER_IP`` wins, then ``MONARCH_CONTROLLER_IP``. Otherwise
    the host's own name is resolved with ``gethostbyname``, which returns
    IPv4. A non-IPv4 result is an error: link-local IPv6 is not usable as
    ``MASTER_ADDR`` here.
    """
    for key in ("AORTA_CONTROLLER_IP", "MONARCH_CONTROLLER_IP"):
        raw = os.environ.get(key)
        if raw:
            if not is_ipv4(raw):
                raise RuntimeError(f"{key} must be an IPv4 address, got {raw!r}")
            return raw
    try:
        ip = socket.gethostbyname(socket.getfqdn())
    except OSError as exc:
        raise RuntimeError(
            "Could not resolve the controller IPv4 address. "
            f"Set AORTA_CONTROLLER_IP. Hostname lookup failed: {exc}"
        ) from exc
    if not is_ipv4(ip):
        raise RuntimeError(
            f"Controller address {ip!r} is not IPv4. Set AORTA_CONTROLLER_IP "
            "to the controller's IPv4 address."
        )
    return ip


def is_launched_rank() -> bool:
    """True when this process should train instead of spawning workers.

    Spur workers export ``AORTA_INNER=1``. Torchrun and Monarch's elastic
    agent set ``TORCHELASTIC_RUN_ID`` together with ``RANK`` and
    ``WORLD_SIZE``. A controller that only inherited ``RANK`` and
    ``WORLD_SIZE`` from the shell still spawns.
    """
    if os.environ.get("AORTA_INNER") == "1":
        return True
    if not os.environ.get("TORCHELASTIC_RUN_ID"):
        return False
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def spawning_controller(workload: str, *configs: Mapping[str, Any] | None) -> bool:
    """True when this process launches workers and must keep the trial output.

    A shell can inherit ``RANK`` from an earlier allocation. Spur and Monarch
    still launch from that process. The dispatcher and the triage runner
    otherwise treat a nonzero ``RANK`` as a peer and discard the result.
    An in-process rank, including one under torchrun or ``AORTA_INNER``,
    stays on the rank-0 write gate.
    """
    if workload != "tiny_transformer" or is_launched_rank():
        return False
    for config in configs:
        if isinstance(config, dict) and config.get("launcher") in {"spur", "monarch"}:
            return True
    return False


def run_training_rank(config: dict[str, Any]) -> dict[str, Any]:
    """Run one training rank and return its :class:`WorkloadResult` as a dict.

    Shared by the in-process path, spur workers, and the Monarch actor.
    Cleanup still runs when setup or run raises, but that path does not
    enter ``dist.barrier``. A cleanup error does not replace the original
    exception. When the run itself finished, a cleanup error is stored on
    ``metrics["cleanup_error"]``. ``KeyboardInterrupt`` is not caught.
    """
    if not isinstance(config, dict):
        raise TypeError(f"training config must be a dict, got {type(config).__name__}")
    # Imported here so a controller that only classifies a launcher does not
    # load PyTorch. ``run_trials`` imports this module for every workload.
    from aorta.workloads.training import TrainingWorkload

    workload = TrainingWorkload(config)
    cleanup_error: str | None = None
    finished = False
    try:
        workload.setup()
        result = workload.run()
        finished = True
    finally:
        try:
            workload.cleanup(collective=finished)
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
            log.warning("training rank cleanup failed", exc_info=True)
    payload = asdict(result)
    if cleanup_error is not None:
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            metrics["cleanup_error"] = cleanup_error
        else:
            payload["metrics"] = {"cleanup_error": cleanup_error}
    return payload


def worker_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Training config for one worker, without controller-only keys.

    ``launcher``, mesh shape, and ``spur_bin`` tell the controller how to
    spawn. Keys starting with ``_aorta_`` are controller metadata. Everything
    else, including model, optimizer, device, and checks, is the job the
    caller asked for.
    """
    if not isinstance(config, dict):
        raise TypeError(f"training config must be a dict, got {type(config).__name__}")
    kept: dict[str, Any] = {}
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        if key in _CONTROLLER_CONFIG_KEYS or key.startswith("_aorta_"):
            continue
        kept[key] = value
    return kept


def resolve_rank0_ipv4(*, spur_bin: str, nodes: int) -> str:
    """IPv4 of the worker that will be ``RANK=0``.

    That rank starts the rendezvous store, so ``MASTER_ADDR`` has to be its
    address. ``AORTA_RANK0_IP`` wins. Otherwise one ``spur srun`` prints each
    node's address and the controller keeps ``SLURM_PROCID`` 0. The
    controller's own address is not a substitute: the controller may be on
    another node.
    """
    raw = os.environ.get("AORTA_RANK0_IP", "").strip()
    if raw:
        if not is_ipv4(raw):
            raise RuntimeError(f"AORTA_RANK0_IP must be an IPv4 address, got {raw!r}")
        return raw
    script = (
        "set -euo pipefail\n"
        "ip=$(python3 -c 'import socket; print(socket.gethostbyname(socket.getfqdn()))')\n"
        'printf "%s %s\\n" "$SLURM_PROCID" "$ip"\n'
    )
    argv = [
        spur_bin,
        "srun",
        "-N",
        str(nodes),
        "--ntasks",
        str(nodes),
        "--ntasks-per-node",
        "1",
        "bash",
        "-c",
        script,
    ]
    try:
        out = subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not resolve rank 0's IPv4 via spur srun. Set AORTA_RANK0_IP "
            "to the IPv4 of the node that runs RANK=0. "
            f"Probe failed: {exc}"
        ) from exc
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] != "0":
            continue
        if not is_ipv4(parts[1]):
            raise RuntimeError(f"Rank 0 address {parts[1]!r} is not IPv4. Set AORTA_RANK0_IP.")
        return parts[1]
    raise RuntimeError(
        "spur srun did not report an IPv4 address for SLURM_PROCID 0. "
        f"Set AORTA_RANK0_IP. Probe output: {out!r}"
    )


def run_worker_rank(config: dict[str, Any]) -> None:
    """Train this rank and, on rank 0, publish the result where the controller reads it.

    Every rank trains. Only rank 0 writes ``rank0.json``. The file is replaced
    into ``AORTA_RESULTS_DIR`` after an fsync so a controller on another host
    sees the finished object, not a partial write.
    """
    result = run_training_rank(worker_training_config(config))
    if os.environ.get("RANK") != "0":
        return
    raw_dir = os.environ.get("AORTA_RESULTS_DIR", "").strip()
    if not raw_dir:
        raise RuntimeError("rank 0 needs AORTA_RESULTS_DIR to publish its result")
    dest = Path(raw_dir) / "rank0.json"
    blob = json.dumps({"result": result})
    temporary = dest.with_suffix(".json.tmp")
    temporary.write_text(blob, encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(dest)


def build_spur_launch(
    *,
    spur_bin: str,
    nodes: int,
    gpus_per_node: int,
    master_addr: str,
    master_port: int,
    config_path: Path,
    results_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    """Argv and env for one ``spur srun`` of :func:`run_worker_rank`."""
    if not is_ipv4(master_addr):
        raise ValueError(f"MASTER_ADDR must be an IPv4 address, got {master_addr!r}")
    if not isinstance(master_port, int) or isinstance(master_port, bool):
        raise TypeError(f"master_port must be an int, got {type(master_port).__name__}")
    argv = [
        spur_bin,
        "srun",
        "-N",
        str(nodes),
        "--ntasks-per-node",
        str(gpus_per_node),
        "--ntasks",
        str(nodes * gpus_per_node),
        "bash",
        "-c",
        SPUR_WORKER_SCRIPT,
        "aorta-inner",
        sys.executable,
        "-c",
        _WORKER_BOOTSTRAP,
    ]
    env = _worker_env(
        master_addr=master_addr,
        master_port=master_port,
        world_size=nodes * gpus_per_node,
        gpus_per_node=gpus_per_node,
    )
    env["AORTA_WORKER_CONFIG"] = str(config_path)
    env["AORTA_RESULTS_DIR"] = str(results_dir)
    return argv, env


def _open_subprocess_logs(config: Mapping[str, Any]) -> tuple[Any, Any]:
    """Open sibling logs for a launcher child, or ``(None, None)``.

    The dispatcher already holds ``<prefix>.stdout.log`` and
    ``<prefix>.stderr.log`` for the Python process. A child inherits the
    OS streams, so those files miss its output. The sibling files are
    ``<prefix>.subprocess.stdout.log`` and ``<prefix>.subprocess.stderr.log``.
    Callers keep the handles open until the child and its process group
    are gone, then close them.
    """
    if not config.get("_aorta_save_logs"):
        return None, None
    prefix = config.get("_aorta_log_prefix")
    if not isinstance(prefix, str) or not prefix:
        return None, None
    opened: list[Any] = []
    try:
        for suffix in ("stdout", "stderr"):
            opened.append(
                open(
                    f"{prefix}.subprocess.{suffix}.log",
                    "w",
                    encoding="utf-8",
                    errors="backslashreplace",
                )
            )
    except OSError as exc:
        for handle in opened:
            handle.close()
        log.warning("could not open subprocess logs under %s: %s", prefix, exc)
        return None, None
    return opened[0], opened[1]


def _close_subprocess_logs(*handles: Any) -> None:
    """Close launcher log files. A close error is logged and does not replace the result."""
    for handle in handles:
        if handle is None:
            continue
        try:
            handle.close()
        except OSError as exc:
            log.warning("could not close subprocess log: %s", exc)


def launch_spur(
    config: dict[str, Any],
    *,
    nodes: int,
    gpus_per_node: int,
    spur_bin: str,
) -> WorkloadResult:
    """Spawn ``spur srun`` and return rank 0's verdict gated on the exit code.

    The worker command is one rank of the resolved config. ``MASTER_ADDR`` is
    rank 0's IPv4. The result file lives under ``AORTA_SHARED_DIR``.
    """
    raw_port = os.environ.get("AORTA_MASTER_PORT")
    if raw_port is None or raw_port == "":
        raise RuntimeError(
            "launcher 'spur' needs AORTA_MASTER_PORT set to a free TCP port. "
            "An ephemeral port cannot be held until the workers bind it."
        )
    master_port = _tcp_port(raw_port, name="AORTA_MASTER_PORT")
    master_addr = resolve_rank0_ipv4(spur_bin=spur_bin, nodes=nodes)
    controller = os.environ.get("AORTA_CONTROLLER_IP", "").strip()
    if controller and controller != master_addr:
        log.info(
            "MASTER_ADDR is rank 0 at %s; AORTA_CONTROLLER_IP %s is the controller",
            master_addr,
            controller,
        )
    work = _shared_workdir()
    stdout_log, stderr_log = _open_subprocess_logs(config)
    try:
        payload = worker_training_config(config)
        config_path = work / "worker_config.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        argv, env = build_spur_launch(
            spur_bin=spur_bin,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            master_addr=master_addr,
            master_port=master_port,
            config_path=config_path,
            results_dir=work,
        )
        log.info(
            "spur launch MASTER_ADDR=%s WORLD_SIZE=%s argv=%s",
            env.get("MASTER_ADDR"),
            env.get("WORLD_SIZE"),
            argv,
        )
        proc: subprocess.Popen[str] | None = None
        pgid: int | None = None
        try:
            proc = subprocess.Popen(
                argv,
                env=env,
                start_new_session=True,
                stdout=stdout_log,
                stderr=stderr_log,
            )
            pgid = _session_pgid(proc)
            proc.wait()
        except KeyboardInterrupt:
            raise
        finally:
            if proc is not None and _needs_group_cleanup(proc, pgid):
                _terminate_process_tree(proc, pgid=pgid)
        exit_code = proc.returncode if proc is not None and isinstance(proc.returncode, int) else 1
        rank0, read_error = _read_rank0_result(work)
        result = verdict_from_workers(exit_code, rank0, launcher="spur")
        if read_error is not None:
            result.failure_details.append(
                {"problems": ["worker_result_unreadable"], "error": read_error}
            )
            result.passed = False
            result.failure_count += 1
        return result
    finally:
        _close_subprocess_logs(stdout_log, stderr_log)
        shutil.rmtree(work, ignore_errors=True)


def monarch_step_prefix(*, scheduler: str, nodes: int, spur_bin: str = "spur") -> list[str]:
    """``srun`` arguments for one task on each of ``nodes`` hosts of the current allocation.

    ``slurm`` attaches to ``MONARCH_SLURM_JOB`` or ``SLURM_JOB_ID`` with
    ``srun --jobid <id> --overlap``. ``spur`` runs ``<spur_bin> srun`` from a
    shell inside a Spur allocation (``SPUR_JOB_ID`` set), where it becomes a
    step of that allocation. Spur's ``--jobid --overlap`` would start one task
    on one node, so it is not used. ``--ntasks`` is explicit in both: a step
    otherwise inherits the allocation's task count.
    """
    counts = [f"-N{nodes}", f"--ntasks={nodes}", "--ntasks-per-node=1"]
    if scheduler == "spur":
        spur_job = os.environ.get("SPUR_JOB_ID", "").strip()
        if not spur_job:
            raise RuntimeError(
                "scheduler 'spur' runs Monarch inside a Spur allocation. Run aorta from "
                "the shell that `spur salloc` opens, so SPUR_JOB_ID is set. "
                "This workload does not request the allocation."
            )
        return [spur_bin, "srun", *counts]
    if scheduler != "slurm":
        raise ValueError(f"scheduler must be 'slurm' or 'spur', got {scheduler!r}")
    explicit = os.environ.get("MONARCH_SLURM_JOB", "").strip()
    job_id = explicit or os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RuntimeError(
            "launcher 'monarch' attaches to an existing allocation. "
            "Set SLURM_JOB_ID or MONARCH_SLURM_JOB. This workload does not submit Slurm jobs."
        )
    if not explicit and os.environ.get("SPUR_JOB_ID", "").strip() == job_id:
        raise RuntimeError(
            f"SLURM_JOB_ID {job_id} is a Spur allocation, not a Slurm one. "
            "Set scheduler: spur in the recipe, or set MONARCH_SLURM_JOB to the Slurm job id."
        )
    return ["srun", "--jobid", job_id, "--overlap", *counts]


def launch_monarch(
    config: dict[str, Any],
    *,
    nodes: int,
    gpus_per_node: int,
    scheduler: str = "slurm",
    spur_bin: str = "spur",
) -> WorkloadResult:
    """Run the training actor under a Monarch proc mesh on an existing allocation.

    Does not request an allocation. With ``scheduler="slurm"``,
    ``SLURM_JOB_ID`` or ``MONARCH_SLURM_JOB`` names the Slurm allocation. With
    ``scheduler="spur"``, the controller runs inside a Spur allocation and the
    workers are ``spur srun`` steps of it.
    """
    imported = _import_monarch()
    actor_mod, spmd_mod, bootstrap, service_env, alloc_ids, serialize_ids, proc_addrs = imported
    # Import after Monarch itself imports, so a missing install still raises
    # the not-installed error. The class lives in its own module so workers
    # unpickle ``aorta.workloads.monarch_trainer.Trainer``.
    from aorta.workloads.monarch_trainer import Trainer

    prefix = monarch_step_prefix(scheduler=scheduler, nodes=nodes, spur_bin=spur_bin)
    ip = controller_ipv4()
    actor_mod.enable_transport(f"tcp://{ip}:0")
    port = _tcp_port(os.environ.get("MONARCH_PORT", "22222"), name="MONARCH_PORT")
    hostnames = _monarch_hostnames(prefix, nodes)
    proc_ids = alloc_ids(nodes)
    env = {**os.environ, service_env: serialize_ids(proc_ids)}
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(key, None)
    worker_cmd = [sys.executable, "-c", bootstrap(port, None)]
    # Spur parses --kill-on-bad-exit but ignores it: its step supervisor always
    # stops the whole step when one task exits abnormally.
    kill_flag = ["--kill-on-bad-exit=1"] if scheduler == "slurm" else []
    argv = [*prefix, *kill_flag, *worker_cmd]
    log.info("monarch controller tcp://%s scheduler=%s hosts=%s", ip, scheduler, hostnames)
    workers: subprocess.Popen[str] | None = None
    worker_pgid: int | None = None
    stdout_log, stderr_log = _open_subprocess_logs(config)
    try:
        workers = subprocess.Popen(
            argv,
            env=env,
            start_new_session=True,
            stdout=stdout_log,
            stderr=stderr_log,
        )
        worker_pgid = _session_pgid(workers)
        hosts = actor_mod.attach_to_workers(
            name="trainers",
            ca="trust_all_connections",
            workers=proc_addrs([f"tcp://{host}:{port}" for host in hostnames], proc_ids),
        )
        actor_config = worker_training_config(config)
        raw = asyncio.run(_monarch_train(hosts, spmd_mod, Trainer, gpus_per_node, actor_config))
        return _verdict_from_monarch(raw)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        log.warning("monarch launch failed: %s", exc, exc_info=True)
        return WorkloadResult(
            passed=False,
            failure_count=1,
            failure_details=[
                {
                    "problems": ["monarch_launch"],
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            ],
            metrics={"launcher": "monarch", "world_size": None, "final_loss": None},
        )
    finally:
        if workers is not None:
            _terminate_process_tree(workers, pgid=worker_pgid)
        _close_subprocess_logs(stdout_log, stderr_log)


def verdict_from_workers(
    exit_code: int,
    rank0: dict[str, Any] | None,
    *,
    launcher: str,
) -> WorkloadResult:
    """Pass only when the workers exited 0 and rank 0's result passed.

    Metrics keep rank 0's ``world_size`` and ``final_loss`` and always
    record ``launcher``. A missing rank-0 file fails the trial.
    """
    if isinstance(rank0, dict):
        result = workload_result_from_mapping(rank0, launcher=launcher)
    else:
        result = WorkloadResult(
            passed=False,
            failure_count=1,
            failure_details=[{"problems": ["missing_rank0_result"]}],
            metrics={"launcher": launcher, "world_size": None, "final_loss": None},
        )
    if exit_code == 0:
        return result
    details = list(result.failure_details)
    details.append({"problems": ["worker_exit"], "exit_code": exit_code})
    metrics = dict(result.metrics)
    metrics["launcher"] = launcher
    return WorkloadResult(
        passed=False,
        failure_count=result.failure_count + 1,
        first_failure_iteration=result.first_failure_iteration,
        failure_details=details,
        total_iterations=result.total_iterations,
        step_times_ms=list(result.step_times_ms),
        elapsed_sec=result.elapsed_sec,
        metrics=metrics,
        main_work_started=result.main_work_started,
        executed_iterations=result.executed_iterations,
        configured_iterations=result.configured_iterations,
    )


def workload_result_from_mapping(data: dict[str, Any], *, launcher: str) -> WorkloadResult:
    """Rebuild a :class:`WorkloadResult` from a rank's serialized result."""
    metrics_raw = data.get("metrics")
    metrics = dict(metrics_raw) if isinstance(metrics_raw, dict) else {}
    metrics["launcher"] = launcher
    if "world_size" not in metrics:
        metrics["world_size"] = None
    if "final_loss" not in metrics:
        metrics["final_loss"] = None
    details_raw = data.get("failure_details")
    if isinstance(details_raw, list):
        details = [item for item in details_raw if isinstance(item, dict)]
    else:
        details = []
    failure_count = data.get("failure_count")
    if isinstance(failure_count, bool) or not isinstance(failure_count, int):
        failure_count = 0 if data.get("passed") is True else 1
    first = data.get("first_failure_iteration")
    if first is not None and (isinstance(first, bool) or not isinstance(first, int)):
        first = None
    step_times_raw = data.get("step_times_ms")
    step_times: list[float] = []
    if isinstance(step_times_raw, list):
        for item in step_times_raw:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                continue
            step_times.append(float(item))
    elapsed = data.get("elapsed_sec")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        elapsed = 0.0
    total = data.get("total_iterations")
    if isinstance(total, bool) or not isinstance(total, int):
        total = 0
    executed = _optional_int(data.get("executed_iterations"))
    configured = _optional_int(data.get("configured_iterations"))
    main_work = data.get("main_work_started")
    if not isinstance(main_work, bool):
        main_work = None
    return WorkloadResult(
        passed=data.get("passed") is True,
        failure_count=failure_count,
        first_failure_iteration=first,
        failure_details=details,
        total_iterations=total,
        step_times_ms=step_times,
        elapsed_sec=float(elapsed),
        metrics=metrics,
        main_work_started=main_work,
        executed_iterations=executed,
        configured_iterations=configured,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _worker_env(
    *,
    master_addr: str,
    master_port: int,
    world_size: int,
    gpus_per_node: int,
) -> dict[str, str]:
    env = dict(os.environ)
    for key in _IDENTITY_ENV:
        env.pop(key, None)
    env["AORTA_INNER"] = "1"
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = str(master_port)
    env["WORLD_SIZE"] = str(world_size)
    env["AORTA_GPUS_PER_NODE"] = str(gpus_per_node)
    return env


def _tcp_port(raw: str, *, name: str) -> int:
    """Return ``raw`` as a TCP port.

    Callers pass an explicit port. Probing an ephemeral port and closing the
    socket leaves a gap before workers bind it, so this function does not
    pick one.
    """
    if not raw.isdigit():
        raise ValueError(f"{name} must be an integer, got {raw!r}")
    port = int(raw)
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be in 1..65535, got {port}")
    return port


def _shared_workdir() -> Path:
    """A fresh directory under ``AORTA_SHARED_DIR`` for this launch's result.

    The directory is explicit so rank 0 and the controller name the same
    path. A node-local default such as ``tempfile``'s ``/tmp`` would make a
    successful rank-0 write look like a missing result.
    """
    raw = os.environ.get("AORTA_SHARED_DIR", "").strip()
    if not raw:
        raise RuntimeError(
            "launcher 'spur' needs AORTA_SHARED_DIR set to a directory that "
            "every node in the allocation can read and write. Rank 0 writes "
            "rank0.json there. Node-local directories such as /tmp are not "
            "visible to a controller on another host."
        )
    root = Path(raw)
    if not root.is_dir():
        raise RuntimeError(f"AORTA_SHARED_DIR {raw!r} is not a directory")
    resolved = root.resolve()
    for prefix in _LOCAL_RESULT_PREFIXES:
        try:
            resolved.relative_to(prefix)
        except ValueError:
            continue
        log.warning(
            "AORTA_SHARED_DIR %s is node-local; rank 0's result will be "
            "invisible to a controller on another host",
            resolved,
        )
        break
    return Path(tempfile.mkdtemp(prefix="aorta-tiny-transformer-", dir=resolved))


def _read_rank0_result(output_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return rank 0's published result, and a read error when the file is unusable.

    Only ``rank0.json`` counts. A directory of earlier trial files must not
    supply another cell's verdict.
    """
    path = output_dir / "rank0.json"
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read worker result %s: %s", path, exc)
        return None, f"{path.name}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path.name}: expected a JSON object"
    result = payload.get("result")
    if isinstance(result, dict):
        return result, None
    if isinstance(payload.get("passed"), bool):
        return payload, None
    return None, f"{path.name}: no rank-0 result"


def _import_monarch() -> tuple[Any, ...]:
    """Import Monarch. Any failure says it is not installed."""
    try:
        import monarch.actor as actor_mod
        import monarch.spmd as spmd_mod
        from monarch._src.job._slurm_batch import _worker_bootstrap
        from monarch._src.job.service_identity import (
            SERVICE_PROC_IDS_ENV,
            allocate_service_proc_ids,
            serialize_service_proc_ids,
            service_proc_addrs,
        )
    except Exception as exc:
        raise RuntimeError(_MONARCH_MISSING.format(exc=exc)) from exc
    else:
        return (
            actor_mod,
            spmd_mod,
            _worker_bootstrap,
            SERVICE_PROC_IDS_ENV,
            allocate_service_proc_ids,
            serialize_service_proc_ids,
            service_proc_addrs,
        )


async def _monarch_train(
    hosts: Any,
    spmd_mod: Any,
    trainer_cls: type,
    gpus_per_node: int,
    config: dict[str, Any],
) -> Any:
    await hosts.initialized
    procs = hosts.spawn_procs(per_host={"gpus": gpus_per_node})
    await procs.logging_option(stream_to_client=True)
    await spmd_mod.setup_torch_elastic_env_async(procs)
    trainers = procs.spawn("trainer", trainer_cls)
    return await trainers.train.call(config)


def _verdict_from_monarch(raw: Any) -> WorkloadResult:
    rows = _rank_rows(raw)
    if not rows:
        return WorkloadResult(
            passed=False,
            failure_count=1,
            failure_details=[{"problems": ["missing_rank0_result"]}],
            metrics={"launcher": "monarch", "world_size": None, "final_loss": None},
        )
    rank0 = rows[0]
    for item in rows:
        metrics = item.get("metrics")
        if isinstance(metrics, dict) and metrics.get("rank") == 0:
            rank0 = item
            break
    result = workload_result_from_mapping(rank0, launcher="monarch")
    if any(item.get("passed") is not True for item in rows):
        result = WorkloadResult(
            passed=False,
            failure_count=max(result.failure_count, 1),
            first_failure_iteration=result.first_failure_iteration,
            failure_details=list(result.failure_details) or [{"problems": ["rank_result_failed"]}],
            total_iterations=result.total_iterations,
            step_times_ms=list(result.step_times_ms),
            elapsed_sec=result.elapsed_sec,
            metrics=dict(result.metrics),
            main_work_started=result.main_work_started,
            executed_iterations=result.executed_iterations,
            configured_iterations=result.configured_iterations,
        )
    return result


def _rank_rows(raw: Any) -> list[dict[str, Any]]:
    items = raw.items if isinstance(raw, dict) else getattr(raw, "items", None)
    if not callable(items):
        raise TypeError(
            f"Monarch train() returned {type(raw).__name__}, expected a mapping of rank results"
        )
    rows: list[dict[str, Any]] = []
    for _point, value in items():
        if not isinstance(value, dict):
            raise TypeError(f"Monarch rank result must be a dict, got {type(value).__name__}")
        rows.append(value)
    return rows


def _monarch_hostnames(prefix: list[str], nodes: int) -> list[str]:
    """Ask each node for the hostname Monarch workers advertise.

    ``prefix`` comes from :func:`monarch_step_prefix`. Order follows
    ``SLURM_NODEID``, which Spur also sets. This runs a step of the current
    allocation; it does not request a new one.
    """
    script = 'echo "$SLURM_NODEID $(python3 -c "import socket; print(socket.gethostname())")"'
    argv = [*prefix, "bash", "-c", script]
    try:
        out = subprocess.check_output(argv, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not list hostnames with {' '.join(prefix)}: {exc}") from exc
    pairs: list[tuple[int, str]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pairs.append((int(parts[0]), parts[1]))
    pairs.sort()
    if len(pairs) != nodes:
        raise RuntimeError(f"srun returned {len(pairs)} hostnames for {nodes} nodes: {out!r}")
    return [host for _, host in pairs]


def _session_pgid(proc: subprocess.Popen[Any]) -> int | None:
    """Process group of a ``start_new_session`` child, saved while the leader is alive.

    After the leader exits, ``getpgid`` on its pid fails even when children
    remain in the group. The caller keeps this value. A pid that is not its
    own process-group leader is not our session (a recycled pid, or a test
    double) and is ignored so cleanup cannot signal an unrelated process.
    """
    pid = getattr(proc, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return None
    except OSError as exc:
        log.warning("getpgid(%s) failed: %s", pid, exc)
        return None
    if pgid != pid or pgid == os.getpgrp():
        return None
    return pgid


def _group_has_members(pgid: int) -> bool:
    """True when any process in ``pgid`` still exists, including after the leader exits."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        log.warning("killpg(%s, 0) failed: %s", pgid, exc)
        return False
    return True


def _needs_group_cleanup(proc: subprocess.Popen[Any], pgid: int | None) -> bool:
    """True when the leader or any of its group is still running."""
    if proc.poll() is None:
        return True
    return pgid is not None and _group_has_members(pgid)


def _wait_for_group(proc: subprocess.Popen[Any], pgid: int | None, timeout: float) -> None:
    """Block until the leader and its group are gone, or ``timeout`` elapses.

    ``Popen.wait`` returns when the leader exits. A child that ignores
    SIGTERM stays in the group, so the wait continues until ``killpg``
    reports that the group is empty.
    """
    deadline = time.monotonic() + timeout
    while _needs_group_cleanup(proc, pgid):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        slice_s = min(0.2, remaining)
        if proc.poll() is None:
            try:
                proc.wait(timeout=slice_s)
            except subprocess.TimeoutExpired:
                pass
            except OSError as exc:
                log.warning(
                    "waiting for child pid %s failed: %s",
                    getattr(proc, "pid", "?"),
                    exc,
                )
                return
        else:
            time.sleep(slice_s)


def _terminate_process_tree(proc: subprocess.Popen[Any], pgid: int | None = None) -> None:
    """SIGTERM the child's process group, then SIGKILL, then reap.

    ``pgid`` is the group captured while the leader was alive. Waiting only
    on the leader would return while a child that ignores SIGTERM still holds
    GPUs or ports. ``KeyboardInterrupt`` during the grace period still
    SIGKILLs the group and is re-raised. ``ProcessLookupError`` is the group
    already being gone. Any other ``OSError`` is logged: the tree may still
    be alive.
    """
    if pgid is None:
        pgid = _session_pgid(proc)
    elif pgid == os.getpgrp():
        pgid = None

    def _send(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                pass
            except OSError as exc:
                log.warning(
                    "killpg(%s, %s) failed: %s; the child process group may still be running",
                    pgid,
                    signal.Signals(sig).name,
                    exc,
                )
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass
        except OSError as exc:
            log.warning(
                "send_signal(%s) to child pid %s failed: %s",
                signal.Signals(sig).name,
                getattr(proc, "pid", "?"),
                exc,
            )

    def _force_kill() -> None:
        if proc.poll() is not None and pgid is not None and _group_has_members(pgid):
            log.warning(
                "process group %s still has members after the leader exited; sending SIGKILL",
                pgid,
            )
        _send(signal.SIGKILL)
        try:
            _wait_for_group(proc, pgid, _REAP_AFTER_KILL_SEC)
        except KeyboardInterrupt:
            pass
        if _needs_group_cleanup(proc, pgid):
            log.warning("process group %s still has members after SIGKILL", pgid)

    _send(signal.SIGTERM)
    try:
        _wait_for_group(proc, pgid, _TERMINATE_GRACE_SEC)
    except KeyboardInterrupt:
        _force_kill()
        raise
    if not _needs_group_cleanup(proc, pgid):
        return
    _force_kill()


__all__ = [
    "SPUR_WORKER_SCRIPT",
    "build_spur_launch",
    "controller_ipv4",
    "is_ipv4",
    "is_launched_rank",
    "spawning_controller",
    "launch_monarch",
    "launch_spur",
    "monarch_step_prefix",
    "resolve_rank0_ipv4",
    "run_training_rank",
    "run_worker_rank",
    "verdict_from_workers",
    "worker_training_config",
]
