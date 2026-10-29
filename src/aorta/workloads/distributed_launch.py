"""Spur and Monarch launchers for the tiny-transformer workload.

Neither launcher is an install dependency. Spur is a subprocess of
``spur_bin`` (default ``spur``). Monarch is imported only when
:func:`launch_monarch` runs.

The spur controller re-execs ``aorta run`` (or the current ``aorta``
invocation) with ``AORTA_INNER=1``. Each worker maps ``SLURM_PROCID`` to
``RANK`` and uses an IPv4 ``MASTER_ADDR``: on this cluster a hostname
resolves to an IPv6 link-local address first. The controller's verdict
passes only when the worker exit code is 0 and rank 0's result passed.

The monarch controller follows the runs/04 sequence: TCP transport on the
controller IPv4, one worker per node, a proc mesh, and
``setup_torch_elastic_env_async``. The actor calls :func:`run_training_rank`
instead of an inlined model. A missing ``monarch`` import raises an error
that names the pytorch_monarch source build.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aorta.workloads._base import WorkloadResult
from aorta.workloads.training import TrainingWorkload

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
# before ``aorta`` starts because the dispatcher uses it to decide who writes.
SPUR_WORKER_SCRIPT = """\
set -euo pipefail
if [ -z "${SLURM_PROCID:-}" ]; then
  echo "spur worker: SLURM_PROCID is required to set RANK" >&2
  exit 2
fi
export RANK="$SLURM_PROCID"
gpus="${AORTA_GPUS_PER_NODE:?AORTA_GPUS_PER_NODE is required}"
export LOCAL_RANK=$(( RANK % gpus ))
: "${WORLD_SIZE:?WORLD_SIZE is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${AORTA_INNER:?AORTA_INNER is required}"
exec "$@"
"""

_MONARCH_MISSING = (
    "Monarch is not installed. launcher 'monarch' needs the pytorch_monarch "
    "source build (venvs/rocm10.1); import failed: {exc!r}"
)


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
            "(hostnames on this cluster resolve to IPv6 link-local first)."
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


def run_training_rank(config: dict[str, Any]) -> dict[str, Any]:
    """Run one training rank and return its :class:`WorkloadResult` as a dict.

    Shared by the in-process path, spur workers, and the Monarch actor.
    Cleanup still runs when setup or run raises. A cleanup error does not
    replace that exception. When the run itself finished, the error is
    stored on ``metrics["cleanup_error"]``. ``KeyboardInterrupt`` is not caught.
    """
    if not isinstance(config, dict):
        raise TypeError(f"training config must be a dict, got {type(config).__name__}")
    workload = TrainingWorkload(config)
    cleanup_error: str | None = None
    try:
        workload.setup()
        result = workload.run()
    finally:
        try:
            workload.cleanup()
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


def build_spur_launch(
    *,
    spur_bin: str,
    nodes: int,
    gpus_per_node: int,
    master_addr: str,
    master_port: int,
    inner_argv: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Argv and env for ``spur srun`` re-executing the inner aorta command."""
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
        "bash",
        "-c",
        SPUR_WORKER_SCRIPT,
        "aorta-inner",
        *inner_argv,
    ]
    env = _worker_env(
        master_addr=master_addr,
        master_port=master_port,
        world_size=nodes * gpus_per_node,
        gpus_per_node=gpus_per_node,
    )
    return argv, env


def inner_aorta_argv(output_dir: Path, config: dict[str, Any]) -> list[str]:
    """Re-exec the current ``aorta`` invocation against ``output_dir``.

    A ``triage run`` keeps its recipe and switches ``--output-dir``. A direct
    ``aorta run`` switches ``--results-dir``. ``--dry-run`` is dropped so
    workers actually train. Any other parent (a test runner, a library
    call) falls back to ``aorta run --workload tiny_transformer``.
    """
    argv = list(sys.argv)
    if argv and Path(argv[0]).name == "aorta":
        argv = _drop_flag(argv, "--dry-run")
        if "triage" in argv:
            return _replace_option(argv, "--output-dir", str(output_dir))
        return _replace_option(argv, "--results-dir", str(output_dir))
    return _fallback_aorta_run(output_dir, config.get("steps"))


def launch_spur(
    config: dict[str, Any],
    *,
    nodes: int,
    gpus_per_node: int,
    spur_bin: str,
) -> WorkloadResult:
    """Spawn ``spur srun`` and return rank 0's verdict gated on the exit code."""
    master_addr = controller_ipv4()
    raw_port = os.environ.get("AORTA_MASTER_PORT")
    if raw_port is None or raw_port == "":
        raise RuntimeError(
            "launcher 'spur' needs AORTA_MASTER_PORT set to a free TCP port. "
            "An ephemeral port cannot be held until the workers bind it."
        )
    master_port = _tcp_port(raw_port, name="AORTA_MASTER_PORT")
    with tempfile.TemporaryDirectory(prefix="aorta-tiny-transformer-") as tmp:
        output = Path(tmp)
        argv, env = build_spur_launch(
            spur_bin=spur_bin,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            master_addr=master_addr,
            master_port=master_port,
            inner_argv=inner_aorta_argv(output, config),
        )
        log.info(
            "spur launch MASTER_ADDR=%s WORLD_SIZE=%s argv=%s",
            env.get("MASTER_ADDR"),
            env.get("WORLD_SIZE"),
            argv,
        )
        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(argv, env=env, start_new_session=True)
            proc.wait()
        except KeyboardInterrupt:
            raise
        finally:
            if proc is not None and proc.poll() is None:
                _terminate_process_tree(proc)
        exit_code = proc.returncode if proc is not None and isinstance(proc.returncode, int) else 1
        rank0, read_error = _read_rank0_result(output)
        result = verdict_from_workers(exit_code, rank0, launcher="spur")
        if read_error is not None:
            result.failure_details.append(
                {"problems": ["worker_result_unreadable"], "error": read_error}
            )
            result.passed = False
            result.failure_count += 1
        return result


def launch_monarch(
    config: dict[str, Any],
    *,
    nodes: int,
    gpus_per_node: int,
) -> WorkloadResult:
    """Run the training actor under a Monarch proc mesh on an existing allocation.

    Does not submit a Slurm job. ``SLURM_JOB_ID`` or ``MONARCH_SLURM_JOB``
    must already name the allocation ``srun --overlap`` attaches to.
    """
    imported = _import_monarch()
    actor_mod, spmd_mod, bootstrap, service_env, alloc_ids, serialize_ids, proc_addrs = imported
    trainer_cls = _trainer_class(actor_mod.Actor, actor_mod.endpoint)
    ip = controller_ipv4()
    actor_mod.enable_transport(f"tcp://{ip}:0")
    job_id = os.environ.get("MONARCH_SLURM_JOB") or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError(
            "launcher 'monarch' attaches to an existing allocation. "
            "Set SLURM_JOB_ID or MONARCH_SLURM_JOB. This workload does not submit Slurm jobs."
        )
    port = _tcp_port(os.environ.get("MONARCH_PORT", "22222"), name="MONARCH_PORT")
    hostnames = _monarch_hostnames(str(job_id), nodes)
    proc_ids = alloc_ids(nodes)
    env = {**os.environ, service_env: serialize_ids(proc_ids)}
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(key, None)
    worker_cmd = [sys.executable, "-c", bootstrap(port, None)]
    argv = [
        "srun",
        "--jobid",
        str(job_id),
        "--overlap",
        f"-N{nodes}",
        "--ntasks-per-node=1",
        "--kill-on-bad-exit=1",
        *worker_cmd,
    ]
    log.info("monarch controller tcp://%s hosts=%s", ip, hostnames)
    workers: subprocess.Popen[str] | None = None
    try:
        workers = subprocess.Popen(argv, env=env, start_new_session=True)
        hosts = actor_mod.attach_to_workers(
            name="trainers",
            ca="trust_all_connections",
            workers=proc_addrs([f"tcp://{host}:{port}" for host in hostnames], proc_ids),
        )
        actor_config = {
            key: value
            for key, value in config.items()
            if not (isinstance(key, str) and key.startswith("_aorta_"))
        }
        raw = asyncio.run(
            _monarch_train(hosts, spmd_mod, trainer_cls, gpus_per_node, actor_config)
        )
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
            _terminate_process_tree(workers)


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


def _drop_flag(argv: list[str], flag: str) -> list[str]:
    return [token for token in argv if token != flag]


def _replace_option(argv: list[str], flag: str, value: str) -> list[str]:
    replaced = False
    out: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == flag:
            out.extend([flag, value])
            replaced = True
            index += 2 if index + 1 < len(argv) else 1
            continue
        if token.startswith(flag + "="):
            out.append(f"{flag}={value}")
            replaced = True
            index += 1
            continue
        out.append(token)
        index += 1
    if not replaced:
        out.extend([flag, value])
    return out


def _fallback_aorta_run(output_dir: Path, steps: object) -> list[str]:
    tail = [
        "run",
        "--workload",
        "tiny_transformer",
        "--trials",
        "1",
        "--results-dir",
        str(output_dir),
    ]
    if isinstance(steps, int) and not isinstance(steps, bool) and steps >= 1:
        tail.extend(["--steps", str(steps)])
    aorta = _which("aorta")
    if aorta:
        return [aorta, *tail]
    code = (
        "import sys\n"
        "from aorta.cli import main\n"
        "sys.argv = ['aorta', *sys.argv[1:]]\n"
        "raise SystemExit(main())\n"
    )
    return [sys.executable, "-c", code, *tail]


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _read_rank0_result(output_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return rank 0's result, and a read error when no result could be parsed."""
    paths = sorted(output_dir.glob("**/trial_*.json"))
    paths.extend(sorted(output_dir.glob("**/result.json")))
    found: list[dict[str, Any]] = []
    read_errors: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read worker result %s: %s", path, exc)
            read_errors.append(f"{path.name}: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        result = payload.get("result")
        if isinstance(result, dict):
            found.append(result)
            continue
        if isinstance(payload.get("passed"), bool):
            found.append(payload)
    for item in found:
        metrics = item.get("metrics")
        if isinstance(metrics, dict) and metrics.get("rank") == 0:
            return item, None
    if found:
        return found[0], None
    if read_errors:
        return None, "; ".join(read_errors)
    return None, None


def _import_monarch() -> tuple[Any, ...]:
    """Import Monarch. Any failure names the source build the cluster uses."""
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


def _trainer_class(actor_base: type, endpoint: Any) -> type:
    class Trainer(actor_base):
        @endpoint
        def train(self, config: dict) -> dict:
            if not isinstance(config, dict):
                raise TypeError(f"train config must be a dict, got {type(config).__name__}")
            return run_training_rank(config)

    Trainer.__module__ = __name__
    Trainer.__qualname__ = "Trainer"
    globals()["Trainer"] = Trainer
    return Trainer


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
            failure_details=list(result.failure_details)
            or [{"problems": ["rank_result_failed"]}],
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


def _monarch_hostnames(job_id: str, nodes: int) -> list[str]:
    """Ask each node for the hostname Monarch workers advertise.

    Order follows ``SLURM_NODEID``. This attaches to ``job_id``; it does
    not submit a new allocation.
    """
    script = (
        'echo "$SLURM_NODEID $(python3 -c "import socket; print(socket.gethostname())")"'
    )
    argv = [
        "srun",
        "--jobid",
        job_id,
        "--overlap",
        f"-N{nodes}",
        "--ntasks-per-node=1",
        "bash",
        "-c",
        script,
    ]
    try:
        out = subprocess.check_output(argv, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Could not list hostnames for Slurm job {job_id} via srun: {exc}"
        ) from exc
    pairs: list[tuple[int, str]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pairs.append((int(parts[0]), parts[1]))
    pairs.sort()
    if len(pairs) != nodes:
        raise RuntimeError(
            f"srun returned {len(pairs)} hostnames for {nodes} nodes: {out!r}"
        )
    return [host for _, host in pairs]


def _terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    """SIGTERM the child's process group, then SIGKILL, then reap.

    ``KeyboardInterrupt`` during the grace period still SIGKILLs the group
    and is re-raised. ``ProcessLookupError`` is the group already being
    gone. Any other ``OSError`` is logged: the tree may still be alive.
    """
    pgid: int | None
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None
    except OSError as exc:
        log.warning("getpgid(%s) failed: %s", getattr(proc, "pid", "?"), exc)
        pgid = None
    if pgid is not None and pgid == os.getpgrp():
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

    def _reap_after_kill() -> None:
        try:
            proc.wait(timeout=_REAP_AFTER_KILL_SEC)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            pass
        except KeyboardInterrupt:
            pass

    _send(signal.SIGTERM)
    try:
        proc.wait(timeout=_TERMINATE_GRACE_SEC)
        return
    except subprocess.TimeoutExpired:
        pass
    except ProcessLookupError:
        return
    except OSError as exc:
        log.warning("waiting for child pid %s failed: %s", getattr(proc, "pid", "?"), exc)
        return
    except KeyboardInterrupt:
        _send(signal.SIGKILL)
        _reap_after_kill()
        raise
    _send(signal.SIGKILL)
    try:
        proc.wait(timeout=_REAP_AFTER_KILL_SEC)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        pass
    except KeyboardInterrupt:
        _reap_after_kill()
        raise


__all__ = [
    "SPUR_WORKER_SCRIPT",
    "build_spur_launch",
    "controller_ipv4",
    "inner_aorta_argv",
    "is_ipv4",
    "is_launched_rank",
    "launch_monarch",
    "launch_spur",
    "run_training_rank",
    "verdict_from_workers",
]
