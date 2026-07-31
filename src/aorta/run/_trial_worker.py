"""Hidden fresh-interpreter entry point for one isolated trial."""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from aorta.run._worker_protocol import (
    PROTOCOL_VERSION,
    read_envelope,
    write_envelope_atomic,
)
from aorta.workloads import Workload

_STATUS_SEVERITY = {
    "ok": 0,
    "workload_failed": 1,
    "workload_setup_failed": 2,
    "infrastructure_failed": 3,
}
_MAX_FAILURE_DETAILS_PER_RANK = 8
_MAX_FAILURE_DETAILS_TOTAL = 256


def _resolve_backend_request(
    workload_cls: type[Workload],
    config: dict[str, Any],
) -> Literal["auto", "gloo", "nccl"]:
    backend = workload_cls.isolated_distributed_backend(config)
    if backend == "auto":
        device_preference = config.get("device", "auto")
        if device_preference == "cpu":
            return "gloo"
        if device_preference == "cuda":
            return "nccl"
    return backend


def _initialize_distributed_worker(
    store_prefix: str,
    backend_request: str,
) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        return
    if backend_request not in {"auto", "gloo", "nccl"}:
        raise ValueError(f"invalid isolated distributed backend {backend_request!r}")
    backend = (
        ("nccl" if torch.cuda.is_available() else "gloo")
        if backend_request == "auto"
        else backend_request
    )
    rank = int(os.environ["RANK"])
    if os.environ.get("TORCHELASTIC_USE_AGENT_STORE", "").lower() == "true":
        use_libuv = (
            os.environ.get(
                "USE_LIBUV",
                "0" if sys.platform == "win32" else "1",
            )
            == "1"
        )
        store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]),
            world_size=world_size,
            is_master=False,
            timeout=timedelta(minutes=5),
            wait_for_workers=False,
            use_libuv=use_libuv,
        )
        prefixed_store = dist.PrefixStore(store_prefix, store)
        dist.init_process_group(
            backend=backend,
            store=prefixed_store,
            rank=rank,
            world_size=world_size,
        )
    else:
        dist.init_process_group(backend=backend)


def _run_request_from_dict(data: dict[str, Any]):
    from aorta.run.dispatcher import RunRequest

    return RunRequest(
        workload=data["workload"],
        trials=1,
        environment=data.get("environment", "local"),
        image=data.get("image"),
        buck_target=data.get("buck_target"),
        mitigations=tuple(data.get("mitigations", ("none",))),
        extra_env=dict(data.get("extra_env", {})),
        trial_isolation=data.get("trial_isolation", "process"),
        cell_name=data.get("cell_name"),
        request_fingerprint=data.get("request_fingerprint"),
        steps=data.get("steps"),
        config_overrides=dict(data.get("config_overrides", {})),
        results_dir=Path(data["results_dir"]),
        collect=tuple(data.get("collect", ())),
        collect_options={
            name: dict(options) for name, options in data.get("collect_options", {}).items()
        },
        sidecar_files=tuple(Path(path) for path in data.get("sidecar_files", ())),
        dataset_index=int(data.get("dataset_index", 0)),
        mitigation_index=int(data.get("mitigation_index", 0)),
        save_logs=bool(data.get("save_logs", False)),
        env_probe=(dict(data["env_probe"]) if data.get("env_probe") is not None else None),
    )


def _synchronize_workload_result(
    workload_result,
    exit_status: str,
    *,
    rank_local: bool,
):
    """Synchronize every rank outcome before workload cleanup."""
    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    try:
        import torch.distributed as dist
    except Exception:
        if requested_world_size > 1:
            raise RuntimeError(
                "distributed trial cannot synchronize rank outcomes because "
                "torch.distributed is unavailable"
            )
        return workload_result, exit_status
    if not dist.is_available() or not dist.is_initialized():
        if requested_world_size > 1:
            raise RuntimeError(
                "distributed trial failed before its worker-owned process "
                "group was initialized"
            )
        return workload_result, exit_status

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_status_error = exit_status in {
        "workload_setup_failed",
        "infrastructure_failed",
    }
    local_details = (
        list(workload_result.failure_details)
        if rank_local or local_status_error
        else []
    )
    local_detail_omitted = 0
    if rank_local and isinstance(workload_result.metrics, dict):
        raw_omitted = workload_result.metrics.get(
            "corruption_details_omitted",
            0,
        )
        if isinstance(raw_omitted, int) and raw_omitted > 0:
            local_detail_omitted = raw_omitted
    local_summary = {
        "exit_status": exit_status,
        "passed": bool(workload_result.passed),
        "failure_count": (
            int(workload_result.failure_count)
            if rank_local or local_status_error
            else 0
        ),
        "first_failure_iteration": (
            workload_result.first_failure_iteration
            if rank_local or local_status_error
            else None
        ),
        "failure_detail_count": len(local_details),
        "failure_detail_omitted": local_detail_omitted,
        "failure_details": copy.deepcopy(
            local_details[:_MAX_FAILURE_DETAILS_PER_RANK]
        ),
    }
    gathered: list[dict[str, Any] | None] | None = (
        [None] * world_size
        if rank == 0
        else None
    )
    dist.gather_object(local_summary, gathered, dst=0)

    broadcast: list[dict[str, Any] | None] = [None]
    if rank == 0:
        assert gathered is not None
        rank_summaries = [entry for entry in gathered if isinstance(entry, dict)]
        if len(rank_summaries) != world_size:
            raise RuntimeError("not every rank produced a valid workload summary")

        canonical_result = asdict(workload_result)
        canonical_status = max(
            (entry["exit_status"] for entry in rank_summaries),
            key=lambda status: _STATUS_SEVERITY.get(status, 99),
        )
        passed_count = sum(bool(entry.get("passed")) for entry in rank_summaries)
        canonical_result["passed"] = (
            passed_count == world_size and canonical_status == "ok"
        )
        merge_rank_failures = rank_local or any(
            entry["exit_status"]
            in {"workload_setup_failed", "infrastructure_failed"}
            for entry in rank_summaries
        )
        if merge_rank_failures:
            canonical_result["failure_count"] = sum(
                int(entry.get("failure_count", 0))
                for entry in rank_summaries
            )
            details: list[dict[str, Any]] = []
            first_failures: list[int] = []
            total_detail_count = 0
            for summary_rank, entry in enumerate(rank_summaries):
                first = entry.get("first_failure_iteration")
                if isinstance(first, int):
                    first_failures.append(first)
                total_detail_count += (
                    int(entry.get("failure_detail_count", 0))
                    + int(entry.get("failure_detail_omitted", 0))
                )
                for raw_detail in entry.get("failure_details", []) or []:
                    if len(details) >= _MAX_FAILURE_DETAILS_TOTAL:
                        break
                    detail = dict(raw_detail)
                    detail.setdefault("rank", summary_rank)
                    details.append(detail)
            canonical_result["failure_details"] = details
            canonical_result["first_failure_iteration"] = (
                min(first_failures) if first_failures else None
            )
        else:
            total_detail_count = len(canonical_result.get("failure_details", []))

        metrics = dict(canonical_result.get("metrics", {}))
        metrics["_rank_outcomes"] = {
            "passed": passed_count,
            "failed": world_size - passed_count,
            "world_size": world_size,
        }
        omitted_details = max(
            0,
            total_detail_count - len(canonical_result.get("failure_details", [])),
        )
        if omitted_details:
            metrics["_failure_details_omitted"] = omitted_details
        if rank_local:
            metrics["corruption_details_omitted"] = sum(
                int(entry.get("failure_detail_omitted", 0))
                for entry in rank_summaries
            )
        canonical_result["metrics"] = metrics
        broadcast[0] = {
            "result": canonical_result,
            "exit_status": canonical_status,
        }

    dist.broadcast_object_list(broadcast, src=0)
    canonical = broadcast[0]
    if not isinstance(canonical, dict):
        raise RuntimeError("rank 0 did not broadcast a valid workload result")
    result_data = canonical.get("result")
    canonical_status = canonical.get("exit_status")
    if not isinstance(result_data, dict) or not isinstance(canonical_status, str):
        raise RuntimeError("broadcast workload result has an invalid shape")

    from aorta.workloads import WorkloadResult

    return WorkloadResult(**result_data), canonical_status


def _destroy_worker_process_group() -> None:
    try:
        import torch.distributed as dist
    except Exception:
        return
    try:
        if dist.is_available() and dist.is_initialized():
            # Do not enter a barrier here. A peer may have failed before it
            # could initialize or synchronize; process exit is the fallback.
            dist.destroy_process_group()
    except Exception:
        # Teardown is best-effort and must not replace the canonical trial
        # result or the original worker exception.
        pass


def _main(request_path: Path, response_path: Path) -> int:
    envelope = read_envelope(request_path)
    nonce = envelope["nonce"]
    trial_id = envelope["trial_id"]
    try:
        # Imports that may load plugin workloads or torch occur only after the
        # worker process has started with its fully resolved environment.
        from aorta.registry import Environment
        from aorta.run._trial_runtime import execute_trial
        from aorta.run.discovery import get_workload_class
        from aorta.run.validation import (
            resolve_trial_isolation,
            validate_launch_mode,
        )

        request = _run_request_from_dict(envelope["run_request"])
        workload_cls = get_workload_class(request.workload)
        validate_launch_mode(workload_cls)
        resolve_trial_isolation(workload_cls, "process")
        backend_config = dict(request.config_overrides)
        if request.steps is not None:
            backend_config["steps"] = request.steps
        startup_env = workload_cls.isolated_startup_env(backend_config)
        if not isinstance(startup_env, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in startup_env.items()
        ):
            raise ValueError(
                f"Workload '{workload_cls.__name__}' returned invalid "
                "isolated_startup_env; expected dict[str, str]"
            )
        missing_startup_env = {
            key: value
            for key, value in startup_env.items()
            if key not in os.environ
        }
        startup_marker = "_AORTA_TRIAL_STARTUP_NONCE"
        if missing_startup_env:
            if os.environ.get(startup_marker) == nonce:
                raise RuntimeError(
                    "isolated startup environment was not preserved across "
                    "worker re-exec"
                )
            restart_env = dict(os.environ)
            restart_env.update(missing_startup_env)
            restart_env[startup_marker] = nonce
            os.execve(
                sys.executable,
                [
                    sys.executable,
                    "-m",
                    "aorta.run._trial_worker",
                    str(request_path),
                    str(response_path),
                ],
                restart_env,
            )
        os.environ.pop(startup_marker, None)
        for key, value in startup_env.items():
            os.environ.setdefault(key, value)
        backend_request = _resolve_backend_request(workload_cls, backend_config)
        _initialize_distributed_worker(
            envelope["store_prefix"],
            backend_request,
        )
        env_descriptor = Environment(**envelope["env_descriptor"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank_local = getattr(workload_cls, "distributed_result_scope", "global") == "rank_local"
        result = execute_trial(
            trial_idx=int(envelope["trial_idx"]),
            workload_cls=workload_cls,
            request=request,
            env_descriptor=env_descriptor,
            mitigation_env=dict(envelope["mitigation_env"]),
            results_dir=Path(envelope["results_dir"]),
            should_write=bool(envelope["should_write"]),
            persist_result=False,
            result_transform=(
                lambda workload_result, status: _synchronize_workload_result(
                    workload_result,
                    status,
                    rank_local=rank_local,
                )
            ),
            skip_cleanup_on_error=world_size > 1,
        )
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "nonce": nonce,
            "trial_id": trial_id,
            "kind": "result",
            "trial_result": result.to_dict(),
        }
        write_envelope_atomic(response_path, response)
        return 0
    except BaseException as exc:
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "nonce": nonce,
            "trial_id": trial_id,
            "kind": "worker_error",
            "error": {
                "phase": "worker",
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        write_envelope_atomic(response_path, response)
        return 1
    finally:
        _destroy_worker_process_group()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m aorta.run._trial_worker REQUEST RESPONSE")
    raise SystemExit(_main(Path(sys.argv[1]), Path(sys.argv[2])))
