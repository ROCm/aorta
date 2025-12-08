"""FSDP2 multi-stream training benchmark with profiling."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler, profile
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from aorta.data import SyntheticDatasetConfig, create_dataloader
from aorta.models import ModelConfig, RankingTransformerModel
from aorta.profiling.stream_profiler import StreamProfiler
from aorta.utils import detect_accelerator, get_device, get_distributed_backend, load_config, merge_cli_overrides, setup_logging

log = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 2e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    warmup_steps: int = 200
    total_steps: int = 2000


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation: int = 1
    max_steps: Optional[int] = None
    grad_clip_norm: float = 1.0
    mixed_precision: str = "bf16"  # options: none, fp16, bf16
    log_interval: int = 10
    output_dir: Path = Path("artifacts")
    inject_allreduce_copies: bool = False  # Inject all_reduce + host-device copies to trigger hang
    allreduce_stress_level: int = 1  # Number of all_reduce ops per iteration (1-10)


@dataclass
class FSDPConfig:
    sharding_strategy: str = "full_shard"
    backward_prefetch: str = "BACKWARD_PRE"
    use_orig_params: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = True
    sync_module_states: bool = True
    param_init_device: str = "cpu"


@dataclass
class CompileConfig:
    enabled: bool = False
    backend: Optional[str] = "inductor"
    mode: Optional[str] = "max-autotune"
    fullgraph: bool = False
    dynamic: bool = False
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DDPConfig:
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False


@dataclass
class ProfilerConfig:
    enabled: bool = True
    wait: int = 1
    warmup: int = 1
    active: int = 2
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = False
    tensorboard: bool = False
    chrome_trace: bool = True
    trace_filename: str = "trace.json"


def _parse_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    config = load_config(config_path)
    config = merge_cli_overrides(config, args.override or [])
    return config


def _build_training_config(raw: Dict[str, Any]) -> TrainingConfig:
    training = raw.get("training", {})
    cfg = TrainingConfig()
    for field in dataclass_fields(TrainingConfig):
        if field.name in training:
            setattr(cfg, field.name, training[field.name])
    cfg.output_dir = Path(cfg.output_dir)
    return cfg


def _build_optimizer_config(raw: Dict[str, Any]) -> OptimizerConfig:
    section = raw.get("optimizer", {})
    cfg = OptimizerConfig()
    for field in dataclass_fields(OptimizerConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_scheduler_config(raw: Dict[str, Any]) -> SchedulerConfig:
    section = raw.get("scheduler", {})
    cfg = SchedulerConfig()
    for field in dataclass_fields(SchedulerConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_model_config(raw: Dict[str, Any]) -> ModelConfig:
    section = raw.get("model", {})
    cfg = ModelConfig()
    for field in dataclass_fields(ModelConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_dataset_config(raw: Dict[str, Any]) -> SyntheticDatasetConfig:
    section = raw.get("dataset", {})
    cfg = SyntheticDatasetConfig()
    for field in dataclass_fields(SyntheticDatasetConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_fsdp_config(raw: Dict[str, Any]) -> FSDPConfig:
    section = raw.get("fsdp", {})
    cfg = FSDPConfig()
    for field in dataclass_fields(FSDPConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_ddp_config(raw: Dict[str, Any]) -> DDPConfig:
    section = raw.get("distributed", {})
    cfg = DDPConfig()
    for field in dataclass_fields(DDPConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_compile_config(raw: Dict[str, Any]) -> CompileConfig:
    section = raw.get("compile", {})
    cfg = CompileConfig()
    for field in dataclass_fields(CompileConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    if cfg.options is None:
        cfg.options = {}
    return cfg


def _build_profiler_config(raw: Dict[str, Any]) -> ProfilerConfig:
    section = raw.get("profiling", {})
    cfg = ProfilerConfig()
    for field in dataclass_fields(ProfilerConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def dataclass_fields(cls) -> Iterable[Any]:
    return getattr(cls, "__dataclass_fields__").values()


def init_distributed(training_cfg: TrainingConfig, log_level: str) -> Dict[str, Any]:
    backend = get_distributed_backend()
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    device = get_device(local_rank)
    torch.cuda.set_device(device)
    training_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(level=log_level, log_file=training_cfg.output_dir / f"rank{rank}.log", rank=rank)

    log.info(
        "Initialised distributed training | backend=%s rank=%s world=%s local_rank=%s device=%s",
        backend,
        rank,
        world_size,
        local_rank,
        device,
    )

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
    }


def build_fsdp_model(
    model_cfg: ModelConfig,
    fsdp_cfg: FSDPConfig,
    compile_cfg: CompileConfig,
    device: torch.device,
) -> FSDP:
    model = RankingTransformerModel(model_cfg)

    sharding = getattr(ShardingStrategy, fsdp_cfg.sharding_strategy.upper())
    backward_prefetch = getattr(BackwardPrefetch, fsdp_cfg.backward_prefetch.upper())

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={nn.TransformerEncoderLayer}
    )

    fsdp_model = FSDP(
        model.to(device),
        sharding_strategy=sharding,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=fsdp_cfg.use_orig_params,
        backward_prefetch=backward_prefetch,
        limit_all_gathers=fsdp_cfg.limit_all_gathers,
        forward_prefetch=fsdp_cfg.forward_prefetch,
        device_id=torch.cuda.current_device(),
        sync_module_states=fsdp_cfg.sync_module_states,
    )
    if compile_cfg.enabled:
        fsdp_model = _maybe_compile(fsdp_model, compile_cfg)
    return fsdp_model


def build_ddp_model(
    model_cfg: ModelConfig,
    ddp_cfg: DDPConfig,
    compile_cfg: CompileConfig,
    device: torch.device,
) -> DDP:
    model = RankingTransformerModel(model_cfg).to(device)
    if compile_cfg.enabled:
        model = _maybe_compile(model, compile_cfg)

    device_ids = None
    if device.type == "cuda":
        device_ids = [device.index if device.index is not None else torch.cuda.current_device()]

    ddp_model = DDP(
        model,
        device_ids=device_ids,
        gradient_as_bucket_view=ddp_cfg.gradient_as_bucket_view,
        static_graph=ddp_cfg.static_graph,
        bucket_cap_mb=ddp_cfg.bucket_cap_mb,
        find_unused_parameters=ddp_cfg.find_unused_parameters,
    )
    return ddp_model


class MetricsLogger:
    def __init__(self, output_dir: Path, rank: int) -> None:
        self.path = output_dir / f"rank_{rank:02d}_metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def log(self, payload: Dict[str, Any]) -> None:
        self.handle.write(json.dumps(payload, default=self._serialize) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    @staticmethod
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return obj


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}


def compute_loss(scores: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    target = batch["target"].to(scores.dtype)
    importance = batch["importance"].to(scores.dtype)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, target, weight=importance)
    return loss.mean()


def collect_rocm_metrics(enabled: bool) -> Dict[str, Any]:
    if not enabled or detect_accelerator() != "amd":
        return {}
    try:
        result = subprocess.run(
            ["rocm-smi", "--showtemp", "--showuse", "--showpower", "--showmemuse"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"rocm_smi_error": "rocm-smi not available"}

    metrics: Dict[str, Any] = {"rocm_smi_exit_code": result.returncode}
    if result.stdout:
        metrics["rocm_smi_output"] = result.stdout.strip()
    if result.stderr:
        metrics["rocm_smi_stderr"] = result.stderr.strip()
    return metrics


def setup_signal_handlers(stop_flag: Dict[str, bool]) -> None:
    def _handle(signum, frame):  # pragma: no cover - signal handling
        stop_flag["stop"] = True
        log.warning("Received signal %s; will stop after current iteration", signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle)


def training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    training_cfg: TrainingConfig,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    environment: Dict[str, Any],
    profiler: StreamProfiler,
    enable_rocm_metrics: bool,
    profiler_cfg: ProfilerConfig,
) -> None:
    rank = environment["rank"]
    world_size = environment["world_size"]
    device = environment["device"]

    scaler: Optional[torch.cuda.amp.GradScaler]
    autocast_dtype: Optional[torch.dtype]
    mp_mode = training_cfg.mixed_precision.lower()
    if mp_mode == "fp16":
        autocast_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    elif mp_mode == "bf16":
        autocast_dtype = torch.bfloat16
        scaler = None
    else:
        autocast_dtype = None
        scaler = None

    metrics_logger = MetricsLogger(training_cfg.output_dir, rank)

    total_steps = training_cfg.max_steps or len(dataloader) * training_cfg.epochs
    global_step = 0
    stop_flag = {"stop": False}
    setup_signal_handlers(stop_flag)

    model.train()

    profiler_dir = training_cfg.output_dir / "torch_profiler"
    with profiler.intercept_distributed_ops():
        with _torch_profiler_context(profiler_cfg, profiler_dir, rank, device) as torch_profiler:
            for epoch in range(training_cfg.epochs):
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(epoch)

                for step, cpu_batch in enumerate(dataloader):
                    profiler.start_iteration(global_step)

                    with profiler.range("aux", f"epoch{epoch}_step{step}_prefetch"):
                        batch = move_batch_to_device(cpu_batch, device)

                    profiler.stream("compute").wait_stream(profiler.stream("aux"))

                    optimizer.zero_grad(set_to_none=True)

                    with profiler.range("compute", f"epoch{epoch}_step{step}_forward"):
                        if autocast_dtype:
                            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                                scores = model(batch)
                                loss = compute_loss(scores, batch)
                        else:
                            scores = model(batch)
                            loss = compute_loss(scores, batch)

                    with profiler.range("compute", f"epoch{epoch}_step{step}_backward"):
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    grad_norm = None
                    if training_cfg.grad_clip_norm is not None and training_cfg.grad_clip_norm > 0:
                        with profiler.range("aux", f"epoch{epoch}_step{step}_grad_clip"):
                            grad_norm = clip_grad_norm_(model.parameters(), training_cfg.grad_clip_norm)

                    with profiler.range("aux", f"epoch{epoch}_step{step}_optimizer"):
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                    # Inject all_reduce operations to trigger hang pattern
                    # Pattern: all_reduce → device-to-device copy → host-device copy → compute blocked
                    if training_cfg.inject_allreduce_copies:
                        import torch.distributed as dist
                        if dist.is_initialized():
                            with profiler.range("aux", f"epoch{epoch}_step{step}_allreduce_sync"):
                                # Perform multiple all_reduce + memory copy cycles
                                # This stresses the pattern: all_reduce → device copies → hipMemcpyWithStream → rocprim deadlock
                                stress_level = min(max(training_cfg.allreduce_stress_level, 1), 10)

                                for i in range(stress_level):
                                    # Create moderately-sized tensors to stress RCCL and memory copy
                                    # Size: ~4MB per tensor
                                    tensor_size = 1024 * 1024  # 1M elements = 4MB in FP32
                                    stress_tensor = torch.randn(tensor_size, device=device, dtype=torch.float32)

                                    # All-reduce operation (collective that triggers RCCL multi-stream)
                                    dist.all_reduce(stress_tensor, op=dist.ReduceOp.AVG)

                                    # Device-to-device copy (triggers hipMemcpyAsync device-to-device)
                                    # This happens when moving data between GPU memory regions or during P2P transfers
                                    device_copy_1 = stress_tensor.clone()  # Explicit device copy
                                    device_copy_2 = device_copy_1.contiguous()  # May trigger another copy if not contiguous

                                    # Force device-to-device copy via different tensor
                                    temp_storage = torch.empty_like(device_copy_2)
                                    temp_storage.copy_(device_copy_2, non_blocking=False)  # Blocking device-to-device copy

                                    # All-reduce on device-copied tensor
                                    dist.all_reduce(temp_storage, op=dist.ReduceOp.SUM)

                                    # Force blocking host-device copy (triggers hipMemcpyWithStream)
                                    # This is the pattern that causes the hang according to customer
                                    tensor_cpu = temp_storage.cpu()  # Device → Host copy
                                    tensor_back = tensor_cpu.to(device, non_blocking=False)  # Host → Device blocking copy

                                    # Additional all_reduce on the copied-back tensor
                                    dist.all_reduce(tensor_back, op=dist.ReduceOp.AVG)

                                    # More device-to-device copies after all-reduce
                                    final_copy = tensor_back.clone()
                                    _ = final_copy.contiguous()

                                    # Clean up
                                    del stress_tensor, device_copy_1, device_copy_2, temp_storage
                                    del tensor_cpu, tensor_back, final_copy

                                # Also all-reduce actual metrics (common pattern) with device copies
                                loss_tensor = loss.detach().clone()
                                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

                                # Device-to-device copy
                                loss_device_copy = loss_tensor.clone()

                                # Host-device copy
                                loss_cpu = loss_device_copy.cpu()
                                _ = loss_cpu.to(device, non_blocking=False)

                    profiler.record_marker("compute", f"epoch{epoch}_step{step}_end")

                    iteration_profile = profiler.end_iteration()

                    iteration_payload = {
                        "rank": rank,
                        "world_size": world_size,
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "loss": float(loss.detach().cpu()),
                        "grad_norm": float(grad_norm.cpu()) if grad_norm is not None else None,
                        "lr": optimizer.param_groups[0]["lr"],
                        "profile": iteration_profile,
                    }

                    iteration_payload.update(collect_rocm_metrics(enable_rocm_metrics))
                    metrics_logger.log(iteration_payload)

                    if global_step % training_cfg.log_interval == 0 and rank == 0:
                        log.info(
                            "epoch=%s step=%s loss=%.5f lr=%.6f overlap=%.3fms compute=%.3fms",
                            epoch,
                            step,
                            iteration_payload["loss"],
                            iteration_payload["lr"],
                            iteration_profile["overlap"]["overlap_ms"].get("compute_comm", 0.0),
                            iteration_profile["overlap"]["per_stream_ms"].get("compute", 0.0),
                        )

                    if torch_profiler is not None:
                        torch_profiler.step()

                    global_step += 1
                    if training_cfg.max_steps and global_step >= training_cfg.max_steps:
                        stop_flag["stop"] = True

                    if stop_flag["stop"]:
                        break

                if stop_flag["stop"]:
                    break

    metrics_logger.close()


def configure_optimizer(model: nn.Module, cfg: OptimizerConfig, dist_mode: str = "ddp") -> torch.optim.Optimizer:
    if cfg.name.lower() == "shampoo":
        from distributed_shampoo import DistributedShampoo, DDPDistributedConfig

        # Configure distributed Shampoo for multi-GPU training
        distributed_config = DDPDistributedConfig(
            communication_dtype=torch.float32,
            num_trainers_per_group=-1,  # Use all ranks in the group
            communicate_params=False,
        )
        log.info("Using DistributedShampoo optimizer with DDPDistributedConfig")
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            epsilon=cfg.eps,
            weight_decay=cfg.weight_decay,
            distributed_config=distributed_config,
        )
    else:
        log.info("Using AdamW optimizer")
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps)
    return optimizer


def configure_scheduler(optimizer: torch.optim.Optimizer, cfg: SchedulerConfig, total_steps: int):
    if cfg.total_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return float(step + 1) / float(max(1, cfg.warmup_steps))
        progress = min(1.0, float(step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _maybe_compile(module: FSDP, cfg: CompileConfig) -> FSDP:
    compile_fn = getattr(torch, "compile", None)
    if not cfg.enabled:
        return module
    if compile_fn is None:  # pragma: no cover - defensive when torch.compile missing
        log.warning("torch.compile not available in this PyTorch build; skipping compilation")
        return module

    kwargs: Dict[str, Any] = {}
    backend = cfg.backend or "inductor"
    accelerator = detect_accelerator()
#    if accelerator == "amd" and backend in {"inductor", "inductor_dynamic"}:
#        log.warning("torch.compile backend '%s' is experimental on ROCm; disabling compilation", backend)
#        return module
    kwargs["backend"] = backend
    if cfg.mode:
        kwargs["mode"] = cfg.mode
    kwargs["fullgraph"] = cfg.fullgraph
    kwargs["dynamic"] = cfg.dynamic
    if cfg.options:
        kwargs["options"] = cfg.options

    try:
        compiled = compile_fn(module, **kwargs)
        log.info(
            "Enabled torch.compile | backend=%s mode=%s fullgraph=%s dynamic=%s",
            cfg.backend,
            cfg.mode,
            cfg.fullgraph,
            cfg.dynamic,
        )
        return compiled
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("torch.compile failed (%s); continuing with eager module", exc, exc_info=True)
        return module


def _restore_rocm_profiler_env() -> None:
    keys = [
        "ROCPROFILER_LOG_LEVEL",
        "ROCPROFILER_ENABLE_TRACING",
        "ROCPROFILER_KERNEL_TIMESTAMPS",
        "ROCPROFILER_DEVICE_CLOCK_SYNC",
    ]
    for key in keys:
        if key in os.environ:
            os.environ.pop(key, None)


@contextlib.contextmanager
def _torch_profiler_context(
    cfg: ProfilerConfig, output_dir: Path, rank: int, device: torch.device
) -> Generator[Optional[profile], None, None]:
    if not cfg.enabled:
        yield None
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_dir = output_dir / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    if detect_accelerator() == "amd":
        _restore_rocm_profiler_env()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    schedule_obj = schedule(wait=cfg.wait, warmup=cfg.warmup, active=cfg.active, repeat=cfg.repeat)

    log.info(
        "Enabling torch profiler | rank=%s wait=%s warmup=%s active=%s repeat=%s",
        rank,
        cfg.wait,
        cfg.warmup,
        cfg.active,
        cfg.repeat,
    )

    prof = profile(
        activities=activities,
        schedule=schedule_obj,
        record_shapes=cfg.record_shapes,
        profile_memory=cfg.profile_memory,
        with_stack=cfg.with_stack,
        with_flops=cfg.with_flops,
    )

    try:
        prof.__enter__()
        yield prof
    finally:
        prof.__exit__(None, None, None)
        produce_tb = cfg.tensorboard
        produce_chrome = cfg.chrome_trace
        try:
            stats_available = prof._stats() is not None  # type: ignore[attr-defined]
        except Exception:
            stats_available = False

        if produce_tb and stats_available:
            try:
                handler = tensorboard_trace_handler(str(rank_dir))
                handler(prof)
            except Exception as exc:  # pragma: no cover - best effort
                log.warning("TensorBoard trace export failed: %s", exc, exc_info=True)

        if produce_chrome and stats_available:
            stem, ext = os.path.splitext(cfg.trace_filename)
            if not ext:
                ext = ".json"
            trace_name = f"{stem}{ext}"
            if cfg.repeat != 1 or cfg.active > 1:
                trace_name = f"{stem}_step{prof.step_num}{ext}"
            try:
                prof.export_chrome_trace(str(rank_dir / trace_name))
            except Exception as exc:  # pragma: no cover - best effort
                log.warning("Chrome trace export failed: %s", exc, exc_info=True)


def main_cli() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="FSDP2 training benchmark with multi-stream profiling")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="Configuration overrides as dotted key=value entries",
    )
    parser.add_argument("--enable-rocm-metrics", action="store_true", help="Collect rocm-smi metrics")
    args = parser.parse_args()
    main(args, enable_rocm_metrics=args.enable_rocm_metrics)


def main(args: Optional[argparse.Namespace] = None, *, enable_rocm_metrics: bool = False) -> None:
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument("--override", type=str, nargs="*", default=None)
        parsed = parser.parse_args()
    else:
        parsed = args

    config = _parse_config(parsed)
    training_cfg = _build_training_config(config)
    optimizer_cfg = _build_optimizer_config(config)
    scheduler_cfg = _build_scheduler_config(config)
    model_cfg = _build_model_config(config)
    dataset_cfg = _build_dataset_config(config)
    fsdp_cfg = _build_fsdp_config(config)
    ddp_cfg = _build_ddp_config(config)
    compile_cfg = _build_compile_config(config)
    profiler_cfg = _build_profiler_config(config)

    log_level = config.get("logging", {}).get("level", "INFO")
    env = init_distributed(training_cfg, log_level)
    rank = env["rank"]

    dataloader = create_dataloader(
        dataset_cfg,
        batch_size=training_cfg.batch_size,
        world_size=env["world_size"],
        rank=rank,
        num_workers=config.get("dataloader", {}).get("num_workers", 4),
        pin_memory=config.get("dataloader", {}).get("pin_memory", True),
    )

    dist_mode = config.get("distributed", {}).get("mode")
    if dist_mode is None:
        dist_mode = "fsdp"
    dist_mode = dist_mode.lower()

    if dist_mode == "ddp":
        model = build_ddp_model(model_cfg, ddp_cfg, compile_cfg, env["device"])
    else:
        model = build_fsdp_model(model_cfg, fsdp_cfg, compile_cfg, env["device"])
    optimizer = configure_optimizer(model, optimizer_cfg, dist_mode)
    scheduler = configure_scheduler(
        optimizer,
        scheduler_cfg,
        training_cfg.max_steps or training_cfg.epochs * len(dataloader),
    )

    profiler = StreamProfiler(env["device"])

    try:
        training_loop(
            model,
            optimizer,
            dataloader,
            training_cfg,
            scheduler,
            env,
            profiler,
            enable_rocm_metrics,
            profiler_cfg,
        )
    finally:
        dist.barrier()
        dist.destroy_process_group()


__all__ = ["main", "main_cli"]
