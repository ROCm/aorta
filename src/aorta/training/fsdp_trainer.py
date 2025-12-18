"""FSDP2 multi-stream training benchmark with profiling."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler, profile
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
from aorta.data import SyntheticDatasetConfig, create_dataloader
from aorta.models import ModelConfig, RankingTransformerModel
from aorta.profiling.stream_profiler import StreamProfiler
from aorta.training.nan_debugger import NaNDebugger
from aorta.utils import detect_accelerator, get_device, get_distributed_backend, load_config, merge_cli_overrides, setup_logging

log = logging.getLogger(__name__)


class TrainingAbortError(RuntimeError):
    """Raised to abort training quickly across ranks (e.g., fatal optimizer assertion)."""


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
    nan_check: bool = False  # Enable expensive NaN/Inf diagnostics (debug only)
    nan_check_interval: int = 1  # Run NaN/Inf checks every N steps when enabled


@dataclass
class FSDPConfig:
    sharding_strategy: str = "full_shard"
    backward_prefetch: str = "BACKWARD_PRE"
    use_orig_params: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = True
    sync_module_states: bool = True
    param_init_device: str = "cpu"
    # For HYBRID_SHARD: GPUs per node (None = auto-detect from LOCAL_WORLD_SIZE env var)
    # Only set this if auto-detection fails or you want to override
    hybrid_shard_gpus_per_node: Optional[int] = None


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


def set_seed(seed: int, rank: int) -> None:
    """Set all random seeds for reproducibility across runs."""
    seed_value = seed + rank
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    log.info("Set random seed=%d for rank=%d (base_seed=%d)", seed_value, rank, seed)


def init_distributed(training_cfg: TrainingConfig, log_level: str) -> Dict[str, Any]:
    
    backend = get_distributed_backend()
    # Reduce timeout for faster debugging of hangs (default 600s -> 120s)
    # This timeout applies to ALL collective operations, not just initialization
    timeout_seconds = int(os.environ.get("TORCH_DIST_INIT_TIMEOUT", "120"))
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
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

    # Create process groups for hybrid_shard strategy
    process_group = None
    if sharding == ShardingStrategy.HYBRID_SHARD:
        process_group = _create_hybrid_shard_process_groups(fsdp_cfg.hybrid_shard_gpus_per_node)
        if process_group is not None:
            log.info("Created custom process groups for HYBRID_SHARD strategy")
            # #region agent log
            rank = dist.get_rank() if dist.is_initialized() else -1
            shard_pg, replicate_pg = process_group
            log.info("[DEBUG_H3] Created HYBRID_SHARD process groups | rank=%d shard_pg_size=%d replicate_pg_size=%d", rank, dist.get_world_size(shard_pg), dist.get_world_size(replicate_pg))
            # #endregion

    fsdp_model = FSDP(
        model.to(device),
        sharding_strategy=sharding,
        process_group=process_group,
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


def _create_hybrid_shard_process_groups(gpus_per_node: Optional[int] = None):
    """
    Create process groups for HYBRID_SHARD strategy.

    Args:
        gpus_per_node: Number of GPUs per node. If None, auto-detects from LOCAL_WORLD_SIZE.
                       This should match the --nproc value from torchrun.

    Returns:
        Tuple of (shard_group, replicate_group) or None
    """
    if not dist.is_initialized():
        return None

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Auto-detect GPUs per node from environment if not provided
    if gpus_per_node is None:
        # torchrun sets LOCAL_WORLD_SIZE to the number of processes per node
        local_world_size_str = os.environ.get("LOCAL_WORLD_SIZE")
        if local_world_size_str:
            gpus_per_node = int(local_world_size_str)
            log.info("Auto-detected gpus_per_node=%d from LOCAL_WORLD_SIZE", gpus_per_node)
        else:
            log.error(
                "Cannot determine gpus_per_node: LOCAL_WORLD_SIZE not set and "
                "hybrid_shard_gpus_per_node not configured. Set hybrid_shard_gpus_per_node in config."
            )
            return None

    # Validate configuration
    if world_size % gpus_per_node != 0:
        log.error(
            "Invalid HYBRID_SHARD config: world_size=%d not divisible by gpus_per_node=%d",
            world_size, gpus_per_node
        )
        return None

    num_nodes = world_size // gpus_per_node
    node_id = rank // gpus_per_node

    if num_nodes <= 1:
        log.warning("HYBRID_SHARD with single node - consider using FULL_SHARD instead")
        return None

    log.info(
        "Creating HYBRID_SHARD process groups | rank=%d world_size=%d num_nodes=%d gpus_per_node=%d node_id=%d",
        rank, world_size, num_nodes, gpus_per_node, node_id
    )

    # Intra-node groups: shard within each node
    for i in range(num_nodes):
        ranks_in_node = list(range(i * gpus_per_node, (i + 1) * gpus_per_node))
        group = dist.new_group(ranks=ranks_in_node)
        if i == node_id:
            my_shard_group = group

    # Inter-node groups: replicate across nodes (same local_rank)
    for local_r in range(gpus_per_node):
        ranks_across_nodes = [node * gpus_per_node + local_r for node in range(num_nodes)]
        group = dist.new_group(ranks=ranks_across_nodes)
        if local_r == local_rank:
            my_replicate_group = group

    log.info(
        "Created process groups | shard_group_size=%d replicate_group_size=%d",
        dist.get_world_size(my_shard_group),
        dist.get_world_size(my_replicate_group),
    )

    return (my_shard_group, my_replicate_group)


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
    nan_debugger: Optional[NaNDebugger] = None,
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
    nan_failure_flag = {"detected": False, "step": -1, "rank": -1}
    setup_signal_handlers(stop_flag)

    model.train()

    profiler_dir = training_cfg.output_dir / "torch_profiler"
    
    # Create TCPStore for coordinating trace export across ranks on NaN detection
    store = None
    if dist.is_initialized():
        try:
            store = dist.TCPStore(
                host_name=os.environ.get("MASTER_ADDR", "localhost"),
                port=int(os.environ.get("MASTER_PORT", "29500")) + 1,  # Use different port
                world_size=world_size,
                is_master=(rank == 0),
                timeout=timedelta(seconds=10),
            )
            log.info("[Coordination] Created TCPStore for NaN coordination | rank=%d", rank)
        except Exception as e:
            log.warning("[Coordination] Failed to create TCPStore: %s | rank=%d", e, rank)
            store = None
    
    with profiler.intercept_distributed_ops():
        with _torch_profiler_context(profiler_cfg, profiler_dir, rank, device) as torch_profiler:
            for epoch in range(training_cfg.epochs):
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(epoch)

                for step, cpu_batch in enumerate(dataloader):
                    # Check if any rank detected NaN and signaled for trace export (non-blocking)
                    if store is not None and not nan_failure_flag["detected"]:
                        nan_signal = None
                        try:
                            # wait with zero timeout acts as a non-blocking presence check
                            store.wait(["nan_detected"], timeout=0.0)
                            nan_signal = store.get("nan_detected")
                        except Exception:
                            nan_signal = None
                        if nan_signal:
                            signal_data = json.loads(nan_signal.decode('utf-8'))
                            nan_failure_flag["detected"] = True
                            nan_failure_flag["step"] = signal_data.get("step", -1)
                            nan_failure_flag["rank"] = signal_data.get("rank", -1)
                            log.error(
                                "[Coordination] Received NaN signal from rank %d at step %d | current_rank=%d current_step=%d",
                                nan_failure_flag["rank"], nan_failure_flag["step"], rank, global_step
                            )
                            # Export this rank's trace
                            if torch_profiler is not None and profiler_cfg.chrome_trace:
                                try:
                                    trace_file = profiler_dir / f"rank{rank}" / f"nan_coordinated_step{global_step}.json"
                                    trace_file.parent.mkdir(parents=True, exist_ok=True)
                                    log.info("[Profiler] Exporting trace due to NaN signal | rank=%d file=%s", rank, trace_file)
                                    torch_profiler.export_chrome_trace(str(trace_file))
                                    log.info("[Profiler] Trace export completed | rank=%d", rank)
                                except Exception as export_err:
                                    log.warning("[Profiler] Failed to export trace: %s | rank=%d", export_err, rank)
                            # Stop training
                            stop_flag["stop"] = True
                            break
                    
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
                    
                    # Check loss for NaN/Inf with diagnostics
                    if nan_debugger is not None:
                        if nan_debugger.check_loss(loss, global_step):
                            log.error("[NaNDebugger] NaN/Inf detected in loss - stopping training")
                            if torch_profiler is not None and profiler_cfg.chrome_trace and getattr(torch_profiler, "_profiler", None) is not None:
                                try:
                                    trace_file = profiler_dir / f"rank{rank}" / f"nan_loss_step{global_step}.json"
                                    trace_file.parent.mkdir(parents=True, exist_ok=True)
                                    log.info("[Profiler] Exporting trace due to NaN loss | rank=%d file=%s", rank, trace_file)
                                    torch_profiler.export_chrome_trace(str(trace_file))
                                except Exception as export_err:
                                    log.warning("[Profiler] Failed to export trace: %s | rank=%d", export_err, rank)
                            else:
                                log.warning("[Profiler] Skipping NaN loss trace export (profiler inactive) | rank=%d", rank)
                            # Signal other ranks best-effort
                            if store is not None and not nan_failure_flag["detected"]:
                                try:
                                    store.set("nan_detected", json.dumps({"step": global_step, "rank": rank}))
                                except Exception as signal_err:
                                    log.warning("[Coordination] Failed to send NaN signal from loss check: %s", signal_err)
                            stop_flag["stop"] = True
                            break

                    with profiler.range("compute", f"epoch{epoch}_step{step}_backward"):
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    
                    # Check gradients for NaN/Inf with diagnostics
                    if nan_debugger is not None:
                        if nan_debugger.check_gradients(global_step):
                            log.error("[NaNDebugger] NaN/Inf detected in gradients - stopping training")
                            if torch_profiler is not None and profiler_cfg.chrome_trace and getattr(torch_profiler, "_profiler", None) is not None:
                                try:
                                    trace_file = profiler_dir / f"rank{rank}" / f"nan_gradients_step{global_step}.json"
                                    trace_file.parent.mkdir(parents=True, exist_ok=True)
                                    log.info("[Profiler] Exporting trace due to NaN gradients | rank=%d file=%s", rank, trace_file)
                                    torch_profiler.export_chrome_trace(str(trace_file))
                                except Exception as export_err:
                                    log.warning("[Profiler] Failed to export trace: %s | rank=%d", export_err, rank)
                            else:
                                log.warning("[Profiler] Skipping NaN gradient trace export (profiler inactive) | rank=%d", rank)
                            # Signal other ranks best-effort
                            if store is not None and not nan_failure_flag["detected"]:
                                try:
                                    store.set("nan_detected", json.dumps({"step": global_step, "rank": rank}))
                                except Exception as signal_err:
                                    log.warning("[Coordination] Failed to send NaN signal from grad check: %s", signal_err)
                            stop_flag["stop"] = True
                            break

                    grad_norm = None
                    if training_cfg.grad_clip_norm is not None and training_cfg.grad_clip_norm > 0:
                        with profiler.range("aux", f"epoch{epoch}_step{step}_grad_clip"):
                            grad_norm = clip_grad_norm_(model.parameters(), training_cfg.grad_clip_norm)

                   
                    with profiler.range("aux", f"epoch{epoch}_step{step}_optimizer"):
                        # #region agent log
                        log.info("[DEBUG_H1_H4] Before optimizer step | rank=%d global_step=%d stop_flag=%s", rank, global_step, stop_flag["stop"])
                        # #endregion
                        step_error_local = False
                        step_error_msg: Optional[str] = None
                        step_error_exception: Optional[Exception] = None
                        try:
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                        except Exception as e:
                            error_str = str(e)
                            is_nan_inf = ("nan" in error_str.lower()) or ("inf" in error_str.lower())
                            if isinstance(e, AssertionError) or is_nan_inf:
                                # Some optimizers explicitly assert/raise when gradients contain NaN/Inf.
                                # Do not synchronize a stop flag here: the optimizer may have been mid-collective.
                                step_error_local = True
                                step_error_msg = error_str
                                step_error_exception = e
                                
                                # Signal to all other ranks that NaN detected (non-blocking)
                                if store is not None and not nan_failure_flag["detected"]:
                                    try:
                                        signal_data = json.dumps({"step": global_step, "rank": rank})
                                        store.set("nan_detected", signal_data)
                                        nan_failure_flag["detected"] = True
                                        nan_failure_flag["step"] = global_step
                                        nan_failure_flag["rank"] = rank
                                        log.info("[Coordination] Sent NaN signal to all ranks | rank=%d step=%d", rank, global_step)
                                    except Exception as signal_err:
                                        log.warning("[Coordination] Failed to send NaN signal: %s", signal_err)
                                
                                # Export this rank's profiler trace
                                if torch_profiler is not None and profiler_cfg.chrome_trace and getattr(torch_profiler, "_profiler", None) is not None:
                                    try:
                                        trace_file = profiler_dir / f"rank{rank}" / f"nan_failure_step{global_step}.json"
                                        trace_file.parent.mkdir(parents=True, exist_ok=True)
                                        log.info("[Profiler] Exporting trace due to optimizer failure | rank=%d file=%s", rank, trace_file)
                                        torch_profiler.export_chrome_trace(str(trace_file))
                                        log.info("[Profiler] Trace export completed | rank=%d", rank)
                                    except Exception as export_err:
                                        log.warning("[Profiler] Failed to export trace: %s | rank=%d", export_err, rank)
                                else:
                                    log.warning("[Profiler] Skipping optimizer-failure trace export (profiler inactive) | rank=%d", rank)
                                
                                # Investigate root cause when optimizer detects NaN/Inf
                                if nan_debugger is not None:
                                    log.error("[NaNDebugger] Optimizer detected NaN/Inf - investigating root cause | rank=%d", rank)
                                    log.error("[NaNDebugger] Optimizer error: %s", error_str)
                                    # Investigate which gradients/parameters contain NaN/Inf
                                    # These checks validate actual NaN/Inf counts to avoid false positives
                                    nan_debugger.investigate_optimizer_failure(global_step, error_str)
                                
                                # Give other ranks time to see signal and export their traces
                                import time
                                time.sleep(2)  # 2 seconds for other ranks to export traces
                            else:
                                raise

                        # #region agent log
                        log.info("[DEBUG_H4] After optimizer step | rank=%d global_step=%d step_error_local=%s step_error_msg=%s", rank, global_step, step_error_local, step_error_msg)
                        # #endregion
                        if step_error_local:
                            # #region agent log
                            log.error("[DEBUG_H4] Raising TrainingAbortError - will skip stress injection | rank=%d global_step=%d", rank, global_step)
                            # #endregion
                            log.error(
                                "Fatal optimizer assertion; aborting training | rank=%d step=%d error=%s",
                                rank,
                                global_step,
                                step_error_msg,
                            )
                            log.info("[DEBUG_RAISE] About to raise TrainingAbortError | rank=%d", rank)
                            raise TrainingAbortError(step_error_msg or "optimizer assertion") from step_error_exception

                        if scheduler is not None:
                            scheduler.step()

                    # Inject all_reduce operations to trigger hang pattern
                    # Pattern: all_reduce → device-to-device copy → host-device copy → compute blocked
                    if training_cfg.inject_allreduce_copies:
                        if dist.is_initialized():
                            # #region agent log
                            log.info("[DEBUG_H1_H4] Entering stress injection block | rank=%d global_step=%d world_size=%d", rank, global_step, world_size)
                            # #endregion
                            with profiler.range("aux", f"epoch{epoch}_step{step}_allreduce_sync"):
                                # Perform multiple all_reduce + memory copy cycles
                                # This stresses the pattern: all_reduce → device copies → hipMemcpyWithStream → rocprim deadlock
                                stress_level = min(max(training_cfg.allreduce_stress_level, 1), 10)

                                for i in range(stress_level):
                                    # #region agent log
                                    if i == 0 or i == stress_level - 1:
                                        log.info("[DEBUG_H1_H5] Stress iteration %d/%d | rank=%d global_step=%d", i, stress_level, rank, global_step)
                                    # #endregion
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
                            # #region agent log
                            log.info("[DEBUG_H1_H4] Completed stress injection block | rank=%d global_step=%d stress_level=%d", rank, global_step, stress_level)
                            # #endregion

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

                    loss_log = training_cfg.output_dir / f"loss_rank{rank}.log"
                    with open(loss_log, "a") as f:
                        f.write(f"step={global_step} epoch={epoch} loss={iteration_payload['loss']:.6f} lr={iteration_payload['lr']:.6f}\n")

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
                    if rank == 0:
                        log.info("Training stopped at epoch=%d step=%d", epoch, step)
                    break

    metrics_logger.close()
    # #region agent log
    log.info("[INSTR_H1_H2_H3] Exiting training_loop normally | rank=%d", rank)
    try:
        with open("/apps/oyazdanb/.cursor/debug.log", "a") as f:
            f.write(json.dumps({"location": "fsdp_trainer.py:889", "message": "Exiting training_loop normally", "data": {"rank": rank}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "hypothesisId": "H1_H2_H3"}) + "\n")
    except: pass
    # #endregion
    
    if rank == 0:
        log.info("Training loop finished. Profiler will export traces in cleanup phase.")
        log.info("Output directory: %s", training_cfg.output_dir)
        log.info("Torch profiler traces: %s", training_cfg.output_dir / "torch_profiler")
        log.info("Loss logs: %s/loss_rank*.log", training_cfg.output_dir)
        log.info("Metrics: %s/rank_*_metrics.jsonl", training_cfg.output_dir)


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
        # #region agent log
        rank = dist.get_rank() if dist.is_initialized() else -1
        world_size = dist.get_world_size() if dist.is_initialized() else -1
        log.info("[DEBUG_H3] Creating DistributedShampoo optimizer | rank=%d world_size=%d dist_mode=%s num_trainers_per_group=-1", rank, world_size, dist_mode)
        # #endregion
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
    def _profiler_active(prof_obj: profile) -> bool:
        return getattr(prof_obj, "_profiler", None) is not None

    def _export_traces(prof_obj: profile, rank_dir: Path) -> None:
        try:
            stats_available = prof_obj._stats() is not None  # type: ignore[attr-defined]
        except Exception:
            stats_available = False

        if not stats_available or not _profiler_active(prof_obj):
            log.warning("Torch profiler inactive or stats unavailable; skipping trace export | rank=%d", rank)
            return

        if cfg.chrome_trace:
            stem, ext = os.path.splitext(cfg.trace_filename)
            if not ext:
                ext = ".json"
            trace_name = f"{stem}{ext}"
            try:
                prof_obj.export_chrome_trace(str(rank_dir / trace_name))
                log.info("Exported chrome trace to %s/%s", rank_dir, trace_name)
            except Exception as exc:
                log.warning("Chrome trace export failed: %s", exc, exc_info=True)

        if cfg.tensorboard:
            try:
                handler = tensorboard_trace_handler(str(rank_dir))
                handler(prof_obj)
                log.info("Exported TensorBoard trace to %s", rank_dir)
            except Exception as exc:
                log.warning("TensorBoard trace export failed: %s", exc, exc_info=True)

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

    # Disable profiler trace export to avoid delays during exception handling
    # Just enter/exit the profiler and let exceptions propagate naturally
    log.info("[INSTR_H2] About to enter torch profiler context | rank=%d", rank)
    try:
        prof.__enter__()
        log.info("[INSTR_H2] Inside torch profiler, about to yield | rank=%d", rank)
        yield prof
        log.info("[INSTR_H2] Back from yield, exiting normally | rank=%d", rank)
    except Exception as prof_exc:
        log.error("[INSTR_H2] Torch profiler caught exception | rank=%d exc_type=%s", rank, type(prof_exc).__name__)
        raise
    finally:
        log.info("[INSTR_H2] Torch profiler finally block | rank=%d", rank)
        # Always clean up profiler, even on exception
        try:
            prof.__exit__(None, None, None)
        except Exception:
            pass  # Ignore cleanup errors
        _export_traces(prof, rank_dir)


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

    set_seed(dataset_cfg.seed, rank)

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
    
    # Initialize NaN debugger for automatic root cause analysis
    nan_debugger = NaNDebugger(
        model=model,
        optimizer=optimizer,
        config=config,
        output_dir=str(training_cfg.output_dir / "nan_diagnostics"),
        rank=rank,
        enabled=True,  # Always enabled for automatic diagnostics
    )
    log.info("[NaNDebugger] Initialized for automatic NaN detection and diagnosis")

    had_fatal_error = False
    # #region agent log
    log.info("[INSTR_H3] About to call training_loop | rank=%d", env["rank"])
    try:
        with open("/apps/oyazdanb/.cursor/debug.log", "a") as f:
            f.write(json.dumps({"location": "fsdp_trainer.py:1145", "message": "About to call training_loop", "data": {"rank": env["rank"]}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "hypothesisId": "H3"}) + "\n")
    except: pass
    # #endregion
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
            nan_debugger,
        )
    except TrainingAbortError as e:
        # #region agent log
        log.info("[INSTR_H3] Caught TrainingAbortError in main() except block | rank=%d", env["rank"])
        try:
            with open("/apps/oyazdanb/.cursor/debug.log", "a") as f:
                f.write(json.dumps({"location": "fsdp_trainer.py:1157", "message": "Caught TrainingAbortError in main", "data": {"rank": env["rank"], "error": str(e)[:200]}, "timestamp": __import__("time").time() * 1000, "sessionId": "debug-session", "hypothesisId": "H3"}) + "\n")
        except: pass
        # #endregion
        log.info("[DEBUG_CATCH] Caught TrainingAbortError in main() | rank=%d error=%s", env["rank"], str(e)[:100])
        had_fatal_error = True
        raise
    finally:
        if dist.is_initialized():
            # #region agent log
            rank = dist.get_rank() if dist.is_initialized() else -1
            log.info("[DEBUG_H2_H4] Entering cleanup finally block | rank=%d had_fatal_error=%s", rank, had_fatal_error)
            # #endregion
            if had_fatal_error:
                # Best-effort fast shutdown on fatal rank-local errors (e.g., optimizer assertions).
                # Avoid barrier: it can hang if ranks diverged mid-collective.
                # #region agent log
                log.info("[DEBUG_H4] Fatal error path - calling dist.abort | rank=%d", rank)
                # #endregion
                abort_fn = getattr(dist, "abort", None)
                if callable(abort_fn):
                    try:
                        abort_fn()
                    except Exception as e:
                        log.warning("dist.abort() failed during cleanup: %s", e)
            else:
                # #region agent log
                log.info("[DEBUG_H2] Normal cleanup - calling barrier | rank=%d", rank)
                # #endregion
                try:
                    dist.barrier()
                except Exception as e:
                    log.warning("Barrier failed during cleanup: %s", e)
            try:
                dist.destroy_process_group()
            except Exception as e:
                log.warning("destroy_process_group failed: %s", e)


__all__ = ["main", "main_cli"]
