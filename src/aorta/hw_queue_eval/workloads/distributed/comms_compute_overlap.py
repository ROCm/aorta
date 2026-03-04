"""
Comms-Compute Overlap Workload

Pattern: Overlap collective communication with compute (GEMM) on separate CUDA streams.
Supports three modes:
  - compute_only:  Run only compute kernels (no communication)
  - comms_only:    Run only collective operations (no compute)
  - comms_compute: Overlap communication and compute on independent streams

Stream assignment (comms_compute mode with compute_streams=N):
  - Stream 0: Communication (collective operations)
  - Streams 1..N: Compute (GEMM operations distributed round-robin)

The number of compute streams can be set independently of the total stream
count passed by the harness via the ``compute_streams`` parameter.

When ``simulate_collectives=False`` the workload uses real
``torch.distributed`` collectives (NCCL / RCCL) and must be launched via
``torchrun``.  Each rank runs on ``cuda:{LOCAL_RANK}``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.profiler import record_function

logger = logging.getLogger(__name__)

from aorta.hw_queue_eval.workloads.base import DistributedWorkload, MultiGPUMixin
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry
from aorta.utils.distributed import (
    cleanup_distributed,
    create_process_groups,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    parse_process_groups,
)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(value) -> torch.dtype:
    """Accept a ``torch.dtype`` or a string name and return a ``torch.dtype``."""
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.lower().strip()
        if key in _DTYPE_MAP:
            return _DTYPE_MAP[key]
        raise ValueError(
            f"Unsupported dtype string '{value}'. "
            f"Supported: {list(_DTYPE_MAP.keys())}"
        )
    raise TypeError(f"Expected torch.dtype or str, got {type(value)}")


@WorkloadRegistry.register
class CommsComputeOverlapWorkload(MultiGPUMixin, DistributedWorkload):
    """
    Configurable comm-compute overlap benchmark.

    Dispatches collective communication and GEMM compute on separate CUDA
    streams so that both can execute concurrently.  The workload is fully
    parameterised so callers can sweep matrix sizes, compute-to-comm ratios,
    and stream counts to characterise overlap behaviour.
    """

    name = "comms_compute_overlap"
    description = "Comm-compute overlap with configurable GEMM and collectives"
    category = "distributed"
    min_streams = 1
    max_streams = 32
    recommended_streams = 4
    switch_latency_sensitivity = "high"
    memory_requirements_gb = 2.0
    multi_gpu_capable = True

    # -- Construction --------------------------------------------------------

    def __init__(
        self,
        mode: str = "comms_compute",
        kernel: str = "gemm",
        mm_dim: Tuple[int, int, int] = (2048, 2048, 2048),
        num_compute_per_iter: int = 10,
        num_coll_per_iter: int = 1,
        collective: str = "all_reduce",
        comm_size_bytes: int = 128 * 1024 * 1024,  # 128 MiB default
        simulate_collectives: bool = True,
        async_op: bool = False,
        backend: str = "nccl",
        process_groups: Optional[str] = None,
        compute_streams: Optional[int] = None,
        comp_data_type: str | torch.dtype = torch.float32,
        comm_data_type: str | torch.dtype = torch.float32,
        use_multi_gpu: bool = True,
    ):
        """
        Args:
            mode: One of ``"compute_only"``, ``"comms_only"``, ``"comms_compute"``.
            kernel: Compute kernel; currently only ``"gemm"`` (expandable later).
            mm_dim: ``(M, N, K)`` dimensions for the GEMM  A[M,K] x B[K,N].
            num_compute_per_iter: Number of GEMM ops per iteration per compute stream.
            num_coll_per_iter: Number of collective ops per iteration on the comm stream.
            collective: Collective op; currently only ``"all_reduce"`` (expandable later).
            comm_size_bytes: Size (in bytes) of the tensor used for the collective.
            simulate_collectives: If ``True`` (default), mock the collective; else use
                real ``torch.distributed`` collectives (requires ``torchrun``).
            async_op: If ``True``, issue non-blocking collectives and wait at the
                end of the iteration.
            backend: Distributed backend (``"nccl"`` or ``"gloo"``).
            process_groups: Process-group specification string, e.g.
                ``"[0,1,2,3],[4,5,6,7]"``.
            compute_streams: Number of compute streams.  When ``None`` (default)
                the count is derived from the total stream count (all streams
                minus the comm stream in ``comms_compute`` mode).  Set explicitly
                to decouple compute parallelism from the harness stream count.
            comp_data_type: Data-type for compute (GEMM) tensors.  Accepts a
                ``torch.dtype`` or a string like ``"float32"``, ``"bfloat16"``,
                ``"float16"``.
            comm_data_type: Data-type for the communication tensor.
            use_multi_gpu: Distribute work across all visible GPUs (simulation
                mode only).
        """
        if mode not in ("compute_only", "comms_only", "comms_compute"):
            raise ValueError(
                f"mode must be one of 'compute_only', 'comms_only', "
                f"'comms_compute', got '{mode}'"
            )
        if kernel != "gemm":
            raise ValueError(f"kernel must be 'gemm' (expandable later), got '{kernel}'")
        if collective != "all_reduce":
            raise ValueError(
                f"collective must be 'all_reduce' (expandable later), got '{collective}'"
            )

        super().__init__(simulate_collectives=simulate_collectives)

        self.mode = mode
        self.kernel = kernel
        self.mm_dim = mm_dim
        self.num_compute_per_iter = num_compute_per_iter
        self.num_coll_per_iter = num_coll_per_iter
        self.collective = collective
        self.comm_size_bytes = comm_size_bytes
        self.use_multi_gpu = use_multi_gpu
        self._requested_compute_streams: Optional[int] = compute_streams
        self.comp_data_type: torch.dtype = _resolve_dtype(comp_data_type)
        self.comm_data_type: torch.dtype = _resolve_dtype(comm_data_type)

        # Distributed-specific parameters
        self._async_op: bool = async_op
        self._backend: str = backend
        self._pg_spec: Optional[str] = process_groups

        # Populated during setup()
        self._A: List[torch.Tensor] = []
        self._B: List[torch.Tensor] = []
        self._C: List[torch.Tensor] = []
        self._comm_tensor: torch.Tensor = torch.empty(0)
        self._comm_scratch: torch.Tensor = torch.empty(0)
        self._num_compute_streams: int = 0
        self._comm_stream_idx: int = 0

        # Distributed state (populated in setup when simulate_collectives=False)
        self._process_groups: Dict[int, dist.ProcessGroup] = {}
        self._active_group: Optional[dist.ProcessGroup] = None
        self._pending_ops: List[dist.Work] = []
        self._rank: int = 0
        self._world_size: int = 1

    # -- Workload interface --------------------------------------------------

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Allocate tensors and decide stream layout."""
        self._stream_count = stream_count
        self._is_setup = True

        # --- Distributed initialisation (real collectives only) ---
        if not self._simulate_collectives:
            init_distributed(backend=self._backend)
            self._rank = get_rank()
            self._world_size = get_world_size()
            local_rank = get_local_rank()
            device = f"cuda:{local_rank}"

            self._setup_multi_gpu(stream_count, device, use_multi_gpu=False)

            # Process groups -- overlapping groups are allowed (e.g.
            # the same ranks can appear in multiple groups).  The workload
            # uses the first group that contains this rank as the active
            # group for collectives.
            if self._pg_spec is not None:
                pg_ranks = parse_process_groups(self._pg_spec)

                # Validate that all ranks are within [0, world_size)
                for pg_id, ranks in pg_ranks.items():
                    for r in ranks:
                        if not (0 <= r < self._world_size):
                            raise RuntimeError(
                                f"Invalid process group spec: rank {r} is out of "
                                f"range for world size {self._world_size} "
                                f"(valid ranks: 0..{self._world_size - 1})."
                            )

                self._process_groups = create_process_groups(
                    pg_ranks, backend=self._backend
                )

                # Pick the first group this rank belongs to
                for pg_id, ranks in pg_ranks.items():
                    if self._rank in ranks:
                        self._active_group = self._process_groups[pg_id]
                        break

                if self._active_group is None:
                    raise RuntimeError(
                        f"Rank {self._rank} does not belong to any of the "
                        f"specified process groups: {self._pg_spec}"
                    )
            else:
                self._active_group = dist.group.WORLD
        else:
            self._setup_multi_gpu(stream_count, device, self.use_multi_gpu)

        # --- Stream layout ---
        self._comm_stream_idx = 0

        if self._requested_compute_streams is not None:

            self._num_compute_streams = self._requested_compute_streams
            # Warn if compute streams exceed available harness streams
            avail = stream_count if self.mode == "compute_only" else max(0, stream_count - 1)
            if avail > 0 and self._num_compute_streams > avail:
                logger.warning(
                    "compute_streams=%d exceeds available streams=%d; "
                    "multiple GEMM sets will serialize on shared streams.",
                    self._num_compute_streams,
                    avail,
                )
        elif self.mode == "comms_only":
            self._num_compute_streams = 0
        elif self.mode == "compute_only":
            self._num_compute_streams = stream_count
        else:
            self._num_compute_streams = max(1, stream_count - 1)

        m, n, k = self.mm_dim

        # --- Allocate compute tensors (one set per compute stream) ---
        self._A = []
        self._B = []
        self._C = []

        if self.mode != "comms_only":
            for i in range(self._num_compute_streams):
                stream_idx = i if self.mode == "compute_only" else i + 1
                target_device = self._get_device_for_stream(
                    stream_idx % stream_count
                )

                a = torch.randn(m, k, dtype=self.comp_data_type, device=target_device)
                b = torch.randn(k, n, dtype=self.comp_data_type, device=target_device)
                c = torch.empty(m, n, dtype=self.comp_data_type, device=target_device)

                self._A.append(a)
                self._B.append(b)
                self._C.append(c)

                self._tensors[f"A_{i}"] = a
                self._tensors[f"B_{i}"] = b
                self._tensors[f"C_{i}"] = c

        # --- Allocate communication tensors ---
        if self.mode != "compute_only":
            elem_size = torch.tensor([], dtype=self.comm_data_type).element_size()
            num_elements = max(1, self.comm_size_bytes // elem_size)
            # Two buffers (comm + scratch) are allocated
            alloc_bytes = num_elements * elem_size * 2
            if alloc_bytes > 4 * 1024 ** 3:
                logger.warning(
                    "Comm tensor allocation is %.1f GiB (comm_size_bytes=%d, "
                    "dtype=%s). This may cause out-of-memory errors.",
                    alloc_bytes / 1024 ** 3,
                    self.comm_size_bytes,
                    self.comm_data_type,
                )
            comm_device = self._get_device_for_stream(self._comm_stream_idx)
            self._comm_tensor = torch.randn(
                num_elements, dtype=self.comm_data_type, device=comm_device
            )
            self._comm_scratch = torch.empty_like(self._comm_tensor)
            self._tensors["comm"] = self._comm_tensor
            self._tensors["comm_scratch"] = self._comm_scratch

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one iteration.

        Depending on the mode the iteration dispatches:
        - **comms_only**: collectives on stream 0
        - **compute_only**: GEMMs across all streams
        - **comms_compute**: collectives on stream 0, GEMMs on streams 1..N
        """
        with record_function(f"comms_compute_overlap::{self.mode}"):
            if self.mode == "comms_only":
                self._run_comms(streams)
            elif self.mode == "compute_only":
                self._run_compute(streams, compute_stream_offset=0)
            else:
                self._run_comms(streams)
                self._run_compute(streams, compute_stream_offset=1)

            if self._pending_ops:
                self._complete_pending_ops()

    # -- Internal helpers ----------------------------------------------------

    def _run_comms(self, streams: List[torch.cuda.Stream]) -> None:
        """Dispatch collective operations on the comm stream."""
        comm_stream = streams[self._comm_stream_idx]
        with torch.cuda.stream(comm_stream):
            for coll_idx in range(self.num_coll_per_iter):
                with record_function(f"comms::all_reduce#{coll_idx}"):
                    if self._simulate_collectives:
                        self._comm_scratch.copy_(self._comm_tensor)
                        self._comm_tensor.add_(self._comm_scratch)
                    else:
                        work = dist.all_reduce(
                            self._comm_tensor,
                            op=dist.ReduceOp.SUM,
                            group=self._active_group,
                            async_op=self._async_op,
                        )
                        if self._async_op and work is not None:
                            self._pending_ops.append(work)

    def _run_compute(
        self, streams: List[torch.cuda.Stream], compute_stream_offset: int
    ) -> None:
        """Dispatch GEMM compute across compute streams."""
        num_available = len(streams) - compute_stream_offset
        if num_available <= 0:
            raise ValueError(
                f"No compute streams available: {len(streams)} total streams "
                f"with compute_stream_offset={compute_stream_offset}."
            )
        for i in range(self._num_compute_streams):
            stream = streams[compute_stream_offset + (i % num_available)]
            a = self._A[i]
            b = self._B[i]
            c = self._C[i]
            with torch.cuda.stream(stream):
                with record_function(f"compute::gemm#stream{compute_stream_offset + i}"):
                    for _ in range(self.num_compute_per_iter):
                        torch.mm(a, b, out=c)

    def _complete_pending_ops(self) -> None:
        """Wait for all outstanding async collective handles."""
        for work in self._pending_ops:
            work.wait()
        self._pending_ops.clear()

    # -- Metrics -------------------------------------------------------------

    def get_throughput_unit(self) -> str:
        if self.mode == "comms_only":
            return "GB/s"
        return "TFLOPS"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        """Compute throughput from iteration count and elapsed time.

        In ``comms_only`` mode the metric is raw data throughput (GB/s).
        In simulation mode this reflects local memory bandwidth of the
        copy+add mock; with real collectives it reflects the end-to-end
        collective data rate (which depends on the algorithm, topology,
        and world size -- not just point-to-point link bandwidth).
        """
        if total_time_sec <= 0:
            return 0.0

        if self.mode == "comms_only":
            if self._comm_tensor.nelement() == 0:
                return 0.0
            total_bytes = (
                iterations
                * self.num_coll_per_iter
                * self._comm_tensor.nelement()
                * self._comm_tensor.element_size()
            )
            return total_bytes / (total_time_sec * 1e9)  # GB/s

        m, n, k = self.mm_dim
        flops_per_gemm = 2 * m * n * k
        total_flops = (
            iterations
            * self.num_compute_per_iter
            * self._num_compute_streams
            * flops_per_gemm
        )
        return total_flops / (total_time_sec * 1e12)  # TFLOPS

    # -- Configuration -------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = {
            "name": self.name,
            "mode": self.mode,
            "kernel": self.kernel,
            "collective": self.collective,
            "mm_dim": self.mm_dim,
            "comp_data_type": str(self.comp_data_type),
            "comm_data_type": str(self.comm_data_type),
            "num_compute_per_iter": self.num_compute_per_iter,
            "num_coll_per_iter": self.num_coll_per_iter,
            "comm_size_bytes": self.comm_size_bytes,
            "num_compute_streams": self._num_compute_streams,
            "requested_compute_streams": self._requested_compute_streams,
            "simulate_collectives": self._simulate_collectives,
            "async_op": self._async_op,
            "backend": self._backend,
            "process_groups": self._pg_spec,
            "rank": self._rank,
            "world_size": self._world_size,
            "stream_count": self._stream_count,
        }
        config.update(self._get_multi_gpu_config())
        return config

    def validate_correctness(
        self, baseline_result: Any, test_result: Any
    ) -> Tuple[bool, str]:
        """Quick sanity check: verify GEMM outputs are finite."""
        if not self._is_setup:
            return True, "Not set up; skipping validation"

        for i, c in enumerate(self._C):
            if not torch.isfinite(c).all():
                return False, f"Compute stream {i} GEMM output contains non-finite values"

        return True, "Correctness validation passed"

    def cleanup(self) -> None:
        """Release resources and tear down distributed state."""
        super().cleanup()
        self._pending_ops.clear()
        self._process_groups.clear()
        self._active_group = None
        if not self._simulate_collectives:
            cleanup_distributed()
