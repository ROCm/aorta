"""Provenance manifest for every environment variable ``aorta env probe`` captures.

``CANONICAL_ENV_VARS`` -- the list the probe reads -- is GENERATED from
``ENV_KNOB_REGISTRY`` below, so the captured set, its classification and its provenance
cannot drift apart. Adding a knob means adding one registry entry; the tests, the docs
table and the coverage audit all read the same manifest.

What a registry entry asserts, and what it does not
--------------------------------------------------
An entry records that a variable is worth *preserving* in a snapshot and what is known
about who reads it. It makes no claim that the installed library supports the variable,
that the process exported it, or that it affected a run. ``_capture_env_vars`` is
``os.environ.get`` over these names: a ``null`` value means the variable was **unset in
this process**, never "unsupported by the library".

* ``library``          -- the component the variable belongs to. For a GEMM-prefix knob
  that is PRESENT in the reference build this is MEASURED: it is the shipped shared
  object whose string table holds the name, per ``scripts/audit_env_knobs.py``. A string
  table shows the name, not a call site, so this is ownership rather than proof of
  consumption -- ``consumer`` carries whatever was actually traced. Knobs marked
  ``ABSENT_FROM_REFERENCE_BUILD`` are declared, not measured: the audit has no binary to
  compare them against.
* ``consumer``         -- what the value reaches, at the granularity that was actually
  verified. For ``category="gemm_diagnostics"`` knobs the verified answer is "log or
  report emission only"; that classification is *recorded*, and is deliberately NOT used
  to exclude the knob from capture.
* ``category``         -- classification, from ``CATEGORIES``.
* ``source_reference`` -- where the entry can be re-verified. For GEMM knobs that is the
  reference build's library, re-checkable by running the audit script. Entries inherited
  from schema <= 1.14 are marked ``INHERITED_UNAUDITED`` rather than given a source they
  were never traced to.
* ``reference_build``  -- the build ``source_reference`` speaks about, since each ROCm
  release ships a different subset.

Auditing coverage
-----------------
``scripts/audit_env_knobs.py`` diffs this registry against the env-var strings in the
installed hipBLASLt / rocBLAS libraries and reports three sets: covered,
not-present-here (fine -- support varies by build; capture still depends only on whether
the process exported the variable), and *uncovered* -- a knob the library exposes that
this registry omits. That direction is the one a hand-written second list cannot check.
"""

from __future__ import annotations

from dataclasses import dataclass

# Provenance sentinels -- shared strings so a reader can grep for the class of evidence.
REFERENCE_BUILD_GEMM = "hipBLASLt 1.4.70002 + rocBLAS 5.0.70002 (ROCm 7.0.2)"
NOT_BUILD_SCOPED = "n/a -- not scoped to a specific library build"

REF_HIPBLASLT_SO = "libhipblaslt.so string table (scripts/audit_env_knobs.py)"
REF_ROCBLAS_SO = "librocblas.so string table (scripts/audit_env_knobs.py)"
REF_BOTH_SO = "libhipblaslt.so + librocblas.so string tables (scripts/audit_env_knobs.py)"
ABSENT_FROM_REFERENCE_BUILD = (
    "absent from the reference build; upstream-only or shipped by a newer ROCm "
    "(scripts/audit_env_knobs.py reports it as not_present)"
)
INHERITED_UNAUDITED = (
    "inherited from schema <= 1.14 (PR #306 / #308); classification not traced to an "
    "upstream call site"
)

#: Controlled vocabulary for ``EnvironmentKnob.category``.
CATEGORIES: tuple[str, ...] = (
    "gpu_scoping",
    "runtime",
    "codegen",
    "collectives",
    "fabric",
    "embedding_backend",
    "conv_backend",
    "attention_backend",
    "framework",
    "loader",
    "gemm_loading",
    "gemm_routing",
    "gemm_numeric",
    "gemm_workspace",
    "gemm_solution_selection",
    "gemm_launch_geometry",
    "gemm_skip_work",
    "gemm_numeric_check",
    "gemm_forward_compat",
    "gemm_diagnostics",
)


@dataclass(frozen=True)
class EnvironmentKnob:
    """One captured environment variable and its provenance.

    Recording a knob preserves declared process configuration. It does not assert that
    the installed library supports it, that it was set, or that it changed a run.
    """

    name: str
    library: str
    consumer: str
    category: str
    source_reference: str
    reference_build: str = NOT_BUILD_SCOPED


#: The manifest. ``CANONICAL_ENV_VARS`` is generated from it, in this order.
ENV_KNOB_REGISTRY: tuple[EnvironmentKnob, ...] = (
    # --- GPU scoping (most common cause of "you see N GPUs, I see M")
    EnvironmentKnob(
        name="HIP_VISIBLE_DEVICES",
        library="hip-runtime",
        consumer="device visibility -- which GPUs the process can see at all",
        category="gpu_scoping",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="ROCR_VISIBLE_DEVICES",
        library="hip-runtime",
        consumer="device visibility -- which GPUs the process can see at all",
        category="gpu_scoping",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- HSA / runtime
    EnvironmentKnob(
        name="HSA_XNACK",
        library="hsa-runtime",
        consumer="HSA runtime memory / paging behaviour",
        category="runtime",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="HSA_KERNARG_POOL_SIZE",
        library="hsa-runtime",
        consumer="HSA runtime memory / paging behaviour",
        category="runtime",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="HSA_NO_SCRATCH_RECLAIM",
        library="hsa-runtime",
        consumer="HSA runtime memory / paging behaviour",
        category="runtime",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="HSA_OVERRIDE_GFX_VERSION",
        library="hsa-runtime",
        consumer="forces a different gfx target than the silicon reports",
        category="runtime",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="HSA_TOOLS_DISABLE_REGISTER",
        library="hsa-runtime",
        consumer="disables HSA tool (profiler / sanitizer) registration",
        category="runtime",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- GPU queue / codegen / build target
    EnvironmentKnob(
        name="GPU_MAX_HW_QUEUES",
        library="hip-runtime",
        consumer="hardware queue count per process",
        category="runtime",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="AMDGCN_USE_BUFFER_OPS",
        library="compiler",
        consumer="code generation for buffer ops",
        category="codegen",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="DISABLE_TF32",
        library="pytorch",
        consumer="disables the TF32/xf32 compute path",
        category="gemm_numeric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="PYTORCH_ROCM_ARCH",
        library="pytorch",
        consumer="gfx targets PyTorch builds / dispatches for",
        category="codegen",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="HIP_LAUNCH_BLOCKING",
        library="hip-runtime",
        consumer="forces synchronous kernel launches; serialises stream overlap",
        category="runtime",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- RCCL / NCCL
    EnvironmentKnob(
        name="NCCL_MAX_NCHANNELS",
        library="rccl",
        consumer="collective transport / channel selection",
        category="collectives",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_P2P_LEVEL",
        library="rccl",
        consumer="collective transport / channel selection",
        category="collectives",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_HCA",
        library="rccl",
        consumer="collective transport / channel selection",
        category="collectives",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_SOCKET_IFNAME",
        library="rccl",
        consumer="collective transport / channel selection",
        category="collectives",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="RCCL_MSCCL_ENABLE",
        library="rccl",
        consumer="collective transport / channel selection",
        category="collectives",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- AINIC (AMD-Pensando RoCE NIC) net-plugin + fabric tuning. Captured so an our-env vs customer-env diff surfaces RoCE/QoS mismatches (GID index, traffic class, DCQCN-adjacent flags) and which net plugin RCCL loads. Absent on non-AINIC nodes -> None.
    EnvironmentKnob(
        name="RCCL_AINIC_ROCE",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_NET_PLUGIN",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_NET",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="RCCL_CTS_OFFLOAD_ENABLED",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_GID_INDEX",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_ROCE_VERSION_NUM",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_TC",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_FIFO_TC",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_GDR_FLUSH_DISABLE",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_GDRCOPY_ENABLE",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_USE_INLINE",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_PCI_RELAXED_ORDERING",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_QPS_PER_CONNECTION",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_PXN_DISABLE",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IGNORE_CPU_AFFINITY",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_NET_OPTIONAL_RECV_COMPLETION",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_TIMEOUT",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_SL",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_IB_SPLIT_DATA_ON_QPS",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_DMABUF_ENABLE",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_CUMEM_ENABLE",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="IONIC_LOCKFREE",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="RCCL_DISABLE_RAIL_TREES",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="RCCL_LL128_FORCE_ENABLE",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="NCCL_WORK_FIFO_BYTES",
        library="rccl",
        consumer="RoCE / fabric / net-plugin tuning as declared to RCCL",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- gfx950 (MI350/MI355X) fence-ordering debug knob from the silent-data-corruption investigation. Captured so a per-rank env diff catches a launcher that exports the override on rank 0 but not the rest -- a half-applied "fix" the diff would otherwise miss.
    EnvironmentKnob(
        name="RCCL_GFX9_CHEAP_FENCE_OFF",
        library="rccl",
        consumer="gfx950 fence-ordering override from the SDC investigation",
        category="fabric",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- FBGEMM
    EnvironmentKnob(
        name="FBGEMM_NO_JK",
        library="fbgemm",
        consumer="TBE kernel variant / bounds-check implementation",
        category="embedding_backend",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="FBGEMM_TBE_V2",
        library="fbgemm",
        consumer="TBE kernel variant / bounds-check implementation",
        category="embedding_backend",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL",
        library="fbgemm",
        consumer="TBE kernel variant / bounds-check implementation",
        category="embedding_backend",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="FBGEMM_BOUNDS_CHECK_INDICES_V2",
        library="fbgemm",
        consumer="TBE kernel variant / bounds-check implementation",
        category="embedding_backend",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- MIOpen kernel DB + selection-mode
    EnvironmentKnob(
        name="MIOPEN_SYSTEM_DB_PATH",
        library="miopen",
        consumer=("convolution kernel database location and solver selection mode"),
        category="conv_backend",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="MIOPEN_USER_DB_PATH",
        library="miopen",
        consumer=("convolution kernel database location and solver selection mode"),
        category="conv_backend",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="MIOPEN_DEBUG_DISABLE_FIND_DB",
        library="miopen",
        consumer=("convolution kernel database location and solver selection mode"),
        category="conv_backend",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="MIOPEN_FIND_MODE",
        library="miopen",
        consumer=("convolution kernel database location and solver selection mode"),
        category="conv_backend",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- SDPA / Flash Attention backend selection (CK vs AOTriton). Note: USE_ROCM_CK_SDPA / USE_ROCM_CK_GEMM are NOT here -- they're build-time cmake flags consumed when the PyTorch wheel is built, not runtime env vars. Captured under composable_kernel.{pytorch_use_ck_sdpa,pytorch_use_ck_gemm}.
    EnvironmentKnob(
        name="TORCH_ROCM_FA_PREFER_CK",
        library="pytorch",
        consumer="SDPA / flash-attention backend choice (CK vs AOTriton)",
        category="attention_backend",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
        library="pytorch",
        consumer="SDPA / flash-attention backend choice (CK vs AOTriton)",
        category="attention_backend",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- GEMM backend preference + hipBLASLt autotune pinning
    EnvironmentKnob(
        name="TORCH_BLAS_PREFER_HIPBLASLT",
        library="pytorch",
        consumer="routes torch GEMMs to hipBLASLt instead of rocBLAS",
        category="gemm_routing",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="TORCH_HIPBLASLT_TUNING_FILE",
        library="pytorch",
        consumer="torch-side hipBLASLt autotune pinning",
        category="gemm_solution_selection",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="TORCH_HIPBLASLT_TUNING_OVERRIDE_FILE",
        library="pytorch",
        consumer="torch-side hipBLASLt autotune pinning",
        category="gemm_solution_selection",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- PyTorch / inductor
    EnvironmentKnob(
        name="TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE",
        library="pytorch",
        consumer="inductor autotuning / caching allocator configuration",
        category="framework",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    EnvironmentKnob(
        name="PYTORCH_CUDA_ALLOC_CONF",
        library="pytorch",
        consumer="inductor autotuning / caching allocator configuration",
        category="framework",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- Dynamic loader -- which shared object actually loads. LD_LIBRARY_PATH can silently point hipBLASLt/rocBLAS/RCCL at a different .so than the installed one ("the version running wasn't the version we thought" -- the failure this tool exists to catch). Verbose and not ROCm-specific, but load-bearing for a library-identity diff.
    EnvironmentKnob(
        name="LD_LIBRARY_PATH",
        library="dynamic-loader",
        consumer=(
            "which shared object actually loads -- can silently swap hipBLASLt / rocBLAS / RCCL"
        ),
        category="loader",
        source_reference=INHERITED_UNAUDITED,
        reference_build=NOT_BUILD_SCOPED,
    ),
    # --- hipBLASLt / rocBLAS / Tensile GEMM configuration.
    #
    # Reference build: hipBLASLt 1.4.70002 + rocBLAS 5.0.70002 on ROCm 7.0.2.
    # Capture is comprehensive: diagnostics and report-only controls are retained and
    # classified, never filtered out. A value is the raw exported process setting; None
    # means unset, not unsupported.
    #
    # Library / kernel loading -- which file backs the kernels.
    EnvironmentKnob(
        name="HIPBLASLT_TENSILE_LIBPATH",
        library="hipblaslt",
        consumer=("overrides the Tensile kernel-DB dir, bypassing tensile_catalog"),
        category="gemm_loading",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_EXT_OP_LIBRARY_PATH",
        library="hipblaslt",
        consumer="overrides the ExtOp library path",
        category="gemm_loading",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_PRELOAD_KERNELS",
        library="hipblaslt",
        consumer=(
            "reaches Debug::Instance().preload() in tensile_host.cpp -- changes the load path and the isPreloaded flag"
        ),
        category="gemm_loading",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_TENSILE_LIBPATH",
        library="rocblas",
        consumer="overrides rocBLAS's Tensile kernel-DB dir",
        category="gemm_loading",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH",
        library="rocblas",
        consumer="per-GEMM Tensile solution override file",
        category="gemm_solution_selection",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    # --- Backend routing + generator choice.
    EnvironmentKnob(
        name="ROCBLAS_USE_HIPBLASLT",
        library="rocblas",
        consumer="routes rocBLAS GEMMs through hipBLASLt instead of Tensile",
        category="gemm_routing",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_USE_HIPBLASLT_BATCHED",
        library="rocblas",
        consumer="same routing decision for batched GEMMs",
        category="gemm_routing",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_USE_ROCROLLER",
        library="hipblaslt",
        consumer="switches to the rocRoller kernel generator",
        category="gemm_routing",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL",
        library="hipblaslt",
        consumer="rocRoller sub-knob: disables custom kernels",
        category="gemm_routing",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_TUNING_OVERRIDE_FILE",
        library="hipblaslt",
        consumer=("native autotune pin, distinct from the TORCH_ prefixed variant"),
        category="gemm_solution_selection",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    # --- Numeric path.
    EnvironmentKnob(
        name="HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32",
        library="hipblaslt",
        consumer="forces the xf32 / TF32 emulation compute path",
        category="gemm_numeric",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_DEFAULT_ATOMICS_MODE",
        library="rocblas",
        consumer="atomic reductions on/off -> numeric determinism",
        category="gemm_numeric",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_INTERNAL_FP16_ALT_IMPL",
        library="rocblas",
        consumer="alternate fp16 implementation",
        category="gemm_numeric",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_INTERNAL_FP16_ALT_IMPL_RNZ",
        library="rocblas",
        consumer="alternate fp16 implementation, round-nearest-zero variant",
        category="gemm_numeric",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_INTERNAL_FORCE_VALU_FOR_DGEMM",
        library="rocblas",
        consumer="forces the VALU arithmetic unit for dgemm",
        category="gemm_numeric",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    # --- Allocator + workspace sizing -- directly relevant to a recycled-buffer race.
    EnvironmentKnob(
        name="ROCBLAS_STREAM_ORDER_ALLOC",
        library="rocblas",
        consumer=("stream-ordered allocation; interacts with cross-stream buffer reuse"),
        category="gemm_workspace",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_DEVICE_MEMORY_SIZE",
        library="rocblas",
        consumer="sets the device workspace size and flips it to user-managed",
        category="gemm_workspace",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_INTERNAL_TRSM_REG_KERNEL_MEM_LIMIT",
        library="rocblas",
        consumer="trsm regular-vs-special kernel switch point",
        category="gemm_workspace",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    # --- Solution selection -- all of these change which kernel is chosen.
    EnvironmentKnob(
        name="TENSILE_SOLUTION_INDEX",
        library="hipblaslt+rocblas",
        consumer="pins a specific Tensile solution, forcing one kernel",
        category="gemm_solution_selection",
        source_reference=REF_BOTH_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_SOLUTION_SELECTION_METHOD",
        library="hipblaslt",
        consumer="changes how Tensile selects a solution",
        category="gemm_solution_selection",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_EXPERIMENTAL_SELECTION",
        library="rocblas",
        consumer="rocBLAS-side Tensile experimental selection path",
        category="gemm_solution_selection",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_TAM_SELECTION_ENABLE",
        library="rocblas",
        consumer="rocBLAS-side Tensile debug-selection path",
        category="gemm_solution_selection",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_NAIVE_SEARCH",
        library="hipblaslt+rocblas",
        consumer="linear property search instead of the matching tree",
        category="gemm_solution_selection",
        source_reference=REF_BOTH_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_METRIC",
        library="hipblaslt+rocblas",
        consumer="the performance metric the matching library optimises for",
        category="gemm_solution_selection",
        source_reference=REF_BOTH_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_PREDICTION_LIB",
        library="hipblaslt",
        consumer="routes selection through the prediction library",
        category="gemm_solution_selection",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_GRIDBASED_KDTREE",
        library="hipblaslt",
        consumer="grid-based selection: k-d tree lookup",
        category="gemm_solution_selection",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_GRIDBASED_BATCH_EXP",
        library="hipblaslt",
        consumer="grid-based selection: batch-experiment path",
        category="gemm_solution_selection",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ANALYTICAL_GEMM_HEURISTICS",
        library="hipblaslt",
        consumer="selects Origami analytical GEMM heuristic behavior",
        category="gemm_solution_selection",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ANALYTICAL_GEMM_HEURISTICS_VARIANCE",
        library="hipblaslt",
        consumer="configures variance used by Origami analytical GEMM heuristics",
        category="gemm_solution_selection",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="GRIDBASED_TOPSOLS",
        library="hipblaslt",
        consumer="sets the number of top GridBased candidate solutions considered",
        category="gemm_solution_selection",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    # --- Stream-K: how the K dimension is split across workgroups, i.e. the launch geometry and the CU occupancy of every GEMM.
    EnvironmentKnob(
        name="TENSILE_STREAMK_DYNAMIC_GRID",
        library="hipblaslt+rocblas",
        consumer=(
            "Stream-K grid sizing / CU cap -- launch geometry and CU occupancy of every GEMM"
        ),
        category="gemm_launch_geometry",
        source_reference=REF_BOTH_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_STREAMK_FIXED_GRID",
        library="hipblaslt+rocblas",
        consumer=(
            "Stream-K grid sizing / CU cap -- launch geometry and CU occupancy of every GEMM"
        ),
        category="gemm_launch_geometry",
        source_reference=REF_BOTH_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_STREAMK_MAX_CUS",
        library="hipblaslt+rocblas",
        consumer=(
            "Stream-K grid sizing / CU cap -- launch geometry and CU occupancy of every GEMM"
        ),
        category="gemm_launch_geometry",
        source_reference=REF_BOTH_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_STREAMK_DATA_PARALLEL",
        library="hipblaslt",
        consumer="Stream-K: forces the data-parallel variant",
        category="gemm_launch_geometry",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_STREAMK_DYNAMIC_WGM",
        library="hipblaslt",
        consumer="Stream-K: dynamic workgroup mapping",
        category="gemm_launch_geometry",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_STREAMK_FULL_TILES",
        library="hipblaslt+rocblas",
        consumer=(
            "Stream-K grid sizing / CU cap -- launch geometry and CU occupancy of every GEMM"
        ),
        category="gemm_launch_geometry",
        source_reference=REF_BOTH_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_STREAMK_GRID_MULTIPLIER",
        library="hipblaslt+rocblas",
        consumer=(
            "Stream-K grid sizing / CU cap -- launch geometry and CU occupancy of every GEMM"
        ),
        category="gemm_launch_geometry",
        source_reference=REF_BOTH_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    # --- Workgroup mapping + StaggerU -- these change tile-to-CU assignment and the per-workgroup memory-access offsets, so they move the timing of a race.
    EnvironmentKnob(
        name="TENSILE_FIXED_WGM",
        library="hipblaslt",
        consumer="pins workgroup / XCC tile-to-CU mapping",
        category="gemm_launch_geometry",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_FIXED_WGMXCC",
        library="hipblaslt",
        consumer="pins workgroup / XCC tile-to-CU mapping",
        category="gemm_launch_geometry",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_FIXED_WGMXCCCHUNK",
        library="hipblaslt",
        consumer="pins workgroup / XCC tile-to-CU mapping",
        category="gemm_launch_geometry",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_DISABLE_STAGGERU",
        library="hipblaslt",
        consumer="StaggerU per-workgroup memory-access offsets",
        category="gemm_launch_geometry",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_FIXED_STAGGERU",
        library="hipblaslt",
        consumer="StaggerU per-workgroup memory-access offsets",
        category="gemm_launch_geometry",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_FIXED_STAGGERU_MAPPING",
        library="hipblaslt",
        consumer="StaggerU per-workgroup memory-access offsets",
        category="gemm_launch_geometry",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_FIXED_STAGGERU_STRIDE_SHIFT",
        library="hipblaslt",
        consumer="StaggerU per-workgroup memory-access offsets",
        category="gemm_launch_geometry",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    # --- Debug bits that SKIP work (unlike TENSILE_DB, which only prints): bit 0 = skipKernelLaunch, bit 1 = skipInitKernelLaunch.
    EnvironmentKnob(
        name="TENSILE_DB2",
        library="hipblaslt+rocblas",
        consumer=(
            "bit 0 gates skipKernelLaunch, bit 1 skipInitKernelLaunch (HipSolutionAdapter.cpp / ContractionSolution.cpp) -- skips work, unlike TENSILE_DB"
        ),
        category="gemm_skip_work",
        source_reference=REF_BOTH_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    # --- Not in the reference build's .so (newer hipBLASLt Stream-K knobs).
    # Kept for forward-compatible diffing. Exported values are captured verbatim even
    # when the local library ignores them; None means only that the process left them unset.
    EnvironmentKnob(
        name="TENSILE_STREAMK5_FORCE_MODE",
        library="hipblaslt",
        consumer=(
            "newer Stream-K knobs; recorded so a customer on a newer library still gets them diffed"
        ),
        category="gemm_forward_compat",
        source_reference=ABSENT_FROM_REFERENCE_BUILD,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_STREAMK_TILES",
        library="hipblaslt",
        consumer=(
            "newer Stream-K knobs; recorded so a customer on a newer library still gets them diffed"
        ),
        category="gemm_forward_compat",
        source_reference=ABSENT_FROM_REFERENCE_BUILD,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_STREAMK_SPLIT",
        library="hipblaslt",
        consumer=(
            "newer Stream-K knobs; recorded so a customer on a newer library still gets them diffed"
        ),
        category="gemm_forward_compat",
        source_reference=ABSENT_FROM_REFERENCE_BUILD,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    # --- In-library numeric checking: extra scan kernels + sync -> can mask/surface the timing race independent of the final values (see block note above). The rocBLAS variant matters concretely here: on the stock 1.4.0 image the repro's GEMMs fall back to rocBLAS, so capturing only the hipBLASLt half would capture the half that is not running.
    EnvironmentKnob(
        name="HIPBLASLT_CHECK_NUMERICS",
        library="hipblaslt",
        consumer=(
            "in-library numeric checking: launches extra scan kernels and adds synchronization, which can mask or surface a timing race"
        ),
        category="gemm_numeric_check",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY",
        library="hipblaslt",
        consumer=(
            "in-library numeric checking: launches extra scan kernels and adds synchronization, which can mask or surface a timing race"
        ),
        category="gemm_numeric_check",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_CHECK_NUMERICS_SCAN_FROM",
        library="hipblaslt",
        consumer=(
            "in-library numeric checking: launches extra scan kernels and adds synchronization, which can mask or surface a timing race"
        ),
        category="gemm_numeric_check",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL",
        library="hipblaslt",
        consumer=(
            "in-library numeric checking: launches extra scan kernels and adds synchronization, which can mask or surface a timing race"
        ),
        category="gemm_numeric_check",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST",
        library="hipblaslt",
        consumer=(
            "in-library numeric checking: launches extra scan kernels and adds synchronization, which can mask or surface a timing race"
        ),
        category="gemm_numeric_check",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_CHECK_NUMERICS",
        library="rocblas",
        consumer=(
            "rocBLAS numeric checking -- the load-bearing half on builds where the GEMMs fall back to rocBLAS"
        ),
        category="gemm_numeric_check",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    # --- Diagnostics / reporting knobs.
    #
    # Captured, with their consumer recorded as report-emission-only. Schema <= 1.15 drafts
    # EXCLUDED these on a behaviour-based rule ("only capture what changes execution").
    # That rule made the probe's output depend on our classification being right -- and the
    # 2026-08-02 audit overturned two of its own name-based verdicts, which is the evidence
    # that the rule was the wrong place to apply the judgement. Capturing them keeps the
    # snapshot a faithful record of declared configuration; the classification survives in
    # ``category``/``consumer`` for anyone triaging a diff.
    EnvironmentKnob(
        name="ANALYTICAL_GEMM_DEBUG",
        library="hipblaslt",
        consumer="enables Origami analytical-model debug output",
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ORIGAMI_LOG_FILE",
        library="hipblaslt",
        consumer="selects the Origami debug log path and output format",
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_LOG_FILE",
        library="hipblaslt",
        consumer="hipBLASLt logging sink / level / mask -- log emission only",
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_LOG_LEVEL",
        library="hipblaslt",
        consumer="hipBLASLt logging sink / level / mask -- log emission only",
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_LOG_MASK",
        library="hipblaslt",
        consumer="hipBLASLt logging sink / level / mask -- log emission only",
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_BENCH_PERF",
        library="hipblaslt",
        consumer=("fills hipblasltClientPerformanceArgs for client-side perf reporting"),
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_BENCH_PERF_ALL",
        library="hipblaslt",
        consumer=(
            "sibling of HIPBLASLT_BENCH_PERF; absent from the reference build, present in ROCm 7.2.3"
        ),
        category="gemm_diagnostics",
        source_reference=ABSENT_FROM_REFERENCE_BUILD,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_BENCH_PRINT_COMMAND",
        library="hipblaslt",
        consumer="prints the reproducing bench command",
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="HIPBLASLT_ENABLE_MARKER",
        library="hipblaslt",
        consumer="emits profiler range markers",
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_ENABLE_MARKER",
        library="hipblaslt",
        consumer="emits profiler range markers",
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_LAYER",
        library="rocblas",
        consumer="rocBLAS log layer mask (trace / bench / profile)",
        category="gemm_diagnostics",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_LOG_PATH",
        library="rocblas",
        consumer="rocBLAS per-layer log sinks",
        category="gemm_diagnostics",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_LOG_TRACE_PATH",
        library="rocblas",
        consumer="rocBLAS per-layer log sinks",
        category="gemm_diagnostics",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_LOG_BENCH_PATH",
        library="rocblas",
        consumer="rocBLAS per-layer log sinks",
        category="gemm_diagnostics",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_LOG_PROFILE_PATH",
        library="rocblas",
        consumer="rocBLAS per-layer log sinks",
        category="gemm_diagnostics",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_VERBOSE_HIPBLASLT_ERROR",
        library="rocblas",
        consumer="verbose backend error reporting",
        category="gemm_diagnostics",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="ROCBLAS_VERBOSE_TENSILE_ERROR",
        library="rocblas",
        consumer="verbose backend error reporting",
        category="gemm_diagnostics",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_DB",
        library="hipblaslt+rocblas",
        consumer=(
            "TensileLite debug print mask -- every bit it sets is a print*, unlike TENSILE_DB2"
        ),
        category="gemm_diagnostics",
        source_reference=REF_BOTH_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_ADAPTIVE_GEMM_LOG",
        library="hipblaslt",
        consumer="adaptive-GEMM logging",
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_AUTO_GSU_ALGO",
        library="hipblaslt",
        consumer=(
            "guards a lone std::cout in calculateAutoGSU -- reporting only, not a selection knob"
        ),
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_SOLUTION_SELECTION_TRACE",
        library="rocblas",
        consumer="traces solution selection",
        category="gemm_diagnostics",
        source_reference=REF_ROCBLAS_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
    EnvironmentKnob(
        name="TENSILE_BENCHMARK",
        library="hipblaslt",
        consumer=("Debug::getBenchmark() has no call site in the reference libraries"),
        category="gemm_diagnostics",
        source_reference=REF_HIPBLASLT_SO,
        reference_build=REFERENCE_BUILD_GEMM,
    ),
)

#: Names the probe reads, generated from the manifest. Order is the manifest's order,
#: which is the order ``env_vars`` keys appear in a snapshot.
CANONICAL_ENV_VARS: tuple[str, ...] = tuple(knob.name for knob in ENV_KNOB_REGISTRY)

#: Registry lookup by name.
ENV_KNOBS_BY_NAME: dict[str, EnvironmentKnob] = {knob.name: knob for knob in ENV_KNOB_REGISTRY}
