# rocprof Tracing Configuration Guide

Detailed guide for configuring rocprofv3 kernel tracing to capture GEMM performance metrics.

## Configuration Methods

### YAML-Based Configuration (Recommended)

Use YAML configuration file for CU utilization metrics:

```yaml
jobs:
  - kernel_include_regex: "(gemm|Cijk_.*)"  # pattern for kernels to trace
    kernel_trace: true                      # enable kernel tracing
    stats: true                             # timing statistics only (not CU utilization)
    output_format: [json, csv]              # add perfetto for Chrome tracing
    sys_trace: false
    advanced_thread_trace: false            # leave false unless ATT decoder is installed
```

Example configuration: `scripts/gemm_analysis/rocprof_cu_only.yaml`

### Running with YAML Config

```bash
bash scripts/gemm_analysis/run_train_various_channels.sh \
  --rocprof \
  --rocprof-input scripts/gemm_analysis/rocprof_cu_only.yaml \
  --channels 28,42,56 \
  --threads 256,512 \
  --config config/gemm_overlap/gemm_test_1.yaml
```

## Configuration Notes

- Kernel filtering/stats come from the YAML
- Current rocprofv3 build ignores CLI kernel filters, use YAML to include/exclude kernels
- Remove `advanced_thread_trace` or keep it `false` unless ATT decoder debs are installed
- `stats: true` only collects timing statistics, NOT CU utilization metrics

## Output Files

rocprof generates 5 files per rank/process:

| File | Description |
|------|-------------|
| `PID_agent_info.csv` | Hardware information about CPUs and GPUs |
| `PID_counter_collection.csv` | **CU utilization metrics (main focus)** |
| `PID_kernel_trace.csv` | Kernel execution timeline data |
| `PID_results.json` | Chrome trace format for visualization |
| `PID_results.csv` | Summary statistics |

## counter_collection.csv Columns

### Kernel Configuration

- `Grid_Size` - Total number of workgroups in the kernel launch
- `Kernel_Name` - Name of the GEMM kernel (e.g., Cijk_Alik_Bljk_SB_MT128x128x32_MI32x32x1x2)
- `Workgroup_Size` - Number of work-items per workgroup

### Memory Configuration

- `LDS_Block_Size` - Local Data Share memory allocation per workgroup
- `Scratch_Size` - Private memory allocation per work-item

### Register Usage

- `VGPR_Count` - Vector General Purpose Registers used
- `Accum_VGPR_Count` - Accumulator VGPRs (for matrix operations)
- `SGPR_Count` - Scalar General Purpose Registers used

### Performance Metrics

- `Counter_Name` - Performance counter being measured
- `Counter_Value` - Value of the performance counter
- `Start_Timestamp` / `End_Timestamp` - Kernel execution timing

## Key Performance Counters

Focus on these counters in `counter_collection.csv`:

| Counter | Description | Purpose |
|---------|-------------|---------|
| `SQ_BUSY_CU_CYCLES` | Percentage of time CUs are active | CU utilization |
| `SQ_WAVES` | Number of active wavefronts | Occupancy indicator |
| `SQ_INSTS_MFMA` | Matrix FMA instructions | Critical for GEMM performance |
| `SQ_INSTS_VALU` | Vector ALU instructions | General compute |

## Command Line Options

- `--rocprof` - Enable rocprofv3 tracing
- `--rocprof-input FILE` - Use YAML configuration file
- `--stats` - Include timing statistics (not CU utilization)
- `--channels VALUES` - Comma-separated NCCL channel values
- `--threads VALUES` - Comma-separated thread values

## Output Location

Traces saved to `rocprof_traces/` in each run directory:

```
experiments/sweep_YYYYMMDD_HHMMSS/
└── 256thread/
    └── nccl_XXchannels/
        └── rocprof_traces/
            ├── PID_agent_info.csv
            ├── PID_counter_collection.csv
            ├── PID_kernel_trace.csv
            ├── PID_results.json
            └── PID_results.csv
```

## Analysis Workflow

1. Run sweep with rocprof enabled
2. Focus on `counter_collection.csv` files
3. Extract CU utilization metrics (`SQ_BUSY_CU_CYCLES`)
4. Correlate with GEMM kernel performance variance
5. Identify bottlenecks (low CU utilization, register pressure)

## Common Issues

### ATT Decoder Not Found

If you see warnings about ATT decoder, set `advanced_thread_trace: false` in YAML.

### Missing Counter Data

Ensure `stats: true` is set in YAML configuration.

### Large Output Files

Use `kernel_include_regex` to filter only GEMM kernels and reduce output size.
