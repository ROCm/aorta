# Magpie Integration

Aorta optionally integrates with [Magpie](https://github.com/AMD-AGI/Magpie), AMD-AGI's GPU kernel evaluation toolkit, to gain two capabilities that aorta does not provide natively:

- **GPU hardware control** -- Lock GPU clocks and set power limits for deterministic, reproducible benchmarks.
- **Magpie report adapter** -- Import Magpie benchmark workspaces into aorta's report pipeline for unified analysis and comparison.

Magpie is a **read-only, optional dependency**. All integration code lives in the aorta codebase; no changes to Magpie are required.

## Installation

Magpie is pulled directly from GitHub (the same pattern used for TraceLens):

```bash
# Install aorta with Magpie support
pip install -e ".[magpie]"

# Or install everything
pip install -e ".[all]"
```

This resolves `magpie-eval` via `git+https://github.com/AMD-AGI/Magpie.git` as declared in `pyproject.toml`.

**Using a local Magpie checkout instead:** If you are developing Magpie locally, install it in editable mode first and then install aorta without the magpie extra so pip does not overwrite your local version:

```bash
# Install your local Magpie checkout
pip install -e /path/to/Magpie

# Then install aorta (skip the magpie extra since it is already satisfied)
pip install -e ".[hw-queue,report]"
```

**Docker:** The `rocm70_9-1` Docker images (`docker/Dockerfile.rocm70_9-1`, `docker/Dockerfile.rocm70_9-1-shampoo`, `docker/rccl_test/Dockerfile.rocm70_9-1`) include Magpie pre-installed alongside TraceLens. No additional install step is needed inside these containers.

When Magpie is not installed, aorta continues to work normally. GPU control flags are silently ignored (with a warning), and the report adapter operates on Magpie's file-based output without importing any Magpie modules.

## GPU Hardware Control

### Why it matters

GPU boosting and thermal throttling introduce variance between benchmark runs. Locking clocks to a fixed level and capping power ensures that `hw_queue_eval` results are reproducible and comparable across runs.

### CLI usage

Both the `run` and `sweep` commands accept GPU control flags:

```bash
# Lock AMD GPU clocks to level 3 (mid-range)
python -m aorta.hw_queue_eval run hetero_kernels --streams 8 --lock-clocks 3

# Lock clocks and cap power at 200W
python -m aorta.hw_queue_eval run hetero_kernels --streams 8 \
    --lock-clocks 3 --power-limit 200

# Sweep with locked clocks for consistent scaling curves
python -m aorta.hw_queue_eval sweep hetero_kernels --streams 1,2,4,8,16 \
    --lock-clocks 5 --power-limit 300
```

| Flag | Type | Description |
| --- | --- | --- |
| `--lock-clocks` | int | AMD GPU clock level (0-7, where 7 is highest) |
| `--power-limit` | int | GPU power cap in watts |

When GPU control is active, the harness:

1. Applies the clock/power settings before the warmup phase.
2. Captures a hardware snapshot (clocks, power, temperature) into the result metadata.
3. Resets the GPU to default settings after measurement completes.

### Programmatic usage

```python
from aorta.utils import GPUControlConfig, GPUControlManager

config = GPUControlConfig(
    enabled=True,
    gpu_clock_level=3,
    power_limit_watts=200,
    reset_on_exit=True,
)

# As a context manager
with GPUControlManager(config) as mgr:
    run_benchmark()
# GPU settings restored automatically

# Or manually
mgr = GPUControlManager(config)
snapshot = mgr.apply()   # returns hardware state dict
run_benchmark()
mgr.reset()
```

### Result metadata

When GPU control is active, the `HarnessResult` metadata includes a `gpu_hardware_state` key with per-device information (clocks, power draw, temperature, etc.) captured at benchmark start.

## Magpie Report Adapter

### Magpie workspace layout

Magpie benchmark runs produce workspaces with this structure:

```
benchmark_{framework}_{timestamp}/
    config.yaml
    benchmark_report.json
    inferencemax_result.json
    torch_trace/
    tracelens_rank0_csvs/
    tracelens_collective_csvs/
```

The adapter reads these workspaces and converts them into a format aorta's report pipeline can consume.

### CLI commands

All commands live under `aorta-report magpie`:

```bash
# List Magpie workspaces in a directory
aorta-report magpie list ./results

# Show normalized report for a workspace
aorta-report magpie show results/benchmark_vllm_20260301_120000

# Import a workspace into aorta-report format
aorta-report magpie import results/benchmark_vllm_20260301_120000 \
    -o aorta_output/run1

# Import and run TraceLens analysis on torch traces
aorta-report magpie import results/benchmark_vllm_20260301_120000 \
    -o aorta_output/run1 --run-tracelens

# Compare two Magpie benchmark runs
aorta-report magpie compare \
    -b results/benchmark_vllm_20260301_120000 \
    -t results/benchmark_vllm_20260301_140000

# Save comparison to JSON
aorta-report magpie compare \
    -b results/benchmark_vllm_20260301_120000 \
    -t results/benchmark_vllm_20260301_140000 \
    -o comparison.json
```

### Comparison output

The `compare` command reports throughput and latency deltas:

- **Throughput** (higher is better): `request_throughput`, `output_throughput`, `total_token_throughput`
- **Latency** (lower is better): `ttft`, `tpot`, `itl`, `e2el` (mean and p99)

Each metric is classified as `better` (>1% improvement), `worse` (>1% regression), or `similar`.

### Programmatic usage

```python
from aorta.report.magpie_adapter import (
    locate_magpie_workspaces,
    read_magpie_report,
    import_magpie_workspace,
    compare_magpie_reports,
)

# Find workspaces
workspaces = locate_magpie_workspaces("./results")

# Read a single report
report = read_magpie_report(workspaces[0])
print(report["throughput"]["request_throughput"])

# Import into aorta format
result = import_magpie_workspace(
    workspace=workspaces[0],
    output_dir="./aorta_output/run1",
    run_tracelens=True,
)

# Compare two runs
comparison = compare_magpie_reports(
    baseline_workspace=workspaces[1],
    test_workspace=workspaces[0],
)
print(comparison["summary"]["overall"])  # "improvement", "regression", or "neutral"
```

## Architecture

```
Magpie (untouched, optional)
  |
  v  (try/except ImportError)
aorta
  ├── src/aorta/utils/gpu_control.py
  │     Imports Magpie.utils.gpu.GPUController / MultiGPUController
  │     Exposes GPUControlConfig + GPUControlManager
  │
  ├── src/aorta/hw_queue_eval/core/harness.py
  │     Uses GPUControlManager in run() and run_workload()
  │
  ├── src/aorta/hw_queue_eval/cli.py
  │     --lock-clocks / --power-limit flags
  │
  ├── src/aorta/report/magpie_adapter.py
  │     Reads Magpie workspace files (no Magpie imports)
  │
  └── src/aorta/report/cli.py
        aorta-report magpie list|show|import|compare
```

## API Reference

### `aorta.utils.gpu_control`

| Symbol | Type | Description |
| --- | --- | --- |
| `HAS_MAGPIE` | `bool` | `True` if Magpie is importable |
| `GPUControlConfig` | dataclass | Power limit, clock levels, device IDs, reset-on-exit |
| `GPUControlManager` | class | Context manager wrapping Magpie's `MultiGPUController` |
| `GPUControlManager.apply()` | method | Apply config, return hardware snapshot dict |
| `GPUControlManager.reset()` | method | Reset GPUs to defaults |
| `GPUControlManager.available` | property | `True` if control is enabled and Magpie is installed |

### `aorta.report.magpie_adapter`

| Function | Description |
| --- | --- |
| `locate_magpie_workspaces(results_dir)` | Find workspace dirs containing `benchmark_report.json` |
| `read_magpie_report(workspace)` | Read and normalize a Magpie benchmark report |
| `import_magpie_workspace(workspace, output_dir, ...)` | Copy workspace into aorta-compatible layout |
| `compare_magpie_reports(baseline, test)` | Compute throughput/latency deltas between two runs |
