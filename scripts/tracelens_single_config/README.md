# TraceLens Single Configuration

Analyze PyTorch profiler traces from one training run.

> Need to compare multiple configs? See [../gemm_analysis/README.md](../gemm_analysis/README.md)

## Usage

```bash
bash scripts/tracelens_single_config/run_tracelens_single_config.sh /path/to/traces
```

Accepts either:
- Parent directory containing torch_profiler/
- torch_profiler/ directory directly

## Expected Structure

```
traces/
└── torch_profiler/
    ├── rank0/
    │   └── trace.json
    ├── rank1/
    │   └── trace.json
    └── ...
```

## Output

```
<base_directory>/tracelens_analysis/
├── individual_reports/
│   ├── perf_rank0.xlsx
│   ├── perf_rank1.xlsx
│   └── ...
└── collective_reports/
    └── collective_all_ranks.xlsx
```

## Examples

```bash
# Parent directory
bash scripts/tracelens_single_config/run_tracelens_single_config.sh \
  /home/oyazdanb/aorta_sonbol/saleel_data/saleelk_hip_runtime_test_traces

# torch_profiler directory directly
bash scripts/tracelens_single_config/run_tracelens_single_config.sh \
  /home/oyazdanb/aorta_sonbol/saleel_data/saleelk_hip_runtime_test_traces/torch_profiler
```

## Script Location

```
scripts/tracelens_single_config/run_tracelens_single_config.sh
```