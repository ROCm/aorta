# TraceLens Single Configuration

Analyze PyTorch profiler traces from one training run.

For multiple configs see [../gemm_analysis/README.md](../gemm_analysis/README.md)

## Quick Start

```bash
# Complete analysis
python scripts/tracelens_single_config/run_full_analysis.py \
  --baseline /path/to/baseline/traces \
  --test /path/to/test/traces \
  --output /path/to/output \
  --all

# Skip TraceLens if already done
python scripts/tracelens_single_config/run_full_analysis.py \
  --baseline /path/to/baseline \
  --test /path/to/test \
  --output /path/to/output \
  --all --skip-tracelens
```

### Flags:
- `--all` - Run everything including final report
- `--gpu-timeline` - GPU timeline comparison
- `--collective` - NCCL collective comparison
- `--final-report` - Create comprehensive Excel report
- `--generate-plots` - Generate visualization plots and HTML report from final report
- `--skip-tracelens` - Skip TraceLens report generation if already done

### Output:
- `final_analysis_report.xlsx` - All comparisons with tables and color scale
  - Color scale on percent_change: Red (worst) -> White (neutral) -> Green (best)

### Using --skip-tracelens

Use the same paths for `--baseline` and `--test`. The script looks for `tracelens_analysis` subdirectory:

```bash
# Expected structure when using --skip-tracelens
baseline/
└── tracelens_analysis/    # From previous run
    ├── individual_reports/
    └── collective_reports/

test/
└── tracelens_analysis/    # From previous run
    ├── individual_reports/
    └── collective_reports/
```

Example:
```bash
# Use same paths, script finds tracelens_analysis inside
python run_full_analysis.py \
  --baseline ~/data/baseline_traces \
  --test ~/data/test_traces \
  --output ~/results \
  --all --skip-tracelens
```


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

## What the Master Script Does

The `run_full_analysis.py` script automatically handles all steps:

1. Runs TraceLens on baseline and test traces
2. Processes GPU timelines using `process_gpu_timeline.py`
3. Combines reports using `combine_reports.py`
4. Adds comparison sheets using `add_comparison_sheets.py` and `add_collective_comparison.py`
5. Creates final report using `create_final_report.py`

All post-processing is handled automatically - no need to run individual scripts.


## Scripts

```
run_full_analysis.py            - Master script for complete pipeline
create_final_report.py          - Create comprehensive Excel report
run_tracelens_single_config.sh  - Main TraceLens report generation
process_gpu_timeline.py         - Aggregate GPU timeline across ranks
combine_reports.py              - Combine two runs
add_comparison_sheets.py        - Add GPU timeline comparison sheets
add_collective_comparison.py    - Add collective/NCCL comparison sheets
```
