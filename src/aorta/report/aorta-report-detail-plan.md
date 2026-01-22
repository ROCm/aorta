# aorta-report Detailed Implementation Plan

## Current State Summary

| Directory | Scripts | Purpose |
|-----------|---------|---------|
| `gemm_analysis/` | 8 Python + 2 Shell | GEMM-specific analysis, sweep directories, HTML reports |
| `tracelens_single_config/` | 9 Python + 2 Shell | Single-config analysis, report comparison, final reports |
| Shared | `tracelens_with_gemm_patch.py` | GEMM-patched TraceLens wrapper |

### Current Scripts Inventory

#### `gemm_analysis/` directory:

**Shell scripts:**
- `run_tracelens_analysis.sh` - Main pipeline for sweep directory analysis
- `run_train_various_channels.sh` - Training with various NCCL channels

**Python scripts:**
- `analyze_gemm_reports.py` - Analyze GEMM reports from Excel
- `create_embeded_html_report.py` - Create HTML report with embedded images
- `enhance_gemm_variance_with_timestamps.py` - Enhance GEMM variance data
- `gemm_report_with_collective_overlap.py` - GEMM report with collective overlap
- `plot_gemm_variance.py` - Plot GEMM variance
- `process_comms.py` - Process communication data
- `process_gpu_timeline.py` - Process GPU timeline

**Support files:**
- `html_template.py` - HTML template for reports
- `rocprof_*.yaml` - rocprof config files

#### `tracelens_single_config/` directory:

**Shell scripts:**
- `run_tracelens_single_config.sh` - TraceLens for single config
- `run_rccl_warp_speed_comparison.sh` - RCCL comparison

**Python scripts:**
- `run_full_analysis.py` - Master pipeline script
- `add_collective_comparison.py` - Add collective comparison sheets
- `add_comparison_sheets.py` - Add comparison sheets
- `combine_reports.py` - Combine reports
- `compare_all_runs.py` - Compare all runs
- `create_final_html.py` - Create final HTML report
- `create_final_plots.py` - Create final plots
- `create_final_report.py` - Create final Excel report
- `process_gpu_timeline.py` - Process GPU timeline (duplicate!)

**Support files:**
- `html_report_config.py` - HTML config

### Key Issues

- Duplicate scripts (`process_gpu_timeline.py` exists in both directories)
- Mixed Shell + Python entry points
- No unified interface
- Hard to discover available functionality
- Inconsistent argument styles

---

## Duplicate Script Analysis: `process_gpu_timeline.py`

Both directories contain a `process_gpu_timeline.py` file with overlapping but different functionality.

### Comparison Summary

| Aspect | `gemm_analysis/` | `tracelens_single_config/` |
|--------|------------------|----------------------------|
| **Lines of Code** | 468 | 101 |
| **Purpose** | Multi-config sweep analysis | Single config analysis |
| **Input Argument** | `--sweep-dir` | `--reports-dir` |
| **File Pattern** | `perf_28ch_rank0.xlsx` | `perf_rank0.xlsx` |

### Detailed Differences

#### 1. Scope & Directory Structure

**`gemm_analysis/`** - Handles **sweep directories** with multiple thread/channel configurations:
```
sweep_dir/
└── tracelens_analysis/
    ├── 256thread/
    │   └── individual_reports/
    │       ├── perf_28ch_rank0.xlsx
    │       ├── perf_28ch_rank1.xlsx
    │       ├── perf_56ch_rank0.xlsx
    │       └── ...
    └── 384thread/
        └── individual_reports/
            └── ...
```

**`tracelens_single_config/`** - Handles **single configuration** flat directory:
```
reports_dir/
├── perf_rank0.xlsx
├── perf_rank1.xlsx
└── ...
```

#### 2. Input Arguments

```python
# gemm_analysis/process_gpu_timeline.py
parser.add_argument("--sweep-dir", required=True, 
    help="Path to sweep directory (e.g., sweep_20251124_222204)")

# tracelens_single_config/process_gpu_timeline.py
parser.add_argument("--reports-dir", required=True, 
    help="Path to individual_reports directory")
```

#### 3. Metadata Handling

**`gemm_analysis/`** extracts and adds rich metadata:
```python
aggregated["thread_config"] = thread_config      # e.g., "256thread"
aggregated["threads_num"] = int(...)             # e.g., 256
aggregated["channel_config"] = channel_config    # e.g., "28ch"
aggregated["channels_num"] = int(...)            # e.g., 28
aggregated["full_config"] = f"{thread_config}_{channel_config}"  # e.g., "256thread_28ch"
aggregated["num_ranks"] = num_ranks
```

**`tracelens_single_config/`** - minimal metadata:
```python
aggregated["num_ranks"] = len(perf_files)
```

#### 4. Output Excel Sheets

**`gemm_analysis/`** creates:
| Sheet | Description |
|-------|-------------|
| `All_Data` | Complete dataset with all configs |
| `Pivot_Time_ms` | Matrix: type × full_config |
| `Pivot_Percent` | Matrix: type × full_config |
| `Summary_By_Config` | Key metrics per configuration |

**`tracelens_single_config/`** creates:
| Sheet | Description |
|-------|-------------|
| `Summary` | Aggregated metrics |
| `All_Ranks_Combined` | Raw data from all ranks |
| `Per_Rank_Time_ms` | Matrix: type × rank |
| `Per_Rank_Percent` | Matrix: type × rank |

#### 5. Output File Location

```python
# gemm_analysis/
output_path = tracelens_dir / f"gpu_timeline_all_configs_{method_suffix}.xlsx"

# tracelens_single_config/
output_path = reports_path.parent / f"gpu_timeline_summary_{method_suffix}.xlsx"
```

### Shared Code (Consolidation Candidates)

Both files have **identical** `geometric_mean()` function:
```python
def geometric_mean(values):
    """Calculate geometric mean, handling zeros."""
    values = np.array(values)
    values = np.where(values == 0, 1e-10, values)
    return np.exp(np.mean(np.log(values)))
```

And **similar** aggregation logic:
```python
agg_func = geometric_mean if use_geo_mean else "mean"
aggregated = (
    combined.groupby("type")
    .agg({"time ms": agg_func, "percent": agg_func})
    .reset_index()
)
```

### Consolidation Recommendation

Create a unified `process_gpu_timeline()` command that:

1. **Auto-detects** input type (sweep dir vs single reports dir)
2. Uses `--mode` flag with options: `auto`, `single`, `sweep`
3. Shares common aggregation logic via a core module
4. Generates appropriate output based on detected/specified mode

**Proposed unified CLI command:**
```python
@process.command("gpu-timeline")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--mode", type=click.Choice(["auto", "single", "sweep"]), default="auto",
              help="Processing mode: auto-detect, single config, or sweep")
@click.option("--geo-mean", is_flag=True, help="Use geometric mean instead of arithmetic mean")
@click.option("--output", "-o", help="Output file path (auto-generated if not specified)")
def process_gpu_timeline(input_dir, mode, geo_mean, output):
    """Process GPU timeline data from TraceLens reports.
    
    Supports both single-config and sweep directory structures.
    Auto-detects the structure by default.
    
    Examples:
        # Auto-detect mode
        aorta-report process gpu-timeline /path/to/reports
        
        # Explicit single config
        aorta-report process gpu-timeline /path/to/individual_reports --mode single
        
        # Sweep directory with geometric mean
        aorta-report process gpu-timeline /path/to/sweep --mode sweep --geo-mean
    """
    from aorta.report.core.gpu_timeline import (
        process_single_config,
        process_sweep_config,
        detect_input_type
    )
    
    if mode == "auto":
        mode = detect_input_type(input_dir)
    
    if mode == "single":
        return process_single_config(input_dir, geo_mean, output)
    else:
        return process_sweep_config(input_dir, geo_mean, output)
```

**Core module structure:**
```python
# aorta/src/aorta/report/core/gpu_timeline.py

def geometric_mean(values):
    """Shared geometric mean calculation."""
    ...

def aggregate_rank_data(rank_data, use_geo_mean):
    """Shared aggregation logic."""
    ...

def detect_input_type(input_dir):
    """Auto-detect if input is single config or sweep."""
    ...

def process_single_config(reports_dir, use_geo_mean, output):
    """Process single configuration (from tracelens_single_config)."""
    ...

def process_sweep_config(sweep_dir, use_geo_mean, output):
    """Process sweep directory (from gemm_analysis)."""
    ...
```

---

## Proposed CLI Architecture

```
aorta-report
├── analyze           # Core analysis commands
│   ├── single        # Single config analysis (was: run_tracelens_single_config.sh)
│   ├── sweep         # Sweep directory analysis (was: run_tracelens_analysis.sh)
│   └── gemm          # GEMM-specific analysis (was: analyze_gemm_reports.py)
│
├── compare           # Comparison commands
│   ├── runs          # Compare multiple runs (was: compare_all_runs.py)
│   ├── reports       # Compare two reports (was: combine_reports.py)
│   └── collective    # Compare collective ops (was: add_collective_comparison.py)
│
├── generate          # Report generation
│   ├── html          # HTML report (was: create_embeded_html_report.py, create_final_html.py)
│   ├── excel         # Excel report (was: create_final_report.py)
│   └── plots         # Generate plots (was: create_final_plots.py, plot_gemm_variance.py)
│
├── process           # Data processing
│   ├── gpu-timeline  # Process GPU timeline (consolidated)
│   ├── comms         # Process communications
│   └── gemm-variance # Enhance GEMM variance
│
└── pipeline          # Full pipelines (composite commands)
    ├── full          # Full analysis pipeline (was: run_full_analysis.py)
    └── gemm          # GEMM-focused pipeline
```

---

## Implementation Plan

### Phase 1: Create CLI Foundation

**Create new package:** `aorta/src/aorta/report/`

```python
# aorta/src/aorta/report/cli.py
import click

@click.group()
@click.version_option()
def cli():
    """TraceLens Analysis CLI - Unified interface for trace analysis."""
    pass

# === ANALYZE GROUP ===
@cli.group()
def analyze():
    """Run TraceLens analysis on traces."""
    pass

@analyze.command("single")
@click.argument("trace_dir", type=click.Path(exists=True))
@click.option("--individual-only", is_flag=True, help="Generate only individual reports")
@click.option("--collective-only", is_flag=True, help="Generate only collective report")
@click.option("--output", "-o", help="Output directory")
def analyze_single(trace_dir, individual_only, collective_only, output):
    """Analyze a single configuration trace directory."""
    # Consolidate run_tracelens_single_config.sh logic
    pass

@analyze.command("sweep")
@click.argument("sweep_dir", type=click.Path(exists=True))
@click.option("--rocprof", is_flag=True, help="Use rocprof traces instead of PyTorch profiler")
@click.option("--output", "-o", help="Output directory")
def analyze_sweep(sweep_dir, rocprof, output):
    """Analyze a sweep directory with multiple configurations."""
    # Consolidate run_tracelens_analysis.sh logic
    pass

@analyze.command("gemm")
@click.argument("reports_dir", type=click.Path(exists=True))
@click.option("--top-k", default=5, help="Number of top kernels to extract")
@click.option("--output", "-o", help="Output CSV file")
def analyze_gemm(reports_dir, top_k, output):
    """Analyze GEMM kernels from TraceLens reports."""
    # From analyze_gemm_reports.py
    pass

# === COMPARE GROUP ===
@cli.group()
def compare():
    """Compare traces and reports."""
    pass

@compare.command("runs")
@click.option("--inputs", "-i", multiple=True, required=True, help="Input directories")
@click.option("--output", "-o", required=True, help="Output directory")
def compare_runs(inputs, output):
    """Compare multiple TraceLens analysis runs."""
    pass

@compare.command("reports")
@click.option("--baseline", "-b", required=True, help="Baseline report")
@click.option("--test", "-t", required=True, help="Test report")
@click.option("--baseline-label", help="Label for baseline")
@click.option("--test-label", help="Label for test")
@click.option("--output", "-o", required=True, help="Output file")
def compare_reports(baseline, test, baseline_label, test_label, output):
    """Combine and compare two reports."""
    pass

# === GENERATE GROUP ===
@cli.group()
def generate():
    """Generate reports and visualizations."""
    pass

@generate.command("html")
@click.option("--sweep1", required=True, help="First sweep directory")
@click.option("--sweep2", help="Second sweep directory (for comparison)")
@click.option("--label1", help="Label for first sweep")
@click.option("--label2", help="Label for second sweep")
@click.option("--output", "-o", required=True, help="Output HTML file")
def generate_html(sweep1, sweep2, label1, label2, output):
    """Generate HTML report with embedded images."""
    pass

@generate.command("excel")
@click.option("--gpu-combined", required=True)
@click.option("--gpu-comparison", required=True)
@click.option("--coll-combined", required=True)
@click.option("--coll-comparison", required=True)
@click.option("--output", "-o", required=True)
def generate_excel(gpu_combined, gpu_comparison, coll_combined, coll_comparison, output):
    """Generate comprehensive Excel report."""
    pass

@generate.command("plots")
@click.option("--input", "-i", required=True, help="Input Excel report")
@click.option("--output", "-o", required=True, help="Output directory")
def generate_plots(input, output):
    """Generate visualization plots."""
    pass

# === PIPELINE GROUP ===
@cli.group()
def pipeline():
    """Run complete analysis pipelines."""
    pass

@pipeline.command("full")
@click.option("--baseline", "-b", required=True, help="Baseline trace directory")
@click.option("--test", "-t", required=True, multiple=True, help="Test trace directory(s)")
@click.option("--output", "-o", required=True, help="Output directory")
@click.option("--skip-tracelens", is_flag=True, help="Skip TraceLens generation")
@click.option("--gpu-timeline/--no-gpu-timeline", default=True)
@click.option("--collective/--no-collective", default=True)
@click.option("--final-report/--no-final-report", default=True)
@click.option("--plots/--no-plots", default=True)
def pipeline_full(baseline, test, output, skip_tracelens, gpu_timeline, collective, final_report, plots):
    """Run complete analysis pipeline with comparisons."""
    # Consolidate run_full_analysis.py
    pass
```

---

### Phase 2: Migrate Logic from Shell to Python

| Shell Script | → | Python Function |
|--------------|---|-----------------|
| `run_tracelens_single_config.sh` | → | `analyze.single()` |
| `run_tracelens_analysis.sh` | → | `analyze.sweep()` |
| `run_rccl_warp_speed_comparison.sh` | → | `compare.rccl()` (new) |

### Phase 3: Consolidate Duplicate Code

- Merge both `process_gpu_timeline.py` files
- Create shared utilities in `aorta/src/aorta/report/utils/`
- Move `html_template.py` to shared location

### Phase 4: File Structure

```
aorta/src/aorta/report/
├── __init__.py
├── __main__.py          # Entry point: python -m aorta.report
├── cli.py               # Click CLI definition
├── commands/
│   ├── __init__.py
│   ├── analyze.py       # analyze subcommands
│   ├── compare.py       # compare subcommands
│   ├── report.py        # report subcommands
│   └── pipeline.py      # pipeline subcommands
├── core/
│   ├── __init__.py
│   ├── tracelens_wrapper.py  # GEMM-patched TraceLens (from tracelens_with_gemm_patch.py)
│   ├── gpu_timeline.py       # Consolidated GPU timeline processing
│   ├── gemm_analysis.py      # GEMM analysis logic
│   └── report_generator.py   # Report generation logic
└── templates/
    ├── html_template.py
    └── html_report_config.py
```

### Phase 5: Entry Points

Add to `pyproject.toml`:

```toml
[project.scripts]
aorta-report = "aorta.report:main"
```

---

## Usage Examples (After Implementation)

```bash
# Single config analysis
aorta-report analyze single /path/to/traces --output ./results

# Sweep analysis  
aorta-report analyze sweep /path/to/sweep --rocprof

# GEMM analysis
aorta-report analyze gemm /path/to/reports --top-k 10 -o gemm_analysis.csv

# Compare two runs
aorta-report compare reports -b baseline.xlsx -t test.xlsx -o comparison.xlsx

# Generate HTML report
aorta-report generate html --sweep1 ./exp1 --sweep2 ./exp2 -o report.html

# Full pipeline
aorta-report pipeline full \
    --baseline /path/to/baseline \
    --test /path/to/test \
    --output /path/to/output \
    --plots

# List available commands
aorta-report --help
aorta-report analyze --help
aorta-report compare --help
```

---

## Migration Strategy

| Phase | Duration | Description |
|-------|----------|-------------|
| **Phase 1** | 1-2 days | Create CLI skeleton with Click |
| **Phase 2** | 2-3 days | Migrate Python scripts as command handlers |
| **Phase 3** | 1-2 days | Convert shell scripts to Python |
| **Phase 4** | 1 day | Add tests and documentation |
| **Phase 5** | Optional | Deprecate old scripts with warnings |

### Backward Compatibility

During migration, keep old scripts working by having them call the new CLI:

```bash
#!/bin/bash
# Legacy wrapper for run_tracelens_single_config.sh
echo "DEPRECATED: Use 'aorta-report analyze single' instead"
exec aorta-report analyze single "$@"
```

---

## Script Mapping Reference

| Old Script | New Command |
|------------|-------------|
| `run_tracelens_single_config.sh` | `aorta-report analyze single` |
| `run_tracelens_analysis.sh` | `aorta-report analyze sweep` |
| `analyze_gemm_reports.py` | `aorta-report analyze gemm` |
| `run_full_analysis.py` | `aorta-report pipeline full` |
| `compare_all_runs.py` | `aorta-report compare runs` |
| `combine_reports.py` | `aorta-report compare reports` |
| `add_collective_comparison.py` | `aorta-report compare collective` |
| `create_embeded_html_report.py` | `aorta-report generate html` |
| `create_final_report.py` | `aorta-report generate excel` |
| `create_final_plots.py` | `aorta-report generate plots` |
| `process_gpu_timeline.py` | `aorta-report process gpu-timeline` |
| `process_comms.py` | `aorta-report process comms` |
| `enhance_gemm_variance_with_timestamps.py` | `aorta-report process gemm-variance` |
| `plot_gemm_variance.py` | `aorta-report generate plots --type gemm-variance` |

---

## Benefits

1. **Discoverability** - `--help` at every level shows available commands
2. **Consistency** - Uniform argument style across all commands
3. **Composability** - Easy to chain commands in scripts
4. **Maintainability** - Single codebase, no shell script maintenance
5. **Testability** - Python functions are easier to unit test
6. **Documentation** - Auto-generated from docstrings and Click decorators

