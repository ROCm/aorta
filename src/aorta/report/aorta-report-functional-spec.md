# aorta-report Functional Specification

**Version:** 1.1  
**Date:** January 2026  
**Status:** Partially Implemented

---

## 1. Current State Summary

| Directory | Scripts | Purpose |
|-----------|---------|---------|
| `gemm_analysis/` | 8 Python + 2 Shell | GEMM-specific analysis, sweep directories, HTML reports |
| `tracelens_single_config/` | 9 Python + 2 Shell | Single-config analysis, report comparison, final reports |
| Shared | `tracelens_with_gemm_patch.py` | GEMM-patched TraceLens wrapper |

### 1.1 Scripts Inventory

#### `gemm_analysis/` directory

| Script | Type | Description |
|--------|------|-------------|
| `run_tracelens_analysis.sh` | Shell | Main pipeline for sweep directory analysis |
| `run_train_various_channels.sh` | Shell | Training with various NCCL channels |
| `analyze_gemm_reports.py` | Python | Analyze GEMM reports from Excel |
| `create_embeded_html_report.py` | Python | Create HTML report with embedded images |
| `enhance_gemm_variance_with_timestamps.py` | Python | Enhance GEMM variance data |
| `gemm_report_with_collective_overlap.py` | Python | GEMM report with collective overlap |
| `plot_gemm_variance.py` | Python | Plot GEMM variance |
| `process_comms.py` | Python | Process communication data |
| `process_gpu_timeline.py` | Python | Process GPU timeline (sweep mode) |
| `html_template.py` | Python | HTML template for reports |

#### `tracelens_single_config/` directory

| Script | Type | Description |
|--------|------|-------------|
| `run_tracelens_single_config.sh` | Shell | TraceLens for single config |
| `run_rccl_warp_speed_comparison.sh` | Shell | RCCL comparison |
| `run_full_analysis.py` | Python | Master pipeline script |
| `add_collective_comparison.py` | Python | Add collective comparison sheets |
| `add_comparison_sheets.py` | Python | Add comparison sheets |
| `combine_reports.py` | Python | Combine reports |
| `compare_all_runs.py` | Python | Compare all runs |
| `create_final_html.py` | Python | Create final HTML report |
| `create_final_plots.py` | Python | Create final plots |
| `create_final_report.py` | Python | Create final Excel report |
| `process_gpu_timeline.py` | Python | Process GPU timeline (single mode) |
| `html_report_config.py` | Python | HTML config |


---

## 2. Proposed CLI Architecture

### 2.1 Command Hierarchy

```
aorta-report
│
├── analyze                 # Core analysis commands
│   ├── single              # Single config analysis
│   ├── sweep               # Sweep directory analysis
│   └── gemm                # GEMM-specific analysis
│
├── compare                 # Comparison commands
│   ├── runs                # Compare multiple runs
│   ├── reports             # Compare two reports
│   └── collective          # Compare collective ops
│
├── generate                # Report generation
│   ├── html                # HTML report generation
│   ├── excel               # Excel report generation
│   └── plots               # Generate visualization plots
│
├── process                 # Data processing utilities
│   ├── gpu-timeline        # Process GPU timeline data
│   ├── comms               # Process communications data
│   └── gemm-variance       # Enhance GEMM variance data
│
└── pipeline                # Full analysis pipelines
    ├── full                # Complete analysis pipeline
    └── gemm                # GEMM-focused pipeline
```

### 2.2 Command Specifications

#### 2.2.1 `analyze` Group

| Command | Arguments | Options | Description |
|---------|-----------|---------|-------------|
| `analyze single` | `TRACE_DIR` | `--individual-only`, `--collective-only`, `-o/--output` | Analyze single configuration trace directory |
| `analyze sweep` | `SWEEP_DIR` | `--rocprof`, `-o/--output` | Analyze sweep directory with multiple configs |
| `analyze gemm` | `REPORTS_DIR` | `--top-k`, `-o/--output` | Analyze GEMM kernels from TraceLens reports |

#### 2.2.2 `compare` Group

| Command | Arguments | Options | Description |
|---------|-----------|---------|-------------|
| `compare runs` | - | `-i/--inputs` (multiple), `-o/--output` | Compare multiple TraceLens analysis runs |
| `compare reports` | - | `-b/--baseline`, `-t/--test`, `--baseline-label`, `--test-label`, `-o/--output` | Combine and compare two reports |
| `compare collective` | - | `--input`, `-o/--output`, `--baseline-label`, `--test-label` | Add collective comparison sheets |

#### 2.2.3 `generate` Group

| Command | Arguments | Options | Description |
|---------|-----------|---------|-------------|
| `generate html` | - | `--mode` (sweep/performance), mode-specific options, `-o/--output` | Generate HTML report with embedded images |
| `generate excel` | - | `--gpu-combined`, `--gpu-comparison`, `--coll-combined`, `--coll-comparison`, `-o/--output` | Generate comprehensive Excel report |
| `generate plots` | - | `-i/--input`, `-o/--output`, `--type` | Generate visualization plots |

##### `generate html` Modes

| Mode | Required Options | Optional Options | Description |
|------|-----------------|------------------|-------------|
| `sweep` | `--sweep1`, `--sweep2` | `--label1`, `--label2` | GEMM variance comparison between two sweeps |
| `performance` | `--plots-dir` | - | GPU/NCCL performance analysis report |

#### 2.2.4 `process` Group ✅ Implemented

| Command | Arguments | Options | Description |
|---------|-----------|---------|-------------|
| `process gpu-timeline` | `INPUT_DIR` | `--mode` (auto/single/sweep), `--geo-mean`, `-o/--output` | Process GPU timeline from reports |
| `process comms` | `SWEEP_DIR` | `-o/--output` | Process NCCL communication data from collective reports |
| `process gemm-variance` | `INPUT_CSV` | `--base-path` (required), `--tolerance`, `-o/--output` | Enhance GEMM variance CSV with kernel timestamps |

#### 2.2.5 `pipeline` Group

| Command | Arguments | Options | Description |
|---------|-----------|---------|-------------|
| `pipeline full` | - | `-b/--baseline`, `-t/--test` (multiple), `-o/--output`, `--skip-tracelens`, `--gpu-timeline/--no-gpu-timeline`, `--collective/--no-collective`, `--final-report/--no-final-report`, `--plots/--no-plots` | Run complete analysis pipeline |
| `pipeline gemm` | - | `--sweep-dir`, `-o/--output`, `--top-k` | Run GEMM-focused analysis pipeline |

### 2.3 Global Options

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--help` | Show help message and exit |
| `-v/--verbose` | Enable verbose output |
| `--quiet` | Suppress non-error output |

---

## 3. Usage Examples

### 3.1 Single Configuration Analysis

```bash
# Analyze a single trace directory
aorta-report analyze single /path/to/traces --output ./results

# Generate only individual reports (skip collective)
aorta-report analyze single /path/to/traces --individual-only

# Generate only collective report (skip individual)
aorta-report analyze single /path/to/traces --collective-only
```

### 3.2 Sweep Analysis

```bash
# Analyze sweep directory with PyTorch profiler traces
aorta-report analyze sweep /path/to/sweep_20251124

# Analyze sweep directory with rocprof traces
aorta-report analyze sweep /path/to/sweep_20251124 --rocprof
```

### 3.3 GEMM Analysis

```bash
# Analyze GEMM kernels with default top 5
aorta-report analyze gemm /path/to/reports

# Analyze GEMM kernels with custom top-k
aorta-report analyze gemm /path/to/reports --top-k 10 -o gemm_analysis.csv
```

### 3.4 Report Comparison

```bash
# Compare two reports
aorta-report compare reports \
    --baseline baseline.xlsx \
    --test test.xlsx \
    --baseline-label "ROCm 6.0" \
    --test-label "ROCm 7.0" \
    --output comparison.xlsx

# Compare multiple runs
aorta-report compare runs \
    -i /path/to/run1 \
    -i /path/to/run2 \
    -i /path/to/run3 \
    -o /path/to/comparison_output
```

### 3.5 Report Generation

```bash
# Generate HTML report - SWEEP MODE (GEMM variance comparison)
aorta-report generate html --mode sweep \
    --sweep1 ./experiments/baseline \
    --sweep2 ./experiments/test \
    --label1 "Baseline" \
    --label2 "Optimized" \
    -o gemm_comparison.html

# Generate HTML report - PERFORMANCE MODE (GPU/NCCL analysis)
aorta-report generate html --mode performance \
    --plots-dir ./output/plots \
    -o performance_report.html

# Generate visualization plots
aorta-report generate plots \
    --input final_report.xlsx \
    --output ./plots/
```

### 3.6 Data Processing

#### GPU Timeline Processing

```bash
# Auto-detect input type and process
aorta-report process gpu-timeline /path/to/reports

# Explicit single config mode (perf_rank*.xlsx files)
aorta-report process gpu-timeline /path/to/individual_reports --mode single

# Sweep mode with geometric mean (perf_*ch_rank*.xlsx files)
aorta-report process gpu-timeline /path/to/sweep --mode sweep --geo-mean

# Custom output path
aorta-report process gpu-timeline /path/to/sweep -o ./results/timeline.xlsx
```

#### NCCL Communication Processing

```bash
# Process NCCL collective reports from sweep directory
aorta-report process comms /path/to/sweep

# Custom output directory
aorta-report process comms /path/to/sweep -o ./nccl_analysis/
```

#### GEMM Variance Timestamp Enhancement

```bash
# Enhance GEMM variance CSV with kernel timestamps
aorta-report process gemm-variance ./gemm_variance.csv --base-path /path/to/sweep

# Custom tolerance and output
aorta-report process gemm-variance ./variance.csv --base-path /path/to/sweep \
    --tolerance 0.02 -o ./enhanced.csv
```

### 3.7 Full Pipeline

```bash
# Run complete analysis with all options
aorta-report pipeline full \
    --baseline /path/to/baseline \
    --test /path/to/test \
    --output /path/to/output \
    --plots

# Skip TraceLens if already generated
aorta-report pipeline full \
    --baseline /path/to/baseline \
    --test /path/to/test \
    --output /path/to/output \
    --skip-tracelens \
    --final-report \
    --plots

# Compare multiple test configurations against baseline
aorta-report pipeline full \
    --baseline /path/to/baseline \
    --test /path/to/test1 \
    --test /path/to/test2 \
    --output /path/to/output
```

### 3.8 Help and Discovery

```bash
# Show all available commands
aorta-report --help

# Show analyze subcommands
aorta-report analyze --help

# Show specific command help
aorta-report analyze single --help
aorta-report pipeline full --help
```

---

## 4. Script Mapping Reference

### 4.1 Shell Scripts → CLI Commands

| Old Script | New Command |
|------------|-------------|
| `run_tracelens_single_config.sh` | `aorta-report analyze single` |
| `run_tracelens_analysis.sh` | `aorta-report analyze sweep` |
| `run_rccl_warp_speed_comparison.sh` | `aorta-report compare rccl` |

### 4.2 Python Scripts → CLI Commands

| Old Script | New Command |
|------------|-------------|
| `analyze_gemm_reports.py` | `aorta-report analyze gemm` |
| `run_full_analysis.py` | `aorta-report pipeline full` |
| `compare_all_runs.py` | `aorta-report compare runs` |
| `combine_reports.py` | `aorta-report compare reports` |
| `add_collective_comparison.py` | `aorta-report compare collective` |
| `add_comparison_sheets.py` | `aorta-report compare reports` (merged) |
| `create_embeded_html_report.py` | `aorta-report generate html --mode sweep` |
| `create_final_html.py` | `aorta-report generate html --mode performance` |
| `create_final_report.py` | `aorta-report generate excel` |
| `create_final_plots.py` | `aorta-report generate plots` |
| `plot_gemm_variance.py` | `aorta-report generate plots --type gemm-variance` |
| `process_gpu_timeline.py` (both) | `aorta-report process gpu-timeline` |
| `process_comms.py` | `aorta-report process comms` |
| `enhance_gemm_variance_with_timestamps.py` | `aorta-report process gemm-variance` |

### 4.3 Migration Quick Reference

```bash
# Old way (shell)
bash ./scripts/tracelens_single_config/run_tracelens_single_config.sh /path/to/traces

# New way (CLI)
aorta-report analyze single /path/to/traces
```

```bash
# Old way (shell)
bash ./scripts/gemm_analysis/run_tracelens_analysis.sh /path/to/sweep --rocprof

# New way (CLI)
aorta-report analyze sweep /path/to/sweep --rocprof
```

```bash
# Old way (python)
python ./scripts/tracelens_single_config/run_full_analysis.py \
    --baseline /path/to/baseline \
    --test /path/to/test \
    --output /path/to/output \
    --all

# New way (CLI)
aorta-report pipeline full \
    --baseline /path/to/baseline \
    --test /path/to/test \
    --output /path/to/output \
    --plots
```

```bash
# Old way (python)
python ./scripts/gemm_analysis/process_gpu_timeline.py --sweep-dir /path/to/sweep --geo-mean

# New way (CLI)
aorta-report process gpu-timeline /path/to/sweep --mode sweep --geo-mean
```

---

## Appendix A: Entry Point Configuration

Add to `pyproject.toml`:

```toml
[project.scripts]
aorta-report = "aorta.report:main"
```

## Appendix B: Package Structure

```
aorta/src/aorta/report/
├── __init__.py
├── __main__.py                  # python -m aorta.report
├── cli.py                       # Click CLI definition
├── analysis/                    # ✅ Implemented - analyze command logic
│   ├── __init__.py
│   ├── tracelens_wrapper.py     # GEMM-patched TraceLens wrapper
│   ├── analyze_gemm.py          # GEMM kernel variance analysis
│   ├── analyze_single.py        # Single configuration analysis
│   └── analyze_sweep.py         # Sweep configuration analysis
├── generators/                  # ✅ Implemented - generate html command
│   ├── __init__.py
│   ├── html_generator.py        # Unified HTML generation entry point
│   ├── sweep_comparison.py      # GEMM sweep comparison mode
│   └── performance_report.py    # GPU/NCCL performance mode
├── templates/                   # ✅ Implemented - HTML templates
│   ├── __init__.py
│   ├── sweep_comparison_template.py
│   └── performance_report_template.py
├── processing/                  # ✅ Implemented - process command logic
│   ├── __init__.py
│   ├── gpu_timeline_single.py   # Single config GPU timeline processing
│   ├── gpu_timeline_sweep.py    # Sweep GPU timeline processing
│   ├── process_comms.py         # NCCL communication data processing
│   └── process_gemm_variance.py # GEMM variance timestamp enhancement
├── ANALYZE_CMD_DEV_DOCS.md      # Developer documentation
├── GENERATE_HTML_DEV_DOCS.md    # Developer documentation
├── PROCESS_CMD_DEV_DOCS.md      # Developer documentation
├── aorta-report-detail-plan.md  # Implementation plan
└── aorta-report-functional-spec.md  # This document
```

## Appendix C: Implementation Status

| Command Group | Status | Notes |
|---------------|--------|-------|
| `analyze` | ✅ Implemented | `single`, `sweep`, `gemm` commands working |
| `compare` | ⏳ Pending | CLI stubs exist, logic not implemented |
| `generate` | ⚠️ Partial | `html` implemented, `excel`/`plots` pending |
| `process` | ✅ Implemented | All commands working (`gpu-timeline`, `comms`, `gemm-variance`) |
| `pipeline` | ⏳ Pending | CLI stubs exist, logic not implemented |

