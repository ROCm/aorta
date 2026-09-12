# HW Queue Eval Pipeline Enhancement Plan for `aorta-report`

**Version**: 1.1  
**Created**: 2026-02-12  
**Status**: Awaiting Review  
**Last Updated**: 2026-02-12

---

## Table of Contents

1. [Overview](#1-overview)
2. [Supported Modes](#2-supported-modes)
3. [JSON Data Structures](#3-json-data-structures)
4. [CLI Command Design](#4-cli-command-design)
5. [Pipeline Steps](#5-pipeline-steps)
6. [Output File Structure](#6-output-file-structure)
7. [Excel Report Structure](#7-excel-report-structure)
8. [Plot Types](#8-plot-types)
9. [Implementation Structure](#9-implementation-structure)
10. [Implementation Tasks](#10-implementation-tasks)
11. [Comparison Logic](#11-comparison-logic)
12. [Reuse from Existing Code](#12-reuse-from-existing-code)
13. [Design Decisions](#13-design-decisions)
14. [Future Enhancements](#14-future-enhancements)

---

## 1. Overview

Add a new `hwqueue` pipeline to `aorta-report` that processes JSON output from `hw_queue_eval` and generates comprehensive reports with visualizations.

**Key Goals:**
- Process single workload runs and sweeps
- Support multi-workload comparison between baseline and test configurations
- Generate consolidated Excel reports (similar to `all_workloads_analysis.xlsx`)
- Create visualizations matching existing `aorta-report` styling
- Generate static HTML reports for easy sharing

---

## 2. Supported Modes

### Mode A: Single Workload - Single Run
**Input**: One JSON file from a single `run` command  
**Use Case**: Quick analysis of one workload at one stream count

### Mode B: Single Workload - Sweep
**Input**: One JSON file from a `sweep` command (contains results for multiple stream counts)  
**Use Case**: Analyze scaling behavior of one workload across stream counts

### Mode C: Multi-Workload Comparison (Baseline vs Test)
**Input**: Two directories, each containing multiple JSON files (one per workload) from `run-priority`  
**Use Case**: Compare all workloads between baseline and test configurations  
**Output**: Consolidated `all_workloads_comparison.xlsx`

**Important Behavior for Mode C:**
- Only operates on **common workloads** present in both baseline and test directories
- Clearly reports missing workloads (present in baseline but not test, or vice versa)
- Does NOT generate per-workload individual files (only consolidated report)

---

## 3. JSON Data Structures

### 3.1 Single Run Result (`HarnessResult.to_json()`)
```json
{
  "throughput": 1234.56,
  "throughput_unit": "ops/sec",
  "latency_ms": {
    "mean": 1.2,
    "p50": 1.1,
    "p95": 1.5,
    "p99": 2.0,
    "min": 0.8,
    "max": 3.0,
    "std": 0.3
  },
  "total_time_ms": 12345.6,
  "stream_count": 8,
  "per_stream_times_ms": [100.5, 102.3, ...],
  "iteration_times_ms": [1.2, 1.1, 1.3, ...],
  "switch_latency": {
    "inter_stream_gap_ms": 0.05,
    "intra_stream_gap_ms": 0.01,
    "estimated_switch_overhead_ms": 0.04
  },
  "memory": {
    "peak_allocated_gb": 2.5,
    "peak_reserved_gb": 3.0,
    "final_allocated_gb": 2.0,
    "final_reserved_gb": 2.5
  },
  "metadata": {...},
  "workload_name": "hetero_kernels",
  "timestamp": "2026-02-12T..."
}
```

### 3.2 Sweep Result (`save_sweep_results()`)
```json
{
  "timestamp": "2026-02-12T...",
  "workload": "hetero_kernels",
  "results": [
    { /* Single run result at stream_count=1 */ },
    { /* Single run result at stream_count=2 */ },
    { /* ... */ }
  ],
  "environment": {
    "hostname": "...",
    "kernel": "...",
    "hip_version": "...",
    "torch_version": "...",
    "gpu_count": 8,
    "gpus": [...]
  },
  "analysis": {
    "stream_counts": [1, 2, 4, 8, 16],
    "throughputs": [...],
    "efficiencies": [...],
    "inflection_point": 8,
    "peak_stream_count": 4
  }
}
```

### 3.3 Multi-Workload Directory Structure
Output from `run-priority` command:
```
results_baseline/
├── environment_info.json
├── hetero_kernels_results.json      # Sweep result for hetero_kernels
├── tiny_kernel_stress_results.json  # Sweep result for tiny_kernel_stress
├── large_gemm_only_results.json
├── moe_results.json
└── ...

results_test/
├── environment_info.json
├── hetero_kernels_results.json
├── tiny_kernel_stress_results.json
└── ...
```

---

## 4. CLI Command Design

### 4.1 Command Signature

```bash
aorta-report pipeline hwqueue [OPTIONS]
```

### 4.2 Options

| Option | Type | Description | Required |
|--------|------|-------------|----------|
| `--input, -i` | PATH | Single JSON file (run or sweep) | For Mode A/B |
| `--baseline-dir` | PATH | Directory with baseline workload results | For Mode C |
| `--test-dir` | PATH | Directory with test workload results | For Mode C |
| `--output, -o` | PATH | Output directory | Yes |
| `--baseline-label` | STRING | Label for baseline in reports | No (default: dir name) |
| `--test-label` | STRING | Label for test in reports | No (default: dir name) |
| `--threshold` | FLOAT | Regression threshold (0.05 = 5%) | No (default: 0.05) |
| `--excel/--no-excel` | FLAG | Generate Excel report | No (default: True) |
| `--plots/--no-plots` | FLAG | Generate plots | No (default: True) |
| `--html/--no-html` | FLAG | Generate HTML report | No (default: True) |

### 4.3 Usage Examples

```bash
# Mode A: Single workload, single run
aorta-report pipeline hwqueue \
    --input results/hetero_kernels_single.json \
    --output ./hwqueue_report/

# Mode B: Single workload, sweep
aorta-report pipeline hwqueue \
    --input results/hetero_kernels_sweep.json \
    --output ./hwqueue_report/

# Mode C: Multi-workload comparison
aorta-report pipeline hwqueue \
    --baseline-dir ./results_baseline/ \
    --test-dir ./results_test/ \
    --baseline-label "ROCm 6.0" \
    --test-label "ROCm 6.1" \
    --output ./comparison_report/

# Mode C with custom threshold
aorta-report pipeline hwqueue \
    --baseline-dir ./results_baseline/ \
    --test-dir ./results_test/ \
    --threshold 0.10 \
    --output ./comparison_report/
```

---

## 5. Pipeline Steps

### 5.1 Mode A: Single Run Analysis

| Step | Action | Output |
|------|--------|--------|
| 1 | Load & validate JSON | Parsed data |
| 2 | Generate summary Excel | `single_run_analysis.xlsx` |
| 3 | Generate plots | `plots/latency_distribution.png`, `plots/per_stream_breakdown.png` |
| 4 | Generate HTML | `hwqueue_report.html` |

### 5.2 Mode B: Sweep Analysis

| Step | Action | Output |
|------|--------|--------|
| 1 | Load & validate sweep JSON | Parsed data with multiple stream counts |
| 2 | Generate summary Excel | `sweep_analysis.xlsx` (multiple sheets) |
| 3 | Generate plots | Scaling curves, efficiency plots, latency heatmaps |
| 4 | Generate HTML | `hwqueue_report.html` |

### 5.3 Mode C: Multi-Workload Comparison

| Step | Action | Output |
|------|--------|--------|
| 1 | Scan baseline & test directories | List of JSON files in each |
| 2 | Identify common workloads | Intersection of workload names |
| 3 | Report missing workloads | Log/print workloads missing from either side |
| 4 | Load common workload results | Dict[workload_name, sweep_data] for each |
| 5 | Compute comparisons | Per-workload, per-stream-count deltas |
| 6 | Generate consolidated Excel | `all_workloads_comparison.xlsx` |
| 7 | Generate comparison plots | Aggregate plots only |
| 8 | Generate HTML report | `hwqueue_comparison_report.html` |

**Missing Workload Handling:**
- Print warning: `"WARNING: Workload 'xyz' found in baseline but not in test - skipping"`
- Print warning: `"WARNING: Workload 'abc' found in test but not in baseline - skipping"`
- Include "Missing Workloads" section in HTML report

---

## 6. Output File Structure

### 6.1 Mode A/B: Single Workload
```
hwqueue_report/
├── hwqueue_analysis.xlsx
├── plots/
│   ├── latency_distribution.png
│   ├── latency_percentiles.png
│   ├── per_stream_breakdown.png
│   ├── throughput_scaling.png        # (sweep only)
│   ├── scaling_efficiency.png        # (sweep only)
│   └── switch_overhead.png
└── hwqueue_report.html
```

### 6.2 Mode C: Multi-Workload Comparison
```
comparison_report/
├── all_workloads_comparison.xlsx     # Consolidated report
├── plots/
│   ├── throughput_comparison_all.png         # All workloads, grouped bars
│   ├── regression_summary.png                # Heatmap of regressions
│   ├── latency_comparison_all.png            # Latency changes
│   └── scaling_comparison_summary.png        # Aggregate scaling view
└── hwqueue_comparison_report.html
```

**Note:** Per-workload individual Excel files are NOT generated (see Design Decisions).

---

## 7. Excel Report Structure

### 7.1 Single Workload - Single Run (`hwqueue_analysis.xlsx`)

**Sheet: Summary**
| Metric | Value |
|--------|-------|
| Workload | hetero_kernels |
| Stream Count | 8 |
| Throughput | 1234.56 ops/sec |
| Mean Latency | 1.2 ms |
| P50 Latency | 1.1 ms |
| P95 Latency | 1.5 ms |
| P99 Latency | 2.0 ms |
| Switch Overhead | 0.04 ms |
| Peak Memory | 2.5 GB |

### 7.2 Single Workload - Sweep (`sweep_analysis.xlsx`)

**Sheet: Summary**
| Metric | Value |
|--------|-------|
| Workload | hetero_kernels |
| Peak Throughput | 1500.0 ops/sec |
| Peak Stream Count | 8 |
| Inflection Point | 16 |

**Sheet: Scaling_Data**
| Streams | Throughput | P50 (ms) | P95 (ms) | P99 (ms) | Efficiency |
|---------|------------|----------|----------|----------|------------|
| 1 | 200.0 | 5.0 | 5.5 | 6.0 | 1.00 |
| 2 | 380.0 | 2.6 | 2.9 | 3.1 | 0.95 |
| 4 | 720.0 | 1.4 | 1.6 | 1.8 | 0.90 |
| 8 | 1200.0 | 0.8 | 1.0 | 1.2 | 0.75 |
| ... | ... | ... | ... | ... | ... |

### 7.3 Multi-Workload Comparison (`all_workloads_comparison.xlsx`)

**Sheet: Summary**
| Workload | Category | Best_Streams | Baseline_Throughput | Test_Throughput | Change_% | Status |
|----------|----------|--------------|---------------------|-----------------|----------|--------|
| hetero_kernels | latency_sensitive | 8 | 1200.0 | 1250.0 | +4.2% | ✓ OK |
| tiny_kernel_stress | latency_sensitive | 8 | 5000.0 | 4800.0 | -4.0% | ✓ OK |
| moe | distributed | 16 | 800.0 | 750.0 | -6.3% | ⚠ REGRESSION |
| ... | ... | ... | ... | ... | ... | ... |

**Sheet: Throughput_by_StreamCount**
| Workload | 1_Base | 1_Test | 1_Δ% | 2_Base | 2_Test | 2_Δ% | 4_Base | 4_Test | 4_Δ% | ... |
|----------|--------|--------|------|--------|--------|------|--------|--------|------|-----|
| hetero_kernels | 200 | 210 | +5.0 | 380 | 400 | +5.3 | 720 | 750 | +4.2 | ... |
| tiny_kernel_stress | 1000 | 980 | -2.0 | 1900 | 1850 | -2.6 | 3600 | 3500 | -2.8 | ... |
| moe | 100 | 98 | -2.0 | 190 | 185 | -2.6 | 360 | 340 | -5.6 | ... |

**Sheet: Latency_P99_by_StreamCount**
| Workload | 1_Base | 1_Test | 1_Δ% | 2_Base | 2_Test | 2_Δ% | ... |
|----------|--------|--------|------|--------|--------|------|-----|
| hetero_kernels | 6.0 | 5.8 | -3.3 | 3.1 | 3.0 | -3.2 | ... |
| tiny_kernel_stress | 1.2 | 1.3 | +8.3 | 0.7 | 0.75 | +7.1 | ... |

**Sheet: Regressions**
| Workload | Stream_Count | Metric | Baseline | Test | Change_% |
|----------|--------------|--------|----------|------|----------|
| moe | 16 | throughput | 800.0 | 750.0 | -6.3 |
| moe | 32 | throughput | 750.0 | 680.0 | -9.3 |
| tiny_kernel_stress | 8 | latency_p99 | 0.5 | 0.58 | +16.0 |

**Sheet: Improvements**
| Workload | Stream_Count | Metric | Baseline | Test | Change_% |
|----------|--------------|--------|----------|------|----------|
| hetero_kernels | 4 | throughput | 720.0 | 800.0 | +11.1 |
| hetero_kernels | 8 | latency_p99 | 1.2 | 1.0 | -16.7 |

**Sheet: Missing_Workloads**
| Workload | Present_In | Missing_From |
|----------|------------|--------------|
| new_workload | test | baseline |
| deprecated_workload | baseline | test |

**Sheet: Environment_Comparison**
| Property | Baseline | Test |
|----------|----------|------|
| Hostname | node1 | node2 |
| HIP Version | 6.0.0 | 6.1.0 |
| Driver | amdgpu | amdgpu |
| GPU Count | 8 | 8 |
| GPU Model | MI300X | MI300X |

---

## 8. Plot Types

### 8.1 Single Run Plots
1. **Latency Distribution Histogram** - `iteration_times_ms` distribution
2. **Latency Percentiles Bar** - mean/p50/p95/p99 comparison
3. **Per-Stream Time Breakdown** - Stacked/grouped bar of `per_stream_times_ms`
4. **Switch Overhead Gauge** - Visual indicator

### 8.2 Sweep Plots
1. **Throughput Scaling Curve** - X: stream count, Y: throughput (with ideal line)
2. **Scaling Efficiency Curve** - Shows diminishing returns
3. **Latency vs Streams** - Multi-line plot (p50, p95, p99)
4. **Latency Variance Heatmap** - P99/P50 ratio across streams
5. **Inflection Point Annotation** - Marked on throughput curve

### 8.3 Comparison Plots (Mode C)
1. **Throughput Comparison (All Workloads)** - Grouped bar chart
2. **Regression/Improvement Heatmap** - Workload × StreamCount matrix, color-coded
3. **Delta Summary Chart** - % change per workload (sorted by change)
4. **Latency Delta Chart** - P99 changes across workloads

**Styling:** All plots will match existing `aorta-report` styling conventions.

---

## 9. Implementation Structure

### 9.1 New Files to Create

```
aorta/src/aorta/report/
├── pipelines/
│   ├── cli.py                      # UPDATE: Add @pipeline.command("hwqueue")
│   ├── hwqueue_pipeline.py         # NEW: Pipeline orchestrator
│   └── __init__.py                 # UPDATE: Export new pipeline
├── processing/
│   └── hwqueue_loader.py           # NEW: JSON loader with validation
├── generators/
│   ├── hwqueue_excel.py            # NEW: Excel report generation
│   ├── hwqueue_plots.py            # NEW: Plot generation
│   └── cli.py                      # UPDATE: Add generate hwqueue subcommand (optional)
└── templates/
    └── hwqueue_report.html         # NEW: HTML template (or extend existing)
```

### 9.2 Module Dependencies

```
pipelines/hwqueue_pipeline.py
    ├── processing/hwqueue_loader.py
    │   └── (JSON validation & loading)
    ├── generators/hwqueue_excel.py
    │   └── comparison/save_with_formatting (reuse)
    ├── generators/hwqueue_plots.py
    │   └── matplotlib/seaborn (match aorta-report style)
    └── generators/html_generator.py (extend for mode="hwqueue")
```

### 9.3 Key Classes/Functions

```python
# processing/hwqueue_loader.py
@dataclass
class SingleRunData:
    """Parsed single run result."""
    workload_name: str
    stream_count: int
    throughput: float
    throughput_unit: str
    latency_ms: Dict[str, float]
    switch_latency: Optional[Dict[str, float]]
    memory: Optional[Dict[str, float]]
    metadata: Dict[str, Any]

@dataclass  
class SweepData:
    """Parsed sweep result."""
    workload_name: str
    results: List[SingleRunData]
    environment: Dict[str, Any]
    analysis: Dict[str, Any]

class HWQueueLoader:
    @staticmethod
    def load_single_run(path: Path) -> SingleRunData
    
    @staticmethod
    def load_sweep(path: Path) -> SweepData
    
    @staticmethod
    def load_directory(path: Path) -> Dict[str, SweepData]
    
    @staticmethod
    def find_common_workloads(
        baseline: Dict[str, SweepData], 
        test: Dict[str, SweepData]
    ) -> Tuple[List[str], List[str], List[str]]
    # Returns: (common, baseline_only, test_only)


# pipelines/hwqueue_pipeline.py
@dataclass
class HWQueuePipelineConfig:
    input_path: Optional[Path] = None       # For Mode A/B
    baseline_dir: Optional[Path] = None     # For Mode C
    test_dir: Optional[Path] = None         # For Mode C
    output_dir: Path = None
    baseline_label: Optional[str] = None
    test_label: Optional[str] = None
    threshold: float = 0.05
    excel: bool = True
    plots: bool = True
    html: bool = True
    verbose: bool = False

@dataclass
class HWQueuePipelineResult:
    success: bool
    mode: str  # "single_run", "sweep", "comparison"
    output_dir: Path
    files_generated: Dict[str, Path]
    common_workloads: List[str] = field(default_factory=list)
    missing_baseline: List[str] = field(default_factory=list)
    missing_test: List[str] = field(default_factory=list)
    regressions: List[Dict] = field(default_factory=list)
    improvements: List[Dict] = field(default_factory=list)
    steps_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

def run_hwqueue_pipeline(config: HWQueuePipelineConfig) -> HWQueuePipelineResult


# generators/hwqueue_excel.py
def generate_single_run_excel(data: SingleRunData, output: Path) -> Path

def generate_sweep_excel(data: SweepData, output: Path) -> Path

def generate_comparison_excel(
    baseline: Dict[str, SweepData],
    test: Dict[str, SweepData],
    common_workloads: List[str],
    missing_baseline: List[str],
    missing_test: List[str],
    baseline_label: str,
    test_label: str,
    threshold: float,
    output: Path
) -> Tuple[Path, List[Dict], List[Dict]]
# Returns: (excel_path, regressions, improvements)


# generators/hwqueue_plots.py
def generate_single_run_plots(data: SingleRunData, output_dir: Path) -> List[Path]

def generate_sweep_plots(data: SweepData, output_dir: Path) -> List[Path]

def generate_comparison_plots(
    baseline: Dict[str, SweepData],
    test: Dict[str, SweepData],
    common_workloads: List[str],
    labels: Tuple[str, str],
    output_dir: Path
) -> List[Path]
```

---

## 10. Implementation Tasks

| # | Task | File(s) | Priority | Complexity | Notes |
|---|------|---------|----------|------------|-------|
| 1 | Create JSON loader with validation | `processing/hwqueue_loader.py` | P0 | Low | |
| 2 | Create single run Excel generator | `generators/hwqueue_excel.py` | P0 | Low | |
| 3 | Create sweep Excel generator | `generators/hwqueue_excel.py` | P0 | Medium | |
| 4 | Create comparison Excel generator | `generators/hwqueue_excel.py` | P0 | Medium | Key deliverable |
| 5 | Create pipeline orchestrator | `pipelines/hwqueue_pipeline.py` | P0 | Medium | |
| 6 | Add CLI command | `pipelines/cli.py` | P0 | Low | |
| 7 | Create single run plots | `generators/hwqueue_plots.py` | P1 | Medium | Match aorta-report style |
| 8 | Create sweep plots | `generators/hwqueue_plots.py` | P1 | Medium | Match aorta-report style |
| 9 | Create comparison plots | `generators/hwqueue_plots.py` | P1 | Medium | Match aorta-report style |
| 10 | Create/extend HTML template | `templates/` + `html_generator.py` | P1 | Low | Static HTML |
| 11 | Update `__init__.py` exports | Multiple | P0 | Low | |
| 12 | Add tests | `tests/` | P2 | Medium | |

---

## 11. Comparison Logic

### 11.1 Metric Comparison Rules

| Metric | Regression If | Improvement If |
|--------|---------------|----------------|
| Throughput | `(test - baseline) / baseline < -threshold` | `> +threshold` |
| Latency (p50/p95/p99) | `(test - baseline) / baseline > +threshold` | `< -threshold` |
| Switch Overhead | `(test - baseline) / baseline > +threshold` | `< -threshold` |

**Note:** A single threshold value is used for all metrics. See Future Enhancements for per-metric thresholds.

### 11.2 Status Labels

| Status | Condition |
|--------|-----------|
| `✓ OK` | Change within ±threshold |
| `⚠ REGRESSION` | Throughput decreased OR latency increased beyond threshold |
| `✓ IMPROVED` | Throughput increased OR latency decreased beyond threshold |

### 11.3 Workload Matching

For Mode C (multi-workload comparison):
1. Scan both directories for `*_results.json` files
2. Extract workload name from filename (e.g., `hetero_kernels_results.json` → `hetero_kernels`)
3. Find intersection (common workloads)
4. Track workloads present in only one directory
5. Only compare common workloads
6. Report missing workloads in output

---

## 12. Reuse from Existing Code

| Component | Source | Usage |
|-----------|--------|-------|
| `save_with_formatting` | `comparison/__init__.py` | Excel styling with formatting |
| `compare_results` | `hw_queue_eval/core/metrics.py` | Single result comparison logic |
| `generate_html` | `generators/html_generator.py` | Extend for `mode="hwqueue"` |
| Pipeline pattern | `pipelines/summary_pipeline.py` | Config/Result dataclasses, step structure |
| Plot styling | `generators/plot_helper/` | Match existing aorta-report plot styles |

---

## 13. Design Decisions

This section documents the decisions made for the initial implementation.

### DD-1: Stream Count Filtering
**Decision:** NOT implemented in initial version  
**Rationale:** Nice-to-have feature, not essential for MVP  
**Future:** Can be added as `--stream-counts` option to filter specific stream counts

### DD-2: Per-Workload Detail Files
**Decision:** NOT generated in Mode C  
**Rationale:** Consolidated Excel is sufficient; per-workload files add complexity and clutter  
**Implementation:** Only `all_workloads_comparison.xlsx` is generated

### DD-3: Plot Styling
**Decision:** Match existing `aorta-report` plot styling  
**Rationale:** Consistency across the tool  
**Implementation:** Reuse plot helpers and color schemes from `generators/plot_helper/`

### DD-4: HTML Interactivity
**Decision:** Static HTML for initial version  
**Rationale:** Simpler implementation, easier to share  
**Future:** Sortable tables can be added as enhancement (see Future Enhancements)

### DD-5: Regression Threshold
**Decision:** Single threshold for all metrics  
**Rationale:** Simpler UX; most users expect uniform threshold  
**Implementation:** `--threshold 0.05` applies to throughput and latency equally  
**Future:** Per-metric thresholds and configurable "regression decision metric" can be added

### DD-6: Missing Workload Handling
**Decision:** Skip missing workloads, report clearly  
**Rationale:** Comparing apples to oranges doesn't make sense  
**Implementation:**
- Only common workloads are compared
- Missing workloads logged to console with WARNING
- Missing workloads listed in Excel sheet and HTML report

---

## 14. Future Enhancements

These features are NOT in scope for the initial implementation but are documented for future reference.

### FE-1: Stream Count Filtering (Nice to Have)
```bash
aorta-report pipeline hwqueue \
    --baseline-dir ... \
    --test-dir ... \
    --stream-counts "4,8,16"  # Only compare these stream counts
```

### FE-2: Sortable HTML Tables (Nice to Have)
Add JavaScript-based sortable tables to HTML reports for better interactivity.

### FE-3: Per-Metric Thresholds
```bash
aorta-report pipeline hwqueue \
    --threshold-throughput 0.05 \
    --threshold-latency 0.10 \
    --threshold-switch 0.15
```

### FE-4: Regression Decision Metric
Allow specifying which metric determines overall regression status:
```bash
aorta-report pipeline hwqueue \
    --regression-metric throughput  # or latency_p99, switch_overhead
```

### FE-5: Per-Workload Detail Files (If Needed)
```bash
aorta-report pipeline hwqueue \
    --baseline-dir ... \
    --test-dir ... \
    --per-workload-files  # Generate individual comparison files
```

### FE-6: JSON Output
```bash
aorta-report pipeline hwqueue \
    --baseline-dir ... \
    --test-dir ... \
    --json  # Output comparison results as JSON for programmatic use
```

---

## Appendix: Example CLI Output (Mode C)

```
$ aorta-report pipeline hwqueue \
    --baseline-dir ./results_baseline/ \
    --test-dir ./results_test/ \
    --baseline-label "ROCm 6.0" \
    --test-label "ROCm 6.1" \
    --output ./comparison_report/

============================================================
HWQUEUE COMPARISON PIPELINE
============================================================
Baseline: ./results_baseline/ (ROCm 6.0)
Test: ./results_test/ (ROCm 6.1)
Output: ./comparison_report/
Threshold: 5.0%

Scanning directories...
  Baseline: 12 workload files found
  Test: 11 workload files found

WARNING: Workload 'deprecated_workload' found in baseline but not in test - skipping
Comparing 11 common workloads...

Processing workloads:
  ✓ hetero_kernels
  ✓ tiny_kernel_stress  
  ✓ large_gemm_only
  ✓ moe
  ... (7 more)

============================================================
PIPELINE COMPLETE!
============================================================

Steps completed:
  ✓ load_baseline_results
  ✓ load_test_results
  ✓ compute_comparisons
  ✓ generate_excel
  ✓ generate_plots
  ✓ generate_html

Workload Summary:
  Common workloads: 11
  Missing from test: 1 (deprecated_workload)
  Missing from baseline: 0

Results:
  Regressions: 3
  Improvements: 5
  Unchanged: 3

Output directory: ./comparison_report/
Generated files:
  - all_workloads_comparison.xlsx
  - plots/ (4 plots)
  - hwqueue_comparison_report.html
```

---

*Document Version: 1.1*  
*Status: Awaiting Review*
