# `analyze` Command Group - Developer Documentation

**Version:** 1.0  
**Date:** January 2026  
**Status:** ✅ Implemented

---

## Table of Contents

1. [Overview](#1-overview)
2. [Command Summary](#2-command-summary)
3. [Source Script Analysis](#3-source-script-analysis)
4. [Implementation Architecture](#4-implementation-architecture)
5. [Command Specifications](#5-command-specifications)
6. [TraceLens Integration](#6-tracelens-integration)
7. [Implementation Order](#7-implementation-order)
8. [Expected Output](#8-expected-output)

---

## 1. Overview

The `analyze` command group provides TraceLens analysis capabilities for PyTorch profiler traces. It consolidates three shell/Python scripts into a unified CLI interface.

### Commands

| Command | Purpose | Source Script |
|---------|---------|---------------|
| `analyze single` | Analyze single configuration traces | `run_tracelens_single_config.sh` |
| `analyze sweep` | Analyze sweep with multiple configs | `run_tracelens_analysis.sh` |
| `analyze gemm` | Extract GEMM kernel variance | `analyze_gemm_reports.py` |

### Key Features

- **Unified interface**: Consistent CLI for all analysis operations
- **GEMM recognition**: Patched TraceLens for ROCm Tensile kernel detection
- **Auto-discovery**: Automatic detection of ranks, threads, and channels
- **Flexible output**: Configurable output directories and formats

---

## 2. Command Summary

### 2.1 `analyze single`

Analyze a single configuration trace directory containing rank subdirectories.

```bash
aorta-report analyze single /path/to/traces [OPTIONS]

Options:
  --individual-only               Generate only individual reports
  --collective-only               Generate only collective report
  --geo-mean                      Use geometric mean for timeline aggregation
  --short-kernel-threshold INT    Threshold for short kernel study (µs)
  --topk-ops INT                  Number of top operations to include
  -o, --output PATH               Output directory
```

**Usage Examples:**
```bash
# Basic analysis (generates individual + collective reports)
aorta-report analyze single /path/to/traces

# Generate only individual reports with geometric mean aggregation
aorta-report analyze single /path/to/traces --individual-only --geo-mean

# Custom output directory
aorta-report analyze single /path/to/traces -o ./results
```

### 2.2 `analyze sweep`

Analyze a sweep directory containing multiple thread/channel configurations.

```bash
aorta-report analyze sweep /path/to/sweep [OPTIONS]

Options:
  --geo-mean          Use geometric mean instead of arithmetic mean
  -o, --output PATH   Output directory
```

**Usage Examples:**
```bash
# Basic sweep analysis
aorta-report analyze sweep /path/to/sweep_20251124

# Use geometric mean for aggregation
aorta-report analyze sweep /path/to/sweep --geo-mean

# Custom output directory
aorta-report analyze sweep /path/to/sweep -o ./analysis_results
```

### 2.3 `analyze gemm`

Extract GEMM kernel variance data from existing TraceLens reports.

```bash
aorta-report analyze gemm /path/to/reports [OPTIONS]

Options:
  -t, --threads INT   Thread configurations to analyze (multiple allowed)
  -c, --channels INT  Channel configurations to analyze (multiple allowed)
  -r, --ranks INT     Ranks to analyze (default: 0-7)
  --top-k INTEGER     Number of top kernels to extract per file (default: 5)
  -o, --output PATH   Output CSV file
```

**Usage Examples:**
```bash
# Basic GEMM analysis with defaults (256/512 threads, 28/42/56/70 channels)
aorta-report analyze gemm /path/to/tracelens_analysis

# Custom thread and channel configurations
aorta-report analyze gemm /path/to/reports -t 256 -t 512 -c 28 -c 42

# Extract top 10 kernels and save to custom file
aorta-report analyze gemm /path/to/reports --top-k 10 -o gemm_analysis.csv

# Specify specific ranks
aorta-report analyze gemm /path/to/reports -r 0 -r 1 -r 2 -r 3
```

---

## 3. Source Script Analysis

### 3.1 `run_tracelens_single_config.sh` (267 lines)

**Location:** `scripts/tracelens_single_config/run_tracelens_single_config.sh`

**Functionality:**
1. Parse options (`--individual-only`, `--collective-only`)
2. Auto-detect trace directory structure:
   - Check if input contains `rank*` directories (is torch_profiler/)
   - Check if input contains `torch_profiler/` subdirectory
3. Create output directory structure
4. Detect number of ranks
5. Generate individual reports (per rank)
6. Generate collective multi-rank report

**TraceLens Commands:**
```bash
# Individual report (per rank)
$TRACELENS_WRAPPER generate_perf_report \
    --profile_json_path "$TRACE" \
    --output_xlsx_path "$OUTPUT" \
    --include_unlinked_kernels \
    --short_kernel_study \
    --short_kernel_threshold_us 50 \
    --topk_ops 100 \
    --topk_roofline_ops 100

# Collective report (all ranks)
$TRACELENS_WRAPPER generate_multi_rank_collective \
    --trace_pattern "$TORCH_PROF_DIR/rank*/trace.json" \
    --world_size $NUM_RANKS \
    --output_xlsx_path "$OUTPUT" \
    --detailed_analysis \
    --use_multiprocessing
```

**Input Structure:**
```
trace_dir/
├── torch_profiler/          # or trace_dir IS torch_profiler/
│   ├── rank0/
│   │   └── *.json
│   ├── rank1/
│   │   └── *.json
│   └── ...
```

**Output Structure:**
```
trace_dir/
└── tracelens_analysis/
    ├── individual_reports/
    │   ├── perf_rank0.xlsx
    │   ├── perf_rank1.xlsx
    │   └── ...
    └── collective_reports/
        └── collective_all_ranks.xlsx
```

---

### 3.2 `run_tracelens_analysis.sh` (423 lines)

**Location:** `scripts/gemm_analysis/run_tracelens_analysis.sh`

**Functionality:**
1. Parse options (`--rocprof`)
2. Auto-discover thread configurations (e.g., `256thread`, `512thread`)
3. Auto-discover channel configurations per thread (e.g., `nccl_28channels`)
4. For each thread/channel/rank combination:
   - Find trace files
   - Generate individual reports
5. Generate collective reports (PyTorch mode only)
6. Generate cross-thread comparisons

**TraceLens Commands:**
```bash
# PyTorch mode - Individual
TraceLens_generate_perf_report_pytorch \
    --profile_json_path "$TRACE" \
    --output_xlsx_path "$OUTPUT" \
    --include_unlinked_kernels \
    --short_kernel_study \
    --short_kernel_threshold_us 50 \
    --topk_ops 100 \
    --enable_kernel_summary \
    --topk_roofline_ops 100

# ROCprof mode - Individual
TraceLens_generate_perf_report_rocprof \
    --profile_json_path "$TRACE" \
    --output_xlsx_path "$OUTPUT" \
    --kernel_details \
    --short_kernel_study \
    --short_kernel_threshold_us 50 \
    --topk_kernels 100

# PyTorch mode - Collective
TraceLens_generate_multi_rank_collective_report_pytorch \
    --trace_pattern "$TRACE_DIR/rank*/trace/pt.trace.json" \
    --world_size 8 \
    --output_xlsx_path "$OUTPUT" \
    --detailed_analysis \
    --use_multiprocessing

# Comparison across threads
TraceLens_compare_perf_reports_pytorch \
    "${reports[@]}" \
    --names "${names[@]}" \
    --sheets gpu_timeline ops_summary \
    -o "$OUTPUT"
```

**Input Structure:**
```
sweep_dir/
├── 256thread/
│   ├── nccl_28channels/
│   │   └── torch_profiler/
│   │       ├── rank0/
│   │       └── ...
│   ├── nccl_42channels/
│   └── ...
└── 512thread/
    └── ...
```

**Output Structure:**
```
sweep_dir/
└── tracelens_analysis/
    ├── 256thread/
    │   ├── individual_reports/
    │   │   ├── perf_28ch_rank0.xlsx
    │   │   ├── perf_28ch_rank1.xlsx
    │   │   └── ...
    │   └── collective_reports/
    │       └── collective_28ch.xlsx
    ├── 512thread/
    │   └── ...
    └── comparisons/
        ├── compare_28ch_rank0_across_threads.xlsx
        └── ...
```

---

### 3.3 `analyze_gemm_reports.py` (344 lines)

**Location:** `scripts/gemm_analysis/analyze_gemm_reports.py`

**Functionality:**
1. Parse command-line arguments
2. Iterate through thread/channel/rank combinations
3. Open each Excel report
4. Read GEMM sheet
5. Extract kernel info and timing data
6. Calculate time variance (max - min)
7. Sort by variance and get top-K
8. Output combined CSV

**Key Functions:**
```python
def process_excel_file(file_path, threads, channel, rank, top_k=5):
    """Process a single Excel file and extract GEMM data."""
    # Opens workbook
    # Reads GEMM sheet
    # Validates column headers
    # Extracts kernel_details, time_min, time_max
    # Calculates time_diff
    # Returns top_k results sorted by variance
```

**Input:** TraceLens Excel reports with GEMM sheet
**Output:** CSV with columns:
- `threads`, `channel`, `rank`
- `kernel_name`
- `kernel_time_min_us`, `kernel_time_max_us`, `time_diff_us`

---

## 4. Implementation Architecture

### 4.1 File Structure

```
src/aorta/report/
├── cli.py                           # CLI definitions (update analyze commands)
├── analysis/                        # NEW: Analysis logic
│   ├── __init__.py                  # Exports public functions
│   ├── tracelens_wrapper.py         # GEMM-patched TraceLens wrapper
│   ├── analyze_single.py            # Single config analysis
│   ├── analyze_sweep.py             # Sweep analysis
│   └── analyze_gemm.py              # GEMM variance analysis
├── generators/                      # HTML generators (existing)
└── templates/                       # HTML templates (existing)
```

### 4.2 Module Responsibilities

#### `analysis/__init__.py`
```python
from .analyze_single import analyze_single_config
from .analyze_sweep import analyze_sweep_config
from .analyze_gemm import analyze_gemm_reports
from .tracelens_wrapper import TraceLensWrapper

__all__ = [
    "analyze_single_config",
    "analyze_sweep_config", 
    "analyze_gemm_reports",
    "TraceLensWrapper",
]
```

#### `analysis/tracelens_wrapper.py`
```python
class TraceLensWrapper:
    """GEMM-patched TraceLens wrapper."""
    
    def __init__(self):
        self._apply_gemm_patches()
    
    def _apply_gemm_patches(self):
        """Apply GEMM recognition patches to TraceLens."""
        # Port from tracelens_with_gemm_patch.py
    
    def generate_perf_report(self, trace_path, output_path, **options):
        """Generate individual performance report."""
    
    def generate_collective_report(self, trace_pattern, world_size, output_path, **options):
        """Generate multi-rank collective report."""
    
    def compare_reports(self, report_paths, names, output_path, sheets=None):
        """Compare multiple performance reports."""
```

#### `analysis/analyze_single.py`
```python
def analyze_single_config(
    trace_dir: Path,
    output_dir: Optional[Path] = None,
    individual_only: bool = False,
    collective_only: bool = False,
    verbose: bool = False,
) -> Path:
    """Analyze a single configuration trace directory."""

def detect_trace_structure(input_dir: Path) -> Tuple[Path, Path]:
    """Auto-detect torch_profiler directory and base directory."""

def discover_ranks(torch_prof_dir: Path) -> List[int]:
    """Discover available ranks in the trace directory."""

def generate_individual_reports(
    wrapper: TraceLensWrapper,
    torch_prof_dir: Path,
    output_dir: Path,
    ranks: List[int],
    verbose: bool,
) -> List[Path]:
    """Generate individual performance reports for each rank."""

def generate_collective_report(
    wrapper: TraceLensWrapper,
    torch_prof_dir: Path,
    output_dir: Path,
    num_ranks: int,
    verbose: bool,
) -> Optional[Path]:
    """Generate multi-rank collective report."""
```

#### `analysis/analyze_sweep.py`
```python
def analyze_sweep_config(
    sweep_dir: Path,
    output_dir: Optional[Path] = None,
    use_geo_mean: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    """Process GPU timeline data from all individual reports in a sweep."""

def process_thread_config(
    thread_config: str,
    tracelens_dir: Path,
    use_geo_mean: bool,
    verbose: bool = False,
) -> List[pd.DataFrame]:
    """Process a single thread configuration."""

def process_channel_config(
    channel_config: str,
    channel_groups: Dict[str, List[tuple]],
    use_geo_mean: bool,
    thread_config: str,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    """Process a single channel configuration."""

def aggregate_rank_data(
    rank_data: List[pd.DataFrame],
    thread_config: str,
    channel_config: str,
    num_ranks: int,
    use_geo_mean: bool,
) -> pd.DataFrame:
    """Aggregate data across ranks and add metadata."""
```

#### `analysis/analyze_gemm.py`
```python
def analyze_gemm_reports(
    reports_dir: Path,
    output_file: Optional[Path] = None,
    top_k: int = 5,
    threads: Optional[List[int]] = None,
    channels: Optional[List[int]] = None,
    ranks: Optional[List[int]] = None,
    verbose: bool = False,
) -> Path:
    """Analyze GEMM reports and extract top kernels by variance."""

def process_excel_file(
    file_path: Path,
    threads: int,
    channel: int,
    rank: int,
    top_k: int,
) -> List[Dict]:
    """Process a single Excel file and extract GEMM data."""

def extract_kernel_name(kernel_info_str: str) -> Optional[str]:
    """Extract kernel name from kernel info string."""
```

### 4.3 Data Flow

```
CLI (cli.py)
    │
    ├── analyze single ───────────►  analysis.analyze_single_config()
    │                                        │
    │                                        ├── detect_trace_structure()
    │                                        ├── discover_ranks()
    │                                        ├── TraceLensWrapper.generate_perf_report()
    │                                        └── TraceLensWrapper.generate_collective_report()
    │
    ├── analyze sweep ────────────►  analysis.analyze_sweep_config()
    │                                        │
    │                                        ├── discover_configurations()
    │                                        ├── process_configuration()
    │                                        │       └── TraceLensWrapper.generate_perf_report()
    │                                        ├── TraceLensWrapper.generate_collective_report()
    │                                        └── TraceLensWrapper.compare_reports()
    │
    └── analyze gemm ─────────────►  analysis.analyze_gemm_reports()
                                             │
                                             ├── process_excel_file()
                                             └── write CSV output
```

---

## 5. Command Specifications

### 5.1 `analyze single`

| Aspect | Details |
|--------|---------|
| **Input** | Directory with torch_profiler/rank* structure |
| **Output** | individual_reports/ and collective_reports/ |
| **Options** | `--individual-only`, `--collective-only`, `-o` |
| **TraceLens** | `generate_perf_report`, `generate_multi_rank_collective` |

### 5.2 `analyze sweep`

| Aspect | Details |
|--------|---------|
| **Input** | Sweep directory with thread/channel structure |
| **Output** | Per-config reports + comparisons |
| **Options** | `--rocprof`, `-o` |
| **TraceLens** | `generate_perf_report`, `generate_collective`, `compare_reports` |

### 5.3 `analyze gemm`

| Aspect | Details |
|--------|---------|
| **Input** | Directory with TraceLens Excel reports |
| **Output** | CSV with GEMM kernel variance |
| **Options** | `--top-k`, `-o` |
| **Dependencies** | `openpyxl` for Excel reading |

---

## 6. TraceLens Integration

### 6.1 GEMM Patch Requirements

The TraceLens wrapper must apply these patches for ROCm GEMM recognition:

1. **`kernel_name_parser`**: Recognize Tensile GEMM patterns (`Cijk_Alik_Bljk_...`)
2. **`Trace2Tree.util`**: Enhanced `is_gemm_kernel()` function
3. **`TraceEventUtils`**: Add GEMM keys for classification
4. **`torch_op_mapping`**: Better GEMM categorization

### 6.2 TraceLens Functions Used

| Function | PyTorch Mode | ROCprof Mode |
|----------|-------------|--------------|
| `generate_perf_report_pytorch` | ✓ | - |
| `generate_perf_report_rocprof` | - | ✓ |
| `generate_multi_rank_collective_report_pytorch` | ✓ | - |
| `compare_perf_reports_pytorch` | ✓ | ✓ |

### 6.3 Common TraceLens Options

```python
# Individual report options
INDIVIDUAL_REPORT_OPTIONS = {
    "include_unlinked_kernels": True,
    "short_kernel_study": True,
    "short_kernel_threshold_us": 50,
    "topk_ops": 100,
    "topk_roofline_ops": 100,
}

# ROCprof specific options
ROCPROF_OPTIONS = {
    "kernel_details": True,
    "topk_kernels": 100,
}

# Collective report options
COLLECTIVE_REPORT_OPTIONS = {
    "detailed_analysis": True,
    "use_multiprocessing": True,
}
```

---

## 7. Implementation Status

### Phase 1: Foundation ✅

1. **Created `analysis/` directory structure** ✅
2. **Implemented `tracelens_wrapper.py`** ✅
   - GEMM patches for ROCm Tensile kernel recognition
   - Wrapper class with methods for TraceLens commands
   - Support for individual, collective, and rocprof reports

### Phase 2: `analyze gemm` ✅

3. **Implemented `analyze_gemm.py`** ✅
   - Ported logic from `analyze_gemm_reports.py`
   - Clean API with configurable threads/channels/ranks
   - Progress reporting and summary statistics

4. **Updated CLI for `analyze gemm`** ✅
   - Connected command to implementation
   - Added multiple options for configuration

### Phase 3: `analyze single` ✅

5. **Implemented `analyze_single.py`** ✅
   - Directory detection logic
   - Report generation with TraceLens wrapper
   - GPU timeline aggregation
   - Status reporting

6. **Updated CLI for `analyze single`** ✅
   - Added geo-mean and threshold options

### Phase 4: `analyze sweep` ✅

7. **Implemented `analyze_sweep.py`** ✅
   - Thread/channel config discovery
   - GPU timeline processing across all configs
   - Excel output with pivot tables

8. **Updated CLI for `analyze sweep`** ✅
   - Added geo-mean option

### Phase 5: Documentation ✅

9. **Updated documentation** ✅
   - This dev docs file
   - Implementation complete

---

## 8. Expected Output

### 8.1 `analyze single` Output

```
============================================================
TraceLens Analysis - Single Configuration
============================================================
Input directory: /path/to/traces
Torch profiler: /path/to/traces/torch_profiler
Detected 8 ranks

Step 1: Generating Individual Reports
  [1/8] Rank 0... ✓ perf_rank0.xlsx
  [2/8] Rank 1... ✓ perf_rank1.xlsx
  [3/8] Rank 2... ✓ perf_rank2.xlsx
  [4/8] Rank 3... ✓ perf_rank3.xlsx
  [5/8] Rank 4... ✓ perf_rank4.xlsx
  [6/8] Rank 5... ✓ perf_rank5.xlsx
  [7/8] Rank 6... ✓ perf_rank6.xlsx
  [8/8] Rank 7... ✓ perf_rank7.xlsx

Step 2: Generating Collective Report
  Processing all 8 ranks... ✓ collective_all_ranks.xlsx

============================================================
Analysis Complete!
============================================================
Output: /path/to/traces/tracelens_analysis/

Generated reports:
  Individual: 8
  Collective: 1
```

### 8.2 `analyze sweep` Output

```
============================================================
TraceLens Analysis - Sweep
============================================================
Sweep directory: /path/to/sweep
Mode: PyTorch profiler

Discovered configurations:
  256thread: 28, 42, 56, 70 channels
  512thread: 28, 42, 56, 70 channels
  Total: 8 configurations × 8 ranks = 64 reports

Step 1: Generating Individual Reports
  256thread/28ch:
    [1/8] Rank 0... ✓
    [2/8] Rank 1... ✓
    ...
  256thread/42ch:
    ...

Step 2: Generating Collective Reports
  256thread/28ch... ✓
  256thread/42ch... ✓
  ...

Step 3: Generating Comparisons
  28ch across threads... ✓
  42ch across threads... ✓
  ...

============================================================
Analysis Complete!
============================================================
Output: /path/to/sweep/tracelens_analysis/

Summary:
  Individual reports: 64
  Collective reports: 8
  Comparisons: 32
```

### 8.3 `analyze gemm` Output

```
============================================================
GEMM Kernel Variance Analysis
============================================================
Base path: /path/to/tracelens_analysis
Configuration:
  Threads: [256, 512]
  Channels: [28, 42, 56, 70]
  Ranks: [0, 1, 2, 3, 4, 5, 6, 7]
  Top K: 5

Processing Excel files...
  [1/64] perf_28ch_rank0.xlsx... 5 kernels found
  [2/64] perf_28ch_rank1.xlsx... 5 kernels found
  ...

============================================================
Analysis Complete!
============================================================
Output: /path/to/tracelens_analysis/top5_gemm_kernels_time_variance.csv

Summary:
  Total kernels extracted: 320
  Unique kernel names: 45
  Max variance: 1234.56 µs
  Avg variance: 89.12 µs

Top 5 kernels by variance:
  1. Cijk_Alik_Bljk_... (256t/28ch/r0): 1234.56 µs
  2. Cijk_Alik_Bljk_... (512t/42ch/r3): 987.65 µs
  ...
```

---

## Appendix A: Migration Checklist

### From `run_tracelens_single_config.sh`
- [x] Directory structure detection
- [x] Rank discovery
- [x] Individual report generation loop
- [x] Symlink creation for collective report
- [x] Collective report generation
- [x] Summary output
- [x] GPU timeline aggregation

### From `run_tracelens_analysis.sh` → `analyze sweep`
- [x] Thread config discovery
- [x] Channel config discovery
- [x] PyTorch trace file finding
- [x] GPU timeline processing per config
- [x] Summary Excel generation with pivot tables

### From `analyze_gemm_reports.py`
- [x] Command-line argument handling
- [x] Excel file processing
- [x] GEMM sheet reading
- [x] Kernel name extraction
- [x] Variance calculation
- [x] CSV output

---

## Appendix B: Error Handling

| Scenario | Handling |
|----------|----------|
| Missing trace file | Log warning, continue with next |
| Missing rank directory | Log warning, continue with next |
| GEMM sheet not found | Log warning, skip file |
| TraceLens import error | Raise with helpful message |
| Permission error | Raise with fix instructions |
| No configurations found | Raise with expected structure |

