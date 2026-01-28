# `process` Command Group - Developer Documentation

**Version:** 1.0  
**Date:** January 2026  
**Status:** ✅ Implemented

---

## Table of Contents

1. [Overview](#1-overview)
2. [Command Summary](#2-command-summary)
3. [process gpu-timeline - Detailed Analysis](#3-process-gpu-timeline---detailed-analysis)
4. [process comms - NCCL Communication Processing](#4-process-comms---nccl-communication-processing)
5. [process gemm-variance - Timestamp Enhancement](#5-process-gemm-variance---timestamp-enhancement)
6. [Implementation Architecture](#6-implementation-architecture)
7. [Implementation Order](#7-implementation-order)
8. [Expected Output Examples](#8-expected-output-examples)

---

## 1. Overview

The `process` command group provides data processing utilities that transform raw TraceLens data into more usable formats.

### Current Status

| Command | Status | Source Script |
|---------|--------|---------------|
| `process gpu-timeline` | ✅ Implemented | Two modes: single and sweep |
| `process comms` | ✅ Implemented | `gemm_analysis/process_comms.py` |
| `process gemm-variance` | ✅ Implemented | `gemm_analysis/enhance_gemm_variance_with_timestamps.py` |

### Key Features

- **GPU Timeline Processing**: Aggregate GPU timeline data across ranks (two modes)
- **NCCL Communication Processing**: Extract and combine communication metrics
- **GEMM Variance Enhancement**: Add temporal information to variance analysis

---

## 2. Command Summary

### 2.1 `process gpu-timeline`

Process GPU timeline data from TraceLens reports with two distinct modes.

```bash
aorta-report process gpu-timeline INPUT_DIR [OPTIONS]

Arguments:
  INPUT_DIR             Path to reports directory or sweep directory

Options:
  --mode [auto|single|sweep]   Processing mode (default: auto)
  --geo-mean                   Use geometric mean instead of arithmetic mean
  -o, --output PATH            Output file path
```

**Usage Examples:**
```bash
# Auto-detect mode
aorta-report process gpu-timeline /path/to/reports

# Explicit single config mode
aorta-report process gpu-timeline /path/to/individual_reports --mode single

# Sweep mode with geometric mean
aorta-report process gpu-timeline /path/to/sweep --mode sweep --geo-mean

# Custom output path
aorta-report process gpu-timeline /path/to/sweep -o ./results/timeline.xlsx
```

### 2.2 `process comms`

Process NCCL collective reports and generate master data files.

```bash
aorta-report process comms SWEEP_DIR [OPTIONS]

Arguments:
  SWEEP_DIR             Path to sweep directory

Options:
  -o, --output PATH     Output directory (default: tracelens_analysis/)
```

**Usage Examples:**
```bash
# Basic usage
aorta-report process comms /path/to/sweep

# Custom output directory
aorta-report process comms /path/to/sweep -o ./nccl_analysis/
```

### 2.3 `process gemm-variance`

Enhance GEMM variance CSV with kernel timestamp information.

```bash
aorta-report process gemm-variance INPUT_CSV [OPTIONS]

Arguments:
  INPUT_CSV             Input CSV file with GEMM variance data

Options:
  --base-path PATH      Base path to sweep directory (required)
  --tolerance FLOAT     Duration matching tolerance (default: 0.01 = 1%)
  -o, --output PATH     Output CSV file
```

**Usage Examples:**
```bash
# Basic usage
aorta-report process gemm-variance ./gemm_variance.csv --base-path /path/to/sweep

# Custom tolerance and output
aorta-report process gemm-variance ./gemm_variance.csv \
    --base-path /path/to/sweep \
    --tolerance 0.02 \
    -o ./enhanced_variance.csv
```

---

## 3. `process gpu-timeline` - Detailed Analysis

### 3.1 Two Distinct Modes

The command supports two processing modes for different directory structures:

| Aspect | Single Config Mode | Sweep Mode |
|--------|-------------------|------------|
| **Source Script** | `tracelens_single_config/process_gpu_timeline.py` | `gemm_analysis/process_gpu_timeline.py` |
| **Lines of Code** | 101 | 468 |
| **Input Argument** | `--reports-dir` | `--sweep-dir` |
| **Input Path** | `individual_reports/` directory | Sweep directory root |
| **File Pattern** | `perf_rank*.xlsx` | `perf_*ch_rank*.xlsx` |
| **Directory Structure** | Flat | Nested with thread/channel hierarchy |

### 3.2 Input Structures

**Single Config Mode:**
```
individual_reports/        # Direct input
├── perf_rank0.xlsx
├── perf_rank1.xlsx
├── perf_rank2.xlsx
├── perf_rank3.xlsx
├── perf_rank4.xlsx
├── perf_rank5.xlsx
├── perf_rank6.xlsx
└── perf_rank7.xlsx
```

**Sweep Mode:**
```
sweep_directory/
└── tracelens_analysis/
    ├── 256thread/
    │   └── individual_reports/
    │       ├── perf_28ch_rank0.xlsx
    │       ├── perf_28ch_rank1.xlsx
    │       ├── perf_28ch_rank2.xlsx
    │       ├── ...
    │       ├── perf_42ch_rank0.xlsx
    │       ├── perf_42ch_rank1.xlsx
    │       └── ...
    └── 512thread/
        └── individual_reports/
            ├── perf_28ch_rank0.xlsx
            └── ...
```

### 3.3 Output Differences

**Single Config Mode Output:** `gpu_timeline_summary_{mean|geomean}.xlsx`

| Sheet | Description |
|-------|-------------|
| `Summary` | Aggregated metrics across ranks |
| `All_Ranks_Combined` | Raw data from all ranks with rank column |
| `Per_Rank_Time_ms` | Pivot table: type × rank (time values) |
| `Per_Rank_Percent` | Pivot table: type × rank (percentages) |

**Sweep Mode Output:** `gpu_timeline_all_configs_{mean|geomean}.xlsx`

| Sheet | Description |
|-------|-------------|
| `All_Data` | Complete dataset with all configs + metadata |
| `Pivot_Time_ms` | Pivot table: type × full_config (time values) |
| `Pivot_Percent` | Pivot table: type × full_config (percentages) |
| `Summary_By_Config` | Key metrics per configuration |

### 3.4 Metadata Columns

**Single Config Mode:**
```python
# Minimal metadata
aggregated["num_ranks"] = len(perf_files)
```

**Sweep Mode:**
```python
# Rich metadata for each configuration
aggregated["thread_config"] = thread_config      # e.g., "256thread"
aggregated["threads_num"] = 256                  # numeric for sorting
aggregated["channel_config"] = channel_config    # e.g., "28ch"  
aggregated["channels_num"] = 28                  # numeric for sorting
aggregated["full_config"] = "256thread_28ch"     # combined identifier
aggregated["num_ranks"] = num_ranks
```

### 3.5 Auto-Detection Logic

```python
def detect_mode(input_dir):
    """Auto-detect processing mode from directory structure."""
    input_path = Path(input_dir)
    
    # Check for sweep structure
    tracelens_dir = input_path / "tracelens_analysis"
    if tracelens_dir.exists():
        thread_dirs = [d for d in tracelens_dir.iterdir() 
                       if d.is_dir() and "thread" in d.name]
        if thread_dirs:
            return "sweep"
    
    # Check for single config structure
    if input_path.name == "individual_reports":
        return "single"
    if list(input_path.glob("perf_rank*.xlsx")):
        return "single"
    
    # Check for sweep files in current directory
    if list(input_path.glob("perf_*ch_rank*.xlsx")):
        return "sweep"
    
    raise ValueError("Could not auto-detect mode")
```

### 3.6 Shared Code

Both modes share identical `geometric_mean()` function:

```python
def geometric_mean(values):
    """Calculate geometric mean, handling zeros."""
    values = np.array(values)
    values = np.where(values == 0, 1e-10, values)
    return np.exp(np.mean(np.log(values)))
```

And similar aggregation logic:

```python
agg_func = geometric_mean if use_geo_mean else "mean"
aggregated = (
    combined.groupby("type")
    .agg({"time ms": agg_func, "percent": agg_func})
    .reset_index()
)
```

---

## 4. `process comms` - NCCL Communication Processing

### 4.1 Source Script

**Location:** `scripts/gemm_analysis/process_comms.py` (291 lines)

### 4.2 Purpose

Process NCCL collective reports from a sweep directory and generate combined CSV/Excel files with communication metrics for analysis and visualization.

### 4.3 Input Structure

```
sweep_dir/
└── tracelens_analysis/
    ├── 256thread/
    │   └── collective_reports/
    │       ├── collective_28ch.xlsx
    │       ├── collective_42ch.xlsx
    │       ├── collective_56ch.xlsx
    │       └── collective_70ch.xlsx
    └── 512thread/
        └── collective_reports/
            ├── collective_28ch.xlsx
            ├── collective_42ch.xlsx
            └── ...
```

### 4.4 Processing Steps

1. **Find Configurations**
   - Discover thread configurations (e.g., `256thread`, `512thread`)
   - For each thread config, find collective report files

2. **Read Data**
   - Open each `collective_*.xlsx` file
   - Read `nccl_summary_implicit_sync` sheet
   - Extract communication metrics

3. **Add Metadata**
   ```python
   df['thread_config'] = thread_config      # e.g., "256thread"
   df['threads_num'] = 256
   df['channel_config'] = channel_config    # e.g., "28ch"
   df['channels_num'] = 28
   df['source_file'] = filename
   df['full_config'] = f"{thread_config}_{channel_config}"
   ```

4. **Create Operation IDs**
   ```python
   # Based on message size
   unique_sizes = sorted(combined_df['Full msg size (MB)'].unique())
   size_to_id = {size: f"OP_{i+1:02d}" for i, size in enumerate(unique_sizes)}
   combined_df['operation_id'] = combined_df['Full msg size (MB)'].map(size_to_id)
   
   # Create readable operation names
   def create_op_name(row):
       size_mb = row['Full msg size (MB)']
       if size_mb < 0.01:
           return f"tiny_{size_mb*1000:.3f}KB"
       elif size_mb < 100:
           return f"medium_{size_mb:.2f}MB"
       else:
           return f"large_{size_mb:.2f}MB"
   ```

5. **Combine and Save**
   - Merge all DataFrames
   - Reorder columns for readability
   - Save as Excel and CSV

### 4.5 Output Files

**`nccl_master_all_configs.xlsx`** - Excel file with pivot table support

**`nccl_master_all_configs.csv`** - CSV for pandas/scripts

### 4.6 Output Columns

```python
column_order = [
    # Unique identifiers
    'operation_id', 'operation_name', 'Full msg size (MB)', 'In msg nelems',
    
    # Configuration
    'threads_num', 'thread_config', 'channels_num', 'channel_config', 'full_config',
    
    # Operation info
    'Collective name', 'dtype', 'Group size', 'count',
    
    # Communication Latency
    'comm_latency_mean', 'comm_latency_median', 'comm_latency_min', 'comm_latency_max',
    'Total comm latency (ms)',
    
    # Algorithm Bandwidth
    'algo bw (GB/s)_mean', 'algo bw (GB/s)_median', 'algo bw (GB/s)_min', 'algo bw (GB/s)_max',
    
    # Bus Bandwidth
    'bus bw (GB/s)_mean', 'bus bw (GB/s)_median', 'bus bw (GB/s)_min', 'bus bw (GB/s)_max',
    
    # Start/End Time Skew
    'skew in start time_mean', 'skew in start time_median', ...
    'skew in end time_mean', 'skew in end time_median', ...
    
    # Process Group Info
    'Process Group Name', 'source_file'
]
```

---

## 5. `process gemm-variance` - Timestamp Enhancement

### 5.1 Source Script

**Location:** `scripts/gemm_analysis/enhance_gemm_variance_with_timestamps.py` (274 lines)

### 5.2 Purpose

Enhance GEMM variance CSV (output from `analyze gemm`) with actual kernel timestamps by finding the specific kernel instances with min and max durations in the original trace files.

### 5.3 Input

1. **GEMM Variance CSV** - Output from `aorta-report analyze gemm`
   - Contains columns: `threads`, `channel`, `rank`, `kernel_name`, `kernel_time_min_us`, `kernel_time_max_us`, `time_diff_us`

2. **Base Path** - Sweep directory containing original trace files

### 5.4 Processing Steps

For each row in the variance CSV:

1. **Find Trace File**
   ```python
   def get_trace_file_path(base_path, threads, channel, rank):
       trace_dir = base_path / f"{threads}thread" / f"nccl_{channel}channels" / \
                   "torch_profiler" / f"rank{rank}"
       # Look for JSON trace files
       trace_files = list(trace_dir.glob("*.json"))
       # Prefer customer_trace files
       for pattern in ["customer_trace*.json", "*.json"]:
           matches = list(trace_dir.glob(pattern))
           if matches:
               return matches[0]
       return None
   ```

2. **Search Trace for Kernel Instances**
   ```python
   def find_min_max_kernel_timestamps(trace_file, kernel_name, 
                                       min_duration_us, max_duration_us, tolerance=0.01):
       with open(trace_file, 'r') as f:
           data = json.load(f)
       
       events = data['traceEvents']
       kernel_instances = []
       
       for event in events:
           if event.get('cat') == 'kernel' and \
              event.get('name', '').startswith(kernel_name):
               duration_us = event.get('dur')
               timestamp_us = event.get('ts')
               kernel_instances.append({
                   'duration_us': duration_us,
                   'timestamp_ms': timestamp_us / 1000.0,
               })
       
       # Sort by duration
       kernel_instances.sort(key=lambda x: x['duration_us'])
       
       min_instance = kernel_instances[0]   # Shortest duration
       max_instance = kernel_instances[-1]  # Longest duration
       
       return {
           'min_timestamp_ms': min_instance['timestamp_ms'],
           'max_timestamp_ms': max_instance['timestamp_ms'],
           'min_duration_found_us': min_instance['duration_us'],
           'max_duration_found_us': max_instance['duration_us'],
       }
   ```

3. **Update DataFrame**
   ```python
   df['min_duration_timestamp_ms'] = ...
   df['max_duration_timestamp_ms'] = ...
   df['time_between_min_max_ms'] = abs(max_ts - min_ts)
   df['min_duration_found_us'] = ...  # For verification
   df['max_duration_found_us'] = ...  # For verification
   ```

### 5.5 Output

Enhanced CSV with additional columns:

| Column | Description |
|--------|-------------|
| `min_duration_timestamp_ms` | When the shortest kernel instance occurred |
| `max_duration_timestamp_ms` | When the longest kernel instance occurred |
| `time_between_min_max_ms` | Time difference between occurrences |
| `min_duration_found_us` | Actual min duration found (for verification) |
| `max_duration_found_us` | Actual max duration found (for verification) |

---

## 6. Implementation Architecture

### 6.1 File Structure

```
src/aorta/report/
├── cli.py                       # CLI definitions (update process commands)
├── analysis/                    # EXISTING - analyze command logic
│   ├── __init__.py
│   ├── tracelens_wrapper.py
│   ├── analyze_gemm.py
│   ├── analyze_single.py
│   └── analyze_sweep.py
├── generators/                  # EXISTING - generate command logic
├── templates/                   # EXISTING - HTML templates
└── processing/                  # NEW - process command logic
    ├── __init__.py              # Package exports
    ├── gpu_timeline_single.py   # Single config GPU timeline processing
    ├── gpu_timeline_sweep.py    # Sweep GPU timeline processing
    ├── process_comms.py         # NCCL communication processing
    └── process_gemm_variance.py # GEMM variance timestamp enhancement
```

### 6.2 Module Responsibilities

#### `processing/__init__.py`
```python
from .gpu_timeline_single import process_single_config
from .gpu_timeline_sweep import process_sweep_config
from .process_comms import process_nccl_data
from .process_gemm_variance import enhance_gemm_variance

__all__ = [
    "process_single_config",
    "process_sweep_config",
    "process_nccl_data",
    "enhance_gemm_variance",
]
```

#### `processing/gpu_timeline_single.py`
```python
def geometric_mean(values: np.ndarray) -> float:
    """Calculate geometric mean, handling zeros."""

def process_single_config(
    reports_dir: Path,
    use_geo_mean: bool = False,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Process GPU timeline from single config individual reports.
    
    Args:
        reports_dir: Path to individual_reports directory
        use_geo_mean: Use geometric mean instead of arithmetic mean
        output_path: Custom output path
        verbose: Print verbose output
    
    Returns:
        Path to output Excel file
    """
```

#### `processing/gpu_timeline_sweep.py`
```python
def process_sweep_config(
    sweep_dir: Path,
    use_geo_mean: bool = False,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Process GPU timeline from sweep directory with multiple configs.
    
    Args:
        sweep_dir: Path to sweep directory
        use_geo_mean: Use geometric mean
        output_path: Custom output path
        verbose: Print verbose output
    
    Returns:
        Path to output Excel file
    """

def parse_perf_filename(filename: str) -> Tuple[str, int]:
    """Parse perf_28ch_rank0.xlsx to extract channel and rank."""

def group_files_by_channel(perf_files: List[str]) -> Dict[str, List[Tuple[int, str]]]:
    """Group performance files by channel configuration."""

def aggregate_rank_data(rank_data, thread_config, channel_config, 
                        num_ranks, use_geo_mean) -> pd.DataFrame:
    """Aggregate data across ranks with metadata."""

def create_pivot_sheet(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Create pivot table from DataFrame."""

def create_summary_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary sheet with key metrics per configuration."""
```

#### `processing/process_comms.py`
```python
def process_nccl_data(
    sweep_dir: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Process NCCL collective reports from sweep directory.
    
    Args:
        sweep_dir: Path to sweep directory
        output_dir: Custom output directory
        verbose: Print verbose output
    
    Returns:
        Tuple of (excel_path, csv_path)
    """

def create_operation_id(size_mb: float) -> str:
    """Create operation ID based on message size."""

def create_operation_name(size_mb: float) -> str:
    """Create readable operation name."""
```

#### `processing/process_gemm_variance.py`
```python
def enhance_gemm_variance(
    input_csv: Path,
    base_path: Path,
    output_csv: Optional[Path] = None,
    tolerance: float = 0.01,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Enhance GEMM variance CSV with timestamp information.
    
    Args:
        input_csv: Input CSV file with GEMM variance data
        base_path: Base path to sweep directory with trace files
        output_csv: Output CSV path
        tolerance: Duration matching tolerance (fraction)
        verbose: Print verbose output
    
    Returns:
        Path to output CSV file
    """

def get_trace_file_path(base_path: Path, threads: int, 
                        channel: int, rank: int) -> Optional[Path]:
    """Find trace file for a given configuration."""

def find_min_max_kernel_timestamps(trace_file: Path, kernel_name: str,
                                   min_duration_us: float, max_duration_us: float,
                                   tolerance: float = 0.01) -> Dict[str, Optional[float]]:
    """Find timestamps for kernel instances with min and max durations."""
```

### 6.3 Data Flow

```
CLI (cli.py)
    │
    ├── process gpu-timeline ──────►  Auto-detect mode
    │                                       │
    │                         ┌─────────────┴─────────────┐
    │                         │                           │
    │                         ▼                           ▼
    │               processing.process_single_config()   processing.process_sweep_config()
    │                         │                           │
    │                         ▼                           ▼
    │               gpu_timeline_summary_*.xlsx   gpu_timeline_all_configs_*.xlsx
    │
    ├── process comms ─────────────►  processing.process_nccl_data()
    │                                       │
    │                                       ▼
    │                               nccl_master_all_configs.xlsx/csv
    │
    └── process gemm-variance ─────►  processing.enhance_gemm_variance()
                                            │
                                            ▼
                                    *_with_timestamps.csv
```

---

## 7. Implementation Status

### Phase 1: Create `processing/` directory structure ✅
- [x] Create `processing/__init__.py`
- [x] Set up exports

### Phase 2: Implement `gpu_timeline_single.py` ✅
- [x] Port logic from `tracelens_single_config/process_gpu_timeline.py`
- [x] Create `process_single_config()` function
- [x] Add proper error handling and progress reporting

### Phase 3: Implement `gpu_timeline_sweep.py` ✅
- [x] Port logic from `gemm_analysis/process_gpu_timeline.py`
- [x] Create `process_sweep_config()` function
- [x] Implement metadata extraction
- [x] Implement pivot table generation

### Phase 4: Update `cli.py` for `process gpu-timeline` ✅
- [x] Update command to call new processing modules
- [x] Implement auto-detection logic

### Phase 5: Implement `process_comms.py` ✅
- [x] Port logic from `gemm_analysis/process_comms.py`
- [x] Create `process_nccl_data()` function
- [x] Add progress reporting and summary statistics
- [x] Update CLI command

### Phase 6: Implement `process_gemm_variance.py` ✅
- [x] Port logic from `enhance_gemm_variance_with_timestamps.py`
- [x] Create `enhance_gemm_variance()` function
- [x] Add progress reporting
- [x] Update CLI command

### Phase 7: Documentation ✅
- [x] Update this document with implementation status
- [x] Add to functional spec

---

## 8. Expected Output Examples

### 8.1 `process gpu-timeline --mode single` Output

```
Processing GPU timeline from: /path/to/individual_reports
Aggregation: Arithmetic Mean
Found 8 rank files
  Rank 0: OK
  Rank 1: OK
  Rank 2: OK
  Rank 3: OK
  Rank 4: OK
  Rank 5: OK
  Rank 6: OK
  Rank 7: OK

Saved: /path/to/gpu_timeline_summary_mean.xlsx

Summary:
              type   time ms   percent
       busy_time    125.450    85.230
       idle_time     21.780    14.770
      total_time    147.230   100.000
computation_time     98.340    66.820
exposed_comm_time    27.110    18.410
```

### 8.2 `process gpu-timeline --mode sweep` Output

```
================================================================================
Processing GPU Timeline data from: /path/to/sweep
Aggregation method: Arithmetic Mean
================================================================================

Found thread configurations: ['256thread', '512thread']

Processing: 256thread
------------------------------------------------------------
  28ch: Processing 8 ranks...
    [OK] Aggregated across 8 ranks
  42ch: Processing 8 ranks...
    [OK] Aggregated across 8 ranks
  56ch: Processing 8 ranks...
    [OK] Aggregated across 8 ranks
  70ch: Processing 8 ranks...
    [OK] Aggregated across 8 ranks

Processing: 512thread
------------------------------------------------------------
  28ch: Processing 8 ranks...
    [OK] Aggregated across 8 ranks
  ...

================================================================================
CREATING OUTPUT FILE
================================================================================
[SAVED] /path/to/tracelens_analysis/gpu_timeline_all_configs_mean.xlsx
  Sheets created:
    1. All_Data - Complete dataset
    2. Pivot_Time_ms - Matrix view of time (ms)
    3. Pivot_Percent - Matrix view of percentages
    4. Summary_By_Config - Key metrics per configuration

================================================================================
SUMMARY
================================================================================

Metric Types Found:
  busy_time                 (8 configurations)
  computation_time          (8 configurations)
  exposed_comm_time         (8 configurations)
  idle_time                 (8 configurations)
  total_time                (8 configurations)

Configurations Processed:
  256thread_28ch            (8 ranks)
  256thread_42ch            (8 ranks)
  256thread_56ch            (8 ranks)
  256thread_70ch            (8 ranks)
  512thread_28ch            (8 ranks)
  512thread_42ch            (8 ranks)
  512thread_56ch            (8 ranks)
  512thread_70ch            (8 ranks)

================================================================================
KEY METRICS COMPARISON (Sorted by Busy Time)
================================================================================

Busy Time (lower is better):
     full_config   time ms   percent
  512thread_70ch    98.234    78.45
  512thread_56ch   102.456    79.12
  256thread_70ch   115.678    82.34
  ...

Idle Time (lower is better):
     full_config   time ms   percent
  512thread_70ch    12.345    9.87
  512thread_56ch    14.567   11.23
  ...

================================================================================
COMPLETE!
================================================================================
Output file: /path/to/tracelens_analysis/gpu_timeline_all_configs_mean.xlsx
Open in Excel to create custom pivots and charts!
================================================================================
```

### 8.3 `process comms` Output

```
================================================================================
Processing NCCL data from: /path/to/sweep
================================================================================

Found thread configurations: ['256thread', '512thread']

Processing: 256thread
------------------------------------------------------------
  Reading: collective_28ch.xlsx
    [OK] Loaded 15 rows
  Reading: collective_42ch.xlsx
    [OK] Loaded 15 rows
  Reading: collective_56ch.xlsx
    [OK] Loaded 15 rows
  Reading: collective_70ch.xlsx
    [OK] Loaded 15 rows

Processing: 512thread
------------------------------------------------------------
  Reading: collective_28ch.xlsx
    [OK] Loaded 15 rows
  ...

================================================================================
COMBINING AND PROCESSING DATA
================================================================================
Total rows: 120
Total columns: 28

Creating unique operation IDs...

================================================================================
SAVING DATA FILE
================================================================================
[SAVED] Excel: nccl_master_all_configs.xlsx
  Rows: 120, Columns: 28
[SAVED] CSV: nccl_master_all_configs.csv
  (Use Excel file for pivot tables, CSV for pandas/scripts)

================================================================================
SUMMARY
================================================================================

Operation ID Mapping:
------------------------------------------------------------
  OP_01:     0.001953 MB  (       512 elements)  tiny_1.953KB
  OP_02:     0.062500 MB  (     16384 elements)  medium_0.06MB
  OP_03:     0.500000 MB  (    131072 elements)  medium_0.50MB
  OP_04:     4.000000 MB  (   1048576 elements)  medium_4.00MB
  OP_05:   128.000000 MB  (  33554432 elements)  large_128.00MB

Configurations:
------------------------------------------------------------
  256thread    28ch     -> 15 operations
  256thread    42ch     -> 15 operations
  256thread    56ch     -> 15 operations
  256thread    70ch     -> 15 operations
  512thread    28ch     -> 15 operations
  512thread    42ch     -> 15 operations
  512thread    56ch     -> 15 operations
  512thread    70ch     -> 15 operations

Total Communication Time by Configuration:
------------------------------------------------------------
  512thread_70ch           :     123.45 ms
  512thread_56ch           :     134.56 ms
  256thread_70ch           :     145.67 ms
  ...

Best Configuration by Operation:
------------------------------------------------------------
  OP_01 (    0.00 MB): 512thread_70ch      (    0.12 ms)
  OP_02 (    0.06 MB): 512thread_56ch      (    0.45 ms)
  OP_03 (    0.50 MB): 256thread_70ch      (    1.23 ms)
  OP_04 (    4.00 MB): 512thread_70ch      (    5.67 ms)
  OP_05 (  128.00 MB): 512thread_70ch      (   89.12 ms)

================================================================================
COMPLETE!
================================================================================
Generated files:
  1. nccl_master_all_configs.xlsx (Excel - use for pivot tables)
  2. nccl_master_all_configs.csv (CSV - use for pandas/scripts)

Recommended workflow:
  1. Open Excel file: libreoffice nccl_master_all_configs.xlsx
  2. Create pivot table: Select all -> Insert -> Pivot Table
  3. Setup: Rows=operation_id, Columns=full_config, Values=comm_latency_mean
================================================================================
```

### 8.4 `process gemm-variance` Output

```
GEMM Variance Timestamp Enhancement
============================================================
Input CSV: ./top5_gemm_kernels_time_variance.csv
Output CSV: ./top5_gemm_kernels_time_variance_with_timestamps.csv
Base path: /path/to/sweep
Tolerance: 1.0%

Processing 320 rows...

Processing row 1/320
  Config: 256thread/28ch/rank0
  Kernel: Cijk_Alik_Bljk_HBH_BH_MT128x128x16_MI16x16x1_SE_1LDSB0_APM1_ABV0_ACED0...
  Duration range: [45.234, 89.456] us
  Using trace: customer_trace_1234567890.json
  Found timestamps: min at 1523.456ms, max at 2345.678ms (diff: 822.222ms)
  Verification: found min=45.234us (expected 45.234us), found max=89.456us (expected 89.456us)

Processing row 2/320
  Config: 256thread/28ch/rank0
  Kernel: Cijk_Alik_Bljk_HBH_BH_MT64x64x32_MI16x16x1_SE_1LDSB0_APM1_ABV0_ACED0...
  Duration range: [32.123, 67.890] us
  Using trace: customer_trace_1234567890.json
  Found timestamps: min at 1678.901ms, max at 2456.789ms (diff: 777.888ms)
  ...

Processing row 320/320
  Config: 512thread/70ch/rank7
  ...

Enhanced CSV saved to: ./top5_gemm_kernels_time_variance_with_timestamps.csv

Summary:
  Total rows: 320
  Rows with timestamps: 312
  Success rate: 97.5%

Time between min/max occurrences:
  Mean: 456.789 ms
  Median: 234.567 ms
  Max: 1234.567 ms
  Min: 12.345 ms

[OK] Enhancement complete!
```

---

## Appendix A: Migration Checklist

### From `tracelens_single_config/process_gpu_timeline.py` ✅
- [x] `geometric_mean()` function
- [x] File pattern matching (`perf_rank*.xlsx`)
- [x] Excel sheet generation (Summary, All_Ranks_Combined, Per_Rank_*)
- [x] Aggregation logic
- [x] Output path generation

### From `gemm_analysis/process_gpu_timeline.py` ✅
- [x] `geometric_mean()` function (shared)
- [x] `parse_perf_filename()` function
- [x] `group_files_by_channel()` function
- [x] Thread/channel config discovery
- [x] `aggregate_rank_data()` with metadata
- [x] `create_pivot_sheet()` function
- [x] `create_summary_sheet()` function
- [x] `print_summary_report()` function

### From `gemm_analysis/process_comms.py` ✅
- [x] Thread config discovery
- [x] Excel reading (`nccl_summary_implicit_sync` sheet)
- [x] Metadata column addition
- [x] Operation ID creation
- [x] Operation name creation
- [x] Column reordering
- [x] Excel/CSV output
- [x] Summary statistics

### From `enhance_gemm_variance_with_timestamps.py` ✅
- [x] `get_trace_file_path()` function
- [x] `find_min_max_kernel_timestamps()` function
- [x] JSON trace parsing
- [x] Kernel instance searching
- [x] Duration tolerance matching
- [x] CSV enhancement
- [x] Progress reporting

---

## Appendix B: Error Handling

| Scenario | Handling |
|----------|----------|
| Directory not found | Raise `FileNotFoundError` with helpful message |
| No Excel files found | Print warning, return `None` |
| Missing sheet in Excel | Print warning, skip file |
| No valid data loaded | Raise `ValueError` |
| Trace file not found | Print warning, skip row |
| No kernel instances found | Print warning, set timestamps to `None` |
| Duration mismatch | Print warning with expected vs found values |

