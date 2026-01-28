# `compare` Command Group - Developer Documentation

**Version:** 1.0  
**Date:** January 2026  
**Status:** ✅ Implemented

---

## Table of Contents

1. [Overview](#1-overview)
2. [Command Specification](#2-command-specification)
3. [Source Script Analysis](#3-source-script-analysis)
4. [Implementation Architecture](#4-implementation-architecture)
5. [Module Details](#5-module-details)
6. [Data Flow](#6-data-flow)
7. [Implementation Order](#7-implementation-order)
8. [Expected Output](#8-expected-output)
9. [Testing Strategy](#9-testing-strategy)

---

## 1. Overview

The `compare` command provides functionality to compare baseline and test TraceLens reports. It supports two comparison types:

| Type | Purpose | Source Scripts |
|------|---------|----------------|
| `gpu_timeline` | Compare GPU timeline reports | `combine_reports.py` + `add_comparison_sheets.py` |
| `collective` | Compare collective/NCCL reports | `combine_reports.py` + `add_collective_comparison.py` |

### Key Design Decisions

1. **Single command with positional type argument** - cleaner than separate commands
2. **Exact Excel file paths** - user specifies exact files, no auto-discovery
3. **2-way comparison only** - baseline vs test (N-way comparison deferred)
4. **Shared combine logic** - reuse same function for both types
5. **Match original behavior** - output same sheets and formatting as original scripts

---

## 2. Command Specification

### 2.1 `aorta-report compare gpu_timeline`

Compare two GPU timeline reports.

```bash
aorta-report compare gpu_timeline \
    --baseline /path/to/baseline/gpu_timeline_summary_mean.xlsx \
    --test /path/to/test/gpu_timeline_summary_mean.xlsx \
    --baseline-label "ROCm 6.0" \
    --test-label "ROCm 7.0" \
    --output /path/to/gpu_comparison.xlsx
```

| Argument/Option | Required | Default | Description |
|-----------------|----------|---------|-------------|
| `--baseline`, `-b` | Yes | - | Path to baseline gpu_timeline_summary_mean.xlsx |
| `--test`, `-t` | Yes | - | Path to test gpu_timeline_summary_mean.xlsx |
| `--baseline-label` | No | grandparent dir name | Label for baseline in output |
| `--test-label` | No | grandparent dir name | Label for test in output |
| `--output`, `-o` | Yes | - | Output Excel file path |

**Label Extraction Logic:**
- If `--baseline-label` not provided: extract grandparent directory name
- Example: `/path/to/56cu_256threads/tracelens_analysis/gpu_timeline.xlsx` → `56cu_256threads`
- Fallback: `"baseline"` if extraction fails

**Output Sheets:**
| Sheet | Description | Source |
|-------|-------------|--------|
| Summary | Combined summaries with `source` column | Combined |
| All_Ranks_Combined | Combined raw data with `source` column | Combined |
| Per_Rank_Time_ms | Combined pivot (time) | Combined |
| Per_Rank_Percent | Combined pivot (percent) | Combined |
| **Comparison_By_Rank** | Per-rank comparison with metrics | NEW |
| **Summary_Comparison** | Overall comparison with metrics | NEW |

---

### 2.2 `aorta-report compare collective`

Compare two collective/NCCL reports.

```bash
aorta-report compare collective \
    --baseline /path/to/baseline/collective_all_ranks.xlsx \
    --test /path/to/test/collective_all_ranks.xlsx \
    --baseline-label "ROCm 6.0" \
    --test-label "ROCm 7.0" \
    --output /path/to/collective_comparison.xlsx
```

| Argument/Option | Required | Default | Description |
|-----------------|----------|---------|-------------|
| `--baseline`, `-b` | Yes | - | Path to baseline collective_all_ranks.xlsx |
| `--test`, `-t` | Yes | - | Path to test collective_all_ranks.xlsx |
| `--baseline-label` | No | grandparent dir name | Label for baseline in output |
| `--test-label` | No | grandparent dir name | Label for test in output |
| `--output`, `-o` | Yes | - | Output Excel file path |

**Sheet Filtering (matches original):**
- Only sheets with `"summary"` in the name are kept
- Non-summary sheets are skipped

**Output Sheets:**
| Sheet | Description | Source |
|-------|-------------|--------|
| nccl_summary_implicit_sync | Combined summary (implicit sync) | Combined |
| nccl_summary_long | Combined summary (long) | Combined |
| **nccl_implicit_sync_cmp** | Comparison for implicit sync | NEW |
| **nccl_long_cmp** | Comparison for long | NEW |

---

## 3. Source Script Analysis

### 3.1 `combine_reports.py` (72 lines)

**Location:** `scripts/tracelens_single_config/combine_reports.py`

**Purpose:** Combine two Excel files by adding a `source` column.

**Key Logic:**
```python
def combine_collective_reports(baseline_path, test_path, output_path, baseline_label, test_label):
    baseline_xl = pd.ExcelFile(baseline_path)
    test_xl = pd.ExcelFile(test_path)
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in baseline_xl.sheet_names:
            if sheet_name not in test_xl.sheet_names:
                continue  # Skip sheets not in both files
            
            baseline_df = pd.read_excel(baseline_path, sheet_name=sheet_name)
            test_df = pd.read_excel(test_path, sheet_name=sheet_name)
            
            baseline_df["source"] = baseline_label
            test_df["source"] = test_label
            
            combined = pd.concat([baseline_df, test_df], ignore_index=True)
            combined.to_excel(writer, sheet_name=sheet_name, index=False)
```

---

### 3.2 `add_comparison_sheets.py` (222 lines)

**Location:** `scripts/tracelens_single_config/add_comparison_sheets.py`

**Purpose:** Add GPU timeline comparison sheets to combined Excel file.

**Key Logic:**

```python
def add_comparison_sheets(input_path, output_path, baseline_label, test_label):
    xl = pd.ExcelFile(input_path)
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 1. Copy all original sheets
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(input_path, sheet_name=sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 2. Read combined data
        all_combined = pd.read_excel(input_path, sheet_name="All_Ranks_Combined")
        
        # Get actual source values from dataframe
        sources = all_combined['source'].unique()
        actual_baseline = sources[0]
        actual_test = sources[1]
        
        # 3. Create Comparison_By_Rank
        baseline_data = all_combined[all_combined["source"] == actual_baseline]
        test_data = all_combined[all_combined["source"] == actual_test]
        
        comparison_by_rank = pd.DataFrame()
        for rank in sorted(baseline_data["rank"].unique()):
            base_rank = baseline_data[baseline_data["rank"] == rank].set_index("type")
            test_rank = test_data[test_data["rank"] == rank].set_index("type")
            
            for metric_type in base_rank.index:
                if metric_type in test_rank.index:
                    base_time = base_rank.loc[metric_type, "time ms"]
                    test_time = test_rank.loc[metric_type, "time ms"]
                    
                    # percent_change: positive when test is faster (takes less time)
                    pct_change = (base_time - test_time) / base_time * 100 if base_time != 0 else 0
                    
                    # Determine status
                    if pct_change > 1:
                        status = "Better"
                    elif pct_change < -1:
                        status = "Worse"
                    else:
                        status = "Similar"
                    
                    # Build row with all metrics
                    row = {
                        "rank": rank,
                        "type": metric_type,
                        f"{baseline_label}_time_ms": base_time,
                        f"{test_label}_time_ms": test_time,
                        "diff_time_ms": test_time - base_time,
                        "percent_change": pct_change,
                        "status": status,
                        "ratio": test_time / base_time if base_time != 0 else 0,
                        f"{baseline_label}_percent": base_rank.loc[metric_type, "percent"],
                        f"{test_label}_percent": test_rank.loc[metric_type, "percent"],
                        "diff_percent": test_rank.loc[metric_type, "percent"] - base_rank.loc[metric_type, "percent"],
                    }
                    comparison_by_rank = pd.concat([comparison_by_rank, pd.DataFrame([row])], ignore_index=True)
        
        comparison_by_rank.to_excel(writer, sheet_name="Comparison_By_Rank", index=False)
        
        # 4. Create Summary_Comparison (similar logic with Summary sheet)
        # ...
        
        # 5. Apply conditional formatting
        ws = writer.sheets["Comparison_By_Rank"]
        # Find percent_change column and apply color scale
        ws.conditional_formatting.add(
            data_range,
            ColorScaleRule(
                start_type="min", start_color="F8696B",  # Red
                mid_type="num", mid_value=0, mid_color="FFFFFF",  # White
                end_type="max", end_color="63BE7B",  # Green
            )
        )
```

**Comparison Columns Created:**
| Column | Formula | Description |
|--------|---------|-------------|
| `{baseline}_time_ms` | baseline value | Time from baseline |
| `{test}_time_ms` | test value | Time from test |
| `diff_time_ms` | test - baseline | Absolute difference |
| `percent_change` | (baseline - test) / baseline × 100 | Positive = faster |
| `status` | Based on percent_change | "Better", "Worse", or "Similar" |
| `ratio` | test / baseline | Ratio comparison |
| `{baseline}_percent` | baseline value | Percent from baseline |
| `{test}_percent` | test value | Percent from test |
| `diff_percent` | test - baseline | Difference in percent |

---

### 3.3 `add_collective_comparison.py` (209 lines)

**Location:** `scripts/tracelens_single_config/add_collective_comparison.py`

**Purpose:** Add NCCL collective comparison sheets.

**Key Differences from GPU Timeline:**

1. **Sheet Filtering:** Only keeps sheets with "summary" in the name
2. **Grouping:** Groups by `['Collective name', 'dtype', 'In msg nelems']`
3. **Multiple Metrics:** Compares multiple NCCL-specific metrics
4. **Semantic Difference:** Latency vs Bandwidth have opposite "better" directions

**Key Logic:**

```python
def add_collective_comparison_sheets(input_path, output_path, baseline_label, test_label):
    xl = pd.ExcelFile(input_path)
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 1. Copy only summary sheets
        for sheet_name in xl.sheet_names:
            if "summary" not in sheet_name.lower():
                continue  # Skip non-summary sheets
            df = pd.read_excel(input_path, sheet_name=sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 2. Process each summary sheet
        for sheet_name in ["nccl_summary_implicit_sync", "nccl_summary_long"]:
            if sheet_name not in xl.sheet_names:
                continue
            
            df = pd.read_excel(input_path, sheet_name=sheet_name)
            
            # Get actual source values
            sources = df['source'].unique()
            actual_baseline = sources[0]
            actual_test = sources[1]
            
            baseline_df = df[df["source"] == actual_baseline]
            test_df = df[df["source"] == actual_test]
            
            # Group columns
            group_cols = ["Collective name", "dtype", "In msg nelems"]
            
            # Metrics to compare
            numeric_cols = [
                "comm_latency_mean",
                "algo bw (GB/s)_mean",
                "bus bw (GB/s)_mean",
                "Total comm latency (ms)",
                "count",
            ]
            
            comparison = pd.DataFrame()
            
            for name, base_group in baseline_df.groupby(group_cols):
                # Find matching test group
                # ... matching logic ...
                
                comp_row = {}
                
                # Copy grouping columns
                for col, val in zip(group_cols, name):
                    comp_row[col] = val
                
                # Compare each metric
                for col in numeric_cols:
                    base_val = base_group[col].values[0]
                    test_val = test_group[col].values[0]
                    
                    comp_row[f"{actual_baseline}_{col}"] = base_val
                    comp_row[f"{actual_test}_{col}"] = test_val
                    comp_row[f"diff_{col}"] = test_val - base_val
                    
                    # percent_change semantics differ by metric type
                    if "latency" in col.lower() or "time" in col.lower():
                        # Lower is better - positive when test is faster
                        pct_change = (base_val - test_val) / base_val * 100
                    elif "bw" in col.lower() or "bandwidth" in col.lower():
                        # Higher is better - positive when test is better
                        pct_change = (test_val - base_val) / base_val * 100
                    else:
                        pct_change = 0
                    
                    comp_row[f"percent_change_{col}"] = pct_change
                    comp_row[f"ratio_{col}"] = test_val / base_val if base_val != 0 else 0
                
                comparison = pd.concat([comparison, pd.DataFrame([comp_row])], ignore_index=True)
            
            # Sheet name: nccl_summary_implicit_sync → nccl_implicit_sync_cmp
            comparison_sheet_name = sheet_name.replace("nccl_summary_", "nccl_") + "_cmp"
            comparison.to_excel(writer, sheet_name=comparison_sheet_name, index=False)
            
            # Apply formatting to all percent_change columns
            # ...
```

**Metrics Compared:**
| Metric | Better Direction | percent_change Formula |
|--------|------------------|------------------------|
| `comm_latency_mean` | Lower | (base - test) / base × 100 |
| `algo bw (GB/s)_mean` | Higher | (test - base) / base × 100 |
| `bus bw (GB/s)_mean` | Higher | (test - base) / base × 100 |
| `Total comm latency (ms)` | Lower | (base - test) / base × 100 |
| `count` | N/A | No percent_change |

---

## 4. Implementation Architecture

### 4.1 File Structure

```
src/aorta/report/
├── comparison/                      # NEW: comparison module
│   ├── __init__.py                  # Package exports
│   ├── combine.py                   # Shared: combine two Excel files
│   ├── gpu_timeline_comparison.py   # GPU timeline comparison logic
│   ├── collective_comparison.py     # Collective/NCCL comparison logic
│   └── formatting.py                # Shared Excel formatting utilities
├── cli.py                           # Update compare commands
└── ... (existing modules)
```

### 4.2 Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `combine.py` | Combine two Excel files with source column |
| `gpu_timeline_comparison.py` | Add Comparison_By_Rank and Summary_Comparison sheets |
| `collective_comparison.py` | Add nccl_*_cmp sheets for each summary sheet |
| `formatting.py` | Color scale formatting, column letter conversion |

### 4.3 Dependency Graph

```
cli.py
    │
    ├── compare gpu_timeline ──► combine.py
    │                                │
    │                                └──► gpu_timeline_comparison.py
    │                                              │
    │                                              └──► formatting.py
    │
    └── compare collective ───► combine.py
                                     │
                                     └──► collective_comparison.py
                                                   │
                                                   └──► formatting.py
```

---

## 5. Module Details

### 5.1 `comparison/__init__.py`

```python
"""Comparison modules for baseline vs test TraceLens reports."""

from .combine import combine_excel_files
from .gpu_timeline_comparison import add_gpu_timeline_comparison
from .collective_comparison import add_collective_comparison
from .formatting import save_with_formatting

__all__ = [
    "combine_excel_files",
    "add_gpu_timeline_comparison",
    "add_collective_comparison",
    "save_with_formatting",
]
```

---

### 5.2 `comparison/combine.py`

```python
"""Shared functionality to combine two Excel files."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def combine_excel_files(
    baseline_path: Path,
    test_path: Path,
    baseline_label: str,
    test_label: str,
    sheets_to_combine: Optional[List[str]] = None,
    filter_summary_only: bool = False,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Combine two Excel files by adding a 'source' column.
    
    Args:
        baseline_path: Path to baseline Excel file
        test_path: Path to test Excel file
        baseline_label: Label for baseline rows in 'source' column
        test_label: Label for test rows in 'source' column
        sheets_to_combine: Specific sheets to combine (None = all common sheets)
        filter_summary_only: If True, only keep sheets with 'summary' in name
        verbose: Print progress messages
    
    Returns:
        Dict mapping sheet_name to combined DataFrame
    
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If no common sheets found
    """
```

**Implementation Notes:**
- Read both Excel files using `pd.ExcelFile`
- Find common sheets (intersection of sheet names)
- If `filter_summary_only`, filter to sheets containing "summary"
- For each sheet: add `source` column, concat, store in dict
- Return dict (don't save yet - let caller handle saving)

---

### 5.3 `comparison/gpu_timeline_comparison.py`

```python
"""GPU timeline comparison logic."""

from typing import Dict

import pandas as pd


def add_gpu_timeline_comparison(
    combined_data: Dict[str, pd.DataFrame],
    baseline_label: str,
    test_label: str,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Add comparison sheets for GPU timeline data.
    
    Args:
        combined_data: Dict from combine_excel_files()
        baseline_label: Label for baseline (for column naming)
        test_label: Label for test (for column naming)
        verbose: Print progress messages
    
    Returns:
        Dict with original sheets + new comparison sheets:
        - 'Comparison_By_Rank': Per-rank comparison
        - 'Summary_Comparison': Overall comparison
    
    Expects combined_data to have:
        - 'All_Ranks_Combined' sheet with: source, rank, type, time ms, percent
        - 'Summary' sheet with: source, type, time ms, percent
    
    Comparison columns created:
        - {baseline_label}_time_ms, {test_label}_time_ms
        - diff_time_ms, percent_change, status, ratio
        - {baseline_label}_percent, {test_label}_percent, diff_percent
    
    percent_change formula: (baseline - test) / baseline × 100
        - Positive = test is faster (better)
        - Negative = test is slower (worse)
    
    status thresholds:
        - "Better" if percent_change > 1
        - "Worse" if percent_change < -1
        - "Similar" otherwise
    """
```

**Implementation Notes:**
- Get actual source values from DataFrame (first = baseline, second = test)
- Create Comparison_By_Rank by iterating over ranks and types
- Create Summary_Comparison from Summary sheet
- Add to result dict and return

---

### 5.4 `comparison/collective_comparison.py`

```python
"""Collective/NCCL comparison logic."""

from typing import Dict

import pandas as pd


# Metrics to compare
NCCL_NUMERIC_COLS = [
    "comm_latency_mean",
    "algo bw (GB/s)_mean",
    "bus bw (GB/s)_mean",
    "Total comm latency (ms)",
    "count",
]

# Grouping columns for NCCL data
NCCL_GROUP_COLS = ["Collective name", "dtype", "In msg nelems"]


def add_collective_comparison(
    combined_data: Dict[str, pd.DataFrame],
    baseline_label: str,
    test_label: str,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Add comparison sheets for collective/NCCL data.
    
    Args:
        combined_data: Dict from combine_excel_files()
        baseline_label: Label for baseline
        test_label: Label for test
        verbose: Print progress messages
    
    Returns:
        Dict with summary sheets + new comparison sheets:
        - 'nccl_implicit_sync_cmp': Comparison for nccl_summary_implicit_sync
        - 'nccl_long_cmp': Comparison for nccl_summary_long
    
    Processes sheets:
        - 'nccl_summary_implicit_sync' → 'nccl_implicit_sync_cmp'
        - 'nccl_summary_long' → 'nccl_long_cmp'
    
    Groups by: ['Collective name', 'dtype', 'In msg nelems']
    
    For each metric, creates columns:
        - {baseline}_{metric}, {test}_{metric}
        - diff_{metric}, percent_change_{metric}, ratio_{metric}
    
    percent_change semantics (positive = better):
        - Latency/time: (baseline - test) / baseline × 100
        - Bandwidth: (test - baseline) / baseline × 100
    """
```

**Implementation Notes:**
- Only process sheets in `["nccl_summary_implicit_sync", "nccl_summary_long"]`
- Use flexible grouping (fall back to just "Collective name" if other cols missing)
- Apply correct percent_change formula based on metric type
- Sheet name transformation: `nccl_summary_X` → `nccl_X_cmp`

---

### 5.5 `comparison/formatting.py`

```python
"""Shared Excel formatting utilities."""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule


# Color constants
RED = "F8696B"
WHITE = "FFFFFF"
GREEN = "63BE7B"


def get_column_letter(col_idx: int) -> str:
    """
    Convert 1-based column index to Excel column letter.
    
    Examples:
        1 → 'A', 26 → 'Z', 27 → 'AA', 28 → 'AB'
    """


def create_color_scale_rule() -> ColorScaleRule:
    """
    Create standard red-white-green color scale rule.
    
    Red (min/negative) → White (0) → Green (max/positive)
    """
    return ColorScaleRule(
        start_type="min",
        start_color=RED,
        mid_type="num",
        mid_value=0,
        mid_color=WHITE,
        end_type="max",
        end_color=GREEN,
    )


def apply_color_scale_to_column(
    worksheet,
    col_idx: int,
    num_rows: int,
) -> None:
    """
    Apply color scale formatting to a specific column.
    
    Args:
        worksheet: openpyxl worksheet
        col_idx: 1-based column index
        num_rows: Number of data rows (excluding header)
    """


def save_with_formatting(
    data: Dict[str, pd.DataFrame],
    output_path: Path,
    format_columns: Dict[str, List[str]],
    verbose: bool = False,
) -> Path:
    """
    Save DataFrames to Excel with conditional formatting.
    
    Args:
        data: Dict[sheet_name, DataFrame]
        output_path: Output file path
        format_columns: Dict[sheet_name, list of column names to format]
        verbose: Print progress
    
    Returns:
        Path to saved file
    
    Example:
        format_columns = {
            "Comparison_By_Rank": ["percent_change"],
            "Summary_Comparison": ["percent_change"],
            "nccl_implicit_sync_cmp": [
                "percent_change_comm_latency_mean",
                "percent_change_algo bw (GB/s)_mean",
            ],
        }
    """
```

---

## 6. Data Flow

### 6.1 `compare gpu_timeline` Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        compare gpu_timeline                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT:                                                                     │
│  ├── baseline.xlsx (gpu_timeline_summary_mean.xlsx)                         │
│  │   ├── Summary                                                            │
│  │   ├── All_Ranks_Combined                                                 │
│  │   ├── Per_Rank_Time_ms                                                   │
│  │   └── Per_Rank_Percent                                                   │
│  │                                                                          │
│  └── test.xlsx (gpu_timeline_summary_mean.xlsx)                             │
│      └── (same sheets)                                                      │
│                                                                             │
│  STEP 1: combine_excel_files()                                              │
│  ────────────────────────────────                                           │
│  For each sheet, add 'source' column and concat:                            │
│      baseline rows → source = baseline_label                                │
│      test rows → source = test_label                                        │
│                                                                             │
│  STEP 2: add_gpu_timeline_comparison()                                      │
│  ────────────────────────────────────────                                   │
│  Create new sheets:                                                         │
│      Comparison_By_Rank: Per-rank comparison                                │
│      Summary_Comparison: Overall comparison                                 │
│                                                                             │
│  STEP 3: save_with_formatting()                                             │
│  ─────────────────────────────────                                          │
│  Save all sheets to Excel with color formatting on percent_change           │
│                                                                             │
│  OUTPUT:                                                                    │
│  └── output.xlsx                                                            │
│      ├── Summary (combined)                                                 │
│      ├── All_Ranks_Combined (combined)                                      │
│      ├── Per_Rank_Time_ms (combined)                                        │
│      ├── Per_Rank_Percent (combined)                                        │
│      ├── Comparison_By_Rank (NEW - with formatting)                         │
│      └── Summary_Comparison (NEW - with formatting)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 `compare collective` Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          compare collective                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT:                                                                     │
│  ├── baseline.xlsx (collective_all_ranks.xlsx)                              │
│  │   ├── nccl_summary_implicit_sync                                         │
│  │   ├── nccl_summary_long                                                  │
│  │   └── (other non-summary sheets - SKIPPED)                               │
│  │                                                                          │
│  └── test.xlsx (collective_all_ranks.xlsx)                                  │
│      └── (same sheets)                                                      │
│                                                                             │
│  STEP 1: combine_excel_files(filter_summary_only=True)                      │
│  ────────────────────────────────────────────────────────                   │
│  Only combine sheets with "summary" in name                                 │
│  Add 'source' column and concat                                             │
│                                                                             │
│  STEP 2: add_collective_comparison()                                        │
│  ──────────────────────────────────────                                     │
│  For each summary sheet, create comparison sheet:                           │
│      nccl_summary_implicit_sync → nccl_implicit_sync_cmp                    │
│      nccl_summary_long → nccl_long_cmp                                      │
│                                                                             │
│  STEP 3: save_with_formatting()                                             │
│  ─────────────────────────────────                                          │
│  Save with color formatting on all percent_change_* columns                 │
│                                                                             │
│  OUTPUT:                                                                    │
│  └── output.xlsx                                                            │
│      ├── nccl_summary_implicit_sync (combined)                              │
│      ├── nccl_summary_long (combined)                                       │
│      ├── nccl_implicit_sync_cmp (NEW - with formatting)                     │
│      └── nccl_long_cmp (NEW - with formatting)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Order

| Phase | Task | Est. Time | Dependencies |
|-------|------|-----------|--------------|
| **1** | Create `comparison/` directory and `__init__.py` | 5 min | None |
| **2** | Implement `formatting.py` | 25 min | Phase 1 |
| **3** | Implement `combine.py` | 20 min | Phase 1 |
| **4** | Implement `gpu_timeline_comparison.py` | 40 min | Phase 2, 3 |
| **5** | Implement `collective_comparison.py` | 40 min | Phase 2, 3 |
| **6** | Update `cli.py` with compare commands | 25 min | Phase 4, 5 |
| **7** | Testing | 30 min | Phase 6 |

**Total estimated time: ~3 hours**

---

## 8. Expected Output

### 8.1 `compare gpu_timeline` Console Output

```
============================================================
GPU Timeline Comparison
============================================================
Baseline: /path/to/56cu_256threads/tracelens_analysis/gpu_timeline_summary_mean.xlsx
Test: /path/to/37cu_384threads/tracelens_analysis/gpu_timeline_summary_mean.xlsx
Baseline label: 56cu_256threads
Test label: 37cu_384threads

Step 1: Combining Excel files
  Loading baseline (56cu_256threads)...
  Loading test (37cu_384threads)...
  Combining sheets:
    Summary: 10 + 10 = 20 rows
    All_Ranks_Combined: 80 + 80 = 160 rows
    Per_Rank_Time_ms: 10 + 10 = 20 rows
    Per_Rank_Percent: 10 + 10 = 20 rows

Step 2: Adding comparison sheets
  Creating Comparison_By_Rank...
    Processing 8 ranks × 10 types = 80 comparisons
  Creating Summary_Comparison...
    Processing 10 types

Step 3: Saving with formatting
  Applying color scale to Comparison_By_Rank.percent_change
  Applying color scale to Summary_Comparison.percent_change

============================================================
Comparison Complete!
============================================================
Output: /path/to/gpu_comparison.xlsx

Sheets:
  - Summary (combined data)
  - All_Ranks_Combined (combined data)
  - Per_Rank_Time_ms (combined data)
  - Per_Rank_Percent (combined data)
  - Comparison_By_Rank (per-rank comparison)
  - Summary_Comparison (overall comparison)

percent_change interpretation:
  Positive = test is faster/better
  Negative = test is slower/worse
```

### 8.2 `compare collective` Console Output

```
============================================================
Collective/NCCL Comparison
============================================================
Baseline: /path/to/56cu_256threads/tracelens_analysis/collective_reports/collective_all_ranks.xlsx
Test: /path/to/37cu_384threads/tracelens_analysis/collective_reports/collective_all_ranks.xlsx
Baseline label: 56cu_256threads
Test label: 37cu_384threads

Step 1: Combining Excel files
  Loading baseline (56cu_256threads)...
  Loading test (37cu_384threads)...
  Filtering to summary sheets only...
  Combining sheets:
    nccl_summary_implicit_sync: 15 + 15 = 30 rows
    nccl_summary_long: 15 + 15 = 30 rows
  Skipped sheets (non-summary):
    - per_rank_comm_details
    - raw_data

Step 2: Adding comparison sheets
  Processing nccl_summary_implicit_sync...
    Grouping by: ['Collective name', 'dtype', 'In msg nelems']
    Created nccl_implicit_sync_cmp (15 rows)
  Processing nccl_summary_long...
    Created nccl_long_cmp (15 rows)

Step 3: Saving with formatting
  Applying color scale to nccl_implicit_sync_cmp:
    - percent_change_comm_latency_mean
    - percent_change_algo bw (GB/s)_mean
    - percent_change_bus bw (GB/s)_mean
    - percent_change_Total comm latency (ms)
  Applying color scale to nccl_long_cmp:
    - (same columns)

============================================================
Comparison Complete!
============================================================
Output: /path/to/collective_comparison.xlsx

Sheets:
  - nccl_summary_implicit_sync (combined data)
  - nccl_summary_long (combined data)
  - nccl_implicit_sync_cmp (comparison)
  - nccl_long_cmp (comparison)

percent_change interpretation:
  For latency/time: Positive = faster (better)
  For bandwidth: Positive = higher bandwidth (better)
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/test_comparison/test_combine.py

def test_combine_excel_files_basic():
    """Test combining two Excel files adds source column."""

def test_combine_excel_files_filter_summary():
    """Test filter_summary_only option works."""

def test_combine_excel_files_missing_sheet():
    """Test handling when sheet only exists in one file."""


# tests/test_comparison/test_gpu_timeline.py

def test_add_gpu_timeline_comparison_creates_sheets():
    """Test that Comparison_By_Rank and Summary_Comparison are created."""

def test_percent_change_calculation():
    """Test percent_change formula is correct."""

def test_status_thresholds():
    """Test Better/Worse/Similar status logic."""


# tests/test_comparison/test_collective.py

def test_add_collective_comparison_creates_sheets():
    """Test comparison sheets are created for each summary sheet."""

def test_latency_percent_change():
    """Test latency metrics use (base-test)/base formula."""

def test_bandwidth_percent_change():
    """Test bandwidth metrics use (test-base)/base formula."""


# tests/test_comparison/test_formatting.py

def test_get_column_letter():
    """Test column index to letter conversion."""
    assert get_column_letter(1) == "A"
    assert get_column_letter(26) == "Z"
    assert get_column_letter(27) == "AA"

def test_color_scale_applied():
    """Test that color scale formatting is applied to correct columns."""
```

### 9.2 Integration Tests

```python
# tests/test_comparison/test_cli_integration.py

def test_compare_gpu_timeline_cli():
    """Test full CLI flow for gpu_timeline comparison."""

def test_compare_collective_cli():
    """Test full CLI flow for collective comparison."""

def test_label_extraction_from_path():
    """Test grandparent directory name extraction."""
```

---

## Appendix A: Label Extraction Logic

```python
def extract_label_from_path(file_path: Path) -> str:
    """
    Extract label from file path using grandparent directory name.
    
    Examples:
        /path/to/56cu_256threads/tracelens_analysis/gpu_timeline.xlsx
        → "56cu_256threads"
        
        /path/to/run1/tracelens_analysis/collective_reports/collective.xlsx
        → "run1" (or "tracelens_analysis" depending on depth)
    
    Fallback: "baseline" or "test" if extraction fails
    """
    try:
        # Go up to grandparent (skip filename and parent directory)
        grandparent = file_path.parent.parent.name
        if grandparent and grandparent not in [".", "..", ""]:
            return grandparent
    except:
        pass
    return None  # Let caller provide default
```

---

## Appendix B: CLI Help Text

```
$ aorta-report compare --help
Usage: aorta-report compare [OPTIONS] COMMAND [ARGS]...

  Compare baseline and test TraceLens reports.

  Supported comparison types:
    gpu_timeline  - Compare GPU timeline reports
    collective    - Compare collective/NCCL reports

Commands:
  collective    Compare two collective/NCCL reports.
  gpu_timeline  Compare two GPU timeline reports.


$ aorta-report compare gpu_timeline --help
Usage: aorta-report compare gpu_timeline [OPTIONS]

  Compare two GPU timeline reports.

  Combines baseline and test files, then adds comparison sheets with diff,
  percent_change, and status columns.

  Output sheets:
    - Summary, All_Ranks_Combined, Per_Rank_* (combined data)
    - Comparison_By_Rank (per-rank comparison)
    - Summary_Comparison (overall comparison)

  Examples:
    aorta-report compare gpu_timeline \
        -b baseline/gpu_timeline_summary_mean.xlsx \
        -t test/gpu_timeline_summary_mean.xlsx \
        -o comparison.xlsx

Options:
  -b, --baseline PATH      Path to baseline gpu_timeline_summary_mean.xlsx
                           [required]
  -t, --test PATH          Path to test gpu_timeline_summary_mean.xlsx
                           [required]
  --baseline-label TEXT    Label for baseline (default: grandparent dir name)
  --test-label TEXT        Label for test (default: grandparent dir name)
  -o, --output PATH        Output Excel file path  [required]
  --help                   Show this message and exit.
```

---

## Appendix C: Migration from Original Scripts

| Original | New CLI Equivalent |
|----------|-------------------|
| `python combine_reports.py --baseline b.xlsx --test t.xlsx --output combined.xlsx` | (intermediate step, now internal) |
| `python add_comparison_sheets.py --input combined.xlsx --output comparison.xlsx` | `aorta-report compare gpu_timeline -b b.xlsx -t t.xlsx -o comparison.xlsx` |
| `python add_collective_comparison.py --input combined.xlsx --output comparison.xlsx` | `aorta-report compare collective -b b.xlsx -t t.xlsx -o comparison.xlsx` |

The new CLI combines both steps (combine + add comparison) into a single command.

