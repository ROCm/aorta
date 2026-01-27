# `generate excel` Command - Developer Documentation

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

---

## 1. Overview

The `generate excel` command creates a comprehensive final report by combining GPU timeline and collective comparison data into a single, well-organized Excel file.

### Key Features

| Feature | Description |
|---------|-------------|
| **Summary Dashboard** | First visible sheet with key metrics and status |
| **Comparison Sheets** | Visible sheets with comparison data |
| **Hidden Raw Data** | Original data hidden but accessible |
| **Excel Tables** | All data formatted as tables with filters |
| **Color Coding** | Red-white-green scale on percent_change columns |

### Source Script

**Location:** `scripts/tracelens_single_config/create_final_report.py` (346 lines)

---

## 2. Command Specification

### Current CLI (Stub)

```bash
aorta-report generate excel \
    --gpu-combined gpu_timeline_combined.xlsx \
    --gpu-comparison gpu_timeline_comparison.xlsx \
    --coll-combined collective_combined.xlsx \
    --coll-comparison collective_comparison.xlsx \
    --baseline-label "ROCm 6.0" \
    --test-label "ROCm 7.0" \
    -o final_analysis_report.xlsx
```

### Arguments

| Option | Required | Description |
|--------|----------|-------------|
| `--gpu-combined` | Yes | GPU timeline combined file (output of `compare gpu_timeline` without comparison sheets) |
| `--gpu-comparison` | Yes | GPU timeline comparison file (output of `compare gpu_timeline`) |
| `--coll-combined` | Yes | Collective combined file (intermediate) |
| `--coll-comparison` | Yes | Collective comparison file (output of `compare collective`) |
| `--baseline-label` | No | Label for baseline (default: "Baseline") |
| `--test-label` | No | Label for test (default: "Test") |
| `-o, --output` | Yes | Output Excel file path |

### Alternative Simplified Interface

Since `compare gpu_timeline` and `compare collective` now produce combined comparison files directly, we could simplify:

```bash
aorta-report generate excel \
    --gpu-comparison gpu_comparison.xlsx \
    --coll-comparison collective_comparison.xlsx \
    --baseline-label "ROCm 6.0" \
    --test-label "ROCm 7.0" \
    -o final_report.xlsx
```

**Decision:** Keep original 4-file interface for now. Can refactor later.

---

## 3. Source Script Analysis

### 3.1 Input Files

The script requires 4 Excel files:

| File | Contents | Source |
|------|----------|--------|
| `gpu_combined` | GPU summary + raw data with source column | `combine_reports.py` |
| `gpu_comparison` | GPU comparison sheets (Summary_Comparison, Comparison_By_Rank) | `add_comparison_sheets.py` |
| `coll_combined` | NCCL summary data with source column | `combine_reports.py` |
| `coll_comparison` | NCCL comparison sheets (nccl_*_cmp) | `add_collective_comparison.py` |

### 3.2 Processing Steps

```python
def create_final_report(gpu_combined, gpu_comparison, coll_combined, coll_comparison, output_file):
    # 1. Create workbook and add sheets
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        
        # 2. Add GPU Timeline sheets (raw → hidden)
        for sheet in gpu_combined:
            rename_and_add(sheet, "GPU_*_Raw")
            mark_as_hidden(sheet)
        
        # 3. Add GPU Comparison sheets (visible)
        for sheet in gpu_comparison:
            if "Comparison" in sheet:
                rename_and_add(sheet, "GPU_*_Cmp")
        
        # 4. Add Collective sheets (raw → hidden)
        for sheet in coll_combined:
            if "summary" in sheet:
                rename_and_add(sheet, "NCCL_*_Raw")
                mark_as_hidden(sheet)
        
        # 5. Add Collective Comparison sheets (visible)
        for sheet in coll_comparison:
            rename_and_add(sheet, "NCCL_*")
        
        # 6. Create Summary Dashboard
        create_dashboard_from_gpu_comparison()
    
    # 7. Post-processing with openpyxl
    wb = load_workbook(output_file)
    
    # 8. Hide raw data sheets
    for sheet in raw_sheets:
        wb[sheet].sheet_state = "hidden"
    
    # 9. Convert all sheets to Excel tables
    for sheet in wb.sheetnames:
        add_excel_table(sheet)
    
    # 10. Add conditional formatting to comparison sheets
    for sheet in comparison_sheets:
        apply_color_scale_to_percent_change_columns(sheet)
    
    # 11. Move Summary_Dashboard to first position
    wb.move_sheet("Summary_Dashboard", offset=-(len(wb.sheetnames)-1))
    
    wb.save(output_file)
```

### 3.3 Sheet Naming Convention

| Original Sheet | Final Name | Visibility |
|----------------|------------|------------|
| Summary | GPU_Summary_Raw | Hidden |
| All_Ranks_Combined | GPU_AllRanks_Raw | Hidden |
| Per_Rank_Time_ms | GPU_Time_Raw | Hidden |
| Per_Rank_Percent | GPU_Pct_Raw | Hidden |
| Summary_Comparison | GPU_Summary_Cmp | Visible |
| Comparison_By_Rank | GPU_ByRank_Cmp | Visible |
| nccl_summary_implicit_sync | NCCL_ImplSync_Raw | Hidden |
| nccl_summary_long | NCCL_Long_Raw | Hidden |
| nccl_implicit_sync_cmp | NCCL_Implicit_sync_cmp | Visible |
| nccl_long_cmp | NCCL_Long_cmp | Visible |
| (generated) | Summary_Dashboard | Visible (1st) |

### 3.4 Summary Dashboard Creation

**Decision:** Include BOTH GPU and NCCL metrics in the Summary Dashboard.

```python
dashboard_data = {
    'Metric': [],
    baseline_label: [],
    test_label: [],
    'Improvement (%)': [],
    'Status': []
}

# For each GPU metric type (busy_time, idle_time, etc.):
for row in gpu_summary_comparison:
    dashboard_data['Metric'].append(f"GPU_{row['type']}")
    dashboard_data[baseline_label].append(row[baseline_time_col])
    dashboard_data[test_label].append(row[test_time_col])
    dashboard_data['Improvement (%)'].append(row['percent_change'])
    dashboard_data['Status'].append('Better' if pct > 0 else 'Worse' if pct < -1 else 'Similar')

# Add NCCL metrics from collective comparison
for sheet in ['nccl_implicit_sync_cmp', 'nccl_long_cmp']:
    # Add latency and bandwidth metrics
    for row in coll_comparison[sheet]:
        # Add total comm latency metric
        dashboard_data['Metric'].append(f"NCCL_{collective_name}_latency")
        # ... add values
```

### 3.5 Excel Table Formatting

```python
def add_excel_table(worksheet, table_name):
    # Create table reference: A1:Z100
    table_ref = f"A1:{get_column_letter(max_col)}{max_row}"
    
    tab = Table(displayName=table_name, ref=table_ref)
    style = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    tab.tableStyleInfo = style
    worksheet.add_table(tab)
```

### 3.6 Conditional Formatting

Applied to columns with "percent_change" in the header:

```python
ColorScaleRule(
    start_type="min", start_color="F8696B",   # Red
    mid_type="num", mid_value=0, mid_color="FFFFFF",  # White
    end_type="max", end_color="63BE7B",       # Green
)
```

---

## 4. Implementation Architecture

### 4.1 File Structure

```
src/aorta/report/
├── generators/                      # Existing
│   ├── __init__.py                  # Add export
│   ├── html_generator.py            # Existing
│   ├── sweep_comparison.py          # Existing
│   ├── performance_report.py        # Existing
│   └── excel_report.py              # NEW: Final Excel report generator
└── cli.py                           # Update generate excel command
```

### 4.2 Relationship with Existing Modules

The `generate excel` command will use:
- `comparison/formatting.py` - For color scale formatting (already implemented)
- Excel table creation - New utility functions

### 4.3 Simplification Consideration

Since `compare gpu_timeline` and `compare collective` now produce files with BOTH combined data AND comparison sheets, we could:

**Option A: Keep current interface (4 files)**
- Matches original script exactly
- More flexible but verbose

**Option B: Simplified interface (2 files)**
- Only needs comparison files (they contain combined data too)
- Cleaner CLI but may need to extract raw data from comparison files

**Recommendation:** Option B with backward compatibility for Option A

---

## 5. Module Details

### 5.1 `generators/excel_report.py`

```python
"""Final Excel report generator.

Creates comprehensive report with:
- Summary Dashboard (first, visible)
- Comparison sheets (visible)
- Raw data sheets (hidden)
- Excel table formatting
- Color-coded percent_change columns
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo

from ..comparison.formatting import get_column_letter, create_color_scale_rule


# Sheet naming mappings
GPU_SHEET_MAPPING = {
    "Summary": "GPU_Summary_Raw",
    "All_Ranks_Combined": "GPU_AllRanks_Raw",
    "Per_Rank_Time_ms": "GPU_Time_Raw",
    "Per_Rank_Percent": "GPU_Pct_Raw",
}

GPU_COMPARISON_MAPPING = {
    "Summary_Comparison": "GPU_Summary_Cmp",
    "Comparison_By_Rank": "GPU_ByRank_Cmp",
}

COLL_SHEET_MAPPING = {
    "nccl_summary_implicit_sync": "NCCL_ImplSync_Raw",
    "nccl_summary_long": "NCCL_Long_Raw",
}


def create_final_excel_report(
    gpu_comparison_path: Path,
    coll_comparison_path: Path,
    output_path: Path,
    baseline_label: str = "Baseline",
    test_label: str = "Test",
    gpu_combined_path: Optional[Path] = None,
    coll_combined_path: Optional[Path] = None,
    verbose: bool = False,
) -> Path:
    """
    Create comprehensive final Excel report.
    
    Args:
        gpu_comparison_path: Path to GPU comparison file
        coll_comparison_path: Path to collective comparison file
        output_path: Output path for final report
        baseline_label: Label for baseline column
        test_label: Label for test column
        gpu_combined_path: Optional separate GPU combined file
        coll_combined_path: Optional separate collective combined file
        verbose: Print progress
    
    Returns:
        Path to created report
    """


def _add_gpu_sheets(
    writer: pd.ExcelWriter,
    gpu_comparison_path: Path,
    gpu_combined_path: Optional[Path],
    verbose: bool,
) -> Tuple[List[str], List[str]]:
    """Add GPU timeline sheets, return (raw_sheets, comparison_sheets)."""


def _add_collective_sheets(
    writer: pd.ExcelWriter,
    coll_comparison_path: Path,
    coll_combined_path: Optional[Path],
    verbose: bool,
) -> Tuple[List[str], List[str]]:
    """Add collective sheets, return (raw_sheets, comparison_sheets)."""


def _create_summary_dashboard(
    writer: pd.ExcelWriter,
    gpu_comparison_path: Path,
    baseline_label: str,
    test_label: str,
    verbose: bool,
) -> str:
    """Create Summary_Dashboard sheet, return sheet name."""


def _apply_post_processing(
    output_path: Path,
    raw_sheets: List[str],
    comparison_sheets: List[str],
    verbose: bool,
) -> None:
    """Apply Excel formatting: hide sheets, add tables, color formatting."""


def add_excel_table(worksheet, table_name: str, start_row: int = 1) -> None:
    """Convert worksheet data to Excel table format."""


def _sanitize_table_name(sheet_name: str) -> str:
    """Create valid Excel table name from sheet name."""
```

### 5.2 Updated `generators/__init__.py`

```python
"""Report generators for HTML and Excel."""

from .html_generator import generate_html, image_to_base64
from .excel_report import create_final_excel_report

__all__ = [
    "generate_html",
    "image_to_base64",
    "create_final_excel_report",
]
```

---

## 6. Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          generate excel                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS:                                                                    │
│  ├── gpu_comparison.xlsx                                                    │
│  │   ├── Summary (combined)                                                 │
│  │   ├── All_Ranks_Combined (combined)                                      │
│  │   ├── Per_Rank_Time_ms (combined)                                        │
│  │   ├── Per_Rank_Percent (combined)                                        │
│  │   ├── Comparison_By_Rank                                                 │
│  │   └── Summary_Comparison                                                 │
│  │                                                                          │
│  └── collective_comparison.xlsx                                             │
│      ├── nccl_summary_implicit_sync (combined)                              │
│      ├── nccl_summary_long (combined)                                       │
│      ├── nccl_implicit_sync_cmp                                             │
│      └── nccl_long_cmp                                                      │
│                                                                             │
│  PROCESSING:                                                                │
│  ────────────                                                               │
│  1. Read all sheets from input files                                        │
│  2. Rename sheets according to naming convention                            │
│  3. Create Summary_Dashboard from GPU comparison data                       │
│  4. Write all sheets to new workbook                                        │
│  5. Post-process with openpyxl:                                             │
│     - Hide raw data sheets                                                  │
│     - Convert to Excel tables                                               │
│     - Apply color formatting                                                │
│     - Move Summary_Dashboard to first position                              │
│                                                                             │
│  OUTPUT:                                                                    │
│  └── final_analysis_report.xlsx                                             │
│      ├── Summary_Dashboard (visible, FIRST)                                 │
│      ├── GPU_Summary_Cmp (visible)                                          │
│      ├── GPU_ByRank_Cmp (visible)                                           │
│      ├── NCCL_Implicit_sync_cmp (visible)                                   │
│      ├── NCCL_Long_cmp (visible)                                            │
│      ├── GPU_Summary_Raw (HIDDEN)                                           │
│      ├── GPU_AllRanks_Raw (HIDDEN)                                          │
│      ├── GPU_Time_Raw (HIDDEN)                                              │
│      ├── GPU_Pct_Raw (HIDDEN)                                               │
│      ├── NCCL_ImplSync_Raw (HIDDEN)                                         │
│      └── NCCL_Long_Raw (HIDDEN)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Order

| Phase | Task | Est. Time |
|-------|------|-----------|
| **1** | Create `generators/excel_report.py` with core functions | 45 min |
| **2** | Implement `_add_gpu_sheets()` | 20 min |
| **3** | Implement `_add_collective_sheets()` | 20 min |
| **4** | Implement `_create_summary_dashboard()` | 25 min |
| **5** | Implement `_apply_post_processing()` | 30 min |
| **6** | Update `generators/__init__.py` | 5 min |
| **7** | Update CLI command in `cli.py` | 15 min |
| **8** | Testing | 20 min |

**Total estimated time: ~3 hours**

---

## 8. Expected Output

### Console Output

```
============================================================
Creating Final Excel Report
============================================================
Output: final_analysis_report.xlsx
Baseline label: ROCm 6.0
Test label: ROCm 7.0

Step 1: Adding GPU Timeline sheets
  Added GPU_Summary_Raw (will be hidden)
  Added GPU_AllRanks_Raw (will be hidden)
  Added GPU_Time_Raw (will be hidden)
  Added GPU_Pct_Raw (will be hidden)
  Added GPU_Summary_Cmp
  Added GPU_ByRank_Cmp

Step 2: Adding Collective/NCCL sheets
  Added NCCL_ImplSync_Raw (will be hidden)
  Added NCCL_Long_Raw (will be hidden)
  Added NCCL_Implicit_sync_cmp
  Added NCCL_Long_cmp

Step 3: Creating Summary Dashboard
  Added Summary_Dashboard

Step 4: Applying formatting
  Hidden: GPU_Summary_Raw
  Hidden: GPU_AllRanks_Raw
  Hidden: GPU_Time_Raw
  Hidden: GPU_Pct_Raw
  Hidden: NCCL_ImplSync_Raw
  Hidden: NCCL_Long_Raw
  Converted to table: Summary_Dashboard
  Converted to table: GPU_Summary_Cmp
  Converted to table: GPU_ByRank_Cmp
  ...
  Applied color scale to GPU_Summary_Cmp column percent_change
  Applied color scale to GPU_ByRank_Cmp column percent_change
  ...
  Moved Summary_Dashboard to first position

============================================================
Report Complete!
============================================================
Output: final_analysis_report.xlsx

Report Structure:
  Visible Sheets (Analysis):
    - Summary_Dashboard
    - GPU_Summary_Cmp
    - GPU_ByRank_Cmp
    - NCCL_Implicit_sync_cmp
    - NCCL_Long_cmp

  Hidden Sheets (Raw Data):
    - GPU_Summary_Raw
    - GPU_AllRanks_Raw
    - GPU_Time_Raw
    - GPU_Pct_Raw
    - NCCL_ImplSync_Raw
    - NCCL_Long_Raw

Features:
  - All data formatted as Excel tables with filters
  - Percent change columns are color-coded (green=better, red=worse)
  - Unhide raw data: Right-click sheet tab → Unhide
```

### Summary Dashboard Content

| Metric | ROCm 6.0 | ROCm 7.0 | Improvement (%) | Status |
|--------|----------|----------|-----------------|--------|
| GPU_busy_time | 125.45 | 118.32 | 5.68 | Better |
| GPU_idle_time | 21.78 | 19.45 | 10.70 | Better |
| GPU_computation_time | 98.34 | 95.12 | 3.27 | Better |
| GPU_exposed_comm_time | 27.11 | 23.20 | 14.42 | Better |
| GPU_total_time | 147.23 | 137.77 | 6.43 | Better |

---

## Appendix A: CLI Update

### Simplified Interface (Recommended)

```python
@generate.command("excel")
@click.option("--gpu-comparison", required=True, type=click.Path(exists=True),
              help="GPU timeline comparison file (from 'compare gpu_timeline')")
@click.option("--coll-comparison", required=True, type=click.Path(exists=True),
              help="Collective comparison file (from 'compare collective')")
@click.option("--baseline-label", default="Baseline",
              help="Label for baseline configuration")
@click.option("--test-label", default="Test",
              help="Label for test configuration")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output Excel file path")
@click.pass_context
def generate_excel(ctx, gpu_comparison, coll_comparison, baseline_label, test_label, output):
    """Generate comprehensive final Excel report.
    
    Combines GPU timeline and collective comparison data into a single
    well-organized Excel report with:
    
    \b
    - Summary Dashboard (first sheet, key metrics at a glance)
    - Comparison sheets (visible, with color-coded changes)
    - Raw data sheets (hidden, accessible via Unhide)
    - Excel table formatting with filters
    
    \b
    Examples:
      aorta-report generate excel \\
          --gpu-comparison gpu_comparison.xlsx \\
          --coll-comparison collective_comparison.xlsx \\
          -o final_report.xlsx
    """
```

### Full Interface (Backward Compatible)

```python
@generate.command("excel")
@click.option("--gpu-comparison", required=True, type=click.Path(exists=True))
@click.option("--coll-comparison", required=True, type=click.Path(exists=True))
@click.option("--gpu-combined", type=click.Path(exists=True),
              help="Optional: Separate GPU combined file")
@click.option("--coll-combined", type=click.Path(exists=True),
              help="Optional: Separate collective combined file")
@click.option("--baseline-label", default="Baseline")
@click.option("--test-label", default="Test")
@click.option("-o", "--output", required=True, type=click.Path())
```

---

## Appendix B: Design Decisions

1. **Interface:** Keep original 4-file interface (can refactor later)

2. **Dashboard Metrics:** Include both GPU and NCCL metrics in Summary Dashboard

3. **Table Style:** Use `TableStyleMedium2` (standard)

4. **Sheet Order:** Dashboard → GPU Comparison → NCCL Comparison → (hidden)

