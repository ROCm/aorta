"""
HW Queue Eval Excel report generator.

Generates Excel reports for:
- Single run analysis (Mode A)
- Sweep analysis (Mode B)

Phase 4 will add comparison mode (Mode C).
"""

from pathlib import Path
<<<<<<< copilot/sub-pr-100-again
from typing import Dict, Any, List, Optional, Set, Tuple
=======
# TODO: Check after final merge - unused typing imports (Dict, Any, List, Optional, Tuple)
# may be needed once comparison mode (Mode C) is implemented in Phase 4.
from typing import Dict, Any, List, Optional, Tuple
>>>>>>> prosenj_cli_hq_eval_report_phase_3

import pandas as pd
from openpyxl import load_workbook
# TODO: Check after final merge - ColorScaleRule, Alignment, Border, Side are unused;
# intended for conditional formatting/styling to be applied in a future phase.
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# Import data classes from loader
from ..processing.hwqueue_loader import SingleRunData, SweepData
from .excel_report import sanitize_table_name, add_excel_table


# TODO: Check after final merge - RED, WHITE, GREEN, HEADER_FILL, HEADER_FONT are defined
# but not yet applied; intended for header/conditional formatting in a future phase.
# Color constants
RED = "F8696B"
WHITE = "FFFFFF"
GREEN = "63BE7B"
HEADER_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
HEADER_FONT = Font(color="FFFFFF", bold=True)


<<<<<<< copilot/sub-pr-100-again
def _sanitize_table_name(name: str) -> str:
    """Create valid Excel table name."""
    table_name = name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    if not table_name[0].isalpha():
        table_name = "Tbl_" + table_name
    return table_name[:255]


def _add_excel_table(
    worksheet,
    table_name: str,
    start_row: int = 1,
    warnings: Optional[List[str]] = None,
) -> bool:
    """Convert worksheet data to Excel table format."""
    max_row = worksheet.max_row
    max_col = worksheet.max_column

    if max_row <= start_row:
        return False

    # Ensure all column headers are strings
    for col_idx in range(1, max_col + 1):
        cell = worksheet.cell(row=start_row, column=col_idx)
        if cell.value is not None and not isinstance(cell.value, str):
            cell.value = str(cell.value)

    start_cell = f"A{start_row}"
    end_cell = f"{get_column_letter(max_col)}{max_row}"
    table_ref = f"{start_cell}:{end_cell}"

    try:
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
        return True
    except Exception as e:
        if warnings is not None:
            warnings.append(f"Could not create table {table_name}: {e}")
        return False


=======
>>>>>>> prosenj_cli_hq_eval_report_phase_3
def _apply_formatting(output_path: Path, verbose: bool = False) -> None:
    """Apply formatting to the Excel file after writing."""
    wb = load_workbook(output_path)
    table_warnings: List[str] = []
    used_table_names: Set[str] = set()

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]

        if ws.max_row <= 1:
            continue

<<<<<<< copilot/sub-pr-100-again
        # Create a table name that is unique workbook-wide
        base_name = _sanitize_table_name(f"HWQ_{sheet_name}")
        table_name = base_name
        counter = 1
        while table_name in used_table_names:
            suffix = f"_{counter}"
            table_name = (base_name[: 255 - len(suffix)] + suffix)
            counter += 1
        used_table_names.add(table_name)

        if _add_excel_table(ws, table_name, warnings=table_warnings):
=======
        # Create unique table name
        table_name = sanitize_table_name(f"HWQ_{sheet_name}")
        if add_excel_table(ws, table_name):
>>>>>>> prosenj_cli_hq_eval_report_phase_3
            if verbose:
                print(f"    Converted to table: {sheet_name}")

        # Auto-adjust column widths
        for col_idx in range(1, ws.max_column + 1):
            max_length = 0
            col_letter = get_column_letter(col_idx)
            for row in range(1, ws.max_row + 1):
                try:
                    cell_value = ws.cell(row=row, column=col_idx).value
                    if cell_value is not None:
                        max_length = max(max_length, len(str(cell_value)))
                except Exception:
                    pass
            ws.column_dimensions[col_letter].width = min(max_length + 2, 50)

    wb.save(output_path)

    if verbose and table_warnings:
        for warning in table_warnings:
            print(f"    Warning: {warning}")


def generate_single_run_excel(
    data: SingleRunData,
    output_path: Path,
    verbose: bool = False,
) -> Path:
    """
    Generate Excel report for single run analysis (Mode A).

    Args:
        data: SingleRunData object from loader
        output_path: Path to output Excel file

    Returns:
        Path to generated Excel file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Generating single run Excel: {output_path.name}")

    # Build summary data
    summary_data = {
        "Metric": [
            "Workload",
            "Stream Count",
            "Throughput",
            "Total Time (ms)",
            "Latency Mean (ms)",
            "Latency P50 (ms)",
            "Latency P95 (ms)",
            "Latency P99 (ms)",
            "Latency Min (ms)",
            "Latency Max (ms)",
            "Latency Std (ms)",
        ],
        "Value": [
            data.workload_name,
            data.stream_count,
            f"{data.throughput:.2f} {data.throughput_unit}",
            f"{data.total_time_ms:.3f}",
            f"{data.latency.mean:.3f}",
            f"{data.latency.p50:.3f}",
            f"{data.latency.p95:.3f}",
            f"{data.latency.p99:.3f}",
            f"{data.latency.min:.3f}",
            f"{data.latency.max:.3f}",
            f"{data.latency.std:.3f}",
        ],
    }

    # Add switch latency if available
    if data.switch_latency:
        summary_data["Metric"].extend([
            "Inter-Stream Gap (ms)",
            "Intra-Stream Gap (ms)",
            "Switch Overhead (ms)",
        ])
        summary_data["Value"].extend([
            f"{data.switch_latency.inter_stream_gap_ms:.3f}",
            f"{data.switch_latency.intra_stream_gap_ms:.3f}",
            f"{data.switch_latency.estimated_switch_overhead_ms:.3f}",
        ])

    # Add memory if available
    if data.memory:
        summary_data["Metric"].extend([
            "Peak Allocated (GB)",
            "Peak Reserved (GB)",
        ])
        summary_data["Value"].extend([
            f"{data.memory.peak_allocated_gb:.2f}",
            f"{data.memory.peak_reserved_gb:.2f}",
        ])

    summary_df = pd.DataFrame(summary_data)

    # Build per-stream data if available
    sheets = {"Summary": summary_df}

    if data.per_stream_times_ms:
        stream_data = {
            "Stream": list(range(len(data.per_stream_times_ms))),
            "Time (ms)": data.per_stream_times_ms,
        }
        sheets["Per_Stream_Times"] = pd.DataFrame(stream_data)

    # Write to Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Apply formatting
    _apply_formatting(output_path, verbose)

    if verbose:
        print(f"    Created {len(sheets)} sheet(s)")

    return output_path


def generate_sweep_excel(
    data: SweepData,
    output_path: Path,
    verbose: bool = False,
) -> Path:
    """
    Generate Excel report for sweep analysis (Mode B).

    Args:
        data: SweepData object from loader
        output_path: Path to output Excel file

    Returns:
        Path to generated Excel file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Generating sweep Excel: {output_path.name}")

    # Build summary sheet
    best_streams, best_throughput = data.get_best_throughput()

    summary_data = {
        "Metric": [
            "Workload",
            "Stream Configurations Tested",
            "Best Stream Count",
            "Peak Throughput",
            "Inflection Point",
        ],
        "Value": [
            data.workload_name,
            len(data.results),
            best_streams,
            f"{best_throughput:.2f}",
            data.analysis.inflection_point if data.analysis.inflection_point else "N/A",
        ],
    }

    # Add environment info if available
    if data.environment.hostname:
        summary_data["Metric"].extend([
            "Hostname",
            "GPU Count",
            "HIP Version",
            "PyTorch Version",
        ])
        summary_data["Value"].extend([
            data.environment.hostname,
            data.environment.gpu_count,
            data.environment.hip_version or "N/A",
            data.environment.torch_version or "N/A",
        ])

    summary_df = pd.DataFrame(summary_data)

    # Build scaling data sheet
    scaling_rows = []
    for result in data.results:
        row = {
            "Stream_Count": result.stream_count,
            "Throughput": result.throughput,
            "Throughput_Unit": result.throughput_unit,
            "Latency_Mean_ms": result.latency.mean,
            "Latency_P50_ms": result.latency.p50,
            "Latency_P95_ms": result.latency.p95,
            "Latency_P99_ms": result.latency.p99,
            "Total_Time_ms": result.total_time_ms,
        }

        # Add efficiency if available from analysis
        if data.analysis.stream_counts and result.stream_count in data.analysis.stream_counts:
            idx = data.analysis.stream_counts.index(result.stream_count)
            if idx < len(data.analysis.efficiencies):
                row["Efficiency"] = data.analysis.efficiencies[idx]

        # Add switch latency if available
        if result.switch_latency:
            row["Inter_Stream_Gap_ms"] = result.switch_latency.inter_stream_gap_ms
            row["Intra_Stream_Gap_ms"] = result.switch_latency.intra_stream_gap_ms
            row["Switch_Overhead_ms"] = result.switch_latency.estimated_switch_overhead_ms

        scaling_rows.append(row)

    scaling_df = pd.DataFrame(scaling_rows)

    # Sort by stream count
    if not scaling_df.empty:
        scaling_df = scaling_df.sort_values("Stream_Count").reset_index(drop=True)

    sheets = {
        "Summary": summary_df,
        "Scaling_Data": scaling_df,
    }

    # Write to Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Apply formatting
    _apply_formatting(output_path, verbose)

    if verbose:
        print(f"    Created {len(sheets)} sheet(s)")
        print(f"    Scaling data: {len(scaling_rows)} configurations")

    return output_path


def generate_hwqueue_excel(
    data: SingleRunData | SweepData,
    output_path: Path,
    verbose: bool = False,
) -> Path:
    """
    Generate Excel report for single run or sweep data.

    Automatically detects data type and calls the appropriate generator.

    Args:
        data: SingleRunData or SweepData object
        output_path: Path to output Excel file

    Returns:
        Path to generated Excel file
    """
    if isinstance(data, SweepData):
        return generate_sweep_excel(data, output_path, verbose)
    elif isinstance(data, SingleRunData):
        return generate_single_run_excel(data, output_path, verbose)
    else:
        raise ValueError(f"Unknown data type: {type(data)}")

