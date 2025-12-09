#!/usr/bin/env python3
"""
Shared utilities for GPU timeline processing.

Common functions used by:
- gemm_analysis/process_gpu_timeline.py
- tracelens_single_config/process_gpu_timeline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Callable, Union


def geometric_mean(values):
    """
    Calculate geometric mean, handling zeros.

    Args:
        values: Array-like of numeric values

    Returns:
        float: Geometric mean of the values
    """
    values = np.array(values)
    values = np.where(values == 0, 1e-10, values)
    return np.exp(np.mean(np.log(values)))


def get_aggregation_func(use_geo_mean: bool) -> Union[Callable, str]:
    """
    Get the appropriate aggregation function.

    Args:
        use_geo_mean: If True, return geometric_mean function; otherwise "mean"

    Returns:
        Aggregation function or string for pandas
    """
    return geometric_mean if use_geo_mean else "mean"


def read_gpu_timeline_from_excel(file_path: Path, rank: int = None) -> Tuple[pd.DataFrame, bool]:
    """
    Read gpu_timeline sheet from an Excel file.

    Args:
        file_path: Path to the Excel file
        rank: Optional rank number to add as column

    Returns:
        Tuple of (DataFrame, success_bool)
    """
    try:
        df = pd.read_excel(file_path, sheet_name="gpu_timeline")
        if rank is not None:
            df["rank"] = rank
        return df, True
    except Exception as e:
        print(f"  Warning: Could not read {file_path.name}: {e}")
        return None, False


def aggregate_gpu_timeline(
    rank_data: List[pd.DataFrame], use_geo_mean: bool = False
) -> pd.DataFrame:
    """
    Aggregate GPU timeline data across multiple ranks.

    Args:
        rank_data: List of DataFrames with gpu_timeline data
        use_geo_mean: If True, use geometric mean; otherwise arithmetic mean

    Returns:
        DataFrame: Aggregated data grouped by 'type'
    """
    combined = pd.concat(rank_data, ignore_index=True)

    agg_func = get_aggregation_func(use_geo_mean)
    aggregated = (
        combined.groupby("type").agg({"time ms": agg_func, "percent": agg_func}).reset_index()
    )

    return aggregated


def print_section(title: str, char: str = "=", width: int = 80):
    """
    Print a formatted section header.

    Args:
        title: Section title to display
        char: Character to use for the separator line
        width: Width of the separator line
    """
    print(f"\n{char * width}")
    print(title)
    print(char * width)


def get_method_suffix(use_geo_mean: bool) -> str:
    """
    Get the file suffix based on aggregation method.

    Args:
        use_geo_mean: Whether geometric mean is used

    Returns:
        str: "geomean" or "mean"
    """
    return "geomean" if use_geo_mean else "mean"


def get_aggregation_description(use_geo_mean: bool) -> str:
    """
    Get a human-readable description of the aggregation method.

    Args:
        use_geo_mean: Whether geometric mean is used

    Returns:
        str: "Geometric Mean" or "Arithmetic Mean"
    """
    return "Geometric Mean" if use_geo_mean else "Arithmetic Mean"
