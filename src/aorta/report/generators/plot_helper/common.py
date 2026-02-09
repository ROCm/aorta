"""Common utilities for plot generation."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Color Palette
# =============================================================================

COLORS = {
    "positive": "#2ecc71",    # Green - improvements
    "negative": "#e74c3c",    # Red - regressions
    "baseline": "#3498db",    # Blue - baseline data
    "test": "#e67e22",        # Orange - test data
    "neutral": "#95a5a6",     # Gray - neutral
}

# Extended palette for multi-series
PALETTE_MULTI = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c"]


# =============================================================================
# Plot Configuration
# =============================================================================

DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (10, 6)


def configure_style() -> None:
    """Configure matplotlib/seaborn style for consistent plots."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": DEFAULT_DPI,
        "savefig.dpi": DEFAULT_DPI,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def remove_spines(ax) -> None:
    """Remove all spines from an axis."""
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)


def save_figure(
    fig,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    close: bool = True,
) -> Path:
    """Save figure and optionally close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return output_path


def get_improvement_colors(values) -> List[str]:
    """Return green/red colors based on positive/negative values."""
    return [COLORS["positive"] if v > 0 else COLORS["negative"] for v in values]
