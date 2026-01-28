"""GEMM variance violin plot."""

from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt

from .common import DEFAULT_DPI, save_figure


def _prepare_violin_data(
    data_dict: Dict[int, List[float]],
    label_fmt: str,
) -> List[Dict[str, Any]]:
    """Prepare data for violin plot from a dictionary."""
    result = []
    for key, values in sorted(data_dict.items()):
        for val in values:
            result.append({"config": label_fmt.format(key), "time_diff": val})
    return result


def plot_variance_violin_combined(
    data: Dict[str, Any],
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Create combined violin plot (1x3 grid) for all dimensions."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    configs = [
        {
            "data": _prepare_violin_data(data["threads"], "{}t"),
            "sort_key": lambda x: int(x[:-1]),
            "color": "lightblue",
            "xlabel": "Threads",
            "title": "By Thread Count",
        },
        {
            "data": _prepare_violin_data(data["channels"], "{}ch"),
            "sort_key": lambda x: int(x[:-2]),
            "color": "lightcoral",
            "xlabel": "Channels",
            "title": "By Channel Count",
        },
        {
            "data": _prepare_violin_data(data["ranks"], "R{}"),
            "sort_key": lambda x: int(x[1:]),
            "color": "lightgreen",
            "xlabel": "Ranks",
            "title": "By Rank",
        },
    ]
    
    for ax, cfg in zip(axes, configs):
        violin_data = cfg["data"]
        if not violin_data:
            ax.set_visible(False)
            continue
        
        configs_list = sorted(
            set(d["config"] for d in violin_data),
            key=cfg["sort_key"],
        )
        values = [
            [d["time_diff"] for d in violin_data if d["config"] == c]
            for c in configs_list
        ]
        
        parts = ax.violinplot(
            values,
            positions=range(len(configs_list)),
            showmeans=True,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(cfg["color"])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(configs_list)))
        ax.set_xticklabels(configs_list)
        ax.set_ylabel("Time Difference (us)", fontsize=12, fontweight="bold")
        ax.set_xlabel(cfg["xlabel"], fontsize=12, fontweight="bold")
        ax.set_title(cfg["title"], fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    
    fig.suptitle(
        "GEMM Kernel Time Variance Distribution",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    
    plt.tight_layout()
    return save_figure(fig, output_dir / "variance_violin_combined.png", dpi)

