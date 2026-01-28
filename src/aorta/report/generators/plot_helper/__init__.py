"""Plot helper functions for summary and GEMM visualizations."""

from .common import configure_style, COLORS, save_figure, get_improvement_colors

# Summary plots
from .summary_dashboard import (
    get_labels_from_excel,
    plot_improvement_chart,
    plot_abs_time_comparison,
)
from .gpu_by_rank import plot_gpu_metrics_by_rank
from .gpu_percent_change import plot_gpu_percent_change_grid
from .gpu_heatmap import plot_gpu_heatmap
from .nccl_charts import plot_nccl_comparison, plot_nccl_percent_change

# GEMM plots
from .gemm_data import read_gemm_csv_data, print_gemm_statistics
from .gemm_boxplots import (
    plot_variance_by_threads,
    plot_variance_by_channels,
    plot_variance_by_ranks,
)
from .gemm_violin import plot_variance_violin_combined
from .gemm_interaction import plot_thread_channel_interaction

__all__ = [
    # Common
    "configure_style",
    "COLORS",
    "save_figure",
    "get_improvement_colors",
    # Summary
    "get_labels_from_excel",
    "plot_improvement_chart",
    "plot_abs_time_comparison",
    "plot_gpu_metrics_by_rank",
    "plot_gpu_percent_change_grid",
    "plot_gpu_heatmap",
    "plot_nccl_comparison",
    "plot_nccl_percent_change",
    # GEMM
    "read_gemm_csv_data",
    "print_gemm_statistics",
    "plot_variance_by_threads",
    "plot_variance_by_channels",
    "plot_variance_by_ranks",
    "plot_variance_violin_combined",
    "plot_thread_channel_interaction",
]

