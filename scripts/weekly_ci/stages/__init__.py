"""
Pipeline stages for Weekly CI Kickoff.

This module exports all pipeline stage functions.
"""

from __future__ import annotations

from .analysis import (
    stage_compare_all_analysis,
    stage_cross_timestamp_comparison,
    stage_pairwise_comparison,
    stage_single_config_analysis,
)
from .build import (
    stage_build_rccl,
    stage_install_dependencies,
    verify_rccl_installation,
)
from .docker import stage_cleanup, stage_docker_setup
from .reporting import (
    generate_dashboard_entry,
    stage_generate_summary,
    stage_update_dashboard,
    update_dashboard_file,
)
from .convert import stage_convert_html_to_md
from .repository import (
    stage_checkout_aorta_report,
    stage_push_results,
)
from .test import (
    stage_find_baseline_experiment_dir,
    stage_find_experiment_dir,
    stage_run_performance_tests,
    validate_experiment_configs,
)
from .validate import stage_validate_environment

__all__ = [
    # Validation
    "stage_validate_environment",
    # Docker
    "stage_docker_setup",
    "stage_cleanup",
    # Build
    "stage_build_rccl",
    "stage_install_dependencies",
    "verify_rccl_installation",
    # Test
    "stage_run_performance_tests",
    "stage_find_experiment_dir",
    "stage_find_baseline_experiment_dir",
    "validate_experiment_configs",
    # Analysis
    "stage_single_config_analysis",
    "stage_pairwise_comparison",
    "stage_compare_all_analysis",
    "stage_cross_timestamp_comparison",
    # Convert
    "stage_convert_html_to_md",
    # Repository
    "stage_checkout_aorta_report",
    "stage_push_results",
    # Reporting
    "stage_generate_summary",
    "stage_update_dashboard",
    "generate_dashboard_entry",
    "update_dashboard_file",
]
