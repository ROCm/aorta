"""
Weekly CI Kickoff Package.

This package provides the standalone implementation of the
RCCL Warp Speed Performance Analysis CI workflow.
"""

from __future__ import annotations

from .config import Config, load_config_file, merge_config, parse_args
from .logging_setup import (
    log_stage_complete,
    log_stage_error,
    log_stage_skip,
    log_stage_start,
    setup_logging,
)
from .utils import (
    check_docker_exists,
    check_docker_running,
    docker_exec,
    find_latest_experiment_dir,
    find_second_latest_experiment_dir,
    get_config_dir_name,
    get_repo_root,
    parse_config_pairs,
    run_command,
)

__all__ = [
    # Config
    "Config",
    "load_config_file",
    "merge_config",
    "parse_args",
    # Logging
    "setup_logging",
    "log_stage_start",
    "log_stage_skip",
    "log_stage_complete",
    "log_stage_error",
    # Utils
    "run_command",
    "docker_exec",
    "get_repo_root",
    "check_docker_running",
    "check_docker_exists",
    "find_latest_experiment_dir",
    "find_second_latest_experiment_dir",
    "parse_config_pairs",
    "get_config_dir_name",
]

