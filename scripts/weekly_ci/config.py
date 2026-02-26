"""
Configuration management for Weekly CI Kickoff.

This module provides:
- Dataclasses for all configuration sections
- YAML config file loading
- CLI argument parsing
- Config merging with CLI precedence
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# =============================================================================
# Configuration Dataclasses
# =============================================================================
@dataclass
class RCCLConfig:
    """RCCL build configuration."""

    branch: str = "warp_speed_v1"
    gpu_target: str = "gfx950"


@dataclass
class TestConfig:
    """Test configuration."""

    config_pairs: str = "56,256 37,384 32,512"
    baseline: str = "56,256"
    training_config: str = "config/single_node/gemm_overlap_comm.yaml"
    experiment_dir: str = ""  # If empty, auto-detect most recent


@dataclass
class AnalysisConfig:
    """Analysis configuration for aorta-report commands."""

    baseline_label: str = ""  # Label for baseline in reports (e.g., "baseline", "v1.0")
    test_label: str = ""  # Label for test in reports (e.g., "test", "v1.1")
    report_label: str = ""  # Override for aorta-report dir & dashboard (default: date from experiment)
    skip_tracelens_single_config: bool = False  # CLI --skip-tracelens: add to single-config only


@dataclass
class DockerConfig:
    """Docker configuration."""

    compose_file: str = "docker/rccl_test/docker-compose.rocm70_9-1.yaml"
    container_name: str = "training-overlap-bugs-rocm70_9-1"
    registry_user: str = ""  # Docker registry username (e.g., rocmshared)
    registry_password: str = ""  # Docker registry password/token
    skip_build: bool = True  # Skip docker build by default (use existing image)
    force_restart: bool = False  # If False, reuse running container; if True, always restart


@dataclass
class SkipConfig:
    """Stage skip configuration."""

    docker_setup: bool = False
    rccl_build: bool = False
    install_deps: bool = False
    performance_tests: bool = False
    single_config_analysis: bool = False
    pairwise_comparison: bool = False
    compare_all_analysis: bool = True  # Skip by default for initial setup
    checkout_aorta_report: bool = False  # Needed for cross-timestamp comparison
    cross_timestamp_comparison: bool = False
    convert_html_to_md: bool = False  # Convert HTML to Markdown before push (skip to disable)
    push_results: bool = True  # Skip by default
    cleanup: bool = True  # Skip by default (leave container running)


@dataclass
class CrossTimestampConfig:
    """Cross-timestamp comparison configuration."""

    baseline_experiment: str = ""  # If empty, auto-detect second-most-recent
    baseline_date: str = ""  # Date directory in aorta-report (e.g., "2026-02-19")
    aorta_report_path: str = "../aorta-report"


@dataclass
class GitConfig:
    """Git configuration for pushing results."""

    user_name: str = "Weekly CI Bot"
    user_email: str = "weekly-ci@aorta.local"
    github_token: str = ""  # Can be set via AORTA_REPORT_GITHUB_TOKEN env var


@dataclass
class OutputConfig:
    """Output configuration."""

    log_dir: str = "logs"
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration container."""

    rccl: RCCLConfig = field(default_factory=RCCLConfig)
    test: TestConfig = field(default_factory=TestConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    skip: SkipConfig = field(default_factory=SkipConfig)
    cross_timestamp: CrossTimestampConfig = field(default_factory=CrossTimestampConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    git: GitConfig = field(default_factory=GitConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Runtime state (populated during execution)
    experiment_dir: Optional[str] = None
    baseline_experiment_dir: Optional[str] = None  # For cross-timestamp comparison
    aorta_report_dir: Optional[Path] = None  # Path to cloned aorta-report


# =============================================================================
# Configuration Loading
# =============================================================================
def load_config_file(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary with configuration values, or empty dict if file not found.
    """
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _get_value(cli_value, yaml_data: dict, yaml_path: list, default):
    """Get value with precedence: CLI > YAML > default.

    Args:
        cli_value: Value from CLI argument (None if not provided).
        yaml_data: Parsed YAML configuration dictionary.
        yaml_path: List of keys to traverse in YAML (e.g., ['rccl', 'branch']).
        default: Default value if not found in CLI or YAML.

    Returns:
        The resolved value with proper precedence.
    """
    # CLI has highest precedence
    if cli_value is not None:
        return cli_value

    # Then YAML
    current = yaml_data
    for key in yaml_path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current if current is not None else default


def merge_config(config: Config, yaml_data: dict, args: argparse.Namespace) -> Config:
    """Merge YAML config and CLI args into Config object.

    CLI arguments always take precedence over YAML values.

    Args:
        config: Base Config object with defaults.
        yaml_data: Parsed YAML configuration dictionary.
        args: Parsed CLI arguments.

    Returns:
        Updated Config object with merged values.
    """
    # RCCL config
    config.rccl.branch = _get_value(
        args.rccl_branch, yaml_data, ["rccl", "branch"], config.rccl.branch
    )
    config.rccl.gpu_target = _get_value(
        args.gpu_target, yaml_data, ["rccl", "gpu_target"], config.rccl.gpu_target
    )

    # Test config
    config.test.config_pairs = _get_value(
        args.config_pairs, yaml_data, ["test", "config_pairs"], config.test.config_pairs
    )
    config.test.baseline = _get_value(
        args.baseline, yaml_data, ["test", "baseline"], config.test.baseline
    )
    config.test.training_config = _get_value(
        args.training_config,
        yaml_data,
        ["test", "training_config"],
        config.test.training_config,
    )
    config.test.experiment_dir = _get_value(
        getattr(args, "experiment_dir", None),
        yaml_data,
        ["test", "experiment_dir"],
        config.test.experiment_dir,
    )

    # Docker config
    config.docker.compose_file = _get_value(
        args.compose_file,
        yaml_data,
        ["docker", "compose_file"],
        config.docker.compose_file,
    )
    config.docker.container_name = _get_value(
        args.container_name,
        yaml_data,
        ["docker", "container_name"],
        config.docker.container_name,
    )
    config.docker.registry_user = _get_value(
        args.docker_user,
        yaml_data,
        ["docker", "registry_user"],
        config.docker.registry_user,
    )
    config.docker.registry_password = _get_value(
        args.docker_password,
        yaml_data,
        ["docker", "registry_password"],
        config.docker.registry_password,
    )

    # Docker build: --docker-build enables it, --no-docker-build disables it
    # Default is to skip build (skip_build=True)
    if args.docker_build:
        config.docker.skip_build = False
    elif args.no_docker_build:
        config.docker.skip_build = True
    else:
        config.docker.skip_build = _get_value(
            None,
            yaml_data,
            ["docker", "skip_build"],
            config.docker.skip_build,
        )

    # Force restart: --force-restart forces container restart even if running
    if args.force_restart:
        config.docker.force_restart = True
    else:
        config.docker.force_restart = _get_value(
            None,
            yaml_data,
            ["docker", "force_restart"],
            config.docker.force_restart,
        )

    # Skip config - CLI flags override YAML
    config.skip.docker_setup = _get_value(
        args.skip_docker_setup if args.skip_docker_setup else None,
        yaml_data,
        ["skip", "docker_setup"],
        config.skip.docker_setup,
    )
    config.skip.rccl_build = _get_value(
        args.skip_rccl_build if args.skip_rccl_build else None,
        yaml_data,
        ["skip", "rccl_build"],
        config.skip.rccl_build,
    )
    config.skip.install_deps = _get_value(
        args.skip_install_deps if args.skip_install_deps else None,
        yaml_data,
        ["skip", "install_deps"],
        config.skip.install_deps,
    )
    config.skip.performance_tests = _get_value(
        args.skip_performance_tests if args.skip_performance_tests else None,
        yaml_data,
        ["skip", "performance_tests"],
        config.skip.performance_tests,
    )
    config.skip.single_config_analysis = _get_value(
        args.skip_single_config_analysis if args.skip_single_config_analysis else None,
        yaml_data,
        ["skip", "single_config_analysis"],
        config.skip.single_config_analysis,
    )
    config.skip.pairwise_comparison = _get_value(
        args.skip_pairwise_comparison if args.skip_pairwise_comparison else None,
        yaml_data,
        ["skip", "pairwise_comparison"],
        config.skip.pairwise_comparison,
    )

    # Handle compare_all_analysis: --no-skip-compare-all enables it
    if args.no_skip_compare_all:
        config.skip.compare_all_analysis = False
    elif args.skip_compare_all:
        config.skip.compare_all_analysis = True
    else:
        config.skip.compare_all_analysis = _get_value(
            None,
            yaml_data,
            ["skip", "compare_all_analysis"],
            config.skip.compare_all_analysis,
        )

    config.skip.checkout_aorta_report = _get_value(
        args.skip_checkout_aorta_report if args.skip_checkout_aorta_report else None,
        yaml_data,
        ["skip", "checkout_aorta_report"],
        config.skip.checkout_aorta_report,
    )
    config.skip.cross_timestamp_comparison = _get_value(
        args.skip_cross_timestamp if args.skip_cross_timestamp else None,
        yaml_data,
        ["skip", "cross_timestamp_comparison"],
        config.skip.cross_timestamp_comparison,
    )
    config.skip.convert_html_to_md = _get_value(
        args.skip_convert_html_to_md if args.skip_convert_html_to_md else None,
        yaml_data,
        ["skip", "convert_html_to_md"],
        config.skip.convert_html_to_md,
    )
    config.skip.push_results = _get_value(
        args.skip_push if args.skip_push else None,
        yaml_data,
        ["skip", "push_results"],
        config.skip.push_results,
    )

    # Cleanup is inverted: --cleanup means DO cleanup (skip.cleanup = False)
    if args.cleanup:
        config.skip.cleanup = False
    else:
        config.skip.cleanup = _get_value(
            None, yaml_data, ["skip", "cleanup"], config.skip.cleanup
        )

    # Cross-timestamp config
    config.cross_timestamp.baseline_experiment = _get_value(
        args.baseline_experiment,
        yaml_data,
        ["cross_timestamp", "baseline_experiment"],
        config.cross_timestamp.baseline_experiment,
    )
    config.cross_timestamp.baseline_date = _get_value(
        getattr(args, "baseline_date", None),
        yaml_data,
        ["cross_timestamp", "baseline_date"],
        config.cross_timestamp.baseline_date,
    )
    config.cross_timestamp.aorta_report_path = _get_value(
        args.aorta_report_path,
        yaml_data,
        ["cross_timestamp", "aorta_report_path"],
        config.cross_timestamp.aorta_report_path,
    )

    # Analysis config (labels for aorta-report)
    config.analysis.baseline_label = _get_value(
        getattr(args, "baseline_label", None),
        yaml_data,
        ["analysis", "baseline_label"],
        config.analysis.baseline_label,
    )
    config.analysis.test_label = _get_value(
        getattr(args, "test_label", None),
        yaml_data,
        ["analysis", "test_label"],
        config.analysis.test_label,
    )
    config.analysis.report_label = _get_value(
        getattr(args, "report_label", None),
        yaml_data,
        ["analysis", "report_label"],
        config.analysis.report_label,
    )

    # skip_tracelens_single_config: CLI only (not in YAML). Only add when --skip-tracelens passed.
    if getattr(args, "skip_tracelens", False):
        config.analysis.skip_tracelens_single_config = True

    # Git config
    config.git.user_name = _get_value(
        getattr(args, "git_user_name", None),
        yaml_data,
        ["git", "user_name"],
        config.git.user_name,
    )
    config.git.user_email = _get_value(
        getattr(args, "git_user_email", None),
        yaml_data,
        ["git", "user_email"],
        config.git.user_email,
    )
    config.git.github_token = _get_value(
        getattr(args, "github_token", None),
        yaml_data,
        ["git", "github_token"],
        config.git.github_token,
    )

    # Output config
    config.output.log_dir = _get_value(
        args.log_dir, yaml_data, ["output", "log_dir"], config.output.log_dir
    )
    config.output.log_level = _get_value(
        args.log_level, yaml_data, ["output", "log_level"], config.output.log_level
    )

    return config


# =============================================================================
# CLI Argument Parsing
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Weekly CI Kickoff - RCCL Warp Speed Performance Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with config file defaults
    python scripts/weekly_ci_kickoff.py

    # Override specific settings
    python scripts/weekly_ci_kickoff.py --config-pairs "56,256 37,384" --skip-rccl-build

    # Use custom config file
    python scripts/weekly_ci_kickoff.py --config my_config.yaml

    # Skip multiple stages
    python scripts/weekly_ci_kickoff.py --skip-docker-setup --skip-rccl-build --skip-install-deps
        """,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/weekly_ci.yaml"),
        help="Path to YAML config file (default: config/weekly_ci.yaml)",
    )

    # Test configuration
    parser.add_argument(
        "--config-pairs",
        type=str,
        default=None,
        help='Space-separated CU,threads pairs (e.g., "56,256 37,384 32,512")',
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help='Baseline configuration (CU,threads format, e.g., "56,256")',
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Explicit experiment directory to use (auto-detect if not specified)",
    )

    # RCCL configuration
    parser.add_argument(
        "--gpu-target",
        type=str,
        default=None,
        help="GPU architecture target (e.g., gfx950, gfx942)",
    )
    parser.add_argument(
        "--rccl-branch", type=str, default=None, help="RCCL branch to test"
    )

    # Docker configuration
    parser.add_argument(
        "--compose-file", type=str, default=None, help="Docker compose file path"
    )
    parser.add_argument(
        "--container-name", type=str, default=None, help="Docker container name"
    )
    parser.add_argument(
        "--docker-user",
        type=str,
        default=None,
        help="Docker registry username (e.g., rocmshared)",
    )
    parser.add_argument(
        "--docker-password",
        type=str,
        default=None,
        help="Docker registry password/token (can also use DOCKER_PASSWORD env var)",
    )
    parser.add_argument(
        "--docker-build",
        action="store_true",
        help="Build Docker image before starting container (default: skip build)",
    )
    parser.add_argument(
        "--no-docker-build",
        action="store_true",
        help="Skip Docker image build, use existing image (default behavior)",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart container even if already running (default: reuse running container)",
    )

    # Skip stages
    parser.add_argument(
        "--skip-docker-setup", action="store_true", help="Skip Docker setup stage"
    )
    parser.add_argument(
        "--skip-rccl-build", action="store_true", help="Skip RCCL build stage"
    )
    parser.add_argument(
        "--skip-install-deps",
        action="store_true",
        help="Skip dependency installation stage",
    )
    parser.add_argument(
        "--skip-performance-tests",
        action="store_true",
        help="Skip performance tests stage",
    )
    parser.add_argument(
        "--skip-single-config-analysis",
        action="store_true",
        help="Skip single-config analysis stage",
    )
    parser.add_argument(
        "--skip-pairwise-comparison",
        action="store_true",
        help="Skip pairwise comparison stage",
    )
    parser.add_argument(
        "--skip-tracelens",
        action="store_true",
        help="Pass --skip-tracelens to single-config aorta-report (when TraceLens already run)",
    )
    parser.add_argument(
        "--skip-compare-all",
        action="store_true",
        help="Skip compare-all-runs analysis stage (default: skipped)",
    )
    parser.add_argument(
        "--no-skip-compare-all",
        action="store_true",
        help="Enable compare-all-runs analysis stage (overrides default skip)",
    )
    parser.add_argument(
        "--skip-checkout-aorta-report",
        action="store_true",
        help="Skip aorta-report checkout stage",
    )
    parser.add_argument(
        "--skip-cross-timestamp",
        action="store_true",
        help="Skip cross-timestamp comparison stage",
    )
    parser.add_argument(
        "--skip-convert-html-to-md",
        action="store_true",
        help="Skip HTML-to-Markdown conversion stage (before push)",
    )
    parser.add_argument(
        "--baseline-experiment",
        type=str,
        default=None,
        help="Previous experiment directory for cross-timestamp comparison (auto-detect if not specified)",
    )
    parser.add_argument(
        "--baseline-date",
        type=str,
        default=None,
        help="Date directory in aorta-report for cross-timestamp baseline (e.g., '2026-02-19')",
    )
    parser.add_argument(
        "--aorta-report-path",
        type=str,
        default=None,
        help="Path to aorta-report repository",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip pushing results to aorta-report",
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Cleanup container after completion"
    )

    # Analysis configuration (labels for aorta-report)
    parser.add_argument(
        "--baseline-label",
        type=str,
        default=None,
        help="Label for baseline in aorta-report output (e.g., 'baseline', 'v1.0')",
    )
    parser.add_argument(
        "--test-label",
        type=str,
        default=None,
        help="Label for test in aorta-report output (e.g., 'test', 'v1.1')",
    )
    parser.add_argument(
        "--report-label",
        type=str,
        default=None,
        help="Override label for aorta-report directory and dashboard entry (default: date from experiment dir)",
    )

    # Git configuration
    parser.add_argument(
        "--git-user-name",
        type=str,
        default=None,
        help="Git user name for commits (default: Weekly CI Bot)",
    )
    parser.add_argument(
        "--git-user-email",
        type=str,
        default=None,
        help="Git user email for commits",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub token for aorta-report (can also use AORTA_REPORT_GITHUB_TOKEN env var)",
    )

    # Output configuration
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None, help="Directory for log files"
    )

    return parser.parse_args()

