# Weekly CI Kickoff - Standalone Script

This document outlines the design for a standalone Python script that replicates the functionality of the 
`.github/workflows/rccl-warp-speed-analysis.yml` CI workflow for local execution.

---

## Overview

Convert the GitHub Actions workflow into a standalone Python script that can be run locally or on any 
machine with Docker, without requiring GitHub Actions infrastructure.

**Key Features:**
- Python-based for better maintainability and cross-platform support
- Configuration via YAML file and/or command-line arguments (CLI takes precedence)
- Skip any stage via configuration
- Comprehensive logging to file and console

---

## Input Parameters

### Configuration File (`config/weekly_ci.yaml`)

```yaml
# RCCL Warp Speed Configuration
rccl:
  branch: "warp_speed_v1"
  gpu_target: "gfx950"

# Test configurations
test:
  config_pairs: "56,256 37,384 32,512"
  baseline: "56,256"
  training_config: "config/single_node/gemm_overlap_comm.yaml"

# Docker settings
docker:
  compose_file: "docker/rccl_test/docker-compose.rocm70_9-1.yaml"
  container_name: "training-overlap-bugs-rocm70_9-1"

# Stage control (set to true to skip)
skip:
  docker_setup: false
  rccl_build: false
  install_deps: false
  performance_tests: false
  pairwise_analysis: false
  compare_all_analysis: true       # Skip by default for initial setup
  checkout_aorta_report: false     # Needed for cross-timestamp comparison
  cross_timestamp_comparison: false
  push_results: true               # Skip by default
  cleanup: true                    # Skip by default (leave container running)

# Cross-timestamp comparison settings
cross_timestamp:
  # Previous experiment directory to compare against (optional)
  # If not specified, auto-detects the second-most-recent experiment
  baseline_experiment: ""
  # Path to aorta-report repository (for finding previous runs)
  aorta_report_path: "../aorta-report"

# Output settings
output:
  log_dir: "logs"
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
```

### Command-Line Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config file | `config/weekly_ci.yaml` |
| `--config-pairs` | Space-separated CU,threads pairs | From config |
| `--baseline` | Baseline configuration (CU,threads) | From config |
| `--training-config` | Path to training config YAML | From config |
| `--gpu-target` | GPU architecture (gfx950, gfx942, etc.) | From config |
| `--rccl-branch` | RCCL branch to test | From config |
| `--compose-file` | Docker compose file path | From config |
| `--container-name` | Docker container name | From config |
| `--skip-docker-setup` | Skip Docker setup stage | From config |
| `--skip-rccl-build` | Skip RCCL build stage | From config |
| `--skip-install-deps` | Skip dependency installation | From config |
| `--skip-performance-tests` | Skip performance tests | From config |
| `--skip-pairwise-analysis` | Skip pairwise analysis | From config |
| `--skip-compare-all` | Skip compare-all-runs analysis | `true` (skipped by default) |
| `--no-skip-compare-all` | Enable compare-all-runs analysis | |
| `--skip-checkout-aorta-report` | Skip aorta-report checkout | From config |
| `--skip-cross-timestamp` | Skip cross-timestamp comparison | From config |
| `--baseline-experiment` | Previous experiment dir for cross-timestamp comparison | Auto-detect |
| `--aorta-report-path` | Path to aorta-report repository | `../aorta-report` |
| `--baseline-date` | Date dir in aorta-report for cross-timestamp baseline | Auto-detect |
| `--baseline-label` | Label for baseline in aorta-report output | From config |
| `--test-label` | Label for test in aorta-report output | From config |
| `--report-label` | Override for aorta-report dir and dashboard entry | Date from experiment |
| `--skip-convert-html-to-md` | Skip HTML-to-Markdown conversion stage | From config |
| `--skip-push` | Skip pushing to aorta-report | From config |
| `--cleanup` | Cleanup container after run | From config |
| `--log-level` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--log-dir` | Directory for log files | `logs` |
| `-h, --help` | Show help message | |

**Note:** Command-line arguments always take precedence over config file values.

---

## Script Structure

```
┌─────────────────────────────────────────────────────────────┐
│  0. Initialize                                              │
│     • Parse config file                                     │
│     • Parse command-line args (override config)             │
│     • Setup logging (file + console)                        │
├─────────────────────────────────────────────────────────────┤
│  1. Validate Environment                        [SKIPPABLE] │
│     • Check Docker is installed and running                 │
│     • Verify docker-compose file exists                     │
│     • Check we're in the aorta repo root                    │
├─────────────────────────────────────────────────────────────┤
│  2. Docker Setup                                [SKIPPABLE] │
│     • Stop/remove existing container                        │
│     • Build and start container                             │
├─────────────────────────────────────────────────────────────┤
│  3. Build RCCL (inside container)               [SKIPPABLE] │
│     • Clone or update rccl repo                             │
│     • Checkout specified branch                             │
│     • Build with GPU target                                 │
├─────────────────────────────────────────────────────────────┤
│  4. Install Dependencies (inside container)     [SKIPPABLE] │
│     • pip install -e . (current package)                    │
│     • pip install -r requirements.txt                       │
│     • pip install analysis packages                         │
├─────────────────────────────────────────────────────────────┤
│  5. Run Performance Tests                       [SKIPPABLE] │
│     • Execute run_rccl_warp_speed_comparison.sh             │
│     • Find generated experiment directory                   │
├─────────────────────────────────────────────────────────────┤
│  6. Single Config Analysis                       [SKIPPABLE] │
│     • aorta-report pipeline summary --test <dir> per config  │
│     • Default: run TraceLens; use --skip-tracelens to skip  │
├─────────────────────────────────────────────────────────────┤
│  7. Pairwise Comparison                         [SKIPPABLE] │
│     • aorta-report pipeline summary --baseline --test       │
│       --skip-tracelens (always)                              │
├─────────────────────────────────────────────────────────────┤
│  8. Run Compare-All Analysis          [SKIPPABLE, OFF*]     │
│     • Merged report of all configurations                   │
│     * Skipped by default for initial setup                  │
├─────────────────────────────────────────────────────────────┤
│  9. Checkout aorta-report                       [SKIPPABLE] │
│     • Clone or update aorta-report repository               │
│     • Required for cross-timestamp comparison               │
├─────────────────────────────────────────────────────────────┤
│ 10. Cross-Timestamp Comparison                  [SKIPPABLE] │
│     • Find previous experiment (from config or auto-detect) │
│     • Compare each config: older timestamp = baseline,      │
│       newer timestamp = test                                │
│     • aorta-report pipeline summary --baseline <old>        │
│       --test <new> --skip-tracelens                         │
├─────────────────────────────────────────────────────────────┤
│ 11. Generate Summary Report                                 │
│     • Print summary to terminal                             │
│     • Save summary to file                                  │
├─────────────────────────────────────────────────────────────┤
│ 12. Push to aorta-report (Optional)             [SKIPPABLE] │
│     • Copy results to dated directory                       │
│     • Commit and push                                       │
│     • Update README dashboard                               │
├─────────────────────────────────────────────────────────────┤
│ 13. Cleanup                                     [SKIPPABLE] │
│     • docker compose down                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Output Directory Structure

```
experiments/rccl_warp_speed_YYYYMMDD_HHMMSS/
├── 56cu_256threads/              # Baseline configuration results
│   ├── traces/
│   ├── logs/
│   ├── metrics.json
│   └── summary/                  # Generated by: aorta-report pipeline summary --test-dir
├── 37cu_384threads/              # Test configuration 1
│   └── summary/
├── 32cu_512threads/              # Test configuration 2
│   └── summary/
├── single_config_results/        # Single-config summaries (Stage 6)
│   ├── single_config_56cu_256threads/
│   ├── single_config_37cu_384threads/
│   └── single_config_32cu_512threads/
├── comparison_results/           # Pairwise comparisons (Stage 7)
│   ├── baseline_vs_37cu_384threads/
│   └── baseline_vs_32cu_512threads/
├── cross_timestamp_comparison/   # Comparison with previous run (older=baseline, newer=test)
│   ├── 56cu_256threads/
│   ├── 37cu_384threads/
│   └── 32cu_512threads/
├── compare_all_runs/             # Optional: Merged comparison (if --no-skip-compare-all)
└── summary.txt                   # Run summary

logs/
├── weekly_ci_20260219_143022.log # Full log with timestamps
└── latest.log -> weekly_ci_20260219_143022.log
```

---

## Key Differences from CI Workflow

| Aspect | CI Workflow | Standalone Script |
|--------|-------------|-------------------|
| Language | YAML + Bash | Python |
| Repository | Checks out repo | Assumes already in repo root |
| Configuration | Workflow inputs | YAML config + CLI args |
| Stage Control | Fixed sequence | Skip any stage |
| Docker Auth | Uses GitHub secrets | Uses local Docker auth |
| Artifacts | Uploads to GitHub | Saves locally in `experiments/` |
| aorta-report | Always pushes | Optional (skip by default) |
| Cross-timestamp | Not available | Compare current vs previous run |
| Logging | GitHub Actions logs | File + console logging |
| Timeout | 300 minutes hard limit | No timeout (user can Ctrl+C) |

---

## Script Code

**Location:** `scripts/weekly_ci_kickoff.py`

```python
#!/usr/bin/env python3
"""
Weekly CI Kickoff - RCCL Warp Speed Performance Analysis

Replicates the functionality of .github/workflows/rccl-warp-speed-analysis.yml
for local execution without GitHub Actions infrastructure.

Usage:
    python scripts/weekly_ci_kickoff.py [OPTIONS]

Examples:
    # Run with config file defaults
    python scripts/weekly_ci_kickoff.py

    # Override specific settings
    python scripts/weekly_ci_kickoff.py --config-pairs "56,256 37,384" --skip-rccl-build

    # Use custom config file
    python scripts/weekly_ci_kickoff.py --config my_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


# =============================================================================
# Configuration
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


@dataclass
class DockerConfig:
    """Docker configuration."""
    compose_file: str = "docker/rccl_test/docker-compose.rocm70_9-1.yaml"
    container_name: str = "training-overlap-bugs-rocm70_9-1"


@dataclass
class SkipConfig:
    """Stage skip configuration."""
    docker_setup: bool = False
    rccl_build: bool = False
    install_deps: bool = False
    performance_tests: bool = False
    pairwise_analysis: bool = False
    compare_all_analysis: bool = True  # Skip by default for initial setup
    checkout_aorta_report: bool = False  # Needed for cross-timestamp comparison
    cross_timestamp_comparison: bool = False
    push_results: bool = True  # Skip by default
    cleanup: bool = True  # Skip by default (leave container running)


@dataclass
class CrossTimestampConfig:
    """Cross-timestamp comparison configuration."""
    baseline_experiment: str = ""  # If empty, auto-detect second-most-recent
    aorta_report_path: str = "../aorta-report"


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
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Runtime state
    experiment_dir: Optional[str] = None
    baseline_experiment_dir: Optional[str] = None  # For cross-timestamp comparison
    aorta_report_dir: Optional[Path] = None  # Path to cloned aorta-report


# =============================================================================
# Logging Setup
# =============================================================================
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Green
        'WARNING': '\033[1;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[1;31m', # Bold Red
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(config: Config) -> logging.Logger:
    """Setup logging to both file and console."""
    logger = logging.getLogger("weekly_ci")
    logger.setLevel(getattr(logging, config.output.log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create log directory
    log_dir = Path(config.output.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"weekly_ci_{timestamp}.log"
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.output.log_level.upper()))
    console_formatter = ColoredFormatter(
        '%(levelname)s | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create symlink to latest log
    latest_link = log_dir / "latest.log"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(log_file.name)
    
    logger.info(f"Logging to: {log_file}")
    return logger


# =============================================================================
# Configuration Loading
# =============================================================================
def load_config_file(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def merge_config(config: Config, yaml_data: dict, args: argparse.Namespace) -> Config:
    """Merge YAML config and CLI args into Config object. CLI takes precedence."""
    
    # Helper to get value with precedence: CLI > YAML > default
    def get_value(cli_value, yaml_path: list, default):
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
    
    # RCCL config
    config.rccl.branch = get_value(
        args.rccl_branch, ['rccl', 'branch'], config.rccl.branch
    )
    config.rccl.gpu_target = get_value(
        args.gpu_target, ['rccl', 'gpu_target'], config.rccl.gpu_target
    )
    
    # Test config
    config.test.config_pairs = get_value(
        args.config_pairs, ['test', 'config_pairs'], config.test.config_pairs
    )
    config.test.baseline = get_value(
        args.baseline, ['test', 'baseline'], config.test.baseline
    )
    config.test.training_config = get_value(
        args.training_config, ['test', 'training_config'], config.test.training_config
    )
    
    # Docker config
    config.docker.compose_file = get_value(
        args.compose_file, ['docker', 'compose_file'], config.docker.compose_file
    )
    config.docker.container_name = get_value(
        args.container_name, ['docker', 'container_name'], config.docker.container_name
    )
    
    # Skip config - CLI flags override YAML
    config.skip.docker_setup = get_value(
        args.skip_docker_setup if args.skip_docker_setup else None,
        ['skip', 'docker_setup'], config.skip.docker_setup
    )
    config.skip.rccl_build = get_value(
        args.skip_rccl_build if args.skip_rccl_build else None,
        ['skip', 'rccl_build'], config.skip.rccl_build
    )
    config.skip.install_deps = get_value(
        args.skip_install_deps if args.skip_install_deps else None,
        ['skip', 'install_deps'], config.skip.install_deps
    )
    config.skip.performance_tests = get_value(
        args.skip_performance_tests if args.skip_performance_tests else None,
        ['skip', 'performance_tests'], config.skip.performance_tests
    )
    config.skip.pairwise_analysis = get_value(
        args.skip_pairwise_analysis if args.skip_pairwise_analysis else None,
        ['skip', 'pairwise_analysis'], config.skip.pairwise_analysis
    )
    # Handle compare_all_analysis: --no-skip-compare-all enables it, --skip-compare-all disables it
    if args.no_skip_compare_all:
        config.skip.compare_all_analysis = False
    elif args.skip_compare_all:
        config.skip.compare_all_analysis = True
    else:
        config.skip.compare_all_analysis = get_value(
            None, ['skip', 'compare_all_analysis'], config.skip.compare_all_analysis
        )
    
    config.skip.checkout_aorta_report = get_value(
        args.skip_checkout_aorta_report if args.skip_checkout_aorta_report else None,
        ['skip', 'checkout_aorta_report'], config.skip.checkout_aorta_report
    )
    config.skip.cross_timestamp_comparison = get_value(
        args.skip_cross_timestamp if args.skip_cross_timestamp else None,
        ['skip', 'cross_timestamp_comparison'], config.skip.cross_timestamp_comparison
    )
    config.skip.push_results = get_value(
        args.skip_push if args.skip_push else None,
        ['skip', 'push_results'], config.skip.push_results
    )
    # Cleanup is inverted: --cleanup means DO cleanup (skip.cleanup = False)
    if args.cleanup:
        config.skip.cleanup = False
    else:
        config.skip.cleanup = get_value(
            None, ['skip', 'cleanup'], config.skip.cleanup
        )
    
    # Cross-timestamp config
    config.cross_timestamp.baseline_experiment = get_value(
        args.baseline_experiment,
        ['cross_timestamp', 'baseline_experiment'], config.cross_timestamp.baseline_experiment
    )
    config.cross_timestamp.aorta_report_path = get_value(
        args.aorta_report_path,
        ['cross_timestamp', 'aorta_report_path'], config.cross_timestamp.aorta_report_path
    )
    
    # Output config
    config.output.log_dir = get_value(
        args.log_dir, ['output', 'log_dir'], config.output.log_dir
    )
    config.output.log_level = get_value(
        args.log_level, ['output', 'log_level'], config.output.log_level
    )
    
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
        """
    )
    
    # Config file
    parser.add_argument(
        '--config', type=Path, default=Path('config/weekly_ci.yaml'),
        help='Path to YAML config file (default: config/weekly_ci.yaml)'
    )
    
    # Test configuration
    parser.add_argument(
        '--config-pairs', type=str, default=None,
        help='Space-separated CU,threads pairs (e.g., "56,256 37,384 32,512")'
    )
    parser.add_argument(
        '--baseline', type=str, default=None,
        help='Baseline configuration (CU,threads format, e.g., "56,256")'
    )
    parser.add_argument(
        '--training-config', type=str, default=None,
        help='Path to training config YAML'
    )
    
    # RCCL configuration
    parser.add_argument(
        '--gpu-target', type=str, default=None,
        help='GPU architecture target (e.g., gfx950, gfx942)'
    )
    parser.add_argument(
        '--rccl-branch', type=str, default=None,
        help='RCCL branch to test'
    )
    
    # Docker configuration
    parser.add_argument(
        '--compose-file', type=str, default=None,
        help='Docker compose file path'
    )
    parser.add_argument(
        '--container-name', type=str, default=None,
        help='Docker container name'
    )
    
    # Skip stages
    parser.add_argument(
        '--skip-docker-setup', action='store_true',
        help='Skip Docker setup stage'
    )
    parser.add_argument(
        '--skip-rccl-build', action='store_true',
        help='Skip RCCL build stage'
    )
    parser.add_argument(
        '--skip-install-deps', action='store_true',
        help='Skip dependency installation stage'
    )
    parser.add_argument(
        '--skip-performance-tests', action='store_true',
        help='Skip performance tests stage'
    )
    parser.add_argument(
        '--skip-pairwise-analysis', action='store_true',
        help='Skip pairwise analysis stage'
    )
    parser.add_argument(
        '--skip-compare-all', action='store_true',
        help='Skip compare-all-runs analysis stage (default: skipped)'
    )
    parser.add_argument(
        '--no-skip-compare-all', action='store_true',
        help='Enable compare-all-runs analysis stage (overrides default skip)'
    )
    parser.add_argument(
        '--skip-checkout-aorta-report', action='store_true',
        help='Skip aorta-report checkout stage'
    )
    parser.add_argument(
        '--skip-cross-timestamp', action='store_true',
        help='Skip cross-timestamp comparison stage'
    )
    parser.add_argument(
        '--baseline-experiment', type=str, default=None,
        help='Previous experiment directory for cross-timestamp comparison (auto-detect if not specified)'
    )
    parser.add_argument(
        '--aorta-report-path', type=str, default=None,
        help='Path to aorta-report repository'
    )
    parser.add_argument(
        '--skip-push', action='store_true',
        help='Skip pushing results to aorta-report'
    )
    parser.add_argument(
        '--cleanup', action='store_true',
        help='Cleanup container after completion'
    )
    
    # Output configuration
    parser.add_argument(
        '--log-level', type=str, default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--log-dir', type=str, default=None,
        help='Directory for log files'
    )
    
    return parser.parse_args()


# =============================================================================
# Utility Functions
# =============================================================================
def run_command(
    cmd: str | list[str],
    logger: logging.Logger,
    check: bool = True,
    capture_output: bool = False,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    if isinstance(cmd, list):
        cmd_str = ' '.join(cmd)
    else:
        cmd_str = cmd
    
    logger.debug(f"Running: {cmd_str}")
    
    result = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        check=check,
        capture_output=capture_output,
        text=True,
        cwd=cwd,
    )
    
    if capture_output and result.stdout:
        logger.debug(f"stdout: {result.stdout.strip()}")
    if capture_output and result.stderr:
        logger.debug(f"stderr: {result.stderr.strip()}")
    
    return result


def docker_exec(
    container: str,
    command: str,
    logger: logging.Logger,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Execute command inside Docker container."""
    full_cmd = f'docker exec {container} bash -c "{command}"'
    return run_command(full_cmd, logger, check=check)


# =============================================================================
# Pipeline Stages
# =============================================================================
def stage_validate_environment(config: Config, logger: logging.Logger) -> bool:
    """Stage 1: Validate environment."""
    logger.info("=" * 60)
    logger.info("Stage 1: Validating environment")
    logger.info("=" * 60)
    
    # Check Docker
    try:
        run_command("docker --version", logger, capture_output=True)
    except subprocess.CalledProcessError:
        logger.error("Docker is not installed or not in PATH")
        return False
    
    try:
        run_command("docker info", logger, capture_output=True)
    except subprocess.CalledProcessError:
        logger.error("Docker daemon is not running")
        return False
    
    # Check docker compose
    try:
        run_command("docker compose version", logger, capture_output=True)
    except subprocess.CalledProcessError:
        logger.error("Docker Compose is not available")
        return False
    
    # Check we're in aorta repo root
    if not Path("pyproject.toml").exists() or not Path("src/aorta").is_dir():
        logger.error("Must run from aorta repository root")
        return False
    
    # Check docker-compose file exists
    if not Path(config.docker.compose_file).exists():
        logger.error(f"Docker compose file not found: {config.docker.compose_file}")
        return False
    
    # Check training config exists
    if not Path(config.test.training_config).exists():
        logger.error(f"Training config not found: {config.test.training_config}")
        return False
    
    logger.info("Environment validation passed")
    return True


def stage_docker_setup(config: Config, logger: logging.Logger) -> bool:
    """Stage 2: Setup Docker container."""
    if config.skip.docker_setup:
        logger.info("Skipping Docker setup (--skip-docker-setup)")
        return True
    
    logger.info("=" * 60)
    logger.info("Stage 2: Setting up Docker container")
    logger.info("=" * 60)
    
    container = config.docker.container_name
    compose_file = config.docker.compose_file
    
    # Cleanup existing container
    logger.info("Cleaning up existing container (if any)...")
    run_command(f"docker stop {container}", logger, check=False)
    run_command(f"docker rm {container}", logger, check=False)
    run_command(f"docker compose -f {compose_file} down", logger, check=False)
    
    # Build and start container
    logger.info("Building Docker container...")
    run_command(f"docker compose -f {compose_file} build", logger)
    
    logger.info("Starting Docker container...")
    run_command(f"docker compose -f {compose_file} up -d", logger)
    
    # Wait for container to be ready
    import time
    time.sleep(5)
    
    # Verify container is running
    result = run_command(
        "docker ps --format '{{.Names}}'",
        logger,
        capture_output=True
    )
    if container not in result.stdout:
        logger.error("Container failed to start")
        return False
    
    logger.info("Docker container is running")
    return True


def stage_build_rccl(config: Config, logger: logging.Logger) -> bool:
    """Stage 3: Build RCCL inside container."""
    if config.skip.rccl_build:
        logger.info("Skipping RCCL build (--skip-rccl-build)")
        return True
    
    logger.info("=" * 60)
    logger.info("Stage 3: Building RCCL")
    logger.info(f"  Branch: {config.rccl.branch}")
    logger.info(f"  GPU Target: {config.rccl.gpu_target}")
    logger.info("=" * 60)
    
    build_script = f"""
        mkdir -p /rccl && cd /rccl

        if [ -d 'rccl' ]; then
            cd rccl
            git fetch origin
            git checkout {config.rccl.branch}
            git pull
        else
            git clone --recursive https://github.com/mustafabar/rccl.git
            cd rccl
            git checkout {config.rccl.branch}
        fi

        echo 'Building RCCL with GPU target: {config.rccl.gpu_target}'
        ./install.sh -l --amdgpu_targets={config.rccl.gpu_target}

        echo 'RCCL build completed. Library location:'
        ls -la /rccl/rccl/build/release/ || echo 'Build directory not found'
    """
    
    docker_exec(config.docker.container_name, build_script, logger)
    logger.info("RCCL build completed")
    return True


def stage_install_dependencies(config: Config, logger: logging.Logger) -> bool:
    """Stage 4: Install Python dependencies."""
    if config.skip.install_deps:
        logger.info("Skipping dependency installation (--skip-install-deps)")
        return True
    
    logger.info("=" * 60)
    logger.info("Stage 4: Installing Python dependencies")
    logger.info("=" * 60)
    
    install_script = """
        pip install -e .
        pip install -r requirements.txt
        pip install pandas openpyxl matplotlib seaborn numpy
    """
    
    docker_exec(config.docker.container_name, install_script, logger)
    logger.info("Dependencies installed")
    return True


def stage_run_performance_tests(config: Config, logger: logging.Logger) -> bool:
    """Stage 5: Run performance tests."""
    if config.skip.performance_tests:
        logger.info("Skipping performance tests (--skip-performance-tests)")
        return True
    
    logger.info("=" * 60)
    logger.info("Stage 5: Running performance tests")
    logger.info(f"  Config pairs: {config.test.config_pairs}")
    logger.info(f"  Training config: {config.test.training_config}")
    logger.info("=" * 60)
    
    test_script = f"""
        export LD_LIBRARY_PATH=/rccl/rccl/build/release:$LD_LIBRARY_PATH
        
        bash ./scripts/tracelens_single_config/run_rccl_warp_speed_comparison.sh \\
            -p "{config.test.config_pairs}" \\
            -c {config.test.training_config}
    """
    
    docker_exec(config.docker.container_name, test_script, logger)
    logger.info("Performance tests completed")
    return True


def stage_find_experiment_dir(config: Config, logger: logging.Logger) -> bool:
    """Find the experiment directory."""
    logger.info("Finding experiment directory...")
    
    # Find most recent experiment directory
    experiment_dirs = sorted(
        Path("experiments").glob("rccl_warp_speed_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not experiment_dirs:
        logger.error("No experiment directory found")
        return False
    
    config.experiment_dir = str(experiment_dirs[0])
    logger.info(f"Found experiment directory: {config.experiment_dir}")
    return True


def stage_pairwise_analysis(config: Config, logger: logging.Logger) -> bool:
    """Stage 6: Run pairwise comparison analysis using aorta-report CLI.
    
    This stage has two steps:
    1. Generate summary for ALL configurations using aorta-report pipeline summary --test-dir
    2. Run pairwise comparisons (baseline vs each test) using aorta-report pipeline summary
       with --baseline and --test flags, with --skip-tracelens
    """
    if config.skip.pairwise_analysis:
        logger.info("Skipping pairwise analysis (--skip-pairwise-analysis)")
        return True
    
    if not config.experiment_dir:
        if not stage_find_experiment_dir(config, logger):
            return False
    
    logger.info("=" * 60)
    logger.info("Stage 6: Running pairwise comparison analysis")
    logger.info("=" * 60)
    
    # Parse baseline
    baseline_cu, baseline_threads = config.test.baseline.split(',')
    baseline_dir = f"{config.experiment_dir}/{baseline_cu}cu_{baseline_threads}threads"
    
    # Step 1: Generate summary for ALL configurations
    logger.info("Step 1: Generating summary for all configurations...")
    
    summary_script = f"""
        echo '========================================'
        echo 'Step 1: Generating summary for ALL configurations'
        echo '========================================'
        
        for pair in {config.test.config_pairs}; do
            CU_COUNT=$(echo $pair | cut -d',' -f1)
            THREADS=$(echo $pair | cut -d',' -f2)
            TEST_DIR="{config.experiment_dir}/${{CU_COUNT}}cu_${{THREADS}}threads"
            
            echo "Generating summary for: $TEST_DIR"
            aorta-report pipeline summary --test-dir "$TEST_DIR"
        done
    """
    
    docker_exec(config.docker.container_name, summary_script, logger)
    logger.info("Step 1 completed: Summaries generated for all configurations")
    
    # Step 2: Run pairwise comparisons (baseline vs each test)
    logger.info("Step 2: Running pairwise comparisons...")
    
    comparison_script = f"""
        OUTPUT_DIR="{config.experiment_dir}/comparison_results"
        mkdir -p "$OUTPUT_DIR"

        for pair in {config.test.config_pairs}; do
            CU_COUNT=$(echo $pair | cut -d',' -f1)
            THREADS=$(echo $pair | cut -d',' -f2)

            # Skip if this is the baseline
            if [ "$CU_COUNT" = "{baseline_cu}" ] && [ "$THREADS" = "{baseline_threads}" ]; then
                continue
            fi

            TEST_DIR="{config.experiment_dir}/${{CU_COUNT}}cu_${{THREADS}}threads"
            COMPARISON_OUTPUT="$OUTPUT_DIR/baseline_vs_${{CU_COUNT}}cu_${{THREADS}}threads"

            echo '========================================'
            echo "Comparing baseline ({baseline_cu} cu, {baseline_threads} threads) vs test ($CU_COUNT cu, $THREADS threads)"
            echo '========================================'

            aorta-report pipeline summary \\
                --baseline "{baseline_dir}" \\
                --test "$TEST_DIR" \\
                --output "$COMPARISON_OUTPUT" \\
                --skip-tracelens
        done
    """
    
    docker_exec(config.docker.container_name, comparison_script, logger)
    logger.info("Step 2 completed: Pairwise comparisons finished")
    logger.info("Pairwise analysis completed")
    return True


def stage_compare_all_analysis(config: Config, logger: logging.Logger) -> bool:
    """Stage 7: Run compare-all-runs analysis.
    
    NOTE: This stage is skipped by default for initial setup.
    Enable with --no-skip-compare-all or set skip.compare_all_analysis: false in config.
    """
    if config.skip.compare_all_analysis:
        logger.info("Skipping compare-all analysis (skipped by default for initial setup)")
        logger.info("  To enable: use --no-skip-compare-all or set skip.compare_all_analysis: false")
        return True
    
    if not config.experiment_dir:
        if not stage_find_experiment_dir(config, logger):
            return False
    
    logger.info("=" * 60)
    logger.info("Stage 7: Running compare-all-runs analysis")
    logger.info("=" * 60)
    
    # Parse baseline
    baseline_cu, baseline_threads = config.test.baseline.split(',')
    baseline_dir = f"{config.experiment_dir}/{baseline_cu}cu_{baseline_threads}threads"
    
    # Build test directories list
    test_dirs = []
    for pair in config.test.config_pairs.split():
        cu, threads = pair.split(',')
        if cu != baseline_cu or threads != baseline_threads:
            test_dirs.append(f"{config.experiment_dir}/{cu}cu_{threads}threads")
    
    test_dirs_str = ' '.join(test_dirs)
    
    logger.info(f"Baseline: {baseline_dir}")
    logger.info(f"Test directories: {test_dirs_str}")
    
    analysis_script = f"""
        OUTPUT_DIR="{config.experiment_dir}/compare_all_runs"
        mkdir -p "$OUTPUT_DIR"

        aorta-report pipeline summary \\
            --baseline "{baseline_dir}" \\
            --test {test_dirs_str} \\
            --output "$OUTPUT_DIR" \\
            --skip-tracelens \\
            --compare-all-runs
    """
    
    docker_exec(config.docker.container_name, analysis_script, logger)
    logger.info("Compare-all-runs analysis completed")
    return True


def stage_checkout_aorta_report(config: Config, logger: logging.Logger) -> bool:
    """Stage 8: Checkout aorta-report repository.
    
    This is needed for cross-timestamp comparison and pushing results.
    """
    if config.skip.checkout_aorta_report:
        logger.info("Skipping aorta-report checkout (--skip-checkout-aorta-report)")
        return True
    
    logger.info("=" * 60)
    logger.info("Stage 8: Checking out aorta-report repository")
    logger.info("=" * 60)
    
    aorta_report_path = Path(config.cross_timestamp.aorta_report_path)
    
    if aorta_report_path.exists():
        logger.info(f"Updating existing aorta-report at: {aorta_report_path}")
        try:
            run_command("git fetch origin", logger, cwd=aorta_report_path)
            run_command("git pull --rebase origin main", logger, cwd=aorta_report_path)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to update aorta-report: {e}")
            logger.info("Continuing with existing version...")
    else:
        logger.info(f"Cloning aorta-report to: {aorta_report_path}")
        try:
            run_command(
                f"git clone git@github.com:ROCm/aorta-report.git {aorta_report_path}",
                logger
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone aorta-report: {e}")
            logger.error("Cross-timestamp comparison and push will not be available")
            return False
    
    config.aorta_report_dir = aorta_report_path.resolve()
    logger.info(f"aorta-report ready at: {config.aorta_report_dir}")
    return True


def stage_cross_timestamp_comparison(config: Config, logger: logging.Logger) -> bool:
    """Stage 9: Cross-timestamp comparison.
    
    Compare each configuration between the current run and a previous run.
    Older timestamp = baseline, newer timestamp = test.
    """
    if config.skip.cross_timestamp_comparison:
        logger.info("Skipping cross-timestamp comparison (--skip-cross-timestamp)")
        return True
    
    if not config.experiment_dir:
        if not stage_find_experiment_dir(config, logger):
            return False
    
    logger.info("=" * 60)
    logger.info("Stage 9: Running cross-timestamp comparison")
    logger.info("=" * 60)
    
    # Find the baseline experiment directory (previous run)
    if config.cross_timestamp.baseline_experiment:
        # Use explicitly provided baseline
        baseline_experiment = Path(config.cross_timestamp.baseline_experiment)
        if not baseline_experiment.exists():
            logger.error(f"Specified baseline experiment not found: {baseline_experiment}")
            return False
        config.baseline_experiment_dir = str(baseline_experiment)
    else:
        # Auto-detect: find the second-most-recent experiment directory
        logger.info("Auto-detecting previous experiment directory...")
        experiment_dirs = sorted(
            Path("experiments").glob("rccl_warp_speed_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if len(experiment_dirs) < 2:
            logger.warning("No previous experiment found for cross-timestamp comparison")
            logger.info("Skipping cross-timestamp comparison (need at least 2 experiments)")
            return True
        
        # Most recent is current, second is baseline
        config.baseline_experiment_dir = str(experiment_dirs[1])
    
    logger.info(f"Current experiment (test): {config.experiment_dir}")
    logger.info(f"Previous experiment (baseline): {config.baseline_experiment_dir}")
    
    # Create output directory for cross-timestamp results
    cross_timestamp_output = Path(config.experiment_dir) / "cross_timestamp_comparison"
    cross_timestamp_output.mkdir(parents=True, exist_ok=True)
    
    # Compare each configuration
    comparison_script = f"""
        OUTPUT_DIR="{cross_timestamp_output}"
        BASELINE_EXPERIMENT="{config.baseline_experiment_dir}"
        CURRENT_EXPERIMENT="{config.experiment_dir}"
        
        echo '========================================'
        echo 'Cross-Timestamp Comparison'
        echo "Baseline (older): $BASELINE_EXPERIMENT"
        echo "Test (newer): $CURRENT_EXPERIMENT"
        echo '========================================'
        
        for pair in {config.test.config_pairs}; do
            CU_COUNT=$(echo $pair | cut -d',' -f1)
            THREADS=$(echo $pair | cut -d',' -f2)
            CONFIG_NAME="${{CU_COUNT}}cu_${{THREADS}}threads"
            
            BASELINE_DIR="$BASELINE_EXPERIMENT/$CONFIG_NAME"
            TEST_DIR="$CURRENT_EXPERIMENT/$CONFIG_NAME"
            COMPARISON_OUTPUT="$OUTPUT_DIR/$CONFIG_NAME"
            
            # Check if baseline config exists
            if [ ! -d "$BASELINE_DIR" ]; then
                echo "Baseline config not found: $BASELINE_DIR (skipping)"
                continue
            fi
            
            echo '----------------------------------------'
            echo "Comparing $CONFIG_NAME: baseline (older) vs test (newer)"
            echo "  Baseline: $BASELINE_DIR"
            echo "  Test: $TEST_DIR"
            echo '----------------------------------------'
            
            aorta-report pipeline summary \\
                --baseline "$BASELINE_DIR" \\
                --test "$TEST_DIR" \\
                --output "$COMPARISON_OUTPUT" \\
                --skip-tracelens
        done
    """
    
    docker_exec(config.docker.container_name, comparison_script, logger)
    logger.info("Cross-timestamp comparison completed")
    logger.info(f"Results saved to: {cross_timestamp_output}")
    return True


def stage_generate_summary(config: Config, logger: logging.Logger) -> bool:
    """Stage 10: Generate summary report."""
    logger.info("=" * 60)
    logger.info("Stage 10: Generating summary report")
    logger.info("=" * 60)
    
    if not config.experiment_dir:
        if not stage_find_experiment_dir(config, logger):
            return False
    
    summary_file = Path(config.experiment_dir) / "summary.txt"
    
    # Build config pairs list
    config_list = '\n'.join(
        f"  - CU={pair.split(',')[0]}, Threads={pair.split(',')[1]}"
        for pair in config.test.config_pairs.split()
    )
    
    summary_content = f"""================================================================================
RCCL Warp Speed Performance Analysis Summary
================================================================================

Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
  Experiment Directory: {config.experiment_dir}
  Baseline: {config.test.baseline} (CU,Threads)
  RCCL Branch: {config.rccl.branch}
  GPU Target: {config.rccl.gpu_target}
  Training Config: {config.test.training_config}

Tested Configurations:
{config_list}

Cross-Timestamp Comparison:
  Previous Experiment (Baseline): {config.baseline_experiment_dir or 'N/A'}

Generated Artifacts:
  - Individual config summaries (aorta-report pipeline summary --test-dir)
  - Pairwise comparison reports (baseline vs each test, --skip-tracelens)
  - Cross-timestamp comparison (previous run vs current run)
  - Compare-all-runs merged report (optional, skipped by default)
  - GPU timeline comparison
  - NCCL/Collective comparison
  - Final analysis report (Excel)
  - Performance visualization plots
  - HTML performance report

Directory Structure:
  {config.experiment_dir}/
  ├── *cu_*threads/               (individual config results + summaries)
  ├── comparison_results/         (pairwise comparisons with --skip-tracelens)
  ├── cross_timestamp_comparison/ (comparison with previous run)
  └── compare_all_runs/           (optional, if --no-skip-compare-all)

Skipped Stages:
  - Docker Setup: {config.skip.docker_setup}
  - RCCL Build: {config.skip.rccl_build}
  - Install Dependencies: {config.skip.install_deps}
  - Performance Tests: {config.skip.performance_tests}
  - Pairwise Analysis: {config.skip.pairwise_analysis}
  - Compare-All Analysis: {config.skip.compare_all_analysis}
  - Checkout aorta-report: {config.skip.checkout_aorta_report}
  - Cross-Timestamp Comparison: {config.skip.cross_timestamp_comparison}
  - Push Results: {config.skip.push_results}
  - Cleanup: {config.skip.cleanup}

================================================================================
"""
    
    summary_file.write_text(summary_content)
    
    # Print to console
    logger.info("\n" + summary_content)
    logger.info(f"Summary saved to: {summary_file}")
    return True


def stage_push_results(config: Config, logger: logging.Logger) -> bool:
    """Stage 11: Push results to aorta-report repository."""
    if config.skip.push_results:
        logger.info("Skipping push to aorta-report (--skip-push is default)")
        return True
    
    logger.info("=" * 60)
    logger.info("Stage 11: Pushing results to aorta-report")
    logger.info("=" * 60)
    
    # Use the aorta-report directory from Stage 8 (checkout), or fallback to default path
    if config.aorta_report_dir and config.aorta_report_dir.exists():
        aorta_report_dir = config.aorta_report_dir
        logger.info(f"Using already checked-out aorta-report: {aorta_report_dir}")
    else:
        aorta_report_dir = Path(config.cross_timestamp.aorta_report_path)
        # Clone or update aorta-report if not already done
        if aorta_report_dir.exists():
            logger.info("Updating existing aorta-report clone...")
            run_command("git pull --rebase origin main", logger, cwd=aorta_report_dir)
        else:
            logger.info("Cloning aorta-report repository...")
            run_command(
                f"git clone git@github.com:ROCm/aorta-report.git {aorta_report_dir}",
                logger
            )
    
    date_dir = datetime.now().strftime('%Y-%m-%d')
    
    # Create date directory and copy results
    target_dir = aorta_report_dir / date_dir / "rccl-warp-speed"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.copytree(config.experiment_dir, target_dir, dirs_exist_ok=True)
    
    # Commit and push
    run_command(
        'git config user.name "Weekly CI Kickoff Script"',
        logger, cwd=aorta_report_dir
    )
    run_command(
        'git config user.email "noreply@amd.com"',
        logger, cwd=aorta_report_dir
    )
    run_command(f"git add {date_dir}", logger, cwd=aorta_report_dir)
    run_command(
        f'git commit -m "Add RCCL warp speed results for {date_dir}"',
        logger, cwd=aorta_report_dir
    )
    run_command("git push origin main", logger, cwd=aorta_report_dir)
    
    logger.info("Results pushed to aorta-report")
    return True


def stage_cleanup(config: Config, logger: logging.Logger) -> bool:
    """Stage 12: Cleanup Docker container."""
    if config.skip.cleanup:
        logger.info("Skipping cleanup (container left running)")
        logger.info(f"  To manually cleanup: docker compose -f {config.docker.compose_file} down")
        return True
    
    logger.info("=" * 60)
    logger.info("Stage 12: Cleaning up Docker container")
    logger.info("=" * 60)
    
    run_command(
        f"docker compose -f {config.docker.compose_file} down",
        logger,
        check=False
    )
    
    logger.info("Cleanup completed")
    return True


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    """Main entry point."""
    # Parse arguments first (needed for config file path)
    args = parse_args()
    
    # Load configuration
    config = Config()
    yaml_data = load_config_file(args.config)
    config = merge_config(config, yaml_data, args)
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Weekly CI Kickoff - RCCL Warp Speed Performance Analysis")
    logger.info("=" * 60)
    logger.info("")
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Config file: {args.config}")
    logger.info(f"  Config pairs: {config.test.config_pairs}")
    logger.info(f"  Baseline: {config.test.baseline}")
    logger.info(f"  Training config: {config.test.training_config}")
    logger.info(f"  GPU target: {config.rccl.gpu_target}")
    logger.info(f"  RCCL branch: {config.rccl.branch}")
    logger.info(f"  Docker compose: {config.docker.compose_file}")
    logger.info(f"  Container: {config.docker.container_name}")
    logger.info("")
    logger.info("Skip settings:")
    logger.info(f"  Docker setup: {config.skip.docker_setup}")
    logger.info(f"  RCCL build: {config.skip.rccl_build}")
    logger.info(f"  Install deps: {config.skip.install_deps}")
    logger.info(f"  Performance tests: {config.skip.performance_tests}")
    logger.info(f"  Pairwise analysis: {config.skip.pairwise_analysis}")
    logger.info(f"  Compare-all analysis: {config.skip.compare_all_analysis}")
    logger.info(f"  Checkout aorta-report: {config.skip.checkout_aorta_report}")
    logger.info(f"  Cross-timestamp comparison: {config.skip.cross_timestamp_comparison}")
    logger.info(f"  Push results: {config.skip.push_results}")
    logger.info(f"  Cleanup: {config.skip.cleanup}")
    logger.info("")
    logger.info("Cross-timestamp settings:")
    logger.info(f"  Baseline experiment: {config.cross_timestamp.baseline_experiment or '(auto-detect)'}")
    logger.info(f"  aorta-report path: {config.cross_timestamp.aorta_report_path}")
    logger.info("")
    
    # Run pipeline stages
    stages = [
        ("Validate Environment", stage_validate_environment),
        ("Docker Setup", stage_docker_setup),
        ("Build RCCL", stage_build_rccl),
        ("Install Dependencies", stage_install_dependencies),
        ("Run Performance Tests", stage_run_performance_tests),
        ("Pairwise Analysis", stage_pairwise_analysis),
        ("Compare-All Analysis", stage_compare_all_analysis),
        ("Checkout aorta-report", stage_checkout_aorta_report),
        ("Cross-Timestamp Comparison", stage_cross_timestamp_comparison),
        ("Generate Summary", stage_generate_summary),
        ("Push Results", stage_push_results),
        ("Cleanup", stage_cleanup),
    ]
    
    for stage_name, stage_func in stages:
        try:
            if not stage_func(config, logger):
                logger.error(f"Stage '{stage_name}' failed")
                return 1
        except Exception as e:
            logger.exception(f"Stage '{stage_name}' raised an exception: {e}")
            return 1
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Weekly CI Kickoff Complete!")
    if config.experiment_dir:
        logger.info(f"Results: {config.experiment_dir}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## Default Configuration File

**Location:** `config/weekly_ci.yaml`

```yaml
# Weekly CI Kickoff Configuration
# ================================
# This file provides default values for the weekly CI kickoff script.
# Command-line arguments always take precedence over these values.

# RCCL build configuration
rccl:
  branch: "warp_speed_v1"
  gpu_target: "gfx950"

# Test configuration
test:
  config_pairs: "56,256 37,384 32,512"
  baseline: "56,256"
  training_config: "config/single_node/gemm_overlap_comm.yaml"

# Docker settings
docker:
  compose_file: "docker/rccl_test/docker-compose.rocm70_9-1.yaml"
  container_name: "training-overlap-bugs-rocm70_9-1"

# Stage control
# Set to true to skip the corresponding stage
skip:
  docker_setup: false
  rccl_build: false
  install_deps: false
  performance_tests: false
  pairwise_analysis: false
  compare_all_analysis: true       # Skip by default for initial setup
  checkout_aorta_report: false     # Needed for cross-timestamp comparison
  cross_timestamp_comparison: false
  convert_html_to_md: false         # Convert HTML to Markdown before push (set true to skip)
  push_results: true               # Skip by default (requires SSH access)
  cleanup: true                    # Skip by default (leave container running)

# Cross-timestamp comparison settings
cross_timestamp:
  # Previous experiment directory to compare against (optional)
  # If not specified, auto-detects the second-most-recent experiment
  baseline_experiment: ""
  # Date directory in aorta-report for baseline (e.g., "2026-02-19")
  baseline_date: ""
  # Path to aorta-report repository (for finding previous runs)
  aorta_report_path: "../aorta-report"

# Analysis labels (for aorta-report commands)
analysis:
  baseline_label: ""    # e.g., "baseline", "v1.0"
  test_label: ""        # e.g., "test", "v1.1"
  report_label: ""      # Override aorta-report dir & dashboard (default: date from experiment)

# Output settings
output:
  log_dir: "logs"
  log_level: "INFO"     # DEBUG, INFO, WARNING, ERROR
```

---

## Usage Examples

### Basic Usage

```bash
# Run with config file defaults
python scripts/weekly_ci_kickoff.py

# View help
python scripts/weekly_ci_kickoff.py --help
```

### Override Configuration

```bash
# Override config pairs and baseline
python scripts/weekly_ci_kickoff.py \
    --config-pairs "56,256 42,384 28,512" \
    --baseline "56,256"

# Use different GPU target
python scripts/weekly_ci_kickoff.py --gpu-target gfx942

# Use different RCCL branch
python scripts/weekly_ci_kickoff.py --rccl-branch warp_speed_v2
```

### Skip Stages

```bash
# Skip RCCL rebuild (use existing build)
python scripts/weekly_ci_kickoff.py --skip-rccl-build

# Skip Docker setup (container already running)
python scripts/weekly_ci_kickoff.py --skip-docker-setup

# Skip multiple stages (e.g., re-run analysis only)
python scripts/weekly_ci_kickoff.py \
    --skip-docker-setup \
    --skip-rccl-build \
    --skip-install-deps \
    --skip-performance-tests

# Enable compare-all analysis (skipped by default)
python scripts/weekly_ci_kickoff.py --no-skip-compare-all

# Skip cross-timestamp comparison
python scripts/weekly_ci_kickoff.py --skip-cross-timestamp

# Run with cleanup
python scripts/weekly_ci_kickoff.py --cleanup
```

### Cross-Timestamp Comparison

```bash
# Auto-detect previous experiment (uses second-most-recent)
python scripts/weekly_ci_kickoff.py

# Specify explicit baseline experiment for comparison
python scripts/weekly_ci_kickoff.py \
    --baseline-experiment experiments/rccl_warp_speed_20260218_120000

# Use custom aorta-report path
python scripts/weekly_ci_kickoff.py \
    --aorta-report-path /path/to/aorta-report

# Override label for aorta-report directory and dashboard (default: date from experiment dir)
python scripts/weekly_ci_kickoff.py --report-label "v1.2.3" ...
python scripts/weekly_ci_kickoff.py --report-label "2026-02-24" ...

# Skip cross-timestamp comparison entirely
python scripts/weekly_ci_kickoff.py --skip-cross-timestamp
```

### Custom Config File

```bash
# Use a completely different configuration
python scripts/weekly_ci_kickoff.py --config config/my_custom_ci.yaml

# Use custom config but override one value
python scripts/weekly_ci_kickoff.py --config config/my_custom_ci.yaml --gpu-target gfx942
```

### Logging Control

```bash
# Enable debug logging
python scripts/weekly_ci_kickoff.py --log-level DEBUG

# Save logs to custom directory
python scripts/weekly_ci_kickoff.py --log-dir /tmp/ci_logs
```

---

## Log Output

The script produces logs in two formats:

### Console Output (Colored)

```
INFO | Logging to: logs/weekly_ci_20260219_143022.log
INFO | ============================================================
INFO | Weekly CI Kickoff - RCCL Warp Speed Performance Analysis
INFO | ============================================================
INFO | Configuration:
INFO |   Config file: config/weekly_ci.yaml
INFO |   Config pairs: 56,256 37,384 32,512
...
```

### Log File (Detailed)

```
2026-02-19 14:30:22 | INFO     | Logging to: logs/weekly_ci_20260219_143022.log
2026-02-19 14:30:22 | INFO     | ============================================================
2026-02-19 14:30:22 | INFO     | Weekly CI Kickoff - RCCL Warp Speed Performance Analysis
2026-02-19 14:30:22 | DEBUG    | Running: docker --version
2026-02-19 14:30:22 | DEBUG    | stdout: Docker version 24.0.7, build afdd53b
...
```

---

## Dependencies

- **Python 3.10+**
- **PyYAML** (`pip install pyyaml`)
- **Docker** with Docker Compose v2
- **Git** (for aorta-report push functionality)
- **GPU** with ROCm support (gfx950, gfx942, etc.)

---

## aorta-report Repository README

The following README template should be placed in the `aorta-report` repository to provide quick access to results and a summary dashboard.

### README Template for aorta-report

````markdown
# RCCL Warp Speed Performance Results

Automated performance tracking for RCCL warp speed configurations on AMD GPUs.

## Quick Start - Running the Weekly CI

```bash
# Clone the aorta repository
git clone git@github.com:ROCm/aorta.git
cd aorta

# Install dependencies
pip install -e .
pip install pyyaml

# Run with defaults (uses config/weekly_ci.yaml)
python scripts/weekly_ci_kickoff.py

# Run with custom configurations
python scripts/weekly_ci_kickoff.py \
    --config-pairs "56,256 37,384 32,512" \
    --baseline "56,256" \
    --gpu-target gfx950

# Skip stages for faster iteration
python scripts/weekly_ci_kickoff.py \
    --skip-docker-setup \
    --skip-rccl-build \
    --skip-install-deps

# Specify explicit baseline for cross-timestamp comparison
python scripts/weekly_ci_kickoff.py \
    --baseline-experiment experiments/rccl_warp_speed_20260218_120000
```

For detailed documentation, see [rccl-warp-speed-standalone.md](https://github.com/ROCm/aorta/blob/main/docs/rccl-warp-speed-standalone.md).

---

## Summary Dashboard

Performance trends comparing each day's run against the previous day (cross-timestamp comparison).

### Legend
- 🟢 **Improvement** (>2% faster)
- 🟡 **Neutral** (within ±2%)
- 🔴 **Regression** (>2% slower)
- ⚪ **No Data** (baseline not available)

| Date | 56cu_256t | 37cu_384t | 32cu_512t | Overall | Details |
|------|-----------|-----------|-----------|---------|---------|
| 2026-02-19 | 🟢 +3.2% | 🟡 +0.8% | 🟢 +2.5% | 🟢 +2.2% | [View](2026-02-19/rccl-warp-speed/) |
| 2026-02-17 | 🟡 -0.5% | 🟢 +4.1% | 🟡 +1.2% | 🟢 +1.6% | [View](2026-02-17/rccl-warp-speed/) |
| 2026-02-15 | 🔴 -3.8% | 🟡 +0.2% | 🔴 -2.9% | 🔴 -2.2% | [View](2026-02-15/rccl-warp-speed/) |
| 2026-02-13 | 🟢 +5.1% | 🟢 +3.7% | 🟢 +4.2% | 🟢 +4.3% | [View](2026-02-13/rccl-warp-speed/) |
| 2026-02-11 | ⚪ N/A | ⚪ N/A | ⚪ N/A | ⚪ N/A | [View](2026-02-11/rccl-warp-speed/) |

### Metrics Tracked
- **GEMM Throughput**: Matrix multiplication performance (TFLOPS)
- **NCCL AllReduce**: Collective communication latency (μs)
- **Overlap Ratio**: Compute-communication overlap efficiency (%)
- **Iteration Time**: End-to-end training step time (ms)

### Understanding Improvement vs Degradation

**Important**: Different metrics have different "good" directions. The dashboard normalizes 
all percentages so that **positive (+%) always means improvement**.

| Metric | Raw Direction | Improvement | Degradation |
|--------|---------------|-------------|-------------|
| **GEMM Throughput** (TFLOPS) | Higher is better ↑ | +% = faster compute | -% = slower |
| **NCCL AllReduce** (μs) | Lower is better ↓ | +% = faster (less latency) | -% = slower |
| **Overlap Ratio** (%) | Higher is better ↑ | +% = better overlap | -% = worse |
| **Iteration Time** (ms) | Lower is better ↓ | +% = faster (less time) | -% = slower |

**Normalization Logic**:
- For **throughput/efficiency** metrics (higher is better): `change = (test - baseline) / baseline`
- For **latency/time** metrics (lower is better): `change = (baseline - test) / baseline`

This ensures the dashboard always shows:
- **Positive (+%)** = Things got faster/better 🟢
- **Negative (-%)** = Things got slower/worse 🔴

**Example**:
```
NCCL AllReduce: baseline=125.4μs, test=121.2μs
  Raw change: (121.2 - 125.4) / 125.4 = -3.3% (latency decreased)
  Normalized: +3.3% improvement (faster communication) 🟢

Iteration Time: baseline=12.45ms, test=12.85ms  
  Raw change: (12.85 - 12.45) / 12.45 = +3.2% (time increased)
  Normalized: -3.2% regression (slower iteration) 🔴
```

---

## Directory Structure

```
aorta-report/
├── README.md                      # This file
├── 2026-02-19/
│   └── rccl-warp-speed/
│       ├── 56cu_256threads/
│       ├── 37cu_384threads/
│       ├── 32cu_512threads/
│       ├── comparison_results/
│       ├── cross_timestamp_comparison/
│       └── summary.txt
├── 2026-02-17/
│   └── rccl-warp-speed/
│       └── ...
└── ...
```

---

## Recent Highlights

### 2026-02-19
- **RCCL Branch**: warp_speed_v1
- **Notable**: 3.2% improvement in 56cu_256t configuration
- **Regression**: None detected

### 2026-02-15
- **RCCL Branch**: warp_speed_v1
- **Notable**: Regression in 56cu_256t and 32cu_512t configs
- **Root Cause**: [Link to investigation]

---

## How Results Are Generated

1. **Performance Tests**: Run via `weekly_ci_kickoff.py` every other day
2. **Pairwise Analysis**: Each config compared against baseline (56cu_256t)
3. **Cross-Timestamp Comparison**: Current run compared to previous run
4. **Summary Generation**: Automated metrics extraction and dashboard update

The cross-timestamp comparison (Step 9) generates the data for this dashboard by comparing:
- **Baseline**: Previous day's experiment directory
- **Test**: Current day's experiment directory

```bash
aorta-report pipeline summary \
    --baseline experiments/rccl_warp_speed_20260217_120000/56cu_256threads \
    --test experiments/rccl_warp_speed_20260219_120000/56cu_256threads \
    --skip-tracelens
```
````

---

## Sample Summary Dashboard (ASCII)

For terminals or plain-text views, here's an ASCII representation of the dashboard:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               RCCL Warp Speed Performance Summary Dashboard                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Date        │ 56cu_256t │ 37cu_384t │ 32cu_512t │ Overall  │ Status        ║
║  ────────────┼───────────┼───────────┼───────────┼──────────┼───────────────║
║  2026-02-19  │  ▲ +3.2%  │  ● +0.8%  │  ▲ +2.5%  │  ▲ +2.2% │ ✓ PASS        ║
║  2026-02-17  │  ● -0.5%  │  ▲ +4.1%  │  ● +1.2%  │  ▲ +1.6% │ ✓ PASS        ║
║  2026-02-15  │  ▼ -3.8%  │  ● +0.2%  │  ▼ -2.9%  │  ▼ -2.2% │ ⚠ REGRESSION  ║
║  2026-02-13  │  ▲ +5.1%  │  ▲ +3.7%  │  ▲ +4.2%  │  ▲ +4.3% │ ✓ PASS        ║
║  2026-02-11  │    N/A    │    N/A    │    N/A    │    N/A   │ ○ BASELINE    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Legend:  ▲ Improvement (>2%)  ● Neutral (±2%)  ▼ Regression (>2%)           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Detailed Metrics (2026-02-19 vs 2026-02-17):
┌──────────────────────────────────────────────────────────────────────────────┐
│ Configuration: 56cu_256threads                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ Metric              │ Baseline (02-17) │ Current (02-19) │ Change           │
│ ────────────────────┼──────────────────┼─────────────────┼──────────────────│
│ GEMM Throughput     │ 48.2 TFLOPS      │ 49.8 TFLOPS     │ ▲ +3.3%          │
│ NCCL AllReduce      │ 125.4 μs         │ 121.2 μs        │ ▲ +3.3% (faster) │
│ Overlap Ratio       │ 78.5%            │ 81.2%           │ ▲ +2.7%          │
│ Iteration Time      │ 12.45 ms         │ 12.08 ms        │ ▲ +3.0% (faster) │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Sample Dashboard

Below is a comprehensive sample of what the aorta-report README dashboard would look like with real data:

### Sample aorta-report/README.md

```markdown
# 🚀 RCCL Warp Speed Performance Results

> Automated performance tracking for RCCL warp speed configurations on AMD MI300X GPUs.
> 
> **RCCL Branch:** `warp_speed_v1` | **GPU Target:** `gfx950` | **Last Updated:** 2026-02-19 14:30 UTC

---

## 📊 Performance Summary Dashboard

Performance trends comparing each day's run against the previous day.
**Positive % = Improvement** (faster/better) | **Negative % = Regression** (slower/worse)

### Weekly Overview

| Date | 56cu_256t | 37cu_384t | 32cu_512t | Overall | Trend | Status | Report |
|:----:|:---------:|:---------:|:---------:|:-------:|:-----:|:------:|:------:|
| **2026-02-19** | 🟢 +3.2% | 🟡 +0.8% | 🟢 +2.5% | 🟢 **+2.2%** | 📈 | ✅ PASS | [📄](2026-02-19/rccl-warp-speed/) |
| 2026-02-17 | 🟡 -0.5% | 🟢 +4.1% | 🟡 +1.2% | 🟢 +1.6% | 📈 | ✅ PASS | [📄](2026-02-17/rccl-warp-speed/) |
| 2026-02-15 | 🔴 -3.8% | 🟡 +0.2% | 🔴 -2.9% | 🔴 -2.2% | 📉 | ⚠️ REGRESS | [📄](2026-02-15/rccl-warp-speed/) |
| 2026-02-13 | 🟢 +5.1% | 🟢 +3.7% | 🟢 +4.2% | 🟢 +4.3% | 📈 | ✅ PASS | [📄](2026-02-13/rccl-warp-speed/) |
| 2026-02-11 | ⚪ — | ⚪ — | ⚪ — | ⚪ — | 🏁 | 🔵 BASE | [📄](2026-02-11/rccl-warp-speed/) |

**Legend:** 🟢 Improvement (>+2%) | 🟡 Neutral (±2%) | 🔴 Regression (<-2%) | ⚪ Baseline

---

## 📈 Latest Results: 2026-02-19

Comparison with previous run (2026-02-17)

### Configuration: 56cu_256threads 🟢 +3.2%

| Metric | 02-17 (Baseline) | 02-19 (Current) | Change | Status |
|--------|:----------------:|:---------------:|:------:|:------:|
| GEMM Throughput | 48.2 TFLOPS | 49.8 TFLOPS | **+3.3%** | 🟢 |
| NCCL AllReduce | 125.4 μs | 121.2 μs | **+3.3%** ↓ | 🟢 |
| Overlap Ratio | 78.5% | 81.2% | **+3.4%** | 🟢 |
| Iteration Time | 12.45 ms | 12.08 ms | **+3.0%** ↓ | 🟢 |

<details>
<summary>📊 View Raw Data</summary>

```json
{
  "config": "56cu_256threads",
  "baseline_date": "2026-02-17",
  "test_date": "2026-02-19",
  "metrics": {
    "gemm_throughput": {"baseline": 48.2, "test": 49.8, "unit": "TFLOPS", "change_pct": 3.32},
    "nccl_allreduce": {"baseline": 125.4, "test": 121.2, "unit": "μs", "change_pct": 3.35},
    "overlap_ratio": {"baseline": 78.5, "test": 81.2, "unit": "%", "change_pct": 3.44},
    "iteration_time": {"baseline": 12.45, "test": 12.08, "unit": "ms", "change_pct": 2.97}
  },
  "overall_improvement_pct": 3.27
}
```
</details>

---

### Configuration: 37cu_384threads 🟡 +0.8%

| Metric | 02-17 (Baseline) | 02-19 (Current) | Change | Status |
|--------|:----------------:|:---------------:|:------:|:------:|
| GEMM Throughput | 52.1 TFLOPS | 52.4 TFLOPS | **+0.6%** | 🟡 |
| NCCL AllReduce | 118.7 μs | 117.8 μs | **+0.8%** ↓ | 🟡 |
| Overlap Ratio | 82.3% | 83.1% | **+1.0%** | 🟡 |
| Iteration Time | 11.82 ms | 11.75 ms | **+0.6%** ↓ | 🟡 |

---

### Configuration: 32cu_512threads 🟢 +2.5%

| Metric | 02-17 (Baseline) | 02-19 (Current) | Change | Status |
|--------|:----------------:|:---------------:|:------:|:------:|
| GEMM Throughput | 54.8 TFLOPS | 56.2 TFLOPS | **+2.6%** | 🟢 |
| NCCL AllReduce | 112.3 μs | 109.5 μs | **+2.5%** ↓ | 🟢 |
| Overlap Ratio | 84.1% | 86.3% | **+2.6%** | 🟢 |
| Iteration Time | 11.23 ms | 10.98 ms | **+2.2%** ↓ | 🟡 |

---

## 📉 Regression Analysis: 2026-02-15

On 2026-02-15, a regression was detected in configurations 56cu_256t and 32cu_512t.

### Root Cause Analysis

| Factor | Status | Notes |
|--------|--------|-------|
| RCCL Version | ✅ Same | warp_speed_v1 |
| ROCm Version | ⚠️ Updated | 7.0.8 → 7.0.9 |
| PyTorch Version | ✅ Same | 2.6.0.dev |
| Kernel Changes | ⚠️ Possible | New GEMM kernel path |

**Action Items:**
- [ ] Bisect ROCm changes between 7.0.8 and 7.0.9
- [ ] Profile GEMM kernel selection logic
- [ ] Compare trace files for 56cu_256t configuration

---

## 🔧 Quick Start

```bash
# Clone aorta and run weekly CI
git clone git@github.com:ROCm/aorta.git && cd aorta
pip install -e . && pip install pyyaml
python scripts/weekly_ci_kickoff.py

# Compare with specific baseline
python scripts/weekly_ci_kickoff.py \
    --baseline-experiment experiments/rccl_warp_speed_20260217_120000
```

📚 [Full Documentation](https://github.com/ROCm/aorta/blob/main/docs/rccl-warp-speed-standalone.md)

---

## 📁 Directory Structure

```
aorta-report/
├── README.md                         ← You are here
├── 2026-02-19/
│   └── rccl-warp-speed/
│       ├── 56cu_256threads/
│       │   ├── summary/
│       │   └── traces/
│       ├── 37cu_384threads/
│       ├── 32cu_512threads/
│       ├── comparison_results/       ← Pairwise within-experiment
│       ├── cross_timestamp_comparison/ ← vs previous day
│       └── summary.txt
└── 2026-02-17/
    └── ...
```
```

---

## Sample HTML Dashboard Preview

The dashboard can also be rendered as an HTML page with interactive charts:

```html
<!-- Sample structure for HTML dashboard -->
<!DOCTYPE html>
<html>
<head>
    <title>RCCL Warp Speed Performance Dashboard</title>
    <style>
        .dashboard { font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; }
        .metric-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin: 8px; }
        .improvement { color: #2e7d32; background: #e8f5e9; }
        .regression { color: #c62828; background: #ffebee; }
        .neutral { color: #f57c00; background: #fff3e0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: center; border-bottom: 1px solid #e0e0e0; }
        th { background: #f5f5f5; font-weight: 600; }
        .trend-up::before { content: "▲ "; color: #2e7d32; }
        .trend-down::before { content: "▼ "; color: #c62828; }
        .trend-flat::before { content: "● "; color: #f57c00; }
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>🚀 RCCL Warp Speed Performance Dashboard</h1>
        <p>Last updated: 2026-02-19 14:30 UTC</p>
        
        <h2>Performance Trend</h2>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>56cu_256t</th>
                    <th>37cu_384t</th>
                    <th>32cu_512t</th>
                    <th>Overall</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>2026-02-19</td>
                    <td class="improvement trend-up">+3.2%</td>
                    <td class="neutral trend-flat">+0.8%</td>
                    <td class="improvement trend-up">+2.5%</td>
                    <td class="improvement trend-up">+2.2%</td>
                    <td>✅ PASS</td>
                </tr>
                <!-- More rows... -->
            </tbody>
        </table>
        
        <h2>Configuration Details</h2>
        <div class="metric-card improvement">
            <h3>56cu_256threads</h3>
            <p><strong>GEMM:</strong> 49.8 TFLOPS (+3.3%)</p>
            <p><strong>NCCL:</strong> 121.2 μs (+3.3% faster)</p>
            <p><strong>Overlap:</strong> 81.2% (+2.7%)</p>
        </div>
    </div>
</body>
</html>
```

---

## Automated Dashboard Update

The `weekly_ci_kickoff.py` script can be extended to automatically update the dashboard.
Add these functions to Stage 11 (Push Results) to calculate and generate the dashboard entry:

### Metric Calculation Logic

```python
# Metric type definitions
LOWER_IS_BETTER = [
    "nccl_latency", "nccl_allreduce", "allreduce_time", "allreduce_latency",
    "iteration_time", "step_time", "latency", "time_ms", "time_us"
]

HIGHER_IS_BETTER = [
    "gemm_throughput", "tflops", "throughput", "bandwidth", "gbps",
    "overlap_ratio", "overlap_efficiency", "efficiency"
]


def calculate_improvement(metric_name: str, baseline_value: float, test_value: float) -> float:
    """
    Calculate improvement percentage with normalization.
    
    Returns:
        Positive value = improvement (things got better/faster)
        Negative value = regression (things got worse/slower)
    
    Args:
        metric_name: Name of the metric (used to determine direction)
        baseline_value: Value from the previous/baseline run
        test_value: Value from the current/test run
    """
    if baseline_value == 0:
        return 0.0
    
    # Normalize metric name for lookup
    metric_lower = metric_name.lower().replace(" ", "_").replace("-", "_")
    
    # Calculate raw percentage change
    raw_change = (test_value - baseline_value) / baseline_value * 100
    
    # Determine if this metric is "lower is better"
    is_lower_better = any(pattern in metric_lower for pattern in LOWER_IS_BETTER)
    
    if is_lower_better:
        # For latency/time: decrease is improvement, so invert the sign
        # baseline=125μs, test=121μs → raw=-3.2% → normalized=+3.2% (improvement)
        return -raw_change
    else:
        # For throughput: increase is improvement, keep the sign
        # baseline=48 TFLOPS, test=50 TFLOPS → raw=+4.2% → normalized=+4.2% (improvement)
        return raw_change


def calculate_overall_improvement(metrics: dict[str, tuple[float, float]]) -> float:
    """
    Calculate overall improvement across multiple metrics.
    
    Args:
        metrics: Dict of metric_name -> (baseline_value, test_value)
    
    Returns:
        Average normalized improvement percentage
    """
    improvements = []
    for metric_name, (baseline, test) in metrics.items():
        imp = calculate_improvement(metric_name, baseline, test)
        improvements.append(imp)
    
    return sum(improvements) / len(improvements) if improvements else 0.0


# Example usage:
# metrics = {
#     "gemm_throughput": (48.2, 49.8),    # TFLOPS, higher is better
#     "nccl_allreduce": (125.4, 121.2),   # μs, lower is better
#     "overlap_ratio": (78.5, 81.2),       # %, higher is better
#     "iteration_time": (12.45, 12.08),   # ms, lower is better
# }
# 
# for name, (base, test) in metrics.items():
#     imp = calculate_improvement(name, base, test)
#     print(f"{name}: {imp:+.1f}%")
#
# Output:
#   gemm_throughput: +3.3%
#   nccl_allreduce: +3.3%
#   overlap_ratio: +3.4%
#   iteration_time: +3.0%
```

### Dashboard Entry Generator

```python
def generate_dashboard_entry(config: Config, logger: logging.Logger) -> str:
    """Generate a dashboard row for the current run."""
    from datetime import datetime
    import json
    
    date_str = datetime.now().strftime('%Y-%m-%d')
    results = {}
    
    # Parse cross-timestamp comparison results
    cross_ts_dir = Path(config.experiment_dir) / "cross_timestamp_comparison"
    
    for config_dir in cross_ts_dir.iterdir():
        if config_dir.is_dir():
            summary_file = config_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                
                # Extract metrics and calculate normalized improvement
                metrics = {}
                for metric_name in ["gemm_throughput", "nccl_allreduce", "overlap_ratio", "iteration_time"]:
                    if f"{metric_name}_baseline" in data and f"{metric_name}_test" in data:
                        metrics[metric_name] = (
                            data[f"{metric_name}_baseline"],
                            data[f"{metric_name}_test"]
                        )
                
                # Calculate overall improvement for this configuration
                if metrics:
                    overall_imp = calculate_overall_improvement(metrics)
                    results[config_dir.name] = overall_imp
    
    # Determine status emoji based on normalized improvement
    def get_emoji(pct: float) -> str:
        if pct > 2:
            return "🟢"  # Improvement
        elif pct < -2:
            return "🔴"  # Regression
        else:
            return "🟡"  # Neutral
    
    # Build table row
    configs = ["56cu_256threads", "37cu_384threads", "32cu_512threads"]
    cells = []
    for cfg in configs:
        pct = results.get(cfg, 0)
        emoji = get_emoji(pct)
        cells.append(f"{emoji} {pct:+.1f}%")
    
    overall = sum(results.values()) / len(results) if results else 0
    overall_emoji = get_emoji(overall)
    
    row = f"| {date_str} | {' | '.join(cells)} | {overall_emoji} {overall:+.1f}% | [View]({date_str}/rccl-warp-speed/) |"
    
    return row


def update_readme_dashboard(aorta_report_dir: Path, new_row: str, logger: logging.Logger) -> None:
    """Insert new dashboard row into aorta-report README."""
    readme_path = aorta_report_dir / "README.md"
    
    if not readme_path.exists():
        logger.error(f"README not found: {readme_path}")
        return
    
    content = readme_path.read_text()
    
    # Find the dashboard table and insert the new row after the header
    # Table format: | Date | 56cu_256t | 37cu_384t | 32cu_512t | Overall | Details |
    marker = "|------|-----------|-----------|-----------|---------|---------|"
    
    if marker in content:
        # Insert new row right after the header separator
        content = content.replace(marker, f"{marker}\n{new_row}")
        readme_path.write_text(content)
        logger.info(f"Dashboard updated with new row: {new_row}")
    else:
        logger.warning("Could not find dashboard table marker in README")
```

---

## Implementation Plan

### Phase 1: Foundation (Days 1-2)

#### Task 1.1: Create Project Structure
```
scripts/
├── weekly_ci_kickoff.py          # Main script
├── weekly_ci/                    # Package directory
│   ├── __init__.py
│   ├── config.py                 # Configuration dataclasses
│   ├── logging_setup.py          # Logging utilities
│   ├── utils.py                  # run_command, docker_exec
│   └── stages/                   # Stage implementations
│       ├── __init__.py
│       ├── validate.py           # Stage 1
│       ├── docker.py             # Stage 2
│       ├── rccl.py               # Stage 3
│       ├── dependencies.py       # Stage 4
│       ├── performance.py        # Stage 5
│       ├── analysis.py           # Stages 6, 7
│       ├── cross_timestamp.py    # Stages 8, 9
│       ├── summary.py            # Stage 10
│       ├── push.py               # Stage 11
│       └── cleanup.py            # Stage 12

config/
└── weekly_ci.yaml                # Default configuration
```

**Deliverables:**
- [x] Create directory structure
- [x] Create `config.py` with all dataclasses
- [x] Create `logging_setup.py` with ColoredFormatter
- [x] Create `utils.py` with run_command and docker_exec
- [x] Create default `config/weekly_ci.yaml`

**Note:** Actual file structure differs slightly from plan:
- `rccl.py` + `dependencies.py` → combined into `build.py`
- `performance.py` → renamed to `test.py`
- `cross_timestamp.py` → merged into `analysis.py` and `repository.py`
- `summary.py` → renamed to `reporting.py`
- `push.py` → merged into `repository.py`
- `cleanup.py` → merged into `docker.py`

---

#### Task 1.2: Configuration System
**File:** `scripts/weekly_ci/config.py`

```python
# Implementation checklist:
- [x] RCCLConfig dataclass
- [x] TestConfig dataclass  
- [x] DockerConfig dataclass
- [x] SkipConfig dataclass
- [x] CrossTimestampConfig dataclass
- [x] OutputConfig dataclass
- [x] Config main dataclass
- [x] load_config_file() function
- [x] merge_config() function
- [x] parse_args() function
- [x] AnalysisConfig dataclass (additional)
- [x] GitConfig dataclass (additional)
```

**Testing:**
```bash
# Test config loading
python -c "from weekly_ci.config import Config, load_config_file; print(load_config_file('config/weekly_ci.yaml'))"
```

---

### Phase 2: Core Stages (Days 3-5)

#### Task 2.1: Stage 1 - Validate Environment
**File:** `scripts/weekly_ci/stages/validate.py`

```python
# Implementation checklist:
- [x] Check Docker installed
- [x] Check Docker daemon running
- [x] Check docker compose available
- [x] Check in aorta repo root (pyproject.toml, src/aorta)
- [x] Check docker-compose file exists
- [x] Check training config exists
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/validate.py`

**Testing:**
```bash
# Should pass in aorta root
python -c "from weekly_ci.stages.validate import stage_validate_environment; ..."

# Should fail outside aorta root
cd /tmp && python -c "..."  # Should error
```

---

#### Task 2.2: Stage 2 - Docker Setup
**File:** `scripts/weekly_ci/stages/docker.py`

```python
# Implementation checklist:
- [x] Check skip flag
- [x] Stop existing container (ignore errors)
- [x] Remove existing container (ignore errors)
- [x] docker compose down (ignore errors)
- [x] docker compose build (conditional on --docker-build flag)
- [x] docker compose up -d
- [x] Wait for container ready (5s sleep)
- [x] Verify container is running
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/docker.py`

**Testing:**
```bash
# Manual test with actual Docker
python scripts/weekly_ci_kickoff.py --skip-rccl-build --skip-install-deps --skip-performance-tests ...
```

---

#### Task 2.3: Stage 3 - Build RCCL
**File:** `scripts/weekly_ci/stages/build.py` (was planned as `rccl.py`)

```python
# Implementation checklist:
- [x] Check skip flag
- [x] Create /rccl directory in container
- [x] Clone or update rccl repo
- [x] Checkout specified branch
- [x] Run install.sh with GPU target
- [x] Verify build output exists
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/build.py` (`stage_build_rccl`)

**Dependencies:** Stage 2 (Docker must be running)

**Testing:**
```bash
# This is a long-running operation - test with --skip-rccl-build first
# Then do a full run to verify
```

---

#### Task 2.4: Stage 4 - Install Dependencies  
**File:** `scripts/weekly_ci/stages/build.py` (was planned as `dependencies.py`)

```python
# Implementation checklist:
- [x] Check skip flag
- [x] pip install -e . (current package)
- [x] pip install -r requirements.txt
- [x] pip install pandas openpyxl matplotlib seaborn numpy
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/build.py` (`stage_install_dependencies`)

**Dependencies:** Stage 2

---

#### Task 2.5: Stage 5 - Run Performance Tests
**File:** `scripts/weekly_ci/stages/test.py` (was planned as `performance.py`)

```python
# Implementation checklist:
- [x] Check skip flag
- [x] Set LD_LIBRARY_PATH for RCCL
- [x] Run run_rccl_warp_speed_comparison.sh
- [x] Find experiment directory (most recent rccl_warp_speed_*)
- [x] Store in config.experiment_dir
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/test.py` (`stage_run_performance_tests`, `stage_find_experiment_dir`)

**Dependencies:** Stages 3, 4

---

### Phase 3: Analysis Stages (Days 6-8)

#### Task 3.1a: Stage 6 - Single Config Analysis
**File:** `scripts/weekly_ci/stages/analysis.py`

```python
# Implementation checklist:
- [x] Check skip flag (skip.single_config_analysis)
- [x] Loop through config_pairs
- [x] Run: aorta-report pipeline summary --test <dir> --output <dir>
- [x] --skip-tracelens: only when --skip-tracelens CLI passed (default: run full TraceLens)
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/analysis.py` (`stage_single_config_analysis`)

**Dependencies:** Stage 5

---

#### Task 3.1b: Stage 7 - Pairwise Comparison
**File:** `scripts/weekly_ci/stages/analysis.py`

```python
# Implementation checklist:
- [x] Check skip flag (skip.pairwise_comparison)
- [x] Parse baseline configuration
- [x] Loop through non-baseline configs
- [x] Run: aorta-report pipeline summary --baseline <base> --test <test> --skip-tracelens (always)
- [x] Support --baseline-label and --test-label arguments
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/analysis.py` (`stage_pairwise_comparison`)

**Dependencies:** Stage 5, 6 (single-config populates tracelens_analysis for pairwise --skip-tracelens)

**Testing:**
```bash
# Test with existing experiment directory
python scripts/weekly_ci_kickoff.py \
    --skip-docker-setup --skip-rccl-build --skip-install-deps --skip-performance-tests
```

---

#### Task 3.2: Stage 7 - Compare-All Analysis
**File:** `scripts/weekly_ci/stages/analysis.py` (same file)

```python
# Implementation checklist:
- [x] Check skip flag (default: skipped)
- [x] Ensure experiment_dir is set
- [x] Build list of test directories
- [x] Run: python run_full_analysis.py --baseline --test <multiple> --skip-tracelens --compare-all-runs
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/analysis.py` (`stage_compare_all_analysis`)

**Note:** This stage is skipped by default for initial setup.

---

### Phase 4: Cross-Timestamp Comparison (Days 9-10)

#### Task 4.1: Stage 8 - Checkout aorta-report
**File:** `scripts/weekly_ci/stages/repository.py` (was planned as `cross_timestamp.py`)

```python
# Implementation checklist:
- [x] Check skip flag
- [x] Check if aorta_report_path exists
- [x] If exists: git fetch && git pull --rebase
- [x] If not exists: git clone
- [x] Store resolved path in config.aorta_report_dir
- [x] Handle clone failures gracefully
- [x] Support SSH and HTTPS authentication
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/repository.py` (`stage_checkout_aorta_report`)

**Note:** Default aorta_report_path changed to `.aorta-report` (inside repo) for Docker accessibility.

**Testing:**
```bash
# Test with existing clone
python scripts/weekly_ci_kickoff.py --skip-docker-setup ... --aorta-report-path ../aorta-report

# Test fresh clone (use temp directory)
python scripts/weekly_ci_kickoff.py ... --aorta-report-path /tmp/aorta-report-test
```

---

#### Task 4.2: Stage 9 - Cross-Timestamp Comparison
**File:** `scripts/weekly_ci/stages/analysis.py` + `test.py` (was planned as `cross_timestamp.py`)

```python
# Implementation checklist:
- [x] Check skip flag
- [x] Ensure experiment_dir is set
- [x] Find baseline experiment:
      - If config.cross_timestamp.baseline_experiment set → use it
      - If config.cross_timestamp.baseline_date set → use aorta-report/{date}/rccl-warp-speed
      - Else → auto-detect from aorta-report (most recent date directory)
      - Else → auto-detect (second-most-recent local experiment)
- [x] Handle case: < 2 experiments (skip gracefully)
- [x] Create cross_timestamp_comparison/ output directory
- [x] For each config_pair:
      - Build baseline and test paths
      - Check baseline exists (skip if not)
      - Run: aorta-report pipeline summary --baseline <old> --test <new> --skip-tracelens
- [x] Store baseline_experiment_dir in config
- [x] Auto-generate --baseline-label and --test-label from directory dates
```

**Status:** ✅ Complete - Implemented in:
- `scripts/weekly_ci/stages/analysis.py` (`stage_cross_timestamp_comparison`)
- `scripts/weekly_ci/stages/test.py` (`stage_find_baseline_experiment_dir`)

**Testing:**
```bash
# Need at least 2 experiment directories
ls experiments/rccl_warp_speed_*

# Test auto-detection
python scripts/weekly_ci_kickoff.py --skip-docker-setup ...

# Test explicit baseline
python scripts/weekly_ci_kickoff.py ... --baseline-experiment experiments/rccl_warp_speed_20260218_120000
```

---

### Phase 5: Summary & Push (Days 11-12)

#### Task 5.1: Stage 10 - Generate Summary
**File:** `scripts/weekly_ci/stages/reporting.py` (was planned as `summary.py`)

```python
# Implementation checklist:
- [x] Ensure experiment_dir is set
- [x] Generate summary content with:
      - Run date
      - Configuration details
      - Tested configurations list
      - Cross-timestamp baseline info
      - Generated artifacts list
      - Directory structure (partial)
      - Skipped stages list (not included)
- [x] Write to {experiment_dir}/summary.txt
- [x] Print to console via logger
- [x] Extract key metrics from comparison results
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/reporting.py` (`stage_generate_summary`)

---

#### Task 5.2: Stage 11 - Push to aorta-report
**File:** `scripts/weekly_ci/stages/repository.py` (was planned as `push.py`)

```python
# Implementation checklist:
- [x] Check skip flag (default: skipped)
- [x] Use existing aorta_report_dir or fallback to config path
- [x] Clone/update if needed (handled by stage 8)
- [x] Create date directory: {date}/rccl-warp-speed/
- [x] Copy experiment results (excluding large files like .rpd, .sqlite)
- [x] Git config user.name and user.email
- [x] Git add, commit, push
- [ ] ❌ Generate and insert dashboard entry (functions exist but NOT called)
- [ ] ❌ update_readme_dashboard() NOT IMPLEMENTED
```

**Status:** ⚠️ Partial - Core push implemented, dashboard integration missing

**Additional:** `generate_dashboard_entry()` and `update_dashboard_file()` exist in `reporting.py` but are NOT called.
The planned `update_readme_dashboard()` function is NOT implemented.

---

#### Task 5.3: Stage 12 - Cleanup
**File:** `scripts/weekly_ci/stages/docker.py` (was planned as `cleanup.py`)

```python
# Implementation checklist:
- [x] Check skip flag (default: skipped, container left running)
- [x] Run: docker compose -f {compose_file} down
- [x] Log manual cleanup command if skipped
```

**Status:** ✅ Complete - Implemented in `scripts/weekly_ci/stages/docker.py` (`stage_cleanup`)

---

### Phase 6: Integration & Testing (Days 13-15)

#### Task 6.1: Main Script Assembly
**File:** `scripts/weekly_ci_kickoff.py`

```python
# Implementation checklist:
- [x] Import all modules
- [x] main() function:
      - Parse args
      - Load config file
      - Merge config
      - Setup logging
      - Log configuration
      - Define stages list (13 stages)
      - Execute stages in order
      - Handle failures
      - Return exit code
- [x] if __name__ == "__main__" block
```

**Status:** ✅ Complete - Fully implemented with 13 stages

---

#### Task 6.2: Integration Testing

```bash
# Test 1: Dry run (skip everything)
python scripts/weekly_ci_kickoff.py \
    --skip-docker-setup \
    --skip-rccl-build \
    --skip-install-deps \
    --skip-performance-tests \
    --skip-pairwise-analysis \
    --skip-cross-timestamp \
    --log-level DEBUG

# Test 2: Analysis only (with existing experiment)
python scripts/weekly_ci_kickoff.py \
    --skip-docker-setup \
    --skip-rccl-build \
    --skip-install-deps \
    --skip-performance-tests

# Test 3: Full run (requires GPU and Docker)
python scripts/weekly_ci_kickoff.py

# Test 4: With custom config
python scripts/weekly_ci_kickoff.py --config config/my_test.yaml
```

---

#### Task 6.3: Unit Tests
**File:** `tests/test_weekly_ci.py`

```python
# Test cases:
- [ ] test_config_loading
- [ ] test_config_merge_cli_precedence
- [ ] test_skip_flags
- [ ] test_calculate_improvement_higher_is_better
- [ ] test_calculate_improvement_lower_is_better
- [ ] test_get_emoji_thresholds
- [ ] test_generate_dashboard_entry
```

**Status:** ❌ NOT IMPLEMENTED - No unit tests created yet

---

### Phase 7: Documentation & Polish (Days 16-17)

#### Task 7.1: Create aorta-report README
**File:** `templates/aorta-report-readme-template.md`

- [ ] Copy README template from this document
- [ ] Include sample dashboard
- [ ] Include setup instructions

**Status:** ❌ NOT IMPLEMENTED

#### Task 7.2: Update Main README
**File:** `README.md`

- [ ] Add section for Weekly CI Kickoff
- [ ] Link to detailed documentation

**Status:** ❌ NOT IMPLEMENTED

#### Task 7.3: Add to pyproject.toml
**File:** `pyproject.toml`

```toml
[project.scripts]
weekly-ci = "weekly_ci:main"
```

**Status:** ❌ NOT IMPLEMENTED

---

### Implementation Summary

| Phase | Tasks | Days | Dependencies | Status |
|-------|-------|------|--------------|--------|
| 1. Foundation | Config, Logging, Utils | 1-2 | None | ✅ Complete |
| 2. Core Stages | Stages 1-5 | 3-5 | Phase 1 | ✅ Complete |
| 3. Analysis | Stages 6-7 | 6-8 | Phase 2 | ✅ Complete |
| 4. Cross-Timestamp | Stages 8-9 | 9-10 | Phase 3 | ✅ Complete |
| 5. Summary & Push | Stages 10-12 | 11-12 | Phase 4 | ✅ Complete |
| 6. Integration | Main script, Tests | 13-15 | All phases | ⚠️ Partial (no unit tests) |
| 7. Documentation | README, polish | 16-17 | Phase 6 | ❌ Not Started |

**Total Estimated Time:** 17 days (can be parallelized)

**Current Status (as of 2026-02-23):**
- Core pipeline (13 stages): ✅ Fully functional
- Dashboard integration: ✅ Implemented (reads Excel files)
- Unit tests: ❌ Not created
- Documentation updates: ❌ Pending

---

### Quick Start for Implementation

```bash
# Step 1: Create directory structure
mkdir -p scripts/weekly_ci/stages
touch scripts/weekly_ci/__init__.py
touch scripts/weekly_ci/stages/__init__.py

# Step 2: Create config file
cat > config/weekly_ci.yaml << 'EOF'
# ... (copy from document)
EOF

# Step 3: Start with config.py
# Copy the dataclasses and config loading code from this document

# Step 4: Implement stages one by one
# Test each stage independently before moving to the next

# Step 5: Assemble main script
# Connect all stages and test end-to-end
```

---

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Docker not available | Skip flags allow testing without Docker |
| aorta-report clone fails | Graceful fallback, continue with local results |
| No previous experiment | Skip cross-timestamp comparison gracefully |
| RCCL build fails | Clear error messages, check GPU target |
| aorta-report CLI changes | Abstract CLI calls for easy updates |

---

## TODO / Future Enhancements

### High Priority (Dashboard Integration) - ✅ COMPLETE
- [x] Wire up `stage_update_dashboard()` call in push stage
- [x] Implement `update_readme_dashboard()` to update aorta-report README markdown table
- [x] Extract metrics from cross-timestamp comparison results for dashboard (via Excel parsing)

### Medium Priority (Testing & Documentation)
- [ ] Create unit tests (`tests/test_weekly_ci.py`)
- [ ] Create aorta-report README template (`templates/aorta-report-readme-template.md`)
- [ ] Update main aorta README with Weekly CI section
- [ ] Add `weekly-ci` entry point to pyproject.toml

### Future Enhancements
- [ ] **Enhance aorta-report to output JSON summary** alongside Excel for easier metric extraction
- [ ] Add `--dry-run` mode to preview commands without execution
- [ ] Add `--resume` to continue from last failed stage
- [ ] Add progress bar for long-running operations
- [ ] Support parallel execution of independent analyses
- [ ] Email/Slack notification on completion
- [ ] Web dashboard for viewing results
- [ ] Integration with pytest for result validation
- [ ] Add trend charts (sparklines) in dashboard
