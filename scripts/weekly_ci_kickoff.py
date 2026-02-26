#!/usr/bin/env python3
"""
Weekly CI Kickoff - RCCL Warp Speed Performance Analysis

This script replicates the CI workflow locally for:
- Building Docker containers
- Cloning and building RCCL
- Running performance tests
- Single-config analysis, pairwise comparison, and cross-timestamp comparison
- Pushing results to aorta-report repository

Usage:
    python scripts/weekly_ci_kickoff.py [OPTIONS]

Examples:
    # Run with defaults
    python scripts/weekly_ci_kickoff.py

    # Skip docker and rccl build (container already running)
    python scripts/weekly_ci_kickoff.py --skip-docker-setup --skip-rccl-build

    # Custom config pairs
    python scripts/weekly_ci_kickoff.py --config-pairs "56,256 37,384"

    # Use custom config file
    python scripts/weekly_ci_kickoff.py --config my_config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from weekly_ci import (
    Config,
    load_config_file,
    log_stage_complete,
    log_stage_error,
    log_stage_skip,
    log_stage_start,
    merge_config,
    parse_args,
    setup_logging,
)
from weekly_ci.stages import (
    stage_build_rccl,
    stage_checkout_aorta_report,
    stage_cleanup,
    stage_compare_all_analysis,
    stage_cross_timestamp_comparison,
    stage_convert_html_to_md,
    stage_docker_setup,
    stage_find_baseline_experiment_dir,
    stage_find_experiment_dir,
    stage_generate_summary,
    stage_install_dependencies,
    stage_pairwise_comparison,
    stage_push_results,
    stage_run_performance_tests,
    stage_single_config_analysis,
    stage_update_dashboard,
    stage_validate_environment,
)


def main() -> int:
    """Main entry point for Weekly CI Kickoff.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config()
    yaml_data = load_config_file(args.config)
    config = merge_config(config, yaml_data, args)

    # Setup logging
    logger = setup_logging(config.output.log_dir, config.output.log_level)

    logger.info("=" * 60)
    logger.info("Weekly CI Kickoff - RCCL Warp Speed Performance Analysis")
    logger.info("=" * 60)
    logger.info("")

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Config file: {args.config}")
    logger.info(f"  RCCL branch: {config.rccl.branch}")
    logger.info(f"  GPU target: {config.rccl.gpu_target}")
    logger.info(f"  Config pairs: {config.test.config_pairs}")
    logger.info(f"  Baseline: {config.test.baseline}")
    logger.info(f"  Docker compose: {config.docker.compose_file}")
    logger.info(f"  Container: {config.docker.container_name}")
    logger.info(f"  Docker build: {'skip' if config.docker.skip_build else 'enabled'}")
    logger.info("")

    # Log skip configuration
    logger.info("Stages to skip:")
    skip_any = False
    if config.skip.docker_setup:
        logger.info("  - docker_setup")
        skip_any = True
    if config.skip.rccl_build:
        logger.info("  - rccl_build")
        skip_any = True
    if config.skip.install_deps:
        logger.info("  - install_deps")
        skip_any = True
    if config.skip.performance_tests:
        logger.info("  - performance_tests")
        skip_any = True
    if config.skip.single_config_analysis:
        logger.info("  - single_config_analysis")
        skip_any = True
    if config.skip.pairwise_comparison:
        logger.info("  - pairwise_comparison")
        skip_any = True
    if config.skip.compare_all_analysis:
        logger.info("  - compare_all_analysis")
        skip_any = True
    if config.skip.checkout_aorta_report:
        logger.info("  - checkout_aorta_report")
        skip_any = True
    if config.skip.cross_timestamp_comparison:
        logger.info("  - cross_timestamp_comparison")
        skip_any = True
    if config.skip.push_results:
        logger.info("  - push_results")
        skip_any = True
    if config.skip.cleanup:
        logger.info("  - cleanup")
        skip_any = True
    if not skip_any:
        logger.info("  (none)")
    logger.info("")

    # Track overall success and repo root
    success = True
    repo_root = None

    try:
        # =====================================================================
        # Stage 1: Validate Environment
        # =====================================================================
        log_stage_start(logger, "1. Validate Environment")
        try:
            repo_root = stage_validate_environment(args.config, logger)
            log_stage_complete(logger, "Validate Environment")
        except Exception as e:
            log_stage_error(logger, "Validate Environment", str(e))
            raise

        # =====================================================================
        # Stage 2: Docker Setup
        # =====================================================================
        if config.skip.docker_setup:
            log_stage_skip(logger, "2. Docker Setup")
        else:
            log_stage_start(logger, "2. Docker Setup")
            try:
                stage_docker_setup(
                    compose_file=config.docker.compose_file,
                    container_name=config.docker.container_name,
                    repo_root=repo_root,
                    logger=logger,
                    registry_user=config.docker.registry_user,
                    registry_password=config.docker.registry_password,
                    skip_build=config.docker.skip_build,
                    force_restart=config.docker.force_restart,
                )
                log_stage_complete(logger, "Docker Setup")
            except Exception as e:
                log_stage_error(logger, "Docker Setup", str(e))
                raise

        # =====================================================================
        # Stage 3: Build RCCL
        # =====================================================================
        if config.skip.rccl_build:
            log_stage_skip(logger, "3. Build RCCL")
        else:
            log_stage_start(logger, "3. Build RCCL")
            try:
                stage_build_rccl(
                    container_name=config.docker.container_name,
                    rccl_branch=config.rccl.branch,
                    gpu_target=config.rccl.gpu_target,
                    logger=logger,
                )
                log_stage_complete(logger, "Build RCCL")
            except Exception as e:
                log_stage_error(logger, "Build RCCL", str(e))
                raise

        # =====================================================================
        # Stage 4: Install Dependencies
        # =====================================================================
        if config.skip.install_deps:
            log_stage_skip(logger, "4. Install Dependencies")
        else:
            log_stage_start(logger, "4. Install Dependencies")
            try:
                stage_install_dependencies(
                    container_name=config.docker.container_name,
                    repo_root=repo_root,
                    logger=logger,
                )
                log_stage_complete(logger, "Install Dependencies")
            except Exception as e:
                log_stage_error(logger, "Install Dependencies", str(e))
                raise

        # =====================================================================
        # Stage 5: Run Performance Tests
        # =====================================================================
        if config.skip.performance_tests:
            log_stage_skip(logger, "5. Run Performance Tests")
        else:
            log_stage_start(logger, "5. Run Performance Tests")
            try:
                stage_run_performance_tests(
                    container_name=config.docker.container_name,
                    config_pairs=config.test.config_pairs,
                    training_config=config.test.training_config,
                    logger=logger,
                )
                log_stage_complete(logger, "Run Performance Tests")
            except Exception as e:
                log_stage_error(logger, "Run Performance Tests", str(e))
                raise

        # =====================================================================
        # Stage 6: Find Experiment Directory
        # =====================================================================
        # Only need to find experiment dir if we're running analysis stages
        need_experiment_dir = (
            not config.skip.single_config_analysis
            or not config.skip.pairwise_comparison
            or not config.skip.compare_all_analysis
            or not config.skip.cross_timestamp_comparison
        )

        if need_experiment_dir:
            log_stage_start(logger, "6. Find Experiment Directory")
            try:
                config.experiment_dir = stage_find_experiment_dir(
                    repo_root=repo_root,
                    logger=logger,
                    explicit_experiment_dir=config.test.experiment_dir,
                )
                log_stage_complete(logger, "Find Experiment Directory")
            except Exception as e:
                log_stage_error(logger, "Find Experiment Directory", str(e))
                raise
        else:
            log_stage_skip(logger, "6. Find Experiment Directory (no analysis stages enabled)")

        # =====================================================================
        # Stage 7: Single Config Analysis
        # =====================================================================
        if config.skip.single_config_analysis:
            log_stage_skip(logger, "7. Single Config Analysis")
        else:
            log_stage_start(logger, "7. Single Config Analysis")
            try:
                stage_single_config_analysis(
                    container_name=config.docker.container_name,
                    experiment_dir=config.experiment_dir,
                    config_pairs=config.test.config_pairs,
                    logger=logger,
                    skip_tracelens=config.analysis.skip_tracelens_single_config,
                )
                log_stage_complete(logger, "Single Config Analysis")
            except Exception as e:
                log_stage_error(logger, "Single Config Analysis", str(e))
                raise

        # =====================================================================
        # Stage 8: Pairwise Comparison
        # =====================================================================
        if config.skip.pairwise_comparison:
            log_stage_skip(logger, "8. Pairwise Comparison")
        else:
            log_stage_start(logger, "8. Pairwise Comparison")
            try:
                stage_pairwise_comparison(
                    container_name=config.docker.container_name,
                    experiment_dir=config.experiment_dir,
                    config_pairs=config.test.config_pairs,
                    baseline=config.test.baseline,
                    logger=logger,
                    baseline_label=config.analysis.baseline_label,
                    test_label=config.analysis.test_label,
                )
                log_stage_complete(logger, "Pairwise Comparison")
            except Exception as e:
                log_stage_error(logger, "Pairwise Comparison", str(e))
                raise

        # =====================================================================
        # Stage 9: Compare All Analysis
        # =====================================================================
        if config.skip.compare_all_analysis:
            log_stage_skip(logger, "9. Compare All Analysis")
        else:
            log_stage_start(logger, "9. Compare All Analysis")
            try:
                stage_compare_all_analysis(
                    container_name=config.docker.container_name,
                    experiment_dir=config.experiment_dir,
                    config_pairs=config.test.config_pairs,
                    baseline=config.test.baseline,
                    logger=logger,
                )
                log_stage_complete(logger, "Compare All Analysis")
            except Exception as e:
                log_stage_error(logger, "Compare All Analysis", str(e))
                raise

        # =====================================================================
        # Stage 10: Checkout aorta-report
        # =====================================================================
        if config.skip.checkout_aorta_report:
            log_stage_skip(logger, "10. Checkout aorta-report")
        else:
            log_stage_start(logger, "10. Checkout aorta-report")
            try:
                config.aorta_report_dir = stage_checkout_aorta_report(
                    aorta_report_path=config.cross_timestamp.aorta_report_path,
                    repo_root=repo_root,
                    logger=logger,
                    git_token=config.git.github_token,
                )
                log_stage_complete(logger, "Checkout aorta-report")
            except Exception as e:
                log_stage_error(logger, "Checkout aorta-report", str(e))
                raise

        # =====================================================================
        # Stage 11: Cross-Timestamp Comparison
        # =====================================================================
        if config.skip.cross_timestamp_comparison:
            log_stage_skip(logger, "11. Cross-Timestamp Comparison")
        else:
            log_stage_start(logger, "11. Cross-Timestamp Comparison")
            try:
                # Find baseline experiment directory for comparison
                config.baseline_experiment_dir = stage_find_baseline_experiment_dir(
                    repo_root=repo_root,
                    baseline_experiment=config.cross_timestamp.baseline_experiment,
                    logger=logger,
                    baseline_date=config.cross_timestamp.baseline_date,
                    aorta_report_dir=config.aorta_report_dir,
                )

                if config.baseline_experiment_dir:
                    stage_cross_timestamp_comparison(
                        container_name=config.docker.container_name,
                        current_experiment_dir=config.experiment_dir,
                        baseline_experiment_dir=config.baseline_experiment_dir,
                        config_pairs=config.test.config_pairs,
                        logger=logger,
                        baseline_label=config.analysis.baseline_label,
                        test_label=config.analysis.test_label,
                    )
                    log_stage_complete(logger, "Cross-Timestamp Comparison")
                else:
                    logger.warning("  Skipping cross-timestamp comparison (no baseline found)")
                    log_stage_complete(logger, "Cross-Timestamp Comparison (partial)")
            except Exception as e:
                log_stage_error(logger, "Cross-Timestamp Comparison", str(e))
                raise

        # =====================================================================
        # Stage 12: Generate Summary & Dashboard
        # =====================================================================
        log_stage_start(logger, "12. Generate Summary & Dashboard")
        try:
            if config.experiment_dir:
                stage_generate_summary(
                    experiment_dir=config.experiment_dir,
                    repo_root=repo_root,
                    config_pairs=config.test.config_pairs,
                    baseline=config.test.baseline,
                    rccl_branch=config.rccl.branch,
                    gpu_target=config.rccl.gpu_target,
                    baseline_experiment_dir=config.baseline_experiment_dir,
                    logger=logger,
                )

                # Update aorta-report README dashboard with cross-timestamp results
                if config.aorta_report_dir:
                    stage_update_dashboard(
                        experiment_dir=config.experiment_dir,
                        repo_root=repo_root,
                        config_pairs=config.test.config_pairs,
                        aorta_report_dir=config.aorta_report_dir,
                        logger=logger,
                        report_label=config.analysis.report_label or None,
                    )
                else:
                    logger.info("  Skipping dashboard update (aorta-report not checked out)")
            else:
                logger.warning("  Skipping summary generation (no experiment directory)")
            log_stage_complete(logger, "Generate Summary & Dashboard")
        except Exception as e:
            log_stage_error(logger, "Generate Summary & Dashboard", str(e))
            # Don't raise for summary failures
            logger.warning("Summary/dashboard generation failed but continuing...")

        # =====================================================================
        # Stage 13: Convert HTML to Markdown
        # =====================================================================
        if config.skip.convert_html_to_md:
            log_stage_skip(logger, "13. Convert HTML to Markdown")
        else:
            log_stage_start(logger, "13. Convert HTML to Markdown")
            try:
                if config.experiment_dir:
                    stage_convert_html_to_md(
                        experiment_dir=config.experiment_dir,
                        repo_root=repo_root,
                        logger=logger,
                    )
                    log_stage_complete(logger, "Convert HTML to Markdown")
                else:
                    logger.warning("  Skipping (no experiment directory)")
                    log_stage_complete(logger, "Convert HTML to Markdown")
            except Exception as e:
                log_stage_error(logger, "Convert HTML to Markdown", str(e))
                logger.warning("HTML conversion failed but continuing...")

        # =====================================================================
        # Stage 14: Push Results
        # =====================================================================
        if config.skip.push_results:
            log_stage_skip(logger, "14. Push Results")
        else:
            log_stage_start(logger, "13. Push Results")
            try:
                if config.aorta_report_dir and config.experiment_dir:
                    stage_push_results(
                        aorta_report_dir=config.aorta_report_dir,
                        experiment_dir=config.experiment_dir,
                        repo_root=repo_root,
                        logger=logger,
                        report_label=config.analysis.report_label or None,
                        git_user_name=config.git.user_name,
                        git_user_email=config.git.user_email,
                    )
                else:
                    if not config.aorta_report_dir:
                        logger.warning("  Skipping push (aorta-report not checked out)")
                    if not config.experiment_dir:
                        logger.warning("  Skipping push (no experiment directory)")
                log_stage_complete(logger, "Push Results")
            except Exception as e:
                log_stage_error(logger, "Push Results", str(e))
                raise

        # =====================================================================
        # Stage 15: Cleanup
        # =====================================================================
        if config.skip.cleanup:
            log_stage_skip(logger, "15. Cleanup")
        else:
            log_stage_start(logger, "15. Cleanup")
            try:
                stage_cleanup(
                    compose_file=config.docker.compose_file,
                    container_name=config.docker.container_name,
                    repo_root=repo_root,
                    logger=logger,
                )
                log_stage_complete(logger, "Cleanup")
            except Exception as e:
                log_stage_error(logger, "Cleanup", str(e))
                # Don't raise for cleanup failures
                logger.warning("Cleanup failed but continuing...")

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        success = False
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        success = False

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("✅ Weekly CI Kickoff completed successfully!")
    else:
        logger.error("❌ Weekly CI Kickoff failed!")
    logger.info("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
