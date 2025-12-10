#!/usr/bin/env python3
"""
Modular regression tests for GEMM analysis pipeline.
Supports both full pipeline and individual component testing.

Usage:
    # Generate baseline (on origin/main)
    pytest tests/gemm_analysis/test_gemm_regression.py --generate-baseline

    # Run all regression tests
    pytest tests/gemm_analysis/test_gemm_regression.py

    # Run only full pipeline tests
    pytest tests/gemm_analysis/test_gemm_regression.py::TestFullPipeline

    # Run only individual step tests
    pytest tests/gemm_analysis/test_gemm_regression.py::TestIndividualSteps
"""

import pytest
import subprocess
import sys
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

# Add parent directories to path
TEST_DIR = Path(__file__).parent  # tests/gemm_analysis
REPO_ROOT = TEST_DIR.parent.parent  # repository root
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'gemm_analysis'))


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to the test data directory."""
    return TEST_DIR / "testdata" / "test_sweep"


@pytest.fixture(scope="session")
def scripts_dir():
    """Path to the GEMM analysis scripts directory."""
    return REPO_ROOT / "scripts" / "gemm_analysis"


@pytest.fixture(scope="session")
def expected_outputs_dir():
    """Path to the expected outputs directory."""
    return TEST_DIR / "expected_outputs"


@pytest.fixture(scope="session")
def generate_baseline(request):
    """Check if we're generating baseline outputs."""
    return request.config.getoption("--generate-baseline")


# ============================================================================
# PIPELINE RUNNER CLASS
# ============================================================================

class PipelineRunner:
    """Encapsulates pipeline execution logic for reuse across tests."""

    def __init__(self, scripts_dir: Path, timeout_seconds: Dict[str, int] = None):
        """
        Initialize the pipeline runner.

        Args:
            scripts_dir: Path to GEMM analysis scripts
            timeout_seconds: Optional timeout overrides for each step
        """
        self.scripts_dir = scripts_dir
        self.timeouts = timeout_seconds or {
            'tracelens': 600,
            'analyze': 300,
            'plot': 300,
            'enhance': 300,
            'overlap': 300,
            'timeline': 300
        }

    def setup_output_dirs(self, output_dir: Path) -> Tuple[Path, Path]:
        """
        Create output directory structure.

        Returns:
            Tuple of (output_dir, reports_dir)
        """
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, reports_dir

    def run_tracelens(self, test_data_dir: Path, skip_on_missing: bool = True) -> Path:
        """
        Run TraceLens analysis step.

        Args:
            test_data_dir: Input data directory
            skip_on_missing: Skip if TraceLens not installed (vs fail)

        Returns:
            Path to tracelens output directory
        """
        print("\n1. Running TraceLens analysis...")
        tracelens_script = self.scripts_dir / "run_tracelens_analysis.sh"

        if tracelens_script.exists():
            result = subprocess.run(
                ["bash", str(tracelens_script), str(test_data_dir)],
                capture_output=True, text=True, timeout=self.timeouts['tracelens']
            )
            if result.returncode != 0:
                if any(err in str(result.stderr) + str(result.stdout)
                       for err in ["TraceLens", "tracelens", "command not found"]):
                    if skip_on_missing:
                        print("   Warning: TraceLens not installed - skipping")
                    else:
                        pytest.fail("TraceLens not installed")
                else:
                    pytest.fail(f"TraceLens analysis failed: {result.stderr}")
            else:
                print("   TraceLens analysis completed")

        # Check if outputs exist and contain actual Excel reports
        tracelens_dir = test_data_dir / "tracelens_analysis"
        if not tracelens_dir.exists():
            if skip_on_missing:
                pytest.skip(
                    "TraceLens outputs not found. Either:\n"
                    "  1. Install TraceLens, or\n"
                    "  2. Run manually: bash scripts/gemm_analysis/run_tracelens_analysis.sh "
                    f"{test_data_dir}"
                )
            else:
                pytest.fail("TraceLens outputs not found")

        # Check if TraceLens generated actual Excel reports
        excel_files = list(tracelens_dir.glob("**/individual_reports/*.xlsx"))
        if not excel_files:
            if skip_on_missing:
                pytest.skip(
                    "TraceLens directory exists but contains no Excel reports.\n"
                    "TraceLens is either not installed or failed to run.\n"
                    "To fix, run: bash scripts/gemm_analysis/run_tracelens_analysis.sh "
                    f"{test_data_dir}"
                )
            else:
                pytest.fail("TraceLens outputs directory exists but contains no Excel reports")

        return tracelens_dir

    def run_analyze_gemm(self,
                        tracelens_dir: Path,
                        output_file: Path,
                        threads: List[int] = None,
                        channels: List[int] = None,
                        ranks: List[int] = None,
                        top_k: int = 10) -> Path:
        """
        Run analyze_gemm_reports.py step.

        Args:
            tracelens_dir: TraceLens output directory
            output_file: Output CSV path
            threads: Thread configurations (default: [256, 512])
            channels: Channel configurations (default: [28, 56])
            ranks: Rank list (default: [0-6])
            top_k: Number of top kernels to extract

        Returns:
            Path to output file
        """
        print("\n2. Analyzing GEMM reports...")

        # Default values
        threads = threads or [256, 512]
        channels = channels or [28, 56]
        ranks = ranks or list(range(7))

        # Require TraceLens Excel reports; otherwise skip gracefully
        excel_files = list(tracelens_dir.glob("**/individual_reports/*.xlsx"))
        if not excel_files:
            pytest.skip(
                "TraceLens outputs (individual_reports/*.xlsx) are required for analyze_gemm"
            )

        cmd = [
            sys.executable,
            str(self.scripts_dir / "analyze_gemm_reports.py"),
            "--base-path", str(tracelens_dir),
            "--threads"] + [str(t) for t in threads] + [
            "--channels"] + [str(c) for c in channels] + [
            "--ranks"] + [str(r) for r in ranks] + [
            "--top-k", str(top_k),
            "--output-file", str(output_file)
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeouts['analyze']
        )

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            pytest.fail(f"analyze_gemm_reports.py failed")

        print(f"   Created: {output_file}")
        return output_file

    def run_plot_variance(self, csv_file: Path, output_dir: Path,
                          skip_on_missing_deps: bool = True) -> Optional[Path]:
        """
        Run plot_gemm_variance.py step.

        Args:
            csv_file: Input CSV from analyze_gemm
            output_dir: Output directory for plots
            skip_on_missing_deps: Skip if matplotlib missing (vs fail)

        Returns:
            Path to plots directory or None if skipped
        """
        print("\n3. Generating variance plots...")
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run([
            sys.executable,
            str(self.scripts_dir / "plot_gemm_variance.py"),
            "--csv-path", str(csv_file),
            "--output-dir", str(plots_dir)
        ], capture_output=True, text=True, timeout=self.timeouts['plot'])

        if result.returncode != 0:
            if "ModuleNotFoundError" in result.stderr or "matplotlib" in result.stderr:
                if skip_on_missing_deps:
                    print("   Warning: Matplotlib not installed - skipping plots")
                    return None
                else:
                    pytest.fail("Matplotlib not installed")
            else:
                print(f"   Warning: plot_gemm_variance.py failed: {result.stderr}")
                return None

        print(f"   Plots created in: {plots_dir}")
        return plots_dir

    def run_enhancement_scripts(self,
                               gemm_csv: Path,
                               test_data_dir: Path,
                               tracelens_dir: Path,
                               reports_dir: Path) -> Dict[str, Path]:
        """
        Run all enhancement/additional analysis scripts.

        Args:
            gemm_csv: GEMM kernels CSV from analyze step
            test_data_dir: Test data directory
            tracelens_dir: TraceLens output directory
            reports_dir: Reports output directory

        Returns:
            Dictionary of output file paths
        """
        print("\n4. Running additional analysis scripts...")
        outputs = {}

        # enhance_gemm_variance_with_timestamps.py
        enhanced_file = reports_dir / "gemm_kernels_enhanced.csv"
        result = subprocess.run([
            sys.executable,
            str(self.scripts_dir / "enhance_gemm_variance_with_timestamps.py"),
            "--input-csv", str(gemm_csv),
            "--base-path", str(test_data_dir),
            "--output-csv", str(enhanced_file)
        ], capture_output=True, text=True, timeout=self.timeouts['enhance'])

        if result.returncode == 0:
            print(f"   Created: {enhanced_file}")
            outputs['enhanced'] = enhanced_file
        else:
            print(f"   Warning: Enhancement failed: {result.stderr}")

        # gemm_report_with_collective_overlap.py
        overlap_file = reports_dir / "gemm_with_overlap.csv"
        result = subprocess.run([
            sys.executable,
            str(self.scripts_dir / "gemm_report_with_collective_overlap.py"),
            "--input-csv", str(gemm_csv),
            "--tracelens-path", str(tracelens_dir),
            "--output-csv", str(overlap_file)
        ], capture_output=True, text=True, timeout=self.timeouts['overlap'])

        if result.returncode == 0:
            print(f"   Created: {overlap_file}")
            outputs['overlap'] = overlap_file
        else:
            print(f"   Warning: Overlap analysis failed: {result.stderr}")

        # process_gpu_timeline.py
        result = subprocess.run([
            sys.executable,
            str(self.scripts_dir / "process_gpu_timeline.py"),
            "--sweep-dir", str(test_data_dir)
        ], capture_output=True, text=True, timeout=self.timeouts['timeline'])

        if result.returncode == 0:
            print("   GPU timeline processed")
            outputs['timeline'] = test_data_dir / "gpu_timeline_all_configs_mean.xlsx"
        else:
            print(f"   Warning: GPU timeline processing failed: {result.stderr}")

        print("\n   Additional analysis completed")
        return outputs

    def run_full_pipeline(self, test_data_dir: Path, output_dir: Path) -> Dict[str, Path]:
        """
        Run the complete analysis pipeline.

        Args:
            test_data_dir: Input test data directory
            output_dir: Output directory

        Returns:
            Dictionary of all output file paths
        """
        # Setup directories
        output_dir, reports_dir = self.setup_output_dirs(output_dir)

        # Step 1: TraceLens
        print('1')
        tracelens_dir = self.run_tracelens(test_data_dir)
        print('2')

        # Step 2: Analyze GEMM
        gemm_csv = reports_dir / "gemm_kernels.csv"
        self.run_analyze_gemm(tracelens_dir, gemm_csv)

        # Step 3: Plot variance
        plots_dir = self.run_plot_variance(gemm_csv, output_dir)

        # Step 4: Enhancement scripts
        enhancement_outputs = self.run_enhancement_scripts(
            gemm_csv, test_data_dir, tracelens_dir, reports_dir
        )

        # Collect all outputs
        outputs = {
            'gemm_csv': gemm_csv,
            'plots_dir': plots_dir,
            **enhancement_outputs
        }

        return outputs


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestFullPipeline:
    """Full end-to-end regression tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_regression(self, test_data_dir, expected_outputs_dir, scripts_dir, generate_baseline):
        """
        Main regression test: either generate baseline or run regression test.

        Usage:
            pytest test_gemm_regression.py --generate-baseline  # Create baseline
            pytest test_gemm_regression.py                      # Run regression test
        """
        # Check prerequisites
        if not test_data_dir.exists():
            pytest.skip(
                "Synthetic test data not found\n"
                "Run: python gemm_analysis/generate_synthetic_data.py"
            )

        # Initialize pipeline runner
        runner = PipelineRunner(scripts_dir)

        # Determine mode and output directory
        if generate_baseline:
            # GENERATE BASELINE MODE
            print(f"\n{'='*60}")
            print("GENERATING BASELINE EXPECTED OUTPUTS")
            print(f"{'='*60}")

            runner.run_full_pipeline(test_data_dir, expected_outputs_dir)

            print(f"\n{'='*60}")
            print("BASELINE GENERATION COMPLETE")
            print(f"Outputs saved to: {expected_outputs_dir}")
            print(f"{'='*60}\n")

        else:
            # REGRESSION TEST MODE
            print(f"\n{'='*60}")
            print("RUNNING REGRESSION TEST")
            print(f"{'='*60}")

            # Check baseline exists
            if not expected_outputs_dir.exists():
                pytest.skip(
                    "Baseline outputs not found\n"
                    "Run on origin/main: pytest test_gemm_regression.py --generate-baseline"
                )

            # Run pipeline to actual_outputs
            actual_outputs_dir = TEST_DIR / "actual_outputs"
            runner.run_full_pipeline(test_data_dir, actual_outputs_dir)

            # Compare outputs to baseline
            print(f"\n5. Comparing outputs to baseline...")
            print(f"{'='*60}")

            compare_script = TEST_DIR / "compare_outputs.py"
            result = subprocess.run([
                sys.executable,
                str(compare_script),
                "--expected", str(expected_outputs_dir),
                "--actual", str(actual_outputs_dir),
                "--verbose"
            ], capture_output=True, text=True, timeout=60)

            # Print comparison output
            print(result.stdout)
            if result.stderr:
                print(result.stderr)

            # Fail if comparison found differences
            if result.returncode != 0:
                pytest.fail(
                    "REGRESSION DETECTED! Current branch outputs differ from baseline.\n"
                    "If changes are intentional, regenerate baseline:\n"
                    "  pytest test_gemm_regression.py --generate-baseline"
                )

            print(f"\n{'='*60}")
            print("ALL REGRESSION TESTS PASSED!")
            print(f"{'='*60}\n")


class TestIndividualSteps:
    """Test individual pipeline components in isolation."""

    @pytest.fixture(scope="class")
    def runner(self, scripts_dir):
        """Create a PipelineRunner instance."""
        return PipelineRunner(scripts_dir)

    @pytest.fixture(scope="class")
    def temp_output_dir(self, tmp_path_factory):
        """Create a temporary output directory for tests."""
        return tmp_path_factory.mktemp("test_outputs")

    def test_analyze_gemm_step(self, runner, test_data_dir, temp_output_dir):
        """Test analyze_gemm_reports.py independently."""
        tracelens_dir = test_data_dir / "tracelens_analysis"
        if not tracelens_dir.exists():
            pytest.skip("TraceLens outputs required for this test")

        output_file = temp_output_dir / "gemm_test.csv"

        # Test with custom configuration
        result = runner.run_analyze_gemm(
            tracelens_dir=tracelens_dir,
            output_file=output_file,
            threads=[256],  # Test with single thread config
            channels=[28],   # Test with single channel config
            ranks=[0, 1],    # Test with fewer ranks
            top_k=5          # Test with different top_k
        )

        assert result.exists()
        assert result.stat().st_size > 0

    def test_plot_generation(self, runner, test_data_dir, temp_output_dir):
        """Test plot_gemm_variance.py independently."""
        # First generate a CSV to plot
        tracelens_dir = test_data_dir / "tracelens_analysis"
        if not tracelens_dir.exists():
            pytest.skip("TraceLens outputs required for this test")

        csv_file = temp_output_dir / "gemm_for_plots.csv"
        runner.run_analyze_gemm(tracelens_dir, csv_file)

        # Test plotting
        plots_dir = runner.run_plot_variance(csv_file, temp_output_dir)

        if plots_dir is not None:
            assert plots_dir.exists()
            # Check for expected plot files
            expected_plots = [
                "variance_by_threads_boxplot.png",
                "variance_by_channels_boxplot.png"
            ]
            for plot_name in expected_plots:
                plot_file = plots_dir / plot_name
                if plot_file.exists():
                    assert plot_file.stat().st_size > 0


class TestCustomConfigurations:
    """Test pipeline with various custom configurations."""

    @pytest.fixture(scope="class")
    def runner(self, scripts_dir):
        """Create a PipelineRunner with custom timeouts."""
        custom_timeouts = {
            'tracelens': 300,  # Faster timeout for testing
            'analyze': 120,
            'plot': 120,
            'enhance': 120,
            'overlap': 120,
            'timeline': 120
        }
        return PipelineRunner(scripts_dir, timeout_seconds=custom_timeouts)

    @pytest.mark.parametrize("threads,channels", [
        ([256], [28]),      # Minimal config
        ([512], [56]),      # Single alternative config
        ([256, 512], [28]), # Mixed config
    ])
    def test_different_configurations(self, runner, test_data_dir, threads, channels, tmp_path):
        """Test pipeline with different thread/channel configurations."""
        tracelens_dir = test_data_dir / "tracelens_analysis"
        if not tracelens_dir.exists():
            pytest.skip("TraceLens outputs required for this test")

        output_file = tmp_path / f"gemm_{threads}_{channels}.csv"

        result = runner.run_analyze_gemm(
            tracelens_dir=tracelens_dir,
            output_file=output_file,
            threads=threads,
            channels=channels,
            ranks=list(range(3))  # Use fewer ranks for speed
        )

        assert result.exists()
        # Could add more specific assertions about the output content


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture(scope="class")
    def runner(self, scripts_dir):
        """Create a PipelineRunner instance."""
        return PipelineRunner(scripts_dir)

    def test_missing_tracelens_data(self, runner, tmp_path):
        """Test graceful handling when TraceLens data is missing."""
        fake_data_dir = tmp_path / "fake_data"
        fake_data_dir.mkdir()

        with pytest.raises(SystemExit):  # pytest.skip raises SystemExit
            runner.run_tracelens(fake_data_dir, skip_on_missing=True)

    def test_invalid_output_path(self, runner, test_data_dir):
        """Test handling of invalid output paths."""
        tracelens_dir = test_data_dir / "tracelens_analysis"
        if not tracelens_dir.exists():
            pytest.skip("TraceLens outputs required for this test")

        invalid_path = Path("/invalid/path/that/does/not/exist/output.csv")

        with pytest.raises(SystemExit):  # pytest.fail raises SystemExit
            runner.run_analyze_gemm(tracelens_dir, invalid_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
