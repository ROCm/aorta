# GEMM Analysis Regression Test Suite

Automated regression testing for the `scripts/gemm_analysis` pipeline to ensure script changes don't introduce errors or produce incorrect reports.

## Overview

This test suite provides comprehensive regression testing by:
1. Using fixed synthetic test data (2 thread configs × 2 channel configs × 7 ranks)
2. Running the complete analysis pipeline on this data
3. Comparing outputs against baseline expected results
4. Allowing tolerance for numeric values and cosmetic changes

## Directory Structure

```
tests/gemm_analysis/
├── testdata/                  # Synthetic test data (not in git, generate locally)
│   └── test_sweep/           # Fixed test data directory
│       ├── 256thread/
│       │   ├── 28channels/
│       │   └── 56channels/
│       └── 512thread/
│           ├── 28channels/
│           └── 56channels/
├── expected_outputs/          # Baseline outputs (not in git, generate locally)
├── actual_outputs/           # Outputs from current branch (temp, not in git)
├── test_gemm_regression.py   # Modular pytest test suite (includes baseline generation)
├── generate_synthetic_data.py # Synthetic data generator
├── compare_outputs.py        # Output comparison logic
└── README.md                # This file
```

## Test Architecture

The test suite is organized into modular components for easy extension:

### PipelineRunner Class
Central class that encapsulates pipeline execution logic with methods for each step:
- `run_tracelens()` - Run TraceLens analysis
- `run_analyze_gemm()` - Analyze GEMM reports
- `run_plot_variance()` - Generate variance plots
- `run_enhancement_scripts()` - Run additional analysis scripts
- `run_full_pipeline()` - Execute complete pipeline

### Test Classes

1. **TestFullPipeline** - Full end-to-end regression test (original test)
2. **TestIndividualSteps** - Test individual pipeline components in isolation
3. **TestCustomConfigurations** - Test with different thread/channel configurations
4. **TestErrorHandling** - Test error conditions and edge cases

### Adding New Tests

To add new tests, create methods in the appropriate test class or add a new test class:

```python
class TestNewFeature:
    def test_my_feature(self, scripts_dir):
        runner = PipelineRunner(scripts_dir)
        # Use runner methods to test specific functionality
```

## Quick Start

### First Time Setup

**Important**: Activate virtual environment and generate test data before running tests (not tracked in git):

```bash
# Activate virtual environment
source ~/venvs/aorta/bin/activate

# Step 1: Generate synthetic test data (only needed once)
python tests/gemm_analysis/generate_synthetic_data.py --output-dir tests/gemm_analysis/testdata

# Step 2: Generate baseline expected outputs (on origin/main)
# NOTE: This automatically runs TraceLens analysis as part of the pipeline
pytest tests/gemm_analysis/test_gemm_regression.py --generate-baseline
```

**Important Note on TraceLens**:
- The end-to-end regression test (`TestFullPipeline`) automatically runs TraceLens analysis
- If TraceLens is not installed, you must manually run it once:
  ```bash
  bash scripts/gemm_analysis/run_tracelens_analysis.sh tests/gemm_analysis/testdata/test_sweep
  ```
- Individual component tests (`TestIndividualSteps`, `TestCustomConfigurations`) require TraceLens outputs to already exist

### Run Full Regression Test Suite

From the repository root:

```bash
# Ensure virtual environment is activated
source ~/venvs/aorta/bin/activate

# Run all regression tests
pytest tests/gemm_analysis/ -v
```

### Run Specific Tests

```bash
# Run full pipeline regression test
pytest tests/gemm_analysis/test_gemm_regression.py::TestFullPipeline -v

# Run individual component tests
pytest tests/gemm_analysis/test_gemm_regression.py::TestIndividualSteps -v

# Run custom configuration tests
pytest tests/gemm_analysis/test_gemm_regression.py::TestCustomConfigurations -v

# Run error handling tests
pytest tests/gemm_analysis/test_gemm_regression.py::TestErrorHandling -v

# Run only fast tests (skip integration)
pytest tests/gemm_analysis/ -m "not integration" -v

# Run only integration tests
pytest tests/gemm_analysis/ -m integration -v
```

## Usage Options

### Test Data Management

```bash
# Generate synthetic test data
python tests/gemm_analysis/generate_synthetic_data.py --output-dir tests/gemm_analysis/testdata

# Regenerate baseline expected outputs
pytest tests/gemm_analysis/test_gemm_regression.py --generate-baseline

# Clean all test artifacts
rm -rf tests/gemm_analysis/testdata tests/gemm_analysis/expected_outputs tests/gemm_analysis/actual_outputs
```

### Pytest Options

```bash
# Run all tests except integration tests
pytest . -m "not integration"

# Run only integration tests
pytest . -m integration

# Run with verbose output
pytest . -v

# Run with coverage
pytest . --cov=scripts/gemm_analysis
```

## Test Coverage

### Scripts Under Test

1. **run_tracelens_analysis.sh** / **.py** - Main analysis orchestrator
2. **analyze_gemm_reports.py** - Extract GEMM kernels from reports
3. **plot_gemm_variance.py** - Generate variance plots
4. **enhance_gemm_variance_with_timestamps.py** - Add timestamp data
5. **gemm_report_with_collective_overlap.py** - Analyze NCCL overlap
6. **process_gpu_timeline.py** - Process GPU timeline data
7. **create_embeded_html_report.py** - Generate HTML comparison reports

### Test Types

1. **Data Integrity Tests** - Verify synthetic test data structure
2. **Script Execution Tests** - Ensure scripts run without errors
3. **Output Comparison Tests** - Validate outputs against baseline
4. **Integration Tests** - Full pipeline end-to-end testing

## Output Comparison Rules

The comparison logic (`compare_outputs.py`) enforces:

### Strict Requirements
- Core numeric metrics must match within tolerance (1e-6)
- Required columns/fields must be present
- Per-GEMM aggregates must be consistent
- Critical data values must match

### Allowed Variations
- Extra columns in outputs (additive changes)
- Column reordering
- HTML styling and layout changes
- Timestamps and build IDs
- File paths (as long as structure is maintained)
- Additional rows/entries (as long as core data is present)

## Adding New Tests

### To add a new analysis script to the test suite:

1. Update `run_regression_tests.sh` to call the new script in `run_analysis_pipeline()`
2. Add test cases in `tests/test_gemm_regression.py`
3. Regenerate baseline outputs:
   ```bash
   tests/gemm_analysis/run_regression_tests.sh --generate-baseline
   ```

### To modify test data:

1. Edit `generate_synthetic_data.py`
2. Regenerate test data and baseline:
   ```bash
   tests/gemm_analysis/run_regression_tests.sh --clean
   tests/gemm_analysis/run_regression_tests.sh --generate-baseline
   ```

## Environment

The test suite requires the aorta virtual environment:

```bash
source ~/venvs/aorta/bin/activate
```

### Dependencies

- Python 3.8+
- pandas
- numpy
- openpyxl
- pytest
- All dependencies from `requirements.txt`

## Continuous Integration

To integrate with CI/CD:

```bash
# In CI pipeline
source ~/venvs/aorta/bin/activate
tests/gemm_analysis/run_regression_tests.sh --generate-data  # Generate test data
tests/gemm_analysis/run_regression_tests.sh --generate-baseline  # If baseline needed
tests/gemm_analysis/run_regression_tests.sh  # Run regression tests
```

Exit codes:
- 0: All tests passed
- 1: Test failures detected

## Troubleshooting

### Common Issues

1. **Test fails with "testdata not found"**
   ```bash
   tests/gemm_analysis/run_regression_tests.sh --generate-data
   ```
   Test data is not tracked in git and must be generated locally first.

2. **Missing virtual environment**
   ```bash
   python3 -m venv ~/venvs/aorta
   source ~/venvs/aorta/bin/activate
   pip install -r requirements.txt
   ```

3. **Baseline outdated**
   ```bash
   tests/gemm_analysis/run_regression_tests.sh --generate-baseline
   ```

4. **Numeric differences**
   - Check `compare_outputs.py` tolerance settings
   - Review actual vs expected values in logs

5. **Missing dependencies**
   ```bash
   pip install pytest pandas numpy openpyxl matplotlib
   ```

## Implementation Status

- ✅ Test directory structure created
- ✅ Synthetic data generator implemented
- ✅ Main test harness script created
- ✅ Output comparison logic with tolerance
- ✅ Pytest test suite
- ✅ Documentation

## Contact

For questions or issues with the test suite, please refer to the test plan in `scripts/gemm_analysis/test_plan.md`.
