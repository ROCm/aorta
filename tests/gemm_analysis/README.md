# GEMM Analysis Regression Test Suite

Automated regression testing for `scripts/gemm_analysis` pipeline.

## Directory Structure

```
tests/gemm_analysis/
├── testdata/                  # Synthetic test data (not in git)
│   └── test_sweep/           # Fixed test data directory
├── expected_outputs/          # Baseline outputs (not in git)
├── actual_outputs/           # Current branch outputs (temp)
├── test_gemm_regression.py   # Pytest test suite
├── generate_synthetic_data.py # Synthetic data generator
├── compare_outputs.py        # Output comparison logic
└── README.md                # This file
```

## Test Architecture

### PipelineRunner Class
Encapsulates pipeline execution with methods:
- `run_tracelens()` - Run TraceLens analysis
- `run_analyze_gemm()` - Analyze GEMM reports
- `run_plot_variance()` - Generate variance plots
- `run_enhancement_scripts()` - Run additional analysis scripts
- `run_full_pipeline()` - Execute complete pipeline

### Test Classes

1. **TestFullPipeline** - Full end-to-end regression test
2. **TestIndividualSteps** - Test individual pipeline components
3. **TestCustomConfigurations** - Test with different thread/channel configurations
4. **TestErrorHandling** - Test error conditions and edge cases

### Test Coverage

Scripts tested:
- run_tracelens_analysis.sh
- analyze_gemm_reports.py
- plot_gemm_variance.py
- enhance_gemm_variance_with_timestamps.py
- gemm_report_with_collective_overlap.py
- process_gpu_timeline.py

Validation:
- TraceLens analysis execution (skips if not installed)
- GEMM report analysis across configurations
- Variance plot generation
- Enhancement script outputs
- GPU timeline processing
- Numeric output consistency (tolerance: 1e-6)
- Script integration and data flow

### Test Data

- Synthetic PyTorch profiler traces (not real traces)
- Configurations: 2 threads (256, 512) × 2 channels (28, 56) × 8 ranks
- 2 batches (ProfilerSteps) per trace
- Based on MI350X trace structure
- Generated via `generate_synthetic_data.py`

### Baseline Comparison

- Baseline in `expected_outputs/` (generated from origin/main)
- Current branch outputs in `actual_outputs/`
- Numeric values: tolerance 1e-6
- Ignores: whitespace, column ordering, timestamps

## Setup

```bash
source ~/venvs/aorta/bin/activate

# Generate test data (once)
python tests/gemm_analysis/generate_synthetic_data.py --output-dir tests/gemm_analysis/testdata

# Generate baseline (on origin/main)
pytest tests/gemm_analysis/test_gemm_regression.py --generate-baseline

# If TraceLens not installed, run manually:
bash scripts/gemm_analysis/run_tracelens_analysis.sh tests/gemm_analysis/testdata/test_sweep
```

## Running Tests

```bash
# All tests
pytest tests/gemm_analysis/ -v

# Specific test class
pytest tests/gemm_analysis/test_gemm_regression.py::TestFullPipeline -v
pytest tests/gemm_analysis/test_gemm_regression.py::TestIndividualSteps -v
pytest tests/gemm_analysis/test_gemm_regression.py::TestCustomConfigurations -v
pytest tests/gemm_analysis/test_gemm_regression.py::TestErrorHandling -v

# Integration tests only
pytest tests/gemm_analysis/ -m integration -v

# Skip integration tests
pytest tests/gemm_analysis/ -m "not integration" -v
```

## Adding New Tests

### Add Test Method to Existing Class

```python
class TestIndividualSteps:
    def test_new_feature(self, runner, test_data_dir, temp_output_dir):
        """Test description."""
        # Setup
        input_file = test_data_dir / "input.csv"
        output_file = temp_output_dir / "output.csv"

        # Execute via runner
        result = runner.run_new_feature(input_file, output_file)

        # Assert
        assert result.exists()
        assert result.stat().st_size > 0
```

### Add New Test Class

```python
class TestNewScript:
    @pytest.fixture(scope="class")
    def runner(self, scripts_dir):
        return PipelineRunner(scripts_dir)

    def test_new_script_execution(self, runner, test_data_dir):
        """Test new script execution."""
        # Add runner method first in PipelineRunner class
        result = runner.run_new_script(test_data_dir)
        assert result.returncode == 0
```

### Add Runner Method

In `PipelineRunner` class:

```python
def run_new_script(self, input_path: Path, output_path: Path) -> Path:
    """
    Run new_script.py step.

    Args:
        input_path: Input file/directory
        output_path: Output file/directory

    Returns:
        Path to output
    """
    print("\nN. Running new script...")

    cmd = [
        sys.executable,
        str(self.scripts_dir / "new_script.py"),
        "--input", str(input_path),
        "--output", str(output_path)
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=self.timeouts.get('new_script', 300)
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        pytest.fail("new_script.py failed")

    print(f"   Created: {output_path}")
    return output_path
```

### Add to Full Pipeline

In `run_full_pipeline()` method:

```python
def run_full_pipeline(self, test_data_dir: Path, output_dir: Path) -> Dict[str, Path]:
    # ... existing steps ...

    # Step N: New script
    new_output = self.run_new_script(input_file, output_file)

    outputs['new_output'] = new_output
    return outputs
```

### Update Baseline Comparison

If script produces output that should be compared:

1. Regenerate baseline:
```bash
pytest tests/gemm_analysis/test_gemm_regression.py --generate-baseline
```

2. Update `compare_outputs.py` if new comparison logic needed

## Output Comparison Rules

### Strict Requirements
- Core numeric metrics match within tolerance (1e-6)
- Required columns/fields present
- Per-GEMM aggregates consistent

### Allowed Variations
- Extra columns (additive changes)
- Column reordering
- HTML styling changes
- Timestamps and build IDs
- File paths (structure maintained)
- Additional rows (core data present)

## Modifying Test Data

1. Edit `generate_synthetic_data.py`
2. Regenerate:
```bash
python tests/gemm_analysis/generate_synthetic_data.py --output-dir tests/gemm_analysis/testdata
bash scripts/gemm_analysis/run_tracelens_analysis.sh tests/gemm_analysis/testdata/test_sweep
pytest tests/gemm_analysis/test_gemm_regression.py --generate-baseline
```

## Troubleshooting

**testdata not found**
```bash
python tests/gemm_analysis/generate_synthetic_data.py --output-dir tests/gemm_analysis/testdata
```

**Baseline outdated**
```bash
pytest tests/gemm_analysis/test_gemm_regression.py --generate-baseline
```

**TraceLens outputs missing**
```bash
bash scripts/gemm_analysis/run_tracelens_analysis.sh tests/gemm_analysis/testdata/test_sweep
```

**Dependencies missing**
```bash
pip install pytest pandas numpy openpyxl matplotlib
```

**Virtual environment not activated**
```bash
source ~/venvs/aorta/bin/activate
```

**Tests fail after generating synthetic data**
```bash
# Run TraceLens analysis on synthetic data
bash scripts/gemm_analysis/run_tracelens_analysis.sh tests/gemm_analysis/testdata/test_sweep
```

## CI/CD Integration

```bash
source ~/venvs/aorta/bin/activate
python tests/gemm_analysis/generate_synthetic_data.py --output-dir tests/gemm_analysis/testdata
bash scripts/gemm_analysis/run_tracelens_analysis.sh tests/gemm_analysis/testdata/test_sweep
pytest tests/gemm_analysis/test_gemm_regression.py --generate-baseline  # If baseline needed
pytest tests/gemm_analysis/ -v
```

Exit codes:
- 0: All tests passed
- 1: Test failures
