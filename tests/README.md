# AORTA Tests

Test suite for the AORTA toolkit.

## Setup

Install pytest and dependencies:

```bash
pip install pytest pytest-cov
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_merge_gpu_trace_ranks.py
```

### Run specific test
```bash
pytest tests/test_merge_gpu_trace_ranks.py::TestMergeGpuTraces::test_basic_merge
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage report
```bash
pytest --cov=src --cov=scripts --cov-report=html
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_merge_gpu_trace_ranks.py  # Tests for trace merger
└── README.md                      # This file
```

## Writing New Tests

Example test structure:

```python
import pytest

class TestYourFeature:
    """Test description."""

    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        return {"key": "value"}

    def test_something(self, mock_data):
        """Test a specific behavior."""
        assert mock_data["key"] == "value"
```
