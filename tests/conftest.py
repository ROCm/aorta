"""
Shared pytest fixtures and configuration for all tests.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers for all tests."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_addoption(parser):
    """Add custom command-line options."""
    # GEMM regression test options
    parser.addoption(
        "--generate-baseline",
        action="store_true",
        default=False,
        help="Generate baseline expected outputs for GEMM regression tests"
    )


@pytest.fixture
def sample_trace_event():
    """Create a sample trace event for testing."""
    return {
        "pid": 100,
        "tid": 1,
        "ts": 1000000,
        "dur": 50000,
        "ph": "X",
        "name": "test_event",
        "cat": "kernel",
        "args": {"some_arg": "value"}
    }
