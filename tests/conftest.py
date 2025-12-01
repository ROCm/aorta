"""
Shared pytest fixtures and configuration for all tests.
"""

import pytest


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
