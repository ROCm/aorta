"""
Shared pytest fixtures and configuration for all tests.
"""

import pytest

# Pre-import Triton once, up front, against the *real* importlib.metadata
# entry-points before any test can replace them with a mock.
#
# Triton discovers its compiler backends lazily on first import via
# ``importlib.metadata.entry_points`` (``triton.backends._discover_backends``).
# Many dispatcher/discovery tests patch ``importlib.metadata.entry_points`` with
# a ``MagicMock`` to fake ``aorta.workloads`` discovery. If Triton's very first
# import happens while that patch is active (e.g. ``run_trials`` -> ``collect_env``
# probes the ``triton`` version), backend discovery raises ``ModuleNotFoundError``
# on the mock, the ``import triton`` fails, and Python drops ``triton`` /
# ``triton.backends`` from ``sys.modules`` while leaving the already-loaded
# ``triton.backends.compiler`` cached. That half-initialized state then makes
# ``torch.use_deterministic_algorithms`` (which does ``import
# triton.backends.compiler``) raise ``AttributeError`` in every later workload
# test -- order-dependent cross-file pollution (issue #270).
#
# Importing Triton here makes it a cache hit everywhere else, so a mocked
# ``entry_points`` never triggers a fresh (failing) Triton import. Best-effort:
# Triton is an optional dependency and absent on CPU-only stacks, where no
# ``triton.backends.compiler`` exists to orphan, so there is nothing to guard.
# Catch ``Exception`` (not just ``ImportError``): a broken Triton/native install
# can fail with OSError/RuntimeError. On any failure, purge partially-imported
# ``triton.*`` modules so the failure leaves a clean slate instead of seeding
# the very half-state (an orphaned ``triton.backends.compiler`` with ``triton``
# gone) that this pre-import exists to prevent.
try:
    import triton  # noqa: F401
except Exception:
    import sys

    for _mod in [m for m in sys.modules if m == "triton" or m.startswith("triton.")]:
        del sys.modules[_mod]


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
