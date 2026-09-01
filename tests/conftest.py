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


class BlockedTooLong(BaseException):
    """A call guarded by :func:`no_hang` did not return in time.

    Deliberately a :class:`BaseException` rather than an :class:`Exception`.
    The code under test is exactly the kind that catches broadly -- collector
    summarization swallows ``Exception`` so an opt-in measurement can never fail
    a healthy trial, and the artifact reader skips on ``OSError`` -- and
    ``TimeoutError`` is an ``OSError``, so a timeout raised as one gets absorbed
    and the test passes while the code hangs. Sitting outside ``Exception`` is
    what makes the alarm reach the test runner.
    """


@pytest.fixture
def no_hang():
    """Bound a call that could block forever, failing instead of stalling.

    Yields a context-manager factory: ``with no_hang(5): ...``. Use it where the
    defect under test is an indefinite block rather than a wrong value, so a
    regression surfaces as a failure. ``SIGALRM`` is the mechanism because the
    block happens inside a single syscall, which a thread-based watchdog cannot
    interrupt. Main-thread only, like any signal-based timeout.
    """
    import contextlib
    import signal

    @contextlib.contextmanager
    def _guard(seconds: int = 10):
        def _raise(_signum, _frame):
            raise BlockedTooLong(f"blocked for more than {seconds}s")

        previous = signal.signal(signal.SIGALRM, _raise)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    return _guard


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
