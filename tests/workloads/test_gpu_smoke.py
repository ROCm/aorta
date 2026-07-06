"""Dependency-free unit tests for the ``gpu_smoke`` workload.

``gpu_smoke`` imports torch lazily inside ``setup()`` / ``run()``, so these
tests install a minimal fake ``torch`` into ``sys.modules`` to exercise the
real code paths (cuda-availability gate, config defaulting incl. an explicit
``steps: 0``, dtype selection, and the tolerance-based pass/fail) without a GPU
or a real torch install.
"""

import sys
import types

import pytest

from aorta.workloads.gpu_smoke import GpuSmokeWorkload


class _FakeScalar:
    def __init__(self, v: float):
        self._v = v

    def item(self) -> float:
        return self._v


class _FakeTensor:
    """All-elements-equal tensor: tracks element count + per-element value."""

    def __init__(self, n: int, val: float = 0.0, err: float = 0.0):
        self.n = n
        self.val = val
        self._err = err  # injected absolute error on the final sum (FAIL-path)

    def add_(self, c: float) -> "_FakeTensor":
        self.val += c
        return self

    def sum(self) -> _FakeScalar:
        return _FakeScalar(self.n * self.val + self._err)


def _make_fake_torch(*, cuda_available: bool = True, sum_err: float = 0.0):
    t = types.ModuleType("torch")
    t.float32 = "float32"
    t.float16 = "float16"
    t.bfloat16 = "bfloat16"

    class _Cuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def set_device(_i):
            pass

        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def get_device_name(_i):
            return "Fake gfx950"

    t.cuda = _Cuda()
    t.device = lambda spec: f"device({spec})"
    t.zeros = lambda n, device=None, dtype=None: _FakeTensor(n, 0.0, err=sum_err)
    return t


@pytest.fixture
def fake_torch(monkeypatch):
    def _install(**kw):
        mod = _make_fake_torch(**kw)
        monkeypatch.setitem(sys.modules, "torch", mod)
        return mod

    return _install


class TestSetup:
    def test_setup_raises_without_cuda(self, fake_torch):
        fake_torch(cuda_available=False)
        wl = GpuSmokeWorkload({})
        with pytest.raises(RuntimeError, match="cuda"):
            wl.setup()

    def test_setup_ok_with_cuda(self, fake_torch):
        fake_torch(cuda_available=True)
        wl = GpuSmokeWorkload({"dtype": "bfloat16"})
        wl.setup()  # should not raise

    def test_setup_raises_on_unknown_dtype(self, fake_torch):
        fake_torch(cuda_available=True)
        wl = GpuSmokeWorkload({"dtype": "flob16"})
        with pytest.raises(ValueError, match="unknown dtype"):
            wl.setup()


class TestRun:
    def test_pass_path(self, fake_torch):
        fake_torch()
        wl = GpuSmokeWorkload({"n": 8, "steps": 3})
        wl.setup()
        r = wl.run()
        assert r.passed is True
        assert r.metrics["sum"] == 24.0  # 8 * 3
        assert r.metrics["expected"] == 24.0
        assert r.executed_iterations == 3
        assert r.total_iterations == 3

    def test_explicit_steps_zero_is_honored(self, fake_torch):
        """`steps: 0` must NOT be treated as missing/defaulted to 1."""
        fake_torch()
        wl = GpuSmokeWorkload({"n": 8, "steps": 0})
        wl.setup()
        r = wl.run()
        assert r.total_iterations == 0
        assert r.configured_iterations == 0
        assert r.metrics["expected"] == 0.0
        assert r.passed is True  # sum 0 == expected 0

    def test_default_steps_when_missing(self, fake_torch):
        fake_torch()
        wl = GpuSmokeWorkload({"n": 4})
        wl.setup()
        r = wl.run()
        assert r.total_iterations == 1  # defaulted
        assert r.metrics["expected"] == 4.0

    def test_fail_path_on_corruption(self, fake_torch):
        """A result outside tolerance fails (simulated silent corruption)."""
        fake_torch(sum_err=5.0)  # push the sum well outside rel/abs tolerance
        wl = GpuSmokeWorkload({"n": 8, "steps": 1})
        wl.setup()
        r = wl.run()
        assert r.passed is False
        assert r.failure_count == 1

    def test_tolerance_absorbs_tiny_error(self, fake_torch):
        """A sub-tolerance rounding error still passes."""
        fake_torch(sum_err=1e-4)
        wl = GpuSmokeWorkload({"n": 8, "steps": 1})
        wl.setup()
        r = wl.run()
        assert r.passed is True

    def test_negative_n_raises(self, fake_torch):
        fake_torch()
        wl = GpuSmokeWorkload({"n": -1})
        wl.setup()
        with pytest.raises(ValueError, match="n must be >= 1"):
            wl.run()

    def test_negative_steps_raises(self, fake_torch):
        fake_torch()
        wl = GpuSmokeWorkload({"n": 8, "steps": -1})
        wl.setup()
        with pytest.raises(ValueError, match="steps must be >= 0"):
            wl.run()
