"""
Tests for GPU hardware control (lock-clock, power-limit).

All subprocess calls are mocked so tests run without real GPUs.
These tests do NOT require PyTorch -- gpu_control.py is loaded directly
by file path so the torch-dependent aorta.utils.__init__ is bypassed.
"""

from unittest.mock import MagicMock, patch
import importlib.util
import os
import subprocess
import sys

# Load gpu_control.py directly to avoid the torch-dependent aorta.utils init chain
_GPU_CTRL_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "src", "aorta", "utils", "gpu_control.py",
)
_spec = importlib.util.spec_from_file_location("gpu_control", _GPU_CTRL_PATH)
_gpu_control = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _gpu_control
_spec.loader.exec_module(_gpu_control)

GPUConfig = _gpu_control.GPUConfig
GPUControlConfig = _gpu_control.GPUControlConfig
GPUControlManager = _gpu_control.GPUControlManager
GPUController = _gpu_control.GPUController
GPUHardwareInfo = _gpu_control.GPUHardwareInfo
GPUVendor = _gpu_control.GPUVendor
MultiGPUConfig = _gpu_control.MultiGPUConfig
MultiGPUController = _gpu_control.MultiGPUController
detect_gpu = _gpu_control.detect_gpu
get_gpu_count = _gpu_control.get_gpu_count


# ---------------------------------------------------------------------------
# GPUVendor / GPUHardwareInfo
# ---------------------------------------------------------------------------


class TestGPUVendor:
    def test_enum_members(self):
        assert GPUVendor.AMD.name == "AMD"
        assert GPUVendor.NVIDIA.name == "NVIDIA"
        assert GPUVendor.UNKNOWN.name == "UNKNOWN"


class TestGPUHardwareInfo:
    def test_to_dict_contains_all_fields(self):
        info = GPUHardwareInfo(
            vendor=GPUVendor.AMD,
            architecture="gfx942",
            device_id=0,
            device_name="MI300X",
            power_current_watts=150.0,
            power_limit_watts=500.0,
            gpu_clock_current=1800,
            temperature=55.0,
        )
        d = info.to_dict()

        assert d["vendor"] == "AMD"
        assert d["architecture"] == "gfx942"
        assert d["device_name"] == "MI300X"
        assert d["power_current_watts"] == 150.0
        assert d["power_limit_watts"] == 500.0
        assert d["gpu_clock_current"] == 1800
        assert d["temperature"] == 55.0

    def test_to_dict_defaults(self):
        info = GPUHardwareInfo(vendor=GPUVendor.UNKNOWN)
        d = info.to_dict()

        assert d["vendor"] == "UNKNOWN"
        assert d["device_id"] == 0
        assert d["power_current_watts"] is None
        assert d["gpu_clock_current"] is None


# ---------------------------------------------------------------------------
# GPUConfig / MultiGPUConfig
# ---------------------------------------------------------------------------


class TestGPUConfig:
    def test_defaults(self):
        cfg = GPUConfig()
        assert cfg.device_id == 0
        assert cfg.power_limit_watts is None
        assert cfg.gpu_clock_level is None
        assert cfg.gpu_clock_mhz is None

    def test_lock_clock_level(self):
        cfg = GPUConfig(gpu_clock_level=3)
        assert cfg.gpu_clock_level == 3

    def test_power_limit(self):
        cfg = GPUConfig(power_limit_watts=200)
        assert cfg.power_limit_watts == 200

    def test_nvidia_clock_range(self):
        cfg = GPUConfig(gpu_clock_mhz=(1200, 1800))
        assert cfg.gpu_clock_mhz == (1200, 1800)


class TestMultiGPUConfig:
    def test_get_config_for_device_uses_default(self):
        default = GPUConfig(power_limit_watts=200, gpu_clock_level=5)
        multi = MultiGPUConfig(default_config=default, device_ids=[0, 1])

        cfg = multi.get_config_for_device(1)
        assert cfg is not None
        assert cfg.device_id == 1
        assert cfg.power_limit_watts == 200
        assert cfg.gpu_clock_level == 5

    def test_get_config_for_device_per_gpu_override(self):
        default = GPUConfig(power_limit_watts=200)
        override = GPUConfig(device_id=1, power_limit_watts=300)
        multi = MultiGPUConfig(
            default_config=default,
            gpu_configs={1: override},
        )

        assert multi.get_config_for_device(0).power_limit_watts == 200
        assert multi.get_config_for_device(1).power_limit_watts == 300

    def test_get_config_for_device_no_config_returns_none(self):
        multi = MultiGPUConfig()
        assert multi.get_config_for_device(0) is None


# ---------------------------------------------------------------------------
# detect_gpu / get_gpu_count
# ---------------------------------------------------------------------------


def _completed(stdout="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class TestDetectGPU:
    def test_detects_amd(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed(
                stdout="  Name:                    amdgcn-amd-amdhsa--gfx942\n"
            )
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            vendor, arch = detect_gpu()
        assert vendor == GPUVendor.AMD
        assert arch == "gfx942"

    def test_detects_nvidia(self):
        with patch.object(_gpu_control, "_detect_amd_gpu", return_value=None), \
             patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed(stdout="8.9\n")
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            vendor, arch = detect_gpu()
        assert vendor == GPUVendor.NVIDIA
        assert arch == "sm_89"

    def test_returns_unknown_when_no_gpu(self):
        with patch.object(_gpu_control, "_detect_amd_gpu", return_value=None), \
             patch.object(_gpu_control, "_detect_nvidia_gpu", return_value=None):
            vendor, arch = detect_gpu()
        assert vendor == GPUVendor.UNKNOWN
        assert arch is None


class TestGetGPUCount:
    def test_nvidia_count(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed(stdout="4\n4\n4\n4\n")
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            assert get_gpu_count() == 4

    def test_amd_count(self):
        def side_effect(cmd, **kw):
            if "nvidia-smi" in cmd:
                raise FileNotFoundError
            return _completed(stdout=(
                "GPU[0]          : 0x740c\n"
                "GPU[1]          : 0x740c\n"
            ))

        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.side_effect = side_effect
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            assert get_gpu_count() == 2

    def test_returns_zero_when_no_tools(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            assert get_gpu_count() == 0


# ---------------------------------------------------------------------------
# GPUController — AMD lock-clock / power-limit
# ---------------------------------------------------------------------------


class TestGPUControllerAMD:
    """Test that the correct rocm-smi commands are issued for AMD GPUs."""

    def _make_controller(self):
        with patch.object(_gpu_control, "detect_gpu", return_value=(GPUVendor.AMD, "gfx942")):
            return GPUController(device_id=0)

    def test_apply_lock_clock(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0, gpu_clock_level=3)
            result = ctrl.apply_config(cfg)

        assert result is True
        mock_sub.run.assert_any_call(
            ["rocm-smi", "-d", "0", "--setsclk", "3"],
            capture_output=True, text=True, timeout=10,
        )

    def test_apply_mem_clock(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0, mem_clock_level=5)
            result = ctrl.apply_config(cfg)

        assert result is True
        mock_sub.run.assert_any_call(
            ["rocm-smi", "-d", "0", "--setmclk", "5"],
            capture_output=True, text=True, timeout=10,
        )

    def test_apply_power_limit(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0, power_limit_watts=200)
            result = ctrl.apply_config(cfg)

        assert result is True
        mock_sub.run.assert_any_call(
            ["rocm-smi", "-d", "0", "--setpoweroverdrive", "200"],
            capture_output=True, text=True, timeout=10,
        )

    def test_apply_combined_lock_clock_and_power(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0, gpu_clock_level=7, power_limit_watts=300)
            result = ctrl.apply_config(cfg)

        assert result is True
        calls = [c.args[0] for c in mock_sub.run.call_args_list]
        assert ["rocm-smi", "-d", "0", "--setpoweroverdrive", "300"] in calls
        assert ["rocm-smi", "-d", "0", "--setsclk", "7"] in calls

    def test_apply_returns_false_on_failure(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed(returncode=1)
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0, gpu_clock_level=3)
            result = ctrl.apply_config(cfg)

        assert result is False

    def test_apply_noop_when_all_none(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0)
            result = ctrl.apply_config(cfg)

        assert result is True
        mock_sub.run.assert_not_called()

    def test_reset_calls_resetclocks(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            result = ctrl.reset_config()

        assert result is True
        mock_sub.run.assert_any_call(
            ["rocm-smi", "-d", "0", "--resetclocks"],
            capture_output=True, text=True, timeout=10,
        )


# ---------------------------------------------------------------------------
# GPUController — NVIDIA lock-clock / power-limit
# ---------------------------------------------------------------------------


class TestGPUControllerNVIDIA:
    """Test that the correct nvidia-smi commands are issued for NVIDIA GPUs."""

    def _make_controller(self):
        with patch.object(_gpu_control, "detect_gpu", return_value=(GPUVendor.NVIDIA, "sm_90")):
            return GPUController(device_id=0)

    def test_apply_lock_gpu_clocks(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0, gpu_clock_mhz=(1200, 1800))
            result = ctrl.apply_config(cfg)

        assert result is True
        mock_sub.run.assert_any_call(
            ["nvidia-smi", "-i", "0", "-lgc", "1200,1800"],
            capture_output=True, text=True, timeout=10,
        )

    def test_apply_lock_mem_clocks(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0, mem_clock_mhz=(800, 1200))
            result = ctrl.apply_config(cfg)

        assert result is True
        mock_sub.run.assert_any_call(
            ["nvidia-smi", "-i", "0", "-lmc", "800,1200"],
            capture_output=True, text=True, timeout=10,
        )

    def test_apply_power_limit(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0, power_limit_watts=250)
            result = ctrl.apply_config(cfg)

        assert result is True
        mock_sub.run.assert_any_call(
            ["nvidia-smi", "-i", "0", "-pl", "250"],
            capture_output=True, text=True, timeout=10,
        )

    def test_apply_combined_lock_clock_and_power(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0, gpu_clock_mhz=(1500, 1500), power_limit_watts=300)
            result = ctrl.apply_config(cfg)

        assert result is True
        calls = [c.args[0] for c in mock_sub.run.call_args_list]
        assert ["nvidia-smi", "-i", "0", "-pl", "300"] in calls
        assert ["nvidia-smi", "-i", "0", "-lgc", "1500,1500"] in calls

    def test_apply_returns_false_on_failure(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed(returncode=1)
            ctrl = self._make_controller()
            cfg = GPUConfig(device_id=0, power_limit_watts=250)
            result = ctrl.apply_config(cfg)

        assert result is False

    def test_reset_calls_rgc_and_rmc(self):
        with patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()
            ctrl = self._make_controller()
            result = ctrl.reset_config()

        assert result is True
        calls = [c.args[0] for c in mock_sub.run.call_args_list]
        assert ["nvidia-smi", "-i", "0", "-rgc"] in calls
        assert ["nvidia-smi", "-i", "0", "-rmc"] in calls


class TestGPUControllerUnknown:
    def test_apply_returns_false(self):
        with patch.object(_gpu_control, "detect_gpu", return_value=(GPUVendor.UNKNOWN, None)):
            ctrl = GPUController(device_id=0)
        assert ctrl.apply_config(GPUConfig()) is False

    def test_reset_returns_false(self):
        with patch.object(_gpu_control, "detect_gpu", return_value=(GPUVendor.UNKNOWN, None)):
            ctrl = GPUController(device_id=0)
        assert ctrl.reset_config() is False


# ---------------------------------------------------------------------------
# GPUControlConfig
# ---------------------------------------------------------------------------


class TestGPUControlConfig:
    def test_defaults(self):
        cfg = GPUControlConfig()
        assert cfg.enabled is False
        assert cfg.power_limit_watts is None
        assert cfg.gpu_clock_level is None
        assert cfg.reset_on_exit is True

    def test_lock_clock_and_power(self):
        cfg = GPUControlConfig(
            enabled=True,
            gpu_clock_level=3,
            power_limit_watts=200,
        )
        assert cfg.enabled is True
        assert cfg.gpu_clock_level == 3
        assert cfg.power_limit_watts == 200


# ---------------------------------------------------------------------------
# GPUControlManager
# ---------------------------------------------------------------------------


class TestGPUControlManager:
    def test_available_when_enabled(self):
        mgr = GPUControlManager(GPUControlConfig(enabled=True))
        assert mgr.available is True

    def test_not_available_when_disabled(self):
        mgr = GPUControlManager(GPUControlConfig(enabled=False))
        assert mgr.available is False

    def test_apply_returns_empty_when_disabled(self):
        mgr = GPUControlManager(GPUControlConfig(enabled=False))
        assert mgr.apply() == {}

    def test_reset_is_noop_when_not_applied(self):
        mgr = GPUControlManager(GPUControlConfig(enabled=True))
        mgr.reset()  # should not raise

    def test_apply_creates_controller_and_applies(self):
        mock_ctrl = MagicMock()
        mock_ctrl.apply_config.return_value = {0: True}
        mock_ctrl.get_all_hardware_info.return_value = {
            0: GPUHardwareInfo(vendor=GPUVendor.AMD, device_id=0)
        }

        with patch.object(_gpu_control, "MultiGPUController", return_value=mock_ctrl):
            cfg = GPUControlConfig(enabled=True, gpu_clock_level=5, power_limit_watts=200)
            mgr = GPUControlManager(cfg)
            snapshot = mgr.apply()

        mock_ctrl.apply_config.assert_called_once()
        applied_multi_cfg = mock_ctrl.apply_config.call_args[0][0]
        assert isinstance(applied_multi_cfg, MultiGPUConfig)

        default = applied_multi_cfg.default_config
        assert default.gpu_clock_level == 5
        assert default.power_limit_watts == 200
        assert "gpu_hardware_state" in snapshot

    def test_reset_calls_reset_all(self):
        mock_ctrl = MagicMock()
        mock_ctrl.apply_config.return_value = {0: True}
        mock_ctrl.get_all_hardware_info.return_value = {}
        mock_ctrl.reset_all.return_value = {0: True}

        with patch.object(_gpu_control, "MultiGPUController", return_value=mock_ctrl):
            cfg = GPUControlConfig(enabled=True, gpu_clock_level=3)
            mgr = GPUControlManager(cfg)
            mgr.apply()
            mgr.reset()

        mock_ctrl.reset_all.assert_called_once()

    def test_context_manager_applies_and_resets(self):
        mock_ctrl = MagicMock()
        mock_ctrl.apply_config.return_value = {0: True}
        mock_ctrl.get_all_hardware_info.return_value = {}
        mock_ctrl.reset_all.return_value = {0: True}

        with patch.object(_gpu_control, "MultiGPUController", return_value=mock_ctrl):
            cfg = GPUControlConfig(enabled=True, gpu_clock_level=3, reset_on_exit=True)
            with GPUControlManager(cfg):
                mock_ctrl.apply_config.assert_called_once()

        mock_ctrl.reset_all.assert_called_once()

    def test_context_manager_skips_reset_when_disabled(self):
        mock_ctrl = MagicMock()
        mock_ctrl.apply_config.return_value = {0: True}
        mock_ctrl.get_all_hardware_info.return_value = {}

        with patch.object(_gpu_control, "MultiGPUController", return_value=mock_ctrl):
            cfg = GPUControlConfig(enabled=True, gpu_clock_level=3, reset_on_exit=False)
            with GPUControlManager(cfg):
                pass

        mock_ctrl.reset_all.assert_not_called()

    def test_snapshot_returns_per_gpu_state(self):
        mock_ctrl = MagicMock()
        mock_ctrl.apply_config.return_value = {0: True, 1: True}
        mock_ctrl.get_all_hardware_info.return_value = {
            0: GPUHardwareInfo(vendor=GPUVendor.AMD, device_id=0, power_current_watts=100.0),
            1: GPUHardwareInfo(vendor=GPUVendor.AMD, device_id=1, power_current_watts=110.0),
        }

        with patch.object(_gpu_control, "MultiGPUController", return_value=mock_ctrl):
            cfg = GPUControlConfig(enabled=True, power_limit_watts=200)
            mgr = GPUControlManager(cfg)
            snapshot = mgr.apply()

        assert "gpu_hardware_state" in snapshot
        assert "gpu_0" in snapshot["gpu_hardware_state"]
        assert "gpu_1" in snapshot["gpu_hardware_state"]
        assert snapshot["gpu_hardware_state"]["gpu_0"]["power_current_watts"] == 100.0


# ---------------------------------------------------------------------------
# MultiGPUController
# ---------------------------------------------------------------------------


class TestMultiGPUController:
    def test_apply_config_to_multiple_gpus(self):
        with patch.object(_gpu_control, "get_gpu_count", return_value=2), \
             patch.object(_gpu_control, "detect_gpu", return_value=(GPUVendor.AMD, "gfx942")), \
             patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()

            ctrl = MultiGPUController(device_ids=[0, 1])
            multi_cfg = MultiGPUConfig(
                default_config=GPUConfig(gpu_clock_level=5, power_limit_watts=200),
                device_ids=[0, 1],
                parallel=False,
            )
            results = ctrl.apply_config(multi_cfg)

        assert results[0] is True
        assert results[1] is True

        cmds = [c.args[0] for c in mock_sub.run.call_args_list]
        assert ["rocm-smi", "-d", "0", "--setsclk", "5"] in cmds
        assert ["rocm-smi", "-d", "1", "--setsclk", "5"] in cmds
        assert ["rocm-smi", "-d", "0", "--setpoweroverdrive", "200"] in cmds
        assert ["rocm-smi", "-d", "1", "--setpoweroverdrive", "200"] in cmds

    def test_reset_all(self):
        with patch.object(_gpu_control, "get_gpu_count", return_value=2), \
             patch.object(_gpu_control, "detect_gpu", return_value=(GPUVendor.AMD, "gfx942")), \
             patch.object(_gpu_control, "subprocess") as mock_sub:
            mock_sub.run.return_value = _completed()

            ctrl = MultiGPUController(device_ids=[0, 1])
            results = ctrl.reset_all(parallel=False)

        assert results[0] is True
        assert results[1] is True

        cmds = [c.args[0] for c in mock_sub.run.call_args_list]
        assert ["rocm-smi", "-d", "0", "--resetclocks"] in cmds
        assert ["rocm-smi", "-d", "1", "--resetclocks"] in cmds
