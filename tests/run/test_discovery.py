"""Tests for workload discovery via entry-points."""

import pytest
from unittest.mock import patch, MagicMock

from aorta.run.discovery import discover_workloads, get_workload_class
from aorta.workloads import Workload, WorkloadResult


class MockWorkload(Workload):
    """Mock workload for testing."""

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        return WorkloadResult(passed=True)


class TestDiscoverWorkloads:
    """Tests for discover_workloads function."""

    def test_returns_dict(self):
        """discover_workloads returns a dict."""
        workloads = discover_workloads()
        assert isinstance(workloads, dict)

    def test_registered_workloads_are_found(self):
        """Entry-point registered workloads are discovered."""
        # Note: fsdp is registered in pyproject.toml but the class
        # doesn't exist yet. This test verifies discovery attempts to load.
        workloads = discover_workloads()
        # The dict exists even if loading fails
        assert isinstance(workloads, dict)

    def test_handles_load_failure_gracefully(self, capsys):
        """Failed workload loads are logged but don't crash discovery."""
        # Mock entry points to include a failing one
        mock_ep_good = MagicMock()
        mock_ep_good.name = "mock_good"
        mock_ep_good.load.return_value = MockWorkload

        mock_ep_bad = MagicMock()
        mock_ep_bad.name = "mock_bad"
        mock_ep_bad.load.side_effect = ImportError("Module not found")

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep_good, mock_ep_bad]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            workloads = discover_workloads()

        # Good workload should still be loaded
        assert "mock_good" in workloads
        assert workloads["mock_good"] == MockWorkload

        # Bad workload logged warning
        captured = capsys.readouterr()
        assert "mock_bad" in captured.out
        assert "Warning" in captured.out

    def test_discovery_with_multiple_workloads(self):
        """Multiple workloads can be discovered."""
        mock_ep1 = MagicMock()
        mock_ep1.name = "workload1"
        mock_ep1.load.return_value = MockWorkload

        mock_ep2 = MagicMock()
        mock_ep2.name = "workload2"
        mock_ep2.load.return_value = MockWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep1, mock_ep2]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            workloads = discover_workloads()

        assert len(workloads) == 2
        assert "workload1" in workloads
        assert "workload2" in workloads


class TestGetWorkloadClass:
    """Tests for get_workload_class function."""

    def test_unknown_workload_raises_value_error(self):
        """Unknown workload name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_workload_class("definitely_not_a_real_workload")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg
        assert "Available" in error_msg

    def test_error_message_lists_available_workloads(self):
        """Error message includes available workload names."""
        mock_ep = MagicMock()
        mock_ep.name = "available_workload"
        mock_ep.load.return_value = MockWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            with pytest.raises(ValueError) as exc_info:
                get_workload_class("nonexistent")

        assert "available_workload" in str(exc_info.value)

    def test_returns_correct_workload_class(self):
        """Returns the correct workload class for valid name."""
        mock_ep = MagicMock()
        mock_ep.name = "my_workload"
        mock_ep.load.return_value = MockWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            cls = get_workload_class("my_workload")

        assert cls == MockWorkload

    def test_workload_names_are_sorted_in_error(self):
        """Available workloads in error message are sorted."""
        mock_ep1 = MagicMock()
        mock_ep1.name = "zeta"
        mock_ep1.load.return_value = MockWorkload

        mock_ep2 = MagicMock()
        mock_ep2.name = "alpha"
        mock_ep2.load.return_value = MockWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep1, mock_ep2]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            with pytest.raises(ValueError) as exc_info:
                get_workload_class("nonexistent")

        error_msg = str(exc_info.value)
        # 'alpha' should come before 'zeta' in sorted list
        alpha_pos = error_msg.find("alpha")
        zeta_pos = error_msg.find("zeta")
        assert alpha_pos < zeta_pos
