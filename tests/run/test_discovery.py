"""Tests for workload discovery via entry-points."""

from importlib.metadata import EntryPoint, EntryPoints
from unittest.mock import MagicMock, patch

import pytest

from aorta.run.discovery import (
    UnknownWorkloadError,
    discover_workloads,
    get_workload_class,
    get_workload_policy,
    get_workload_startup_env,
)
from aorta.run.validation import (
    IN_PROCESS_ONLY_POLICY,
    PROCESS_REQUIRED_POLICY,
)
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
        """discover_workloads returns a dict.

        Use a patched empty entry-point set so the test result is not
        sensitive to whatever plugins happen to be installed in the
        ambient environment (which could pollute the dict and trigger
        warning logs unrelated to this assertion).
        """
        mock_eps = MagicMock()
        mock_eps.select.return_value = []
        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            workloads = discover_workloads()
        assert isinstance(workloads, dict)
        assert workloads == {}

    def test_registered_workloads_are_found(self):
        """A registered entry-point is found and resolved to its class.

        Patch ``entry_points`` to a deterministic single-entry fixture
        so the test does not depend on whether ``aorta`` is installed
        in editable mode (entry-points present) vs. non-editable mode
        (entry-points missing) in the current environment.
        """
        mock_ep = MagicMock()
        mock_ep.name = "registered_one"
        mock_ep.load.return_value = MockWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            workloads = discover_workloads()

        assert workloads == {"registered_one": MockWorkload}

    def test_race_entry_point_resolves(self):
        """The `race` entry-point resolves to RaceWorkload."""
        from aorta.workloads.race import RaceWorkload

        mock_ep = MagicMock()
        mock_ep.name = "race"
        mock_ep.load.return_value = RaceWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("importlib.metadata.entry_points", return_value=mock_eps):
            workloads = discover_workloads()

        assert workloads == {"race": RaceWorkload}

    def test_handles_load_failure_gracefully(self, caplog):
        """Failed workload loads are logged but don't crash discovery."""
        import logging

        mock_ep_good = MagicMock()
        mock_ep_good.name = "mock_good"
        mock_ep_good.load.return_value = MockWorkload

        mock_ep_bad = MagicMock()
        mock_ep_bad.name = "mock_bad"
        mock_ep_bad.load.side_effect = ImportError("Module not found")

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep_good, mock_ep_bad]

        with caplog.at_level(logging.WARNING, logger="aorta.run.discovery"):
            with patch("importlib.metadata.entry_points", return_value=mock_eps):
                workloads = discover_workloads()

        # Good workload should still be loaded
        assert "mock_good" in workloads
        assert workloads["mock_good"] == MockWorkload

        # Bad workload logged a warning via the module logger,
        # WITH exc_info attached (otherwise plugin load failures are
        # essentially undiagnosable -- the most common cause is an
        # ImportError chain inside the plugin module).
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "mock_bad" in r.getMessage()
        ]
        assert warnings, "expected a warning naming the failed entry point"
        assert any(r.exc_info is not None and r.exc_info[0] is ImportError for r in warnings), (
            "warning must carry exc_info so the traceback survives"
        )

    def test_skips_non_workload_subclass(self, caplog):
        """Entry points that don't resolve to a Workload subclass are skipped."""
        import logging

        # A plain function -- not a class at all.
        def not_a_class():
            return "nope"

        # A class, but not a Workload subclass.
        class NotAWorkload:
            pass

        ep_func = MagicMock()
        ep_func.name = "bad_func"
        ep_func.load.return_value = not_a_class

        ep_class = MagicMock()
        ep_class.name = "bad_class"
        ep_class.load.return_value = NotAWorkload

        ep_good = MagicMock()
        ep_good.name = "good"
        ep_good.load.return_value = MockWorkload

        mock_eps = MagicMock()
        mock_eps.select.return_value = [ep_func, ep_class, ep_good]

        with caplog.at_level(logging.WARNING, logger="aorta.run.discovery"):
            with patch("importlib.metadata.entry_points", return_value=mock_eps):
                workloads = discover_workloads()

        assert "good" in workloads
        assert "bad_func" not in workloads
        assert "bad_class" not in workloads
        # Both invalid entries logged a warning.
        msgs = [r.getMessage() for r in caplog.records]
        assert any("bad_func" in m for m in msgs)
        assert any("bad_class" in m for m in msgs)

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
        with pytest.raises(UnknownWorkloadError) as exc_info:
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


class TestGetWorkloadPolicy:
    def test_unknown_workload_uses_specific_error(self):
        entries = EntryPoints([])
        with patch("importlib.metadata.entry_points", return_value=entries):
            with pytest.raises(UnknownWorkloadError, match="not found"):
                get_workload_policy("missing")

    def test_policy_resolves_without_loading_workload_module(self):
        entries = EntryPoints(
            [
                EntryPoint(
                    name="env_sensitive",
                    value="module_that_must_not_import:Workload",
                    group="aorta.workloads",
                ),
                EntryPoint(
                    name="env_sensitive",
                    value="aorta.run.validation:PROCESS_REQUIRED_POLICY",
                    group="aorta.workload_policies",
                ),
            ]
        )
        with patch("importlib.metadata.entry_points", return_value=entries):
            assert get_workload_policy("env_sensitive") == PROCESS_REQUIRED_POLICY

    def test_unregistered_policy_defaults_to_in_process_only(self):
        entries = EntryPoints(
            [
                EntryPoint(
                    name="legacy",
                    value="legacy_plugin:LegacyWorkload",
                    group="aorta.workloads",
                )
            ]
        )
        with patch("importlib.metadata.entry_points", return_value=entries):
            assert get_workload_policy("legacy") == IN_PROCESS_ONLY_POLICY

    def test_unknown_policy_target_is_rejected_without_import(self):
        entries = EntryPoints(
            [
                EntryPoint(
                    name="broken",
                    value="broken_plugin:BrokenWorkload",
                    group="aorta.workloads",
                ),
                EntryPoint(
                    name="broken",
                    value="broken_plugin:UNRECOGNIZED_POLICY",
                    group="aorta.workload_policies",
                ),
            ]
        )
        with patch("importlib.metadata.entry_points", return_value=entries):
            with pytest.raises(ValueError, match="unsupported isolation policy"):
                get_workload_policy("broken")


def test_startup_env_provider_loads_without_workload_entry_point(tmp_path, monkeypatch):
    module = tmp_path / "startup_provider.py"
    module.write_text(
        "def provide(config):\n"
        "    return {'EARLY_VALUE': str(config.get('value', 4))}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    entries = EntryPoints(
        [
            EntryPoint(
                name="early",
                value="must_not_import:Workload",
                group="aorta.workloads",
            ),
            EntryPoint(
                name="early",
                value="startup_provider:provide",
                group="aorta.workload_startup_env",
            ),
        ]
    )

    with patch("importlib.metadata.entry_points", return_value=entries):
        assert get_workload_startup_env("early", {"value": 7}) == {
            "EARLY_VALUE": "7"
        }
