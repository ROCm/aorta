"""Schema + end-to-end tests for verdict-keyed retention (issue #231).

Two layers:

* **Schema** -- the ``retain`` block parses onto ``ProbeExtras.retain``
  via :func:`aorta.probe.recipe_builder.build_probe_recipe_from_dict`,
  with the same strict-validation contract as ``stop_after``.
* **End-to-end** -- :class:`aorta.workloads._subprocess.SubprocessWorkload`
  prunes each trial's heavy artifacts according to the verdict->level
  mapping, while the trial record (``result.json``) always survives. This
  includes the issue's disk acceptance criterion (a multi-trial cell with
  a heavy collector retains ~one heavy artifact per *failing* trial, not
  one per trial).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.probe.recipe_builder import build_probe_recipe_from_dict
from aorta.run.retention import RETAIN_LEVELS
from aorta.triage.recipe import _RETAIN_LEVELS, RecipeSchemaError, RetainPolicy
from aorta.workloads._subprocess import (
    CONFIG_KEY_LOG_PREFIX,
    CONFIG_KEY_PROBE_EXTRAS,
    CONFIG_KEY_SUBPROCESS_ARGV,
    SubprocessWorkload,
)


def _probe_dict(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "mode": "probe",
        "trials": 1,
        "mitigation_axis": ["none"],
        "diagnostic_axis": ["none"],
    }
    base.update(overrides)
    return base


# ---- schema ---------------------------------------------------------------


def test_retain_omitted_is_none():
    r = build_probe_recipe_from_dict(_probe_dict(), [])
    assert r.probe_extras.retain is None


def test_retain_parses_onto_probe_extras():
    r = build_probe_recipe_from_dict(
        _probe_dict(retain={"on_fail": "full", "on_pass": "summary", "on_error": "log"}),
        [],
    )
    assert r.probe_extras.retain == RetainPolicy(on_fail="full", on_pass="summary", on_error="log")


def test_retain_partial_defaults_to_full():
    r = build_probe_recipe_from_dict(_probe_dict(retain={"on_pass": "none"}), [])
    assert r.probe_extras.retain == RetainPolicy(on_fail="full", on_pass="none", on_error="full")


def test_retain_rejects_unknown_key():
    with pytest.raises(RecipeSchemaError, match="unknown keys"):
        build_probe_recipe_from_dict(_probe_dict(retain={"on_skip": "full"}), [])


def test_retain_rejects_bad_level():
    with pytest.raises(RecipeSchemaError, match="must be one of"):
        build_probe_recipe_from_dict(_probe_dict(retain={"on_fail": "everything"}), [])


def test_retain_rejects_non_mapping():
    with pytest.raises(RecipeSchemaError, match="must be a mapping"):
        build_probe_recipe_from_dict(_probe_dict(retain=["full"]), [])


@pytest.mark.parametrize("bad_value", [{"nested": "full"}, ["full"]])
def test_retain_rejects_unhashable_level_value(bad_value):
    # An unhashable YAML node (dict/list) as a level value must surface a
    # clean RecipeSchemaError, not a raw TypeError from the set-membership
    # check (`value not in _RETAIN_LEVELS` hashes `value`).
    with pytest.raises(RecipeSchemaError, match="must be one of"):
        build_probe_recipe_from_dict(_probe_dict(retain={"on_fail": bad_value}), [])


def test_retain_multiple_bad_keys_report_deterministically():
    # With several invalid level values, the validator iterates keys in a
    # fixed (sorted) order so the surfaced error is stable across runs /
    # hash seeds. Sorted order is on_error < on_fail < on_pass, so on_fail
    # is reported first here.
    bad = {"on_pass": "bogus_p", "on_fail": "bogus_f"}
    with pytest.raises(RecipeSchemaError, match=r"retain\.on_fail: must be one of"):
        build_probe_recipe_from_dict(_probe_dict(retain=bad), [])


def test_schema_levels_match_engine_ladder():
    """Drift guard: the schema's accepted levels must equal the engine's."""
    assert _RETAIN_LEVELS == set(RETAIN_LEVELS)


# ---- end-to-end through SubprocessWorkload --------------------------------


def _run_trial(tmp_path: Path, *, argv: list[str], trial_idx: int, retain: dict | None):
    """Run one probe trial in its own cell dir; return (trial_dir, result)."""
    cell_dir = tmp_path / f"cell_{trial_idx}"
    workload_subdir = cell_dir / "_subprocess"
    workload_subdir.mkdir(parents=True, exist_ok=True)
    prefix = workload_subdir / f"trial_d0_m0_t{trial_idx}"
    extras: dict = {
        "cell_name": "none-none",
        "env_passthrough_mode": "inherit",
        "timeout_per_trial": None,
        "cell_env_vars": {},
    }
    if retain is not None:
        extras["retain"] = retain
    wl = SubprocessWorkload(
        {
            CONFIG_KEY_SUBPROCESS_ARGV: argv,
            CONFIG_KEY_LOG_PREFIX: str(prefix),
            CONFIG_KEY_PROBE_EXTRAS: extras,
        }
    )
    wl.setup()
    trial_dir = cell_dir / f"trial_{trial_idx}"
    # Simulate a collector dropping a heavy artifact during the trial.
    (trial_dir / "trace.bin").write_text("x" * 4096, encoding="utf-8")
    result = wl.run()
    return trial_dir, result


_POLICY = {"on_fail": "full", "on_pass": "summary", "on_error": "log"}


def test_pass_trial_drops_heavy_keeps_record(tmp_path: Path):
    trial_dir, result = _run_trial(tmp_path, argv=["true"], trial_idx=0, retain=_POLICY)
    assert result.passed is True
    assert not (trial_dir / "trace.bin").exists()  # heavy dropped at summary
    assert (trial_dir / "result.json").is_file()  # record survives


def test_fail_trial_keeps_heavy(tmp_path: Path):
    trial_dir, result = _run_trial(tmp_path, argv=["false"], trial_idx=0, retain=_POLICY)
    assert result.passed is False
    assert (trial_dir / "trace.bin").is_file()  # full -> heavy kept
    assert (trial_dir / "result.json").is_file()


def test_default_keeps_everything(tmp_path: Path):
    """No ``retain`` block -> legacy keep-everything behaviour."""
    trial_dir, _ = _run_trial(tmp_path, argv=["true"], trial_idx=0, retain=None)
    assert (trial_dir / "trace.bin").is_file()
    assert (trial_dir / "result.json").is_file()


def test_record_survives_at_none_level(tmp_path: Path):
    trial_dir, _ = _run_trial(
        tmp_path, argv=["true"], trial_idx=0, retain={"on_pass": "none"}
    )
    assert (trial_dir / "result.json").is_file()
    assert not (trial_dir / "trace.bin").exists()
    assert not (trial_dir / "stdout.log").exists()  # none drops logs too


def test_non_dict_retain_payload_warns_and_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A malformed programmatic ``retain`` payload (e.g. a bare string, not a
    mapping) must not sink the trial -- retention is best-effort, so warn +
    skip and keep every artifact (the schema path rejects this, but a
    programmatic caller can bypass it)."""
    import logging

    with caplog.at_level(logging.WARNING):
        trial_dir, result = _run_trial(
            tmp_path,
            argv=["true"],
            trial_idx=0,
            retain="summary",  # type: ignore[arg-type]
        )
    assert result.passed is True
    assert (trial_dir / "trace.bin").is_file()  # nothing pruned
    assert (trial_dir / "result.json").is_file()
    assert any("expected a mapping" in r.getMessage() for r in caplog.records)


def _read_result(trial_dir: Path) -> dict:
    return json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))


def test_retention_outcome_recorded_in_result_json(tmp_path: Path):
    """A pruning trial stamps the applied level + deleted list into the
    record so a missing heavy artifact is auditable post-bundle (oyazdanb)."""
    trial_dir, _ = _run_trial(tmp_path, argv=["true"], trial_idx=0, retain=_POLICY)
    retention = _read_result(trial_dir).get("capture", {}).get("retention")
    assert retention is not None, "capture.retention missing from result.json"
    assert retention.get("level") == "summary"  # on_pass=summary
    assert "trace.bin" in retention.get("deleted", [])
    assert retention.get("freed_bytes", 0) >= 4096


def test_retention_record_absent_without_policy(tmp_path: Path):
    """No ``retain`` block -> no retention audit key (keep-everything)."""
    trial_dir, _ = _run_trial(tmp_path, argv=["true"], trial_idx=0, retain=None)
    assert "retention" not in _read_result(trial_dir).get("capture", {})


def test_retention_record_on_full_is_noop_but_audited(tmp_path: Path):
    """A failing trial at ``full`` keeps everything; the audit record still
    notes the level so a reader sees retention was active (deleted empty)."""
    trial_dir, _ = _run_trial(tmp_path, argv=["false"], trial_idx=0, retain=_POLICY)
    retention = _read_result(trial_dir).get("capture", {}).get("retention")
    assert retention is not None, "capture.retention missing from result.json"
    assert retention.get("level") == "full"
    assert retention.get("deleted") == []
    assert (trial_dir / "trace.bin").is_file()  # nothing pruned at full


def test_disk_criterion_heavy_kept_only_for_fails(tmp_path: Path):
    """A 10-trial cell with 3 fails retains 3 heavy artifacts, not 10."""
    n_trials = 10
    fail_idxs = {2, 5, 8}
    heavy_kept = 0
    records = 0
    for i in range(n_trials):
        argv = ["false"] if i in fail_idxs else ["true"]
        trial_dir, _ = _run_trial(tmp_path, argv=argv, trial_idx=i, retain=_POLICY)
        if (trial_dir / "trace.bin").is_file():
            heavy_kept += 1
        if (trial_dir / "result.json").is_file():
            records += 1
    assert heavy_kept == len(fail_idxs)  # ~3 heavy, not 10
    assert records == n_trials  # every trial's bookkeeping survives
