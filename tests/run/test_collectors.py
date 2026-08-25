"""Tests for the collector registry (:mod:`aorta.run.collectors`).

The registry is the dispatch layer between the reserved ``_aorta_collect*``
config keys the dispatcher threads into every trial and the per-collector
packages under :mod:`aorta.instrumentation`. These tests pin the contracts a
caller depends on: ordering, no-op passthrough, cross-collector conflict
rejection, nesting relative to the mirage emulator wrap, and the fail-soft
summary merge.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
from pathlib import Path

import pytest

from aorta.emulation.mirage_launch import (
    CONFIG_KEY_ENVIRONMENT,
    ENV_MIRAGE_BIN,
    wrap_argv_for_environment,
)
from aorta.instrumentation import proton, rocprof
from aorta.registry.types import Environment
from aorta.run.collectors import (
    CONFIG_KEY_COLLECT,
    CONFIG_KEY_COLLECT_DIR,
    CONFIG_KEY_COLLECT_OPTIONS,
    CONFIG_KEY_RESULTS_ROOT,
    KNOWN_RECIPES,
    WRAP_ORDER,
    CollectorSpec,
    active_collectors,
    summarize_collectors,
    validate_collectors,
    wrap_argv_for_collectors,
)

_INNER = ["python", "train.py", "--steps", "10"]


@pytest.fixture
def profilers_on_path(tmp_path, monkeypatch):
    """Fake ``rocprofv3`` + ``env`` on ``$PATH`` so both wraps can be built."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("rocprofv3", "env", "mirage"):
        fake = bin_dir / name
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv(rocprof.ENV_ROCPROF_BIN, raising=False)
    monkeypatch.delenv(proton.ENV_PROTON_PYTHON, raising=False)
    monkeypatch.delenv(ENV_MIRAGE_BIN, raising=False)
    return bin_dir


def _config(names, *, collect_dir=None, options=None, results_root=None):
    config = {CONFIG_KEY_COLLECT: list(names)}
    if collect_dir is not None:
        config[CONFIG_KEY_COLLECT_DIR] = str(collect_dir)
    if options is not None:
        config[CONFIG_KEY_COLLECT_OPTIONS] = options
    if results_root is not None:
        config[CONFIG_KEY_RESULTS_ROOT] = str(results_root)
    return config


# ---- Registry shape ------------------------------------------------------


def test_known_recipes_contains_both_profilers():
    assert {"rocprof", "proton"} <= KNOWN_RECIPES


def test_known_recipes_keeps_the_validated_only_names():
    """Existing recipes name these; dropping one would break them."""
    assert {"numerics", "layer_numerics", "amd_log"} <= KNOWN_RECIPES


def test_wrap_order_puts_rocprof_outermost():
    """rocprof runs a whole command under the profiler while Proton takes over
    a Python script's execution, so rocprof has to be the outer process for the
    pair to compose at all."""
    assert WRAP_ORDER == ("rocprof", "proton")


def test_wrap_order_names_are_known_recipes():
    assert set(WRAP_ORDER) <= KNOWN_RECIPES


def test_collector_spec_is_frozen():
    spec = CollectorSpec("x", None, lambda opts: {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "y"  # type: ignore[misc]


# ---- active_collectors ---------------------------------------------------


def test_active_collectors_empty_without_the_key():
    assert active_collectors({}) == ()


def test_active_collectors_empty_for_non_sequence():
    assert active_collectors({CONFIG_KEY_COLLECT: "rocprof"}) == ()


def test_active_collectors_orders_by_wrap_order():
    assert active_collectors(_config(["proton", "rocprof"])) == ("rocprof", "proton")


def test_active_collectors_appends_non_wrapping_names_after():
    assert active_collectors(_config(["layer_numerics", "proton"])) == (
        "proton",
        "layer_numerics",
    )


def test_active_collectors_dedups_non_wrapping_names():
    assert active_collectors(_config(["layer_numerics", "layer_numerics"])) == ("layer_numerics",)


def test_active_collectors_drops_unknown_and_non_string_entries():
    """``run_trials`` and the recipe loader reject these up front, so anything
    left here came from a hand-built config and must not crash a trial."""
    assert active_collectors(_config(["rocprof", "nope", 7, None])) == ("rocprof",)


def test_active_collectors_accepts_a_tuple():
    assert active_collectors({CONFIG_KEY_COLLECT: ("rocprof",)}) == ("rocprof",)


# ---- validate_collectors -------------------------------------------------


def test_validate_collectors_accepts_empty():
    validate_collectors([])


def test_validate_collectors_accepts_valid_options():
    validate_collectors(["rocprof"], {"rocprof": {"trace": "kernel,hip"}})


def test_validate_collectors_rejects_bad_rocprof_option():
    with pytest.raises(ValueError, match="rocprof: unknown option"):
        validate_collectors(["rocprof"], {"rocprof": {"nope": "1"}})


def test_validate_collectors_rejects_bad_proton_option():
    with pytest.raises(ValueError, match="proton option 'backend'"):
        validate_collectors(["proton"], {"proton": {"backend": "nope"}})


def test_validate_collectors_ignores_unknown_names():
    """The caller already checked against KNOWN_RECIPES; its message stays the
    one the operator sees."""
    validate_collectors(["not_a_collector"])


def test_validate_collectors_accepts_anything_for_wrapper_owned_collectors():
    validate_collectors(["layer_numerics"], {"layer_numerics": {"NANLOG_SPEC": "{}"}})


@pytest.mark.parametrize("backend", ["auto", "rocprofiler", "roctracer"])
def test_validate_collectors_rejects_queue_interception_conflict(backend):
    """rocprof and a queue-intercepting Proton backend both install an HSA
    queue interceptor; the second to attach reports nothing."""
    with pytest.raises(ValueError, match="queue interceptor"):
        validate_collectors(["rocprof", "proton"], {"proton": {"backend": backend}})


def test_validate_collectors_conflict_fires_on_the_proton_default():
    """``collect: [rocprof, proton]`` with no options is the conflicting pair.

    The default backend is ``auto``, which on AMD resolves to ``rocprofiler``
    or ``roctracer`` -- both intercepting. The guard cannot prove the pairing is
    safe, so it rejects it and says what ``auto`` resolves to.
    """
    with pytest.raises(ValueError, match="'auto'.*resolves to rocprofiler or roctracer"):
        validate_collectors(["rocprof", "proton"])


def test_validate_collectors_conflict_names_the_explicit_backend():
    with pytest.raises(ValueError, match="'roctracer'"):
        validate_collectors(["rocprof", "proton"], {"proton": {"backend": "roctracer"}})


def test_validate_collectors_conflict_message_names_the_escape_hatch():
    with pytest.raises(ValueError, match="instrumentation"):
        validate_collectors(["rocprof", "proton"])


def test_validate_collectors_allows_instrumentation_backend_alongside_rocprof():
    """Intra-kernel measurement needs no queue interception, so the pair runs."""
    validate_collectors(["rocprof", "proton"], {"proton": {"backend": "instrumentation"}})


def test_validate_collectors_order_independent():
    with pytest.raises(ValueError, match="queue interceptor"):
        validate_collectors(["proton", "rocprof"])


# ---- wrap_argv_for_collectors: no-op path -------------------------------


def test_wrap_is_a_noop_without_any_collector():
    """A run without ``--collect`` must be byte-for-byte what it was."""
    assert wrap_argv_for_collectors({}, _INNER) == _INNER


def test_wrap_is_a_noop_for_validated_only_collectors(tmp_path):
    config = _config(["layer_numerics"], collect_dir=tmp_path)
    assert wrap_argv_for_collectors(config, _INNER) == _INNER


def test_wrap_returns_a_new_list(tmp_path):
    inner = list(_INNER)
    assert wrap_argv_for_collectors({}, inner) is not inner


def test_wrap_skips_when_no_collect_dir_was_threaded(profilers_on_path, caplog):
    """The dispatcher only injects the collect dir on the artifact-writing rank;
    a non-writing rank has nowhere to put artifacts, so it runs unprofiled
    rather than scattering files into the cwd."""
    with caplog.at_level("WARNING"):
        assert wrap_argv_for_collectors(_config(["rocprof"]), _INNER) == _INNER
    assert CONFIG_KEY_COLLECT_DIR in caplog.text


def test_wrap_fails_when_the_output_dir_cannot_be_created(profilers_on_path, tmp_path):
    """An artifact directory that cannot be prepared is a setup failure.

    The collector would have nowhere to write, so letting the trial proceed
    would run it unprofiled while the operator believes it was measured --
    the same contract as a missing rocprofv3.
    """
    blocker = tmp_path / "blocked"
    blocker.write_text("")  # a file where the collector wants a directory
    config = _config(["rocprof"], collect_dir=blocker)
    with pytest.raises(RuntimeError, match="cannot prepare the rocprof artifact directory"):
        wrap_argv_for_collectors(config, _INNER)


def test_wrap_refuses_a_symlink_in_any_component_below_the_trusted_root(
    profilers_on_path, tmp_path
):
    """A symlink *anywhere* at or below the trusted root redirects the delete.

    Checking only the leaf and its parent is not enough: ``is_dir()`` and
    ``rmtree()`` follow links in every component, so a link further up -- but
    still inside the payload-writable run tree -- reaches outside just as well.
    Here ``<results>/linked -> <outside>`` with the collector root two levels
    below it, and neither the leaf nor its parent is itself a symlink.
    """
    results = tmp_path / "results"
    results.mkdir()
    outside = tmp_path / "outside"
    (outside / "trial_d0_m0_t0" / "rocprof").mkdir(parents=True)
    victim = outside / "trial_d0_m0_t0" / "rocprof" / "precious.txt"
    victim.write_text("not yours to delete", encoding="utf-8")
    (results / "linked").symlink_to(outside, target_is_directory=True)

    collect_dir = results / "linked" / "trial_d0_m0_t0"
    assert not collect_dir.is_symlink()  # the gap: nothing local looks wrong

    with pytest.raises(RuntimeError, match="cannot prepare the rocprof artifact directory"):
        wrap_argv_for_collectors(
            _config(["rocprof"], collect_dir=collect_dir, results_root=results.resolve()), _INNER
        )
    assert victim.read_text(encoding="utf-8") == "not yours to delete"


def test_wrap_allows_a_results_dir_that_legitimately_lives_under_a_symlink(
    profilers_on_path, tmp_path
):
    """The operator's own layout above the trusted root must keep working.

    A ``--results-dir`` under a symlink (a mounted scratch path, a symlinked
    home) is ordinary. The dispatcher canonicalizes that path *before* launch
    and threads the resolved prefix as both the trust anchor and the collect
    dir, so the payload-symlink walk never sees the operator's link.
    """
    real = tmp_path / "real_results"
    real.mkdir()
    link = tmp_path / "results_link"
    link.symlink_to(real, target_is_directory=True)

    argv = wrap_argv_for_collectors(
        _config(["rocprof"], collect_dir=link.resolve() / "trial_d0_m0_t0", results_root=link.resolve()),
        _INNER,
    )
    assert argv != _INNER  # the collector attached rather than being refused
    assert (real / "trial_d0_m0_t0" / "rocprof").is_dir()


def test_wrap_refuses_to_clear_through_a_symlinked_collect_root(profilers_on_path, tmp_path):
    """A symlinked collector root must not let ``rmtree`` escape the run tree.

    ``Path.is_dir()`` follows symlinks in *every* component, so a per-trial
    collector root that is a symlink would resolve the output directory through
    it and delete the link target -- a tree outside the results directory
    entirely. This is the destructive case, so it fails closed.
    """
    # ``precious`` sits outside the results directory, which is the boundary
    # the guard is defending -- not merely outside the trial directory.
    results = tmp_path / "results"
    results.mkdir()
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "rocprof").mkdir()
    (outside / "rocprof" / "keep_me.txt").write_text("not yours to delete", encoding="utf-8")

    link = results / "trial_d0_m0_t0"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="cannot prepare the rocprof artifact directory"):
        wrap_argv_for_collectors(
            _config(["rocprof"], collect_dir=link, results_root=results.resolve()), _INNER
        )
    # The whole point: the tree behind the symlink is untouched.
    assert (outside / "rocprof" / "keep_me.txt").read_text(encoding="utf-8") == (
        "not yours to delete"
    )


def test_wrap_refuses_a_symlink_to_a_sibling_inside_the_results_tree(
    profilers_on_path, tmp_path
):
    """Containment alone is not enough: a link to a sibling still resolves inside.

    ``trial -> <results>/other_trial`` (or ``rocprof -> <results>``) stays
    inside the trusted root after ``resolve()``, so a containment-only guard
    would let ``rmtree`` erase the sibling. The payload-symlink walk refuses
    any link at or below the anchor even when the target is in-tree.
    """
    results = tmp_path / "results"
    results.mkdir()
    sibling = results / "trial_other"
    (sibling / "rocprof").mkdir(parents=True)
    victim = sibling / "rocprof" / "keep_me.txt"
    victim.write_text("sibling trial", encoding="utf-8")

    collect_dir = results / "trial_d0_m0_t0"
    collect_dir.symlink_to(sibling, target_is_directory=True)

    with pytest.raises(RuntimeError, match="cannot prepare the rocprof artifact directory"):
        wrap_argv_for_collectors(
            _config(["rocprof"], collect_dir=collect_dir, results_root=results.resolve()),
            _INNER,
        )
    assert victim.read_text(encoding="utf-8") == "sibling trial"


def test_wrap_refuses_a_collector_subdir_symlinked_at_the_results_root(
    profilers_on_path, tmp_path
):
    """``<trial>/rocprof -> <results>`` would let ``rmtree`` wipe the whole tree."""
    results = tmp_path / "results"
    collect_dir = results / "trial_d0_m0_t0"
    collect_dir.mkdir(parents=True)
    (collect_dir / "rocprof").symlink_to(results, target_is_directory=True)
    marker = results / "keep_me.txt"
    marker.write_text("results tree", encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot prepare the rocprof artifact directory"):
        wrap_argv_for_collectors(
            _config(["rocprof"], collect_dir=collect_dir, results_root=results.resolve()),
            _INNER,
        )
    assert marker.read_text(encoding="utf-8") == "results tree"


# ---- wrap_argv_for_collectors: attaching -------------------------------


def test_wrap_creates_the_output_subdir(profilers_on_path, tmp_path):
    """Pre-created so the trial tree has the same shape whether or not the
    workload produced GPU activity -- rocprofv3 writes nothing when it did not."""
    wrap_argv_for_collectors(_config(["rocprof"], collect_dir=tmp_path), _INNER)
    assert (tmp_path / rocprof.OUTPUT_SUBDIR).is_dir()


def test_wrap_rocprof_points_at_its_own_subdir(profilers_on_path, tmp_path):
    argv = wrap_argv_for_collectors(_config(["rocprof"], collect_dir=tmp_path), _INNER)
    assert argv[argv.index("-d") + 1] == str(tmp_path / rocprof.OUTPUT_SUBDIR)
    assert argv[-len(_INNER) :] == _INNER


def test_wrap_proton_points_at_its_own_subdir(profilers_on_path, tmp_path):
    argv = wrap_argv_for_collectors(_config(["proton"], collect_dir=tmp_path), _INNER)
    expected = tmp_path / proton.OUTPUT_SUBDIR / proton.PROFILE_BASENAME
    assert argv[argv.index("-n") + 1] == str(expected)


def test_wrap_forwards_per_collector_options(profilers_on_path, tmp_path):
    config = _config(
        ["rocprof"],
        collect_dir=tmp_path,
        options={"rocprof": {"trace": "kernel,hip", "summary_units": "msec"}},
    )
    argv = wrap_argv_for_collectors(config, _INNER)
    assert "--hip-trace" in argv
    assert argv[argv.index("-u") + 1] == "msec"


def test_wrap_ignores_options_for_other_collectors(profilers_on_path, tmp_path):
    config = _config(
        ["rocprof"],
        collect_dir=tmp_path,
        options={"proton": {"backend": "roctracer"}},
    )
    assert "--hip-trace" not in wrap_argv_for_collectors(config, _INNER)


def test_wrap_tolerates_a_malformed_options_payload(profilers_on_path, tmp_path):
    config = _config(["rocprof"], collect_dir=tmp_path, options=["rocprof"])
    assert wrap_argv_for_collectors(config, _INNER)[-len(_INNER) :] == _INNER


def test_wrap_coerces_non_string_option_values(profilers_on_path, tmp_path):
    config = _config(["rocprof"], collect_dir=tmp_path, options={"rocprof": {"stats": 1}})
    assert "--stats" in wrap_argv_for_collectors(config, _INNER)


def test_wrap_propagates_an_invalid_option(profilers_on_path, tmp_path):
    config = _config(["rocprof"], collect_dir=tmp_path, options={"rocprof": {"trace": "nope"}})
    with pytest.raises(ValueError, match="unknown domain"):
        wrap_argv_for_collectors(config, _INNER)


def test_wrap_propagates_an_unattachable_collector(tmp_path, monkeypatch):
    """Requesting a measurement that cannot be taken is a clean setup failure,
    not a silently unprofiled run."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv(rocprof.ENV_ROCPROF_BIN, raising=False)
    with pytest.raises(rocprof.RocprofUnavailableError):
        wrap_argv_for_collectors(_config(["rocprof"], collect_dir=tmp_path), _INNER)


def test_wrap_forwards_env_to_proton(profilers_on_path, tmp_path):
    config = _config(["proton"], collect_dir=tmp_path, options={"proton": {"backend": "roctracer"}})
    argv = wrap_argv_for_collectors(config, _INNER, env={"HIP_VISIBLE_DEVICES": "1"})
    assert "ROCR_VISIBLE_DEVICES=1" in argv


# ---- Nesting order ------------------------------------------------------


def test_wrap_nests_rocprof_outside_proton(profilers_on_path, tmp_path):
    config = _config(
        ["rocprof", "proton"],
        collect_dir=tmp_path,
        options={"proton": {"backend": "instrumentation"}},
    )
    argv = wrap_argv_for_collectors(config, _INNER)
    assert Path(argv[0]).name == "rocprofv3"
    sep = argv.index("--")
    inner = argv[sep + 1 :]
    assert inner[1:3] == ["-m", proton.PROTON_MODULE]
    assert inner[-len(_INNER) + 1 :] == _INNER[1:]


def test_wrap_nesting_is_independent_of_the_request_order(profilers_on_path, tmp_path):
    options = {"proton": {"backend": "instrumentation"}}
    forward = wrap_argv_for_collectors(
        _config(["rocprof", "proton"], collect_dir=tmp_path, options=options), _INNER
    )
    reverse = wrap_argv_for_collectors(
        _config(["proton", "rocprof"], collect_dir=tmp_path, options=options), _INNER
    )
    assert forward == reverse


def test_emulator_wraps_outside_the_collectors(profilers_on_path, tmp_path):
    """The caller applies the collector wrap first and the mirage wrap after,
    so the profiler runs *inside* the emulated environment. The other order
    would profile the emulator's own launcher instead of the workload."""
    collected = wrap_argv_for_collectors(_config(["rocprof"], collect_dir=tmp_path), _INNER)
    emulated = wrap_argv_for_environment(
        {
            CONFIG_KEY_ENVIRONMENT: asdict(
                Environment(name="emu", emulator="rocjitsu", mirage_profile="mi350x")
            )
        },
        collected,
    )
    assert Path(emulated[0]).name == "mirage"
    assert emulated[1:4] == ["run", "--profile", "mi350x"]
    assert Path(emulated[emulated.index("--") + 1]).name == "rocprofv3"


# ---- summarize_collectors -----------------------------------------------


def test_summarize_is_empty_without_a_collect_dir():
    assert summarize_collectors(_config(["rocprof"])) == {}


def test_summarize_is_empty_without_any_collector(tmp_path):
    assert summarize_collectors({CONFIG_KEY_COLLECT_DIR: str(tmp_path)}) == {}


def test_summarize_skips_validated_only_collectors(tmp_path):
    config = _config(["layer_numerics"], collect_dir=tmp_path)
    assert summarize_collectors(config) == {}


def test_summarize_reports_the_artifact_dir_for_an_empty_capture(tmp_path):
    (tmp_path / rocprof.OUTPUT_SUBDIR).mkdir()
    config = _config(["rocprof"], collect_dir=tmp_path)
    assert summarize_collectors(config) == {
        "rocprof_artifact_dir": str(tmp_path / rocprof.OUTPUT_SUBDIR)
    }


def test_summarize_parses_real_rocprof_artifacts(tmp_path):
    out_dir = tmp_path / rocprof.OUTPUT_SUBDIR
    out_dir.mkdir()
    (out_dir / "aorta_kernel_stats.csv").write_text(
        '"Name","Calls","TotalDurationNs"\n"sgemm",23,539404\n', encoding="utf-8"
    )
    metrics = summarize_collectors(_config(["rocprof"], collect_dir=tmp_path))
    assert metrics["rocprof_kernel_count"] == 23
    assert metrics["rocprof_gpu_time_ms"] == pytest.approx(0.539404)


def test_summarize_merges_both_collectors(tmp_path):
    (tmp_path / rocprof.OUTPUT_SUBDIR).mkdir()
    (tmp_path / proton.OUTPUT_SUBDIR).mkdir()
    config = _config(["rocprof", "proton"], collect_dir=tmp_path)
    metrics = summarize_collectors(config)
    assert "rocprof_artifact_dir" in metrics
    assert "proton_artifact_dir" in metrics


def test_summarize_metric_keys_are_namespaced_by_collector(tmp_path):
    """Both collectors write into the same flat ``metrics`` mapping, so their
    keys must not collide."""
    (tmp_path / rocprof.OUTPUT_SUBDIR).mkdir()
    (tmp_path / proton.OUTPUT_SUBDIR).mkdir()
    metrics = summarize_collectors(_config(["rocprof", "proton"], collect_dir=tmp_path))
    assert all(key.startswith(("rocprof_", "proton_")) for key in metrics)


def test_summarize_survives_a_parser_that_raises(tmp_path, monkeypatch, caplog):
    """An opt-in measurement must never turn a healthy trial into a failure."""

    def boom(_out_dir):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(rocprof, "parse_summary", boom)
    (tmp_path / rocprof.OUTPUT_SUBDIR).mkdir()
    with caplog.at_level("WARNING"):
        assert summarize_collectors(_config(["rocprof"], collect_dir=tmp_path)) == {}
    assert "summary parsing failed" in caplog.text


def test_summarize_does_not_re_resolve_a_results_dir_swapped_for_a_symlink(tmp_path, caplog):
    """The saved trust anchor must not be resolved again after the payload runs.

    If the profiled process replaces the results directory with a symlink to
    ``/outside``, resolving *both* the candidate and the anchor through that
    link would make containment succeed and parse planted metrics. The
    dispatcher stores a pre-launch ``resolve()``; this test keeps that string
    and swaps the directory out from under it.
    """
    results = tmp_path / "results"
    trial = results / "trial_d0_m0_t0"
    (trial / rocprof.OUTPUT_SUBDIR).mkdir(parents=True)
    trusted = str(results.resolve())
    collect = str(trial)

    outside = tmp_path / "outside"
    planted = outside / "trial_d0_m0_t0" / rocprof.OUTPUT_SUBDIR
    planted.mkdir(parents=True)
    (planted / "aorta_kernel_stats.csv").write_text(
        '"Name","Calls","TotalDurationNs"\n"sgemm",7,1000\n', encoding="utf-8"
    )

    results.rename(tmp_path / "results_orig")
    (tmp_path / "results").symlink_to(outside, target_is_directory=True)

    with caplog.at_level("WARNING"):
        metrics = summarize_collectors(
            _config(["rocprof"], collect_dir=collect, results_root=trusted)
        )
    assert metrics == {}
    assert "rocprof_kernel_count" not in metrics
    assert "symlink" in caplog.text
