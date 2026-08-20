"""Tests for the ``rocprof`` collector package (no GPU, no rocprofv3 needed).

Covers the platform-side contract: the option schema, the ``rocprofv3 ... --``
argv prefix, binary resolution, and fail-soft summary parsing.

The parser fixtures under ``fixtures/rocprof/`` are real ``rocprofv3`` 1.0.1
output captured on a gfx950 host, because the layout depends on a flag:

* ``flat_with_o/`` -- what ``-d <dir> -o <stem>`` produces, flat in ``<dir>``
  as ``<stem>_kernel_stats.csv`` / ``_kernel_trace.csv`` / ``_domain_stats.csv``.
* ``nested_no_o/`` -- what the same run WITHOUT ``-o`` produces: a
  ``<hostname>/`` directory holding ``<pid>_kernel_stats.csv`` and friends.

Both must parse, because an operator pointing the parser at their own
``rocprofv3 -d`` tree will have whichever one they ran.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aorta.instrumentation.rocprof import (
    ENV_ROCPROF_BIN,
    OPTION_KEYS,
    OUTPUT_BASENAME,
    OUTPUT_SUBDIR,
    SUMMARY_FILE_STEM,
    SUMMARY_FILENAME,
    RocprofUnavailableError,
    build_argv_prefix,
    parse_summary,
    resolve_binary,
    validate_options,
    wrap_argv,
)
from aorta.run.collectors import KNOWN_RECIPES

FIXTURES = Path(__file__).parent / "fixtures" / "rocprof"

# Totals of the real capture checked in under fixtures/rocprof/flat_with_o.
_FLAT_KERNEL = "sgemm_tiled(float const*, float const*, float*, int)"
_FLAT_CALLS = 23
_FLAT_TOTAL_NS = 539404.0
# The trace fixture is the first 5 dispatch rows of the same capture; summing
# their (End - Start) spans is what the stats-less fallback must produce.
_TRACE_ROWS = 5
_TRACE_TOTAL_NS = float(
    (1434937317879637 - 1434937317843317)
    + (1434937317902677 - 1434937317879637)
    + (1434937317925477 - 1434937317902677)
    + (1434937317983997 - 1434937317960637)
    + (1434937318006638 - 1434937317983997)
)


@pytest.fixture
def rocprofv3_on_path(tmp_path, monkeypatch):
    """Put a fake executable ``rocprofv3`` on ``$PATH`` so resolution succeeds."""
    fake = tmp_path / "rocprofv3"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv(ENV_ROCPROF_BIN, raising=False)
    return fake


# ---- Registration + package surface --------------------------------------


def test_registered_as_known_recipe():
    assert "rocprof" in KNOWN_RECIPES


def test_output_subdir_is_rocprof():
    assert OUTPUT_SUBDIR == "rocprof"


def test_summary_filename_is_what_rocprofv3_actually_writes():
    """rocprofv3 derives the summary name from ``-o`` + the stem + ``.txt``.

    ``--summary-output-file`` takes a stem relative to ``-d``, not a path, so
    the file that appears is ``<basename>_<stem>.txt``. Verified on hardware:
    ``-o aorta --summary-output-file rocprof_summary`` produces
    ``aorta_rocprof_summary.txt``.
    """
    assert SUMMARY_FILENAME == f"{OUTPUT_BASENAME}_{SUMMARY_FILE_STEM}.txt"
    assert not Path(SUMMARY_FILE_STEM).is_absolute()
    assert "/" not in SUMMARY_FILE_STEM


# ---- Option validation ---------------------------------------------------


def test_validate_options_defaults():
    effective = validate_options(None)
    assert effective["trace"] == "kernel"
    assert effective["output_format"] == "csv"
    assert effective["stats"] == "true"
    # Optional knobs stay absent so the argv omits their flags entirely.
    assert "pmc" not in effective
    assert "summary_units" not in effective
    assert "kernel_include_regex" not in effective


def test_validate_options_empty_mapping_matches_none():
    assert validate_options({}) == validate_options(None)


def test_validate_options_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown option"):
        validate_options({"traces": "kernel"})


def test_validate_options_unknown_key_message_names_valid_keys():
    with pytest.raises(ValueError, match="kernel_include_regex"):
        validate_options({"nope": "1"})


def test_validate_options_rejects_non_string_value():
    with pytest.raises(ValueError, match="must be a string"):
        validate_options({"stats": True})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    "domain", sorted(["kernel", "hip", "hip_runtime", "memory_copy", "rccl", "marker", "scratch"])
)
def test_validate_options_accepts_every_trace_domain(domain):
    assert validate_options({"trace": domain})["trace"] == domain


def test_validate_options_trace_accepts_comma_and_space_lists():
    assert validate_options({"trace": "kernel,hip"})["trace"] == "kernel,hip"
    assert validate_options({"trace": "kernel hip"})["trace"] == "kernel,hip"


def test_validate_options_trace_dedups_preserving_order():
    assert validate_options({"trace": "hip,kernel,hip"})["trace"] == "hip,kernel"


def test_validate_options_trace_normalises_case():
    assert validate_options({"trace": "KERNEL"})["trace"] == "kernel"


def test_validate_options_rejects_unknown_trace_domain():
    with pytest.raises(ValueError, match="unknown domain"):
        validate_options({"trace": "kernel,gpu"})


def test_validate_options_rejects_empty_trace():
    with pytest.raises(ValueError, match="at least one domain"):
        validate_options({"trace": " , "})


@pytest.mark.parametrize("fmt", ["csv", "json", "pftrace", "otf2", "rocpd"])
def test_validate_options_accepts_every_output_format(fmt):
    assert validate_options({"output_format": fmt})["output_format"] == fmt


def test_validate_options_rejects_unknown_output_format():
    with pytest.raises(ValueError, match="output_format"):
        validate_options({"output_format": "parquet"})


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "0", "false", "no", "off"])
def test_validate_options_accepts_boolean_spellings(value):
    assert validate_options({"stats": value})["stats"] == value


def test_validate_options_rejects_non_boolean_stats():
    with pytest.raises(ValueError, match="expected a boolean"):
        validate_options({"stats": "maybe"})


@pytest.mark.parametrize("unit", ["sec", "msec", "usec", "nsec"])
def test_validate_options_accepts_every_summary_unit(unit):
    assert validate_options({"summary_units": unit})["summary_units"] == unit


def test_validate_options_rejects_unknown_summary_unit():
    with pytest.raises(ValueError, match="summary_units"):
        validate_options({"summary_units": "minutes"})


def test_validate_options_rejects_empty_kernel_include_regex():
    with pytest.raises(ValueError, match="kernel_include_regex"):
        validate_options({"kernel_include_regex": "   "})


def test_validate_options_keeps_regex_verbatim():
    """A free-form value must not be case-folded -- a regex is case-sensitive."""
    assert validate_options({"kernel_include_regex": "Cijk|GEMM"})["kernel_include_regex"] == (
        "Cijk|GEMM"
    )


def test_validate_options_rejects_empty_pmc():
    with pytest.raises(ValueError, match="at least one counter"):
        validate_options({"pmc": ","})


def test_validate_options_does_not_mutate_input():
    supplied = {"trace": "KERNEL"}
    validate_options(supplied)
    assert supplied == {"trace": "KERNEL"}


# ---- Argv construction ---------------------------------------------------


def test_build_argv_prefix_shape(rocprofv3_on_path, tmp_path):
    argv = build_argv_prefix(tmp_path / "out")
    assert argv[0] == str(rocprofv3_on_path)
    assert argv[-1] == "--"
    assert argv[1:3] == ["-d", str(tmp_path / "out")]
    assert argv[3:5] == ["-o", OUTPUT_BASENAME]
    assert "--kernel-trace" in argv
    assert "--stats" in argv
    assert ["--output-format", "csv"] == argv[5:7]


def test_build_argv_prefix_summary_goes_to_a_file_not_stderr(rocprofv3_on_path, tmp_path):
    """``-S`` alone prints to stderr, which the probe classifier would read."""
    argv = build_argv_prefix(tmp_path / "out")
    assert "-S" in argv
    assert argv[argv.index("--summary-output-file") + 1] == SUMMARY_FILE_STEM


def test_build_argv_prefix_multi_domain_trace(rocprofv3_on_path, tmp_path):
    argv = build_argv_prefix(tmp_path, {"trace": "kernel,hip,memory_copy"})
    assert "--kernel-trace" in argv
    assert "--hip-trace" in argv
    assert "--memory-copy-trace" in argv


def test_build_argv_prefix_stats_off_omits_flag(rocprofv3_on_path, tmp_path):
    assert "--stats" not in build_argv_prefix(tmp_path, {"stats": "false"})


def test_build_argv_prefix_optional_flags_omitted_by_default(rocprofv3_on_path, tmp_path):
    argv = build_argv_prefix(tmp_path)
    assert "--pmc" not in argv
    assert "-u" not in argv
    assert "--kernel-include-regex" not in argv


def test_build_argv_prefix_carries_optional_flags(rocprofv3_on_path, tmp_path):
    argv = build_argv_prefix(
        tmp_path,
        {"summary_units": "msec", "kernel_include_regex": "Cijk|gemm"},
    )
    assert argv[argv.index("-u") + 1] == "msec"
    assert argv[argv.index("--kernel-include-regex") + 1] == "Cijk|gemm"


def test_build_argv_prefix_pmc_is_last_flag_before_separator(rocprofv3_on_path, tmp_path):
    """``--pmc`` is variadic, so anything after it would be eaten as a counter."""
    argv = build_argv_prefix(tmp_path, {"pmc": "SQ_WAVES,GRBM_COUNT"})
    pmc = argv.index("--pmc")
    assert argv[pmc + 1 : -1] == ["SQ_WAVES", "GRBM_COUNT"]
    assert argv[-1] == "--"


def test_build_argv_prefix_accepts_str_out_dir(rocprofv3_on_path):
    argv = build_argv_prefix("relative/out")
    assert argv[argv.index("-d") + 1] == str(Path("relative/out"))


def test_build_argv_prefix_propagates_option_error(rocprofv3_on_path, tmp_path):
    with pytest.raises(ValueError, match="unknown option"):
        build_argv_prefix(tmp_path, {"nope": "1"})


def test_wrap_argv_puts_command_after_the_separator(rocprofv3_on_path, tmp_path):
    argv = wrap_argv(["/tmp/gemm", "512", "20"], tmp_path)
    sep = argv.index("--")
    assert argv[sep + 1 :] == ["/tmp/gemm", "512", "20"]


def test_wrap_argv_does_not_mutate_the_input(rocprofv3_on_path, tmp_path):
    inner = ["/tmp/gemm", "512"]
    wrap_argv(inner, tmp_path)
    assert inner == ["/tmp/gemm", "512"]


def test_wrap_argv_ignores_env(rocprofv3_on_path, tmp_path):
    """rocprofv3 is flag-configured; the ``env`` kwarg exists for the registry's
    uniform signature (Proton needs it) and must not change the argv."""
    with_env = wrap_argv(["/tmp/gemm"], tmp_path, env={"HIP_VISIBLE_DEVICES": "1"})
    assert with_env == wrap_argv(["/tmp/gemm"], tmp_path)


# ---- Binary resolution ---------------------------------------------------


def test_resolve_binary_finds_path_entry(rocprofv3_on_path):
    assert resolve_binary() == str(rocprofv3_on_path)


def test_resolve_binary_honours_env_override_path(tmp_path, monkeypatch):
    custom = tmp_path / "my-rocprof"
    custom.write_text("#!/bin/sh\nexit 0\n")
    custom.chmod(0o755)
    monkeypatch.setenv(ENV_ROCPROF_BIN, str(custom))
    assert resolve_binary() == str(custom)


def test_resolve_binary_expands_user_in_override(tmp_path, monkeypatch):
    custom = tmp_path / "my-rocprof"
    custom.write_text("#!/bin/sh\nexit 0\n")
    custom.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(ENV_ROCPROF_BIN, "~/my-rocprof")
    assert resolve_binary() == str(custom)


def test_resolve_binary_override_bare_name_searched_on_path(tmp_path, monkeypatch):
    fake = tmp_path / "rocprof-custom"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv(ENV_ROCPROF_BIN, "rocprof-custom")
    assert resolve_binary() == str(fake)


def test_resolve_binary_rejects_non_executable_override(tmp_path, monkeypatch):
    dud = tmp_path / "not-executable"
    dud.write_text("")
    monkeypatch.setenv(ENV_ROCPROF_BIN, str(dud))
    with pytest.raises(RocprofUnavailableError, match="does not"):
        resolve_binary()


def test_resolve_binary_rejects_missing_override_on_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv(ENV_ROCPROF_BIN, "definitely-not-a-profiler-9f8d7")
    with pytest.raises(RocprofUnavailableError, match="was not found"):
        resolve_binary()


def test_resolve_binary_missing_names_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv(ENV_ROCPROF_BIN, raising=False)
    with pytest.raises(RocprofUnavailableError, match=ENV_ROCPROF_BIN):
        resolve_binary()


def test_wrap_argv_raises_when_profiler_missing(tmp_path, monkeypatch):
    """A requested-but-unattachable collector is a clean setup failure."""
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv(ENV_ROCPROF_BIN, raising=False)
    with pytest.raises(RocprofUnavailableError):
        wrap_argv(["/tmp/gemm"], tmp_path)


# ---- Summary parsing: the two real layouts -------------------------------


def test_parse_summary_flat_layout_from_dash_o():
    metrics = parse_summary(FIXTURES / "flat_with_o")
    assert metrics["rocprof_kernel_count"] == _FLAT_CALLS
    assert metrics["rocprof_gpu_time_ms"] == pytest.approx(_FLAT_TOTAL_NS / 1e6)
    assert metrics["rocprof_top_kernel_ms"] == pytest.approx(_FLAT_TOTAL_NS / 1e6)
    assert metrics["rocprof_top_kernels"] == [_FLAT_KERNEL]
    assert metrics["rocprof_artifact_dir"] == str(FIXTURES / "flat_with_o")


def test_flat_fixture_contains_the_summary_file_the_package_names():
    """The fixture is a hardware capture (normalised only by the repo's
    end-of-file hook, which drops rocprofv3's trailing blank line), so it also
    pins the summary filename: ``-o aorta --summary-output-file rocprof_summary``
    really does produce ``aorta_rocprof_summary.txt``, flat beside the CSVs."""
    summary = FIXTURES / "flat_with_o" / SUMMARY_FILENAME
    assert summary.is_file()
    assert "ROCPROFV3 SUMMARY" in summary.read_text(encoding="utf-8")
    # No stray directories: a spliced absolute path would have created some.
    assert not [path for path in (FIXTURES / "flat_with_o").iterdir() if path.is_dir()]


def test_parse_summary_nested_layout_without_dash_o():
    """Without ``-o``, rocprofv3 nests under ``<hostname>/<pid>_*.csv``."""
    metrics = parse_summary(FIXTURES / "nested_no_o")
    assert metrics["rocprof_kernel_count"] == 8
    assert metrics["rocprof_gpu_time_ms"] == pytest.approx(81720 / 1e6)
    assert metrics["rocprof_top_kernels"] == [_FLAT_KERNEL]


def test_parse_summary_emits_only_expected_keys():
    assert set(parse_summary(FIXTURES / "flat_with_o")) == {
        "rocprof_artifact_dir",
        "rocprof_kernel_count",
        "rocprof_gpu_time_ms",
        "rocprof_top_kernel_ms",
        "rocprof_top_kernels",
    }


def test_parse_summary_numeric_metrics_are_plain_numbers():
    """perf.md's metrics table only picks up ``int`` / ``float`` values."""
    metrics = parse_summary(FIXTURES / "flat_with_o")
    for key in ("rocprof_kernel_count", "rocprof_gpu_time_ms", "rocprof_top_kernel_ms"):
        assert isinstance(metrics[key], (int, float))
        assert not isinstance(metrics[key], bool)


def test_parse_summary_falls_back_to_kernel_trace(tmp_path):
    """With ``stats`` off there is no ``_kernel_stats.csv``; sum trace spans."""
    trace = FIXTURES / "flat_with_o" / "aorta_kernel_trace.csv"
    (tmp_path / "aorta_kernel_trace.csv").write_bytes(trace.read_bytes())
    metrics = parse_summary(tmp_path)
    assert metrics["rocprof_kernel_count"] == _TRACE_ROWS
    assert metrics["rocprof_gpu_time_ms"] == pytest.approx(_TRACE_TOTAL_NS / 1e6)


def test_parse_summary_prefers_stats_over_trace():
    """Both files are present in the real capture; the stats aggregate wins."""
    metrics = parse_summary(FIXTURES / "flat_with_o")
    assert metrics["rocprof_kernel_count"] == _FLAT_CALLS
    assert metrics["rocprof_kernel_count"] != _TRACE_ROWS


# ---- Summary parsing: fail-soft ------------------------------------------


def test_parse_summary_missing_dir_is_empty(tmp_path):
    assert parse_summary(tmp_path / "never-created") == {}


def test_parse_summary_file_instead_of_dir_is_empty(tmp_path):
    path = tmp_path / "not-a-dir"
    path.write_text("")
    assert parse_summary(path) == {}


def test_parse_summary_empty_dir_reports_only_the_artifact_dir(tmp_path):
    """Verified on hardware: a command with no GPU work makes rocprofv3 write
    NOTHING at all. That is a legitimate outcome, not a failure."""
    assert parse_summary(tmp_path) == {"rocprof_artifact_dir": str(tmp_path)}


def test_parse_summary_malformed_csv_degrades_without_raising():
    metrics = parse_summary(FIXTURES / "malformed")
    assert metrics == {"rocprof_artifact_dir": str(FIXTURES / "malformed")}


def test_parse_summary_header_only_csv_degrades(tmp_path):
    (tmp_path / "aorta_kernel_stats.csv").write_text(
        '"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"\n',
        encoding="utf-8",
    )
    assert parse_summary(tmp_path) == {"rocprof_artifact_dir": str(tmp_path)}


def test_parse_summary_skips_rows_with_unparseable_duration(tmp_path):
    (tmp_path / "aorta_kernel_stats.csv").write_text(
        '"Name","Calls","TotalDurationNs"\n'
        '"good",2,1000000\n'
        '"bad",1,"not-a-number"\n'
        ",3,5000\n",
        encoding="utf-8",
    )
    metrics = parse_summary(tmp_path)
    assert metrics["rocprof_top_kernels"] == ["good"]
    assert metrics["rocprof_gpu_time_ms"] == pytest.approx(1.0)
    assert metrics["rocprof_kernel_count"] == 2


def test_parse_summary_counts_one_dispatch_when_calls_is_unreadable(tmp_path):
    """A stats row without a readable ``Calls`` column still evidences one
    dispatch, so it counts as one rather than being dropped."""
    (tmp_path / "aorta_kernel_stats.csv").write_text(
        '"Name","TotalDurationNs"\n"k",1000000\n', encoding="utf-8"
    )
    assert parse_summary(tmp_path)["rocprof_kernel_count"] == 1


def test_parse_summary_takes_a_zero_call_count_at_its_word(tmp_path):
    """``rocprof_kernel_count`` claims dispatches, so a readable zero must not
    be rounded up to one -- that would over-count the metric its name
    promises."""
    (tmp_path / "aorta_kernel_stats.csv").write_text(
        '"Name","Calls","TotalDurationNs"\n"idle",0,1000\n"real",2,2000\n',
        encoding="utf-8",
    )
    assert parse_summary(tmp_path)["rocprof_kernel_count"] == 2


def test_parse_summary_trace_skips_non_dispatch_and_inverted_rows(tmp_path):
    (tmp_path / "aorta_kernel_trace.csv").write_text(
        '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"\n'
        '"KERNEL_DISPATCH","k",100,1100\n'
        '"MEMORY_COPY","memcpy",0,9999999\n'
        '"KERNEL_DISPATCH","backwards",500,100\n',
        encoding="utf-8",
    )
    metrics = parse_summary(tmp_path)
    assert metrics["rocprof_top_kernels"] == ["k"]
    assert metrics["rocprof_kernel_count"] == 1


def test_parse_summary_aggregates_multiple_stats_files(tmp_path):
    """A multi-process capture writes one stats CSV per rank directory."""
    for idx, total in ((0, 1000000), (1, 3000000)):
        rank_dir = tmp_path / f"host{idx}"
        rank_dir.mkdir()
        (rank_dir / f"{idx}_kernel_stats.csv").write_text(
            '"Name","Calls","TotalDurationNs"\n' f'"shared",1,{total}\n',
            encoding="utf-8",
        )
    metrics = parse_summary(tmp_path)
    assert metrics["rocprof_gpu_time_ms"] == pytest.approx(4.0)
    assert metrics["rocprof_kernel_count"] == 2


def test_parse_summary_ranks_top_kernel_by_total_time(tmp_path):
    (tmp_path / "aorta_kernel_stats.csv").write_text(
        '"Name","Calls","TotalDurationNs"\n'
        '"small",100,1000\n'
        '"big",1,9000000\n'
        '"medium",2,4000000\n',
        encoding="utf-8",
    )
    metrics = parse_summary(tmp_path)
    assert metrics["rocprof_top_kernels"][:2] == ["big", "medium"]
    assert metrics["rocprof_top_kernel_ms"] == pytest.approx(9.0)


def test_parse_summary_caps_top_kernels(tmp_path):
    rows = "".join(f'"k{i}",1,{(20 - i) * 1000}\n' for i in range(12))
    (tmp_path / "aorta_kernel_stats.csv").write_text(
        '"Name","Calls","TotalDurationNs"\n' + rows, encoding="utf-8"
    )
    assert len(parse_summary(tmp_path)["rocprof_top_kernels"]) == 5


def test_parse_summary_survives_unreadable_file(tmp_path):
    csv_path = tmp_path / "aorta_kernel_stats.csv"
    csv_path.write_text('"Name","Calls","TotalDurationNs"\n"k",1,1000\n', encoding="utf-8")
    csv_path.chmod(0o000)
    try:
        if os.access(csv_path, os.R_OK):
            pytest.skip("running as a user that can read mode-000 files")
        assert parse_summary(tmp_path) == {"rocprof_artifact_dir": str(tmp_path)}
    finally:
        csv_path.chmod(0o644)


def test_parse_summary_accepts_str_path():
    assert parse_summary(str(FIXTURES / "flat_with_o"))["rocprof_kernel_count"] == _FLAT_CALLS


# ---- Docs / schema agreement --------------------------------------------


def test_option_keys_match_the_validator():
    """Every declared key must be accepted; the docs table is generated from
    this tuple, so a key here that the validator rejects is a doc lie."""
    samples = {
        "trace": "kernel",
        "output_format": "csv",
        "stats": "1",
        "pmc": "SQ_WAVES",
        "kernel_include_regex": "gemm",
        "summary_units": "msec",
    }
    assert set(samples) == set(OPTION_KEYS)
    assert validate_options(samples)
