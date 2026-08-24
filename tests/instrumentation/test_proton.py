"""Tests for the ``proton`` collector package (no GPU, no Triton needed).

Covers the option schema, both attach modes (``cli`` argv rewrite and ``env``
variable bundle), the ``HIP_VISIBLE_DEVICES`` -> ``ROCR_VISIBLE_DEVICES``
translation Proton needs on AMD, and fail-soft ``.hatchet`` parsing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aorta.instrumentation.proton import (
    AUTO_BACKEND,
    BACKENDS,
    ENV_PREFIX,
    ENV_PROTON_PYTHON,
    OPTION_KEYS,
    OUTPUT_SUBDIR,
    PROFILE_BASENAME,
    PROTON_MODULE,
    QUEUE_INTERCEPTING_BACKENDS,
    ProtonWrapError,
    build_argv_prefix,
    build_env,
    mode_argument,
    parse_profile,
    parse_summary,
    resolve_python,
    validate_options,
    wrap_argv,
)
from aorta.run.collectors import KNOWN_RECIPES

FIXTURES = Path(__file__).parent / "fixtures" / "proton"


@pytest.fixture
def env_on_path(tmp_path, monkeypatch):
    """Put a fake ``env(1)`` on ``$PATH`` so the device-translation prefix builds."""
    fake = tmp_path / "env"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    return fake


# ---- Registration + package surface --------------------------------------


def test_registered_as_known_recipe():
    assert "proton" in KNOWN_RECIPES


def test_output_subdir_is_proton():
    assert OUTPUT_SUBDIR == "proton"


def test_queue_intercepting_backends_are_a_subset_of_backends():
    assert QUEUE_INTERCEPTING_BACKENDS <= BACKENDS


def test_auto_counts_as_queue_intercepting():
    """``auto`` resolves to ``rocprofiler`` or ``roctracer`` on AMD, so the
    rocprof-conflict guard cannot treat it as safe."""
    assert AUTO_BACKEND in QUEUE_INTERCEPTING_BACKENDS


def test_auto_is_not_a_proton_backend_name():
    """It is aorta's spelling for "omit ``-b``", so it must never be passed
    through to Proton, whose argparse would reject it."""
    assert AUTO_BACKEND == "auto"
    assert "-b" not in build_argv_prefix("/tmp/x", {"backend": AUTO_BACKEND})


def test_rocprofiler_stays_a_valid_backend():
    """``rocprofiler`` is the documented forward path on AMD and is accepted by
    current upstream Triton. Older Triton (3.7.x) offers only
    cupti/roctracer/instrumentation, but the validator must not narrow to
    whichever Triton happens to be installed on the host doing the validating --
    a recipe is authored once and run on many hosts."""
    assert "rocprofiler" in BACKENDS
    assert validate_options({"backend": "rocprofiler"})["backend"] == "rocprofiler"


# ---- Option validation ---------------------------------------------------


def test_validate_options_defaults():
    effective = validate_options(None)
    assert effective == {
        "mode": "cli",
        "backend": AUTO_BACKEND,
        "context": "shadow",
        "data": "tree",
    }


def test_validate_options_empty_mapping_matches_none():
    assert validate_options({}) == validate_options(None)


def test_validate_options_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown option"):
        validate_options({"backends": "roctracer"})


def test_validate_options_rejects_non_string_value():
    with pytest.raises(ValueError, match="must be a string"):
        validate_options({"backend": 1})  # type: ignore[dict-item]


@pytest.mark.parametrize("mode", ["cli", "env"])
def test_validate_options_accepts_every_mode(mode):
    assert validate_options({"mode": mode})["mode"] == mode


def test_validate_options_rejects_unknown_mode():
    with pytest.raises(ValueError, match="'mode'"):
        validate_options({"mode": "library"})


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_validate_options_accepts_every_backend(backend):
    assert validate_options({"backend": backend})["backend"] == backend


def test_validate_options_rejects_unknown_backend():
    with pytest.raises(ValueError, match="'backend'"):
        validate_options({"backend": "rocprof"})


@pytest.mark.parametrize("context", ["shadow", "python"])
def test_validate_options_accepts_every_context(context):
    assert validate_options({"context": context})["context"] == context


def test_validate_options_rejects_unknown_context():
    with pytest.raises(ValueError, match="'context'"):
        validate_options({"context": "cpp"})


@pytest.mark.parametrize("data", ["tree", "trace"])
def test_validate_options_accepts_every_data_format(data):
    assert validate_options({"data": data})["data"] == data


def test_validate_options_rejects_unknown_data_format():
    with pytest.raises(ValueError, match="'data'"):
        validate_options({"data": "hatchet"})


def test_validate_options_normalises_case_and_whitespace():
    assert validate_options({"backend": " ROCTRACER "})["backend"] == "roctracer"


@pytest.mark.parametrize("name", ["default", "mma", "pcsampling"])
def test_validate_options_accepts_every_instrumentation_mode(name):
    effective = validate_options({"backend": "instrumentation", "instrumentation_mode": name})
    assert effective["instrumentation_mode"] == name


def test_validate_options_rejects_unknown_instrumentation_mode():
    with pytest.raises(ValueError, match="instrumentation_mode"):
        validate_options({"backend": "instrumentation", "instrumentation_mode": "nope"})


@pytest.mark.parametrize("granularity", ["cta", "warp", "warp_4", "warp_group_8"])
def test_validate_options_accepts_granularities(granularity):
    effective = validate_options({"backend": "instrumentation", "granularity": granularity})
    assert effective["granularity"] == granularity


def test_validate_options_rejects_unknown_granularity():
    with pytest.raises(ValueError, match="'granularity'"):
        validate_options({"backend": "instrumentation", "granularity": "thread"})


@pytest.mark.parametrize("key", ["instrumentation_mode", "granularity"])
def test_validate_options_rejects_intra_kernel_knob_on_wrong_backend(key):
    """Proton silently ignores these outside the instrumentation backend, so
    accepting them would hand back a profile the operator did not ask for."""
    with pytest.raises(ValueError, match="backend: instrumentation"):
        validate_options({"backend": "roctracer", key: "cta" if key == "granularity" else "mma"})


def test_validate_options_does_not_mutate_input():
    supplied = {"backend": "ROCTRACER"}
    validate_options(supplied)
    assert supplied == {"backend": "ROCTRACER"}


def test_option_keys_match_the_validator():
    samples = {
        "mode": "cli",
        "backend": "instrumentation",
        "context": "shadow",
        "data": "tree",
        "instrumentation_mode": "mma",
        "granularity": "warp",
    }
    assert set(samples) == set(OPTION_KEYS)
    assert validate_options(samples)


# ---- --mode rendering ----------------------------------------------------


def test_mode_argument_absent_without_intra_kernel_knobs():
    assert mode_argument(validate_options(None)) is None


def test_mode_argument_name_only():
    effective = validate_options({"backend": "instrumentation", "instrumentation_mode": "mma"})
    assert mode_argument(effective) == "mma"


def test_mode_argument_granularity_only_defaults_the_name():
    effective = validate_options({"backend": "instrumentation", "granularity": "warp"})
    assert mode_argument(effective) == "default:granularity=warp"


def test_mode_argument_name_and_granularity():
    effective = validate_options(
        {"backend": "instrumentation", "instrumentation_mode": "mma", "granularity": "cta"}
    )
    assert mode_argument(effective) == "mma:granularity=cta"


# ---- Interpreter resolution ---------------------------------------------


def test_resolve_python_prefers_the_workloads_own_interpreter():
    assert resolve_python(["python3.11", "script.py"]) == "python3.11"


def test_resolve_python_accepts_an_absolute_interpreter():
    assert resolve_python(["/opt/venv/bin/python", "script.py"]) == "/opt/venv/bin/python"


def test_resolve_python_falls_back_to_sys_executable():
    assert resolve_python(["/tmp/gemm", "512"]) == sys.executable


def test_resolve_python_empty_argv_falls_back():
    assert resolve_python([]) == sys.executable


def test_resolve_python_env_override_wins(monkeypatch):
    monkeypatch.setenv(ENV_PROTON_PYTHON, "/rocm/bin/python")
    assert resolve_python(["python", "script.py"]) == "/rocm/bin/python"


def _console_script(tmp_path, first_line: str, name: str = "pytest"):
    """Write an executable ``pytest`` console script on ``$PATH``."""
    script = tmp_path / "scriptbin" / name
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"{first_line}\nraise SystemExit(0)\n")
    script.chmod(0o755)
    return script


def test_resolve_python_reads_the_pytest_shebang(tmp_path, monkeypatch):
    """The wrap replaces the command's interpreter, so a bare ``pytest`` must
    keep the one its console script names -- otherwise a venv's test run gets
    profiled under aorta's interpreter and a different dependency set."""
    script = _console_script(tmp_path, "#!/opt/venv/bin/python3.12")
    monkeypatch.delenv(ENV_PROTON_PYTHON, raising=False)
    monkeypatch.setenv("PATH", str(script.parent))
    assert resolve_python(["pytest", "-q"]) == "/opt/venv/bin/python3.12"


def test_resolve_python_reads_an_env_style_shebang(tmp_path, monkeypatch):
    script = _console_script(tmp_path, "#!/usr/bin/env python3")
    monkeypatch.delenv(ENV_PROTON_PYTHON, raising=False)
    monkeypatch.setenv("PATH", str(script.parent))
    assert resolve_python([str(script), "-q"]) == "python3"


def test_resolve_python_ignores_a_non_python_shebang(tmp_path, monkeypatch):
    """A shell wrapper names no interpreter we can run Proton under, so the
    fallback is honest rather than a guess at ``/bin/sh``."""
    script = _console_script(tmp_path, "#!/bin/sh")
    monkeypatch.delenv(ENV_PROTON_PYTHON, raising=False)
    monkeypatch.setenv("PATH", str(script.parent))
    assert resolve_python(["pytest"]) == sys.executable


def test_resolve_python_falls_back_when_pytest_is_not_on_path(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_PROTON_PYTHON, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert resolve_python(["pytest"]) == sys.executable


def test_resolve_python_override_beats_the_shebang(tmp_path, monkeypatch):
    script = _console_script(tmp_path, "#!/opt/venv/bin/python3.12")
    monkeypatch.setenv("PATH", str(script.parent))
    monkeypatch.setenv(ENV_PROTON_PYTHON, "/rocm/bin/python")
    assert resolve_python(["pytest"]) == "/rocm/bin/python"


# ---- CLI-mode argv construction -----------------------------------------


def test_build_argv_prefix_shape(tmp_path):
    argv = build_argv_prefix(tmp_path, python="/rocm/bin/python")
    assert argv[:3] == ["/rocm/bin/python", "-m", PROTON_MODULE]
    assert argv[argv.index("-n") + 1] == str(tmp_path / PROFILE_BASENAME)
    assert argv[argv.index("--context") + 1] == "shadow"
    assert argv[argv.index("--data") + 1] == "tree"
    assert "--mode" not in argv


def test_build_argv_prefix_omits_backend_flag_by_default(tmp_path):
    """``backend: auto`` is the absence of ``-b``, which is the only spelling
    that survives Proton's own version skew: 3.7.x's CLI lists only
    cupti/roctracer/instrumentation and exits at argparse on anything else,
    while newer Triton prefers ``rocprofiler``. Dropping the flag hands the
    choice to Proton's ``_select_backend()`` on whichever version is installed.
    """
    assert "-b" not in build_argv_prefix(tmp_path)
    assert AUTO_BACKEND not in build_argv_prefix(tmp_path)


@pytest.mark.parametrize("backend", ["rocprofiler", "roctracer", "instrumentation", "cupti"])
def test_build_argv_prefix_passes_an_explicit_backend(tmp_path, backend):
    argv = build_argv_prefix(tmp_path, {"backend": backend})
    assert argv[argv.index("-b") + 1] == backend


def test_build_argv_prefix_uses_module_not_console_script(tmp_path):
    """The ``proton`` console script is shebanged to whichever interpreter
    installed Triton, which is typically not the workload's interpreter."""
    argv = build_argv_prefix(tmp_path)
    assert "proton" not in {Path(argv[0]).name}
    assert argv[1] == "-m"


def test_build_argv_prefix_defaults_to_sys_executable(tmp_path):
    assert build_argv_prefix(tmp_path)[0] == sys.executable


def test_build_argv_prefix_includes_mode_for_intra_kernel(tmp_path):
    argv = build_argv_prefix(
        tmp_path, {"backend": "instrumentation", "instrumentation_mode": "mma"}
    )
    assert argv[argv.index("--mode") + 1] == "mma"


def test_build_argv_prefix_propagates_option_error(tmp_path):
    with pytest.raises(ValueError, match="unknown option"):
        build_argv_prefix(tmp_path, {"nope": "1"})


def test_wrap_argv_cli_rewrites_a_python_launch(tmp_path):
    argv = wrap_argv(["python", "vecadd.py", "--size", "1024"], tmp_path, env={})
    assert argv[:3] == ["python", "-m", PROTON_MODULE]
    # No ``--`` separator: Proton's front-end takes the target with REMAINDER.
    assert "--" not in argv
    assert argv[-3:] == ["vecadd.py", "--size", "1024"]


def test_wrap_argv_cli_keeps_interpreter_flags_in_front_of_dash_m(tmp_path):
    argv = wrap_argv(["python", "-u", "vecadd.py"], tmp_path, env={})
    assert argv[:4] == ["python", "-u", "-m", PROTON_MODULE]
    assert argv[-1] == "vecadd.py"


def test_wrap_argv_cli_accepts_pytest_target(tmp_path):
    """Proton's front-end documents ``proton [options] pytest ...``."""
    argv = wrap_argv(["pytest", "-k", "gemm"], tmp_path, env={})
    assert argv[1:3] == ["-m", PROTON_MODULE]
    assert argv[-3:] == ["pytest", "-k", "gemm"]


def test_wrap_argv_cli_accepts_the_dash_m_pytest_spelling(tmp_path):
    """``python -m pytest`` is the usual venv/CI spelling of ``pytest``."""
    argv = wrap_argv(["python", "-m", "pytest", "-k", "gemm"], tmp_path, env={})
    assert argv[:3] == ["python", "-m", PROTON_MODULE]
    # Normalised onto the bare target: Proton dispatches on basename ==
    # 'pytest' and runs ``pytest.main(args)``, so ``-m`` must not survive
    # into the target -- Proton's own ``-m`` is ``--mode``.
    assert argv[-3:] == ["pytest", "-k", "gemm"]


def test_dash_m_pytest_and_bare_pytest_wrap_to_the_same_target(tmp_path):
    """The finding behind this test: same target, different spelling."""
    from_module = wrap_argv(["python", "-m", "pytest", "tests/", "-q"], tmp_path, env={})
    from_bare = wrap_argv(["pytest", "tests/", "-q"], tmp_path, env={})
    target = ["pytest", "tests/", "-q"]
    assert from_module[-3:] == target
    assert from_bare[-3:] == target
    # Only argv[0] may differ: the bare spelling has no interpreter to reuse,
    # so ``resolve_python`` falls back to ``sys.executable``.
    assert from_module[1:] == from_bare[1:]


def test_wrap_argv_cli_accepts_the_attached_dash_m_spelling(tmp_path):
    argv = wrap_argv(["python", "-mpytest", "-k", "gemm"], tmp_path, env={})
    assert argv[:3] == ["python", "-m", PROTON_MODULE]
    assert argv[-3:] == ["pytest", "-k", "gemm"]


def test_wrap_argv_cli_keeps_interpreter_flags_before_a_dash_m_target(tmp_path):
    argv = wrap_argv(["python", "-u", "-m", "pytest", "-x"], tmp_path, env={})
    assert argv[:4] == ["python", "-u", "-m", PROTON_MODULE]
    assert argv[-2:] == ["pytest", "-x"]


def test_wrap_argv_cli_rejects_a_module_proton_cannot_run(tmp_path):
    """``runpy.run_path`` takes a path, so a module name has no equivalent."""
    with pytest.raises(ProtonWrapError, match="cannot wrap 'python -m torch.distributed.run'"):
        wrap_argv(["python", "-m", "torch.distributed.run", "train.py"], tmp_path, env={})


def test_module_rejection_names_the_spellings_that_do_work(tmp_path):
    with pytest.raises(ProtonWrapError) as excinfo:
        wrap_argv(["python", "-m", "http.server"], tmp_path, env={})
    message = str(excinfo.value)
    assert "runpy.run_path" in message
    assert "pytest" in message
    assert "mode: env" in message


def test_wrap_argv_cli_rejects_dash_m_with_no_module(tmp_path):
    with pytest.raises(ProtonWrapError, match="no module name"):
        wrap_argv(["python", "-m"], tmp_path, env={})


def test_unknown_interpreter_flag_error_names_the_dash_m_escape(tmp_path):
    """The supported-spellings list must stay in step with the code."""
    with pytest.raises(ProtonWrapError) as excinfo:
        wrap_argv(["python", "-c", "print(1)"], tmp_path, env={})
    assert "'-m' with ['pytest']" in str(excinfo.value)


def test_wrap_argv_cli_rejects_a_non_python_command(tmp_path):
    with pytest.raises(ProtonWrapError, match="mode: env"):
        wrap_argv(["/tmp/aorta_hip_gemm", "512"], tmp_path, env={})


def test_wrap_argv_cli_rejects_unknown_interpreter_flag(tmp_path):
    with pytest.raises(ProtonWrapError, match="interpreter option"):
        wrap_argv(["python", "-c", "print(1)"], tmp_path, env={})


def test_wrap_argv_cli_rejects_interpreter_with_no_script(tmp_path):
    with pytest.raises(ProtonWrapError, match="needs a script path"):
        wrap_argv(["python", "-u"], tmp_path, env={})


def test_wrap_argv_cli_rejects_empty_argv(tmp_path):
    with pytest.raises(ProtonWrapError, match="empty argv"):
        wrap_argv([], tmp_path, env={})


def test_wrap_argv_does_not_mutate_the_input(tmp_path):
    inner = ["python", "vecadd.py"]
    wrap_argv(inner, tmp_path, env={})
    assert inner == ["python", "vecadd.py"]


# ---- Device-variable translation ----------------------------------------


def test_wrap_argv_translates_hip_visible_devices(tmp_path, env_on_path):
    """Proton on AMD does not honour ``HIP_VISIBLE_DEVICES`` for the
    queue-intercepting backends, so a device-pinned cell would otherwise
    profile the wrong GPU."""
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": "roctracer"},
        env={"HIP_VISIBLE_DEVICES": "1"},
    )
    assert argv[0] == str(env_on_path)
    assert "-u" in argv[:3]
    assert "HIP_VISIBLE_DEVICES" in argv
    assert "ROCR_VISIBLE_DEVICES=1" in argv
    assert argv[argv.index("-m") - 1].endswith("python")


def test_wrap_argv_keeps_an_explicit_rocr_visible_devices(tmp_path, env_on_path):
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": "rocprofiler"},
        env={"HIP_VISIBLE_DEVICES": "1", "ROCR_VISIBLE_DEVICES": "0"},
    )
    assert "ROCR_VISIBLE_DEVICES=0" in argv


def test_wrap_argv_translates_an_empty_hip_visible_devices(tmp_path, env_on_path):
    """Proton rejects ``HIP_VISIBLE_DEVICES`` on presence, not on value.

    An empty device list is also a meaningful selection -- it conventionally
    hides every device -- so it must be carried across to ROCR rather than
    dropped, which would silently expose the GPUs the trial hid.
    """
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": "roctracer"},
        env={"HIP_VISIBLE_DEVICES": ""},
    )
    assert "HIP_VISIBLE_DEVICES" in argv
    assert "ROCR_VISIBLE_DEVICES=" in argv


def test_wrap_argv_keeps_an_explicitly_empty_rocr_visible_devices(tmp_path, env_on_path):
    """An empty ROCR value is an explicit "hide everything", so the documented
    "explicit ROCR wins" precedence must not overwrite it with the HIP list."""
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": "roctracer"},
        env={"HIP_VISIBLE_DEVICES": "1", "ROCR_VISIBLE_DEVICES": ""},
    )
    assert "ROCR_VISIBLE_DEVICES=" in argv
    assert "ROCR_VISIBLE_DEVICES=1" not in argv


def test_env_mode_fails_when_env_binary_is_missing(tmp_path, monkeypatch):
    """``mode: env`` always has an ``AORTA_PROTON_*`` bundle to deliver, so a
    missing ``env(1)`` would hand back the bare command and profile nothing."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(ProtonWrapError, match="env"):
        wrap_argv(["/tmp/gemm", "512"], tmp_path, {"mode": "env"}, env={})


def test_wrap_argv_no_env_prefix_for_instrumentation_backend(tmp_path, env_on_path):
    """The instrumentation backend installs no queue interceptor, so the
    device variables mean what they normally mean and are left alone."""
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": "instrumentation"},
        env={"HIP_VISIBLE_DEVICES": "1"},
    )
    assert argv[0] == "python"


def test_wrap_argv_no_env_prefix_when_no_device_pin(tmp_path, env_on_path):
    argv = wrap_argv(["python", "vecadd.py"], tmp_path, env={})
    assert argv[0] == "python"


def test_wrap_argv_fails_when_env_binary_is_missing(tmp_path, monkeypatch):
    """No ``env(1)`` means the device translation cannot be applied at all.

    Argv rewriting is the collector seam's only environment channel, so
    continuing would hand the command back unchanged -- with
    ``HIP_VISIBLE_DEVICES`` still set, which Proton rejects outright. A
    measurement that cannot be taken is a setup failure, not a silent
    downgrade.
    """
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(ProtonWrapError, match="env"):
        wrap_argv(
            ["python", "vecadd.py"],
            tmp_path,
            {"backend": "roctracer"},
            env={"HIP_VISIBLE_DEVICES": "1"},
        )


def test_wrap_argv_reads_os_environ_by_default(tmp_path, env_on_path, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    argv = wrap_argv(["python", "vecadd.py"], tmp_path, {"backend": "roctracer"})
    assert "ROCR_VISIBLE_DEVICES=3" in argv


def test_wrap_argv_never_mutates_the_supplied_env(tmp_path, env_on_path):
    env = {"HIP_VISIBLE_DEVICES": "1"}
    wrap_argv(["python", "vecadd.py"], tmp_path, {"backend": "roctracer"}, env=env)
    assert env == {"HIP_VISIBLE_DEVICES": "1"}


# ---- env-mode ------------------------------------------------------------


def test_build_env_bundle(tmp_path):
    env = build_env(tmp_path)
    assert env[f"{ENV_PREFIX}DIR"] == str(tmp_path)
    assert env[f"{ENV_PREFIX}NAME"] == str(tmp_path / PROFILE_BASENAME)
    assert env[f"{ENV_PREFIX}CONTEXT"] == "shadow"
    assert env[f"{ENV_PREFIX}DATA"] == "tree"
    assert f"{ENV_PREFIX}MODE" not in env
    assert all(key.startswith(ENV_PREFIX) for key in env)


def test_build_env_omits_backend_by_default(tmp_path):
    """Absent means ``proton.start(backend=None)``: Proton's own selection, the
    env-mode counterpart of dropping ``-b`` from the CLI wrap."""
    assert f"{ENV_PREFIX}BACKEND" not in build_env(tmp_path)


def test_build_env_carries_an_explicit_backend(tmp_path):
    env = build_env(tmp_path, {"backend": "rocprofiler"})
    assert env[f"{ENV_PREFIX}BACKEND"] == "rocprofiler"


def test_build_env_includes_mode_for_intra_kernel(tmp_path):
    env = build_env(tmp_path, {"backend": "instrumentation", "granularity": "warp"})
    assert env[f"{ENV_PREFIX}MODE"] == "default:granularity=warp"


def test_build_env_accepts_str_out_dir():
    assert build_env("relative/out")[f"{ENV_PREFIX}DIR"] == str(Path("relative/out"))


def test_wrap_argv_env_mode_leaves_the_command_alone(tmp_path, env_on_path):
    argv = wrap_argv(["/tmp/aorta_hip_gemm", "512"], tmp_path, {"mode": "env"}, env={})
    assert argv[0] == str(env_on_path)
    assert argv[-2:] == ["/tmp/aorta_hip_gemm", "512"]
    assert PROTON_MODULE not in argv
    assert f"{ENV_PREFIX}DATA=tree" in argv


def test_wrap_argv_env_mode_wraps_a_non_python_command(tmp_path, env_on_path):
    """The escape hatch the CLI-mode error message points at."""
    argv = wrap_argv(["/bin/true"], tmp_path, {"mode": "env"}, env={})
    assert argv[-1] == "/bin/true"


def test_wrap_argv_env_mode_still_translates_devices(tmp_path, env_on_path):
    argv = wrap_argv(
        ["/bin/true"],
        tmp_path,
        {"mode": "env", "backend": "roctracer"},
        env={"HIP_VISIBLE_DEVICES": "1"},
    )
    assert "ROCR_VISIBLE_DEVICES=1" in argv
    assert "HIP_VISIBLE_DEVICES" in argv


# ---- .hatchet parsing ----------------------------------------------------


def _hatchet(tmp_path: Path, payload, name: str = "proton.hatchet") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _tree(children):
    return [{"frame": {"name": "ROOT"}, "metrics": {}, "children": children}]


def _leaf(name, time_ns, count=1):
    return {
        "frame": {"name": name, "type": "function"},
        "metrics": {"count": count, "time (ns)": time_ns},
        "children": [],
    }


def test_parse_summary_real_capture_fixture():
    """The checked-in fixture is a real Proton ``data: tree`` capture.

    Produced on a gfx950 host by running the shipped ``triton-vecadd`` example
    under ``python -m triton.profiler.proton -b roctracer --context shadow
    --data tree``. Every leaf carries ``count`` and ``time (ns)`` alongside
    ``device_id`` / ``device_type``, and the Triton kernel (``add_kernel``) is
    the hot one -- the rest are the torch elementwise/reduction kernels the
    payload's own correctness check dispatches.
    """
    metrics = parse_summary(FIXTURES / "tree_capture")
    assert metrics["proton_kernel_count"] == 12
    assert metrics["proton_gpu_time_ms"] == pytest.approx(0.054402)
    assert metrics["proton_top_kernel_ms"] == pytest.approx(0.01752)
    assert metrics["proton_top_kernels"][0] == "add_kernel"
    assert metrics["proton_top_kernel_ms"] < metrics["proton_gpu_time_ms"]
    assert metrics["proton_artifact_dir"] == str(FIXTURES / "tree_capture")


def test_parse_summary_emits_only_expected_keys(tmp_path):
    _hatchet(tmp_path, _tree([_leaf("k", 1_000_000)]))
    assert set(parse_summary(tmp_path)) == {
        "proton_artifact_dir",
        "proton_kernel_count",
        "proton_gpu_time_ms",
        "proton_top_kernel_ms",
        "proton_top_kernels",
    }


def test_parse_summary_numeric_metrics_are_plain_numbers(tmp_path):
    _hatchet(tmp_path, _tree([_leaf("k", 1_000_000)]))
    metrics = parse_summary(tmp_path)
    for key in ("proton_kernel_count", "proton_gpu_time_ms", "proton_top_kernel_ms"):
        assert isinstance(metrics[key], (int, float))
        assert not isinstance(metrics[key], bool)


def test_parse_summary_converts_time_units(tmp_path):
    _hatchet(tmp_path, _tree([_leaf("k", 2_500_000)]))
    assert parse_summary(tmp_path)["proton_gpu_time_ms"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("unit", "value", "expected_ms"),
    [("ns", 1_000_000, 1.0), ("us", 1500, 1.5), ("ms", 2.5, 2.5), ("s", 0.25, 250.0)],
)
def test_parse_summary_accepts_every_time_unit(tmp_path, unit, value, expected_ms):
    node = {
        "frame": {"name": "k"},
        "metrics": {"count": 1, f"time ({unit})": value},
        "children": [],
    }
    _hatchet(tmp_path, _tree([node]))
    assert parse_summary(tmp_path)["proton_gpu_time_ms"] == pytest.approx(expected_ms)


def test_parse_summary_ignores_cpu_time_and_inclusive_columns(tmp_path):
    node = {
        "frame": {"name": "k"},
        "metrics": {"count": 1, "cpu_time (ns)": 9_000_000, "time (ns) (inc)": 9_000_000},
        "children": [],
    }
    _hatchet(tmp_path, _tree([node]))
    assert parse_summary(tmp_path) == {"proton_artifact_dir": str(tmp_path)}


def test_parse_summary_only_counts_leaves(tmp_path):
    """Interior frames carry inclusive time; counting them double-counts."""
    parent = {
        "frame": {"name": "parent"},
        "metrics": {"count": 1, "time (ns)": 10_000_000},
        "children": [_leaf("child", 4_000_000)],
    }
    _hatchet(tmp_path, _tree([parent]))
    metrics = parse_summary(tmp_path)
    assert metrics["proton_top_kernels"] == ["child"]
    assert metrics["proton_gpu_time_ms"] == pytest.approx(4.0)


def test_parse_summary_sums_repeated_kernel_names(tmp_path):
    _hatchet(
        tmp_path,
        _tree([_leaf("k", 1_000_000, count=3), _leaf("k", 2_000_000, count=2)]),
    )
    metrics = parse_summary(tmp_path)
    assert metrics["proton_gpu_time_ms"] == pytest.approx(3.0)
    assert metrics["proton_kernel_count"] == 5


def test_parse_summary_defaults_missing_count_to_one(tmp_path):
    node = {"frame": {"name": "k"}, "metrics": {"time (ns)": 1_000_000}, "children": []}
    _hatchet(tmp_path, _tree([node]))
    assert parse_summary(tmp_path)["proton_kernel_count"] == 1


def test_parse_summary_ranks_and_caps_top_kernels(tmp_path):
    leaves = [_leaf(f"k{i}", (12 - i) * 1_000_000) for i in range(12)]
    _hatchet(tmp_path, _tree(leaves))
    metrics = parse_summary(tmp_path)
    assert metrics["proton_top_kernels"] == ["k0", "k1", "k2", "k3", "k4"]
    assert metrics["proton_top_kernel_ms"] == pytest.approx(12.0)


def test_parse_summary_aggregates_multiple_profiles(tmp_path):
    _hatchet(tmp_path, _tree([_leaf("k", 1_000_000)]), name="a.hatchet")
    _hatchet(tmp_path, _tree([_leaf("k", 3_000_000)]), name="b.hatchet")
    assert parse_summary(tmp_path)["proton_gpu_time_ms"] == pytest.approx(4.0)


def test_parse_summary_finds_nested_profiles(tmp_path):
    nested = tmp_path / "rank0"
    nested.mkdir()
    _hatchet(nested, _tree([_leaf("k", 1_000_000)]))
    assert parse_summary(tmp_path)["proton_gpu_time_ms"] == pytest.approx(1.0)


# ---- .hatchet parsing: fail-soft ----------------------------------------


def test_parse_summary_missing_dir_is_empty(tmp_path):
    assert parse_summary(tmp_path / "never-created") == {}


def test_parse_summary_file_instead_of_dir_is_empty(tmp_path):
    path = tmp_path / "not-a-dir"
    path.write_text("")
    assert parse_summary(path) == {}


def test_parse_summary_empty_dir_reports_only_the_artifact_dir(tmp_path):
    assert parse_summary(tmp_path) == {"proton_artifact_dir": str(tmp_path)}


def test_parse_summary_malformed_json_degrades_without_raising(tmp_path):
    (tmp_path / "proton.hatchet").write_text("{not json at all", encoding="utf-8")
    assert parse_summary(tmp_path) == {"proton_artifact_dir": str(tmp_path)}


def test_parse_summary_unexpected_json_shape_degrades(tmp_path):
    _hatchet(tmp_path, {"totally": "different"})
    assert parse_summary(tmp_path) == {"proton_artifact_dir": str(tmp_path)}


def test_parse_summary_trace_format_has_no_tree_to_walk(tmp_path):
    """``data: trace`` produces a chrome trace, not a hatchet tree."""
    _hatchet(tmp_path, {"traceEvents": [{"name": "k", "dur": 5}]})
    assert parse_summary(tmp_path) == {"proton_artifact_dir": str(tmp_path)}


def test_parse_summary_skips_unnamed_and_untimed_leaves(tmp_path):
    _hatchet(
        tmp_path,
        _tree(
            [
                {"frame": {"name": ""}, "metrics": {"time (ns)": 5_000_000}, "children": []},
                {"frame": {"name": "no-time"}, "metrics": {"count": 1}, "children": []},
                _leaf("good", 1_000_000),
            ]
        ),
    )
    metrics = parse_summary(tmp_path)
    assert metrics["proton_top_kernels"] == ["good"]


def test_parse_summary_tolerates_non_numeric_time(tmp_path):
    node = {
        "frame": {"name": "k"},
        "metrics": {"count": 1, "time (ns)": "n/a"},
        "children": [],
    }
    _hatchet(tmp_path, _tree([node]))
    assert parse_summary(tmp_path) == {"proton_artifact_dir": str(tmp_path)}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1_000_000])
def test_parse_summary_skips_non_finite_and_negative_time(tmp_path, bad):
    """A ``NaN`` / infinite / negative duration must not reach
    ``proton_gpu_time_ms``: the first two serialise as non-standard JSON
    tokens, and none of them is a time a kernel can take."""
    _hatchet(tmp_path, _tree([_leaf("bad", bad), _leaf("good", 1_000_000)]))
    metrics = parse_summary(tmp_path)
    assert metrics["proton_top_kernels"] == ["good"]
    assert metrics["proton_gpu_time_ms"] == pytest.approx(1.0)


def test_parse_summary_still_reads_a_good_time_key_after_a_bad_one(tmp_path):
    """The scan continues past an unusable ``time (...)`` entry rather than
    giving up on the leaf, so a profile carrying both still reports."""
    node = {
        "frame": {"name": "k"},
        "metrics": {"count": 1, "time (s)": float("nan"), "time (ns)": 2_000_000},
        "children": [],
    }
    _hatchet(tmp_path, _tree([node]))
    assert parse_summary(tmp_path)["proton_gpu_time_ms"] == pytest.approx(2.0)


def test_parse_summary_drops_metrics_when_the_total_overflows(tmp_path):
    """``_time_ms`` checks each leaf, but finite leaves still sum to infinity.

    Spelled in ``ms`` so the values reach the accumulator unscaled -- the same
    magnitude in ``ns`` is divided down to a comfortably finite number.
    """

    def leaf(name):
        return {
            "frame": {"name": name},
            "metrics": {"count": 1, "time (ms)": 1e308},
            "children": [],
        }

    _hatchet(tmp_path, _tree([leaf("a"), leaf("b"), leaf("c")]))
    assert parse_summary(tmp_path) == {"proton_artifact_dir": str(tmp_path)}


@pytest.mark.parametrize("bad_count", [float("inf"), 10**400, -5])
def test_parse_summary_survives_an_unusable_count(tmp_path, bad_count):
    """``int(float(...))`` raises ``OverflowError`` for an infinite or huge
    count, which would escape ``parse_summary()``'s never-raises contract."""
    _hatchet(tmp_path, _tree([_leaf("k", 1_000_000, count=bad_count)]))
    metrics = parse_summary(tmp_path)
    assert metrics["proton_kernel_count"] == 1
    assert metrics["proton_gpu_time_ms"] == pytest.approx(1.0)


def test_parse_summary_tolerates_non_numeric_count(tmp_path):
    node = {
        "frame": {"name": "k"},
        "metrics": {"count": "many", "time (ns)": 1_000_000},
        "children": [],
    }
    _hatchet(tmp_path, _tree([node]))
    assert parse_summary(tmp_path)["proton_kernel_count"] == 1


def test_parse_summary_tolerates_non_dict_children(tmp_path):
    node = {"frame": {"name": "k"}, "metrics": {"time (ns)": 1_000_000}, "children": ["junk"]}
    _hatchet(tmp_path, _tree([node]))
    assert parse_summary(tmp_path) == {"proton_artifact_dir": str(tmp_path)}


def test_parse_summary_ignores_device_metadata_elements(tmp_path):
    """A real hatchet file is a list: the tree plus device metadata dicts."""
    payload = [
        {"frame": {"name": "ROOT"}, "metrics": {}, "children": [_leaf("k", 1_000_000)]},
        {"device": {"0": {"arch": "gfx950", "num_sms": 256}}},
    ]
    _hatchet(tmp_path, payload)
    assert parse_summary(tmp_path)["proton_top_kernels"] == ["k"]


def test_parse_profile_missing_file_is_empty(tmp_path):
    assert parse_profile(tmp_path / "nope.hatchet") == ({}, {})


def test_parse_summary_accepts_str_path(tmp_path):
    _hatchet(tmp_path, _tree([_leaf("k", 1_000_000)]))
    assert parse_summary(str(tmp_path))["proton_top_kernels"] == ["k"]
