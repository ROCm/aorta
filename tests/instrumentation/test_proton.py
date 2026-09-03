"""Tests for the ``proton`` collector package (no GPU, no Triton needed).

Covers the option schema, both attach modes (``cli`` argv rewrite and ``env``
variable bundle), the ``HIP_VISIBLE_DEVICES`` -> ``ROCR_VISIBLE_DEVICES``
translation Proton needs on AMD, and fail-soft ``.hatchet`` parsing.

It also carries one guard over the *sibling* GPU module's source
(``test_gpu_smoke_subprocesses_all_use_the_shared_child_budget``): the GPU legs
are path-skipped on most PRs, so a convention that only they exercise needs a
CPU test to hold it. Because that guard reads one fixed file, its own detection
is exercised separately against snippets -- a guard checked only against source
that already complies cannot say what it would reject.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

from aorta.instrumentation.proton import (
    AUTO_BACKEND,
    BACKEND_MODES,
    BACKENDS,
    ENV_PREFIX,
    ENV_PROTON_PYTHON,
    HOOKS,
    MODE_BEARING_KEYS,
    OPTION_KEYS,
    OUTPUT_SUBDIR,
    PROFILE_BASENAME,
    PROTON_MODULE,
    QUEUE_INTERCEPTING_BACKENDS,
    ProtonWrapError,
    _parse,
    build_argv_prefix,
    build_env,
    mode_argument,
    parse_profile,
    parse_summary,
    parse_summary_from_streams,
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


@pytest.mark.parametrize(
    ("backend", "backend_mode"),
    sorted((backend, mode) for backend, modes in BACKEND_MODES.items() for mode in modes),
)
def test_validate_options_accepts_every_backend_mode_of_its_backend(backend, backend_mode):
    effective = validate_options({"backend": backend, "backend_mode": backend_mode})
    assert effective["backend_mode"] == backend_mode


def test_validate_options_rejects_a_backend_mode_the_backend_does_not_have():
    """The domains differ per backend -- ``pcsampling`` is a rocprofiler / cupti
    mode, and roctracer only has ``periodic_flushing``. A flat union would
    accept this and fail later inside Proton."""
    with pytest.raises(ValueError, match="not one of.*for backend: roctracer"):
        validate_options({"backend": "roctracer", "backend_mode": "pcsampling"})


#: A valid (backend, value) pair for each ``--mode``-bearing option, so the
#: parametrised tests below cover whatever ``MODE_BEARING_KEYS`` holds; a key
#: added there without an entry here fails with the KeyError naming it.
_MODE_BEARING_SAMPLE: dict[str, tuple[str, str]] = {
    "backend_mode": ("cupti", "pcsampling"),
    "instrumentation_mode": ("instrumentation", "mma"),
    "granularity": ("instrumentation", "warp"),
}


def test_mode_bearing_sample_covers_every_declared_key():
    assert set(_MODE_BEARING_SAMPLE) == set(MODE_BEARING_KEYS)


@pytest.mark.parametrize("key", MODE_BEARING_KEYS)
def test_validate_options_does_not_gate_mode_knobs_on_the_attach_mode(key):
    """``mode: cli`` with a ``--mode`` knob is accepted, deliberately. Triton
    3.8.0 forwards ``mode=args.mode`` into ``start()``, so refusing it would
    reject a recipe that is correct on the current release -- and aorta validates
    in its own interpreter, so it cannot tell which Triton will run the wrap.
    Whether the value lands is documented per version, not enforced here."""
    backend, value = _MODE_BEARING_SAMPLE[key]
    effective = validate_options({"mode": "cli", "backend": backend, key: value})
    assert effective[key] == value


def test_validate_options_rejects_unknown_backend_mode():
    with pytest.raises(ValueError, match="not one of"):
        validate_options({"backend": "rocprofiler", "backend_mode": "sampling"})


@pytest.mark.parametrize("backend", ["auto", "instrumentation"])
def test_validate_options_rejects_backend_mode_without_a_backend_to_validate_it(backend):
    """``auto`` resolves at runtime, so its mode domain is unknown here; the
    instrumentation backend's modes are the ``instrumentation_mode`` pair."""
    with pytest.raises(ValueError, match="requires an explicit backend"):
        validate_options({"backend": backend, "backend_mode": "pcsampling"})


@pytest.mark.parametrize("intra_kernel_key", ["instrumentation_mode", "granularity"])
def test_validate_options_rejects_backend_mode_beside_an_intra_kernel_knob(intra_kernel_key):
    """Both render Proton's single ``--mode``, so the pair has no rendering.

    The message must name the collision rather than the backend gate that would
    also have rejected this: the fix is dropping one option, not changing the
    backend.
    """
    value = "cta" if intra_kernel_key == "granularity" else "mma"
    with pytest.raises(ValueError, match="conflicts with"):
        validate_options(
            {
                "backend": "instrumentation",
                "backend_mode": "pcsampling",
                intra_kernel_key: value,
            }
        )


@pytest.mark.parametrize("hook", sorted(HOOKS))
def test_validate_options_accepts_every_hook(hook):
    assert validate_options({"hook": hook})["hook"] == hook


def test_validate_options_rejects_unknown_hook():
    with pytest.raises(ValueError, match="'hook'"):
        validate_options({"hook": "launch"})


def test_hook_is_not_gated_on_a_backend():
    """Proton's ``-k`` is backend-independent: it registers a launch hook that
    records Triton kernel metadata whichever backend measures."""
    for backend in sorted(BACKENDS):
        assert validate_options({"backend": backend, "hook": "triton"})["hook"] == "triton"


def test_validate_options_does_not_mutate_input():
    supplied = {"backend": "ROCTRACER"}
    validate_options(supplied)
    assert supplied == {"backend": "ROCTRACER"}


def test_option_keys_match_the_validator():
    """Two samples rather than one: ``backend_mode`` and the intra-kernel pair
    render the same ``--mode`` and are rejected together, so no single mapping
    can carry every declared key."""
    intra_kernel = {
        "mode": "cli",
        "backend": "instrumentation",
        "context": "shadow",
        "data": "tree",
        "instrumentation_mode": "mma",
        "granularity": "warp",
        "hook": "triton",
    }
    whole_kernel = {
        "mode": "env",
        "backend": "roctracer",
        "backend_mode": "periodic_flushing",
        "context": "python",
        "data": "trace",
    }
    assert set(intra_kernel) | set(whole_kernel) == set(OPTION_KEYS)
    assert validate_options(intra_kernel)
    assert validate_options(whole_kernel)


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


def test_mode_argument_renders_backend_mode():
    """``backend_mode`` is Proton's ``--mode`` for the whole-kernel backends, so
    it renders as the bare name -- no ``granularity=`` grammar, which belongs to
    the instrumentation backend."""
    effective = validate_options({"backend": "roctracer", "backend_mode": "periodic_flushing"})
    assert mode_argument(effective) == "periodic_flushing"


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


def test_build_argv_prefix_includes_mode_for_a_backend_mode(tmp_path):
    argv = build_argv_prefix(tmp_path, {"backend": "cupti", "backend_mode": "pcsampling"})
    assert argv[argv.index("--mode") + 1] == "pcsampling"


def test_build_argv_prefix_omits_mode_when_no_knob_asks_for_it(tmp_path):
    assert "--mode" not in build_argv_prefix(tmp_path, {"backend": "cupti"})


def test_build_argv_prefix_omits_the_hook_flag_by_default(tmp_path):
    assert "-k" not in build_argv_prefix(tmp_path)


def test_build_argv_prefix_renders_the_hook(tmp_path):
    argv = build_argv_prefix(tmp_path, {"hook": "triton"})
    assert argv[argv.index("-k") + 1] == "triton"


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


# ---- CLI mode cannot pin a queue-intercepting backend -------------------
#
# Verified on gfx950 / ROCm 7.0.2 / Triton 3.7.1 against the shipped
# triton-vecadd payload: no ``-b`` gives a ~3 KB hatchet holding 27 dispatches,
# while ``-b roctracer`` gives a 160-byte hatchet whose ROOT frame has empty
# metrics -- from a run that exits 0. Proton's front-end calls
# ``_select_backend()`` only when ``-b`` is absent, and that call is what
# initialises the HIP driver; a queue interceptor that starts first records
# nothing. In aorta that surfaced as a trial with no proton metrics at all,
# since ``parse_summary`` degrades to the artifact directory for an empty tree.
#
# The guard covers ``roctracer`` and nothing else, and the exclusions are not
# one rule but three:
#
# * ``instrumentation`` needs no interceptor and captures correctly under a CLI
#   pin, so refusing it would cost a working configuration; what ``mode: cli``
#   costs there is the ``--mode`` knobs, which the schema documents.
# * ``rocprofiler`` has the *opposite* contract. Triton 3.8.0 calls
#   ``rocprofiler_force_configure`` from an ``__attribute__((constructor))`` in
#   ``libproton.so``, and its source warns that letting HSA come up first -- "a
#   torch import chain" -- yields an empty dispatch buffer. The CLI path loads
#   libproton before the payload, so it is the ordering upstream *wants*.
# * ``auto`` is the ``-b``-absent path, i.e. the working one.


def test_wrap_argv_cli_refuses_to_pin_roctracer(tmp_path):
    with pytest.raises(ProtonWrapError, match="cannot pin backend"):
        wrap_argv(["python", "vecadd.py"], tmp_path, {"backend": "roctracer"}, env={})


@pytest.mark.parametrize("backend", ["rocprofiler", "instrumentation", "cupti"])
def test_wrap_argv_cli_allows_the_backends_whose_ordering_it_does_not_break(tmp_path, backend):
    """Only ``roctracer`` is refused. ``rocprofiler`` in particular must stay
    allowed: refusing it would push operators onto the env-mode ordering its own
    upstream source warns produces an empty dispatch buffer."""
    argv = wrap_argv(["python", "vecadd.py"], tmp_path, {"backend": backend}, env={})
    assert argv[argv.index("-b") + 1] == backend


def test_cli_backend_pin_rejection_names_the_mechanism_and_the_route(tmp_path):
    """The message has to carry the reason, because the alternative outcome is
    a green trial: an operator who is only told "not supported" will reach for
    the pin again on the next Triton."""
    with pytest.raises(ProtonWrapError) as excinfo:
        wrap_argv(["python", "vecadd.py"], tmp_path, {"backend": "roctracer"}, env={})
    message = str(excinfo.value)
    assert "_select_backend()" in message
    assert "empty ROOT frame" in message
    assert "mode: env" in message


def test_wrap_argv_cli_still_allows_the_default_backend(tmp_path):
    """``auto`` is the ``-b``-absent path, so it is the one that works."""
    argv = wrap_argv(["python", "vecadd.py"], tmp_path, {"backend": AUTO_BACKEND}, env={})
    assert "-b" not in argv


@pytest.mark.parametrize("backend", ["instrumentation", "cupti"])
def test_wrap_argv_cli_still_pins_a_non_intercepting_backend(tmp_path, backend):
    """Neither installs an HSA queue interceptor, so neither depends on
    attaching before the runtime comes up."""
    argv = wrap_argv(["python", "vecadd.py"], tmp_path, {"backend": backend}, env={})
    assert argv[argv.index("-b") + 1] == backend


@pytest.mark.parametrize("backend", ["roctracer", "rocprofiler"])
def test_wrap_argv_env_mode_pins_the_same_backend_happily(tmp_path, env_on_path, backend):
    """``mode: env`` accepts either backend -- but what the payload must then do
    differs, and the collector cannot enforce it from here.

    For ``roctracer`` this is the route the CLI rejection names: the payload
    starts Proton after its own ``import torch`` has brought the runtime up. For
    ``rocprofiler`` the responsibility is the opposite one -- import Proton
    *before* torch, so its ``libproton.so`` constructor configures the SDK
    before HSA exists (see ``amd-rocprofiler/gelu.py``). Env mode is safe for
    both; only the payload's import order distinguishes them.
    """
    argv = wrap_argv(
        ["python", "pipeline.py"],
        tmp_path,
        {"mode": "env", "backend": backend},
        env={},
    )
    assert f"{ENV_PREFIX}BACKEND={backend}" in argv


def test_cli_pin_is_refused_before_the_argv_shape_is_checked(tmp_path):
    """Both failures are fixed by ``mode: env``, and this is the one whose
    absence is invisible, so it is the one to report."""
    with pytest.raises(ProtonWrapError, match="cannot pin backend"):
        wrap_argv(["/tmp/aorta_hip_gemm", "512"], tmp_path, {"backend": "roctracer"}, env={})


# ---- Device-variable translation ----------------------------------------


def test_wrap_argv_translates_hip_visible_devices(tmp_path, env_on_path):
    """Proton on AMD does not honour ``HIP_VISIBLE_DEVICES`` for the
    queue-intercepting backends, so a device-pinned cell would otherwise
    profile the wrong GPU.

    Spelled with ``backend: auto`` because that is the only queue-intercepting
    backend ``mode: cli`` will pin, and the assertion below is about where the
    ``env(1)`` prefix sits relative to the CLI wrap.
    """
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": AUTO_BACKEND},
        env={"HIP_VISIBLE_DEVICES": "1"},
    )
    assert argv[0] == str(env_on_path)
    assert "-u" in argv[:3]
    assert "HIP_VISIBLE_DEVICES" in argv
    assert "ROCR_VISIBLE_DEVICES=1" in argv
    assert argv[argv.index("-m") - 1].endswith("python")


def test_wrap_argv_translates_cuda_visible_devices(tmp_path, env_on_path):
    """Proton rejects the CUDA spelling on AMD too, and it is the likelier one.

    ROCm's PyTorch presents its devices as ``cuda``, so an operator pinning a
    GPU commonly reaches for ``CUDA_VISIBLE_DEVICES``. Handling only the HIP
    spelling left that trial reaching Proton with a variable it refuses.

    ``mode: env`` throughout the explicitly-pinned cases below: an explicit
    ``roctracer`` / ``rocprofiler`` is refused in ``mode: cli``, and the
    translation is the same code either way.
    """
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"mode": "env", "backend": "roctracer"},
        env={"CUDA_VISIBLE_DEVICES": "2"},
    )
    assert "CUDA_VISIBLE_DEVICES" in argv
    assert "ROCR_VISIBLE_DEVICES=2" in argv


def test_wrap_argv_unsets_both_device_spellings_and_prefers_hip(tmp_path, env_on_path):
    """Both are unset; HIP supplies the ROCR value, matching ROCm's preference."""
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"mode": "env", "backend": "rocprofiler"},
        env={"HIP_VISIBLE_DEVICES": "1", "CUDA_VISIBLE_DEVICES": "2"},
    )
    assert "HIP_VISIBLE_DEVICES" in argv
    assert "CUDA_VISIBLE_DEVICES" in argv
    assert "ROCR_VISIBLE_DEVICES=1" in argv


def test_wrap_argv_does_not_touch_cuda_visible_devices_under_auto(tmp_path):
    """``auto`` resolves to CUPTI on NVIDIA, where a CUDA pin is real.

    Translating it there would silently drop the device restriction and profile
    the wrong GPU, so the CUDA spelling is only rewritten once AMD is
    established -- by an explicitly AMD backend, or by HIP_VISIBLE_DEVICES
    being set alongside it.
    """
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": "auto"},
        env={"CUDA_VISIBLE_DEVICES": "2"},
    )
    assert "CUDA_VISIBLE_DEVICES" not in argv
    assert not any(part.startswith("ROCR_VISIBLE_DEVICES=") for part in argv)


def test_wrap_argv_translates_cuda_under_auto_when_rocr_establishes_amd(tmp_path, env_on_path):
    """``ROCR_VISIBLE_DEVICES`` is ROCm-only, so its presence establishes AMD.

    It is the variable Proton *reads* on AMD, so a trial that sets it is already
    on that path -- leaving ``CUDA_VISIBLE_DEVICES`` beside it just hands Proton
    a variable it refuses before profiling starts.
    """
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": "auto"},
        env={"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "2"},
    )
    assert "CUDA_VISIBLE_DEVICES" in argv
    # The explicit ROCR value wins; it is not overwritten by the CUDA list.
    assert "ROCR_VISIBLE_DEVICES=0" in argv


def test_wrap_argv_translates_cuda_under_auto_when_hip_establishes_amd(tmp_path, env_on_path):
    """HIP_VISIBLE_DEVICES present means AMD, so the CUDA pin is rewritten too."""
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": "auto"},
        env={"HIP_VISIBLE_DEVICES": "1", "CUDA_VISIBLE_DEVICES": "2"},
    )
    assert "HIP_VISIBLE_DEVICES" in argv
    assert "CUDA_VISIBLE_DEVICES" in argv
    assert "ROCR_VISIBLE_DEVICES=1" in argv


def test_wrap_argv_still_translates_hip_under_auto(tmp_path, env_on_path):
    """HIP_VISIBLE_DEVICES is an AMD signal by itself, so ``auto`` still acts."""
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": "auto"},
        env={"HIP_VISIBLE_DEVICES": "1"},
    )
    assert "HIP_VISIBLE_DEVICES" in argv
    assert "ROCR_VISIBLE_DEVICES=1" in argv


def test_wrap_argv_leaves_device_vars_alone_on_a_non_intercepting_backend(tmp_path):
    """``instrumentation`` installs no queue interceptor, so nothing to rewrite."""
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"backend": "instrumentation"},
        env={"CUDA_VISIBLE_DEVICES": "2", "HIP_VISIBLE_DEVICES": "1"},
    )
    assert "CUDA_VISIBLE_DEVICES" not in argv
    assert "ROCR_VISIBLE_DEVICES=2" not in argv


def test_wrap_argv_keeps_an_explicit_rocr_visible_devices(tmp_path, env_on_path):
    argv = wrap_argv(
        ["python", "vecadd.py"],
        tmp_path,
        {"mode": "env", "backend": "rocprofiler"},
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
        {"mode": "env", "backend": "roctracer"},
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
        {"mode": "env", "backend": "roctracer"},
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
            {"backend": AUTO_BACKEND},
            env={"HIP_VISIBLE_DEVICES": "1"},
        )


def test_wrap_argv_reads_os_environ_by_default(tmp_path, env_on_path, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    argv = wrap_argv(["python", "vecadd.py"], tmp_path, {"backend": AUTO_BACKEND})
    assert "ROCR_VISIBLE_DEVICES=3" in argv


def test_wrap_argv_never_mutates_the_supplied_env(tmp_path, env_on_path):
    env = {"HIP_VISIBLE_DEVICES": "1"}
    wrap_argv(["python", "vecadd.py"], tmp_path, {"mode": "env", "backend": "roctracer"}, env=env)
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


def test_build_env_includes_mode_for_a_backend_mode(tmp_path):
    env = build_env(tmp_path, {"backend": "rocprofiler", "backend_mode": "pcsampling"})
    assert env[f"{ENV_PREFIX}MODE"] == "pcsampling"


def test_build_env_omits_the_hook_by_default(tmp_path):
    assert f"{ENV_PREFIX}HOOK" not in build_env(tmp_path)


def test_build_env_carries_the_hook(tmp_path):
    """The env bundle mirrors the CLI flags one-for-one, so a payload driving
    Proton itself can forward ``hook`` into ``proton.start()``."""
    assert build_env(tmp_path, {"hook": "triton"})[f"{ENV_PREFIX}HOOK"] == "triton"


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


def test_parse_summary_omits_the_count_when_a_leaf_has_none(tmp_path):
    """A leaf with no ``count`` yields no ``proton_kernel_count``, not a 1.

    The checked-in real capture has ``count: 6`` on its ``add_kernel`` leaf,
    which is the proof that a leaf is an aggregate over launches rather than a
    single dispatch -- so there is nothing 1 could legitimately mean here.
    """
    node = {"frame": {"name": "k"}, "metrics": {"time (ns)": 1_000_000}, "children": []}
    _hatchet(tmp_path, _tree([node]))
    metrics = parse_summary(tmp_path)
    assert "proton_kernel_count" not in metrics
    assert metrics.get("proton_gpu_time_ms") == pytest.approx(1.0)


def test_parse_summary_real_fixture_counts_aggregate_launches(tmp_path):
    """Guards the premise above: the real fixture's count exceeds its leaf count."""
    metrics = parse_summary(FIXTURES / "tree_capture")
    assert metrics.get("proton_kernel_count", 0) > len(metrics.get("proton_top_kernels", []))


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
    count, which would escape ``parse_summary()``'s never-raises contract.

    The count is omitted rather than defaulted: a Proton leaf aggregates every
    launch of its kernel (the real fixture carries ``count: 6``), so 1 is not
    a safe stand-in. The timing is unaffected and still published.
    """
    _hatchet(tmp_path, _tree([_leaf("k", 1_000_000, count=bad_count)]))
    metrics = parse_summary(tmp_path)
    assert "proton_kernel_count" not in metrics
    assert metrics.get("proton_gpu_time_ms") == pytest.approx(1.0)


def test_parse_summary_tolerates_non_numeric_count(tmp_path):
    node = {
        "frame": {"name": "k"},
        "metrics": {"count": "many", "time (ns)": 1_000_000},
        "children": [],
    }
    _hatchet(tmp_path, _tree([node]))
    metrics = parse_summary(tmp_path)
    assert "proton_kernel_count" not in metrics
    assert metrics.get("proton_gpu_time_ms") == pytest.approx(1.0)


def test_parse_summary_one_unreadable_leaf_drops_only_the_count(tmp_path):
    """A single bad leaf must not poison the timings of its siblings."""
    good = _leaf("good", 2_000_000, count=4)
    bad = {
        "frame": {"name": "bad"},
        "metrics": {"time (ns)": 1_000_000},  # no ``count`` at all
        "children": [],
    }
    _hatchet(tmp_path, _tree([good, bad]))
    metrics = parse_summary(tmp_path)
    assert "proton_kernel_count" not in metrics
    assert metrics.get("proton_gpu_time_ms") == pytest.approx(3.0)
    assert metrics.get("proton_top_kernels") == ["good", "bad"]


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


@pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(), reason="descriptor accounting needs /proc"
)
def test_parse_summary_holds_one_profile_open_at_a_time(tmp_path, monkeypatch):
    """A per-rank capture must not need a descriptor per rank.

    Proton writes a profile per rank, so a large distributed run can hold more
    ``.hatchet`` files than ``RLIMIT_NOFILE`` allows; opening them all up front
    hit ``EMFILE`` partway through and reported totals covering only the ranks
    before the limit. The count is compared across two tree sizes because an
    absolute bound would pass for the eager version on a small tree.
    """
    original = _parse.parse_profile_stream

    def peak_open_fds(root, ranks):
        root.mkdir()
        for rank in range(ranks):
            _hatchet(root, _tree([_leaf("k", 1_000_000)]), name=f"rank{rank:03d}.hatchet")
        parse_summary(root)  # warm any lazy imports before the baseline
        baseline = len(os.listdir("/proc/self/fd"))
        seen = []

        def counting_parse(stream):
            seen.append(len(os.listdir("/proc/self/fd")))
            return original(stream)

        monkeypatch.setattr(_parse, "parse_profile_stream", counting_parse)
        try:
            metrics = parse_summary(root)
        finally:
            monkeypatch.setattr(_parse, "parse_profile_stream", original)
        assert metrics.get("proton_kernel_count") == ranks
        assert len(seen) == ranks
        return max(seen) - baseline

    small = peak_open_fds(tmp_path / "small", 8)
    large = peak_open_fds(tmp_path / "large", 64)
    assert small == large
    assert large <= 2


def test_parse_summary_from_streams_consumes_a_lazy_iterator(tmp_path):
    """The stream entrypoint takes an iterator, not just a sequence.

    Both callers now pass a generator that opens one profile at a time, so a
    ``len()`` or a second pass would break them.
    """
    path = _hatchet(tmp_path, _tree([_leaf("k", 2_000_000, count=3)]))
    metrics = parse_summary_from_streams(
        str(tmp_path), (p.open(encoding="utf-8") for p in [path])
    )
    assert metrics.get("proton_kernel_count") == 3
    assert metrics.get("proton_gpu_time_ms") == pytest.approx(2.0)


#: Callables that block on a child process and take a ``timeout`` keyword.
#: Matched on the trailing name only, so ``subprocess.run(...)``,
#: ``run(...)`` after ``from subprocess import run`` and
#: ``proc.communicate(...)`` all resolve the same way.
_BLOCKS_ON_A_CHILD = frozenset(
    {"run", "call", "check_call", "check_output", "communicate", "wait", "wait_for"}
)

#: Child launches that cannot be given a timeout at all, so there is no
#: bounded spelling of them to accept.
_UNBOUNDABLE_CHILD_LAUNCH = frozenset({"system", "popen"})


def _called_name(node: ast.Call) -> str | None:
    """Trailing identifier of a call's callee, ignoring how it was reached."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _child_budget_violations(source: str, filename: str = "<snippet>") -> dict[str, list]:
    """Ways ``source`` escapes the shared child budget, keyed by how.

    ``narrowed``: a ``timeout=`` that is not ``_CHILD_TIMEOUT_S``, on any call
    shape. ``unbounded``: a blocking child call with no ``timeout=`` keyword.
    ``unboundable``: a launch that takes no timeout at all.
    """
    tree = ast.parse(source, filename=filename)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return {
        "narrowed": [
            (keyword.lineno, ast.unparse(keyword.value))
            for node in calls
            for keyword in node.keywords
            if keyword.arg == "timeout"
            and not (
                isinstance(keyword.value, ast.Name) and keyword.value.id == "_CHILD_TIMEOUT_S"
            )
        ],
        "unbounded": [
            (node.lineno, ast.unparse(node.func))
            for node in calls
            if (_called_name(node) or "") in _BLOCKS_ON_A_CHILD
            and not any(keyword.arg == "timeout" for keyword in node.keywords)
        ],
        "unboundable": [
            (node.lineno, ast.unparse(node.func))
            for node in calls
            if (_called_name(node) or "") in _UNBOUNDABLE_CHILD_LAUNCH
        ],
    }


@pytest.mark.parametrize(
    ("kind", "snippet"),
    [
        pytest.param("narrowed", "subprocess.run(argv, timeout=3600)", id="literal-timeout"),
        pytest.param("narrowed", "proc.communicate(timeout=600)", id="literal-on-communicate"),
        pytest.param("unbounded", "subprocess.run(argv)", id="no-timeout-at-all"),
        pytest.param("unbounded", "run(argv, check=True)", id="no-timeout-bare-import"),
        pytest.param("unbounded", "subprocess.check_output(argv)", id="no-timeout-check-output"),
        pytest.param("unbounded", "proc.communicate()", id="no-timeout-communicate"),
        pytest.param("unbounded", "proc.wait()", id="no-timeout-wait"),
        pytest.param("unboundable", "os.system(cmd)", id="os-system"),
        # Spellings a human reviewer mutation-tested against the keyword-only
        # version of this guard and found it green on. Kept as their own cases
        # so the three cannot regress independently of the ones above.
        pytest.param(
            "unbounded",
            "subprocess.run(argv, capture_output=True, text=True)",
            id="reviewer-kwarg-deleted",
        ),
        pytest.param("unbounded", "proc.wait(3600)", id="reviewer-positional-wait"),
        pytest.param(
            "unbounded",
            'subprocess.run(argv, **{"timeout": 3600})',
            id="reviewer-kwargs-unpack",
        ),
    ],
)
def test_the_child_budget_guard_reports_each_kind_of_escape(kind, snippet):
    """Each escape must be reported, and filed as its own kind.

    The guard below reads one fixed file, so on its own it can only say that
    *today's* calls are bounded. These snippets are what say it would notice a
    new one -- the first version keyed on ``timeout=`` being present, so
    ``subprocess.run(argv)``, the most likely regression of all, sailed past it.

    What it does not claim to cover: a launch spelled with a name outside
    :data:`_BLOCKS_ON_A_CHILD` and :data:`_UNBOUNDABLE_CHILD_LAUNCH` (say
    ``os.spawnv`` or a bare ``Popen`` that is never waited on). A budget passed
    positionally is reported as missing rather than accepted, which is the
    fail-closed direction.
    """
    found = _child_budget_violations(snippet)
    assert found[kind], f"{snippet!r} not reported as {kind}: {found}"
    assert not [k for k in found if k != kind and found[k]], f"{snippet!r} misfiled: {found}"


@pytest.mark.parametrize(
    "snippet",
    [
        pytest.param("subprocess.run(argv, timeout=_CHILD_TIMEOUT_S)", id="run"),
        pytest.param("proc.communicate(timeout=_CHILD_TIMEOUT_S)", id="communicate"),
        pytest.param("json.load(handle)", id="unrelated-call"),
        pytest.param("shutil.which('proton')", id="no-child"),
    ],
)
def test_the_child_budget_guard_accepts_what_it_should(snippet):
    """A guard that flags correct code gets disabled, so pin the negatives too."""
    assert not any(_child_budget_violations(snippet).values()), snippet


def test_gpu_smoke_subprocesses_all_use_the_shared_child_budget():
    """Every child launch in the GPU smoke module must carry ``_CHILD_TIMEOUT_S``.

    ``test_proton_smoke_gpu.py`` sizes ``_CHILD_TIMEOUT_S`` so a wedged payload
    surfaces as ``TimeoutExpired`` naming that payload, rather than as the CI
    job hitting its own 60-minute cap -- which cancels the run and takes the
    junit report with it (ROCm/aorta#434). The calls most likely to wedge are
    the ones that build their own argv instead of going through ``_capture``.

    Three ways to leave that budget, and the guard rejects each separately. A
    ``timeout=`` literal is a *narrowed* budget, and it can appear on any call
    shape, so that check stays keyed on the keyword rather than the callee:
    ``from subprocess import run``, a ``Popen.communicate`` or an ``asyncio``
    wait would each escape a callee-shaped check while leaving the same hole.
    An omitted ``timeout=`` is an *absent* budget -- ``subprocess.run(argv)``
    is completely unbounded -- and nothing about the keyword can catch that, so
    that check does look at the callee, matching its trailing name so the
    import spelling still does not matter. It requires the keyword spelling:
    ``communicate`` and ``wait`` would also take the budget positionally, and
    asking for the keyword is what lets the guard see it. ``os.system`` and
    ``os.popen`` are the third -- no timeout exists to pass, so they are
    rejected outright.

    Checked from the CPU suite because the GPU legs are path-skipped on most
    PRs, so a regression here would otherwise reach `main` unobserved. Read
    from the AST rather than grepped so a reflowed call still matches.
    """
    source = Path(__file__).with_name("test_proton_smoke_gpu.py")
    found = _child_budget_violations(source.read_text(encoding="utf-8"), filename=str(source))
    narrowed, unbounded, unboundable = (
        found["narrowed"],
        found["unbounded"],
        found["unboundable"],
    )

    assert not narrowed, (
        f"{source.name} sets a per-call timeout instead of _CHILD_TIMEOUT_S at "
        f"{narrowed}. A wedged Proton payload would then run past the GPU job's "
        "own cap, which cancels the job and loses its report -- see ROCm/aorta#434."
    )
    assert not unbounded, (
        f"{source.name} waits on a child without a timeout= keyword at {unbounded}. "
        "Pass timeout=_CHILD_TIMEOUT_S, or launch through _capture, so a wedge "
        "fails this test's payload rather than the whole GPU job -- see "
        "ROCm/aorta#434."
    )
    assert not unboundable, (
        f"{source.name} launches a child that cannot be bounded at {unboundable}. "
        "os.system and os.popen take no timeout, so a wedge there runs until the "
        "GPU job's own cap. Use subprocess with timeout=_CHILD_TIMEOUT_S -- see "
        "ROCm/aorta#434."
    )
