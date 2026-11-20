"""CPU fakes only: these tests do not compile or execute GPU code."""

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

HELPER = (
    Path(__file__).resolve().parents[3]
    / ".agents/skills/rocm-sanitizer-triage/scripts/waitcheck_compile.py"
)
SPEC = importlib.util.spec_from_file_location("waitcheck_compile_test", HELPER)
compiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiler)


class JIT:
    arg_names = ["x", "N", "BLOCK"]

    def run(self, *args, **kwargs):
        raise AssertionError("Kernel must never run")


class Heuristics:
    def __init__(self, fn, values):
        self.fn, self.values = fn, values


class Autotuner:
    def __init__(self, fn):
        self.fn = fn


@pytest.fixture
def setup(tmp_path, monkeypatch):
    module = ModuleType("waitcheck_fixture")
    module.JIT, module.Heuristics, module.Autotuner = JIT, Heuristics, Autotuner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = tmp_path / "source.py"
    source.write_text("from waitcheck_fixture import JIT, Heuristics, Autotuner\nkernel = JIT()\n")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "signature": {"x": "*fp32", "N": "i32"},
                "constexprs": {"BLOCK": 128},
                "options": {"num_warps": 4},
            }
        )
    )
    calls = []

    def compile_fake(ast_source, target, options):
        calls.append((ast_source, target, options))
        return SimpleNamespace(asm={"hsaco": b"\x7fELF" + b"\0" * 60})

    api = SimpleNamespace(
        version="fake-3.6",
        JITFunction=JIT,
        Heuristics=Heuristics,
        Autotuner=Autotuner,
        ASTSource=lambda **kwargs: kwargs,
        GPUTarget=lambda *args: args,
        compile=compile_fake,
    )
    args = argparse.Namespace(
        source=str(source),
        config=str(config),
        kernel="kernel",
        arch="gfx950",
        warp_size=64,
        output=str(tmp_path / "out"),
        python_path=[],
    )
    return args, api, calls


def run(setup):
    args, api, _ = setup
    return compiler.build(args, backend_loader=lambda: api)


def test_jit_explicit_target_and_receipt_preserve_inputs(setup):
    args, _, calls = setup
    before = [Path(path).read_bytes() for path in (args.source, args.config)]
    record = run(setup)
    assert record["status"] == "built; not checked by Waitcheck"
    assert calls[0][1] == ("hip", "gfx950", 64)
    assert calls[0][2] == {"num_warps": 4}
    blob = (Path(args.output) / "kernel.hsaco").read_bytes()
    assert record["artifact"]["sha256"] == hashlib.sha256(blob).hexdigest()
    assert not record["source"]["changed"] and not record["config"]["changed"]
    assert before == [Path(path).read_bytes() for path in (args.source, args.config)]
    assert record["verification"]["native_entry_selectors"] == "unresolved"
    with pytest.raises(FileExistsError):
        run(setup)
    assert (Path(args.output) / "kernel.hsaco").read_bytes() == blob


def test_staticmethod_dotted_symbol(setup):
    args, _, _ = setup
    Path(args.source).write_text(
        "from waitcheck_fixture import JIT\nclass Owner:\n    kernel = staticmethod(JIT())\n"
    )
    args.kernel = "Owner.kernel"
    record = run(setup)
    assert record["status"].startswith("built;")
    assert record["wrappers"] == [{"type": "staticmethod"}]


def test_pointer_tuple_reaches_aot_grouped_ordered_and_preserves_config(setup):
    args, _, calls = setup
    config = json.loads(Path(args.config).read_text())
    config["signature"]["x"] = ["*fp32", "*i16"]
    Path(args.config).write_text(json.dumps(config))
    before = Path(args.config).read_bytes()
    record = run(setup)
    assert record["status"].startswith("built;")
    assert calls[0][0]["signature"]["x"] == ("*fp32", "*i16")
    assert list(calls[0][0]["signature"]) == ["x", "N"]
    assert record["build_config"]["signature"]["x"] == ["*fp32", "*i16"]
    assert Path(args.config).read_bytes() == before


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [None],
        [1],
        ["i32"],
        ["constexpr"],
        [["*fp32"]],
        {"member": "*fp32"},
        "",
        ["*"],
        ["*k"],
        ["*fp32 bad"],
    ],
)
def test_invalid_tuple_signature_fails_before_source_import(setup, bad):
    args, _, calls = setup
    config = json.loads(Path(args.config).read_text())
    config["signature"]["x"] = bad
    Path(args.config).write_text(json.dumps(config))
    Path(args.source).write_text("raise AssertionError('source must not be imported')\n")
    record = run(setup)
    assert record["status"] == "failed" and not calls
    assert record["error"]["type"] == "ValueError"
    assert "signature values" in record["error"]["message"]
    assert not (Path(args.output) / "kernel.hsaco").exists()


def test_heuristics_evaluate_in_order_with_concrete_arguments(setup):
    args, _, calls = setup
    Path(args.source).write_text(
        "from waitcheck_fixture import JIT, Heuristics\nkernel = Heuristics(JIT(), {'BLOCK': lambda a: a['N'] * 2, 'N': lambda a: a['BLOCK']})\n"
    )
    Path(args.config).write_text(
        json.dumps(
            {
                "signature": {"x": "*fp32"},
                "constexprs": {},
                "options": {},
                "heuristic_args": {"N": 64},
            }
        )
    )
    assert run(setup)["status"].startswith("built;")
    assert calls[0][0]["constexprs"] == {"BLOCK": 128, "N": 128}


@pytest.mark.parametrize("expression", ["a['missing']", "a['x'].dtype"])
def test_missing_or_tensor_dependent_heuristic_is_explicit_failure(setup, expression):
    args, _, calls = setup
    Path(args.source).write_text(
        "from waitcheck_fixture import JIT, Heuristics\nkernel = Heuristics(JIT(), {'BLOCK': lambda a: "
        + expression
        + "})\n"
    )
    record = run(setup)
    assert record["status"] == "failed" and not calls
    assert "concrete known arguments" in record["error"]["message"]
    assert "Traceback" in (Path(args.output) / "build.log").read_text()


@pytest.mark.parametrize(
    "definition, symbol, message",
    [
        ("kernel = Autotuner(JIT())", "kernel", "Autotuner"),
        ("kernel = JIT()", "missing", "missing"),
        ("kernel = type('Unknown', (), {'fn': JIT()})()", "kernel", "Unsupported kernel/wrapper"),
    ],
)
def test_unresolved_types_and_exact_symbol_never_fallback(setup, definition, symbol, message):
    args, _, calls = setup
    Path(args.source).write_text(
        "from waitcheck_fixture import JIT, Autotuner\n" + definition + "\n"
    )
    args.kernel = symbol
    record = run(setup)
    assert record["status"] == "failed" and not calls
    assert message in record["error"]["message"]


def test_missing_dependency_records_failure(setup):
    args, _, calls = setup
    Path(args.source).write_text("import missing_waitcheck_dependency_abc123\n")
    record = run(setup)
    assert record["error"]["type"] == "ModuleNotFoundError"
    assert not calls
    assert json.loads((Path(args.output) / "build.json").read_text())["status"] == "failed"


def test_nonfinite_constexpr_keeps_value_but_tags_manifest(setup):
    args, _, calls = setup
    Path(args.config).write_text(
        '{"signature":{"x":"*fp32","N":"i32"},"constexprs":{"BLOCK":-Infinity},"options":{}}'
    )
    record = run(setup)
    assert math.isinf(calls[0][0]["constexprs"]["BLOCK"])
    assert record["effective_constexprs"]["BLOCK"] == {"nonfinite": "-Infinity"}
    json.loads(
        (Path(args.output) / "build.json").read_text(),
        parse_constant=lambda token: pytest.fail(token),
    )


def test_source_change_during_import_marks_build_failed(setup):
    args, _, _ = setup
    with Path(args.source).open("a") as stream:
        stream.write("from pathlib import Path\nPath(__file__).write_text('# source changed')\n")
    record = run(setup)
    assert record["source"]["changed"]
    assert record["status"] == "failed"
    assert "Input bytes changed" in record["input_change_error"]


def test_bad_artifact_and_compiler_error_are_recorded(setup):
    args, api, _ = setup
    api.compile = lambda *a, **k: SimpleNamespace(asm={"hsaco": b"not ELF"})
    record = run(setup)
    assert record["status"] == "failed"
    assert "ELF-like" in record["error"]["message"]


def test_backend_unavailable_still_records_inputs_and_failure(setup):
    args, _, _ = setup

    def missing():
        raise ModuleNotFoundError("No module named triton")

    record = compiler.build(args, backend_loader=missing)
    assert record["status"] == "failed"
    assert record["triton_version"] == "unavailable; not loaded"
    assert record["error"]["type"] == "ModuleNotFoundError"
    assert record["config"]["sha256_before"] == record["config"]["sha256_after"]


def test_compiler_exception_records_log_and_no_artifact(setup):
    args, api, _ = setup

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic backend failure")

    api.compile = fail
    record = run(setup)
    assert record["status"] == "failed"
    assert record["error"]["message"] == "synthetic backend failure"
    assert not (Path(args.output) / "kernel.hsaco").exists()
    assert "synthetic backend failure" in (Path(args.output) / "build.log").read_text()


def test_explicit_dependency_path_and_process_settings_restored(setup, tmp_path):
    args, _, _ = setup
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    (dependency / "compile_fixture_dependency.py").write_text("KNOWN = 17\n")
    with Path(args.source).open("a") as stream:
        stream.write("from compile_fixture_dependency import KNOWN\nassert KNOWN == 17\n")
    args.python_path = [str(dependency)]
    before_path, before_bytecode = sys.path[:], sys.dont_write_bytecode
    try:
        record = run(setup)
        assert record["status"].startswith("built;")
        assert record["python_paths"] == [str(dependency.resolve())]
        assert not (dependency / "__pycache__").exists()
        assert sys.path == before_path and sys.dont_write_bytecode == before_bytecode
    finally:
        sys.modules.pop("compile_fixture_dependency", None)


@pytest.mark.parametrize("change", ["target", "missing_argument", "duplicate_key"])
def test_config_ambiguity_and_target_override_refused(setup, change):
    args, _, calls = setup
    path = Path(args.config)
    if change == "duplicate_key":
        path.write_text('{"signature":{},"signature":{},"constexprs":{},"options":{}}')
    else:
        config = json.loads(path.read_text())
        if change == "target":
            config["options"]["arch"] = "gfx1250"
        else:
            del config["signature"]["N"]
        path.write_text(json.dumps(config))
    assert run(setup)["status"] == "failed"
    assert not calls


def test_source_exit_zero_is_a_failed_build_with_context(setup):
    args, _, calls = setup
    Path(args.source).write_text("raise SystemExit(0)\n")
    record = run(setup)
    assert record["status"] == "failed" and not calls
    assert record["error"]["type"] == "SystemExit"
    assert "SystemExit: 0" in (Path(args.output) / "build.log").read_text()


@pytest.mark.parametrize("comparison", ["known_args", "heuristic_result"])
@pytest.mark.parametrize(
    "explicit, supplied",
    [
        (True, 1),
        (1, True),
        (128, 128.0),
        (128.0, 128),
        (0.0, -0.0),
        ([True], [1]),
        ({"nested": [128]}, {"nested": [128.0]}),
    ],
)
def test_constant_type_or_signed_zero_conflicts_fail_before_compile(
    setup, comparison, explicit, supplied
):
    args, _, calls = setup
    config = json.loads(Path(args.config).read_text())
    config["constexprs"]["BLOCK"] = explicit
    if comparison == "known_args":
        config["heuristic_args"] = {"BLOCK": supplied}
    else:
        Path(args.source).write_text(
            "from waitcheck_fixture import JIT, Heuristics\n"
            "kernel = Heuristics(JIT(), {'BLOCK': lambda args: " + repr(supplied) + "})\n"
        )
    Path(args.config).write_text(json.dumps(config))
    before = [Path(path).read_bytes() for path in (args.source, args.config)]
    record = run(setup)
    assert record["status"] == "failed" and not calls
    assert "conflict" in record["error"]["message"].lower()
    assert before == [Path(path).read_bytes() for path in (args.source, args.config)]


@pytest.mark.parametrize("comparison", ["known_args", "heuristic_result"])
@pytest.mark.parametrize("constant", [True, 128, 128.0, -0.0, {"nested": [True, 128.0]}])
def test_exact_constant_types_and_values_still_build(setup, comparison, constant):
    args, _, calls = setup
    config = json.loads(Path(args.config).read_text())
    config["constexprs"]["BLOCK"] = constant
    if comparison == "known_args":
        config["heuristic_args"] = {"BLOCK": constant}
    else:
        Path(args.source).write_text(
            "from waitcheck_fixture import JIT, Heuristics\n"
            "kernel = Heuristics(JIT(), {'BLOCK': lambda args: " + repr(constant) + "})\n"
        )
    Path(args.config).write_text(json.dumps(config))
    record = run(setup)
    assert record["status"].startswith("built;")
    actual = calls[0][0]["constexprs"]["BLOCK"]
    assert type(actual) is type(constant)
    if type(constant) is float:
        assert struct.pack("!d", actual) == struct.pack("!d", constant)
    else:
        assert actual == constant
    assert not record["source"]["changed"] and not record["config"]["changed"]


def test_heuristic_tuple_does_not_replace_explicit_list(setup):
    args, _, calls = setup
    config = json.loads(Path(args.config).read_text())
    config["constexprs"]["BLOCK"] = [128]
    Path(args.config).write_text(json.dumps(config))
    Path(args.source).write_text(
        "from waitcheck_fixture import JIT, Heuristics\n"
        "kernel = Heuristics(JIT(), {'BLOCK': lambda args: (128,)})\n"
    )
    assert run(setup)["status"] == "failed"
    assert not calls


class DriverConfig:
    def __init__(self, previous):
        self._active = previous
        self.default_resolutions = 0

    @property
    def active(self):
        if self._active is None:
            self.default_resolutions += 1
            raise AssertionError("Lazy hardware driver must not be resolved")
        return self._active

    def set_active(self, value):
        self._active = value


def target_context_setup(setup, previous=None):
    args, api, _ = setup
    args.source_target = "compile"
    api.DriverBase = object
    api.driver_config = DriverConfig(previous)
    fixture = sys.modules["waitcheck_fixture"]
    fixture.driver, fixture.seen = api.driver_config, []
    Path(args.source).write_text(
        "from waitcheck_fixture import JIT, Heuristics, driver, seen\n"
        "seen.append(driver.active.get_current_target())\n"
        "kernel = JIT()\n"
    )
    return api.driver_config, fixture.seen


@pytest.mark.parametrize("previous", [None, object()])
def test_compile_target_precedes_import_and_restores_without_hardware(setup, previous):
    args, _, calls = setup
    driver, seen = target_context_setup(setup, previous)
    before = Path(args.source).read_bytes()
    record = run(setup)
    assert record["status"].startswith("built;")
    assert seen == [calls[0][1]] == [("hip", "gfx950", 64)]
    assert driver._active is previous and driver.default_resolutions == 0
    assert record["source_target_context"]["observed_target_queries"] == 1
    assert record["source_target_context"]["previous_active_restored"] is True
    assert Path(args.source).read_bytes() == before


def test_target_context_covers_heuristic_and_compile(setup):
    args, api, _ = setup
    driver, seen = target_context_setup(setup)
    Path(args.source).write_text(
        "from waitcheck_fixture import JIT, Heuristics, driver, seen\n"
        "seen.append(driver.active.get_current_target())\n"
        "kernel = Heuristics(JIT(), {'BLOCK': lambda _: 128 if "
        "driver.active.get_current_target()[1] == 'gfx950' else 64})\n"
    )
    original_compile = api.compile

    def compile_checked(ast_source, target, options):
        seen.append(driver.active.get_current_target())
        return original_compile(ast_source, target, options)

    api.compile = compile_checked
    record = run(setup)
    assert record["status"].startswith("built;")
    assert seen == [("hip", "gfx950", 64)] * 2
    assert record["source_target_context"]["observed_target_queries"] == 3
    assert driver._active is None and driver.default_resolutions == 0


@pytest.mark.parametrize("stage", ["import", "compile"])
def test_target_context_restores_after_failed_stages(setup, stage):
    args, api, _ = setup
    previous = object()
    driver, _ = target_context_setup(setup, previous)
    if stage == "import":
        with Path(args.source).open("a") as stream:
            stream.write("raise RuntimeError('source failure')\n")
    else:

        def broken_compile(*args, **kwargs):
            raise RuntimeError("compile failure")

        api.compile = broken_compile
    record = run(setup)
    assert record["status"] == "failed"
    assert driver._active is previous and driver.default_resolutions == 0
    assert record["source_target_context"]["previous_active_restored"] is True
    assert not (Path(args.output) / "kernel.hsaco").exists()


@pytest.mark.parametrize("catch", [False, True])
def test_runtime_request_is_refused_even_when_source_catches_it(setup, catch):
    args, _, calls = setup
    driver, _ = target_context_setup(setup)
    body = "driver.active.get_current_device()\n"
    if catch:
        body = "try:\n    driver.active.get_current_device()\nexcept RuntimeError:\n    pass\n"
    with Path(args.source).open("a") as stream:
        stream.write(body)
    record = run(setup)
    assert record["status"] == "failed" and not calls
    assert record["source_target_context"]["refused_requests"] == ["get_current_device"]
    assert driver._active is None and driver.default_resolutions == 0


def test_source_replacing_provider_is_refused_and_previous_restored(setup):
    args, _, calls = setup
    previous = object()
    driver, _ = target_context_setup(setup, previous)
    with Path(args.source).open("a") as stream:
        stream.write("driver.set_active(object())\n")
    record = run(setup)
    assert record["status"] == "failed" and not calls
    assert driver._active is previous


def test_restoration_failure_invalidates_compiler_success(setup):
    args, api, calls = setup
    previous = object()
    target_context_setup(setup, previous)

    class RefusesRestore(DriverConfig):
        def set_active(self, value):
            self._active = object() if value is previous else value

    driver = api.driver_config = RefusesRestore(previous)
    sys.modules["waitcheck_fixture"].driver = driver
    record = run(setup)
    assert record["status"] == "failed" and len(calls) == 1
    assert record["source_target_context"]["previous_active_restored"] is False
    assert not (Path(args.output) / "kernel.hsaco").exists()


def test_unsupported_target_api_fails_before_source_import(setup):
    args, api, calls = setup
    _, seen = target_context_setup(setup)
    api.driver_config = SimpleNamespace(set_active=lambda value: None)
    record = run(setup)
    assert record["status"] == "failed" and not seen and not calls
    assert "Unsupported Triton DriverConfig" in record["error"]["message"]


def test_partial_provider_install_failure_restores_before_source_import(setup):
    _, api, calls = setup
    previous = object()
    _, seen = target_context_setup(setup, previous)

    class PartialInstall(DriverConfig):
        def set_active(self, value):
            self._active = value
            if value is not previous:
                raise RuntimeError("provider install failure")

    driver = api.driver_config = PartialInstall(previous)
    sys.modules["waitcheck_fixture"].driver = driver
    record = run(setup)
    assert record["status"] == "failed" and not seen and not calls
    assert driver._active is previous and driver.default_resolutions == 0
    assert record["source_target_context"]["previous_active_restored"] is True


def test_public_active_ignoring_stored_provider_is_refused_before_import(setup):
    _, api, calls = setup
    previous = SimpleNamespace(get_current_target=lambda: ("hip", "gfx942", 64))
    _, seen = target_context_setup(setup, previous)

    class IgnoresStoredActive(DriverConfig):
        @property
        def active(self):
            return previous

    driver = api.driver_config = IgnoresStoredActive(previous)
    sys.modules["waitcheck_fixture"].driver = driver
    record = run(setup)
    assert record["status"] == "failed" and not seen and not calls
    assert "Public Triton active provider" in record["error"]["message"]
    assert driver._active is previous and driver.default_resolutions == 0
    assert record["source_target_context"]["previous_active_restored"] is True


def test_default_mode_keeps_existing_runtime_target(setup):
    args, _, calls = setup
    host = SimpleNamespace(get_current_target=lambda: ("hip", "gfx942", 64))
    driver, seen = target_context_setup(setup, host)
    args.source_target = "runtime"
    record = run(setup)
    assert record["status"].startswith("built;")
    assert seen == [("hip", "gfx942", 64)]
    assert calls[0][1] == ("hip", "gfx950", 64)
    assert driver._active is host
    assert record["source_target_context"] == {"mode": "runtime", "runtime_target": "not observed"}
