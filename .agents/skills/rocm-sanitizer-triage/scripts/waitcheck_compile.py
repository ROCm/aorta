"""Prepare one explicitly configured Triton HIP object for later Waitcheck use.

Importing source and evaluating heuristics executes Python, without a sandbox.
Imports may probe hardware or have other side effects. This helper does not call
kernel.run, tune, benchmark, or allocate tensors. Bound the process externally.
Supports JITFunction, staticmethod and concrete Heuristics; refuses Autotuner.
API basis: triton-lang/triton v3.6.0 compiler/compiler.py and runtime/autotuner.py.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import re
import struct
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_value(value):
    """Keep legitimate non-finite constants in memory; tag them only in receipts."""
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "nonfinite": "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        }
    if isinstance(value, dict):
        return {str(key): manifest_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [manifest_value(child) for child in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("Unsupported concrete value type: " + type(value).__name__)


def same_constant(left, right):
    """Compare specialization values without Python's numeric type coercion."""
    if type(left) is not type(right):
        return False
    if type(left) is float:
        # IEEE bits preserve signed zero and non-finite values, including NaN payloads.
        return struct.pack("!d", left) == struct.pack("!d", right)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            same_constant(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is dict:
        return len(left) == len(right) and all(
            any(
                same_constant(key, other_key) and same_constant(child, other_child)
                for other_key, other_child in right.items()
            )
            for key, child in left.items()
        )
    if left is None or type(left) in (str, int, bool):
        return left == right
    raise TypeError("Unsupported specialization constant type: " + type(left).__name__)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate config key: " + key)
        result[key] = value
    return result


def compiler_signature(signature):
    """Decode flat JSON pointer-type arrays as single ordered tuple arguments."""
    result = {}
    for name, value in signature.items():
        if isinstance(value, str) and value:
            result[name] = value
        elif (
            isinstance(value, list)
            and value
            and all(
                isinstance(member, str)
                and member.startswith("*")
                and member not in ("*", "*k")
                and not any(char.isspace() for char in member)
                for member in value
            )
        ):
            result[name] = tuple(value)
        else:
            raise ValueError(
                "signature values must be nonempty Triton type strings or nonempty flat pointer-type arrays"
            )
    return result


def read_config(path):
    config = json.loads(path.read_bytes(), object_pairs_hook=unique_object)
    allowed = {"signature", "constexprs", "options", "heuristic_args"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Config must contain only signature, constexprs, options, heuristic_args")
    for key in ("signature", "constexprs", "options", "heuristic_args"):
        if key not in config and key == "heuristic_args":
            config[key] = {}
        if not isinstance(config.get(key), dict):
            raise ValueError("Config requires an object: " + key)
    compiler_signature(config["signature"])  # Validate before importing source.
    if set(config["options"]) & {"arch", "warp_size", "target", "backend", "ir_override"}:
        raise ValueError("Options cannot override the explicit target or source IR")
    return config


def load_triton():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.driver import DriverBase
    from triton.compiler import ASTSource
    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.driver import driver
    from triton.runtime.jit import JITFunction

    return SimpleNamespace(
        compile=triton.compile,
        version=triton.__version__,
        GPUTarget=GPUTarget,
        ASTSource=ASTSource,
        Autotuner=Autotuner,
        Heuristics=Heuristics,
        JITFunction=JITFunction,
        DriverBase=DriverBase,
        driver_config=driver,
    )


@contextlib.contextmanager
def source_target_context(api, target, mode, record):
    """Opt-in target queries for offline imports; never pretend a GPU is present.

    DriverConfig.set_active and its stored _active slot are checked against the
    inspected Triton v3.6.0 contract. Do not read .active/.default to save state:
    those properties can initialize a hardware driver. Other versions must pass
    this guard and native validation; this context is not a Python sandbox.
    """
    if mode not in ("runtime", "compile"):
        raise ValueError("source_target must be runtime or compile")
    if mode == "runtime":
        record["source_target_context"] = {"mode": mode, "runtime_target": "not observed"}
        yield lambda: None
        return
    details = {
        "mode": mode,
        "target": record["requested_target"],
        "observed_target_queries": 0,
        "refused_requests": [],
        "previous_active_restored": False,
    }
    record["source_target_context"] = details
    driver = getattr(api, "driver_config", None)
    base = getattr(api, "DriverBase", None)
    if driver is None or base is None or not callable(getattr(driver, "set_active", None)):
        raise RuntimeError("Triton lacks the required compile-only target context API")
    state = vars(driver)
    if "_active" not in state:
        raise RuntimeError("Unsupported Triton DriverConfig state; source not imported")
    previous = state["_active"]

    def refuse(name):
        details["refused_requests"].append(name)
        raise RuntimeError(
            "Compile-only target context cannot provide GPU runtime operation: " + name
        )

    class CompileTargetDriver(base):
        def __init__(self):
            pass  # No hardware-driver initialization.

        @staticmethod
        def is_active():
            return False  # No device availability is asserted.

        def get_current_target(self):
            details["observed_target_queries"] += 1
            return target

        def map_python_to_cpp_type(self, *args, **kwargs):
            return refuse("map_python_to_cpp_type")

        def get_active_torch_device(self, *args, **kwargs):
            return refuse("get_active_torch_device")

        def get_benchmarker(self, *args, **kwargs):
            return refuse("get_benchmarker")

        def __getattr__(self, name):
            return refuse(name)

    provider = CompileTargetDriver()

    def ensure_selected():
        if details["refused_requests"]:
            raise RuntimeError("Source or compiler attempted unsupported GPU runtime operations")
        if vars(driver).get("_active") is not provider:
            raise RuntimeError("Source or compiler changed the selected import target provider")
        if driver.active is not provider:
            raise RuntimeError(
                "Public Triton active provider differs from the selected import target"
            )

    try:
        driver.set_active(provider)
        ensure_selected()
        yield ensure_selected
        ensure_selected()
    finally:
        driver.set_active(previous)
        details["previous_active_restored"] = vars(driver).get("_active", object()) is previous
        if not details["previous_active_restored"]:
            raise RuntimeError("Could not restore the previous active Triton target provider")


def exact_symbol(module, symbol):
    value = module
    for component in symbol.split("."):
        if not component.isidentifier():
            raise ValueError("Kernel must be an exact dotted Python symbol")
        value = inspect.getattr_static(value, component)
    return value


def unwrap(value, config, api, wrappers):
    constants = dict(config["constexprs"])
    known = dict(config["heuristic_args"])
    for key, child in constants.items():
        if key in known and not same_constant(known[key], child):
            raise ValueError("Conflicting heuristic argument/constexpr: " + key)
        known[key] = child
    visited = set()
    while not isinstance(value, api.JITFunction):
        if id(value) in visited:
            raise ValueError("Cyclic kernel wrapper")
        visited.add(id(value))
        if isinstance(value, staticmethod):
            wrappers.append({"type": "staticmethod"})
            value = value.__func__
        elif isinstance(value, api.Autotuner):
            wrappers.append({"type": "Autotuner", "status": "refused"})
            raise TypeError(
                "Unresolved Triton Autotuner is unsupported; no tuning or hooks were run"
            )
        elif isinstance(value, api.Heuristics):
            selected = {"type": "Heuristics", "values": {}}
            wrappers.append(selected)
            for key, heuristic in value.values.items():
                try:
                    result = heuristic(dict(known))
                    manifest_value(result)  # Reject tensors and other unresolved objects.
                except Exception as error:
                    raise ValueError(
                        f"Heuristic {key} requires concrete known arguments; tensor-dependent/missing values are unsupported: {error}"
                    ) from error
                if key in constants and not same_constant(constants[key], result):
                    raise ValueError("Heuristic conflicts with explicit constexpr: " + key)
                constants[key] = known[key] = result
                selected["values"][key] = manifest_value(result)
            value = value.fn
        else:
            raise TypeError("Unsupported kernel/wrapper type: " + type(value).__name__)
    names = set(value.arg_names)
    if (set(config["signature"]) | set(constants)) - names:
        raise ValueError("Signature/constexpr names are absent from the requested JIT function")
    if names - set(config["signature"]) - set(constants):
        raise ValueError("Every JIT argument needs an explicit signature or constexpr")
    return value, constants


def build(args, backend_loader=None):
    source, config_path = Path(args.source).resolve(), Path(args.config).resolve()
    output = Path(args.output).absolute()
    output.mkdir()  # Must be fresh; never merge or overwrite an earlier attempt.
    record = {
        "status": "failed",
        "utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "interpreter": sys.executable,
        "python": sys.version,
        "helper_sha256": file_hash(Path(__file__)),
        "kernel_symbol": args.kernel,
        "requested_target": {"backend": "hip", "arch": args.arch, "warp_size": args.warp_size},
        "source": {"path": str(source)},
        "config": {"path": str(config_path)},
        "python_paths": list(args.python_path),
        "wrappers": [],
        "triton_version": "unavailable; not loaded",
        "verification": {
            "object_target": "not independently inspected",
            "native_entry_selectors": "unresolved",
            "waitcheck": "not checked",
        },
        "limitations": [
            "Source imports/heuristics execute Python and may probe hardware or have side effects.",
            "Only source/config input bytes are hashed; dependencies are not pinned.",
            "Requested arch/wave recorded; no complete architecture compatibility validation.",
            "Log captures Python stdout/stderr, not every native/subprocess file descriptor.",
        ],
    }
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    module_name = "_waitcheck_source_" + hashlib.sha256(str(source).encode()).hexdigest()
    previous = sys.modules.get(module_name)
    try:
        with (
            (output / "build.log").open("x", encoding="utf-8") as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            try:
                for label, path in (("source", source), ("config", config_path)):
                    record[label]["sha256_before"] = file_hash(path)
                if not re.fullmatch(r"gfx[0-9a-f]+", args.arch) or args.warp_size not in (32, 64):
                    raise ValueError("Supply a gfx target and explicit warp size 32 or 64")
                config = read_config(config_path)
                record["build_config"] = manifest_value(config)
                dependencies = [str(Path(path).resolve(strict=True)) for path in args.python_path]
                if any(not Path(path).is_dir() for path in dependencies):
                    raise ValueError("Each --python-path must be a directory")
                record["python_paths"] = dependencies
                sys.path[:0] = dependencies
                sys.dont_write_bytecode = True
                api = (backend_loader or load_triton)()
                record["triton_version"] = api.version
                target = api.GPUTarget("hip", args.arch, args.warp_size)
                with source_target_context(
                    api, target, getattr(args, "source_target", "runtime"), record
                ) as ensure_target:
                    spec = importlib.util.spec_from_file_location(module_name, source)
                    if spec is None or spec.loader is None:
                        raise ValueError("Source must be an importable Python file")
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    ensure_target()
                    fn, constants = unwrap(
                        exact_symbol(module, args.kernel), config, api, record["wrappers"]
                    )
                    ensure_target()
                    record["effective_constexprs"] = manifest_value(constants)
                    compiled = api.compile(
                        api.ASTSource(
                            fn=fn,
                            signature=compiler_signature(config["signature"]),
                            constexprs=constants,
                        ),
                        target=target,
                        options=config["options"],
                    )
                    blob = compiled.asm.get("hsaco")
                    if (
                        not isinstance(blob, bytes)
                        or len(blob) < 16
                        or not blob.startswith(b"\x7fELF")
                    ):
                        raise ValueError("Compiler returned no nonempty ELF-like hsaco bytes")
                with (output / "kernel.hsaco").open("xb") as stream:
                    stream.write(blob)
                record["artifact"] = {
                    "path": "kernel.hsaco",
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "bytes": len(blob),
                }
                record["status"] = "built; not checked by Waitcheck"
            except BaseException as error:
                record["error"] = {"type": type(error).__name__, "message": str(error)}
                traceback.print_exc()
                if not isinstance(error, (Exception, SystemExit)):
                    raise
    finally:
        sys.path[:], sys.dont_write_bytecode = old_path, old_bytecode
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        for label, path in (("source", source), ("config", config_path)):
            try:
                record[label]["sha256_after"] = file_hash(path)
            except OSError as error:
                record[label]["after_error"] = str(error)
            record[label]["changed"] = record[label].get("sha256_before") != record[label].get(
                "sha256_after"
            )
        if any(record[label]["changed"] for label in ("source", "config")):
            record["status"] = "failed"
            record["input_change_error"] = (
                "Input bytes changed during the attempt; artifact is not accepted"
            )
        with (output / "build.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "kernel", "config", "arch", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--warp-size", required=True, type=int, choices=(32, 64))
    parser.add_argument("--python-path", action="append", default=[])
    parser.add_argument(
        "--source-target",
        choices=("runtime", "compile"),
        default="runtime",
        help="compile: supply only the chosen target to Triton queries during source import/AOT; no GPU runtime",
    )
    args = parser.parse_args(argv)
    try:
        result = build(args)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(str(Path(args.output) / "build.json"))
    return 0 if result["status"].startswith("built;") else 1


if __name__ == "__main__":
    sys.exit(main())
