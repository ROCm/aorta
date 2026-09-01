"""Recipe schema and execution for ``mode: sanitizer`` runs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from aorta.triage.recipe import RecipeSchemaError, load_recipe_mapping

from .consan import resolve_consan_hook
from .models import CheckResult, ExecutionState, SelectionRequirement, Verdict
from .pipeline import DEFAULT_TIMEOUT_SECONDS, run_sanitizers
from .report import build_report, write_report
from .selection import (
    observations_from_consan_repro,
    observations_from_dispatch_csv,
    observations_from_gemm_csv,
    observations_from_kernel_list,
    observations_from_rocprof_trace,
    select_kernels,
    sha256_code_object,
)

_SANITIZER_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "mode",
        "ticket",
        "description",
        "sanitizer_plan",
    }
)
_DEFERRED_SOURCE_KINDS = frozenset({"workload", "model", "module"})
_SUPPORTED_SOURCE_KINDS = frozenset(
    {"gemm_csv", "dispatch_csv", "rocprof_trace", "consan_repro", "kernel_list", "kernel"}
)
_SUPPORTED_REQUIREMENTS = frozenset({"top_time", "top_dispatch_count"})
_SUPPORTED_SANITIZERS = frozenset({"waitcheck", "consan"})
_SUPPORTED_SCOPES = frozenset({"kernel"})
_SUPPORTED_CONSAN_POLICIES = frozenset({"strict", "lenient"})
_SUPPORTED_MISSING_BACKEND = frozenset({"fail"})
_KERNEL_SPEC_FIELDS = frozenset(
    {"name", "code_object", "code_object_sha256", "code_object_index", "entry_offset"}
)


@dataclass(frozen=True)
class KernelSourceSpec:
    """A fully-specified kernel identity from a ``kernel`` / ``kernel_list`` source.

    Preserves the code-object path, SHA-256, image index, and entry offset that
    exact-entry Waitcheck needs, rather than collapsing the source to a name.
    """

    name: str
    code_object: str | None = None
    code_object_sha256: str | None = None
    code_object_index: int | None = None
    entry_offset: int | None = None

    def to_entry(self) -> dict[str, object]:
        entry: dict[str, object] = {"name": self.name}
        if self.code_object is not None:
            entry["code_object"] = self.code_object
        if self.code_object_sha256 is not None:
            entry["code_object_sha256"] = self.code_object_sha256
        if self.code_object_index is not None:
            entry["code_object_index"] = self.code_object_index
        if self.entry_offset is not None:
            entry["entry_offset"] = self.entry_offset
        return entry


def _resolve_kernel_object(
    spec: KernelSourceSpec, *, recipe_path: Path
) -> KernelSourceSpec:
    """Resolve a ``kind: kernel`` spec's ``code_object`` and auto-hash it at load.

    A ``kernel`` / ``kernel_list`` source names a code object by path rather than
    carrying the runtime digest (the object is rebuilt per run, so its SHA is
    host-specific and must not be committed). We resolve that path relative to the
    recipe file's directory -- so a lane runs identically from CI's repo root or
    standalone -- and, when no explicit ``code_object_sha256`` was given, hash the
    resolved object at load time so Waitcheck can build a whole-object-scan
    identity (mirrors ``observations_from_gemm_csv``).

    An explicit digest is preserved verbatim (a real file mismatch still trips
    Waitcheck's existing ``code_object_digest_mismatch`` fail-closed check). A
    missing / unreadable object degrades gracefully: the digest stays absent and
    the lane fails closed (``code_object_identity_required``) exactly as before.
    Harmless for ConSan, which never digests the object.
    """

    if spec.code_object is None:
        return spec
    resolved = _resolve_path(spec.code_object, recipe_path=recipe_path)
    sha256 = spec.code_object_sha256
    if sha256 is None:
        sha256 = sha256_code_object(resolved)
    return replace(spec, code_object=str(resolved), code_object_sha256=sha256)


def _identity_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise RecipeSchemaError(f"{field} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        base = 16 if text.lower().startswith("0x") else 10
        try:
            result = int(text, base)
        except ValueError as exc:
            raise RecipeSchemaError(f"{field} must be an integer") from exc
    else:
        raise RecipeSchemaError(f"{field} must be an integer")
    if result < 0:
        raise RecipeSchemaError(f"{field} must be non-negative")
    return result


def _parse_kernel_spec(entry: object, *, context: str) -> KernelSourceSpec:
    if isinstance(entry, str):
        name = entry.strip()
        if not name:
            raise RecipeSchemaError(f"{context} name must be a non-empty string")
        return KernelSourceSpec(name=name)
    if not isinstance(entry, dict) or not all(isinstance(key, str) for key in entry):
        raise RecipeSchemaError(f"{context} must be a string or an object")
    unknown = set(entry) - _KERNEL_SPEC_FIELDS
    if unknown:
        raise RecipeSchemaError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RecipeSchemaError(f"{context}.name must be a non-empty string")
    code_object = entry.get("code_object")
    if code_object is not None and not isinstance(code_object, str):
        raise RecipeSchemaError(f"{context}.code_object must be a string")
    sha256 = entry.get("code_object_sha256")
    if sha256 is not None and not isinstance(sha256, str):
        raise RecipeSchemaError(f"{context}.code_object_sha256 must be a string")
    index = entry.get("code_object_index")
    offset = entry.get("entry_offset")
    return KernelSourceSpec(
        name=name.strip(),
        code_object=code_object,
        code_object_sha256=sha256,
        code_object_index=(
            None if index is None else _identity_int(index, field=f"{context}.code_object_index")
        ),
        entry_offset=(
            None if offset is None else _identity_int(offset, field=f"{context}.entry_offset")
        ),
    )


@dataclass(frozen=True)
class SanitizerRecipe:
    ticket: str
    target: str
    source_kind: str
    source_path: Path | None
    isa_dir: Path | None
    consan_command: Path | None
    consan_log: bool
    scope_kind: str
    requirement: SelectionRequirement
    top_n: int
    sanitizers: tuple[str, ...]
    consan_policy: str
    on_missing_backend: str
    report_name: str
    repro_variant: str | None = None
    kernel_specs: tuple[KernelSourceSpec, ...] = ()
    timeout_seconds: float | None = None

    @property
    def recipe_dir(self) -> Path | None:
        if self.source_path is None:
            return None
        return self.source_path.parent


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RecipeSchemaError(f"{name} must be an object")
    return value


def _require_str(block: Mapping[str, object], key: str) -> str:
    value = block.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RecipeSchemaError(f"{key} must be a non-empty string")
    return value.strip()


def _resolve_path(raw: str, *, recipe_path: Path) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (recipe_path.parent / candidate).resolve()


def _optional_timeout_seconds(block: Mapping[str, object]) -> float | None:
    """Parse an optional positive ``timeout_seconds`` (seconds) from a block.

    Absent keeps the pipeline default. A bool, non-number, or non-positive value
    is rejected so a malformed knob fails at load rather than silently disabling
    the ceiling.
    """

    if "timeout_seconds" not in block:
        return None
    value = block.get("timeout_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeSchemaError(
            "sanitizer_plan.policy.timeout_seconds must be a positive number"
        )
    if value <= 0:
        raise RecipeSchemaError(
            "sanitizer_plan.policy.timeout_seconds must be a positive number"
        )
    return float(value)


def load_sanitizer_recipe(path: Path) -> SanitizerRecipe:
    """Load and validate a ``mode: sanitizer`` recipe."""

    data = load_recipe_mapping(path)
    if not isinstance(data, dict):
        raise RecipeSchemaError("recipe root must be an object")
    unknown = set(data) - _SANITIZER_TOP_LEVEL
    if unknown:
        raise RecipeSchemaError(f"unknown top-level keys: {', '.join(sorted(unknown))}")
    if data.get("mode") != "sanitizer":
        raise RecipeSchemaError("mode must be 'sanitizer'")
    if data.get("schema_version") != 1:
        raise RecipeSchemaError("schema_version must be 1")

    plan = _require_mapping(data.get("sanitizer_plan"), name="sanitizer_plan")
    target = _require_str(plan, "target")
    source = _require_mapping(plan.get("source"), name="sanitizer_plan.source")
    source_kind = _require_str(source, "kind")
    if source_kind in _DEFERRED_SOURCE_KINDS:
        raise RecipeSchemaError(
            f"sanitizer_plan.source.kind={source_kind!r} is not supported yet"
        )
    if source_kind not in _SUPPORTED_SOURCE_KINDS:
        raise RecipeSchemaError(f"unsupported sanitizer_plan.source.kind={source_kind!r}")

    scope = _require_mapping(plan.get("scope"), name="sanitizer_plan.scope")
    scope_kind = _require_str(scope, "kind")
    if scope_kind not in _SUPPORTED_SCOPES:
        raise RecipeSchemaError(f"unsupported sanitizer_plan.scope.kind={scope_kind!r}")
    selection = _require_mapping(plan.get("selection"), name="sanitizer_plan.selection")
    requirement_raw = _require_str(selection, "requirement")
    if requirement_raw not in _SUPPORTED_REQUIREMENTS:
        raise RecipeSchemaError(f"unsupported selection.requirement={requirement_raw!r}")
    top_n = selection.get("top_n")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise RecipeSchemaError("sanitizer_plan.selection.top_n must be a positive integer")

    sanitizers_raw = plan.get("sanitizers")
    if not isinstance(sanitizers_raw, list) or not sanitizers_raw:
        raise RecipeSchemaError("sanitizer_plan.sanitizers must be a non-empty list")
    sanitizers = tuple(dict.fromkeys(str(item) for item in sanitizers_raw))
    unknown_sanitizers = sorted(set(sanitizers) - _SUPPORTED_SANITIZERS)
    if unknown_sanitizers:
        raise RecipeSchemaError(f"unsupported sanitizers: {unknown_sanitizers}")

    policy = _require_mapping(plan.get("policy"), name="sanitizer_plan.policy")
    consan_policy = _require_str(policy, "consan_policy")
    if consan_policy not in _SUPPORTED_CONSAN_POLICIES:
        raise RecipeSchemaError(
            f"unsupported sanitizer_plan.policy.consan_policy={consan_policy!r}"
        )
    on_missing_backend = _require_str(policy, "on_missing_backend")
    if on_missing_backend not in _SUPPORTED_MISSING_BACKEND:
        raise RecipeSchemaError(
            f"unsupported sanitizer_plan.policy.on_missing_backend={on_missing_backend!r}"
        )
    timeout_seconds = _optional_timeout_seconds(policy)
    output = _require_mapping(plan.get("output"), name="sanitizer_plan.output")
    report_name = _require_str(output, "report")

    source_path: Path | None = None
    isa_dir: Path | None = None
    consan_command: Path | None = None
    consan_log_raw = source.get("consan_log", True)
    if not isinstance(consan_log_raw, bool):
        raise RecipeSchemaError("sanitizer_plan.source.consan_log must be a boolean")
    consan_log = consan_log_raw
    repro_variant: str | None = None
    kernel_specs: tuple[KernelSourceSpec, ...] = ()

    if source_kind in {"gemm_csv", "dispatch_csv", "rocprof_trace"}:
        source_path = _resolve_path(_require_str(source, "path"), recipe_path=path)
    elif source_kind == "consan_repro":
        source_path = _resolve_path(_require_str(source, "hip"), recipe_path=path)
        repro_variant = _require_str(source, "variant")
    elif source_kind == "kernel_list":
        names = source.get("kernels")
        if not isinstance(names, list) or not names:
            raise RecipeSchemaError("sanitizer_plan.source.kernels must be a non-empty list")
        kernel_specs = tuple(
            _parse_kernel_spec(item, context=f"sanitizer_plan.source.kernels[{index}]")
            for index, item in enumerate(names)
        )
    elif source_kind == "kernel":
        kernel = _require_mapping(source.get("kernel"), name="sanitizer_plan.source.kernel")
        kernel_specs = (_parse_kernel_spec(kernel, context="sanitizer_plan.source.kernel"),)

<<<<<<< Updated upstream
    if source_kind in {"kernel", "kernel_list"}:
        # Resolve each kernel's code object relative to the recipe and auto-hash it
        # (see _resolve_kernel_object) so a digest-less waitcheck lane can run.
        kernel_specs = tuple(
            _resolve_kernel_object(spec, recipe_path=path) for spec in kernel_specs
        )

    # For every resolvable (non-repro) source kind an optional
    # ``source.consan_command`` points ConSan at a caller-supplied command or
    # code-object loader (e.g. the consan_app.py HIP loader), resolved relative
    # to the recipe file. Absent, the non-repro kinds keep today's
    # ``not_checked`` behavior; the ``consan_repro`` kind uses ``command``.
    if source_kind != "consan_repro" and "consan_command" in source:
        consan_command = _resolve_path(_require_str(source, "consan_command"), recipe_path=path)
=======
    # ConSan executes a program rather than a kernel, so every source kind that can
    # name a kernel also needs a way to say which binary dispatches it. Gating this
    # to consan_repro limited ConSan to the two built-in repro variants and left
    # kernel/kernel_list recipes with no runnable command at all.
    if "command" in source:
        consan_command = _resolve_path(_require_str(source, "command"), recipe_path=path)
>>>>>>> Stashed changes

    if "isa_dir" in source:
        isa_dir = _resolve_path(_require_str(source, "isa_dir"), recipe_path=path)
    if source_kind == "gemm_csv" and isa_dir is None:
        raise RecipeSchemaError("sanitizer_plan.source.isa_dir is required for gemm_csv")

    ticket = _require_str(data, "ticket") if "ticket" in data else path.stem
    return SanitizerRecipe(
        ticket=ticket,
        target=target,
        source_kind=source_kind,
        source_path=source_path,
        isa_dir=isa_dir,
        consan_command=consan_command,
        consan_log=consan_log,
        scope_kind=scope_kind,
        requirement=SelectionRequirement(requirement_raw),
        top_n=top_n,
        sanitizers=sanitizers,
        consan_policy=consan_policy,
        on_missing_backend=on_missing_backend,
        report_name=report_name,
        repro_variant=repro_variant,
        kernel_specs=kernel_specs,
        timeout_seconds=timeout_seconds,
    )


def _resolve_observations(recipe: SanitizerRecipe) -> tuple:
    target = recipe.target
    if recipe.source_kind == "gemm_csv":
        if recipe.source_path is None or recipe.isa_dir is None:
            raise ValueError("gemm_csv source requires path and isa_dir")
        return observations_from_gemm_csv(
            recipe.source_path,
            target=target,
            isa_dir=recipe.isa_dir,
        )
    if recipe.source_kind == "dispatch_csv":
        if recipe.source_path is None:
            raise ValueError("dispatch_csv source requires path")
        return observations_from_dispatch_csv(recipe.source_path, target=target)
    if recipe.source_kind == "rocprof_trace":
        if recipe.source_path is None:
            raise ValueError("rocprof_trace source requires path")
        return observations_from_rocprof_trace(recipe.source_path, target=target)
    if recipe.source_kind == "consan_repro":
        return observations_from_consan_repro(
            recipe.repro_variant or "clean",
            target=target,
        )
    if recipe.source_kind in {"kernel_list", "kernel"}:
        return observations_from_kernel_list(
            [spec.to_entry() for spec in recipe.kernel_specs], target=target
        )
    raise ValueError(f"unsupported source kind {recipe.source_kind!r}")


def _env_dir(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    root = Path(raw)
    return root if root.is_dir() else None


def _resolve_waitcheck_binary() -> Path | None:
    """Locate ``rj_waitcheck`` in a prebuilt bundle or a CMake build tree.

    Prefers the flattened prebuilt sanitizer bundle published by
    ROCm/rocm-systems (``$ROCJITSU_PREBUILT/bin/rj_waitcheck``), then falls back
    to the raw CMake build-tree layout (``$ROCJITSU_BUILD/tools/rj_waitcheck``).
    """

    prebuilt = _env_dir("ROCJITSU_PREBUILT")
    if prebuilt is not None:
        candidate = prebuilt / "bin" / "rj_waitcheck"
        if candidate.is_file():
            return candidate
    build = _env_dir("ROCJITSU_BUILD")
    if build is not None:
        candidate = build / "tools" / "rj_waitcheck"
        if candidate.is_file():
            return candidate
    return None


def execute_sanitizer_run(
    recipe_path: Path,
    *,
    output_dir: Path,
    dry_run: bool = False,
) -> Path:
    """Resolve a sanitizer recipe, run checks, and return the report path."""

    recipe = load_sanitizer_recipe(recipe_path)
    observations = _resolve_observations(recipe)
    worklist = select_kernels(
        observations,
        requirement=recipe.requirement,
        top_n=recipe.top_n,
    )

    report_path = output_dir / recipe.report_name
    if dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        checks = tuple(
            CheckResult(
                sanitizer=sanitizer,
                state=ExecutionState.NOT_CHECKED,
                verdict=Verdict.NOT_CHECKED,
                reason="dry_run",
            )
            for sanitizer in recipe.sanitizers
        )
        write_report(
            build_report(target=recipe.target, worklist=worklist, checks=checks),
            report_path,
        )
        return report_path

    waitcheck_binary = _resolve_waitcheck_binary()
    consan_hook = resolve_consan_hook()
    consan_command = recipe.consan_command
    # Cross-check the executed repro against the single selected identity so a
    # recipe cannot name one repro while ConSan runs against another selection.
    consan_target = observations[0].identity if len(observations) == 1 else None

    run_sanitizers(
        worklist,
        target=recipe.target,
        sanitizers=recipe.sanitizers,
        output_dir=output_dir,
        waitcheck_binary=waitcheck_binary,
        consan_command=consan_command,
        consan_hook=consan_hook,
        consan_log=recipe.consan_log,
        consan_policy=recipe.consan_policy,
        on_missing_backend=recipe.on_missing_backend,
        consan_target=consan_target,
        report_name=recipe.report_name,
        timeout_seconds=(
            recipe.timeout_seconds
            if recipe.timeout_seconds is not None
            else DEFAULT_TIMEOUT_SECONDS
        ),
    )
    return report_path
