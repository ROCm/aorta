"""Recipe schema and execution for ``mode: sanitizer`` runs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aorta.triage.recipe import RecipeSchemaError, load_recipe_mapping

from .models import CheckResult, ExecutionState, SelectionRequirement, Verdict
from .pipeline import run_sanitizers
from .report import build_report, write_report
from .selection import (
    observations_from_consan_repro,
    observations_from_dispatch_csv,
    observations_from_gemm_csv,
    observations_from_kernel_list,
    observations_from_rocprof_trace,
    select_kernels,
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
    kernel_names: tuple[str, ...] = ()

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
    on_missing_backend = _require_str(policy, "on_missing_backend")
    output = _require_mapping(plan.get("output"), name="sanitizer_plan.output")
    report_name = _require_str(output, "report")

    source_path: Path | None = None
    isa_dir: Path | None = None
    consan_command: Path | None = None
    consan_log = bool(source.get("consan_log", True))
    repro_variant: str | None = None
    kernel_names: tuple[str, ...] = ()

    if source_kind in {"gemm_csv", "dispatch_csv", "rocprof_trace"}:
        source_path = _resolve_path(_require_str(source, "path"), recipe_path=path)
    elif source_kind == "consan_repro":
        source_path = _resolve_path(_require_str(source, "hip"), recipe_path=path)
        repro_variant = _require_str(source, "variant")
        if "command" in source:
            consan_command = _resolve_path(_require_str(source, "command"), recipe_path=path)
    elif source_kind == "kernel_list":
        names = source.get("kernels")
        if not isinstance(names, list) or not names:
            raise RecipeSchemaError("sanitizer_plan.source.kernels must be a non-empty list")
        kernel_names = tuple(str(name) for name in names)
    elif source_kind == "kernel":
        kernel = _require_mapping(source.get("kernel"), name="sanitizer_plan.source.kernel")
        kernel_names = (_require_str(kernel, "name"),)

    if "isa_dir" in source:
        isa_dir = _resolve_path(_require_str(source, "isa_dir"), recipe_path=path)

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
        kernel_names=kernel_names,
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
        return observations_from_kernel_list(recipe.kernel_names, target=target)
    raise ValueError(f"unsupported source kind {recipe.source_kind!r}")


def _resolve_rocjitsu_build() -> Path | None:
    raw = os.environ.get("ROCJITSU_BUILD", "").strip()
    if not raw:
        return None
    build = Path(raw)
    return build if build.is_dir() else None


def _resolve_waitcheck_binary(build: Path | None) -> Path | None:
    if build is None:
        return None
    candidate = build / "tools" / "rj_waitcheck"
    return candidate if candidate.is_file() else None


def _resolve_consan_hook(build: Path | None) -> Path | None:
    if build is None:
        return None
    candidate = (
        build / "lib" / "rocjitsu" / "src" / "rocjitsu" / "hooks" / "librocjitsu_dbi_hooks.so"
    )
    return candidate if candidate.is_file() else None


def execute_sanitizer_run(
    recipe_path: Path,
    *,
    output_dir: Path,
    dry_run: bool = False,
) -> Path:
    """Resolve a sanitizer recipe, run checks, and return the report path."""

    recipe = load_sanitizer_recipe(recipe_path)
    observations = _resolve_observations(recipe)
    if recipe.scope_kind == "kernel" and recipe.source_kind not in {"kernel", "consan_repro"}:
        worklist = select_kernels(
            observations,
            requirement=recipe.requirement,
            top_n=min(recipe.top_n, len(observations)),
        )
    else:
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

    build = _resolve_rocjitsu_build()
    waitcheck_binary = _resolve_waitcheck_binary(build)
    consan_hook = _resolve_consan_hook(build)
    consan_command = recipe.consan_command
    if consan_command is None and recipe.source_kind == "consan_repro":
        consan_command = output_dir / "consan_repro"

    report = run_sanitizers(
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
    )
    _ = report
    return report_path if report_path.is_file() else output_dir / "sanitizer_report.json"
