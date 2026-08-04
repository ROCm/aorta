"""Recipe-oriented kernel sanitizer primitives.

Provides deterministic kernel selection, kernel-source resolvers (GEMM CSV,
rocprofiler trace, dispatch CSV, kernel list, ConSan repro), static Waitcheck
(exact-entry and whole-code-object scan), an executable ConSan path that runs a
targeted repro under the RocJITsu DBI hook, typed reports, and a strict
``mode: sanitizer`` recipe loader. ConSan on supported targets (e.g. gfx950)
runs for real; it falls back to fail-closed ``not_checked`` only when no
targeted repro command / hook is provisioned.
"""

from .backends import support
from .consan import (
    ConSanMode,
    ParsedCombinedOutput,
    evaluate_record_replay,
    parse_record_replay_output,
    run_consan,
    scoped_consan_not_checked,
)
from .models import (
    REPORT_SCHEMA,
    WORKLIST_SCHEMA,
    CheckResult,
    ExecutionState,
    ExecutionSummary,
    Finding,
    FindingSeverity,
    KernelCheckResult,
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    ObjectCoverage,
    SanitizerReport,
    SelectionRequirement,
    Verdict,
)
from .pipeline import run_sanitizers
from .recipe import SanitizerRecipe, execute_sanitizer_run, load_sanitizer_recipe
from .report import build_report, read_report, write_report
from .selection import (
    observations_from_consan_repro,
    observations_from_dispatch_csv,
    observations_from_dispatch_rows,
    observations_from_gemm_csv,
    observations_from_kernel_list,
    observations_from_magpie_report,
    observations_from_magpie_workspace,
    observations_from_rocprof_rows,
    observations_from_rocprof_trace,
    select_kernels,
)
from .waitcheck import (
    parse_waitcheck_jsonl,
    parse_waitcheck_text,
    run_waitcheck,
    waitcheck_argv,
)

__all__ = [
    "REPORT_SCHEMA",
    "WORKLIST_SCHEMA",
    "CheckResult",
    "ConSanMode",
    "ExecutionState",
    "ExecutionSummary",
    "Finding",
    "FindingSeverity",
    "KernelCheckResult",
    "KernelIdentity",
    "KernelObservation",
    "KernelWorklist",
    "ObjectCoverage",
    "ParsedCombinedOutput",
    "SanitizerRecipe",
    "SanitizerReport",
    "SelectionRequirement",
    "Verdict",
    "build_report",
    "evaluate_record_replay",
    "execute_sanitizer_run",
    "load_sanitizer_recipe",
    "observations_from_consan_repro",
    "observations_from_dispatch_csv",
    "observations_from_dispatch_rows",
    "observations_from_gemm_csv",
    "observations_from_kernel_list",
    "observations_from_magpie_report",
    "observations_from_magpie_workspace",
    "observations_from_rocprof_rows",
    "observations_from_rocprof_trace",
    "parse_record_replay_output",
    "parse_waitcheck_jsonl",
    "parse_waitcheck_text",
    "read_report",
    "run_consan",
    "run_sanitizers",
    "run_waitcheck",
    "scoped_consan_not_checked",
    "select_kernels",
    "support",
    "waitcheck_argv",
    "write_report",
]
