"""Recipe-oriented kernel sanitizer primitives.

Phase 0 provides deterministic kernel selection, exact-entry static Waitcheck,
typed reports, and fail-closed ConSan capability reporting. Dynamic ConSan does
not run until RocJITsu supports an allowlist for the resolved worklist.
"""

from .consan import (
    ConSanMode,
    ParsedCombinedOutput,
    evaluate_record_replay,
    parse_record_replay_output,
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
from .report import build_report, read_report, write_report
from .selection import (
    observations_from_dispatch_csv,
    observations_from_dispatch_rows,
    observations_from_magpie_report,
    observations_from_magpie_workspace,
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
    "SanitizerReport",
    "SelectionRequirement",
    "Verdict",
    "build_report",
    "evaluate_record_replay",
    "observations_from_dispatch_csv",
    "observations_from_dispatch_rows",
    "observations_from_magpie_report",
    "observations_from_magpie_workspace",
    "parse_record_replay_output",
    "parse_waitcheck_jsonl",
    "parse_waitcheck_text",
    "read_report",
    "run_sanitizers",
    "run_waitcheck",
    "scoped_consan_not_checked",
    "select_kernels",
    "waitcheck_argv",
    "write_report",
]
