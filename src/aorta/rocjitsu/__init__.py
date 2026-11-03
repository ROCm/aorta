"""Reusable rocjitsu debugging runtime."""

from aorta.rocjitsu.cia_report import REPORT_SCHEMA, attach_archive_run
from aorta.rocjitsu.comparison import (
    ComparisonStatus,
    ShadowComparison,
    compare_job,
    compare_reports,
    summarize,
)
from aorta.rocjitsu.recording import (
    PINNED_ROCJITSU_COMMIT,
    RECORDING_SCHEMA,
    CodeObjectFacts,
    LiveRecording,
    OperationObservation,
    OutputCheck,
    OutputCheckState,
    inspect_code_object,
    render_output_check,
    write_recording,
)
from aorta.rocjitsu.rjdebug import Client, EndpointError, Reply
from aorta.rocjitsu.runtime import (
    ArchiveOperation,
    ArchiveRunConfig,
    ArchiveRunResult,
    run_archive_episode,
)

__all__ = [
    "REPORT_SCHEMA",
    "RECORDING_SCHEMA",
    "PINNED_ROCJITSU_COMMIT",
    "ArchiveOperation",
    "ArchiveRunConfig",
    "ArchiveRunResult",
    "Client",
    "CodeObjectFacts",
    "ComparisonStatus",
    "EndpointError",
    "LiveRecording",
    "OperationObservation",
    "OutputCheck",
    "OutputCheckState",
    "Reply",
    "ShadowComparison",
    "attach_archive_run",
    "compare_job",
    "compare_reports",
    "inspect_code_object",
    "render_output_check",
    "run_archive_episode",
    "summarize",
    "write_recording",
]
