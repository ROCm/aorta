"""Persist archive-demo output beside, never inside, a CIA Autopsy bundle."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aorta.rocjitsu.runtime import ArchiveRunResult

REPORT_SCHEMA = "aorta.rj_expert_report/0.1"
_TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _within(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise ValueError(f"path escapes jobs root: {candidate}")
    return candidate


def _write_atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _operation_dict(operation: Any) -> dict[str, Any]:
    return asdict(operation)


def _transcript(run: ArchiveRunResult) -> str:
    lines = [f"# rocjitsu archive demo: {run.config.scenario_id}", ""]
    for operation in run.operations:
        lines.extend(
            [
                f"## Turn {operation.step}",
                "",
                "### Prompt",
                "",
                "```text",
                operation.prompt,
                "```",
                "",
                "### Reply",
                "",
                "```json",
                operation.reply,
                "```",
                "",
                "### Tool output",
                "",
                "```text",
                operation.rendered_result or "(no tool output)",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def attach_archive_run(jobs_root: Path, job_id: str, run: ArchiveRunResult) -> Path:
    """Attach *run* to an existing CIA job as an isolated storage-only report."""
    root = jobs_root.resolve()
    if not job_id or Path(job_id).name != job_id:
        raise ValueError(f"invalid CIA job ID: {job_id!r}")
    literal_job_dir = root / job_id
    if literal_job_dir.is_symlink():
        raise ValueError(f"CIA job path must not be a symlink: {job_id}")
    job_dir = _within(root, literal_job_dir)
    job_record_path = job_dir / "job.json"
    if not job_record_path.is_file():
        raise FileNotFoundError(f"CIA job does not exist: {job_id}")
    job_record = json.loads(job_record_path.read_text(encoding="utf-8"))
    if job_record.get("job_id") != job_id:
        raise ValueError(f"CIA job record does not match directory: {job_id}")
    if job_record.get("status") not in _TERMINAL_JOB_STATUSES:
        raise ValueError(f"CIA job is not terminal: {job_id}")

    literal_expert_dir = job_dir / "rj_expert"
    if literal_expert_dir.is_symlink():
        raise ValueError("rocjitsu expert directory must not be a symlink")
    try:
        literal_expert_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FileExistsError(
            f"rocjitsu expert attachment already exists: {literal_expert_dir}"
        ) from exc
    expert_dir = _within(job_dir, literal_expert_dir)
    report_path = expert_dir / "report.json"

    operations_path = expert_dir / "operations.jsonl"
    operations = "".join(
        json.dumps(_operation_dict(operation), sort_keys=True) + "\n"
        for operation in run.operations
    )
    _write_atomic(operations_path, operations)
    _write_atomic(expert_dir / "transcript.md", _transcript(run))

    report = {
        "schema": REPORT_SCHEMA,
        "mode": "archive_demo",
        "advisory_only": True,
        "cia_job_binding": "storage_only",
        "live_job_input_analyzed": False,
        "affects_autopsy": False,
        "confirmation": "not_evaluated",
        "job_id": job_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "path": str(run.config.archive.resolve()),
            "scenario_id": run.config.scenario_id,
            "scenario_digest": run.scenario_digest,
            "split": run.config.split,
            "reward": run.config.reward,
            "prompt_profile": run.config.prompt_profile,
        },
        "model": {
            "id": run.model,
            "endpoint": run.endpoint,
            "server": run.server,
            "system_prompt_sha256": run.system_prompt_sha256,
            "seed": run.config.seed,
        },
        "status": run.status,
        "answer": run.answer,
        "archive_outcome": run.outcome,
        "archive_score": run.score,
        "turns": len(run.operations),
        "seconds": round(run.seconds, 3),
        "error": run.error,
        "operations": operations_path.name,
        "transcript": "transcript.md",
        "limitations": [
            "replays one recorded corpus-v2 scenario",
            "does not analyze the bound CIA job input",
            "does not inspect a live code-object architecture",
            "does not run a live output check",
            "does not claim a live rocjitsu commit",
        ],
    }
    _write_atomic(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report_path


__all__ = ["REPORT_SCHEMA", "attach_archive_run"]
