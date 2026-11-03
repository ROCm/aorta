"""Record reviewed outcomes and export only approved offline-training evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class FeedbackOutcome(str, Enum):
    FIXED = "fixed"
    NOT_FIXED = "not_fixed"
    PARTIALLY_FIXED = "partially_fixed"
    NOT_TESTED = "not_tested"


@dataclass(frozen=True)
class FeedbackRecord:
    job_id: str
    outcome: FeedbackOutcome
    verified: bool
    verified_by: str
    notes: str
    created_at: str
    expert_report_sha256: str


def _job(jobs_root: Path, job_id: str) -> Path:
    root = jobs_root.resolve()
    if not job_id or Path(job_id).name != job_id:
        raise ValueError(f"invalid CIA job ID: {job_id!r}")
    job = root / job_id
    if job.is_symlink():
        raise ValueError("CIA job path must not be a symlink")
    resolved = job.resolve()
    if resolved.parent != root:
        raise ValueError("CIA job escapes jobs root")
    record = json.loads((resolved / "job.json").read_text(encoding="utf-8"))
    if record.get("job_id") != job_id:
        raise ValueError("CIA job record does not match its directory")
    if record.get("status") not in {"completed", "failed", "cancelled"}:
        raise ValueError("CIA job is not terminal")
    return resolved


def _write_exclusive(path: Path, document: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")


def record_feedback(
    jobs_root: Path,
    job_id: str,
    outcome: FeedbackOutcome,
    *,
    verified: bool = False,
    verified_by: str = "",
    notes: str = "",
) -> Path:
    """Store one immutable user outcome beside an expert report."""
    if verified and not verified_by.strip():
        raise ValueError("verified feedback requires verified_by")
    if verified and outcome is FeedbackOutcome.NOT_TESTED:
        raise ValueError("not_tested feedback cannot be verified")
    job = _job(jobs_root, job_id)
    expert_dir = job / "rj_expert"
    if expert_dir.is_symlink():
        raise ValueError("rocjitsu expert directory must not be a symlink")
    report = expert_dir / "report.json"
    report_bytes = report.read_bytes()
    record = FeedbackRecord(
        job_id=job_id,
        outcome=outcome,
        verified=verified,
        verified_by=verified_by.strip(),
        notes=notes.strip(),
        created_at=datetime.now(timezone.utc).isoformat(),
        expert_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
    )
    path = expert_dir / "feedback.json"
    _write_exclusive(path, asdict(record))
    return path


def export_verified(
    jobs_root: Path,
    output: Path,
    *,
    include_operations: bool = False,
) -> int:
    """Export reviewed outcomes; unverified user feedback is never training data."""
    rows = []
    for feedback_path in sorted(jobs_root.glob("*/rj_expert/feedback.json")):
        job = _job(jobs_root, feedback_path.parents[1].name)
        expert_dir = job / "rj_expert"
        if expert_dir.is_symlink() or feedback_path.resolve().parent != expert_dir.resolve():
            raise ValueError("feedback path escapes the CIA job")
        feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
        if feedback.get("verified") is not True:
            continue
        report_path = expert_dir / "report.json"
        report_bytes = report_path.read_bytes()
        if hashlib.sha256(report_bytes).hexdigest() != feedback.get("expert_report_sha256"):
            raise ValueError(f"expert report changed after feedback: {job.name}")
        report = json.loads(report_bytes)
        row: dict[str, Any] = {"feedback": feedback, "expert_report": report}
        if include_operations:
            operations_name = str(report.get("operations") or "operations.jsonl")
            if Path(operations_name).name != operations_name:
                raise ValueError(f"invalid operations path in expert report: {job.name}")
            operations = expert_dir / operations_name
            row["operations"] = [
                json.loads(line)
                for line in operations.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record")
    record.add_argument("--jobs-root", type=Path, required=True)
    record.add_argument("--job-id", required=True)
    record.add_argument("--outcome", choices=[item.value for item in FeedbackOutcome], required=True)
    record.add_argument("--verified", action="store_true")
    record.add_argument("--verified-by", default="")
    record.add_argument("--notes", default="")
    export = subparsers.add_parser("export")
    export.add_argument("--jobs-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--include-operations", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "record":
            path = record_feedback(
                args.jobs_root,
                args.job_id,
                FeedbackOutcome(args.outcome),
                verified=args.verified,
                verified_by=args.verified_by,
                notes=args.notes,
            )
            print(path)
        else:
            count = export_verified(
                args.jobs_root,
                args.output,
                include_operations=args.include_operations,
            )
            print(f"{args.output}: {count} verified record(s)")
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FeedbackOutcome",
    "FeedbackRecord",
    "export_verified",
    "record_feedback",
]
