"""Attach one archive-backed rocjitsu episode to an existing CIA job."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from aorta.rocjitsu.cia_report import attach_archive_run
from aorta.rocjitsu.rjdebug import Client
from aorta.rocjitsu.runtime import ArchiveRunConfig, run_archive_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        client = Client(
            args.endpoint,
            args.model,
            api_key=os.environ.get("AORTA_RJ_EXPERT_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            timeout=args.timeout,
        )
        run = run_archive_episode(
            ArchiveRunConfig(
                archive=args.archive,
                scenario_id=args.scenario,
                split=args.split,
            ),
            client,
        )
        report = attach_archive_run(args.jobs_root, args.job_id, run)
    except (OSError, SystemExit, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(report)
    return 1 if run.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
