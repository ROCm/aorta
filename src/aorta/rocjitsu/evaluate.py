"""Compare stored expert reports with independent ConSan evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aorta.rocjitsu.comparison import (
    ComparisonStatus,
    ShadowComparison,
    compare_job,
    summarize,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparisons = []
    for report in sorted(args.jobs_root.glob("*/rj_expert/report.json")):
        output = compare_job(report.parents[1])
        comparisons.append(json.loads(output.read_text(encoding="utf-8")))
    summary = summarize(
        ShadowComparison(
            status=ComparisonStatus(item["status"]),
            expert_class=item.get("expert_class", ""),
            expert_kernel=item.get("expert_kernel", ""),
            consan_kernels=tuple(item.get("consan_kernels", [])),
            rationale=item.get("rationale", ""),
        )
        for item in comparisons
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "comparisons": comparisons}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
