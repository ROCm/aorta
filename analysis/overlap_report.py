"""Generate overlap efficiency reports from profiling logs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))


@dataclass
class IterationStats:
    source: Path
    rank: int
    epoch: int
    step: int
    global_step: int
    loss: float
    overlap_ms: float
    compute_ms: float
    reducescatter_ms: float
    allreduce_ms: float


def load_records(log_dir: Path) -> List[IterationStats]:
    records: List[IterationStats] = []
    for file in sorted(log_dir.glob("rank_*_metrics.jsonl")):
        rank = int(file.stem.split("_")[1])
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                overlap = payload["profile"]["overlap"]
                records.append(
                    IterationStats(
                        source=file,
                        rank=rank,
                        epoch=payload["epoch"],
                        step=payload["step"],
                        global_step=payload["global_step"],
                        loss=payload["loss"],
                        overlap_ms=overlap["overlap_ms"].get("compute_comm", 0.0),
                        compute_ms=overlap["per_stream_ms"].get("compute", 0.0),
                        reducescatter_ms=overlap["overlap_ms"].get("compute_reducescatter", 0.0),
                        allreduce_ms=overlap["overlap_ms"].get("compute_allreduce", 0.0),
                    )
                )
    return records


def aggregate(records: Iterable[IterationStats]) -> Dict[str, float]:
    records = list(records)
    if not records:
        return {"count": 0}
    count = len(records)
    aggregate_metrics = {
        "count": count,
        "mean_loss": sum(r.loss for r in records) / count,
        "mean_overlap_ms": sum(r.overlap_ms for r in records) / count,
        "mean_compute_ms": sum(r.compute_ms for r in records) / count,
        "mean_allreduce_overlap_ms": sum(r.allreduce_ms for r in records) / count,
        "mean_reducescatter_overlap_ms": sum(r.reducescatter_ms for r in records) / count,
    }
    compute = aggregate_metrics["mean_compute_ms"]
    aggregate_metrics["overlap_ratio"] = (
        aggregate_metrics["mean_overlap_ms"] / compute if compute > 0 else 0.0
    )
    return aggregate_metrics


def plot_time_series(records: List[IterationStats], out_path: Path, label: str) -> None:
    if not records:
        return
    records = sorted(records, key=lambda r: r.global_step)
    steps = [r.global_step for r in records]
    overlap = [r.overlap_ms for r in records]
    compute = [r.compute_ms for r in records]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, overlap, label=f"{label} compute-comm overlap")
    plt.plot(steps, compute, label=f"{label} compute duration")
    plt.xlabel("Global Step")
    plt.ylabel("Milliseconds")
    plt.title(f"Overlap vs Compute Timeline ({label})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compare_datasets(reference: List[IterationStats], candidate: List[IterationStats]) -> Dict[str, float]:
    ref_metrics = aggregate(reference)
    cand_metrics = aggregate(candidate)
    result = {
        "reference_overlap_ratio": ref_metrics.get("overlap_ratio", 0.0),
        "candidate_overlap_ratio": cand_metrics.get("overlap_ratio", 0.0),
    }
    result["ratio_delta"] = result["candidate_overlap_ratio"] - result["reference_overlap_ratio"]
    return result


def build_report(
    log_dirs: List[Tuple[str, Path]],
    output_dir: Path,
    reference_name: Optional[str],
    candidate_name: Optional[str],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"datasets": {}}
    cached_records: Dict[str, List[IterationStats]] = {}

    for label, path in log_dirs:
        records = load_records(path)
        cached_records[label] = records
        metrics = aggregate(records)
        report["datasets"][label] = metrics
        plot_time_series(records, output_dir / f"{label}_timeline.png", label)

    if reference_name and candidate_name:
        report["comparison"] = compare_datasets(
            cached_records.get(reference_name, []), cached_records.get(candidate_name, [])
        )

    report_path = output_dir / "summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Generate compute-communication overlap report")
    parser.add_argument("--log-dir", type=Path, action="append", required=True, help="Log directory")
    parser.add_argument("--label", type=str, action="append", required=True, help="Label per log directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for reports")
    parser.add_argument("--reference", type=str, help="Label for reference platform")
    parser.add_argument("--candidate", type=str, help="Label for candidate platform")
    args = parser.parse_args()

    if len(args.log_dir) != len(args.label):
        raise SystemExit("--log-dir and --label counts must match")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    build_report(list(zip(args.label, args.log_dir)), args.output, args.reference, args.candidate)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
