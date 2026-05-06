"""Tests for ``aorta.report.generators.kernel_report.generate_kernel_report``."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


_ensure_pkg("aorta", _REPO_SRC / "aorta")
_ensure_pkg("aorta.report", _REPO_SRC / "aorta" / "report")
_ensure_pkg("aorta.report.analysis", _REPO_SRC / "aorta" / "report" / "analysis")
_ensure_pkg("aorta.report.generators", _REPO_SRC / "aorta" / "report" / "generators")


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_SRC / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the correlator first so kernel_report's relative import resolves.
_load(
    "aorta.report.analysis.kernel_correlator",
    "aorta/report/analysis/kernel_correlator.py",
)
_kr = _load(
    "aorta.report.generators.kernel_report",
    "aorta/report/generators/kernel_report.py",
)
generate_kernel_report = _kr.generate_kernel_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload(
    *,
    rank: int,
    global_step: int,
    loss: float,
    kernel_summary: dict[str, int] | None = None,
    event_count: int = 0,
) -> dict:
    return {
        "rank": rank,
        "global_step": global_step,
        "epoch": 0,
        "step": global_step,
        "loss": loss,
        "profile": {"overlap": {"overlap_ms": {}, "per_stream_ms": {}}},
        "kernel_trace": {
            "summary": kernel_summary or {},
            "event_count": event_count,
        },
    }


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# generate_kernel_report end-to-end
# ---------------------------------------------------------------------------


class TestGenerateKernelReport:
    def test_writes_summary_csv_and_html(self, tmp_path: Path):
        metrics = tmp_path / "metrics"
        metrics.mkdir()
        _write(
            metrics / "rank_0_metrics.jsonl",
            [
                _payload(rank=0, global_step=0, loss=1.0, kernel_summary={"bo_move": 1}),
                _payload(rank=0, global_step=1, loss=float("nan"), kernel_summary={"bo_move": 2}),
            ],
        )
        out = tmp_path / "report"
        artifacts = generate_kernel_report(metrics, out, lookback_iterations=2)

        assert set(artifacts) == {"summary_json", "findings_csv", "html_report"}
        # All three artifacts exist and are non-empty.
        for path in artifacts.values():
            assert path.exists()
            assert path.stat().st_size > 0

        # Summary JSON shape.
        summary = json.loads(artifacts["summary_json"].read_text())
        assert summary["summary"]["total_iterations"] == 2
        assert summary["summary"]["nan_iterations"] == 1
        assert len(summary["findings"]) == 1

    def test_csv_findings_has_dynamic_kernel_columns(self, tmp_path: Path):
        metrics = tmp_path / "metrics"
        metrics.mkdir()
        _write(
            metrics / "rank_0_metrics.jsonl",
            [
                _payload(rank=0, global_step=0, loss=1.0, kernel_summary={"kfd_evict": 1}),
                _payload(
                    rank=0,
                    global_step=1,
                    loss=float("nan"),
                    kernel_summary={"kfd_evict": 2, "bo_move": 3},
                ),
            ],
        )
        out = tmp_path / "report"
        artifacts = generate_kernel_report(metrics, out)

        rows = list(csv.DictReader(artifacts["findings_csv"].open()))
        assert len(rows) == 1
        # Dynamic column names use a ``kernel_`` prefix per
        # iter_findings_table; the totals must reflect window+target.
        assert rows[0]["kernel_kfd_evict"] == "3"
        assert rows[0]["kernel_bo_move"] == "3"

    def test_html_marks_nan_loss(self, tmp_path: Path):
        metrics = tmp_path / "metrics"
        metrics.mkdir()
        _write(
            metrics / "rank_0_metrics.jsonl",
            [
                _payload(rank=0, global_step=0, loss=1.0),
                _payload(rank=0, global_step=1, loss=float("nan")),
            ],
        )
        out = tmp_path / "report"
        artifacts = generate_kernel_report(metrics, out)
        html = artifacts["html_report"].read_text()
        # The NaN row gets the .nan CSS class for visual emphasis.
        assert "class='nan'" in html
        # Source line is rendered escaped.
        assert str(metrics) in html

    def test_no_findings_writes_header_matching_with_findings_path(self, tmp_path: Path):
        metrics = tmp_path / "metrics"
        metrics.mkdir()
        _write(
            metrics / "rank_0_metrics.jsonl",
            [_payload(rank=0, global_step=0, loss=1.0)],
        )
        out = tmp_path / "report"
        artifacts = generate_kernel_report(metrics, out)
        # The empty-findings CSV must carry the same static-column
        # prefix that ``iter_findings_table`` always emits, so a
        # downstream tool can count on these columns being present
        # regardless of whether NaNs were detected. Pre-fix this row
        # said only "rank,global_step,loss" -- inconsistent with the
        # populated path -- and Copilot flagged it on PR #162.
        assert artifacts["findings_csv"].read_text().strip() == (
            "rank,global_step,loss,lookback_iterations"
        )
        assert "No NaN iterations detected" in artifacts["html_report"].read_text()


class TestCsvColumnOrderIsStable:
    """B3: column order is documented contract.

    Pre-fix the populated path used ``sorted(...)`` on every key in
    every row, which mixed static and dynamic columns alphabetically
    (``global_step, kernel_bo_move, kernel_kfd_evict, lookback_iterations,
    loss, rank``). Post-fix the order is fixed: static prefix in source
    order, then sorted ``kernel_*`` columns.
    """

    def test_static_columns_come_first_in_documented_order(self, tmp_path: Path):
        metrics = tmp_path / "metrics"
        metrics.mkdir()
        _write(
            metrics / "rank_0_metrics.jsonl",
            [
                _payload(rank=0, global_step=0, loss=1.0, kernel_summary={"bo_move": 1}),
                _payload(
                    rank=0,
                    global_step=1,
                    loss=float("nan"),
                    kernel_summary={"bo_move": 2, "kfd_evict": 1},
                ),
            ],
        )
        out = tmp_path / "report"
        artifacts = generate_kernel_report(metrics, out)
        header = artifacts["findings_csv"].read_text().splitlines()[0].split(",")
        assert header[:4] == ["rank", "global_step", "loss", "lookback_iterations"]
        # Dynamic kernel columns are sorted alphabetically and only
        # appear after the static prefix.
        dynamic = header[4:]
        assert dynamic == sorted(dynamic)
        assert all(name.startswith("kernel_") for name in dynamic)
        assert "kernel_bo_move" in dynamic
        assert "kernel_kfd_evict" in dynamic

    def test_creates_output_dir_if_missing(self, tmp_path: Path):
        metrics = tmp_path / "metrics"
        metrics.mkdir()
        _write(
            metrics / "rank_0_metrics.jsonl",
            [_payload(rank=0, global_step=0, loss=1.0)],
        )
        out = tmp_path / "deeply" / "nested" / "report"
        artifacts = generate_kernel_report(metrics, out)
        assert out.is_dir()
        assert all(path.parent == out for path in artifacts.values())
