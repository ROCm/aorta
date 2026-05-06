"""Tests for ``aorta.report.analysis.kernel_correlator``.

Loaded via ``importlib.util`` to dodge the heavy ``aorta.report``
``__init__`` chain (openpyxl/pandas/matplotlib are not in the test venv).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"


def _ensure_pkg(name: str, path: Path) -> None:
    """Pre-register a parent package as an empty namespace module.

    Required so that ``from ..analysis.kernel_correlator import ...``
    relative imports inside ``kernel_report`` resolve when we direct-load
    the leaf module.
    """
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


_ensure_pkg("aorta", _REPO_SRC / "aorta")
_ensure_pkg("aorta.report", _REPO_SRC / "aorta" / "report")
_ensure_pkg("aorta.report.analysis", _REPO_SRC / "aorta" / "report" / "analysis")


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_SRC / relpath)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {relpath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_kc = _load(
    "aorta.report.analysis.kernel_correlator",
    "aorta/report/analysis/kernel_correlator.py",
)
KernelEventCorrelator = _kc.KernelEventCorrelator
IterationRecord = _kc.IterationRecord
iter_findings_table = _kc.iter_findings_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for payload in payloads:
            fh.write(json.dumps(payload) + "\n")


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


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


class TestLoadMetrics:
    def test_parses_well_formed_lines(self, tmp_path: Path):
        path = tmp_path / "rank_0_metrics.jsonl"
        _write_jsonl(
            path,
            [
                _payload(rank=0, global_step=0, loss=1.5, kernel_summary={"kfd_evict": 1}),
                _payload(rank=0, global_step=1, loss=1.4, kernel_summary={"bo_move": 2}),
            ],
        )
        records = KernelEventCorrelator().load_metrics(path)
        assert len(records) == 2
        assert records[0].rank == 0
        assert records[0].kernel_summary["kfd_evict"] == 1
        assert records[1].kernel_summary["bo_move"] == 2

    def test_skips_blank_and_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "rank_0_metrics.jsonl"
        path.write_text(
            json.dumps(_payload(rank=0, global_step=0, loss=1.0))
            + "\n"
            + "\n"
            + "{not json}\n"
            + json.dumps(_payload(rank=0, global_step=1, loss=2.0))
            + "\n",
            encoding="utf-8",
        )
        records = KernelEventCorrelator().load_metrics(path)
        assert len(records) == 2

    def test_loss_nan_is_recognised(self, tmp_path: Path):
        path = tmp_path / "rank_0_metrics.jsonl"
        # Write NaN as a JSON `null` (most loggers) and as the string
        # "NaN" both end up as float('nan').
        path.write_text(
            json.dumps({**_payload(rank=0, global_step=0, loss=1.0), "loss": None})
            + "\n"
            + json.dumps({**_payload(rank=0, global_step=1, loss=1.0), "loss": "NaN"})
            + "\n",
            encoding="utf-8",
        )
        records = KernelEventCorrelator().load_metrics(path)
        assert all(math.isnan(r.loss) for r in records)


class TestLoadMetricsGlobOrders:
    def test_files_concatenated_and_sorted_by_step_then_rank(self, tmp_path: Path):
        _write_jsonl(
            tmp_path / "rank_0_metrics.jsonl",
            [_payload(rank=0, global_step=1, loss=1.0), _payload(rank=0, global_step=0, loss=1.0)],
        )
        _write_jsonl(
            tmp_path / "rank_1_metrics.jsonl",
            [_payload(rank=1, global_step=0, loss=1.0)],
        )
        records = KernelEventCorrelator().load_metrics_glob(tmp_path)
        assert [(r.global_step, r.rank) for r in records] == [(0, 0), (0, 1), (1, 0)]


# ---------------------------------------------------------------------------
# NaN findings + lookback
# ---------------------------------------------------------------------------


class TestFindFailures:
    def test_finds_each_nan_iteration(self, tmp_path: Path):
        path = tmp_path / "rank_0_metrics.jsonl"
        # Two NaN rows on rank 0, separated by a healthy row.
        records = [
            _payload(rank=0, global_step=0, loss=1.0),
            _payload(rank=0, global_step=1, loss=float("nan")),
            _payload(rank=0, global_step=2, loss=1.5),
            _payload(rank=0, global_step=3, loss=float("nan")),
        ]
        _write_jsonl(path, records)
        loaded = KernelEventCorrelator().load_metrics(path)
        findings = KernelEventCorrelator().find_failures(loaded)
        assert [f.target.global_step for f in findings] == [1, 3]

    def test_lookback_window_respects_configured_size(self, tmp_path: Path):
        path = tmp_path / "rank_0_metrics.jsonl"
        rows = [
            _payload(
                rank=0,
                global_step=i,
                loss=1.0 if i < 5 else float("nan"),
                kernel_summary={"kfd_evict": i},
            )
            for i in range(6)
        ]
        _write_jsonl(path, rows)
        loaded = KernelEventCorrelator().load_metrics(path)
        findings = KernelEventCorrelator(lookback_iterations=3).find_failures(loaded)
        assert len(findings) == 1
        # Lookback window should include steps 2, 3, 4 (the 3 preceding
        # the NaN at step 5).
        steps = [r.global_step for r in findings[0].preceding_window]
        assert steps == [2, 3, 4]

    def test_kernel_event_total_sums_window_plus_target(self, tmp_path: Path):
        path = tmp_path / "rank_0_metrics.jsonl"
        rows = [
            _payload(rank=0, global_step=0, loss=1.0, kernel_summary={"bo_move": 5}),
            _payload(rank=0, global_step=1, loss=1.0, kernel_summary={"bo_move": 7}),
            _payload(
                rank=0,
                global_step=2,
                loss=float("nan"),
                kernel_summary={"bo_move": 3, "kfd_evict": 1},
            ),
        ]
        _write_jsonl(path, rows)
        loaded = KernelEventCorrelator().load_metrics(path)
        findings = KernelEventCorrelator(lookback_iterations=2).find_failures(loaded)
        assert len(findings) == 1
        total = findings[0].kernel_event_total
        assert total["bo_move"] == 15
        assert total["kfd_evict"] == 1


class TestSummarise:
    def test_summary_counts_iterations_and_nans(self, tmp_path: Path):
        path = tmp_path / "rank_0_metrics.jsonl"
        rows = [
            _payload(rank=0, global_step=0, loss=1.0, kernel_summary={"kfd_evict": 2}),
            _payload(rank=0, global_step=1, loss=float("nan")),
            _payload(rank=1, global_step=0, loss=1.0),
        ]
        _write_jsonl(path, rows)
        loaded = KernelEventCorrelator().load_metrics(path)
        summary = KernelEventCorrelator().summarise(loaded)
        assert summary["total_iterations"] == 3
        assert summary["nan_iterations"] == 1
        assert summary["kernel_event_totals"]["kfd_evict"] == 2
        assert summary["ranks"] == [0, 1]


class TestIterFindingsTable:
    def test_flatten_findings_into_dict_rows(self, tmp_path: Path):
        path = tmp_path / "rank_0_metrics.jsonl"
        rows = [
            _payload(rank=0, global_step=0, loss=1.0, kernel_summary={"bo_move": 2}),
            _payload(rank=0, global_step=1, loss=float("nan"), kernel_summary={"bo_move": 5}),
        ]
        _write_jsonl(path, rows)
        loaded = KernelEventCorrelator().load_metrics(path)
        findings = KernelEventCorrelator().find_failures(loaded)
        table = list(iter_findings_table(findings))
        assert len(table) == 1
        row = table[0]
        assert row["rank"] == 0
        assert row["global_step"] == 1
        assert row["lookback_iterations"] == 1
        assert row["kernel_bo_move"] == 7  # 2 + 5


# ---------------------------------------------------------------------------
# PR #162 round 2 (C1): explicit JSON ``null`` values must not crash
# ``_payload_to_record`` with ``int(None)`` / ``float(None)`` TypeError.
# Older / partial trainer schemas (rank crashed mid-iteration, custom
# integrations) can emit explicit nulls instead of omitting the key.
# ---------------------------------------------------------------------------


class TestNullSafePayloadCoercion:
    def test_null_event_count_is_treated_as_fallback_length(self, tmp_path: Path):
        path = tmp_path / "rank_0_metrics.jsonl"
        # Note: ``event_count`` is *explicitly null*, not omitted, so
        # ``dict.get("event_count", default)`` returns None rather than
        # the default. Pre-fix this raised ``TypeError: int() argument
        # must be a string... not 'NoneType'``.
        path.write_text(
            json.dumps(
                {
                    "rank": 0,
                    "global_step": 0,
                    "epoch": 0,
                    "step": 0,
                    "loss": 1.0,
                    "profile": {"overlap": {"overlap_ms": {}, "per_stream_ms": {}}},
                    "kernel_trace": {
                        "summary": {"kfd_evict": 1},
                        "event_count": None,
                        "events": [
                            {"type": "kfd_evict"},
                            {"type": "kfd_evict"},
                        ],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        records = KernelEventCorrelator().load_metrics(path)
        assert len(records) == 1
        # Fallback should derive from len(events) when event_count is null.
        assert records[0].kernel_event_count == 2

    def test_null_event_count_without_events_array_defaults_to_zero(self, tmp_path: Path):
        path = tmp_path / "rank_0_metrics.jsonl"
        path.write_text(
            json.dumps(
                {
                    "rank": 0,
                    "global_step": 0,
                    "epoch": 0,
                    "step": 0,
                    "loss": 1.0,
                    "profile": {"overlap": {"overlap_ms": {}, "per_stream_ms": {}}},
                    "kernel_trace": {
                        "summary": {},
                        "event_count": None,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        records = KernelEventCorrelator().load_metrics(path)
        assert len(records) == 1
        assert records[0].kernel_event_count == 0

    def test_null_summary_value_defaults_to_zero(self, tmp_path: Path):
        # A single null inside ``summary`` (e.g. partially-finalised
        # iteration) must not poison the whole row -- we accept the row
        # with that single counter coerced to zero.
        path = tmp_path / "rank_0_metrics.jsonl"
        path.write_text(
            json.dumps(
                {
                    "rank": 0,
                    "global_step": 0,
                    "epoch": 0,
                    "step": 0,
                    "loss": 1.0,
                    "profile": {"overlap": {"overlap_ms": {}, "per_stream_ms": {}}},
                    "kernel_trace": {
                        "summary": {"kfd_evict": 4, "bo_move": None},
                        "event_count": 4,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        records = KernelEventCorrelator().load_metrics(path)
        assert len(records) == 1
        assert records[0].kernel_summary["kfd_evict"] == 4
        assert records[0].kernel_summary["bo_move"] == 0

    def test_null_top_level_int_fields_default_to_zero(self, tmp_path: Path):
        # ``rank``, ``global_step``, ``epoch``, ``step`` are all coerced
        # via the same helper; nulls must not crash.
        path = tmp_path / "rank_0_metrics.jsonl"
        path.write_text(
            json.dumps(
                {
                    "rank": None,
                    "global_step": None,
                    "epoch": None,
                    "step": None,
                    "loss": None,
                    "profile": {"overlap": {"overlap_ms": {"H2D": None}, "per_stream_ms": {}}},
                    "kernel_trace": {"summary": {}, "event_count": 0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        records = KernelEventCorrelator().load_metrics(path)
        assert len(records) == 1
        assert records[0].rank == 0
        assert records[0].global_step == 0
        assert records[0].epoch == 0
        assert records[0].step == 0
        assert math.isnan(records[0].loss)
        assert records[0].overlap_ms["H2D"] == 0.0

    def test_kernel_trace_block_completely_absent(self, tmp_path: Path):
        # Pre-existing safety net -- no ``kernel_trace`` key at all
        # (older logs from before the eBPF integration). The fallback
        # path must still produce a usable record.
        path = tmp_path / "rank_0_metrics.jsonl"
        path.write_text(
            json.dumps(
                {
                    "rank": 0,
                    "global_step": 7,
                    "epoch": 0,
                    "step": 7,
                    "loss": 1.0,
                    "profile": {"overlap": {"overlap_ms": {}, "per_stream_ms": {}}},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        records = KernelEventCorrelator().load_metrics(path)
        assert len(records) == 1
        assert records[0].kernel_event_count == 0
        assert records[0].kernel_summary == {}
