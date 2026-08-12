"""Dual ROCm stack definitions for the nightly dashboard.

The dashboard maintains two tracks:
  * **customer** — pinned stack AMD recommends externally (``dashboard_stacks.yaml``)
  * **latest** — newest ROCm version present in published nightly history
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STACKS_PATH = _REPO_ROOT / "config" / "ci" / "dashboard_stacks.yaml"

_ROCM_RE = re.compile(r"(\d+(?:\.\d+)*)")


def load_dashboard_stacks() -> dict[str, dict[str, Any]]:
    """Return stack id -> config (cached per process)."""
    if not hasattr(load_dashboard_stacks, "_cache"):
        load_dashboard_stacks._cache = None  # type: ignore[attr-defined]
    if load_dashboard_stacks._cache is not None:  # type: ignore[attr-defined]
        return load_dashboard_stacks._cache  # type: ignore[attr-defined]
    if not _STACKS_PATH.is_file():
        load_dashboard_stacks._cache = _default_stacks()  # type: ignore[attr-defined]
        return load_dashboard_stacks._cache  # type: ignore[attr-defined]
    doc = yaml.safe_load(_STACKS_PATH.read_text(encoding="utf-8")) or {}
    stacks = doc.get("stacks") or {}
    if not isinstance(stacks, dict):
        stacks = {}
    load_dashboard_stacks._cache = {str(k): dict(v) for k, v in stacks.items()}  # type: ignore[attr-defined]
    return load_dashboard_stacks._cache  # type: ignore[attr-defined]


def _default_stacks() -> dict[str, dict[str, Any]]:
    return {
        "customer": {
            "id": "customer",
            "label": "Customer certified stack",
            "lede": "ROCm + PyTorch versions we recommend to customers.",
            "rocm": "7.2.0",
            "pytorch_prefix": "2.9",
        },
        "latest": {
            "id": "latest",
            "label": "Latest ROCm stack",
            "lede": "Newest ROCm build exercised in nightly CI.",
            "match": "newest_rocm",
        },
    }


def rocm_sort_key(raw: Any) -> tuple[int, ...]:
    """Sortable tuple from a ROCm version string (e.g. ``7.2.26015-…`` -> (7, 2, 26015))."""
    text = str(raw or "").strip()
    match = _ROCM_RE.match(text)
    if not match:
        return (0,)
    parts: list[int] = []
    for piece in match.group(1).split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            break
    return tuple(parts) or (0,)


def _build_rocm(build: dict[str, Any]) -> str:
    return str(build.get("rocm") or "").strip()


def _build_pytorch(build: dict[str, Any]) -> str:
    return str(build.get("torch") or "").strip()


def matches_customer_stack(build: dict[str, Any], stack: dict[str, Any]) -> bool:
    """True when a nightly ``build`` block matches the customer pin."""
    want_rocm = str(stack.get("rocm") or "").strip()
    rocm = _build_rocm(build)
    if want_rocm and not rocm.startswith(want_rocm):
        return False
    prefix = str(stack.get("pytorch_prefix") or stack.get("pytorch") or "").strip()
    torch = _build_pytorch(build)
    if prefix and not torch.startswith(prefix):
        return False
    return bool(rocm or torch)


def newest_rocm_key(results: list[dict[str, Any]]) -> tuple[int, ...]:
    keys = [rocm_sort_key(_build_rocm(doc.get("build") or {})) for doc in results]
    keys = [k for k in keys if k != (0,)]
    return max(keys) if keys else (0,)


def filter_results_for_stack(
    results: list[dict[str, Any]],
    stack: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the sub-history for one dashboard stack track."""
    if str(stack.get("match") or "") == "newest_rocm":
        target = newest_rocm_key(results)
        if target == (0,):
            return list(results)
        return [
            doc for doc in results
            if rocm_sort_key(_build_rocm(doc.get("build") or {})) == target
        ]
    return [
        doc for doc in results
        if matches_customer_stack(doc.get("build") or {}, stack)
    ]


def partition_results_by_stack(
    results: list[dict[str, Any]],
    stacks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    cfg = stacks if stacks is not None else load_dashboard_stacks()
    return {sid: filter_results_for_stack(results, stack) for sid, stack in cfg.items()}


def stack_toolchain_chips(build: dict[str, Any]) -> list[tuple[str, str]]:
    """Label/value pairs for PyTorch + ROCm (+ HIP when present) on one run."""
    chips: list[tuple[str, str]] = []
    torch = _build_pytorch(build)
    if torch:
        chips.append(("PyTorch", torch))
    rocm = _build_rocm(build)
    if rocm:
        chips.append(("ROCm", rocm))
    hip = str(build.get("hip") or "").strip()
    if hip:
        chips.append(("HIP", hip))
    return chips
