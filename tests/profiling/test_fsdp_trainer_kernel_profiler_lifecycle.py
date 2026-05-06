"""Structural test for the FSDP trainer's kernel-profiler lifecycle.

Cannot exercise ``aorta.training.fsdp_trainer.main`` end-to-end in this
venv (no torch / no rendezvous), so this test asserts the structural
property directly via AST inspection: ``kernel_profiler.start()`` must
live inside the same ``try`` block whose ``finally`` calls
``dist.destroy_process_group()``. Pre-fix (Copilot review on PR #162)
``start()`` was *outside* that block, so a profiler-startup raise
(``skip_if_unavailable=False`` and bpftrace missing, or the new
``BpftraceRunner._await_startup`` permission failure) would leak the
process group and hang the rendezvous on every other rank.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TRAINER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "aorta"
    / "training"
    / "fsdp_trainer.py"
)


def _find_main_function(tree: ast.AST) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("fsdp_trainer.main() not found")


def _is_attr_call(node: ast.AST, target_attr: str) -> bool:
    """Return True if ``node`` is an ``ast.Expr`` wrapping ``X.target_attr(...)``."""
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr == target_attr


def _is_dist_destroy_call(node: ast.AST) -> bool:
    """``dist.destroy_process_group()`` -- the cleanup we care about."""
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "destroy_process_group"
        and isinstance(func.value, ast.Name)
        and func.value.id == "dist"
    )


def _walk_calls(nodes: list[ast.AST], attr: str) -> list[ast.Call]:
    """Collect all ``X.attr(...)`` calls anywhere under ``nodes``."""
    found: list[ast.Call] = []
    for stmt in nodes:
        for sub in ast.walk(stmt):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == attr
            ):
                found.append(sub)
    return found


@pytest.fixture(scope="module")
def main_fn() -> ast.FunctionDef:
    tree = ast.parse(_TRAINER_PATH.read_text())
    return _find_main_function(tree)


class TestKernelProfilerStartIsInsideCleanupTry:
    """Pin the start-in-try invariant after Copilot review on PR #162."""

    def test_kernel_profiler_start_lives_inside_a_try_with_dist_destroy_in_finally(
        self, main_fn: ast.FunctionDef
    ):
        cleanup_try_blocks: list[ast.Try] = []
        for node in ast.walk(main_fn):
            if not isinstance(node, ast.Try):
                continue
            # Look for `dist.destroy_process_group()` anywhere in finalbody.
            if any(_is_dist_destroy_call(stmt) for stmt in node.finalbody):
                cleanup_try_blocks.append(node)

        assert len(cleanup_try_blocks) == 1, (
            f"expected exactly one try/finally that tears down the process "
            f"group; found {len(cleanup_try_blocks)}"
        )
        cleanup_try = cleanup_try_blocks[0]

        starts = _walk_calls(cleanup_try.body, "start")
        # Filter to ``kernel_profiler.start(...)`` specifically.
        kernel_starts = [
            call
            for call in starts
            if isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "kernel_profiler"
        ]
        assert kernel_starts, (
            "kernel_profiler.start() must be inside the same try-block whose "
            "finally calls dist.destroy_process_group(); pre-fix it was outside, "
            "so a start-time failure (bpftrace missing under "
            "skip_if_unavailable=False, or the new _await_startup permission "
            "raise) would leak the process group on every other rank"
        )

    def test_kernel_profiler_stop_remains_in_finally(self, main_fn: ast.FunctionDef):
        cleanup_try_blocks = [
            node
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Try)
            and any(_is_dist_destroy_call(stmt) for stmt in node.finalbody)
        ]
        assert cleanup_try_blocks, "no cleanup try/finally found"
        cleanup_try = cleanup_try_blocks[0]

        # ``kernel_profiler.stop()`` should be part of the finally block.
        stops = _walk_calls(cleanup_try.finalbody, "stop")
        kernel_stops = [
            call
            for call in stops
            if isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "kernel_profiler"
        ]
        assert kernel_stops, (
            "kernel_profiler.stop() must run from the cleanup finally so "
            "bpftrace is reaped on training crash"
        )
