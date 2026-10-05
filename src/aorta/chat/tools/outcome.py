"""Structured status carried alongside user-readable tool output."""

from __future__ import annotations


class ToolOutput(str):
    """A string result with machine-readable execution status."""

    failed: bool

    def __new__(cls, value: object, *, failed: bool = False) -> "ToolOutput":
        result = super().__new__(cls, str(value))
        result.failed = failed
        return result


def tool_failure(message: object) -> ToolOutput:
    """Return a readable failure that progress reporting can classify safely."""
    return ToolOutput(message, failed=True)


def tool_result_failed(result: object) -> bool:
    """Whether *result* explicitly records a failed tool execution."""
    return isinstance(result, ToolOutput) and result.failed
