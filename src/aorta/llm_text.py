"""Dependency-light parsing of model answer text."""

from __future__ import annotations

_REASONING_OPEN = "<think>"
_REASONING_CLOSE = "</think>"


def strip_code_fence(content: str) -> str:
    """Unwrap one leading fenced block."""
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
    return "\n".join(body).strip()


def strip_reasoning(content: str) -> str:
    """Drop one terminated leading Qwen/DeepSeek reasoning block."""
    text = content.strip()
    if text.startswith(_REASONING_OPEN):
        end = text.find(_REASONING_CLOSE, len(_REASONING_OPEN))
        if end == -1:
            raise ValueError("reasoning block never closed; the generation ended before an answer")
        answer = text[end + len(_REASONING_CLOSE) :]
    elif _REASONING_CLOSE in text and not text.startswith(("{", "```")):
        answer = text.split(_REASONING_CLOSE, 1)[1]
    else:
        return text
    answer = answer.strip()
    if not answer:
        raise ValueError("reply was reasoning with no answer after it")
    return answer


def answer_text(content: str) -> str:
    """Return the JSON-bearing answer after reasoning and fence removal."""
    return strip_code_fence(strip_reasoning(content))


__all__ = ["answer_text", "strip_code_fence", "strip_reasoning"]
