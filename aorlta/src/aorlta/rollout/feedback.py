"""Compile feedback for repair turns (design Section 6.1, Feedback content).

The stand-in template:

    Your code did not compile (hipcc --offload-arch=gfx950 -c). Errors:
    1. {cls} at line {line}, column {col}: `{token}`
    2. ...
    Fix the errors and reply with the complete corrected source in a single ```cpp code block.

- `cls` is the clang message with every quoted name replaced by `<name>`. A quoted span counts as a
  name when it holds a letter, digit or underscore, so `expected ';'` keeps its `';'`.
- `line` and `col` come from the diagnostic.
- `token` is the model's own token at that position in its own source: the longest match of
  `[A-Za-z_][A-Za-z0-9_]*|[0-9]+|\\S` starting at the 1-based byte column `col`.
- Only errors are listed, in diagnostic order. Warnings, notes, include stacks, source excerpts and
  paths are never included.
- An error located in another file (a header) takes the position of its first note in the candidate's
  own source, and has no position if there is none.
- Exact duplicates (same class, position and token), which clang can print once per compilation
  pass, are listed once.
- Entries are added while the numbered list stays within `max_chars`. The cut falls at an entry
  boundary, followed by a final line `(N more errors not shown)`.

Errors are structured first (`CompileResult.errors`), and the text is built from that structure only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .templates import Tokenizer, repair_suffix

HEADER = "Your code did not compile (hipcc --offload-arch=gfx950 -c). Errors:"
FOOTER = "Fix the errors and reply with the complete corrected source in a single ```cpp code block."
MORE = "({n} more errors not shown)"
MAX_CHARS = 2000

_LOC_RE = re.compile(
    r"^(?P<file>[^\s:][^:]*):(?P<line>[0-9]+):(?P<col>[0-9]+): (?P<sev>fatal error|error|warning|note|remark): (?P<msg>.*)$"
)
_NOLOC_RE = re.compile(r"^(?:(?P<file>[^\s:][^:]*): )?(?P<sev>fatal error|error|warning|note|remark): (?P<msg>.*)$")
_QUOTED_RE = re.compile(r"'[^'\n]*'")
_NAMEISH_RE = re.compile(r"[A-Za-z0-9_]")
_PART = r"[\w.+@~-]+(?::[\w.+@~-]+)*"  # a path component, which may hold a colon
# an absolute path, or (what a space in a path leaves) a relative one
_PATH_RE = re.compile(rf"(?<![\w<>/.])(?:/{_PART})+/?|(?<![\w<>/.:]){_PART}(?:/{_PART})+/?")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|\S")
_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")
_CONTROL_TEXTS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")
ERROR_SEVERITIES = ("error", "fatal error")


@dataclass(frozen=True)
class Diagnostic:
    file: str | None
    line: int | None
    col: int | None
    severity: str
    message: str


def parse_diagnostics(stderr: str) -> list[Diagnostic]:
    """Every clang diagnostic line, located (`file:line:col: severity: message`) or not
    (`clang: error: ...`). Include stacks, excerpts, carets and summaries are skipped."""
    out = []
    for raw in _LINE_BREAK_RE.split(stderr):
        m = _LOC_RE.match(raw)
        if m:
            out.append(Diagnostic(m["file"], int(m["line"]), int(m["col"]), m["sev"], m["msg"]))
            continue
        m = _NOLOC_RE.match(raw)
        if m:
            out.append(Diagnostic(m["file"], None, None, m["sev"], m["msg"]))
    return out


def error_class(message: str) -> str:
    """The message with quoted names replaced by `<name>`, and any path by `<path>`."""

    def quoted(m: re.Match) -> str:
        return "<name>" if _NAMEISH_RE.search(m.group(0)) else m.group(0)

    cls = _QUOTED_RE.sub(quoted, message.strip())
    cls = _PATH_RE.sub("<path>", cls)
    for text in _CONTROL_TEXTS:
        cls = cls.replace(text, "<name>")
    return cls


def token_at(source: str, line: int | None, col: int | None) -> str | None:
    """The model's own token at 1-based `line` and 1-based byte column `col` of `source`, or None when
    the position is outside the source, splits a character, or holds whitespace."""
    if line is None or col is None or line < 1 or col < 1:
        return None
    lines = _LINE_BREAK_RE.split(source)
    if line > len(lines):
        return None
    text = lines[line - 1]
    raw = text.encode("utf-8")
    if col - 1 >= len(raw):
        return None
    try:
        start = len(raw[: col - 1].decode("utf-8"))
    except UnicodeDecodeError:
        return None
    m = _TOKEN_RE.match(text, start)
    return m.group(0) if m else None


def _same_file(reported: str | None, source_name: str) -> bool:
    return reported is not None and os.path.normpath(reported) == os.path.normpath(source_name)


def compile_errors(stderr: str, source: str, source_name: str) -> list[dict]:
    """Structured errors `[{"cls", "line", "col", "token"}]`, in diagnostic order."""
    diags = parse_diagnostics(stderr)
    out: list[dict] = []
    seen: set[tuple] = set()
    for i, d in enumerate(diags):
        if d.severity not in ERROR_SEVERITIES:
            continue
        line, col = (d.line, d.col) if _same_file(d.file, source_name) else (None, None)
        if line is None and d.line is not None:
            for note in diags[i + 1:]:
                if note.severity != "note":
                    break
                if note.line is not None and _same_file(note.file, source_name):
                    line, col = note.line, note.col
                    break
        entry = {"cls": error_class(d.message), "line": line, "col": col, "token": token_at(source, line, col)}
        key = (entry["cls"], line, col, entry["token"])
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def format_entry(n: int, error: dict) -> str:
    text = f"{n}. {error['cls']}"
    if error.get("line") is not None:
        text += f" at line {error['line']}, column {error['col']}"
        if error.get("token") is not None:
            text += f": `{error['token']}`"
    return text


@dataclass(frozen=True)
class Feedback:
    text: str
    shown: int  # entries in the list
    total: int  # errors given


def build_feedback(errors: Sequence[dict], max_chars: int = MAX_CHARS) -> Feedback:
    """The feedback text for one failed compile. The numbered list stays within `max_chars`."""
    if not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError(f"max_chars must be an int >= 1, got {max_chars!r}")
    entries: list[str] = []
    used = 0
    for error in errors:
        entry = format_entry(len(entries) + 1, error)
        cost = len(entry) + (1 if entries else 0)
        if used + cost > max_chars:
            break
        entries.append(entry)
        used += cost
    lines = [HEADER, *entries]
    hidden = len(errors) - len(entries)
    if hidden:
        lines.append(MORE.format(n=hidden))
    lines.append(FOOTER)
    return Feedback("\n".join(lines), len(entries), len(errors))


def numbered_list(feedback_text: str) -> str:
    """The numbered list inside a feedback text (what `max_chars` bounds), for checks."""
    lines = feedback_text.split("\n")
    if len(lines) < 2 or lines[0] != HEADER or lines[-1] != FOOTER:
        raise ValueError("not a stand-in feedback text")
    body = lines[1:-1]
    if body and re.fullmatch(r"\([0-9]+ more errors not shown\)", body[-1]):
        body = body[:-1]
    return "\n".join(body)


def feedback_token_bound(tokenizer: Tokenizer, max_chars: int, thinking: str) -> int:
    """Tokens of the repair suffix around the longest list, for sizing `max_model_len`: the fixed text plus at most
    one token per character of the list [E; the per-chain length check before each submit is the exact guard]."""
    fixed = tokenizer.encode(repair_suffix(HEADER + "\n" + MORE.format(n=99999) + "\n" + FOOTER, thinking))
    return len(fixed) + max_chars
