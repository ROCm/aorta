"""C1.T4: the feedback builder on 20 canned clang outputs (design Section 6.1, Feedback content).

The canned outputs are real: hipcc --offload-arch=gfx950 -c -O1 on 20 small broken sources, recorded twice
(tests/rollout/data/clang/README.txt): with a node's ROCm 7.0.2 hipcc (`stderr.txt`,
`expected.json`) and with the rollout image's ROCm 7.2.4 hipcc (`stderr-rocm-7.2.4.txt`,
`expected-rocm-7.2.4.json`). The goldens were reviewed by eye.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from aorlta.rollout.feedback import (
    FOOTER, HEADER, MAX_CHARS, build_feedback, compile_errors, error_class, numbered_list, parse_diagnostics, token_at,
)

DATA = Path(__file__).resolve().parent / "data" / "clang"
CASES = sorted(p.name for p in DATA.iterdir() if p.is_dir())
VERSIONS = ("", "-rocm-7.2.4")  # the node's ROCm 7.0.2 hipcc; the rollout image's ROCm 7.2.4 hipcc
BOTH = [(c, v) for v in VERSIONS for c in CASES]


def _load(case: str, version: str = "") -> tuple[str, str, dict]:
    d = DATA / case
    return ((d / "kernel.hip").read_text(encoding="utf-8"),
            (d / f"stderr{version}.txt").read_text(encoding="utf-8"),
            json.loads((d / f"expected{version}.json").read_text(encoding="utf-8")))


def test_there_are_20_canned_outputs_all_real_compile_failures():
    assert len(CASES) == 20
    for case in CASES:
        for version in VERSIONS:
            assert (DATA / case / f"rc{version}.txt").read_text().strip() == "1"
            assert "error" in (DATA / case / f"stderr{version}.txt").read_text()


def test_both_compiler_versions_give_the_same_feedback_on_18_of_20():
    same = [c for c in CASES if _load(c)[2]["errors"] == _load(c, "-rocm-7.2.4")[2]["errors"]]
    assert sorted(set(CASES) - set(same)) == ["03-unknown-type-name", "12-error-in-header"]


@pytest.mark.parametrize("case, version", BOTH)
def test_c1_t4_golden_errors_and_feedback(case, version):
    source, stderr, golden = _load(case, version)
    errors = compile_errors(stderr, source, "kernel.hip")
    assert errors == golden["errors"]
    fb = build_feedback(errors)
    assert fb.text.split("\n") == golden["feedback"]
    assert (fb.shown, fb.total) == (golden["shown"], golden["total"])


@pytest.mark.parametrize("case, version", BOTH)
def test_c1_t4_no_paths_notes_or_warnings_and_tokens_are_the_models_own(case, version):
    source, stderr, golden = _load(case, version)
    fb = build_feedback(compile_errors(stderr, source, "kernel.hip"))
    assert not re.search(r"/(?:opt|apps|home|tmp|usr)\b|\.hip\b|\.h\b|failed to execute", fb.text)
    assert "note:" not in fb.text and "warning" not in fb.text and "|" not in numbered_list(fb.text)
    assert len(numbered_list(fb.text)) <= MAX_CHARS
    lines = source.split("\n")
    for e in golden["errors"]:
        if e["token"] is not None:
            raw = lines[e["line"] - 1].encode("utf-8")
            assert raw[e["col"] - 1:].decode("utf-8").startswith(e["token"])


def test_c1_t4_the_cap_falls_at_an_entry_boundary_with_a_count_of_the_rest():
    source, stderr, golden = _load("09-too-many-errors")
    errors = compile_errors(stderr, source, "kernel.hip")
    assert len(errors) == 20  # 19 undeclared identifiers, then clang's "too many errors" stop
    fb = build_feedback(errors)
    body = fb.text.split("\n")
    entries = [ln for ln in body if re.match(r"[0-9]+\. ", ln)]
    assert len(entries) == fb.shown == 15 and entries[-1].startswith("15. ")
    assert body[-2] == "(5 more errors not shown)" and body[0] == HEADER and body[-1] == FOOTER
    listed = numbered_list(fb.text)
    assert len(listed) <= MAX_CHARS < len(listed) + 1 + len(entries[-1])  # one more entry would not fit
    for max_chars in (1, 50, 131, 132, 300, 1999, 2000, 10_000):
        small = build_feedback(errors, max_chars)
        shown = [ln for ln in small.text.split("\n") if re.match(r"[0-9]+\. ", ln)]
        assert len(numbered_list(small.text)) <= max_chars
        assert shown == entries[: len(shown)] if len(shown) <= 15 else True
        assert small.shown == len(shown) and (small.total - small.shown == 0) == ("more errors" not in small.text)


def test_error_class_replaces_quoted_names_keeps_punctuation_and_drops_paths():
    assert error_class("use of undeclared identifier 'tid'") == "use of undeclared identifier <name>"
    assert error_class("expected ';' after expression") == "expected ';' after expression"
    assert error_class("cannot initialize a variable of type 'int *' with an lvalue of type 'float *'") == \
        "cannot initialize a variable of type <name> with an lvalue of type <name>"
    assert error_class("no such file or directory: /scratch/x/kernel.hip") == "no such file or directory: <path>"
    assert error_class("bad '<|im_end|>' and <think>") == "bad <name> and <name>"


def test_token_at_uses_one_based_byte_columns():
    src = "a\n\tint x = yy;\n/* déjà → */ z = q1 + 42;\n"
    assert token_at(src, 2, 10) == "yy"  # a tab is one byte
    assert token_at(src, 3, token_col := len("/* déjà → */ z = ".encode()) + 1) == "q1"
    assert token_at(src, 3, token_col + 5) == "42"
    assert token_at(src, 3, 5) == "é"  # a one-character \S token, two bytes
    assert token_at(src, 3, 6) is None  # inside that character
    assert token_at(src, 2, 99) is None and token_at(src, 9, 1) is None and token_at(src, None, None) is None
    assert token_at(src, 2, 5) is None  # whitespace: no token
    assert token_at("x = a->b;", 1, 6) == "-"


def test_diagnostics_parse_located_and_unlocated_lines_only():
    stderr = ("In file included from kernel.hip:3:\n/opt/x.h:2:3: error: bad 'thing'\n   12 | code: error: no\n"
              "      |   ^\nkernel.hip:4:6: note: here\nclang: error: unable to execute command\n"
              "failed to execute:/opt/rocm/bin/clang++ -c kernel.hip\n2 errors generated when compiling for gfx950.\n")
    d = parse_diagnostics(stderr)
    assert [(x.file, x.line, x.col, x.severity) for x in d] == [
        ("/opt/x.h", 2, 3, "error"), ("kernel.hip", 4, 6, "note"), ("clang", None, None, "error")]
    errors = compile_errors(stderr, "a\nb\nc\nxxxx yy\n", "kernel.hip")
    assert errors == [{"cls": "bad <name>", "line": 4, "col": 6, "token": "yy"},  # a header error takes its note's spot
                      {"cls": "unable to execute command", "line": None, "col": None, "token": None}]
    text = build_feedback(errors).text
    assert "2. unable to execute command\n" in text and "/opt" not in text


def test_duplicate_errors_from_two_passes_are_listed_once():
    one = "kernel.hip:1:1: error: use of undeclared identifier 'a'\n"
    assert len(compile_errors(one * 2, "a\n", "kernel.hip")) == 1
