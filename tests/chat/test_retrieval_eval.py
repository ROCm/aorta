"""The retrieval eval's scoring, and the shipped question set's integrity.

Decision 19b ships BGE-small now and defers the model bake-off to measurement.
That only works if the measurement is trustworthy, so :func:`score` is pure and
pinned here. The end-to-end run needs a built index and a loaded model, so it is
deliberately not a test -- a test that skips everywhere gets trusted anyway.

The question set is validated as data: every entry parses, and the expected
sources are real paths in this tree. A question expecting a file that no longer
exists scores a permanent miss and quietly drags the baseline down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from aorta.chat.rag.eval import (
    QUESTIONS_FILE,
    EvalError,
    Question,
    evaluate,
    load_questions,
    score,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _q(*expected: str) -> Question:
    return Question(question="does it work?", expected=expected)


class TestScoring:
    def test_a_first_place_hit_scores_perfectly(self):
        result = score([_q("cli/chat.py")], [["src/aorta/cli/chat.py", "other.py"]], k=5)
        assert result.recall == 1.0
        assert result.mrr == 1.0
        assert result.results[0].rank == 1

    def test_a_third_place_hit_scores_one_third(self):
        result = score([_q("c.py")], [["a.py", "b.py", "c.py"]], k=5)
        assert result.recall == 1.0
        assert result.mrr == pytest.approx(1 / 3)
        assert result.results[0].rank == 3

    def test_a_miss_scores_zero_on_both(self):
        result = score([_q("z.py")], [["a.py", "b.py"]], k=5)
        assert result.recall == 0.0
        assert result.mrr == 0.0
        assert not result.results[0].hit

    def test_a_hit_past_k_is_a_miss(self):
        """The graph only sees the top k, so neither should the score."""
        result = score([_q("c.py")], [["a.py", "b.py", "c.py"]], k=2)
        assert result.recall == 0.0

    def test_any_one_acceptable_source_counts(self):
        """Several files can legitimately answer one question.

        Requiring the whole set would measure the question's phrasing rather
        than the embedder.
        """
        result = score([_q("a.py", "b.py")], [["b.py"]], k=5)
        assert result.recall == 1.0

    def test_the_best_ranked_acceptable_source_wins(self):
        result = score([_q("b.py", "a.py")], [["a.py", "b.py"]], k=5)
        assert result.results[0].rank == 1

    def test_scores_average_over_the_whole_set(self):
        result = score([_q("a.py"), _q("z.py")], [["a.py"], ["b.py"]], k=5)
        assert result.recall == 0.5
        assert result.mrr == 0.5

    def test_an_empty_set_scores_zero_rather_than_dividing_by_zero(self):
        assert score([], [], k=5).recall == 0.0

    def test_mismatched_lengths_are_a_programming_error(self):
        with pytest.raises(ValueError, match="1 questions but 2 result lists"):
            score([_q("a.py")], [["a.py"], ["b.py"]], k=5)

    def test_misses_are_listed_for_diagnosis(self):
        """An aggregate that moved 0.03 says nothing about which question moved."""
        result = score([_q("a.py"), _q("z.py")], [["a.py"], ["b.py"]], k=5)
        assert len(result.misses) == 1
        assert result.misses[0].question.expected == ("z.py",)

    def test_the_summary_names_both_metrics_and_the_depth(self):
        text = score([_q("a.py")], [["a.py"]], k=7).summary()
        assert "recall@7" in text
        assert "MRR" in text


class TestSuffixMatching:
    """Sources are relative to whichever corpus base built the index."""

    def test_a_ci_built_path_matches_a_package_relative_expectation(self):
        result = score([_q("cli/chat.py")], [["src/aorta/cli/chat.py"]], k=5)
        assert result.recall == 1.0

    def test_a_package_built_path_matches_too(self):
        result = score([_q("cli/chat.py")], [["cli/chat.py"]], k=5)
        assert result.recall == 1.0

    def test_a_partial_component_does_not_match(self):
        """``at.py`` must not satisfy an expectation of ``chat.py``."""
        result = score([_q("hat.py")], [["src/aorta/cli/chat.py"]], k=5)
        assert result.recall == 0.0

    def test_windows_separators_are_normalised(self):
        result = score([_q("cli/chat.py")], [["src\\aorta\\cli\\chat.py"]], k=5)
        assert result.recall == 1.0


class TestShippedQuestionSet:
    def test_it_parses(self):
        questions = load_questions()
        assert len(questions) >= 10

    def test_every_question_has_at_least_one_expected_source(self):
        assert all(question.expected for question in load_questions())

    def test_every_expected_source_exists_in_this_tree(self):
        """A question expecting a deleted file is a permanent, silent miss."""
        package = _REPO_ROOT / "src" / "aorta"
        missing = [
            expected
            for question in load_questions()
            for expected in question.expected
            if not (package / expected).exists() and not (_REPO_ROOT / expected).exists()
        ]
        assert missing == []

    def test_questions_are_phrased_as_questions_not_as_symbol_lookups(self):
        """Retrieving a file whose name is in the query measures nothing."""
        for question in load_questions():
            stems = {Path(expected).stem for expected in question.expected}
            words = set(question.question.lower().replace("?", "").split())
            assert not (stems & words), question.question

    def test_it_is_declared_as_package_data(self):
        """Otherwise `index eval` works from a checkout and not from a wheel."""
        pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "chat/rag/eval_questions.json" in pyproject

    def test_the_shipped_file_is_where_the_module_looks(self):
        assert QUESTIONS_FILE.exists()


class TestLoadFailures:
    def test_a_missing_file_is_reported(self, tmp_path: Path):
        with pytest.raises(EvalError, match="no question set at"):
            load_questions(tmp_path / "absent.json")

    def test_malformed_json_is_reported(self, tmp_path: Path):
        path = tmp_path / "q.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(EvalError, match="could not read"):
            load_questions(path)

    def test_an_empty_set_is_reported(self, tmp_path: Path):
        path = tmp_path / "q.json"
        path.write_text(json.dumps({"questions": []}), encoding="utf-8")
        with pytest.raises(EvalError, match="holds no questions"):
            load_questions(path)

    def test_a_bare_list_is_accepted(self, tmp_path: Path):
        path = tmp_path / "q.json"
        path.write_text(json.dumps([{"question": "q?", "expected": ["a.py"]}]), encoding="utf-8")
        assert load_questions(path)[0].expected == ("a.py",)

    def test_a_string_expected_value_is_accepted(self, tmp_path: Path):
        path = tmp_path / "q.json"
        path.write_text(json.dumps([{"question": "q?", "expected": "a.py"}]), encoding="utf-8")
        assert load_questions(path)[0].expected == ("a.py",)

    def test_a_question_missing_its_expected_key_names_the_index(self, tmp_path: Path):
        path = tmp_path / "q.json"
        path.write_text(json.dumps([{"question": "q?"}]), encoding="utf-8")
        with pytest.raises(EvalError, match="question 0 is missing"):
            load_questions(path)


class TestEvaluate:
    def test_it_scores_an_injected_retriever_without_an_index(self, monkeypatch):
        from aorta.chat.config import settings

        monkeypatch.setattr(settings, "embedding_model", "fake/model")
        questions = [_q("hit.py"), _q("never.py")]

        def _search(query: str, k: int) -> list[Document]:
            return [Document(page_content="x", metadata={"source": "src/hit.py"})]

        result = evaluate(questions, k=4, search=_search)
        assert result.recall == 0.5
        assert result.embedding_model == "fake/model"
        assert result.k == 4

    def test_the_json_form_carries_the_per_question_detail(self, monkeypatch):
        def _search(query: str, k: int) -> list[Document]:
            return [Document(page_content="x", metadata={"source": "a.py"})]

        payload = evaluate([_q("a.py")], k=3, search=_search).to_dict()
        assert payload["recall"] == 1.0
        assert payload["questions"][0]["rank"] == 1
        assert payload["questions"][0]["retrieved"] == ["a.py"]

    def test_a_document_with_no_source_metadata_is_a_miss_not_a_crash(self):
        def _search(query: str, k: int) -> list[Document]:
            return [Document(page_content="x", metadata={})]

        assert evaluate([_q("a.py")], k=3, search=_search).recall == 0.0
