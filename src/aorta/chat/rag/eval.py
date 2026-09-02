"""A retrieval eval small enough to actually run, per Decision 19b.

BGE-small was chosen over MiniLM on published MTEB figures, which are
general-text benchmarks over a Python codebase and markdown docs -- a weak
proxy. A code-specialised embedder may well beat it here. The point of this
module is that the next person to raise that question can answer it with
numbers from this corpus instead of a leaderboard screenshot:

    aorta chat index build            # with AORTA_CHAT_EMBEDDING_MODEL=A
    aorta chat index eval
    aorta chat index build            # with AORTA_CHAT_EMBEDDING_MODEL=B
    aorta chat index eval

Deliberately not a pytest suite. It needs a built index and a loaded model, so
it cannot run in the mock-only CI gate, and pretending otherwise would produce a
test that is skipped everywhere and trusted anyway. What *is* unit-tested is the
scoring: :func:`score` is pure, so recall and MRR are pinned without a model.

The metrics are the two that matter for a fixed-k RAG prompt. Recall@k is
whether the answer was available to the model at all -- a miss there is a
question the assistant cannot get right however good the LLM is. MRR says
whether it arrived near the top, which decides how much of the context window it
has to survive.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Question set shipped with the package, so ``index eval`` works with no
#: arguments. Paths are matched as suffixes (see :func:`_matches`), which keeps
#: the file robust against a module moving within its package.
QUESTIONS_FILE = Path(__file__).with_name("eval_questions.json")

#: Default retrieval depth. Matches ``settings.retriever_k``'s default so the
#: measurement describes what the graph actually gets, not a friendlier number.
DEFAULT_K = 12


class EvalError(RuntimeError):
    """The question set is missing or malformed."""


@dataclass(frozen=True)
class Question:
    """One probe: what a user would ask, and which files should come back.

    ``expected`` is a list of *acceptable* sources, not a required set. Several
    files can legitimately answer "how does the CLI defer its chat imports", so
    a hit on any one of them counts -- scoring the whole set as required would
    measure the question's phrasing rather than the embedder.
    """

    question: str
    expected: tuple[str, ...]
    note: str = ""


@dataclass
class QuestionResult:
    question: Question
    #: Retrieved source paths, best-first.
    retrieved: tuple[str, ...] = ()
    #: 1-based position of the first acceptable source, or 0 for a miss.
    rank: int = 0

    @property
    def hit(self) -> bool:
        return self.rank > 0


@dataclass
class EvalResult:
    """Scores over a whole question set, plus the per-question detail.

    The per-question detail is not optional colour: an aggregate that moved
    0.03 tells you nothing about *which* question regressed, and that is the
    only actionable part of a comparison between two embedders.
    """

    k: int
    embedding_model: str = ""
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def recall(self) -> float:
        """Fraction of questions with an acceptable source anywhere in the top k."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.hit) / len(self.results)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank of the first acceptable source; a miss scores 0."""
        if not self.results:
            return 0.0
        return sum(1.0 / r.rank if r.hit else 0.0 for r in self.results) / len(self.results)

    @property
    def misses(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.hit]

    def summary(self) -> str:
        return (
            f"{len(self.results)} questions, k={self.k}: "
            f"recall@{self.k} {self.recall:.2f}, MRR {self.mrr:.3f}, "
            f"{len(self.misses)} miss(es)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "embedding_model": self.embedding_model,
            "recall": round(self.recall, 4),
            "mrr": round(self.mrr, 4),
            "questions": [
                {
                    "question": r.question.question,
                    "expected": list(r.question.expected),
                    "rank": r.rank,
                    "retrieved": list(r.retrieved),
                }
                for r in self.results
            ],
        }


def load_questions(path: str | Path | None = None) -> list[Question]:
    """Read a question set from JSON, defaulting to the shipped one."""
    target = Path(path) if path else QUESTIONS_FILE
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvalError(f"no question set at {target}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"could not read the question set at {target}: {exc}") from exc

    entries = raw.get("questions") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise EvalError(f"{target} holds no questions (expected a list under 'questions')")

    questions: list[Question] = []
    for index, entry in enumerate(entries):
        try:
            expected = entry["expected"]
            questions.append(
                Question(
                    question=entry["question"],
                    expected=tuple(expected) if isinstance(expected, list) else (expected,),
                    note=entry.get("note", ""),
                )
            )
        except (KeyError, TypeError) as exc:
            raise EvalError(f"{target}: question {index} is missing {exc}") from exc
    return questions


def _matches(retrieved: str, expected: str) -> bool:
    """Whether a retrieved source path satisfies an expected one.

    Suffix matching, because ``source`` metadata is relative to whichever corpus
    base built the index: a CI-built index says ``src/aorta/cli/chat.py`` while
    an index built from the installed package says ``cli/chat.py``. Pinning the
    full path would make the question set valid for exactly one build mode.
    """
    left = retrieved.replace("\\", "/").strip("/")
    right = expected.replace("\\", "/").strip("/")
    return left == right or left.endswith("/" + right)


def score(questions: list[Question], retrieved_per_question: list[list[str]], k: int) -> EvalResult:
    """Score pre-retrieved results. Pure, so the metrics are unit-testable."""
    if len(questions) != len(retrieved_per_question):
        raise ValueError(
            f"{len(questions)} questions but {len(retrieved_per_question)} result lists"
        )
    result = EvalResult(k=k)
    for question, retrieved in zip(questions, retrieved_per_question, strict=True):
        top = tuple(retrieved[:k])
        rank = 0
        for position, source in enumerate(top, start=1):
            if any(_matches(source, expected) for expected in question.expected):
                rank = position
                break
        result.results.append(QuestionResult(question=question, retrieved=top, rank=rank))
    return result


def evaluate(
    questions: list[Question] | None = None,
    k: int = DEFAULT_K,
    search: Callable[[str, int], list[Any]] | None = None,
) -> EvalResult:
    """Run the question set against the live index.

    Args:
        questions: Defaults to the shipped set.
        k: Retrieval depth.
        search: Injection point for tests; defaults to the real retriever.
    """
    questions = questions if questions is not None else load_questions()
    if search is None:
        from aorta.chat.rag.retriever import search_docs

        search = search_docs

    retrieved: list[list[str]] = []
    for question in questions:
        documents = search(question.question, k)
        retrieved.append([doc.metadata.get("source", "") for doc in documents])

    result = score(questions, retrieved, k)
    # The provider that produced the vectors this run retrieved against, so a
    # before/after comparison is not silently attributed to the local model.
    from aorta.chat.rag.embeddings.factory import get_provider

    result.embedding_model = get_provider().model_id()
    return result


__all__ = [
    "DEFAULT_K",
    "QUESTIONS_FILE",
    "EvalError",
    "EvalResult",
    "Question",
    "QuestionResult",
    "evaluate",
    "load_questions",
    "score",
]
