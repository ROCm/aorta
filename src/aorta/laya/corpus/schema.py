"""What a labelled example is, and how it round-trips through JSONL.

One record type for all three corpora, because they are all scored by the same
harness and a second shape would mean a second scorer. What differs between them
is the ``decision`` slug, the question type and where the label came from -- all
of which travel in the record, so a corpus file is self-describing and a mixed
file still scores correctly.

Two fields exist purely so a number can be argued with later:

* ``source`` names the file the state was read out of. An eval that says
  accuracy moved and cannot say which artifact moved it is not evidence.
* ``baselines`` carries what the *current* system answered for this same
  example, recorded at build time from artifacts rather than recomputed. The
  Phase 1 gate has to beat the existing DSPy assessment's own agreement with
  the eventual autopsy category, and that comparison is only honest if the
  baseline's answer is the one it actually gave on that run rather than one
  re-derived from a prompt that has since been edited.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aorta.laya.predictor import Choice, Noul, Question

#: Every decision these builders can label, and which track owns it.
#:
#: Spelled out rather than derived from the builders, because the eval groups by
#: decision and a typo would silently produce a group of one that scores 1.0.
DECISIONS: dict[str, str] = {
    "watch_healthy": "Track A: is this log delta healthy, so no ReAct call is needed",
    "watch_signal": "Track A: which of Watch's seven signal slugs this delta is",
    "proposer_mitigation": "Track B: which candidate mitigation clears the cell (one choice)",
    "proposer_candidate": "Track B: would this one candidate clear the cell (one noul each)",
    "proposer_stop": "Track B: should the mitigation search stop here",
    "proposer_category": "Track B: which probe category the sweep reaches",
    "log_finder_useful": "Track C: is this listed file worth watching",
}


class CorpusError(RuntimeError):
    """A corpus file is missing, unreadable, or not a corpus."""


@dataclass(frozen=True)
class LabelledExample:
    """One state, one typed question, one label that came from an observation.

    *label* is an option name: ``"true"`` / ``"false"`` for a noul, and one of
    ``question.options`` for a choice. A label outside the option set is refused
    at construction rather than at scoring time, because at scoring time it
    reads as the model getting the answer wrong.
    """

    decision: str
    state: str
    question: Question
    label: str
    join_key: str
    source: str = ""
    baselines: tuple[tuple[str, str], ...] = ()
    #: One decision that several rows jointly answer, or "" for a row that stands
    #: alone.
    #:
    #: Two corpora ask N separate yes/no questions where the real decision is
    #: "which one of these N" -- the probe agent picking a mitigation, the log
    #: finder picking files out of a listing. Accuracy over the rows is the wrong
    #: score for that: one candidate in twenty-one is labelled true, so answering
    #: false to everything scores 0.95 while choosing nothing. The group is what
    #: lets :attr:`aorta.laya.eval.EvalResult.group_top1` score the choice that was
    #: actually being made.
    #:
    #: It is also the split key wherever it is set, because rows that jointly
    #: answer one question must not be divided between the temperature fit and the
    #: scored set. That is why it is the *step* for the proposer rather than the
    #: individual question: the choice-shaped row and the noul-shaped rows for one
    #: step share it, so a head-to-head between the two shapes compares them over
    #: the same steps rather than over two different halves of the corpus.
    group: str = ""

    def __post_init__(self) -> None:
        if self.label not in self.question.options:
            raise CorpusError(
                f"{self.decision}/{self.join_key}: label {self.label!r} is not one of "
                f"the offered options {list(self.question.options)}"
            )

    def baseline(self, name: str) -> str | None:
        """What the named existing system answered, or None if it did not."""
        for recorded, answer in self.baselines:
            if recorded == name:
                return answer
        return None

    def to_dict(self) -> dict[str, Any]:
        question: dict[str, Any] = {"type": "noul"}
        if isinstance(self.question, Choice):
            question = {
                "type": "choice",
                "options": list(self.question.options),
                "criteria": [list(pair) for pair in self.question.criteria],
            }
        elif self.question.when_true or self.question.when_false:
            question["when_true"] = self.question.when_true
            question["when_false"] = self.question.when_false
        question["instructions"] = self.question.question
        return {
            "decision": self.decision,
            "join_key": self.join_key,
            "group": self.group,
            "state": self.state,
            "question": question,
            "label": self.label,
            "source": self.source,
            "baselines": {name: answer for name, answer in self.baselines},
        }

    @classmethod
    def from_dict(cls, raw: Any) -> LabelledExample:
        if not isinstance(raw, dict):
            raise CorpusError(f"expected a JSON object per line, got {type(raw).__name__}")
        try:
            spec = raw["question"]
            kind = spec["type"]
            if kind == "choice":
                question: Question = Choice(
                    question=spec["instructions"],
                    options=tuple(spec["options"]),
                    criteria=tuple((name, text) for name, text in spec.get("criteria") or ()),
                )
            elif kind == "noul":
                question = Noul(
                    question=spec["instructions"],
                    when_true=spec.get("when_true", ""),
                    when_false=spec.get("when_false", ""),
                )
            else:
                # ``score`` is the third primitive and is deliberately not
                # modelled anywhere in this package; a corpus carrying one was
                # written by something else.
                raise CorpusError(f"unsupported question type {kind!r}")
            return cls(
                decision=raw["decision"],
                state=raw["state"],
                question=question,
                label=raw["label"],
                join_key=raw.get("join_key", ""),
                source=raw.get("source", ""),
                baselines=tuple(sorted((raw.get("baselines") or {}).items())),
                # Tolerated as absent so a corpus written before the field existed
                # still reads. It scores as a row that stands alone, which is what
                # it was when it was written.
                group=raw.get("group", ""),
            )
        except (KeyError, TypeError) as exc:
            raise CorpusError(f"malformed example: missing or wrong-typed {exc}") from exc


@dataclass
class BuildResult:
    """What a builder produced, and everything it declined to produce.

    The skip counts are not diagnostics; they are half the output. A builder
    that walked four hundred job directories and emitted nine examples has told
    you something important about Phase 0, and a bare list of nine examples has
    not. Likewise *warnings*: a corpus whose negative class is systematically
    shorter than its positive class will train a length detector, and that has
    to be said at the moment the file is written rather than discovered when the
    accuracy looks too good.
    """

    examples: list[LabelledExample] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)
    scanned: int = 0

    def skip(self, reason: str, count: int = 1) -> None:
        self.skipped[reason] += count

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def by_decision(self) -> Counter[str]:
        return Counter(example.decision for example in self.examples)

    def summary(self) -> str:
        counts = ", ".join(
            f"{decision} {count}" for decision, count in sorted(self.by_decision().items())
        )
        return (
            f"{len(self.examples)} example(s) from {self.scanned} candidate(s)"
            + (f": {counts}" if counts else "")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": len(self.examples),
            "scanned": self.scanned,
            "by_decision": dict(sorted(self.by_decision().items())),
            "skipped": dict(sorted(self.skipped.items())),
            "warnings": list(self.warnings),
        }


def write_corpus(path: str | Path, examples: Iterable[LabelledExample]) -> int:
    """Write *examples* as JSONL and return how many were written.

    JSONL rather than one JSON array, for the reason the run artifacts and the
    Watch events file are JSONL: a corpus is appended to over weeks as more jobs
    accumulate, and a truncated write costs one line rather than the file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")
            written += 1
    return written


def read_corpus(path: str | Path) -> list[LabelledExample]:
    """Read a JSONL corpus, naming the line that is wrong.

    A malformed line raises rather than being skipped. The retrieval eval's
    question set takes the same line for the same reason: a corpus that quietly
    drops a tenth of itself reports a score over a set nobody chose, and the
    number looks entirely normal.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CorpusError(
            f"no corpus at {target}. Build one first with 'aorta laya corpus <kind>'."
        ) from exc
    except OSError as exc:
        raise CorpusError(f"could not read the corpus at {target}: {exc}") from exc

    examples: list[LabelledExample] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(LabelledExample.from_dict(json.loads(line)))
        except (json.JSONDecodeError, CorpusError) as exc:
            raise CorpusError(f"{target}:{number}: {exc}") from exc
    if not examples:
        raise CorpusError(f"{target} holds no examples")
    return examples


def iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Every JSON object in a JSONL artifact, skipping what cannot be parsed.

    Tolerant where :func:`read_corpus` is strict, and the difference is whose
    file it is. This reads artifacts written by a live poll loop or an agent run
    that may have been killed mid-write, where a truncated tail is expected;
    that reads a corpus this package wrote itself, where it is a bug.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


__all__ = [
    "DECISIONS",
    "BuildResult",
    "CorpusError",
    "LabelledExample",
    "iter_json_lines",
    "read_corpus",
    "write_corpus",
]
