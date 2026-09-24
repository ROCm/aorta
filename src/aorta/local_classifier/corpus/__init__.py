"""Phase 0: turn artifacts already on disk into labelled JSONL.

Three builders, one record type. Each reads what a real run left behind and
labels it from an *observation* -- an Autopsy verdict that arrived later, a probe
cell that passed, a file Watch read bytes out of -- never from the rules of the
prompt it is meant to replace. Labelling by the rules is how a classifier comes
to reproduce the heuristic that is already in the cheap tier, and the resulting
accuracy looks like success.

Every builder returns a :class:`~aorta.local_classifier.corpus.schema.BuildResult` carrying
what it declined to label and why, because on this codebase's artifacts the skip
counts are the more interesting half of the output.
"""

from __future__ import annotations

from aorta.local_classifier.corpus.log_finder import build_log_finder_corpus
from aorta.local_classifier.corpus.proposer import build_proposer_corpus
from aorta.local_classifier.corpus.schema import (
    DECISIONS,
    BuildResult,
    CorpusError,
    LabelledExample,
    read_corpus,
    write_corpus,
)
from aorta.local_classifier.corpus.watch import build_watch_corpus

#: Corpus kind -> the builder and what it reads, for the CLI and for a reader.
BUILDERS = {
    "watch": (build_watch_corpus, "CIA job directories (CIA_JOBS_ROOT, default ~/cia-jobs)"),
    "proposer": (build_proposer_corpus, "aorta agent mitigate run directories"),
    "log-finder": (build_log_finder_corpus, "CIA job directories, for the listing decision"),
}

__all__ = [
    "BUILDERS",
    "DECISIONS",
    "BuildResult",
    "CorpusError",
    "LabelledExample",
    "build_log_finder_corpus",
    "build_proposer_corpus",
    "build_watch_corpus",
    "read_corpus",
    "write_corpus",
]
