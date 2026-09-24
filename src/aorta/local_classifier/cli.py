"""``python -m aorta.local_classifier`` -- build the decision corpora and measure a model.

Maintainer tooling, not a product command: it is deliberately absent from the
``aorta`` console script, and is reached through this package's ``__main__``
the way ``python -m aorta.report`` and ``python -m aorta.hw_queue_eval`` are.
The only Click code for :mod:`aorta.local_classifier`, which otherwise contains
none, on the same split ``aorta chat`` uses and for the same reason.

**Module scope is click plus stdlib, deliberately.** Nothing in
:mod:`aorta.local_classifier` pulls torch by itself, so the letter of the rule
that ``aorta/cli/chat.py`` follows would not require deferring these imports --
but the reason does. ``eval`` resolves a checkpoint and the log-finder corpus
builder reaches into ``aorta.cia.watch`` and so into dspy, and neither belongs
on the path of ``--help``. The same discipline also means a missing extra is
reported as a sentence rather than as a traceback, which is the whole point of
``_load`` in the chat command.

Two commands, in the order Phase 0 and Phase 1 happen:

    python -m aorta.local_classifier corpus watch --root ~/cia-jobs --output watch.jsonl
    python -m aorta.local_classifier eval --corpus watch.jsonl --backend laya-typed-decisions

``eval`` refuses to run against the fake predictor for anything but a smoke test
of its own plumbing, and says so. A fake answers from a hash, and a results file
full of hashes is indistinguishable from a measurement.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import click

#: Corpus kinds, hard-coded for the reason ``aorta/cli/chat.py`` hard-codes its
#: provider list: a ``click.Choice`` is built when the decorator runs, so
#: enumerating the builders here would import them on every ``aorta --help``.
#: ``tests/cia/test_local_classifier_cli.py`` fails if this list and ``corpus.BUILDERS``
#: drift apart.
_CORPUS_KINDS = ("watch", "proposer", "log-finder")

#: Likewise against ``aorta.local_classifier.predictor.CHECKPOINTS``, plus ``fake``.
_BACKENDS = ("fake", "laya", "laya-typed-decisions")

_INSTALL_HINT = (
    "A real typed-decision predictor needs the local-classifier extra.\n"
    "Install it with:  pip install 'amd-aorta[local-classifier]'\n"
    "It brings torch, which is why it is its own extra: nothing reachable from "
    "the chat-cli extra may resolve torch (Decision 19a, enforced in "
    "nightly.yml and release.yml)."
)


def _logging(verbose: bool) -> None:
    """Show progress, which is minutes long for a corpus walk and otherwise silent."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    for noisy in ("filelock", "huggingface_hub", "urllib3", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _guard(action: Any) -> Any:
    """Run *action*, turning this package's known failures into a clean CLI error.

    The messages these carry are the deliverable -- "no corpus at X, build one
    first" and the refusal to score a hash are both written to be read by the
    person whose command just stopped -- so they are surfaced verbatim rather
    than wrapped in a traceback. Same shape as ``_guard`` in ``cli/chat.py``.
    """
    from aorta.local_classifier.corpus import CorpusError
    from aorta.local_classifier.eval import EvalError
    from aorta.local_classifier.gate import NotMeasurableError
    from aorta.local_classifier.predictor import ClassifierUnavailableError

    try:
        return action()
    except (CorpusError, EvalError, NotMeasurableError) as exc:
        raise click.ClickException(str(exc)) from exc
    except ClassifierUnavailableError as exc:
        raise click.ClickException(f"{exc}\n\n{_INSTALL_HINT}") from exc


@click.group(name="local_classifier")
def cli() -> None:
    """Build the local-classifier decision corpora and measure a model against them offline.

    Phase 0 is 'corpus': it joins artifacts already on disk into labelled JSONL,
    labelling from what was observed rather than from the rules of the prompt
    being replaced. Phase 1 is 'eval': it scores a checkpoint against the
    majority-class floor, the existing regex and the current DSPy assessment's
    own agreement with the eventual Autopsy category, and decides go/no-go.
    """


@cli.command(name="corpus")
@click.argument("kind", type=click.Choice(_CORPUS_KINDS))
@click.option(
    "--root",
    default=None,
    help="Artifact root to walk. Defaults to CIA_JOBS_ROOT or ~/cia-jobs for the "
    "Watch and log-finder corpora; required for the proposer corpus.",
)
@click.option("--output", default=None, help="Where to write the JSONL. Omit to only report.")
@click.option("--json", "as_json", is_flag=True, help="Emit the build report as JSON.")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def corpus(kind: str, root: str | None, output: str | None, as_json: bool, verbose: bool) -> None:
    """Build one labelled corpus from artifacts already on disk.

    The skip counts and warnings are not diagnostics; they are half the output. A
    builder that walked four hundred job directories and labelled nine examples
    has said something important about whether Phase 1 can run at all, and a bare
    count of nine has not. Read them before reading any score.
    """
    _logging(verbose)
    from aorta.local_classifier.corpus import BUILDERS, write_corpus

    builder, _reads = BUILDERS[kind]
    resolved = _resolve_root(kind, root)
    result = _guard(lambda: builder(resolved))

    written = 0
    if output and result.examples:
        written = write_corpus(output, result.examples)
    elif output:
        # Writing an empty file would leave something for `eval` to find and
        # refuse later, three steps from the command that produced nothing.
        click.echo(
            f"warning: no examples were labelled, so {output} was not written.", err=True
        )

    if as_json:
        payload = result.to_dict()
        payload["root"] = str(resolved)
        payload["output"] = output if written else None
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(f"{kind}: {result.summary()}")
        click.echo(f"  root       {resolved}")
        if written:
            click.echo(f"  wrote      {written} example(s) to {output}")
        for reason, count in sorted(result.skipped.items()):
            click.echo(f"  skipped {count:>5}  {reason}")
        for warning in result.warnings:
            click.echo(f"warning: {warning}", err=True)

    if not result.examples:
        # Non-zero, because "built nothing" is the Phase 0 gate failing and a
        # script that treats it as success will go on to eval an absent corpus.
        raise click.exceptions.Exit(1)


def _resolve_root(kind: str, root: str | None) -> Path:
    """Where to walk, with the same default the agents themselves use.

    ``CIA_JOBS_ROOT`` then ``~/cia-jobs`` mirrors ``aorta.cia.triage``, so a
    corpus is built over the directories Watch actually wrote to rather than over
    a second convention invented here. The proposer corpus has no such default:
    an agent run's output directory is chosen per run, and guessing at one would
    walk somebody's home directory.
    """
    if root:
        return Path(root).expanduser()
    if kind == "proposer":
        raise click.UsageError(
            "--root is required for the proposer corpus: it is the --output an "
            "'aorta agent mitigate' run was given, and there is no default worth guessing."
        )
    return Path(os.environ.get("CIA_JOBS_ROOT") or Path.home() / "cia-jobs")


@cli.command(name="eval")
@click.option("--corpus", "corpus_path", required=True, help="Labelled JSONL from 'corpus'.")
@click.option(
    "--backend",
    type=click.Choice(_BACKENDS),
    default="laya-typed-decisions",
    show_default=True,
    help="Which predictor to measure. 'fake' answers from a hash and is refused.",
)
@click.option(
    "--checkpoint",
    default=None,
    help="A local checkpoint directory, overriding --backend. This is how a "
    "fine-tune is scored against the two published checkpoints.",
)
@click.option("--device", default=None, help="Torch device for the checkpoint (default: auto).")
@click.option(
    "--holdout",
    default=None,
    type=float,
    help="Fraction held out of the temperature fit (default 0.2). Pass 0 to fit "
    "in-sample on a corpus too small to split, which the report then says.",
)
@click.option("--no-latency", is_flag=True, help="Skip the CPU latency measurement.")
@click.option("--no-census", is_flag=True, help="Skip the token-length census.")
@click.option("--output", default=None, help="Write the full result as JSON here.")
@click.option("--json", "as_json", is_flag=True, help="Emit the result as JSON on stdout.")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def eval_command(
    corpus_path: str,
    backend: str,
    checkpoint: str | None,
    device: str | None,
    holdout: float | None,
    no_latency: bool,
    no_census: bool,
    output: str | None,
    as_json: bool,
    verbose: bool,
) -> None:
    """Score a checkpoint against the corpus and the three baselines.

    Run it once per candidate -- base 'laya' as a floor, 'laya-typed-decisions',
    then a fine-tune reached through --backend once one exists -- so the go/no-go
    is settled with numbers from these artifacts rather than from a model card.

    The verdict is per decision, because a track may clear while Watch does not,
    and the answer to that is to re-sequence around whichever cleared.
    """
    _logging(verbose)
    from aorta.local_classifier.eval import DEFAULT_HOLDOUT, load_corpus
    from aorta.local_classifier.gate import run_gate
    from aorta.local_classifier.predictor import make_predictor

    examples = _guard(lambda: load_corpus(corpus_path))
    predictor = _guard(
        lambda: make_predictor(backend, device=device, checkpoint=checkpoint)
    )
    fraction = DEFAULT_HOLDOUT if holdout is None else holdout
    result = _guard(
        lambda: run_gate(
            examples,
            predictor,
            # 0 means "do not split", which reads better on a command line than
            # a flag that has to be explained as the absence of a number.
            holdout=fraction if fraction else None,
            with_latency=not no_latency,
            with_census=not no_census,
        )
    )

    payload = result.to_dict()
    payload["corpus"] = corpus_path
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        click.echo(f"wrote {output}", err=True)

    if as_json:
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(result.summary())
        click.echo("")
        for item in result.verdict.decisions:
            mark = "[ ok ]" if item.passed else "[FAIL]"
            click.echo(f"{mark} {item.decision:<20} {item.reason}")

    if not result.verdict.passed:
        # The kill criterion has an exit code, so this works as a gate in a
        # script rather than as a paragraph somebody has to read.
        raise click.exceptions.Exit(1)


def main() -> None:
    """Entry point for ``python -m aorta.local_classifier``."""
    cli(prog_name="python -m aorta.local_classifier")


__all__ = ["cli", "main"]
