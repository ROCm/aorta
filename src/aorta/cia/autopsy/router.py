from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dspy

from aorta.agent.llm import AUTOPSY_CATEGORIES, LAYA_CATEGORY_QUESTION
from aorta.cia.autopsy.adapters.base import resolve_in_bundle as _resolve_in_bundle
from aorta.cia.llm import build_lm

if TYPE_CHECKING:  # Annotations only: ``from __future__ import annotations``
    # means this costs nothing at run time, which is what the import-boundary
    # probe in tests/cli/test_chat_boundaries.py measures.
    from aorta.laya.predictor import Choice, LayaPredictor


# ---------------------------------------------------------------------------
# Tools — the rule-based adapters become callable tools for the LLM
# ---------------------------------------------------------------------------

def classify_matrix(matrix_json: str) -> dict:
    """Run the rule-based AortaMatrixAdapter classifier on a matrix JSON string.
    Returns {category, confidence, signals, rationale}."""
    from aorta.cia.autopsy.adapters.aorta_matrix import classify_matrix as _classify
    try:
        matrix = json.loads(matrix_json)
        result = _classify(matrix)
        return {
            "category": result.category,
            "confidence": result.confidence,
            "signals": result.signals,
            "rationale": result.rationale,
        }
    except Exception as e:
        return {"error": str(e)}


def scan_stderr(log_text: str) -> dict:
    """Run regex patterns on log text to detect NaN, hang, OOM signals.
    Returns {signal, alert, hits}."""
    from aorta.cia.autopsy.adapters.stderr_watch import scan_stderr_text
    scan = scan_stderr_text(log_text)
    return {
        "signal": scan.signal,
        "alert": scan.alert,
        "hits": [{"line": ln, "text": ex} for ln, ex in scan.hits[:10]],
    }


def scan_sanitizer(report_json: str) -> dict:
    """Run the rule-based sanitizer classifier on a sanitizer_report.json string.
    Returns {category, confidence, signals, rationale, per_sanitizer}."""
    from aorta.cia.autopsy.adapters.sanitizer_report import classify_sanitizer
    try:
        report = json.loads(report_json)
        result = classify_sanitizer(report)
        return {
            "category": result.category,
            "confidence": result.confidence,
            "signals": result.signals,
            "rationale": result.rationale,
            "target": report.get("target"),
            "overall_verdict": report.get("overall_verdict"),
            "per_sanitizer": {
                str(c.get("sanitizer")): {"state": c.get("state"), "verdict": c.get("verdict")}
                for c in (report.get("checks") or [])
            },
        }
    except Exception as e:
        return {"error": str(e)}


def resolve_in_bundle(uri: str, bundle_root: str) -> Path | None:
    """*uri* as a path inside *bundle_root*, or None if it points outside.

    The uri arrives from the evidence list and from the model's own tool call,
    and both used to be joined to the bundle root and read. What came back went
    into the router's context and could be quoted into the rationale, which is
    written to the report and sent to a model.

    The policy itself lives with BundleContext, which is what defines a bundle;
    this is the same check the adapters get, spelled for a string root.
    """
    return _resolve_in_bundle(Path(bundle_root), uri)


def evidence_reader(bundle_root: Path):
    """A ``read_evidence_file`` tool bound to *bundle_root*.

    The root is captured here rather than taken as an argument. It used to be
    a parameter of the tool itself, and the tool is one the ReAct model calls
    with arguments it writes -- so the fence and the thing being fenced came
    from the same place. ``read_evidence_file("/etc/passwd", bundle_root="/")``
    passed the containment check, because the path genuinely was inside the
    root it had been handed. What came back went into the router's context and
    could be quoted into a rationale, which is written to the report and sent
    to a model.

    A closure is the whole fix: the model can still name any URI it likes, and
    every one of them is resolved against the root the orchestrator opened the
    bundle with.
    """
    root = Path(bundle_root).resolve()

    def read_evidence_file(uri: str) -> str:
        """Read a specific evidence file from the bundle by its URI.
        Returns up to 200 lines of the file content."""
        try:
            target = _resolve_in_bundle(root, uri)
            if target is None:
                return f"[refused: {uri} is outside the bundle]"
            if not target.is_file():
                return f"[file not found: {uri}]"
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[:200])
        except Exception as e:
            return f"[error: {e}]"

    return read_evidence_file


def list_signals(evidence_json: str) -> list[str]:
    """Extract all unique signal slugs from the evidence list."""
    try:
        evidence = json.loads(evidence_json)
        return list(dict.fromkeys(e.get("signal", "") for e in evidence if e.get("signal")))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# DSPy signature + module
# ---------------------------------------------------------------------------

_CATEGORY_DESC = "Tier-1 category, exactly one of: " + " | ".join(sorted(AUTOPSY_CATEGORIES))


def coerce_category(category: str) -> str:
    """A category outside the shared vocabulary is not a verdict.

    The field is free text, so a model can return a plausible-looking label
    nobody downstream knows how to act on. Reporting that as 'unknown' is
    honest; passing it through would put an unhandled string in the report.
    """
    cleaned = (category or "").strip()
    return cleaned if cleaned in AUTOPSY_CATEGORIES else "unknown"


# ---------------------------------------------------------------------------
# The Laya tier — the category and its confidence, from a calibrated encoder
# ---------------------------------------------------------------------------

#: The checkpoint the tier resolves when nothing names one. Spelled the same as
#: ``aorta.cia.watch.watcher`` and ``aorta.cia.watch.log_finder`` spell it, so
#: the three CIA surfaces read the same way.
DEFAULT_LAYA_BACKEND = "laya-typed-decisions"

#: The three environment variables that configure this tier, mirroring the
#: ``watch.laya`` and ``log_finder.laya`` blocks of ``watch_config.yaml``:
#: ``enabled`` / ``backend`` / one threshold named for what it gates.
#:
#: An environment variable rather than a config key because Autopsy has no
#: config file. It is reached from ``aorta.cia.triage`` and from Watch's
#: ``trigger_autopsy``, neither of which carries settings, and inventing a
#: fourth configuration surface to hold one flag would be a worse trade than
#: the asymmetry.
LAYA_ENABLED_ENV = "CIA_AUTOPSY_LAYA_ENABLED"
LAYA_BACKEND_ENV = "CIA_AUTOPSY_LAYA_BACKEND"
LAYA_ESCALATION_THRESHOLD_ENV = "CIA_AUTOPSY_LAYA_ESCALATION_THRESHOLD"

_TRUE = frozenset({"1", "true", "yes", "on"})


def laya_config_from_env() -> dict[str, Any]:
    """The tier's configuration, off unless an operator turned it on.

    ``escalation_threshold`` is absent rather than defaulted, and that absence
    is load-bearing. ``aorta.cia.autopsy.escalation`` will not compare a Laya
    probability against a cutoff nobody derived for it, so leaving this unset --
    which is every installation today -- is what keeps the escalation on the
    figure 0.85 was actually chosen against. A default here would be a number
    invented to fill a slot, which is the defect this whole phase is about.
    """
    config: dict[str, Any] = {
        "enabled": os.environ.get(LAYA_ENABLED_ENV, "").strip().lower() in _TRUE,
        "backend": os.environ.get(LAYA_BACKEND_ENV, "").strip() or DEFAULT_LAYA_BACKEND,
    }
    raw = os.environ.get(LAYA_ESCALATION_THRESHOLD_ENV, "").strip()
    if raw:
        try:
            config["escalation_threshold"] = float(raw)
        except ValueError:
            # Not fatal, and deliberately not silent. An unparseable threshold
            # means the operator meant to gate on a derived number and did not,
            # and falling through to the rule-based figure is the safe end of
            # that mistake -- but they have to be told, or they will read the
            # report as though their cutoff had applied.
            print(
                f"[autopsy] {LAYA_ESCALATION_THRESHOLD_ENV}={raw!r} is not a number; "
                "the escalation stays on the rule-based confidence"
            )
    return config


def category_question() -> "Choice":
    """The one question this tier asks, over the whole shared vocabulary.

    The wording is :data:`aorta.agent.llm.LAYA_CATEGORY_QUESTION`, imported
    rather than written again. Track B found that the corpus builder and the
    inference path had independently phrased the same question, which does not
    fail loudly -- a temperature is fitted per question, so a question reworded
    at the call site is calibrated against nothing while still returning
    something that looks like a probability. No corpus labels this decision yet,
    so nothing pins the wording from the other side; importing the one public
    definition is how the builder that eventually does find what to use.

    **Sharing the string is not sharing a fit, and must not be read as one.**
    The proposer asks this over ``PROBE_CATEGORIES`` and this asks it over the
    whole of ``AUTOPSY_CATEGORIES``, which is a strictly larger set, and the
    card fits one temperature per (question type, option count) -- so the two
    land in different buckets whatever the instruction says. What the shared
    constant buys is that they are the same *question*, which is the part a
    reader cannot check by grepping for a number.

    No ``criteria``. The proposer's category choice offers its options by name
    alone, and a gloss written here would make this a different question from
    that one while looking like the same one, which is the failure above with an
    extra step. Glosses belong in whichever corpus first labels this decision,
    and then in both call sites at once.
    """
    from aorta.laya.predictor import Choice

    return Choice(
        question=LAYA_CATEGORY_QUESTION,
        options=tuple(sorted(AUTOPSY_CATEGORIES)),
    )


@dataclass(frozen=True)
class LayaCategory:
    """What the tier answered about one bundle, and under what conditions.

    Carried on the returned ``dspy.Prediction`` as ``laya`` rather than written
    from here, following the shape ``aorta.cia.watch.watcher`` established: this
    module has no idea where a report is going, and the orchestrator that does
    is the one that should decide what lands in it.

    ``options`` and ``bucket`` record what was on offer and which of the
    library's calibration buckets that put the question in. Decision 22 asks a
    report to record the temperature fit alongside the checkpoint, because a
    probability is a function of both, and ``bucket`` is the index into the fit.

    **This surface is the one that lands in the worst bucket.**
    ``AUTOPSY_CATEGORIES`` has eleven members, so the question is ``choice:11+``
    -- where 0.3.5 ships a temperature of about 0.1, below 1 and therefore
    sharpening rather than softening, which is why the library clamps it and
    warns. Watch's slug choice is seven and the probe agent's category choice is
    eight; both are under the line and this one is over it. Clamping stops the
    sharpening and does not make the bucket calibrated, so :attr:`caveat` is not
    an edge case here -- on the default checkpoint it is the expected state.

    :attr:`clamped` is three-valued and ``None`` means *unknown*, not clean.
    Unknown is what a predictor with no weights reports, and what a ``laya``
    below 0.3.5 reports -- and on that version the answer is not that nothing
    was clamped, it is that the sharpening happened and nobody was told. A
    consumer collapsing ``None`` to ``False`` would read the worst case as the
    best one.
    """

    model_id: str
    category: str
    probability: float
    options: int
    escalation_threshold: float | None
    #: Which shipped temperature answered, from ``bucket_for``. Pure, so it is
    #: named whether or not a checkpoint ever loaded.
    bucket: str = ""
    #: What the loaded checkpoint knows and the question alone cannot say.
    clamped: bool | None = None
    caveat: str = ""

    def as_report_fields(self) -> dict[str, Any]:
        """The provenance block an Autopsy report carries inline.

        Inline rather than beside the run, which is rule 2 of Decision 22: a
        report is copied into a ticket and read on its own, and a verdict that
        cannot say which weights produced it cannot be compared with the next
        one. A checkpoint swap changes categories with no error and no diff.

        ``clamped`` is here as well as ``caveat`` because the two readers are
        different. A person reads the sentence; a later automated consumer
        deciding whether ``confidence`` may be aggregated needs a field it can
        branch on, and substring-matching a prose fragment for "NOT CALIBRATED"
        is how that consumer comes to disagree with this one.
        """
        return {
            "model_id": self.model_id,
            "category": self.category,
            "probability": round(self.probability, 4),
            "question": LAYA_CATEGORY_QUESTION,
            "options": self.options,
            "bucket": self.bucket,
            "clamped": self.clamped,
            "caveat": self.caveat,
            "escalation_threshold": self.escalation_threshold,
        }


class _LayaCategoryTier:
    """One forward pass, one typed answer, and no prose.

    The tier answers the two fields of ``TriageDecision`` that are a
    classification -- ``category`` and ``confidence`` -- and touches none of the
    three that are writing. That split is forced rather than chosen: Laya emits
    no tokens, so it cannot produce the ``rationale`` a reader acts on, and
    ``next_probe`` is a recommendation the rationale has to argue for.

    Its input is the friendliest in the system for an encoder. ``evidence_json``
    is a list of adapter findings keyed by signal slug, assembled by
    deterministic code from files in the bundle, and it is the same string the
    LLM is handed -- so the two differ in model and not in input, which is what
    makes a later comparison between them a comparison.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        predictor: "LayaPredictor | None" = None,
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.backend = str(cfg.get("backend", DEFAULT_LAYA_BACKEND) or "")
        # No default, and see ``laya_config_from_env``: ``None`` is what keeps
        # the escalation on a number whose cutoff was derived against it.
        raw_threshold = cfg.get("escalation_threshold")
        self.escalation_threshold = (
            None if raw_threshold is None else float(raw_threshold)
        )
        # The injection point, the same one ``LayaAgentPredictor`` offers through
        # ``load=``: a test drives the tier with ``FakeLayaPredictor`` and no
        # checkpoint goes anywhere near CI.
        self._predictor = predictor

    def observe(self, evidence_json: str) -> LayaCategory | None:
        """Classify one bundle's evidence, or None when the tier cannot run.

        None is the whole of the failure handling, and the caller reads it as
        "ask the LLM the way you always did". Autopsy runs once per failure, so
        unlike Watch's poll loop there is nothing to be gained by remembering
        the failure -- but there is something to lose by raising, because the
        alternative to a Laya category is a perfectly good LLM one.
        """
        if not self.enabled:
            return None
        predictor = self._resolve()
        if predictor is None:
            return None
        question = category_question()
        try:
            from aorta.laya.predictor import ChoiceAnswer, ask_one, bucket_for

            (answer,) = ask_one(predictor, evidence_json, [question])
            if not isinstance(answer, ChoiceAnswer):
                raise TypeError(
                    f"{type(predictor).__name__} answered the category choice with "
                    f"{type(answer).__name__}"
                )
            # Read *after* the answers, and the order is load-bearing.
            # ``calibration()`` does not trigger a load, so asking before the
            # forward pass would report "not loaded yet" and stamp CALIBRATION
            # UNKNOWN onto a report whose calibration was perfectly knowable.
            # By here the weights are resident and the temperature tables are
            # there to compare.
            calibration = predictor.calibration()
            bucket = bucket_for(question)
            # Derived from the question object that was actually asked, not
            # from ``len(AUTOPSY_CATEGORIES)``. There is one question here
            # today, so the two agree; the moment a second one is added --
            # a noul on whether the bundle is even conclusive, say -- a
            # count-derived caveat would say nothing about it, which is the
            # mistake Track B made and corrected.
            caveat = calibration.caveat(question)
            clamped = calibration.is_clamped(bucket)
        except Exception as exc:  # noqa: BLE001 - an autopsy must survive a bad tier
            # Reading the calibration is inside this block rather than beside
            # it, so a predictor whose ``calibration()`` cannot be read costs
            # the whole tier instead of only the disclosure. Losing a good
            # classification to a broken accessor is the smaller harm: the
            # other way round puts a probability into report.json with its
            # calibration unexamined, which is the defect this integration
            # exists to remove, reappearing inside the fix for it.
            #
            # Said out loud. An operator who turned this on and silently got the
            # old behaviour back has no other way to find out, and the report
            # they read afterwards names the LLM as its source, which is true
            # but does not explain why.
            print(
                f"[autopsy] the Laya category tier failed; the router classifies "
                f"as before: {type(exc).__name__}: {exc}"
            )
            return None

        return LayaCategory(
            model_id=predictor.model_id(),
            # ``coerce_category`` runs over this downstream and will pass it
            # through. An option-marker head scores the labels it was offered
            # and can return nothing else, so the guard has stopped being the
            # thing that stands between the report and an unhandled string.
            category=answer.option,
            # ``probability``, not the library's own ``confidence`` field. That
            # one is normalised Shannon entropy over the distribution, which
            # measures how peaked an answer is rather than how likely the top
            # option is to be right; ``docs/laya-packaging.md`` has the argument.
            probability=answer.probability,
            options=len(answer.probabilities),
            escalation_threshold=self.escalation_threshold,
            bucket=bucket,
            clamped=clamped,
            caveat=caveat,
        )

    def _resolve(self) -> "LayaPredictor | None":
        """The predictor, or None when this tier cannot run.

        ``fake`` is refused rather than resolved, the same refusal Watch's tier
        makes and for the same reason: ``FakeLayaPredictor`` answers from a
        blake2b hash of the question and the state, so it is stable, arbitrary,
        and indistinguishable from a model with an opinion. A hashed category
        would reach ``report.json``, and from there the corpus builders that
        read ``report.json`` as ground truth. Tests inject a fake directly,
        which is an explicit act by a caller who knows.
        """
        if self._predictor is not None:
            return self._predictor
        if not self.backend or self.backend == "fake":
            print(
                f"[autopsy] {LAYA_ENABLED_ENV} is set but no real checkpoint is named "
                f"(backend={self.backend!r}); the Laya category tier is off. A hashed "
                "fake verdict is not a measurement and must not be recorded as one."
            )
            return None
        try:
            from aorta.laya.predictor import make_predictor

            self._predictor = make_predictor(self.backend)
        except Exception as exc:  # noqa: BLE001 - see observe()
            print(
                f"[autopsy] could not build the Laya predictor {self.backend!r}; "
                f"the category tier is off: {type(exc).__name__}: {exc}"
            )
            return None
        return self._predictor


class TriageDecision(dspy.Signature):
    """
    You are the Autopsy router for GPU cluster training failures on AMD hardware.
    You have structured evidence from tool runs (adapters). Use the tools to
    inspect the evidence, then emit a triage decision.

    Rules:
    - Always call list_signals first to see what signals are present.
    - If AORTA_MATRIX_REPRO and AORTA_MITIGATION_CLEAN are both present,
      category is almost certainly numeric_silent.
    - If only WATCH_NUMERIC_NAN with no matrix, confidence should be ~0.62
      and next_probe should be 'aorta sweep run'. This applies ONLY when the
      watch signal stands alone — if any DBG_* signal is also present, use the
      debugger rule below instead, because the log signature is then the weakest
      evidence in the bundle rather than the only evidence.
    - DBG_DEVICE_ASSERT means a device-side assert trapped on the GPU and a
      debugger read the stopped wave before its registers were lost, naming the
      kernel, the source line and the workgroup. DBG_NAN_TRAP means that captured
      state included a non-finite value. Together they are the hardest evidence
      an Autopsy bundle can carry — harder than a matrix, and harder than a log
      signature, which only says a bad number reached the loss without saying
      where it came from. Category is numeric_silent with confidence >= 0.9, and
      next_probe is 'none': the failure has already been caught in the act, so
      there is nothing left to reproduce. Quote the captured register values and
      the source line from the evidence excerpts in the rationale.
    - If AORTA_MATRIX_INFRA_OK (smoke only, no repro cells), next_probe is
      'aorta sweep run' (need production matrix).
    - SAN_CONSAN_RACE means ConSan observed a real device-side ordering violation
      during record/replay: category is gpu_race with confidence >= 0.9. This is
      not illegal_mem — the access is in bounds, it is just unsynchronised. Name
      the kernel and code object from the evidence excerpt in the rationale.
    - SAN_WAITCHECK_HAZARD alone is a *static* missing-s_waitcnt warning, not an
      observed failure: category is gpu_race but confidence ~0.55, and next_probe
      is 'aorta sweep run' to confirm it dynamically with ConSan.
    - SAN_NOT_CHECKED means the sanitizer backend never ran, so the run proves
      nothing: category is tooling_gap, confidence 0.0. Never read it as clean.
    - SAN_CLEAN with no other failure signal means the sanitizers passed; prefer
      unknown over inventing a failure.
    - Use scan_sanitizer on a sanitizer_report.json URI to read its verdicts.
    - Do not guess — cite specific signal slugs and evidence URIs in rationale.
    - next_probe must be exactly 'aorta sweep run' or 'none'.
    """
    evidence_json: str = dspy.InputField(desc="JSON list of adapter evidence items with signals and URIs")
    job_context: str = dspy.InputField(desc="job_id, node, recipe")

    category: str = dspy.OutputField(desc=_CATEGORY_DESC)
    confidence: float = dspy.OutputField(desc="0.0-1.0")
    rationale: str = dspy.OutputField(desc="One paragraph citing specific signal slugs and evidence URIs")
    next_probe: str = dspy.OutputField(desc="'aorta sweep run' or 'none'")
    next_probe_reason: str = dspy.OutputField(desc="Why this probe is needed, or empty if none")


class TriageDecisionWithAssignedCategory(dspy.Signature):
    """
    You are the Autopsy router for GPU cluster training failures on AMD hardware.
    You have structured evidence from tool runs (adapters), and a category that a
    calibrated classifier has already assigned to this same evidence. Use the
    tools to inspect the evidence, then explain the failure and say what to run
    next.

    You are not asked for a category or for a confidence on this path, and there
    is nowhere to write one. If the evidence does not support the assigned
    category, say so plainly in the rationale rather than arguing around it — a
    disagreement recorded is worth more to the reader than a rationale that
    reads as though it agreed.

    Rules:
    - Always call list_signals first to see what signals are present.
    - WATCH_NUMERIC_NAN standing alone, with no matrix, means the only evidence
      is a log signature, so next_probe is 'aorta sweep run'. This applies ONLY
      when the watch signal stands alone — if any DBG_* signal is also present,
      use the debugger rule below instead, because the log signature is then the
      weakest evidence in the bundle rather than the only evidence.
    - DBG_DEVICE_ASSERT means a device-side assert trapped on the GPU and a
      debugger read the stopped wave before its registers were lost, naming the
      kernel, the source line and the workgroup. DBG_NAN_TRAP means that captured
      state included a non-finite value. Together they are the hardest evidence
      an Autopsy bundle can carry — harder than a matrix, and harder than a log
      signature, which only says a bad number reached the loss without saying
      where it came from. next_probe is 'none': the failure has already been
      caught in the act, so there is nothing left to reproduce. Quote the
      captured register values and the source line from the evidence excerpts in
      the rationale.
    - If AORTA_MATRIX_INFRA_OK (smoke only, no repro cells), next_probe is
      'aorta sweep run' (need production matrix).
    - SAN_CONSAN_RACE means ConSan observed a real device-side ordering violation
      during record/replay. The access is in bounds, it is just unsynchronised;
      do not describe it as an illegal access. Name the kernel and code object
      from the evidence excerpt in the rationale.
    - SAN_WAITCHECK_HAZARD alone is a *static* missing-s_waitcnt warning on a
      path that may never execute, not an observed failure, so next_probe is
      'aorta sweep run' to confirm it dynamically with ConSan.
    - SAN_NOT_CHECKED means the sanitizer backend never ran, so the run proves
      nothing. Never read it as clean.
    - SAN_CLEAN with no other failure signal means the sanitizers passed; do not
      describe a failure the evidence does not show.
    - Use scan_sanitizer on a sanitizer_report.json URI to read its verdicts.
    - Do not guess — cite specific signal slugs and evidence URIs in rationale.
    - next_probe must be exactly 'aorta sweep run' or 'none'.
    """
    evidence_json: str = dspy.InputField(desc="JSON list of adapter evidence items with signals and URIs")
    job_context: str = dspy.InputField(desc="job_id, node, recipe")
    assigned_category: str = dspy.InputField(
        desc="The category already assigned to this evidence. Explain it, or say why it is wrong."
    )

    rationale: str = dspy.OutputField(desc="One paragraph citing specific signal slugs and evidence URIs")
    next_probe: str = dspy.OutputField(desc="'aorta sweep run' or 'none'")
    next_probe_reason: str = dspy.OutputField(desc="Why this probe is needed, or empty if none")


class TriageRouter(dspy.Module):
    #: Autopsy weighs several signals against each other and writes the
    #: rationale a reader acts on, which is the longest reasoning in the
    #: pipeline; it runs once per failure where Watch polls throughout. The
    #: ReAct trajectory below has to fit its tool calls *and* its answer inside
    #: this, and a reasoning model bills its reasoning against it too.
    #:
    #: A budget is all this module pins. It used to name a model as well, which
    #: made the choice of model a property of the code rather than of the
    #: deployment. The operator configures one through the environment and the
    #: agents follow it.
    MAX_TOKENS = 8192

    def __init__(
        self,
        bundle_root: Path | str,
        *,
        laya: Mapping[str, Any] | None = None,
        predictor: "LayaPredictor | None" = None,
    ):
        # Built here, per bundle, because one of the tools is bound to a root
        # and a module shared across jobs would carry the first job's root into
        # the second. Autopsy runs once per failure, so this costs nothing that
        # matters.
        self._tools = [
            classify_matrix,
            scan_stderr,
            scan_sanitizer,
            evidence_reader(Path(bundle_root)),
            list_signals,
        ]
        # Bound to this module rather than configured globally: whichever agent
        # reached DSPy first would otherwise decide what Autopsy reasons with.
        self._lm = build_lm(max_tokens=self.MAX_TOKENS)
        self.laya_tier = _LayaCategoryTier(
            laya_config_from_env() if laya is None else laya, predictor
        )
        # The LLM-classifies module is built whether or not the tier is on, and
        # not only because a construction-time ``set_lm`` is what binds the
        # budget. It is also the fallback: the tier can be configured on and
        # still fail to answer on a node with no staged weights, and when it
        # does, Autopsy has to classify exactly the way it did before anyone
        # turned it on.
        self.react = self._react(TriageDecision)

    def _react(self, signature: type[dspy.Signature]) -> dspy.Module:
        react = dspy.ReAct(signature, tools=self._tools, max_iters=6)
        react.set_lm(self._lm)
        return react

    def forward(self, evidence: list[dict[str, Any]], job_context: str) -> dspy.Prediction:
        """Classify the bundle and explain it, in one or two pieces.

        The Laya tier is asked first, before an LLM call is spent, so that the
        prompt can be chosen by what actually answered rather than by what was
        configured. When it answers, the model is handed a signature with no
        ``category`` and no ``confidence`` field at all, which is the only way
        the prompt's hand-fitted calibration curve genuinely leaves: a model
        asked for a number it was given no rule for would invent a worse one
        than the rule it lost.
        """
        evidence_json = json.dumps(evidence)
        observation = self.laya_tier.observe(evidence_json)
        if observation is None:
            prediction = self.react(
                evidence_json=evidence_json,
                job_context=job_context,
            )
            prediction.category = coerce_category(getattr(prediction, "category", ""))
            return prediction

        prediction = self._react(TriageDecisionWithAssignedCategory)(
            evidence_json=evidence_json,
            job_context=job_context,
            assigned_category=observation.category,
        )
        # Kept as a guard and no longer load-bearing. An option-marker head
        # scores the labels it was offered and cannot name one that was not, so
        # the only thing this can now catch is the option set drifting away from
        # AUTOPSY_CATEGORIES -- which is worth catching, and is not what it was
        # written for.
        prediction.category = coerce_category(observation.category)
        prediction.confidence = observation.probability
        prediction.laya = observation
        return prediction
