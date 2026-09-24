"""Phase 0: that the labels come from observations, and that gaps are reported.

The single most important property of these builders is negative: a label must
never be derived from the rules of the prompt being replaced. Labelling the
log-finder corpus by "prefer a high mtime, prefer 'log' in the name" would teach
a classifier the heuristic already in ``_scan_by_extension``, and the resulting
accuracy would look like success while buying nothing. So the tests below are as
much about what is *not* labelled, and about the warnings that say why, as they
are about the records that come out.

The fixtures write the artifacts a real run leaves behind -- a bundle with the
delta ``write_bundle`` persists, a ``report.json`` with the category Autopsy
reached, an ``agent_log.jsonl`` with the events ``run_agent_loop`` appends -- so
these exercise the joins rather than a convenient reshaping of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from aorta.local_classifier.corpus import (
    BUILDERS,
    build_log_finder_corpus,
    build_proposer_corpus,
    build_watch_corpus,
    read_corpus,
    write_corpus,
)
from aorta.local_classifier.corpus.schema import CorpusError, LabelledExample
from aorta.local_classifier.corpus.watch import (
    CATEGORY_TO_SIGNAL,
    UNLABELLABLE_CATEGORIES,
    watch_questions,
    watch_signals,
)
from aorta.local_classifier.predictor import Choice, Noul

_DELTA = "\n".join(f"step {index} loss nan" for index in range(20))


@pytest.fixture
def jobs_root(tmp_path: Path):
    """Build CIA job directories the way a triage run leaves them."""

    def _build(
        job_id: str = "cia-1",
        *,
        delta: str | None = _DELTA,
        category: str | None = "numeric_silent",
        events: list[dict] | None = None,
        watch_signal: str = "WATCH_NUMERIC_NAN",
        log_path: str = "/var/log/job.log",
        watch_files: list[str] | None = None,
        cursors: dict[str, int] | None = None,
        files: dict[str, str] | None = None,
    ) -> Path:
        root = tmp_path / "cia-jobs"
        job_dir = root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "log_path": log_path,
                    "watch_files": watch_files if watch_files is not None else [],
                }
            ),
            encoding="utf-8",
        )
        if delta is not None:
            bundle = job_dir / "bundle" / "logs"
            bundle.mkdir(parents=True, exist_ok=True)
            (bundle / "watch.stderr.log").write_text(delta, encoding="utf-8")
            (job_dir / "bundle" / "manifest.yaml").write_text(
                yaml.dump({"job_id": job_id, "metadata": {"watch_signal": watch_signal}}),
                encoding="utf-8",
            )
        if category is not None:
            (job_dir / "bundle").mkdir(parents=True, exist_ok=True)
            (job_dir / "bundle" / "report.json").write_text(
                json.dumps({"category": category, "confidence": 0.9}), encoding="utf-8"
            )
        if events:
            (job_dir / "events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
            )
        if cursors is not None:
            (job_dir / "watch_cursors.json").write_text(json.dumps(cursors), encoding="utf-8")
        for name, content in (files or {}).items():
            target = job_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return root

    return _build


class TestWatchCorpus:
    def test_the_delta_is_the_state_and_the_autopsy_category_is_the_label(self, jobs_root):
        """The join the whole corpus rests on, and it is a *later* observation.

        The label is what the failure turned out to be after Autopsy read the
        whole bundle -- not what Watch guessed at the time, which is the baseline.
        """
        result = build_watch_corpus(jobs_root(category="oom_fragment"))
        signals = [e for e in result.examples if e.decision == "watch_signal"]
        assert len(signals) == 1
        assert signals[0].state == _DELTA
        assert signals[0].label == "WATCH_OOM"
        assert signals[0].join_key == "cia-1"

    def test_an_alerting_delta_is_also_a_negative_for_the_clean_gate(self, jobs_root):
        result = build_watch_corpus(jobs_root())
        healthy = [e for e in result.examples if e.decision == "watch_healthy"]
        assert [e.label for e in healthy] == ["false"]

    def test_a_job_that_never_alerted_is_skipped_with_a_reason(self, jobs_root):
        result = build_watch_corpus(jobs_root(delta=None, category=None))
        assert not result.examples
        assert any("never alerted" in reason for reason in result.skipped)

    def test_a_bundle_with_no_autopsy_report_has_no_label(self, jobs_root):
        result = build_watch_corpus(jobs_root(category=None))
        assert not result.examples
        assert any("no Autopsy report" in reason for reason in result.skipped)

    def test_an_unknown_verdict_is_not_ground_truth(self, jobs_root):
        """An Autopsy that could not tell says nothing about the log."""
        result = build_watch_corpus(jobs_root(category="unknown"))
        assert not result.examples
        assert any("not ground truth" in reason for reason in result.skipped)

    def test_a_tooling_gap_is_not_read_as_clean(self, jobs_root):
        """SAN_NOT_CHECKED means the instrument never ran, so the run proves nothing."""
        result = build_watch_corpus(jobs_root(category="tooling_gap"))
        assert not result.examples

    def test_a_category_outside_the_map_is_refused_rather_than_bucketed(self, jobs_root):
        """An unmapped category silently becoming WATCH_UNKNOWN_ERROR is taxonomy drift."""
        result = build_watch_corpus(jobs_root(category="quantum_tunnelling"))
        assert not result.examples
        assert any("CATEGORY_TO_SIGNAL" in warning for warning in result.warnings)

    def test_the_dspy_baseline_is_recovered_from_the_events_file(self, jobs_root):
        """Recorded rather than recomputed: the prompt has been edited since.

        The bar Phase 1 has to clear is the answer the assessment actually gave on
        the day, not one re-derived from today's prompt.
        """
        root = jobs_root(
            category="oom_fragment",
            events=[
                {
                    "event_type": "watchdog_alert",
                    "signal": "WATCH_UNKNOWN_ERROR",
                    "confidence": 0.82,
                    "excerpt": "something went wrong",
                }
            ],
        )
        example = next(e for e in build_watch_corpus(root).examples)
        assert example.baseline("dspy_signal") == "WATCH_UNKNOWN_ERROR"
        assert example.baseline("dspy_confidence") == "0.82"

    def test_the_regex_baseline_is_told_apart_by_its_signature(self, jobs_root):
        """``sanitizer_assessment()`` is the only tier pairing 1.0 with a [sanitizer] line."""
        root = jobs_root(
            events=[
                {
                    "event_type": "watchdog_alert",
                    "signal": "WATCH_UNKNOWN_ERROR",
                    "confidence": 1.0,
                    "excerpt": "[sanitizer] consan: verdict=fail state=ran findings=3",
                }
            ]
        )
        example = next(e for e in build_watch_corpus(root).examples)
        assert example.baseline("regex_signal") == "WATCH_UNKNOWN_ERROR"

    def test_the_manifest_signal_is_the_fallback_when_events_are_gone(self, jobs_root):
        example = next(e for e in build_watch_corpus(jobs_root()).examples)
        assert example.baseline("dspy_signal") == "WATCH_NUMERIC_NAN"

    def test_a_healthy_excerpt_is_used_and_warned_about(self, jobs_root):
        """Half a corpus is a real starting point; a silent half is not.

        Watch caps a watchdog_ok excerpt at 500 characters while a bundle carries
        up to 4000, so the two classes differ in length before they differ in
        content and a classifier can score well by measuring length.
        """
        root = jobs_root(
            events=[
                {
                    "event_type": "watchdog_ok",
                    "event_id": "e1",
                    "signal": "WATCH_CLEAN",
                    "excerpt": "step 41 loss 0.98 throughput 1200 tok/s, all ranks reporting",
                }
            ]
        )
        result = build_watch_corpus(root)
        healthy = [e for e in result.examples if e.decision == "watch_healthy"]
        assert "true" in {e.label for e in healthy}
        assert any("500 characters" in warning for warning in result.warnings)

    def test_an_empty_healthy_excerpt_is_not_a_state(self, jobs_root):
        """A healthy assessment quotes nothing, so most watchdog_ok events carry nothing."""
        root = jobs_root(
            events=[{"event_type": "watchdog_ok", "event_id": "e1", "excerpt": ""}]
        )
        result = build_watch_corpus(root)
        assert any("no excerpt" in reason for reason in result.skipped)

    def test_a_corpus_with_no_healthy_rows_says_the_gate_cannot_be_measured(self, jobs_root):
        result = build_watch_corpus(jobs_root())
        assert any("clean-gate cannot be measured" in warning for warning in result.warnings)

    def test_the_collapse_onto_one_slug_is_counted(self, jobs_root):
        """Four categories have no slug of their own, so plain accuracy misleads."""
        result = build_watch_corpus(jobs_root(category="illegal_mem"))
        assert any("WATCH_UNKNOWN_ERROR" in warning for warning in result.warnings)

    def test_unreachable_slugs_are_named(self, jobs_root):
        """WATCH_LOSS_STALL has no category that maps to it, whatever is scanned."""
        result = build_watch_corpus(jobs_root())
        assert any("WATCH_LOSS_STALL" in warning for warning in result.warnings)

    def test_a_missing_jobs_root_is_reported_rather_than_crashing(self, tmp_path: Path):
        result = build_watch_corpus(tmp_path / "absent")
        assert not result.examples
        assert any("nothing to join on" in warning for warning in result.warnings)

    def test_a_short_delta_carries_nothing_to_read(self, jobs_root):
        result = build_watch_corpus(jobs_root(delta="oom"))
        assert any("too short" in reason for reason in result.skipped)


class TestTheSignalVocabularies:
    def test_every_mapped_slug_is_one_the_question_actually_offers(self):
        """Read off the question, not off a list beside it.

        The ``criteria`` keys are the answer space, so a slug dropped from them is
        a slug the model is never offered -- and a label for it would still have
        validated against a parallel list.
        """
        assert set(CATEGORY_TO_SIGNAL.values()) <= set(watch_signals())

    def test_watch_clean_is_not_an_option_on_the_signal_choice(self):
        """It is the answer to the healthy noul, so offering it twice lets one pass disagree."""
        assert "WATCH_CLEAN" not in watch_signals()

    def test_the_questions_come_from_the_tier_that_asks_them(self):
        """No second copy here: one wording, fitted and asked.

        Byte-identity was confirmed across the move before the copies were
        deleted, so this changed no emitted row.
        """
        from aorta.cia.watch.watcher import healthy_question, signal_question

        healthy, signal = watch_questions()
        assert healthy == healthy_question()
        assert signal == signal_question()

    def test_the_slug_list_is_read_off_the_question_rather_than_beside_it(self):
        assert watch_signals() == watch_questions()[1].options

    def test_the_builder_keeps_no_copy_of_the_question_text(self):
        import ast

        from aorta.cia.watch.watcher import CLASSIFIER_HEALTHY_QUESTION, CLASSIFIER_SIGNAL_QUESTION
        from aorta.local_classifier.corpus import watch as module

        literals = {
            " ".join(node.value.split())
            for node in ast.walk(ast.parse(Path(module.__file__).read_text(encoding="utf-8")))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert CLASSIFIER_HEALTHY_QUESTION not in literals
        assert CLASSIFIER_SIGNAL_QUESTION not in literals

    def test_every_autopsy_category_is_either_mapped_or_deliberately_unlabellable(self):
        """A category added to the shared vocabulary must not go quietly unlabelled.

        ``AUTOPSY_CATEGORIES`` is the vocabulary both front doors answer in, and it
        has grown before -- the plan describes it as nine members where the code
        now holds eleven. A new one arriving with no entry here silently drops
        every job that reached it out of the corpus, which reads as "few jobs
        alerted" rather than as "the taxonomy moved".
        """
        from aorta.agent.llm import AUTOPSY_CATEGORIES

        accounted = set(CATEGORY_TO_SIGNAL) | UNLABELLABLE_CATEGORIES
        missing = sorted(AUTOPSY_CATEGORIES - accounted)
        assert not missing, (
            f"{missing} have no entry in CATEGORY_TO_SIGNAL and are not listed as "
            "unlabellable. Decide which, deliberately: a Watch slug it maps to, or "
            "a category that says nothing about the log."
        )

    def test_nothing_is_both_mapped_and_unlabellable(self):
        assert not (set(CATEGORY_TO_SIGNAL) & UNLABELLABLE_CATEGORIES)

    def test_the_map_is_lossy_and_that_is_recorded_not_hidden(self):
        """Four categories share one slug and one slug is unreachable. Both are findings."""
        collapsed = [c for c, s in CATEGORY_TO_SIGNAL.items() if s == "WATCH_UNKNOWN_ERROR"]
        assert len(collapsed) >= 4
        assert set(watch_signals()) - set(CATEGORY_TO_SIGNAL.values())


@pytest.fixture
def agent_run(tmp_path: Path):
    """Build an ``aorta agent mitigate`` run directory the way the loop leaves one."""

    def _build(
        *,
        events: list[dict],
        cells: dict[str, str],
        category: str | None = None,
        ticket: str = "TICKET-1",
    ) -> Path:
        root = tmp_path / "agent-runs"
        run_dir = root / ticket
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "agent_log.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
        )
        for cell, verdict in cells.items():
            trial = run_dir / cell / "trial_0"
            trial.mkdir(parents=True, exist_ok=True)
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "cell_name": cell,
                        "verdict": verdict,
                        "failure_detectors_fired": [] if verdict == "pass" else ["tier1:exit"],
                        "exit_code": 0 if verdict == "pass" else 1,
                    }
                ),
                encoding="utf-8",
            )
        if category is not None:
            (run_dir / "report.json").write_text(
                json.dumps({"category": category}), encoding="utf-8"
            )
        return root

    return _build


_CONVERGED = [
    {"type": "session_start", "symptom": "loss goes NaN at step 50"},
    {
        "type": "llm_step",
        "category": "illegal_mem",
        "hypothesis": "try it",
        "next_mitigations": ["hsa_xnack"],
        "confidence": 0.5,
        "stop": False,
    },
    {"type": "mitigation_tried", "mitigation": "hsa_xnack"},
    {"type": "iteration_complete", "iteration": 1},
    {
        "type": "llm_step",
        "category": "numeric_silent",
        "hypothesis": "try the other",
        "next_mitigations": ["tf32_off"],
        "confidence": 0.6,
        "stop": False,
    },
    {"type": "mitigation_tried", "mitigation": "tf32_off"},
    {"type": "converged", "winning_mitigation": "tf32_off"},
]

_CONVERGED_CELLS = {
    "none-none": "fail",
    "hsa_xnack-none": "fail",
    "tf32_off-none": "pass",
}


class TestProposerCorpus:
    def test_the_label_is_the_mitigation_that_actually_cleared_a_cell(self, agent_run):
        """A probe cell passing is a measured fact, which is what makes it a label."""
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        result = build_proposer_corpus(root)
        picks = [e for e in result.examples if e.decision == "proposer_mitigation"]
        assert picks
        assert {e.label for e in picks} == {"tf32_off"}

    def test_the_candidate_set_is_what_was_still_untried(self, agent_run):
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        picks = [
            e
            for e in build_proposer_corpus(root).examples
            if e.decision == "proposer_mitigation"
        ]
        first = next(e for e in picks if e.join_key.endswith("step0"))
        assert set(first.question.options) == {"hsa_xnack", "tf32_off"}

    def test_a_step_after_the_winner_was_tried_is_not_labelled(self, agent_run):
        """The answer is no longer in the option set, so there is nothing to ask.

        Reachable because ``agent_log.jsonl`` is append-only across resumes: a
        second session against the same ticket writes more ``llm_step`` events
        after the first session's ``converged``. Labelling those with the winner
        would offer a candidate list the answer is not in.
        """
        resumed = [
            *_CONVERGED,
            {"type": "session_start", "symptom": "loss goes NaN at step 50"},
            {
                "type": "llm_step",
                "category": "numeric_silent",
                "hypothesis": "resumed, trying something new",
                "next_mitigations": ["rccl_blocking"],
                "confidence": 0.3,
                "stop": False,
            },
            {"type": "mitigation_tried", "mitigation": "rccl_blocking"},
        ]
        result = build_proposer_corpus(agent_run(events=resumed, cells=_CONVERGED_CELLS))
        assert any("already been tried" in reason for reason in result.skipped)

    def test_the_state_hides_cells_a_later_iteration_produced(self, agent_run):
        """Otherwise the model is shown the result of the mitigation it must pick.

        The run directory holds every cell the whole run produced. Showing all of
        them to a question asked at step 0 scores beautifully and measures nothing.
        """
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        first = next(
            e
            for e in build_proposer_corpus(root).examples
            if e.join_key.endswith("step0") and e.decision == "proposer_mitigation"
        )
        assert "tf32_off-none" not in first.state
        assert "none-none" in first.state

    def test_the_state_is_the_json_the_proposer_prompt_sends(self, agent_run):
        """So a Phase 1 comparison differs in model rather than in input."""
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        example = next(iter(build_proposer_corpus(root).examples))
        state = json.loads(example.state)
        assert sorted(state) == ["already_tried", "candidates", "cell_summaries", "symptom"]
        assert state["symptom"] == "loss goes NaN at step 50"

    def test_stopping_is_labelled_false_while_something_untried_remains(self, agent_run):
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        stops = [e for e in build_proposer_corpus(root).examples if e.decision == "proposer_stop"]
        assert stops
        assert {e.label for e in stops} == {"false"}

    def test_stopping_is_labelled_true_once_the_candidates_are_exhausted(self, agent_run):
        events = [
            {"type": "session_start", "symptom": "hangs"},
            {"type": "mitigation_tried", "mitigation": "hsa_xnack"},
            {
                "type": "llm_step",
                "category": "rccl_hang",
                "hypothesis": "nothing left",
                "next_mitigations": [],
                "confidence": 0.9,
                "stop": True,
            },
            {"type": "search_stopped", "outcome": "exhausted_candidates",
             "stop_reason": "exhausted_candidates"},
        ]
        root = agent_run(events=events, cells={"none-none": "fail", "hsa_xnack-none": "fail"})
        result = build_proposer_corpus(root)
        stops = [e for e in result.examples if e.decision == "proposer_stop"]
        # Every candidate was already tried before the only step, so the loop's own
        # short-circuit covers it and there is no question left to ask.
        assert not stops
        assert any("remaining candidates" in reason for reason in result.skipped)

    def test_the_questions_come_from_the_module_that_owns_the_decision(self):
        """Not a second phrasing here, which is what this module shipped first.

        A corpus labelled against one wording and a proposer asking another does
        not fail; it answers slightly worse for a reason nobody would go looking
        for. The questions belong beside the probe agent's decision, and this
        module already imported ``PROBE_CATEGORIES`` from there.
        """
        from aorta.agent.llm import (
            CLASSIFIER_CATEGORY_QUESTION,
            CLASSIFIER_MITIGATION_QUESTION,
            CLASSIFIER_STOP_QUESTION,
            CLASSIFIER_STOP_WHEN_FALSE,
            CLASSIFIER_STOP_WHEN_TRUE,
        )
        from aorta.local_classifier.corpus import proposer as module

        assert module._STOP.question == CLASSIFIER_STOP_QUESTION
        assert module._STOP.when_true == CLASSIFIER_STOP_WHEN_TRUE
        assert module._STOP.when_false == CLASSIFIER_STOP_WHEN_FALSE
        source = Path(module.__file__).read_text(encoding="utf-8")
        for canonical in (CLASSIFIER_MITIGATION_QUESTION, CLASSIFIER_CATEGORY_QUESTION, CLASSIFIER_STOP_QUESTION):
            assert canonical not in source, (
                f"{canonical!r} is spelled out in the corpus builder again; import it "
                "from aorta.agent.llm instead."
            )

    def test_the_emitted_examples_carry_the_canonical_questions(self, agent_run):
        """Behavioural, not textual: what the rows actually ask.

        Checking the import alone would pass a module that imported the constants
        and then built its questions from something else.
        """
        from aorta.agent.llm import (
            CLASSIFIER_CATEGORY_QUESTION,
            CLASSIFIER_MITIGATION_QUESTION,
            CLASSIFIER_STOP_QUESTION,
        )

        root = agent_run(
            events=_CONVERGED, cells=_CONVERGED_CELLS, category="oom_fragment"
        )
        asked = {
            example.decision: example.question.question
            for example in build_proposer_corpus(root).examples
        }
        assert asked["proposer_mitigation"] == CLASSIFIER_MITIGATION_QUESTION
        assert asked["proposer_category"] == CLASSIFIER_CATEGORY_QUESTION
        assert asked["proposer_stop"] == CLASSIFIER_STOP_QUESTION

    def test_the_seam_holds_none_of_the_question_text(self):
        """It serves three tracks, and would become a string registry if it did.

        The import direction is the test: a model-agnostic seam that reached into
        ``aorta.agent`` for one of its three consumers' phrasings would have to do
        the same for Watch and for the log finder.
        """
        import ast

        from aorta.agent.llm import CLASSIFIER_MITIGATION_QUESTION, CLASSIFIER_STOP_QUESTION
        from aorta.local_classifier import predictor

        source = Path(predictor.__file__).read_text(encoding="utf-8")
        assert CLASSIFIER_MITIGATION_QUESTION not in source
        assert CLASSIFIER_STOP_QUESTION not in source
        # Imports rather than text: the docstring names ``aorta.agent.llm`` on
        # purpose, because the seam's shape is deliberately modelled on it.
        imported = {
            name
            for node in ast.walk(ast.parse(source))
            for name in (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
        }
        assert not [name for name in imported if name.startswith("aorta.agent")]

    def test_the_proposers_own_answers_are_recorded_as_baselines(self, agent_run):
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        first = next(
            e for e in build_proposer_corpus(root).examples if e.join_key.endswith("step0")
        )
        assert first.baseline("dspy_mitigation") == "hsa_xnack"
        assert first.baseline("dspy_category") == "illegal_mem"
        assert first.baseline("dspy_stop") == "false"

    def test_both_question_shapes_are_emitted_for_the_same_step(self, agent_run):
        """Phase 1 measures which shape is better instead of the shapes being argued.

        Track B's case for per-candidate nouls is sound a priori and is still an
        empirical claim about a model nobody has run on ROCm logs. Emitting both
        costs one row per candidate over a state that is already built.
        """
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        examples = build_proposer_corpus(root).examples
        shapes = {e.decision for e in examples}
        assert "proposer_mitigation" in shapes
        assert "proposer_candidate" in shapes

    def test_one_noul_per_remaining_candidate_labelled_by_what_cleared_the_cell(
        self, agent_run
    ):
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        nouls = [
            e
            for e in build_proposer_corpus(root).examples
            if e.decision == "proposer_candidate" and e.join_key.startswith("TICKET-1:step0")
        ]
        assert {e.join_key.rsplit(":", 1)[1] for e in nouls} == {"hsa_xnack", "tf32_off"}
        assert {e.join_key.rsplit(":", 1)[1] for e in nouls if e.label == "true"} == {"tf32_off"}

    def test_the_candidates_share_one_state_so_the_shape_is_one_forward_pass(self, agent_run):
        """The efficiency claim, which only holds if the candidate varies in the
        question and not in the state.

        ``build_sequence`` lays out one sequence per question with the state
        appended, and collates them into one batch. N questions over one state is a
        single pass; N *states* is N passes. Putting the candidate in the state
        would have turned twenty-one questions into twenty-one forward passes,
        which is the cost the shape exists to avoid.
        """
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        nouls = [
            e
            for e in build_proposer_corpus(root).examples
            if e.decision == "proposer_candidate" and e.join_key.startswith("TICKET-1:step0")
        ]
        assert len({e.state for e in nouls}) == 1
        assert len({e.question.question for e in nouls}) == len(nouls)

    def test_each_candidate_is_named_in_its_own_question(self, agent_run):
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        nouls = [
            e for e in build_proposer_corpus(root).examples if e.decision == "proposer_candidate"
        ]
        for example in nouls:
            assert example.join_key.rsplit(":", 1)[1] in example.question.question

    def test_a_candidate_noul_is_out_of_the_clamped_bucket(self, agent_run):
        """The reason the shape is worth measuring: it avoids the bad bucket rather
        than merely disclosing it."""
        from aorta.local_classifier.predictor import bucket_for

        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        examples = build_proposer_corpus(root).examples
        nouls = [e for e in examples if e.decision == "proposer_candidate"]
        assert nouls
        assert {bucket_for(e.question) for e in nouls} == {"noul:2"}

    def test_both_shapes_share_the_step_as_their_group(self, agent_run):
        """So the split keeps a step whole and the two shapes see the same steps."""
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        examples = [
            e
            for e in build_proposer_corpus(root).examples
            if e.join_key.startswith("TICKET-1:step0")
        ]
        assert {e.group for e in examples} == {"TICKET-1:step0"}

    def test_the_row_count_asymmetry_between_the_shapes_is_stated(self, agent_run):
        """"The corpus has 400 examples" hides which decision has four hundred."""
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        result = build_proposer_corpus(root)
        assert any("per-decision counts" in warning for warning in result.warnings)

    def test_the_candidate_class_imbalance_is_stated(self, agent_run):
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        result = build_proposer_corpus(root)
        assert any("group_top1" in warning for warning in result.warnings)

    def test_no_category_is_labelled_without_an_independent_verdict(self, agent_run):
        """The plan asks for one; nothing on disk supplies it without circularity.

        ``agent_log.jsonl``'s category and ``agent_report.md``'s are both the
        proposer's own, so labelling with either trains the classifier on the
        output of the thing it replaces.
        """
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS)
        result = build_proposer_corpus(root)
        assert not any(e.decision == "proposer_category" for e in result.examples)
        assert any("circular" in reason for reason in result.skipped)
        assert any("Autopsy report" in warning for warning in result.warnings)

    def test_an_autopsy_verdict_beside_the_run_does_supply_one(self, agent_run):
        root = agent_run(
            events=_CONVERGED, cells=_CONVERGED_CELLS, category="oom_fragment"
        )
        categories = [
            e for e in build_proposer_corpus(root).examples if e.decision == "proposer_category"
        ]
        assert categories
        assert {e.label for e in categories} == {"oom_fragment"}

    def test_an_evidence_only_category_is_not_offered_to_the_proposer(self, agent_run):
        """PROBE_CATEGORIES is AUTOPSY_CATEGORIES minus what only an instrument can see.

        A gpu_race verdict cannot be reached by trying mitigations, so labelling a
        proposer question with it would teach it to guess a category it has no way
        to establish.
        """
        root = agent_run(events=_CONVERGED, cells=_CONVERGED_CELLS, category="gpu_race")
        result = build_proposer_corpus(root)
        assert not any(e.decision == "proposer_category" for e in result.examples)

    def test_a_run_that_never_converged_is_reported(self, agent_run):
        events = [
            {"type": "session_start", "symptom": "hangs"},
            {
                "type": "llm_step",
                "category": "rccl_hang",
                "next_mitigations": ["hsa_xnack"],
                "hypothesis": "try",
                "confidence": 0.4,
                "stop": False,
            },
            {"type": "mitigation_tried", "mitigation": "hsa_xnack"},
        ]
        result = build_proposer_corpus(
            agent_run(events=events, cells={"none-none": "fail", "hsa_xnack-none": "fail"})
        )
        assert any("never converged" in reason for reason in result.skipped)
        assert not any(e.decision == "proposer_mitigation" for e in result.examples)

    def test_a_missing_run_root_is_reported_rather_than_crashing(self, tmp_path: Path):
        result = build_proposer_corpus(tmp_path / "absent")
        assert not result.examples
        assert any("no proposer calls" in warning for warning in result.warnings)


def _render_listing(job_dir: Path) -> str:
    """A stand-in for tier 3's ``_dir_listing``, in exactly its rendered shape.

    Injected rather than letting the builder reach the real one, because the real
    one lives in ``aorta.cia.watch.log_finder`` and needs the agents' extra;
    ``test_the_default_listing_is_tier_threes_own`` pins the wiring instead. The
    format matters -- ``path  size=N  mtime=T`` -- because the builder parses its
    candidate set back out of the rendered text so the options offered are exactly
    what the listing contains.
    """
    lines = []
    for path in sorted(job_dir.rglob("*")):
        if path.is_file():
            stat = path.stat()
            lines.append(f"{path}  size={stat.st_size}  mtime={stat.st_mtime}")
    return "\n".join(lines)


def _readable_job(jobs_root, *, log_path: str = "", files: dict[str, str] | None = None) -> Path:
    """A job whose log Watch is observed to have read bytes from."""
    root = jobs_root(
        delta=None,
        category=None,
        log_path=log_path,
        files=files
        or {"train.log": "x" * 400, "extra.txt": "y" * 400, "weights.ckpt": "z" * 400},
    )
    job_dir = root / "cia-1"
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "cia-1",
                "log_path": log_path,
                "watch_files": [str(job_dir / "train.log")],
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "watch_cursors.json").write_text(
        json.dumps({str(job_dir / "train.log"): 400}), encoding="utf-8"
    )
    return root


class TestLogFinderCorpus:
    def test_a_file_watch_read_bytes_from_is_labelled_useful(self, jobs_root):
        """Observed usefulness, not the prompt's rules.

        ``weights.ckpt`` is labelled false here because Watch never read it, not
        because its extension is on a skip list -- labelling by the rules would
        teach the classifier the heuristic already in ``_scan_by_extension``.
        """
        result = build_log_finder_corpus(_readable_job(jobs_root), listing=_render_listing)
        labels = {Path(e.join_key.split(":", 1)[1]).name: e.label for e in result.examples}
        assert labels["train.log"] == "true"
        assert labels["weights.ckpt"] == "false"
        assert labels["extra.txt"] == "false"

    def test_every_example_shares_the_one_listing_as_its_state(self, jobs_root):
        """Per-file nouls over one state: one forward pass for the whole directory."""
        result = build_log_finder_corpus(_readable_job(jobs_root), listing=_render_listing)
        assert len({e.state for e in result.examples}) == 1

    def test_each_listed_file_gets_its_own_question(self, jobs_root):
        """The shape the tier's own docstring rules out, asserted from the corpus side.

        Every row used to carry one shared question object with the filename in
        the join key alone. N questions about one state are told apart by their
        text and nothing else, so that corpus was identical states carrying
        identical questions with opposing labels -- contradictory rather than
        hard, and unrankable by ``group_top1`` because every row scored the same.
        """
        result = build_log_finder_corpus(_readable_job(jobs_root), listing=_render_listing)
        assert len({e.question.question for e in result.examples}) == len(result.examples)

    def test_each_file_is_named_in_its_own_question(self, jobs_root):
        result = build_log_finder_corpus(_readable_job(jobs_root), listing=_render_listing)
        for example in result.examples:
            name = Path(example.join_key.split(":", 1)[1]).name
            assert name in example.question.question

    def test_the_emitted_examples_carry_the_canonical_question(self, jobs_root):
        """Behavioural, not textual: what the rows actually ask.

        Checking the import alone would pass a builder that imported the template
        and then filled it in differently -- and the label *is* part of the
        question, so an absolute path where the tier uses a relative one labels a
        question the tier never asks just as surely as a reworded one does.
        """
        from aorta.cia.watch.log_finder import listing_label, useful_question

        root = _readable_job(jobs_root)
        job_dir = root / "cia-1"
        result = build_log_finder_corpus(root, listing=_render_listing)
        asked = {e.question for e in result.examples}
        expected = {
            useful_question(listing_label(job_dir, e.join_key.split(":", 1)[1]))
            for e in result.examples
        }
        assert asked == expected

    def test_the_tier_asks_exactly_what_the_corpus_labelled(self, jobs_root):
        """Train and serve, end to end, over the same directory.

        The two paths derive their candidates separately -- the builder off the
        rendered listing, the tier off its own ``_candidates`` -- so this is the
        assertion that the wording, the template filling and the label all line
        up at once. The tier's set is a subset because ``_excluded`` drops a
        checkpoint before asking, while the corpus labels it as a negative.
        """
        pytest.importorskip("dspy", reason="the tier needs the [cia] extra")
        from aorta.cia.watch.log_finder import (
            LogFinder,
            _dir_listing,
            useful_question,
        )

        root = _readable_job(jobs_root)
        job_dir = root / "cia-1"
        labelled = {e.question.question for e in build_log_finder_corpus(root).examples}
        finder = LogFinder(config={})
        asked = {
            useful_question(label).question
            for label, _path in finder._candidates(_dir_listing(job_dir), job_dir)
        }

        assert asked, "the tier found no candidates, so this proves nothing"
        assert asked <= labelled

    def test_a_question_is_per_path_not_per_filename(self, jobs_root):
        """Two files can share a name; the question they are told apart by cannot.

        ``logs/train.log`` beside ``train.log`` is an ordinary job directory, and
        labelling both with the bare filename would put two opposing labels back
        on one question -- the same defect, one level down from the wording.
        """
        root = _readable_job(
            jobs_root, files={"train.log": "x" * 400, "logs/train.log": "y" * 400}
        )
        result = build_log_finder_corpus(root, listing=_render_listing)
        named = [e for e in result.examples if e.join_key.endswith("train.log")]

        assert len(named) == 2
        assert len({e.question.question for e in named}) == 2

    def test_it_says_which_extra_when_the_question_cannot_be_reached(self, monkeypatch):
        """The question lives with the tier, so a base install gets advice not a traceback."""
        import sys

        from aorta.local_classifier.corpus.log_finder import _useful_question

        monkeypatch.setitem(sys.modules, "aorta.cia.watch.log_finder", None)
        with pytest.raises(CorpusError, match=r"amd-aorta\[cia\]"):
            _useful_question(Path("."), "train.log")

    def test_the_candidates_are_exactly_what_the_listing_holds(self, jobs_root):
        """The security property: a path not in the listing cannot be named."""
        result = build_log_finder_corpus(_readable_job(jobs_root), listing=_render_listing)
        for example in result.examples:
            assert example.join_key.split(":", 1)[1] in example.state

    def test_selection_alone_is_not_evidence_of_usefulness(self, jobs_root):
        """A file Watch resolved and found empty all run was not worth watching."""
        root = jobs_root(
            delta=None, category=None, log_path="", files={"a.log": "x" * 400, "b.log": "y" * 400}
        )
        job_dir = root / "cia-1"
        (job_dir / "job.json").write_text(
            json.dumps(
                {"job_id": "cia-1", "log_path": "", "watch_files": [str(job_dir / "a.log")]}
            ),
            encoding="utf-8",
        )
        (job_dir / "watch_cursors.json").write_text(
            json.dumps({str(job_dir / "a.log"): 0}), encoding="utf-8"
        )
        result = build_log_finder_corpus(root, listing=_render_listing)
        assert not result.examples
        assert any("read no bytes" in reason for reason in result.skipped)

    def test_a_declared_log_path_means_the_llm_tier_never_ran(self, jobs_root):
        """The finding that bounds how much of this corpus can exist at all.

        ``poll.py`` prefers a declared path outright and ``aorta.cia.triage``
        always sets one, so on a machine whose jobs all came through triage,
        tier 3 has never executed.
        """
        root = _readable_job(jobs_root, log_path="/var/log/job.log")
        result = build_log_finder_corpus(root, listing=_render_listing)
        assert any("never consulted" in warning for warning in result.warnings)

    def test_the_class_imbalance_is_stated(self, jobs_root):
        """A job directory holds far more files than logs, so accuracy misleads."""
        result = build_log_finder_corpus(_readable_job(jobs_root), listing=_render_listing)
        assert any("group_top1" in warning for warning in result.warnings)

    def test_one_directory_is_one_ranking_decision(self, jobs_root):
        """Every file in a listing carries the job as its group.

        Track C ranks the listing and takes ``max_files`` off the front, so the
        score has to be over the directory and not over the files -- and the split
        must not divide one listing between the fit and the scored set.
        """
        result = build_log_finder_corpus(_readable_job(jobs_root), listing=_render_listing)
        assert {e.group for e in result.examples} == {"cia-1"}

    def test_a_single_file_directory_is_not_an_ambiguous_one(self, jobs_root):
        """One file is not a choice, so there is no decision to label."""
        root = _readable_job(jobs_root)
        one_entry = list(root.glob("cia-1/train.log"))[0]
        result = build_log_finder_corpus(
            root, listing=lambda _job_dir: f"{one_entry}  size=400  mtime=1.0"
        )
        assert any("fewer than two" in reason for reason in result.skipped)

    def test_a_missing_jobs_root_is_reported_rather_than_crashing(self, tmp_path: Path):
        result = build_log_finder_corpus(tmp_path / "absent")
        assert not result.examples
        assert any("no listings" in warning for warning in result.warnings)

    def test_the_default_listing_is_tier_threes_own(self, jobs_root):
        """What the injection point must not be allowed to drift from.

        Skipped without the agents' extra, because the real routine imports dspy;
        the builder says exactly that rather than reimplementing the listing.
        """
        pytest.importorskip("dspy", reason="the real _dir_listing needs the [cia] extra")
        result = build_log_finder_corpus(_readable_job(jobs_root))
        assert result.examples
        assert "size=" in next(iter(result.examples)).state

    def test_without_the_cia_extra_it_says_which_extra(self, monkeypatch):
        """Advice, not a traceback -- and not a quiet reimplementation either."""
        import sys

        from aorta.local_classifier.corpus.log_finder import _dir_listing

        # ``None`` in sys.modules is the documented way to make an import fail,
        # and it covers the installed case as well as the absent one.
        monkeypatch.setitem(sys.modules, "aorta.cia.watch.log_finder", None)
        with pytest.raises(CorpusError, match=r"amd-aorta\[cia\]"):
            _dir_listing(Path("."))


class TestRoundTrip:
    def test_a_corpus_survives_being_written_and_read(self, tmp_path: Path):
        examples = [
            LabelledExample(
                decision="watch_healthy",
                state="a log",
                question=Noul(question="clean?", when_true="yes", when_false="no"),
                label="true",
                join_key="job-1",
                source="/somewhere",
                baselines=(("dspy_signal", "WATCH_CLEAN"),),
                group="job-1",
            ),
            LabelledExample(
                decision="watch_signal",
                state="another log",
                question=Choice(
                    question="which?",
                    options=("nan", "oom"),
                    criteria=(("nan", "a non-finite value"),),
                ),
                label="oom",
                join_key="job-2",
            ),
        ]
        path = tmp_path / "corpus.jsonl"
        assert write_corpus(path, examples) == 2
        assert read_corpus(path) == examples

    def test_a_corpus_written_before_groups_existed_still_reads(self, tmp_path: Path):
        """Absent means "this row stands alone", which is what it was when written."""
        path = tmp_path / "corpus.jsonl"
        path.write_text(
            json.dumps(
                {
                    "decision": "watch_healthy",
                    "state": "a log",
                    "question": {"type": "noul", "instructions": "clean?"},
                    "label": "true",
                    "join_key": "job-1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        assert read_corpus(path)[0].group == ""

    def test_a_missing_corpus_says_how_to_build_one(self, tmp_path: Path):
        with pytest.raises(CorpusError, match="python -m aorta.local_classifier corpus"):
            read_corpus(tmp_path / "absent.jsonl")

    def test_a_malformed_line_names_itself(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        path.write_text('{"decision": "x"}\n', encoding="utf-8")
        with pytest.raises(CorpusError, match=r"corpus.jsonl:1"):
            read_corpus(path)

    def test_a_malformed_line_is_not_silently_dropped(self, tmp_path: Path):
        """A corpus that quietly loses a tenth of itself reports a score over a set
        nobody chose, and the number looks entirely normal."""
        path = tmp_path / "corpus.jsonl"
        path.write_text("{not json\n", encoding="utf-8")
        with pytest.raises(CorpusError):
            read_corpus(path)

    def test_an_empty_corpus_is_reported(self, tmp_path: Path):
        path = tmp_path / "corpus.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="no examples"):
            read_corpus(path)

    def test_a_score_question_is_refused(self, tmp_path: Path):
        """The third primitive is modelled nowhere here, so a corpus with one is foreign."""
        path = tmp_path / "corpus.jsonl"
        path.write_text(
            json.dumps(
                {
                    "decision": "watch_healthy",
                    "state": "s",
                    "question": {"type": "score", "instructions": "how bad?"},
                    "label": "1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(CorpusError, match="unsupported question type"):
            read_corpus(path)


def test_every_builder_is_reachable_by_name():
    """The CLI's ``--kind`` choices come from here, so a gap is a dead command."""
    assert sorted(BUILDERS) == ["log-finder", "proposer", "watch"]
    for builder, reads in BUILDERS.values():
        assert callable(builder)
        assert reads
