"""The opt-in decision log records policy steps without recording the paste."""

from __future__ import annotations

import json
import logging
import stat
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from aorta.chat import decision_log
from aorta.cia.autopsy.reporter import build_report

QUERY = (
    "customer kernel at /home/customer7/private/kernel.hip on 10.20.30.40\n"
    "```cpp\n__global__ void secret_kernel(float *out) { out[0] = 7; }\n```"
)
PLAN = "Read /home/customer7/private/kernel.hip, then run the secret kernel."
REASON = (
    "WaitCheck returns instruction evidence from "
    "/home/customer7/private/kernel.hip on 10.20.30.40."
)
SECRET_SELECTOR_REASON = (
    "Choose triage_kernel_source because secret_kernel contains customer "
    "credential ghp_0123456789abcdefghijklmnopqrstuvwxyz."
)
TOOL_ARGUMENTS = "{'source': '__global__ void secret_kernel() {}'}"
TOOL_OUTPUT = (
    "Job cia-20260918-123456-abcd — customer run\n"
    "Autopsy verdict:\n"
    "  category: gpu_race\n"
    "  confidence: 0.95\n"
    "private output token 8491"
)
REPLY = "The private kernel has a race; add a barrier."
DECISION_LOG_DOC = " ".join(
    (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "chat"
        / "configuration.md"
    )
    .read_text(encoding="utf-8")
    .split()
)


def state() -> dict:
    return {
        "route": "action",
        "candidate_tools": ["triage_kernel_source", "read_file"],
        "selection_rationale": REASON,
        "plan": PLAN,
        "tool_trace": [
            f"[triage_kernel_source({TOOL_ARGUMENTS})] →\n{TOOL_OUTPUT}"
        ],
        "critic_feedback": None,
        "command_output": REPLY,
        "messages": [AIMessage(content=REPLY)],
    }


@pytest.fixture(autouse=True)
def private_state_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv(decision_log.SESSION_LOG_ENV, raising=False)
    decision_log._warned_full.clear()


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class TestTheOptIn:
    @pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
    def test_it_is_off_by_default_and_for_false_values(self, monkeypatch, value):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, value)

        assert decision_log.session_log_mode() is None

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "summary"])
    def test_summary_values(self, monkeypatch, value):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, value)

        assert decision_log.session_log_mode() == "summary"

    def test_full_is_explicit(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "full")

        assert decision_log.session_log_mode() == "full"

    def test_disabled_logging_creates_no_state_directory(self):
        path = decision_log.record_turn(
            session_id="session-a",
            turn=1,
            query=QUERY,
            reply=REPLY,
            state=state(),
        )

        assert path is None
        assert not decision_log.decision_log_path("session-a").exists()


class TestSummaryModeStoresDecisionsNotContent:
    def test_content_is_reduced_to_shape_and_digest(self):
        summary = decision_log.summarize_text(QUERY)

        assert summary["characters"] == len(QUERY)
        assert summary["lines"] == len(QUERY.splitlines())
        assert summary["fences"] == 2
        assert len(summary["sha256"]) == 64
        assert "secret_kernel" not in json.dumps(summary)

    def test_no_prompt_plan_argument_output_or_answer_is_written(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        path = decision_log.record_turn(
            session_id="session-a",
            turn=3,
            query=QUERY,
            reply=REPLY,
            state=state(),
        )

        assert path is not None
        blob = path.read_text(encoding="utf-8")
        for secret in (
            "secret_kernel",
            "private output token 8491",
            "add a barrier",
            TOOL_ARGUMENTS,
        ):
            assert secret not in blob

    def test_the_selector_reason_is_digested_not_persisted(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        secret_state = state()
        secret_state["selection_rationale"] = SECRET_SELECTOR_REASON
        path = decision_log.record_turn(
            session_id="session-a",
            turn=1,
            query=QUERY,
            reply=REPLY,
            state=secret_state,
        )

        selection = next(
            record for record in _records(path) if record["event"] == "selection"
        )
        assert selection["reason"] == decision_log.summarize_text(
            SECRET_SELECTOR_REASON
        )
        blob = path.read_text(encoding="utf-8")
        assert "secret_kernel" not in blob
        assert "ghp_0123456789abcdefghijklmnopqrstuvwxyz" not in blob

    def test_route_ranking_tool_order_and_critic_are_recorded(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        path = decision_log.record_turn(
            session_id="session-a",
            turn=7,
            query=QUERY,
            reply=REPLY,
            state=state(),
        )
        records = _records(path)

        assert next(r for r in records if r["event"] == "route")["route"] == "action"
        selection = next(r for r in records if r["event"] == "selection")
        assert selection["ranked_tools"] == [
            "triage_kernel_source",
            "read_file",
        ]
        tool_event = next(r for r in records if r["event"] == "tool")
        assert tool_event["position"] == 1
        assert tool_event["tool"] == "triage_kernel_source"
        assert next(r for r in records if r["event"] == "critic")["accepted"] is True

    def test_cia_outcome_is_extracted_without_retaining_output(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        path = decision_log.record_turn(
            session_id="session-a",
            turn=1,
            query=QUERY,
            reply=REPLY,
            state=state(),
        )
        tool_event = next(
            record for record in _records(path) if record["event"] == "tool"
        )

        assert tool_event["cia_results"] == [
            {
                "job_id": "cia-20260918-123456-abcd",
                "category": "gpu_race",
                "confidence": 0.95,
            }
        ]
        assert "private output token" not in json.dumps(tool_event)

    def test_no_answer_is_not_mislabeled_as_critic_acceptance(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        incomplete = state()
        incomplete["command_output"] = None
        path = decision_log.record_turn(
            session_id="session-a",
            turn=1,
            query=QUERY,
            reply="",
            state=incomplete,
        )

        critic = next(
            record for record in _records(path) if record["event"] == "critic"
        )
        assert critic["accepted"] is None

    def test_every_event_has_the_stable_join_key_and_empty_resolution(
        self, monkeypatch
    ):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        path = decision_log.record_turn(
            session_id="session-a",
            turn=9,
            query=QUERY,
            reply=REPLY,
            state=state(),
        )

        for record in _records(path):
            assert (record["session_id"], record["turn"]) == ("session-a", 9)
            assert record["resolution"] is None


class TestPrivateStorage:
    def test_one_json_line_is_written_per_event(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        path = decision_log.record_turn(
            session_id="session-a",
            turn=1,
            query=QUERY,
            reply=REPLY,
            state=state(),
        )

        records = _records(path)
        assert [record["event"] for record in records] == [
            "input",
            "route",
            "selection",
            "plan",
            "tool",
            "critic",
            "answer",
        ]

    def test_sessions_get_different_files(self):
        assert decision_log.decision_log_path(
            "session-a"
        ) != decision_log.decision_log_path("session-b")

    def test_a_caller_controlled_id_cannot_name_a_path(self):
        path = decision_log.decision_log_path("../../outside")

        assert path.parent.name == "sessions"
        assert path.name.endswith(".jsonl")
        assert ".." not in path.name

    def test_directory_and_file_modes_are_private(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        path = decision_log.record_turn(
            session_id="session-a",
            turn=1,
            query=QUERY,
            reply=REPLY,
            state=state(),
        )

        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_a_symlink_is_not_followed(self, monkeypatch, caplog):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        path = decision_log.decision_log_path("session-a")
        path.parent.mkdir(parents=True)
        victim = path.parent.parent / "victim"
        victim.write_text("do not replace\n", encoding="utf-8")
        path.symlink_to(victim)

        with caplog.at_level(logging.WARNING, logger=decision_log.__name__):
            written = decision_log.record_turn(
                session_id="session-a",
                turn=1,
                query=QUERY,
                reply=REPLY,
                state=state(),
            )

        assert written is None
        assert victim.read_text(encoding="utf-8") == "do not replace\n"
        assert caplog.records


class TestFullModeIsLoudAndExplicit:
    def test_it_keeps_verbatim_content(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "full")
        path = decision_log.record_turn(
            session_id="session-a",
            turn=1,
            query=QUERY,
            reply=REPLY,
            state=state(),
        )
        blob = path.read_text(encoding="utf-8")

        assert "secret_kernel" in blob
        assert "private output token 8491" in blob
        assert REPLY in blob
        selection = next(
            record for record in _records(path) if record["event"] == "selection"
        )
        assert selection["reason"] == REASON

    def test_it_warns_once_per_session_and_names_the_file(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "full")

        with caplog.at_level(logging.WARNING, logger=decision_log.__name__):
            first = decision_log.record_turn(
                session_id="session-a",
                turn=1,
                query=QUERY,
                reply=REPLY,
                state=state(),
            )
            decision_log.record_turn(
                session_id="session-a",
                turn=2,
                query=QUERY,
                reply=REPLY,
                state=state(),
            )

        warnings = [
            record for record in caplog.records if "Full chat session" in record.message
        ]
        assert len(warnings) == 1
        assert str(first) in warnings[0].message


class TestThePublicPrivacyContractMatchesTheModes:
    def test_summary_names_selector_rationale_as_digested_content(self):
        assert (
            "Questions, selector rationale, plans, tool arguments, tool output, "
            "critic feedback, and answers are not stored in summary mode."
            in DECISION_LOG_DOC
        )
        assert (
            "character/byte/line/fence counts and a SHA-256 digest"
            in DECISION_LOG_DOC
        )

    def test_full_names_verbatim_rationale_and_unscrubbed_locations(self):
        assert "including selector rationale, verbatim" in DECISION_LOG_DOC
        assert (
            "does not scrub filesystem paths or IP addresses from the rationale"
            in DECISION_LOG_DOC
        )


class TestInvokeAgentIsTheSharedRecordingSeam:
    async def test_awaited_graph_records_its_returned_state(
        self, monkeypatch
    ):
        import aorta.chat.session as session

        class Graph:
            async def ainvoke(self, _initial):
                decision_log.note_tool_call(
                    "triage_kernel_source",
                    {"source": "__global__ void secret_native_call() {}"},
                )
                return state()

        monkeypatch.setattr(session, "agent_graph", Graph())
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")

        reply, _history, _state = await session.invoke_agent(
            QUERY, [], session_id="shared-session", turn=4
        )

        path = decision_log.decision_log_path("shared-session")
        assert reply == REPLY
        records = _records(path)
        assert {record["turn"] for record in records} == {4}
        tool_event = next(record for record in records if record["event"] == "tool")
        assert tool_event["arguments"]["characters"] > 0
        assert "secret_native_call" not in json.dumps(tool_event)

    async def test_streaming_path_records_the_same_returned_state(
        self, monkeypatch
    ):
        import aorta.chat.session as session

        class Graph:
            async def astream(self, _initial, stream_mode):
                assert "values" in stream_mode
                yield "values", state()

        monkeypatch.setattr(session, "agent_graph", Graph())
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")

        reply, _history, _state = await session.invoke_agent(
            QUERY,
            [],
            on_step=AsyncCallback(),
            session_id="stream-session",
            turn=2,
        )

        path = decision_log.decision_log_path("stream-session")
        assert reply == REPLY
        assert any(record["event"] == "selection" for record in _records(path))

    async def test_a_graph_failure_is_recorded_without_masking_it(
        self, monkeypatch
    ):
        import aorta.chat.session as session

        class Broken:
            async def ainvoke(self, _initial):
                raise RuntimeError("graph exploded")

        monkeypatch.setattr(session, "agent_graph", Broken())
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")

        with pytest.raises(RuntimeError, match="graph exploded"):
            await session.invoke_agent(
                QUERY, [], session_id="failed-session", turn=5
            )

        record = _records(decision_log.decision_log_path("failed-session"))[0]
        assert record["event"] == "failure"
        assert record["error_type"] == "RuntimeError"
        assert "graph exploded" not in json.dumps(record)


class AsyncCallback:
    async def __call__(self, _node: str, _delta: dict) -> None:
        pass


#: What ``list_cluster_jobs`` prints. Its lines carry a bare id rather than a
#: ``Job `` prefix, and it is registered without ``allow_cluster_jobs``, so it
#: is the tool most likely to have produced a record.
JOB_LISTING = (
    "Recent CIA jobs (jobs root: /home/me/cia-jobs):\n"
    "  cia-20260918-064455-4fd02e  recipe=kernel  status=completed"
    "  verdict=gpu_race @ 0.82\n"
    "      bundle: /home/me/cia-jobs/cia-20260918-064455-4fd02e/bundle\n"
    "  cia-20260918-071122-9ab31c  recipe=asm  status=completed"
    "  verdict=numeric_instability @ 0.41\n"
    "      bundle: /home/me/cia-jobs/cia-20260918-071122-9ab31c/bundle"
)

#: One result naming two jobs, each with its own verdict.
TWO_JOBS = (
    "Job cia-20260918-064455-4fd02e (slurm 1) - racy kernel\n"
    "  category:   gpu_race\n"
    "  confidence: 0.82\n"
    "Job cia-20260918-071122-9ab31c (slurm 2) - the fixed rebuild\n"
    "  category:   numeric_instability\n"
    "  confidence: 0.41"
)


class TestEveryNamedJobIsFoundAndKeepsItsOwnVerdict:
    """Both of these were found by running the extraction, not by reading it."""

    def test_a_job_listing_yields_its_ids(self):
        """Before: no ids at all, because the lines carry no ``Job `` prefix."""
        results = decision_log._cia_results(JOB_LISTING)

        assert [result["job_id"] for result in results] == [
            "cia-20260918-064455-4fd02e",
            "cia-20260918-071122-9ab31c",
        ]

    def test_the_bundle_path_is_not_read_as_another_job(self):
        """The id repeats inside the path on the next line, and is not a job."""
        results = decision_log._cia_results(JOB_LISTING)

        assert len(results) == 2

    def test_a_listing_without_a_parsable_verdict_says_so(self):
        """``verdict=x @ y`` is a third rendering; null beats guessing at it.

        The id is what makes the row joinable, and the report under
        ``jobs_root`` is a better source for a verdict than a rendered line.
        """
        results = decision_log._cia_results(JOB_LISTING)

        assert [result["category"] for result in results] == [None, None]
        assert [result["confidence"] for result in results] == [None, None]

    def test_each_job_keeps_its_own_verdict(self):
        """Before: the second job was recorded as ``gpu_race @ 0.82``.

        The direction of that error is what makes it worth fixing. A job whose
        verdict is missing is absent from whatever reads this; a job wearing
        another job's category is an answer, and a wrong one.
        """
        assert decision_log._cia_results(TWO_JOBS) == [
            {
                "job_id": "cia-20260918-064455-4fd02e",
                "category": "gpu_race",
                "confidence": 0.82,
            },
            {
                "job_id": "cia-20260918-071122-9ab31c",
                "category": "numeric_instability",
                "confidence": 0.41,
            },
        ]

    def test_one_job_named_twice_is_recorded_once_and_keeps_its_verdict(self):
        """Announce-then-finish is the ordinary shape of a progress log.

        Deduplicating to the *first* mention recorded the row and dropped the
        answer: the span for a repeated job ended at its own second mention, so
        everything after it -- which is where a verdict appears, because a job
        has no verdict when it starts -- was never read.
        """
        results = decision_log._cia_results(
            "Job cia-20260918-064455-4fd02e started\n"
            "Job cia-20260918-064455-4fd02e done\n"
            "  category: gpu_race\n  confidence: 0.82"
        )

        assert results == [
            {
                "job_id": "cia-20260918-064455-4fd02e",
                "category": "gpu_race",
                "confidence": 0.82,
            }
        ]

    def test_a_job_named_again_after_another_job_keeps_its_verdict(self):
        """The case that decides the rule, rather than just the reported input.

        Bounding a span at the next *distinct* id fixes the announce-then-finish
        shape above and still loses this one: the verdict follows a mention of
        A that is not A's first, with B named in between. Every span that
        follows a mention of A therefore belongs to A, not only the first.
        """
        results = decision_log._cia_results(
            "Job cia-20260918-064455-4fd02e started\n"
            "Job cia-20260918-071122-9ab31c started\n"
            "Job cia-20260918-064455-4fd02e done\n"
            "  category: gpu_race\n  confidence: 0.82"
        )

        assert results == [
            {
                "job_id": "cia-20260918-064455-4fd02e",
                "category": "gpu_race",
                "confidence": 0.82,
            },
            {
                "job_id": "cia-20260918-071122-9ab31c",
                "category": None,
                "confidence": None,
            },
        ]

    def test_a_bundle_path_repeating_the_id_does_not_truncate_the_verdict(self):
        """One reported job, and it used to lose its own verdict.

        ``_JOB_ID`` matches a bare id at the start of a line as well as a
        ``Job`` prefix, and a listing prints the bundle path under the job --
        which repeats the id. So this needs no duplicate *announcement* to hit
        the same defect, which is what makes it more than a curiosity of the
        input the review quoted.
        """
        results = decision_log._cia_results(
            "Job cia-20260918-064455-4fd02e\n"
            "  cia-20260918-064455-4fd02e/bundle.tar.gz\n"
            "  category: numeric_instability\n  confidence: 0.41"
        )

        assert results == [
            {
                "job_id": "cia-20260918-064455-4fd02e",
                "category": "numeric_instability",
                "confidence": 0.41,
            }
        ]

    def test_widening_the_read_does_not_leak_a_verdict_backwards(self):
        """The narrowness control, and the property this PR exists to protect.

        A span begins at a mention of its own job and ends before the next job
        is named, so text following a mention of B is B's however many times A
        was named earlier. If this ever fails, the fix above has reintroduced
        exactly the misattribution the PR removed.
        """
        results = decision_log._cia_results(
            "Job cia-20260918-064455-4fd02e started\n"
            "Job cia-20260918-071122-9ab31c\n"
            "  category: gpu_race\n  confidence: 0.82"
        )

        assert results == [
            {
                "job_id": "cia-20260918-064455-4fd02e",
                "category": None,
                "confidence": None,
            },
            {
                "job_id": "cia-20260918-071122-9ab31c",
                "category": "gpu_race",
                "confidence": 0.82,
            },
        ]

    def test_the_first_verdict_wins_so_the_change_is_additive_only(self):
        """Why the first match across the spans wins rather than the last.

        This is the safety property of widening the read, so it is pinned with
        two *different* verdicts rather than the same one twice -- which is the
        only input that can tell first-wins from last-wins apart. Taking the
        first means a job that already had a verdict reports the same one it
        always did, and the change can only turn a ``None`` into a verdict,
        never one verdict into another.
        """
        first_only = (
            "Job cia-20260918-064455-4fd02e\n"
            "  category: gpu_race\n  confidence: 0.82"
        )
        then_revised = (
            f"{first_only}\n"
            "Job cia-20260918-064455-4fd02e\n"
            "  category: numeric_instability\n  confidence: 0.41"
        )

        assert decision_log._cia_results(then_revised) == decision_log._cia_results(
            first_only
        )
        assert decision_log._cia_results(then_revised)[0]["category"] == "gpu_race"


class TestOnlyARealIdCountsAsAJobBeingNamed:
    """The line-start branch reads *any* leading token, and had to stop.

    ``nan-`` is a live prefix here, so an indented numeric line beginning with
    the word ``nan-`` was read as a job being announced. The damage is not a
    stray row: a phantom id makes the turn look like it ran a CIA job, so the
    record grows a ``jobs_root`` and a result list describing work that never
    happened. The prefixed (``Job ``) and JSON forms stay deliberately loose,
    because there the *context* is the evidence; a bare token at a line start
    has no context, so it has to look like ``new_job_id()`` produced it.
    """

    def test_a_line_that_merely_starts_with_nan_is_not_a_job(self):
        """Before: one result, id ``nan-trap``, wearing the verdict below it."""
        assert (
            decision_log._cia_results(
                "  nan-trap value mean_sq=0\n"
                "  category: gpu_race\n"
                "  confidence: 0.9\n"
            )
            == []
        )

    def test_a_bare_id_at_a_line_start_is_still_a_job(self):
        """The narrowness control: the tightening must not cost the listing.

        A listing prints its ids bare and indented, with no ``Job`` prefix, so
        if the lookahead were even slightly too strict this whole shape would
        go dark -- which is the defect the line-start branch was added to fix.
        """
        results = decision_log._cia_results(JOB_LISTING)

        assert [result["job_id"] for result in results] == [
            "cia-20260918-064455-4fd02e",
            "cia-20260918-071122-9ab31c",
        ]

    def test_a_prefixed_id_that_is_not_the_generated_shape_is_still_read(self):
        """``Job `` is the context, so the id after it stays unconstrained.

        Hand-written ids appear in fixtures and in a user typing a job name;
        the lookahead is the price of a *bare* token, not of every id.
        """
        results = decision_log._cia_results(
            "Job cia-a1\n  category: gpu_race\n  confidence: 0.82"
        )

        assert results == [
            {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.82}
        ]


class TestAVerdictPrintedBeforeItsIdStillLands:
    """Spans run forward from an id, and JSON does not promise that order.

    ``json.dumps(..., sort_keys=True)`` puts ``category`` and ``confidence``
    before ``job_id``, so every verdict sits *behind* the id that owns it and
    the forward span sees nothing. Reading the whole output instead would fix
    one job and break two, so the fix is structural: parse the JSON and take
    each object whole. The line-oriented fallback below stays for the renderings
    that are not JSON, and only where it cannot misattribute.
    """

    def _dumped(self, *jobs: dict) -> str:
        payload = jobs[0] if len(jobs) == 1 else {"results": list(jobs)}
        return json.dumps(payload, sort_keys=True, indent=2)

    def test_one_job_dumped_with_sorted_keys_keeps_its_verdict(self):
        """Before: ``category`` and ``confidence`` were both ``None``."""
        output = self._dumped(
            {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.82}
        )

        assert decision_log._cia_results(output) == [
            {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.82}
        ]

    def test_two_jobs_dumped_with_sorted_keys_do_not_swap_verdicts(self):
        """The case a whole-output fallback cannot serve, hence the pre-pass.

        With two jobs and one shared text there is no way to tell whose verdict
        is whose, so a fallback would hand both jobs the first verdict it found
        -- the misattribution this PR exists to remove, reintroduced by the fix
        for its sibling. Structure is what separates them.
        """
        output = self._dumped(
            {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.82},
            {"job_id": "cia-b2", "category": "numeric_instability",
             "confidence": 0.41},
        )

        assert decision_log._cia_results(output) == [
            {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.82},
            {
                "job_id": "cia-b2",
                "category": "numeric_instability",
                "confidence": 0.41,
            },
        ]

    def test_a_nested_object_does_not_inherit_a_parent_verdict(self):
        """Each object is taken whole and never merged with what encloses it.

        Otherwise a summary verdict at the top level would be copied down onto
        every job in the list beneath it, which is misattribution again -- this
        time sourced from a field that is not even about a single job.
        """
        output = json.dumps(
            {
                "category": "gpu_race",
                "confidence": 0.82,
                "results": [{"job_id": "cia-a1"}],
            },
            sort_keys=True,
        )

        assert decision_log._cia_results(output) == [
            {"job_id": "cia-a1", "category": None, "confidence": None}
        ]

    def test_a_job_named_twice_in_json_keeps_the_verdict_of_its_later_row(self):
        """The JSON twin of ``test_one_job_named_twice_is_recorded_once``.

        Announce-then-finish is no less ordinary in a ``results`` array than in
        a progress log, and taking the first object and discarding the rest
        recorded ``cia-a1`` with ``category: None`` -- the record asserting
        that a job which reached a verdict reached none, about output that
        states the verdict two lines down.
        """
        output = json.dumps(
            {
                "results": [
                    {"job_id": "cia-a1", "status": "running"},
                    {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.82},
                ]
            },
            sort_keys=True,
        )

        assert decision_log._cia_results(output) == [
            {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.82}
        ]

    def test_a_later_row_fills_only_the_half_the_earlier_one_left_empty(self):
        """First valid value wins, per half -- the text reader's rule exactly.

        Per half rather than per object, because a row that names a category
        and no confidence is a verdict that *was* found; and first rather than
        last, so a stale row printed after a final one cannot overwrite an
        answer that was already being reported.
        """
        output = json.dumps(
            {
                "results": [
                    {"job_id": "cia-a1", "category": "gpu_race"},
                    {"job_id": "cia-a1", "category": "numeric_instability",
                     "confidence": 0.41},
                ]
            },
            sort_keys=True,
        )

        assert decision_log._cia_results(output) == [
            {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.41}
        ]

    def test_a_repeated_job_keeps_the_position_its_first_mention_earned(self):
        """Document order is the order the jobs were announced in.

        Re-appending on the later mention would reorder the rows against the
        output they were read from, which is the one thing a reader uses to
        pair a row with what they saw.
        """
        output = json.dumps(
            {
                "results": [
                    {"job_id": "cia-a1"},
                    {"job_id": "cia-b2", "category": "gpu_race", "confidence": 0.5},
                    {"job_id": "cia-a1", "category": "hang", "confidence": 0.9},
                ]
            },
            sort_keys=True,
        )

        assert [row["job_id"] for row in decision_log._cia_results(output)] == [
            "cia-a1",
            "cia-b2",
        ]

    def test_a_locator_for_an_already_recorded_job_is_still_walked_through(self):
        """A pointer at a job is not a mention of its verdict, ever.

        The locator check runs before the merge on purpose. With the order the
        other way a bundle pointer whose id was already recorded returned at
        the duplicate check and never descended, so a *different* job nested
        under it was dropped -- the id being known turning the walk off for
        everything below it.
        """
        output = json.dumps(
            {
                "results": [{"job_id": "cia-a1", "category": "gpu_race",
                             "confidence": 0.82}],
                "bundle": {
                    "job_id": "cia-a1",
                    "root": "/jobs/cia-a1",
                    "child": {"job_id": "cia-b2", "category": "hang",
                              "confidence": 0.3},
                },
            },
            # Key order left alone here, unlike its neighbours: the claim is
            # that the locator is descended into *after* its id is already
            # recorded, and sorting would put `bundle` first and walk it before
            # anything had claimed `cia-a1`.
        )

        assert decision_log._cia_results(output) == [
            {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.82},
            {"job_id": "cia-b2", "category": "hang", "confidence": 0.3},
        ]

    def test_a_json_field_of_the_wrong_type_reads_as_no_verdict(self):
        """A number is not a category and ``"high"`` is not a confidence.

        Being JSON says the shape is machine-written, not that the values are
        the ones this record can carry, so they go through the same vocabulary
        the line reader uses. ``None`` is the honest answer; coercing is how a
        record ends up asserting something nobody said.
        """
        output = json.dumps(
            {"job_id": "cia-x", "category": 3, "confidence": "high"}, sort_keys=True
        )

        assert decision_log._cia_results(output) == [
            {"job_id": "cia-x", "category": None, "confidence": None}
        ]

    def test_json_that_names_no_job_does_not_suppress_the_line_reader(self):
        """Why the pre-pass answers ``None`` and not ``[]`` when it finds nothing.

        A tool that returns JSON wrapping a *rendered* log has its ids inside a
        string, where a structural read cannot see them and the line reader can.
        If "not applicable" and "applicable, no jobs" collapsed into the same
        empty answer, every such output would go dark -- the pre-pass added for
        one shape silently disabling extraction for another.

        The id is what is asserted, not the verdict: the verdict patterns are
        line-anchored and ``json.dumps`` escapes the newlines inside the string,
        so no reader recovers one here. Finding the job is the whole claim, and
        it is the part the collapse would destroy.
        """
        output = json.dumps(
            {
                "ok": True,
                "stdout": "Job cia-a1 done\n  category: gpu_race",
            },
            sort_keys=True,
        )

        assert [
            result["job_id"] for result in decision_log._cia_results(output)
        ] == ["cia-a1"]

    def test_one_job_in_plain_text_recovers_a_verdict_printed_above_it(self):
        """The fallback, and the only shape it is allowed to apply to.

        Not every rendering is JSON. When exactly one job is named and its
        forward spans yield nothing, the whole output is the span -- safe
        precisely because there is one job to attribute to.
        """
        results = decision_log._cia_results(
            "  category: gpu_race\n  confidence: 0.82\nJob cia-a1 done"
        )

        assert results == [
            {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.82}
        ]

    def test_two_jobs_in_plain_text_do_not_fall_back(self):
        """The bound on the fallback, stated as a test rather than a comment.

        Both verdicts here precede both ids, so neither job can be served
        without guessing. Two ``None``s are a record that is quiet about what
        it does not know; two copies of ``gpu_race`` would be a record that is
        wrong about one of them.
        """
        results = decision_log._cia_results(
            "  category: gpu_race\n  confidence: 0.82\n"
            "Job cia-a1 done\nJob cia-b2 done"
        )

        assert [result["category"] for result in results] == [None, None]


class TestTheInTreeAutopsyReportIsReadAsTheVerdictItIs:
    """`report.json` puts the id under `bundle` and the verdict beside it.

    The walker takes "an object carrying a `job_id`" as a result row, which is
    right for every shape above and wrong for this one: the id-bearing object
    is a *locator* -- `{"job_id": ..., "root": ...}` -- and the verdict lives on
    its parent. Walking it as the row recorded the id, left `category` and
    `confidence` `None`, claimed the output as structured, and never fell
    through to the line reader that would have found them. A record that is
    joinable and wrong is worse than one that is not joinable: it asserts that
    this job reached no verdict, about the one file whose purpose is to carry
    one.

    Built from `build_report` rather than from a literal, so the day the report
    schema moves these tests move with it instead of pinning a shape nothing
    emits.
    """

    def _report(self, **overrides) -> dict:
        fields = {
            "session_id": "session-a",
            "generated_at": "2026-01-01T00:00:00Z",
            "bundle_job_id": "cia-20260101-000000-abcdef",
            "bundle_root": "/jobs/cia-20260101-000000-abcdef",
            "kb_version": None,
            "category": "gpu_race",
            "confidence": 0.82,
            "rationale": "consan named the kernel",
            "evidence": [{"source": "consan", "detail": "race at 0x28"}],
            "next_probes": [{"mitigation": "hsa_no_sdma"}],
            "tooling_gaps": [],
        }
        fields.update(overrides)
        return build_report(**fields)

    @pytest.mark.parametrize("sort_keys", [False, True])
    def test_a_raw_report_dump_keeps_its_verdict(self, sort_keys):
        """Before: the id landed and both halves of the verdict were dropped.

        Both key orders, because `sort_keys=True` is the rendering the whole
        JSON pre-pass exists for and insertion order is what `json.dumps`
        gives by default -- and the defect was indifferent to which.
        """
        output = json.dumps(self._report(), sort_keys=sort_keys, indent=2)

        assert decision_log._cia_results(output) == [
            {
                "job_id": "cia-20260101-000000-abcdef",
                "category": "gpu_race",
                "confidence": 0.82,
            }
        ]

    def test_the_locator_does_not_claim_the_output_from_the_line_reader(self):
        """The mechanism, asserted separately from the result.

        The structural read now finds no result row in a report, so it answers
        `None` -- "this is not JSON with a result object in it" -- and the line
        reader gets its turn. Asserting only the verdict above would pass just
        as well if the fix had merged the parent's keys downward, which is the
        thing `test_a_nested_object_does_not_inherit_a_parent_verdict` forbids.
        """
        output = json.dumps(self._report(), sort_keys=True)

        assert decision_log._cia_results_from_json(output) is None

    def test_a_locator_key_that_sorts_after_the_verdict_still_lands(self):
        """Why declining is safe here, and not a swap of one ordering bug for another.

        Handing this back to the line reader looks like a return to the
        span-order dependence the pre-pass was added to remove. It is not, and
        the reason is that a report names exactly one job: rename `bundle` to
        anything sorting after `category` and the forward span yields nothing,
        so the one-job whole-output fallback answers -- which is safe precisely
        because there is one job to attribute to.
        """
        report = json.loads(json.dumps(self._report()))
        report["source"] = report.pop("bundle")
        output = json.dumps(report, sort_keys=True)

        assert decision_log._cia_results(output) == [
            {
                "job_id": "cia-20260101-000000-abcdef",
                "category": "gpu_race",
                "confidence": 0.82,
            }
        ]

    def test_locators_that_carry_their_own_verdicts_are_still_results(self):
        """Narrowness, and the reason the predicate names the verdict keys too.

        Declining every object with a `root` key would drop a verdict an object
        genuinely states about itself. The decline is for a locator that says
        nothing about the outcome, not for every object that points at a
        directory.

        Two jobs, because one would not discriminate: with a single job the
        line reader's whole-output fallback recovers the verdict anyway, so a
        version of this check that declined on `root` alone would still pass.
        With two, declining hands `cia-a1` the span that runs into `cia-b2`'s
        keys -- the misattribution the structural read exists to prevent.
        """
        output = json.dumps(
            {
                "results": [
                    {
                        "job_id": "cia-a1",
                        "root": "/jobs/cia-a1",
                        "category": "gpu_race",
                        "confidence": 0.82,
                    },
                    {
                        "job_id": "cia-b2",
                        "root": "/jobs/cia-b2",
                        "category": "numeric_silent",
                        "confidence": 0.41,
                    },
                ]
            },
            sort_keys=True,
        )

        assert decision_log._cia_results(output) == [
            {"job_id": "cia-a1", "category": "gpu_race", "confidence": 0.82},
            {"job_id": "cia-b2", "category": "numeric_silent", "confidence": 0.41},
        ]

    def test_a_bare_id_in_a_results_list_is_still_taken_verdictless(self):
        """The other half of the narrowness, and the one that costs the most.

        `test_a_nested_object_does_not_inherit_a_parent_verdict` requires this
        row to be recorded with two `None`s rather than handed the summary
        verdict above it. Declining on "no verdict keys" alone would instead
        drop it into the line reader, where one job plus the whole-output
        fallback hands it exactly the `gpu_race` that test forbids. `root` is
        what separates the two cases.
        """
        output = json.dumps(
            {
                "category": "gpu_race",
                "confidence": 0.82,
                "results": [{"job_id": "cia-a1"}],
            },
            sort_keys=True,
        )

        assert decision_log._cia_results(output) == [
            {"job_id": "cia-a1", "category": None, "confidence": None}
        ]


class TestTheFieldsThatMakeARecordResolvable:
    """Four additions, each answering a question a record could not."""

    def _one(self, monkeypatch, event: str, **overrides) -> dict:
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        turn_state = state()
        turn_state.update(overrides.pop("state", {}))
        path = decision_log.record_turn(
            session_id="session-a",
            turn=1,
            query=QUERY,
            reply=REPLY,
            state=turn_state,
            **overrides,
        )
        return next(
            record for record in _records(path) if record["event"] == event
        )

    def test_the_jobs_root_sits_beside_the_ids(self, monkeypatch, tmp_path):
        """Without it a recorded id resolves against nothing.

        The bundle, the autopsy report and the probe cells that would say what
        actually happened all live under this root, and no other field names
        it -- so it is what an empty ``resolution`` is waiting on.
        """
        monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(tmp_path / "cia-jobs"))
        from aorta.chat import config

        config.reset_settings()
        # `finally`, like the test below: cleanup after the assert runs only on
        # the pass, so a failure here leaves the patched settings singleton in
        # place for every test that follows and turns one red into many.
        try:
            tool_event = self._one(monkeypatch, "tool")
        finally:
            config.reset_settings()

        assert tool_event["jobs_root"] == str(tmp_path / "cia-jobs")

    def test_the_root_is_raw_in_summary_mode_and_the_docs_say_so(
        self, monkeypatch, tmp_path
    ):
        """The one field summary mode does not summarise, pinned to its disclosure.

        Every other free-text field goes through ``_content`` and becomes
        counts plus a digest. ``jobs_root`` cannot: a digest of a path resolves
        nothing, which is the whole reason it is recorded. So summary mode --
        which ``configuration.md`` describes as privacy-preserving, and warns
        about paths only for *full* mode -- retains an absolute path holding
        the operator's username.

        That is the right behaviour and the wrong documentation, so this pins
        both halves together: the rawness, and the sentence disclosing it. A
        future field that escapes summarisation gets caught by the first
        assertion; a doc rewrite that drops the warning gets caught by the
        second.
        """
        root = tmp_path / "cia-jobs"
        monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(root))
        from aorta.chat import config

        config.reset_settings()
        try:
            # `SESSION_LOG_ENV=1` is summary mode; `_one` sets it.
            tool_event = self._one(monkeypatch, "tool")
        finally:
            config.reset_settings()

        assert tool_event["mode"] == "summary"
        # Verbatim, not counts-and-a-digest the way `output` beside it is.
        assert tool_event["jobs_root"] == str(root)
        assert isinstance(tool_event["output"], dict), (
            "the control: summary mode still summarises everything else, so a "
            "mode that stopped redacting at all would not pass here"
        )

        doc = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "chat"
            / "configuration.md"
        ).read_text(encoding="utf-8")
        summary_half = doc.split("AORTA_CHAT_SESSION_LOG=full")[0]
        assert "jobs_root" in summary_half, (
            "configuration.md documents summary mode without naming jobs_root, "
            "so an operator reading it would not know an absolute path "
            "containing their username is retained."
        )

    def test_full_mode_keeps_the_rationale_verbatim_and_the_docs_say_so(
        self, monkeypatch
    ):
        """The same pairing as above, for the field the page described wrongly.

        ``configuration.md`` promised the selector's reason had "filesystem
        paths and IP addresses always scrubbed regardless of ``--no-redact``".
        The decision log runs no scrubber in either mode -- ``_content`` either
        summarises or returns the text -- and ``--no-redact`` governs what is
        sent *to the model*, not what is written here. So an operator was
        promised a protection that does not exist, on the mode that needs it
        most, two paragraphs above the page's own correct warning.

        The behaviour is right: full mode exists to keep a transcript. Both
        halves are pinned together so the next rewrite cannot re-promise it.
        """
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "full")
        rationale = "read /home/pat/secrets.log after reaching 10.1.2.3"
        turn_state = state()
        turn_state["selection_rationale"] = rationale
        path = decision_log.record_turn(
            session_id="session-a",
            turn=1,
            query=QUERY,
            reply=REPLY,
            state=turn_state,
        )
        selection = next(
            record for record in _records(path) if record["event"] == "selection"
        )

        assert selection["reason"] == rationale, (
            "full mode stores the rationale as it was written; nothing here "
            "scrubs it"
        )

        # Joined into one line first: the claim was split across a wrap, and a
        # re-wrap must not be able to slip it past this test.
        doc = " ".join(
            (
                Path(__file__).resolve().parents[2]
                / "docs"
                / "chat"
                / "configuration.md"
            )
            .read_text(encoding="utf-8")
            .split()
        )
        assert "scrubbed regardless of `--no-redact`" not in doc, (
            "configuration.md promises the decision log scrubs the selector "
            "rationale. It does not, in either mode."
        )
        assert "verbatim and unscrubbed" in doc, (
            "the bullet has to say what full mode actually retains, or an "
            "operator reading the list stops before the warning below it"
        )

    def test_a_tool_that_named_no_job_carries_no_root(self, monkeypatch):
        """It is only meaningful where there is an id to resolve."""
        tool_event = self._one(
            monkeypatch,
            "tool",
            state={"tool_trace": ["[read_file(path='x')] →\nfile contents"]},
        )

        assert "jobs_root" not in tool_event

    def test_the_front_door_is_recorded_on_every_event(self, monkeypatch):
        """A browser demo and a scripted CLI run are not the same population."""
        record = self._one(monkeypatch, "route", front_door="ui")

        assert record["front_door"] == "ui"

    def test_an_unstated_front_door_is_null_rather_than_guessed(
        self, monkeypatch
    ):
        assert self._one(monkeypatch, "route")["front_door"] is None

    def test_the_turn_duration_is_recorded_on_the_answer(self, monkeypatch):
        """A four-minute turn and a four-second one are different decisions."""
        record = self._one(monkeypatch, "answer", duration_seconds=214.7)

        assert record["duration_seconds"] == 214.7

    def test_the_critic_iteration_count_is_recorded(self, monkeypatch):
        """``accepted`` says the answer was taken; this says what that cost."""
        record = self._one(monkeypatch, "critic", state={"iteration": 3})

        assert record["accepted"] is True
        assert record["iterations"] == 3

    def test_a_failed_turn_carries_the_time_it_spent_failing(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        path = decision_log.record_failure(
            session_id="session-a",
            turn=1,
            query=QUERY,
            error=RuntimeError("boom"),
            front_door="cli",
            duration_seconds=12.5,
        )
        record = _records(path)[0]

        assert record["duration_seconds"] == 12.5
        assert record["front_door"] == "cli"


class TestCliUsesOneExplicitSessionKey:
    async def test_ask_once_forwards_the_session_and_turn(
        self, monkeypatch
    ):
        from aorta.cli import chat

        seen: dict = {}

        async def invoke(query, history, **decision):
            seen.update(decision)
            return "answer", history, {}

        monkeypatch.setattr(chat, "_render_plain", lambda _reply: None)

        _history, ok = await chat._ask_once(
            invoke,
            "question",
            [],
            "plain",
            False,
            session_id="cli-session",
            turn=6,
        )

        assert ok is True
        assert seen == {
            "session_id": "cli-session",
            "turn": 6,
            "front_door": "cli",
        }
