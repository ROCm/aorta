"""The opt-in decision log records policy steps without recording the paste."""

from __future__ import annotations

import json
import logging
import stat
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from aorta.chat import decision_log

QUERY = (
    "customer kernel at /home/customer7/private/kernel.hip on 10.20.30.40\n"
    "```cpp\n__global__ void secret_kernel(float *out) { out[0] = 7; }\n```"
)
PLAN = "Read /home/customer7/private/kernel.hip, then run the secret kernel."
REASON = (
    "WaitCheck returns instruction evidence from "
    "/home/customer7/private/kernel.hip on 10.20.30.40."
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

    def test_the_selector_reason_is_kept_but_always_scrubbed(self, monkeypatch):
        monkeypatch.setenv(decision_log.SESSION_LOG_ENV, "1")
        path = decision_log.record_turn(
            session_id="session-a",
            turn=1,
            query=QUERY,
            reply=REPLY,
            state=state(),
        )

        selection = next(
            record for record in _records(path) if record["event"] == "selection"
        )
        assert "WaitCheck returns instruction evidence" in selection["reason"]
        assert "/home/customer7" not in selection["reason"]
        assert "10.20.30.40" not in selection["reason"]
        assert "<PATH:" in selection["reason"]

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
        assert seen == {"session_id": "cli-session", "turn": 6}
