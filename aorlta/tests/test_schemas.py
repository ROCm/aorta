"""The Section 6.1/6.2 records round-trip through JSON and are decoded strictly."""

import json

import pytest

from aorlta.schemas import (
    Candidate,
    ChainRecord,
    CompileResult,
    LegJob,
    LegResult,
    PromptRecord,
    RewardResult,
    SchemaError,
    TaskRef,
    TrainingSample,
    TurnRecord,
    from_dict,
    to_dict,
)

H = "a" * 64
TASK = TaskRef("c-1", None, None, "frozen", H)
COMPILE = CompileResult(False, 1, 0.61, [{"cls": "use of undeclared identifier <name>", "line": 12, "col": 5, "token": "tid"}], H)
TURN = TurnRecord(0, 1905968311, H, 6097, [10, 11, 248046], [-0.1, -2.25, -0.0], {"type": "stop", "matched": 248046},
                  6000, 6096, 1793600000.0, 1793600010.5, COMPILE, H, "pass")
LEG = LegResult("san.asan", "finding", None, [{"kind": "heap-buffer-overflow"}], 1, 1, "w3",
                {"before": "c1", "after": "c2", "clean": True}, False, 1.0, 2.0, 3.5)
RECORDS = [
    TASK,
    PromptRecord("standin:k1", [1, 2, 3], None, H, TASK, "standin"),
    COMPILE,
    TURN,
    ChainRecord("r/1/standin:k1/0", "standin:k1", 0, 2, "r:v0", [TURN], 0, "compiled"),
    TrainingSample("r/1/standin:k1/0", 1, "standin:k1", [1, 2, 3], [10, 11, 248046], [-0.1, -2.25, -0.0]),
    Candidate("r/1/standin:k1/0", "r", 1, "standin:k1", 0, "r:v0", "```cpp\n```", 3, 4096, TASK, "fake", None),
    LEG,
    RewardResult("r/1/standin:k1/0", -0.177, "scored", None, {"base": -0.17, "broke_healthy": 0.0, "length": -0.007},
                 {"name": "standin.v11", "version": "11", "code_sha256": H}, {"san.asan": LEG}, True, None, False, H,
                 {"queue": 0.1, "run": 2.0, "total": 2.1}),
    LegJob("j1", "san.asan", "r/1/standin:k1/0", None, H, H, 2, "w1", {"repeats": 3}, 5.0),
]


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: type(r).__name__)
def test_records_round_trip_through_json(record):
    data = json.loads(json.dumps(to_dict(record)))
    assert from_dict(type(record), data) == record


def test_group_key():
    assert RECORDS[5].group_key == (1, "standin:k1")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(extra=1), "unknown key"),
        (lambda d: d.pop("state"), "missing required key"),
        (lambda d: d.update(state="ok"), "LegResult.state: expected one of"),
        (lambda d: d.update(repeats=True), "LegResult.repeats: expected an int"),
        (lambda d: d.update(t_queued="1.0"), "LegResult.t_queued: expected a finite number"),
        (lambda d: d.update(findings={"a": 1}), "LegResult.findings: expected a list"),
    ],
)
def test_malformed_leg_results_are_refused(mutate, message):
    data = to_dict(LEG)
    mutate(data)
    with pytest.raises(SchemaError, match=message):
        from_dict(LegResult, data)


def test_nested_errors_name_their_path():
    data = to_dict(RECORDS[8])
    data["legs"]["san.asan"]["state"] = "clean-ish"
    with pytest.raises(SchemaError, match=r"RewardResult\.legs\.san\.asan\.state"):
        from_dict(RewardResult, data)
