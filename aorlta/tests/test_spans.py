"""The span recorder (Section 5 item 5)."""

import threading
from pathlib import Path

import pytest

from aorlta import spans
from aorlta.atomic import PartialFileError
from aorlta.spans import MAX_ATTR_STR, Span, SpanError, SpanRecorder, read_jsonl


class Clock:
    """A monotonic clock the test moves by hand, and a wall clock that can jump."""

    def __init__(self, mono: float = 100.0, wall: float = 1_793_600_000.0):
        self.mono, self.wall_value = mono, wall

    def monotonic(self) -> float:
        return self.mono

    def wall(self) -> float:
        return self.wall_value


def recorder(clock: Clock) -> SpanRecorder:
    return SpanRecorder(monotonic=clock.monotonic, wall=clock.wall)


def test_a_span_is_timed_on_the_monotonic_clock_and_reported_in_utc():
    clock = Clock()
    rec = recorder(clock)
    clock.mono += 2.0
    with rec.span("gen.turn0", 1, engine_id=2) as handle:
        clock.mono += 3.5
        clock.wall_value -= 3600  # a wall-clock step during the span changes nothing
        handle.set(chains=64)
    (span,) = rec.spans()
    assert span == Span("gen.turn0", 1, 1_793_600_002.0, 1_793_600_005.5, {"engine_id": 2, "chains": 64})
    assert span.duration == 3.5


def test_a_failing_block_still_ends_its_span_and_raises():
    clock = Clock()
    rec = recorder(clock)
    with pytest.raises(KeyError):
        with rec.span("merge", 3):
            clock.mono += 1.0
            raise KeyError("x")
    (span,) = rec.spans(3)
    assert span.attrs == {"error": "KeyError"} and span.duration == 1.0
    assert rec.open_spans() == []


def test_overlapping_spans_from_threads():
    clock = Clock()
    rec = recorder(clock)
    handles = [rec.start("san.asan", 2, batch=i) for i in range(8)]
    assert len(rec.open_spans(2)) == 8
    with pytest.raises(SpanError, match="still has open spans"):
        rec.close_iteration(2)
    threads = [threading.Thread(target=h.end) for h in handles]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(s.attrs["batch"] for s in rec.close_iteration(2)) == list(range(8))


def test_a_span_ends_once():
    rec = recorder(Clock())
    handle = rec.start("reload", 1)
    handle.end()
    with pytest.raises(SpanError, match="already ended"):
        handle.end()
    with pytest.raises(SpanError, match="already closed"):
        handle.set(x=1)


def test_set_racing_an_end_on_another_thread_is_kept_or_refused(monkeypatch):
    """The other thread ends the span while set() is under way: the attribute is in the span or set() raises; it is
    never accepted and then dropped."""
    rec = recorder(Clock())
    handle = rec.start("gen.turn0", 1)
    in_set, ended = threading.Event(), threading.Event()
    real = spans.validate_attrs

    def validate_attrs(attrs):
        if "late" in attrs:
            in_set.set()
            ended.wait(5)
        return real(attrs)

    monkeypatch.setattr(spans, "validate_attrs", validate_attrs)
    refused = []

    def set_late():
        try:
            handle.set(late=1)
        except SpanError as exc:
            refused.append(exc)

    setter = threading.Thread(target=set_late)
    setter.start()
    assert in_set.wait(5)
    span = handle.end()
    ended.set()
    setter.join(10)
    assert ("late" in span.attrs) != bool(refused)


def test_an_end_arriving_while_set_updates_waits_for_it():
    rec = recorder(Clock())
    handle = rec.start("gen.turn0", 1)
    in_update, ended = threading.Event(), threading.Event()

    class SlowAttrs(dict):
        def update(self, *args, **kwargs):
            in_update.set()
            ended.wait(1)  # without the recorder's lock, the end below completes meanwhile
            super().update(*args, **kwargs)

    handle._attrs = SlowAttrs(handle._attrs)
    setter = threading.Thread(target=handle.set, kwargs={"late": 1})
    setter.start()
    assert in_update.wait(5)
    span = handle.end()
    ended.set()
    setter.join(10)
    assert span.attrs == {"late": 1}


def test_external_spans():
    rec = recorder(Clock())
    span = rec.record("ref.native", 4, 10.0, 12.5, worker_id="sim-3")
    assert span.duration == 2.5 and rec.spans(4) == [span]
    with pytest.raises(SpanError, match="ends before it starts"):
        rec.record("ref.native", 4, 12.0, 11.0)
    with pytest.raises(SpanError):
        rec.record("ref.native", 4, float("nan"), 11.0)


@pytest.mark.parametrize(
    "attrs, message",
    [
        ({"prompt": "x" * (MAX_ATTR_STR + 1)}, "not text"),
        ({"nested": {"a": 1}}, "scalars"),
        ({"nested": [[1]]}, "scalars"),
        ({"gap": float("inf")}, "non-finite"),
        ({"obj": object()}, "scalars"),
    ],
)
def test_attrs_carry_ids_and_numbers_only(attrs, message):
    rec = recorder(Clock())
    with pytest.raises(SpanError, match=message):
        rec.start("record", 1, **attrs)


@pytest.mark.parametrize("name, iteration", [("", 1), ("has space", 1), ("gen.turn0", -1), ("gen.turn0", 1.0), (3, 1)])
def test_names_and_iterations_are_checked(name, iteration):
    with pytest.raises(SpanError):
        recorder(Clock()).start(name, iteration)


def test_iteration_zero_is_run_setup():
    rec = recorder(Clock())
    rec.start("engine.start", 0).end()
    assert [s.iteration for s in rec.spans(0)] == [0]


def test_jsonl_round_trip_is_atomic_and_canonical(tmp_path: Path):
    clock = Clock()
    rec = recorder(clock)
    for i, name in enumerate(["gen.turn0", "gate.parse", "san.asan"]):
        with rec.span(name, 1, i=i, legs=["a", "b"], ok=True, frac=0.25):
            clock.mono += 0.5
    path = rec.write_jsonl(tmp_path / "spans-1.jsonl", iteration=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["spans-1.jsonl"]
    assert read_jsonl(path) == rec.spans(1)
    first = path.read_text().splitlines()[0]
    assert first == ('{"attrs":{"frac":0.25,"i":0,"legs":["a","b"],"ok":true},"iteration":1,"name":"gen.turn0",'
                     '"t_end":1793600000.5,"t_start":1793600000.0}')


def test_readers_never_open_partial_files(tmp_path: Path):
    p = tmp_path / "spans.jsonl.partial"
    p.write_text("")
    with pytest.raises(PartialFileError):
        read_jsonl(p)


def test_a_corrupt_span_file_is_refused_with_its_line(tmp_path: Path):
    p = tmp_path / "spans.jsonl"
    good = '{"attrs":{},"iteration":1,"name":"a","t_end":2.0,"t_start":1.0}'
    p.write_text(good + "\n" + good.replace('"t_end":2.0', '"t_end":0.5') + "\n")
    with pytest.raises(SpanError, match=":2:"):
        read_jsonl(p)
    p.write_text(good.replace('"attrs":{},', "") + "\n")
    with pytest.raises(SpanError, match=":1:"):
        read_jsonl(p)
