"""C5 step 4: reload replies, the generation barrier, retry and restart (design Section 6.5, A5, A7)."""

from __future__ import annotations

import pytest

from aorlta.sync import reload as R
from aorlta.sync import writer as W
from aorlta.sync.merge import TARGET_KEY_RE

OK = "Succeeded to update model weights from /dev/shm/aorlta/r/w1 (248 checkpoint tensors read)"


@pytest.mark.parametrize("reply, ok, n", [
    ((True, OK, 0), True, 248),
    ([True, OK, 0], True, 248),
    ((True, OK.replace("248", "247"), 0), False, 247),
    ((True, OK.replace("248", "738"), 0), False, 738),
    ((False, OK, 0), False, None),
    ((1, OK, 0), False, None),  # success must be exactly True
    (("True", OK, 0), False, None),
    ((True, "Succeeded to update model weights", 0), False, None),
    ((True, OK + " (12 checkpoint tensors read)", 0), False, None),  # two counts: ambiguous
    ((True,), False, None),
    (None, False, None),
    ({"ok": True}, False, None),
    ("(True, '... (248 checkpoint tensors read)', 0)", False, None),
])
def test_parse_reload_reply_fails_closed(reply, ok, n):
    got = R.parse_reload_reply(reply, 248)
    assert got.ok is ok
    assert got.tensors_read == n


class FakeEngine:
    def __init__(self, engine_id, replies):
        self.engine_id = engine_id
        self.replies = list(replies)
        self.calls = []

    def update_weights_from_disk(self, model_path, flush_cache=True):
        self.calls.append((model_path, flush_cache))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def slot(tiny_base, tmp_path):
    from safetensors import safe_open

    with safe_open(str(tiny_base / "model.safetensors"), framework="pt") as f:
        weights = {k: f.get_tensor(k) for k in f.keys() if TARGET_KEY_RE.fullmatch(k)}
    return W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=1).path


def _ok(slot, n=31):
    return (True, f"Succeeded to update model weights from {slot} ({n} checkpoint tensors read)", 0)


def test_reload_all_flushes_and_accepts_the_exact_count(slot):
    engines = [FakeEngine(i, [_ok(slot)]) for i in range(3)]
    results = R.reload_all(engines, slot, 31, in_flight=lambda: 0)
    assert [r.ok for r in results] == [True] * 3 and all(r.tensors_read == 31 for r in results)
    assert all(e.calls == [(str(slot), True)] for e in engines)


def test_a_reload_with_requests_in_flight_is_refused(slot):
    e = FakeEngine(0, [_ok(slot)])
    with pytest.raises(R.BarrierViolation):
        R.reload_all([e], slot, 31, in_flight=lambda: 2)
    assert e.calls == []


def test_retry_once_then_restart_then_stop(slot):
    bad = (True, f"Succeeded to update model weights from {slot} (30 checkpoint tensors read)", 0)
    e = FakeEngine(0, [bad, _ok(slot)])
    (r,) = R.reload_all([e], slot, 31, in_flight=lambda: 0)
    assert r.ok and r.attempts == 2 and not r.restarted

    restarted = []
    e = FakeEngine(0, [bad, RuntimeError("engine died"), _ok(slot)])
    (r,) = R.reload_all([e], slot, 31, in_flight=lambda: 0, restart=restarted.append)
    assert r.ok and r.attempts == 3 and r.restarted and restarted == [e]

    e = FakeEngine(0, [bad, bad, bad])
    with pytest.raises(R.ReloadError, match="30 checkpoint tensors read"):
        R.reload_all([e], slot, 31, in_flight=lambda: 0, restart=lambda eng: None)

    e = FakeEngine(0, [(False, "refused: a request pins the cache", 0)] * 2)
    with pytest.raises(R.ReloadError, match="refused"):
        R.reload_all([e], slot, 31, in_flight=lambda: 0)


def test_one_failing_engine_stops_the_reload(slot):
    engines = [FakeEngine(0, [_ok(slot)]), FakeEngine(1, [(False, "no", 0)] * 2)]
    with pytest.raises(R.ReloadError, match="engine 1"):
        R.reload_all(engines, slot, 31, in_flight=lambda: 0)


def test_unfinished_slots_are_never_reloaded(slot, tmp_path):
    part = tmp_path / "r" / "w0.partial"
    part.mkdir(parents=True)
    with pytest.raises(R.ReloadError):
        R.reload_all([FakeEngine(0, [_ok(part)])], part, 31, in_flight=lambda: 0)
    empty = tmp_path / "r" / "w0"
    empty.mkdir()
    with pytest.raises(R.ReloadError):
        R.reload_all([FakeEngine(0, [_ok(empty)])], empty, 31, in_flight=lambda: 0)


def test_the_tiny_engine_reports_what_tokenspeed_reports(slot, engine_factory):
    e = engine_factory()
    (r,) = R.reload_all([e], slot, 31, in_flight=lambda: e.busy)
    assert r.ok and r.tensors_read == 31


def test_version_book():
    book = R.VersionBook()
    book.record(0, 3, "/x/w1")
    book.record(1, 3, "/x/w1")
    assert book.all_at(3) and book.version(1) == 3
    book.record(1, 2, "/x/w0")
    assert not book.all_at(3)
    assert book.as_dict()[1] == {"version": 2, "slot": "/x/w0"}
