"""jobs/aorlta-checkers-smoke.py, the remote checkers' pre-flight, against checkers that never answer."""

import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

PKG = Path(__file__).resolve().parents[2]


def _smoke():
    spec = importlib.util.spec_from_file_location("checkers_smoke", PKG / "jobs" / "aorlta-checkers-smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_checker_that_never_answers_fails_the_pre_flight_at_the_limit(tmp_path, monkeypatch):
    smoke, closed = _smoke(), []

    class Hung:  # a server that keeps its connection open and never answers
        legs = ["san.asan"]

        def __init__(self, sock, name, worker_id):
            self.name, self.done = name, threading.Event()

        def controls(self):
            return []

        def run(self, job):
            self.done.wait(60)
            raise ConnectionError("closed")

        def health(self):
            raise AssertionError("a closed adapter is not asked")

        def close(self):
            closed.append(self.name)
            self.done.set()

    monkeypatch.setattr(smoke, "load_config", lambda path: SimpleNamespace(reward=SimpleNamespace(remote=None)))
    monkeypatch.setattr(smoke, "job_limit", lambda cfg: 0.5)
    monkeypatch.setattr(smoke, "servers", lambda remote: [("standin.aorta_asan", gpu, tmp_path / f"{gpu}.sock")
                                                          for gpu in (0, 1)])
    monkeypatch.setattr(smoke, "RemoteAdapter", Hung)
    src = tmp_path / "fixture.hip"
    src.write_text("int main() { return 0; }\n")
    monkeypatch.setattr(smoke, "sources", lambda out: {"tiny_vecadd": src, "asan_overflow": src})
    t0 = time.monotonic()
    assert smoke.main("run.yaml", str(tmp_path / "out")) == 1
    assert time.monotonic() - t0 < 20  # two limits, not the 60 s the servers would hold each job
    rows = json.loads((tmp_path / "out" / "smoke-results.json").read_text())["rows"]
    assert [(r["gpu"], r["reason"]) for r in rows] == [(0, "no answer in 0.5 s"), (1, "no answer in 0.5 s")]
    assert closed == ["standin.aorta_asan", "standin.aorta_asan"]


def test_a_job_gets_the_longest_of_the_services_pool_timeouts():
    from aorlta.config import RewardTimeouts

    cfg = SimpleNamespace(reward=SimpleNamespace(timeouts=RewardTimeouts(gate=1.0, asan_request=1200.0)))
    assert _smoke().job_limit(cfg) == 1200.0
