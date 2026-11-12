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


def test_consan_on_a_kernel_without_sites_passes_only_as_not_applicable(tmp_path, monkeypatch):
    """tiny_vecadd has no LDS, barrier, atomic or fence site: ConSan's NOT_CHECKED is expected, another unsettled
    result (here an adapter error) is not."""
    smoke = _smoke()
    from aorlta.reward.adapters.base import map_aorta_check
    from aorlta.schemas import LegResult

    status = SimpleNamespace(value="not_checked")  # aorta's ConSan result for a kernel without sites
    _, not_applicable = map_aorta_check(SimpleNamespace(state=status, verdict=status, findings=(), reason=(
        "consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object")))

    def server(reason):
        class ConSan:
            legs = ["san.consan"]

            def __init__(self, sock, name, worker_id):
                pass

            def controls(self):
                return []

            def run(self, job):
                state, why = ("unsettled", reason) if "tiny_vecadd" in job.job_id else ("clean", None)
                return LegResult(leg="san.consan", state=state, reason=why, findings=[], repeats=1, attempts=1,
                                 worker_id=None, bracket=None, cache_hit=False, t_queued=0.0, t_start=0.0, t_end=0.0)

            def health(self):
                return {}

            def close(self):
                pass

        return ConSan

    monkeypatch.setattr(smoke, "load_config", lambda path: SimpleNamespace(reward=SimpleNamespace(remote=None)))
    monkeypatch.setattr(smoke, "job_limit", lambda cfg: 5.0)
    monkeypatch.setattr(smoke, "servers", lambda remote: [("standin.aorta_consan", 0, tmp_path / "0.sock")])
    src = tmp_path / "fixture.hip"
    src.write_text("__global__ void k() {}\n")
    monkeypatch.setattr(smoke, "sources", lambda out: {"tiny_vecadd": src, "lds_reduce": src})
    for reason, rc in ((not_applicable, 0), ("adapter_error", 1), ("harness_build_failed", 1), ("check_timeout", 1)):
        monkeypatch.setattr(smoke, "RemoteAdapter", server(reason))
        assert smoke.main("run.yaml", str(tmp_path / reason)) == rc, reason
