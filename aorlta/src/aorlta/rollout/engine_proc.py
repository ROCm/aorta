"""One rollout engine in its own spawned process, driven over a pipe (design Section 6.1,
Engine start). Copied from an earlier prototype, with its known fixes:

- each engine has its own table of waiters, so one engine's death fails only its own requests (the
  prototype shared one table and failed every engine's waiters);
- a request has a timeout; a timed-out request is reported as such and its late reply is dropped;
- the child sees one GPU from its first import, with the environment set under a lock, so engines
  can be started from several threads;
- `get_server_info()` and `flush_cache()` are never called (A9): the resolved ServerArgs are read from
  `engine.tokenizer_manager.server_args`;
- the same process body can host the fake engine (`factory="fake"`), so the pipe protocol, the death
  path and the timeout path are tested on CPU.

Messages (driver -> worker), each a tuple whose first item is the kind:
  ("gen", rid, input_ids, sampling_params, extra) -> ("gen", rid, slim result | {"error": str})
  ("reload", rid, path, flush_cache)              -> ("ctl", rid, {"ok", "reply", "value", "sec"})
  ("stop", rid)                                   -> ("ctl", rid, {"stopped": True})
`extra` holds further `Engine.generate` keyword arguments (for example `logprob_start_len` for the
landing check). AORLTA never sends `n`, so one request is one completion (Section 6.1, Sampling).
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

READY_KEYS = (
    "sampling_backend", "enable_output_logprobs", "enable_prefix_caching", "dtype", "disable_kvstore",
    "disable_autotune", "max_num_seqs", "max_model_len", "gpu_memory_utilization", "port", "seed", "numerics",
    "page_size", "max_total_tokens", "weight_version",
)
_ENV_LOCK = threading.Lock()
_GPU_ENV = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


class EngineStartError(RuntimeError):
    """An engine did not start, or started with settings AORLTA refuses."""


def slim(out: Any) -> dict:
    """The fields of an `Engine.generate` result that AORLTA reads. Log-probs are (logprob, token id,
    None) triples at 3b684074 (A1); entries that are not are reported, not guessed."""
    if not isinstance(out, dict):
        return {"error": f"generate returned {type(out).__name__}, not a dict", "kind": "engine"}
    meta = out.get("meta_info") or {}
    res: dict[str, Any] = {
        "output_ids": [int(t) for t in (out.get("output_ids") or [])],
        "lp_vals": None,
        "lp_ids": None,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "finish_reason": meta.get("finish_reason"),
        "weight_version": meta.get("weight_version"),
        "e2e_latency": meta.get("e2e_latency"),
    }
    for key, vals, ids in (("output_token_logprobs", "lp_vals", "lp_ids"), ("input_token_logprobs", "in_lp_vals", "in_lp_ids")):
        lp = meta.get(key)
        if lp is None:
            continue
        try:
            res[vals] = [None if x[0] is None else float(x[0]) for x in lp]
            res[ids] = [int(x[1]) for x in lp]
        except (TypeError, ValueError, IndexError) as exc:
            res[vals] = res[ids] = None
            res[f"{key}_malformed"] = f"{type(exc).__name__}: {exc}"[:300]
    return res


def reply_value(reply: Any) -> list | None:
    """A reload reply as plain values when it is a tuple of scalars (TokenSpeed's `(ok, message, ...)`), so that
    the driver parses the reply itself rather than its `repr`, which is cut at 500 characters."""
    if not isinstance(reply, tuple) or not all(v is None or isinstance(v, (bool, int, float, str)) for v in reply):
        return None
    return [v[:4000] if isinstance(v, str) else v for v in reply]


def _server_info(engine: Any) -> dict:
    sa = engine.tokenizer_manager.server_args
    info = dataclasses.asdict(sa) if dataclasses.is_dataclass(sa) else dict(vars(sa))
    return {k: info.get(k) for k in READY_KEYS if k in info}


def _make_engine(factory: str, engine_index: int, engine_kwargs: dict, fake: Any) -> Any:
    if factory == "tokenspeed":
        from tokenspeed.runtime.entrypoints.engine import Engine

        return Engine(**engine_kwargs)
    if factory == "fake":
        from .fake_engine import FakeEngine

        return FakeEngine(fake, engine_id=engine_index, **engine_kwargs)
    raise EngineStartError(f"unknown engine factory {factory!r}")


def worker_main(conn: Any, engine_index: int, gpu: str, engine_kwargs: dict, factory: str, fake: Any) -> None:
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    send_lock = threading.Lock()

    def send(msg: tuple) -> None:
        with send_lock:
            conn.send(msg)

    try:
        t0 = time.monotonic()
        engine = _make_engine(factory, engine_index, engine_kwargs, fake)
        send(("ready", engine_index, {"startup_sec": round(time.monotonic() - t0, 1), "pid": os.getpid(),
                                      "info": _server_info(engine)}))
    except BaseException:
        send(("ready", engine_index, {"error": traceback.format_exc()[-4000:]}))
        return

    workers = max(256, 2 * int(engine_kwargs.get("max_num_seqs") or 0))
    pool = ThreadPoolExecutor(max_workers=workers)

    def do_gen(rid: int, ids: list[int], sp: dict, extra: dict) -> None:
        try:
            out = engine.generate(input_ids=ids, sampling_params=sp, return_logprob=True, **extra)
            send(("gen", rid, slim(out)))
        except BaseException:
            send(("gen", rid, {"error": traceback.format_exc()[-4000:], "kind": "engine"}))

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        kind, rid = msg[0], msg[1]
        if kind == "gen":
            pool.submit(do_gen, rid, msg[2], msg[3], msg[4] if len(msg) > 4 else {})
            continue
        t0 = time.monotonic()
        try:
            if kind == "reload":
                reply = engine.update_weights_from_disk(msg[2], flush_cache=bool(msg[3]))
                res: dict[str, Any] = {"reply": repr(reply)[:500], "value": reply_value(reply),
                                       "ok": isinstance(reply, tuple) and len(reply) > 0 and reply[0] is True}
            elif kind == "stop":
                send(("ctl", rid, {"stopped": True}))
                break
            else:
                res = {"error": f"unknown kind {kind!r}"}
        except BaseException:
            res = {"error": traceback.format_exc()[-4000:]}
        res["sec"] = round(time.monotonic() - t0, 3)
        send(("ctl", rid, res))
    try:
        engine.shutdown()
    except BaseException:
        pass


class _Slot:
    __slots__ = ("event", "payload")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.payload: dict | None = None


class EngineProc:
    """Driver-side handle on one engine process. `generate` is thread-safe and blocking."""

    def __init__(
        self,
        index: int,
        gpu: str,
        engine_kwargs: dict,
        *,
        factory: str = "tokenspeed",
        fake: Any = None,
        ready_timeout_s: float = 1200.0,
    ):
        self.index, self.gpu = index, str(gpu)
        self.engine_kwargs = dict(engine_kwargs)
        self.factory, self.fake = factory, fake
        self.ready_timeout_s = ready_timeout_s
        self.ready: dict | None = None
        self._proc: Any = None
        self._conn: Any = None
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._pending: dict[int, _Slot] = {}
        self._counter = 0
        self._dead = True
        self.death: str | None = None

    @property
    def alive(self) -> bool:
        return not self._dead

    def start(self) -> dict:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        with _ENV_LOCK:
            saved = {k: os.environ.get(k) for k in _GPU_ENV}
            os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
            try:
                proc = ctx.Process(
                    target=worker_main, args=(child, self.index, self.gpu, self.engine_kwargs, self.factory, self.fake),
                    daemon=False,
                )
                proc.start()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        child.close()
        self._proc, self._conn = proc, parent
        if not parent.poll(self.ready_timeout_s):
            self._terminate()
            raise EngineStartError(f"engine {self.index} not ready after {self.ready_timeout_s:g} s")
        try:
            _, _, ready = parent.recv()
        except (EOFError, OSError) as exc:
            self._terminate()
            raise EngineStartError(f"engine {self.index} died during start-up ({exc})") from None
        if "error" in ready:
            self._terminate()
            raise EngineStartError(f"engine {self.index} failed to start:\n{ready['error']}")
        self.ready = ready
        self._dead, self.death = False, None
        threading.Thread(target=self._read, name=f"engine-{self.index}-reader", daemon=True).start()
        return ready

    def _read(self) -> None:
        conn = self._conn
        while True:
            try:
                _, rid, payload = conn.recv()
            except (EOFError, OSError):
                self._fail_all(f"engine {self.index} connection lost")
                return
            with self._lock:
                slot = self._pending.pop(rid, None)
            if slot is not None:  # None: the waiter timed out and left
                slot.payload = payload
                slot.event.set()

    def _fail_all(self, why: str) -> None:
        with self._lock:
            self._dead = True
            self.death = self.death or why
            waiting = list(self._pending.values())
            self._pending.clear()
        for slot in waiting:
            slot.payload = {"error": why, "kind": "dead"}
            slot.event.set()

    def _call(self, head: str, *rest: Any, timeout_s: float | None) -> dict:
        with self._lock:
            if self._dead:
                return {"error": f"engine {self.index} is dead ({self.death})", "kind": "dead"}
            self._counter += 1
            rid = self._counter
            slot = _Slot()
            self._pending[rid] = slot
        try:
            with self._send_lock:
                self._conn.send((head, rid, *rest))
        except (OSError, ValueError) as exc:
            with self._lock:
                self._pending.pop(rid, None)
            self._fail_all(f"send to engine {self.index} failed: {exc}")
            return {"error": f"send to engine {self.index} failed: {exc}", "kind": "dead"}
        if not slot.event.wait(timeout_s):
            with self._lock:
                self._pending.pop(rid, None)
            if slot.event.is_set():  # the reply landed while we were leaving
                return slot.payload or {"error": "empty reply", "kind": "engine"}
            return {"error": f"engine {self.index}: no reply within {timeout_s:g} s", "kind": "timeout"}
        return slot.payload or {"error": "empty reply", "kind": "engine"}

    def generate(self, ids: list[int], sp: dict, *, timeout_s: float, extra: dict | None = None) -> dict:
        return self._call("gen", list(ids), dict(sp), dict(extra or {}), timeout_s=timeout_s)

    def reload(self, path: str, *, flush_cache: bool = True, timeout_s: float = 600.0) -> dict:
        return self._call("reload", str(path), bool(flush_cache), timeout_s=timeout_s)

    def _terminate(self) -> None:
        proc = self._proc
        if proc is not None and proc.is_alive():
            proc.kill()
            proc.join(timeout=30)

    def kill(self, why: str = "killed by the driver") -> None:
        """End the process now: every waiter fails at once, as on a real death."""
        with self._lock:
            self.death = self.death or why
        self._terminate()
        self._fail_all(why)

    def stop(self, timeout_s: float = 60.0) -> None:
        if self.alive:
            self._call("stop", timeout_s=timeout_s)
        proc = self._proc
        if proc is not None:
            proc.join(timeout=timeout_s)
        self._terminate()
        self._fail_all(f"engine {self.index} stopped")


class InProcessEngine:
    """The EngineProc interface around an engine object in this process (the fake engine, in tests).
    Requests run on a thread pool, so the timeout path behaves as with a process."""

    def __init__(self, index: int, engine: Any, *, workers: int = 512):
        self.index, self.gpu = index, "inproc"
        self.engine = engine
        self.ready: dict | None = None
        self.death: str | None = None
        self._dead = True
        self._pool = ThreadPoolExecutor(max_workers=workers)

    @property
    def alive(self) -> bool:
        return not self._dead

    def start(self) -> dict:
        self.ready = {"startup_sec": 0.0, "pid": os.getpid(), "info": _server_info(self.engine)}
        self._dead, self.death = False, None
        if hasattr(self.engine, "restart"):
            self.engine.restart()
        return self.ready

    def generate(self, ids: list[int], sp: dict, *, timeout_s: float, extra: dict | None = None) -> dict:
        if self._dead:
            return {"error": f"engine {self.index} is dead ({self.death})", "kind": "dead"}

        def run() -> dict:
            try:
                return slim(self.engine.generate(input_ids=list(ids), sampling_params=dict(sp), return_logprob=True,
                                                 **dict(extra or {})))
            except Exception as exc:  # noqa: BLE001 - every engine failure becomes a result
                kind = "dead" if self._dead else "engine"
                return {"error": f"{type(exc).__name__}: {exc}"[:2000], "kind": kind}

        fut = self._pool.submit(run)
        try:
            return fut.result(timeout=timeout_s)
        except FutureTimeout:
            return {"error": f"engine {self.index}: no reply within {timeout_s:g} s", "kind": "timeout"}

    def reload(self, path: str, *, flush_cache: bool = True, timeout_s: float = 600.0) -> dict:
        """As the worker's reload, for an engine object that has `update_weights_from_disk` (WP3's tiny engine)."""
        if self._dead:
            return {"error": f"engine {self.index} is dead ({self.death})", "kind": "dead"}
        if not hasattr(self.engine, "update_weights_from_disk"):
            return {"error": "the in-process engine does not reload", "ok": False}
        t0 = time.monotonic()
        try:
            reply = self.engine.update_weights_from_disk(str(path), flush_cache=bool(flush_cache))
        except Exception as exc:  # noqa: BLE001 - a failed reload is reported, never raised
            return {"error": f"{type(exc).__name__}: {exc}"[:2000], "sec": round(time.monotonic() - t0, 3)}
        return {"reply": repr(reply)[:500], "value": reply_value(reply),
                "ok": isinstance(reply, tuple) and len(reply) > 0 and reply[0] is True,
                "sec": round(time.monotonic() - t0, 3)}

    def kill(self, why: str = "killed by the driver") -> None:
        self._dead, self.death = True, self.death or why
        if hasattr(self.engine, "kill"):
            self.engine.kill()

    def stop(self, timeout_s: float = 60.0) -> None:
        self._dead = True
