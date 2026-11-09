"""Check adapters served from another process (design Section 6.2, Plug-in adapters; Section 4, the
development layout's GPUs 4-5 for the stand-in ConSan/ASan legs).

aorta's ConSan hook needs glibc >= 2.38 and device ASan the ROCm 7.0.2 ASan runtime; WP5 ran both only in
rocm/dev-ubuntu-24.04:7.0.2, while the orchestrator runs in the trainer image (ROCm 7.2.4).
So `jobs/aorlta-run.sbatch` starts one `serve` process per remote adapter in that image, each seeing only its own
GPU, and the reward service's workers reach it through `RemoteAdapter`, a CheckAdapter whose calls are requests on
a Unix socket. Each direction carries one JSON object per line:

    {"op": "open", "adapter": NAME, "worker_id": WID}  -> {"ok": true, "name", "version", "legs", "resource",
                                                          "max_packages", "inband_controls", "controls": [...]}
    {"op": "run", "job": LegJob}                        -> {"ok": true, "result": LegResult}
    {"op": "run_batch", "jobs": [LegJob, ...]}          -> {"ok": true, "results": [LegResult, ...]}
    {"op": "health"}                                    -> {"ok": true, "health": {...}}
    anything the server cannot do                       -> {"ok": false, "error": "..."}

One connection is one worker's adapter instance: the server builds it from the profile at "open" and closes it when
the connection ends, which also interrupts the job it is running (the watchdog's restart closes the connection).
`health` uses its own short connection, so a heartbeat never waits behind a running job.

Like every adapter, the client never raises from `run`: a lost connection, a malformed reply, or a result for
another job or leg is unsettled(adapter_error), never clean. A server that cannot be reached when a worker starts
is an error at construction (`RemoteUnavailable`), so a run never starts on checkers that are not there.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import select
import socket
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..schemas import ControlSpec, LegJob, LegResult, SchemaError, from_dict, to_dict
from .adapters.base import unsettled

log = logging.getLogger(__name__)

PROTOCOL = "aorlta.remote/1"
HEALTH_TIMEOUT_S = 5.0
OPEN_TIMEOUT_S = 60.0
MAX_LINE = 64 << 20  # a batch of 16 results is far below this


class RemoteUnavailable(RuntimeError):
    """The checker server could not be reached, or refused to open the adapter."""


def socket_path(socket_dir: str | Path, adapter: str, gpu: int | None = None) -> Path:
    """The server's socket: `<adapter>.sock` when one GPU serves the adapter, `<adapter>.gpu<N>.sock` for each of
    several."""
    return Path(socket_dir) / (f"{adapter}.sock" if gpu is None else f"{adapter}.gpu{gpu}.sock")


def servers(remote: Any) -> list[tuple[str, int, Path]]:
    """Every checker server a `reward.remote` section asks for, as (adapter, GPU, socket): what
    `jobs/aorlta-run.sbatch` starts and what the workers connect to."""
    out = []
    for name in sorted(remote.adapters):
        gpus = remote.gpus(name)
        for gpu in gpus:
            out.append((name, gpu, socket_path(remote.socket_dir, name, gpu if len(gpus) > 1 else None)))
    return out


class RemoteFactory:
    """The reward service's factory for one remote adapter served on several GPUs: each call (one worker) opens a
    RemoteAdapter on the server with the fewest adapters still open, the lowest index on a tie. A worker the watchdog
    restarts has its adapter closed first, so its replacement takes the server it left. `replicas` tells the service
    how many servers there are."""

    def __init__(self, paths: Sequence[str | Path], adapter: str, *, connect_timeout_s: float = 120.0,
                 clock: Callable[[], float] = time.time):
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError(f"{adapter}: no server to connect to")
        self.adapter, self.connect_timeout_s, self.clock = adapter, connect_timeout_s, clock
        self.replicas = len(self.paths)
        self._open: list[list[RemoteAdapter]] = [[] for _ in self.paths]
        self._lock = threading.Lock()

    def load(self) -> list[int]:
        """Adapters still open on each server."""
        with self._lock:
            self._open = [[a for a in opened if not a._closed] for opened in self._open]
            return [len(opened) for opened in self._open]

    def __call__(self, worker_id: str) -> RemoteAdapter:
        load = self.load()
        i = min(range(len(load)), key=lambda k: (load[k], k))
        adapter = RemoteAdapter(self.paths[i], self.adapter, worker_id, connect_timeout_s=self.connect_timeout_s,
                                clock=self.clock)
        with self._lock:
            self._open[i].append(adapter)
        return adapter


# --- the wire --------------------------------------------------------------------------------------------------


class _Conn:
    """A line-delimited JSON connection over a stream socket."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = b""

    def send(self, obj: dict) -> None:
        self.sock.sendall(json.dumps(obj, sort_keys=True).encode("utf-8") + b"\n")

    def recv(self) -> dict | None:
        """The next object, or None at end of stream."""
        while b"\n" not in self._buf:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                return None
            self._buf += chunk
            if len(self._buf) > MAX_LINE:
                raise ValueError("a line longer than the protocol allows")
        line, self._buf = self._buf.split(b"\n", 1)
        obj = json.loads(line.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("a message must be a JSON object")
        return obj

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def _connect(path: Path, timeout_s: float) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    s.connect(str(path))
    return s


# --- the client --------------------------------------------------------------------------------------------------


class RemoteAdapter:
    """A CheckAdapter (and BatchCheckAdapter, when the served adapter is one) whose work runs in the server at
    `path`. Built once per worker by the reward service's factory."""

    def __init__(self, path: str | Path, adapter: str, worker_id: str, *, connect_timeout_s: float = 120.0,
                 clock: Callable[[], float] = time.time):
        self.path, self.adapter, self.worker_id, self._clock = Path(path), adapter, worker_id, clock
        self._lock = threading.Lock()
        self._conn: _Conn | None = None
        self._closed = False
        deadline = time.monotonic() + connect_timeout_s
        last = "never tried"
        while True:  # the server may still be starting when the first worker is made
            try:
                sock = _connect(self.path, OPEN_TIMEOUT_S)
                break
            except OSError as exc:
                last = f"{type(exc).__name__}: {exc}"
                if time.monotonic() >= deadline:
                    raise RemoteUnavailable(f"{adapter}: cannot connect to {self.path} ({last})") from None
                time.sleep(0.5)
        conn = _Conn(sock)
        try:
            conn.send({"op": "open", "adapter": adapter, "worker_id": worker_id, "protocol": PROTOCOL})
            info = conn.recv()
        except (OSError, ValueError) as exc:
            conn.close()
            raise RemoteUnavailable(f"{adapter}: open failed on {self.path} ({type(exc).__name__}: {exc})") from None
        if not info or not info.get("ok") or info.get("protocol") != PROTOCOL:
            conn.close()
            raise RemoteUnavailable(f"{adapter}: the server refused to open it: {(info or {}).get('error')!r}")
        sock.settimeout(None)  # a job runs as long as it runs; the reward service's watchdog bounds it
        self._conn = conn
        self.name = str(info["name"])
        self.version = str(info["version"])
        self.legs = tuple(info["legs"])
        self.resource = str(info["resource"])
        self._controls = [from_dict(ControlSpec, c) for c in info.get("controls") or []]
        if info.get("max_packages") is not None:  # a batched endpoint (the ASan pool)
            self.max_packages = int(info["max_packages"])
            self.inband_controls = bool(info.get("inband_controls", True))

    def _call(self, request: dict) -> dict:
        with self._lock:
            conn = self._conn
            if conn is None or self._closed:
                raise ConnectionError("closed")
            conn.send(request)
            reply = conn.recv()
        if reply is None:
            raise ConnectionError("the server closed the connection")
        if not reply.get("ok"):
            raise RuntimeError(str(reply.get("error"))[:300])
        return reply

    def run(self, job: LegJob) -> LegResult:
        t0 = self._clock()
        try:
            reply = self._call({"op": "run", "job": to_dict(job)})
            return self._result(job, reply.get("result"), t0)
        except Exception as exc:  # noqa: BLE001 - the protocol forbids raising; nothing here is clean
            log.warning("remote %s: run %s: %s", self.adapter, job.job_id, type(exc).__name__)
            return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())

    def run_batch(self, jobs: Sequence[LegJob]) -> list[LegResult]:
        jobs = list(jobs)
        t0 = self._clock()
        try:
            reply = self._call({"op": "run_batch", "jobs": [to_dict(j) for j in jobs]})
            results = reply.get("results")
            if not isinstance(results, list) or len(results) != len(jobs):
                raise ValueError("a batch reply with the wrong number of results")
            return [self._result(j, r, t0) for j, r in zip(jobs, results)]
        except Exception as exc:  # noqa: BLE001
            log.warning("remote %s: batch of %d: %s", self.adapter, len(jobs), type(exc).__name__)
            t1 = self._clock()
            return [unsettled(j, "adapter_error", t_start=t0, t_end=t1) for j in jobs]

    def _result(self, job: LegJob, data: Any, t0: float) -> LegResult:
        try:
            result = from_dict(LegResult, data)
        except SchemaError:
            return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())
        if result.leg != job.leg:
            return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())  # not this job's result
        return result

    def controls(self) -> list[ControlSpec]:
        return list(self._controls)

    def health(self) -> dict:
        """A prompt answer on its own connection: responsive only if the server answers within HEALTH_TIMEOUT_S."""
        if self._closed:
            return {"name": self.adapter, "responsive": False}
        try:
            conn = _Conn(_connect(self.path, HEALTH_TIMEOUT_S))
            try:
                conn.send({"op": "health"})
                reply = conn.recv()
            finally:
                conn.close()
            health = dict((reply or {}).get("health") or {})
            health["responsive"] = bool(reply and reply.get("ok") and health.get("responsive", True))
            return health
        except (OSError, ValueError):
            return {"name": self.adapter, "responsive": False}

    def close(self) -> None:
        self._closed = True
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()  # the server closes this worker's adapter, which stops its subprocess


# --- the server --------------------------------------------------------------------------------------------------


def _adapter_info(adapter: Any) -> dict:
    out = {"ok": True, "protocol": PROTOCOL, "name": adapter.name, "version": adapter.version,
           "legs": list(adapter.legs), "resource": adapter.resource,
           "controls": [to_dict(c) for c in adapter.controls()]}
    if callable(getattr(adapter, "run_batch", None)) and hasattr(adapter, "max_packages"):
        out["max_packages"] = int(adapter.max_packages)
        out["inband_controls"] = bool(getattr(adapter, "inband_controls", True))
    return out


class Server:
    """Serves one adapter name at `path`; `factory(worker_id)` builds a fresh instance per connection."""

    def __init__(self, path: str | Path, adapter: str, factory: Callable[[str], Any]):
        self.path, self.adapter, self.factory = Path(path), adapter, factory
        self.connections = 0
        self.jobs = 0
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def start(self) -> "Server":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.unlink()  # a socket left by an earlier server on this node
        except FileNotFoundError:
            pass
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(self.path))
        s.listen(128)
        self._sock = s
        threading.Thread(target=self._accept, name=f"remote-{self.adapter}-accept", daemon=True).start()
        return self

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                sock, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(sock,), daemon=True).start()

    def _serve(self, sock: socket.socket) -> None:
        conn = _Conn(sock)
        adapter = None
        try:
            while True:
                request = conn.recv()
                if request is None:
                    return
                op = request.get("op")
                if op == "health":
                    conn.send({"ok": True, "health": {"name": self.adapter, "responsive": True, "pid": os.getpid(),
                                                      "connections": self.connections, "jobs": self.jobs}})
                elif op == "open":
                    if adapter is not None or request.get("adapter") != self.adapter:
                        conn.send({"ok": False, "error": f"this server serves {self.adapter!r}, once per connection"})
                        return
                    if request.get("protocol") != PROTOCOL:
                        conn.send({"ok": False, "error": f"protocol {request.get('protocol')!r}, not {PROTOCOL}"})
                        return
                    try:
                        adapter = self.factory(str(request.get("worker_id")))
                        info = _adapter_info(adapter)
                    except Exception as exc:  # noqa: BLE001 - the client raises RemoteUnavailable on this
                        conn.send({"ok": False, "error": f"cannot open {self.adapter}: {type(exc).__name__}: {exc}"[:300]})
                        return
                    with self._lock:
                        self.connections += 1
                    conn.send(info)
                elif op in ("run", "run_batch") and adapter is not None:
                    reply = self._work(conn, adapter, op, request)
                    if reply is None:
                        return  # the client left mid-job
                    conn.send(reply)
                else:
                    conn.send({"ok": False, "error": f"unexpected op {op!r}"})
        except (OSError, ValueError) as exc:
            log.info("remote %s: connection ended (%s)", self.adapter, type(exc).__name__)
        finally:
            if adapter is not None:
                close = getattr(adapter, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:  # noqa: BLE001
                        log.warning("remote %s: adapter close failed", self.adapter)
            conn.close()

    def _work(self, conn: _Conn, adapter: Any, op: str, request: dict) -> dict | None:
        """Run the job on a thread while watching the socket: a client that hangs up (a worker the watchdog
        restarted) closes the adapter, which stops its subprocess. Returns None if the client left."""
        out: dict[str, Any] = {}

        def work() -> None:
            try:
                if op == "run":
                    job = from_dict(LegJob, request.get("job"))
                    out["reply"] = {"ok": True, "result": to_dict(adapter.run(job))}
                else:
                    jobs = [from_dict(LegJob, j) for j in request.get("jobs") or []]
                    out["reply"] = {"ok": True, "results": [to_dict(r) for r in adapter.run_batch(jobs)]}
                with self._lock:
                    self.jobs += 1
            except Exception as exc:  # noqa: BLE001 - reported to the client, which makes the leg unsettled
                out["reply"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while t.is_alive():
            readable, _, _ = select.select([conn.sock], [], [], 0.5)
            if readable:
                try:
                    peek = conn.sock.recv(1, socket.MSG_PEEK)
                except OSError:
                    peek = b""
                if not peek:
                    close = getattr(adapter, "close", None)
                    if close is not None:
                        close()
                    t.join()
                    return None
                time.sleep(0.05)  # a request before the reply would break the protocol; let the join see it
        return out["reply"]

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# --- the command line ---------------------------------------------------------------------------------------


def standin_factory(config_path: str, adapter: str) -> Callable[[str], Any]:
    """The run config's profile factory for one stand-in adapter, built as the reward service builds it in-process."""
    import dataclasses

    from ..config import load_config
    from .clock import Clock
    from .factories import standin_factories
    from .profiles import load_profile

    cfg = load_config(config_path)
    profile = load_profile(cfg.reward.profile)
    factories = standin_factories(profile, Clock(), timeouts=dataclasses.asdict(cfg.reward.timeouts),
                                  check_s=cfg.reward.gpu_check_s)
    if adapter not in factories:
        raise SystemExit(f"profile {profile.name!r} has no stand-in adapter {adapter!r}")
    return factories[adapter]


def node_health(out_dir: str) -> int:
    """aorta's ASan node-health canary on the GPUs this process sees: 0 healthy, 2 failed, 3 not verified."""
    from .adapters import aorta_import

    src = os.environ.get("AORLTA_AORTA_SRC", "/opt/aorta/src")
    asan, why = aorta_import.load("asan", src)
    asan_run, why2 = aorta_import.load("asan_run", src)
    if asan is None or asan_run is None:
        print(json.dumps({"status": "not_verified", "reason": f"aorta import failed: {why or why2}"}))
        return 3
    runtime = asan.resolve_asan_runtime()
    if runtime is None:
        print(json.dumps({"status": "not_verified", "reason": "no ASan runtime (AORTA_ASAN_ROCM_LIB)"}))
        return 3
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    health = asan_run.check_node_health(Path(out_dir), runtime)
    status = health.status.value
    print(json.dumps({"status": status, "reason": health.reason, "gpus": health.gpus, "seconds": health.seconds,
                      "evidence": health.evidence_dir}))
    return {"healthy": 0, "failed": 2}.get(status, 3)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aorlta.reward.remote")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve one stand-in adapter of the run config's profile")
    s.add_argument("--config", required=True)
    s.add_argument("--adapter", required=True)
    s.add_argument("--socket", required=True)
    h = sub.add_parser("node-health", help="aorta's ASan node-health canary")
    h.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.cmd == "node-health":
        return node_health(args.out)
    server = Server(args.socket, args.adapter, standin_factory(args.config, args.adapter)).start()
    print(f"serving {args.adapter} at {args.socket} (pid {os.getpid()})", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
