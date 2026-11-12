"""C7: the eval hook (design Section 6.7). The interface is a proposal, not yet used by runs.

Files, not a network API, so it works across team boundaries:

- `EvalRequest` is written atomically to `<inbox>/<request_id>.json`; the partner job writes an `EvalResult` to
  `<outbox>/<request_id>.json` (atomically too: `.partial` names are never read).
- The checkpoint a request names is `eval/checkpoints/<export_id>/`: `merged/` (the merged bf16 Hugging Face
  directory: the base's untouched tensors plus W_eff, exactly the model that was sampled and trained), `adapter/`
  (the PEFT export, A and B in fp32, with a warning that applying it unmerged gives a numerically different model),
  and `manifest.json` (sha256 of every file). The request carries the manifest file's sha256.

Cadence: a `validation` request every `orchestrator.eval_every` iterations and a `heldout` request
at the final iteration; none when `eval_every` is null. Requests are asynchronous and training never waits for a
result: the export runs on a background thread after the iteration commits, and the loop only waits for it before
the next publication (whose cleanup removes the slot the export reads). At most `max_checkpoints` (3) eval
checkpoints exist at once; one is deleted when every request naming it has a result. A due request that would make
a fourth is skipped and logged, never queued: skipping keeps disk bounded with an eval that is down.

Every event (requested, skipped, export_failed, result, ignored, deleted) goes to `eval/ledger.jsonl`, so results
join to iterations by `policy_version`, and the hook's state is rebuilt from it after a resume. A result for an
unknown request id, or one that does not validate, is ignored and logged once. The eval being down never stops
training.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..atomic import PARTIAL_SUFFIX, atomic_write_json, read_text
from ..config import ConfigError, plain, strict_decode
from ..episodes.writer import AppendLedger
from ..ids import IdError, canonical_json, policy_version, sha256_file, sha256_hex, sha256_json, validate_run_id

log = logging.getLogger(__name__)

FORMATS = ("hf_merged_bf16", "peft_adapter")
SETS = ("validation", "heldout")
MODES = ("full", "evidence_only")
STATUSES = ("done", "failed", "refused")
SOURCES = ("standin",)
MANIFEST_SCHEMA = "aorlta.eval_manifest/1"
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,200}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?Z")
MAX_CHECKPOINTS = 3
ADAPTER_WARNING = ("AORLTA trains on the merged bf16 weights (W_eff) that the engine serves. Applying this adapter "
                   "unmerged (W + (alpha/r) B A in higher precision) gives a numerically different model from the one "
                   "that was sampled and trained; use merged/ to evaluate the trained policy (Section 6.5).")


class EvalSchemaError(ValueError):
    pass


def utc_now(wall: Callable[[], float] = time.time) -> str:
    t = _dt.datetime.fromtimestamp(wall(), _dt.UTC)
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --- the files ---------------------------------------------------------------------------------------------------


@dataclass
class EvalRequest:  # written atomically to <inbox>/<request_id>.json
    request_id: str
    run_id: str
    iteration: int
    policy_version: str
    checkpoint: dict  # {"path", "format": "hf_merged_bf16" | "peft_adapter", "manifest_sha256", "base_sha256"}
    eval: dict  # {"set": "validation" | "heldout", "modes": ["full", "evidence_only"], "samples_per_item": 4}
    requested_at: str
    source: str | None = None  # optional: the data the checkpoint was trained on ("standin")


@dataclass
class EvalResult:  # written by the partner job to <outbox>/<request_id>.json
    request_id: str
    status: str  # "done" | "failed" | "refused"
    aggregates: dict  # per-sample verified repairs (full, evidence-only), best-of-4, worst-of-4, pass-count
    #                   histogram, paired diffs vs the base adapter
    eval_version: str
    completed_at: str


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise EvalSchemaError(message)


def _request_id(value: Any, where: str) -> None:
    _check(isinstance(value, str) and REQUEST_ID_RE.fullmatch(value) is not None and not value.endswith(PARTIAL_SUFFIX),
           f"{where}: request_id must match {REQUEST_ID_RE.pattern} and not end in {PARTIAL_SUFFIX}, got {value!r:.80}")


def validate_request(r: EvalRequest) -> None:
    _request_id(r.request_id, "request")
    where = f"request {r.request_id}"
    try:
        validate_run_id(r.run_id)
        _check(r.iteration >= 1, f"{where}: iteration must be >= 1, got {r.iteration}")
        _check(r.policy_version == policy_version(r.run_id, r.iteration),
               f"{where}: policy_version {r.policy_version!r} is not the weights after iteration {r.iteration}")
    except IdError as exc:
        raise EvalSchemaError(f"{where}: {exc}") from None
    c = r.checkpoint
    _check(set(c) == {"path", "format", "manifest_sha256", "base_sha256"},
           f"{where}: checkpoint keys must be path, format, manifest_sha256, base_sha256; got {sorted(c)}")
    _check(isinstance(c["path"], str) and Path(c["path"]).is_absolute(), f"{where}: checkpoint.path must be absolute")
    _check(c["format"] in FORMATS, f"{where}: checkpoint.format must be one of {FORMATS}, got {c['format']!r}")
    for k in ("manifest_sha256", "base_sha256"):
        _check(isinstance(c[k], str) and SHA256_RE.fullmatch(c[k]) is not None, f"{where}: checkpoint.{k} is not a sha256")
    e = r.eval
    _check(set(e) == {"set", "modes", "samples_per_item"},
           f"{where}: eval keys must be set, modes, samples_per_item; got {sorted(e)}")
    _check(e["set"] in SETS, f"{where}: eval.set must be one of {SETS}, got {e['set']!r}")
    modes = e["modes"]
    _check(isinstance(modes, list) and bool(modes) and len(set(modes)) == len(modes) and all(m in MODES for m in modes),
           f"{where}: eval.modes must be a non-empty list of distinct {MODES}, got {modes!r}")
    n = e["samples_per_item"]
    _check(isinstance(n, int) and not isinstance(n, bool) and n >= 1, f"{where}: eval.samples_per_item must be >= 1")
    _check(UTC_RE.fullmatch(r.requested_at) is not None, f"{where}: requested_at must be an ISO-8601 UTC time ending Z")
    _check(r.source is None or r.source in SOURCES, f"{where}: source must be one of {SOURCES} or absent, got {r.source!r}")


def validate_result(r: EvalResult) -> None:
    _request_id(r.request_id, "result")
    where = f"result {r.request_id}"
    _check(r.status in STATUSES, f"{where}: status must be one of {STATUSES}, got {r.status!r}")
    _check(bool(r.eval_version), f"{where}: eval_version must be non-empty")
    _check(UTC_RE.fullmatch(r.completed_at) is not None, f"{where}: completed_at must be an ISO-8601 UTC time ending Z")


def _decode(cls: type, data: Any) -> Any:
    try:
        return strict_decode(cls, data, cls.__name__)
    except ConfigError as exc:
        raise EvalSchemaError(str(exc)) from None


def request_from_dict(data: Any) -> EvalRequest:
    r = _decode(EvalRequest, data)
    validate_request(r)
    return r


def result_from_dict(data: Any) -> EvalResult:
    r = _decode(EvalResult, data)
    validate_result(r)
    return r


def write_request(path: Path, r: EvalRequest) -> Path:
    validate_request(r)
    return atomic_write_json(path, plain(r))


def read_request(path: Path) -> EvalRequest:
    return request_from_dict(json.loads(read_text(path)))


def write_result(path: Path, r: EvalResult) -> Path:
    validate_result(r)
    return atomic_write_json(path, plain(r))


def read_result(path: Path) -> EvalResult:
    return result_from_dict(json.loads(read_text(path)))


def files_manifest(root: Path) -> dict[str, dict]:
    return {p.relative_to(root).as_posix(): {"sha256": sha256_file(p), "size": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file()}


def verify_export(export_dir: Path, manifest_sha256: str) -> list[str]:
    """Problems with an eval checkpoint against its request's manifest hash (empty: it verifies)."""
    mpath = export_dir / "manifest.json"
    if not mpath.is_file():
        return ["no manifest.json"]
    if sha256_file(mpath) != manifest_sha256:
        return ["manifest.json does not match the request's manifest_sha256"]
    manifest = json.loads(read_text(mpath))
    on_disk = {k: v for k, v in files_manifest(export_dir).items() if k != "manifest.json"}
    return [] if on_disk == manifest.get("files") else ["files differ from the manifest"]


# --- the hook ------------------------------------------------------------------------------------------------------


class EvalHook:
    def __init__(self, eval_dir: Path, *, run_id: str, every: int | None, final_iteration: int, exporter: Any,
                 base_sha256: str, max_checkpoints: int = MAX_CHECKPOINTS, inbox: Path | None = None,
                 outbox: Path | None = None, modes: Sequence[str] = MODES, samples_per_item: int = 4,
                 spans: Any = None, wall: Callable[[], float] = time.time, background: bool = True,
                 point: Callable[[str, int], None] = lambda name, iteration: None, source: str | None = None):
        if every is not None and every < 1:
            raise ValueError("every must be >= 1 or None")
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be >= 1")
        if source is not None and source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES} or None, got {source!r}")
        self.dir = Path(eval_dir)
        self.run_id, self.every, self.final_iteration = run_id, every, final_iteration
        self.source = source
        self.exporter, self.base_sha256 = exporter, base_sha256
        self.max_checkpoints = max_checkpoints
        self.inbox = Path(inbox) if inbox else self.dir / "inbox"
        self.outbox = Path(outbox) if outbox else self.dir / "outbox"
        self.checkpoints = self.dir / "checkpoints"
        self.modes, self.samples_per_item = list(modes), samples_per_item
        self.spans, self.wall, self.background, self.point = spans, wall, background, point
        self.ledger = AppendLedger(self.dir / "ledger.jsonl")
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._inflight = 0
        self.errors: list[str] = []
        self._load()

    # state, rebuilt from the ledger

    def _load(self) -> None:
        self.requests: dict[str, dict] = {}  # request_id -> {iteration, export_id, policy_version, set}
        self.resolved: dict[str, str] = {}  # request_id -> status
        self.handled: set[tuple[int, str]] = set()  # (iteration, set) requested, skipped or failed
        self.ignored: set[str] = set()
        self.events: list[dict] = []
        for e in self.ledger.read():
            self._apply(e)

    def _apply(self, e: dict) -> None:
        self.events.append(e)
        ev = e.get("event")
        if ev == "requested":
            self.requests[e["request_id"]] = {k: e[k] for k in ("iteration", "export_id", "policy_version", "set")}
        if ev in ("requested", "skipped", "export_failed"):
            self.handled.add((e["iteration"], e["set"]))
        elif ev == "result":
            self.resolved[e["request_id"]] = e["status"]
        elif ev == "ignored":
            self.ignored.add(e["request_id"])

    def _record(self, **e: Any) -> None:
        e.setdefault("t", self.wall())
        with self._lock:
            self.ledger.append([e])
            self._apply(e)

    # cadence

    def due_sets(self, n: int) -> list[str]:
        if self.every is None:
            return []
        sets = []
        if n % self.every == 0:
            sets.append("validation")
        if n == self.final_iteration:
            sets.append("heldout")
        return sets

    def export_id(self, n: int) -> str:
        return f"{self.run_id}.it{n:06d}"

    def _dirs(self) -> None:
        # Created on first use, not at construction: the run directory must hold nothing before run_plan.json.
        for d in (self.inbox, self.outbox, self.checkpoints):
            d.mkdir(parents=True, exist_ok=True)

    def _exports_held(self) -> int:
        self._dirs()
        on_disk = [p for p in self.checkpoints.iterdir() if p.is_dir() and not p.name.endswith(PARTIAL_SUFFIX)]
        return len(on_disk) + self._inflight

    # after an iteration commits

    def after_commit(self, n: int) -> list[str]:
        """Start the export for the requests due at iteration `n`; returns the sets it will request."""
        sets = [s for s in self.due_sets(n) if (n, s) not in self.handled]
        if not sets:
            return []
        version = policy_version(self.run_id, n)
        with self._lock:
            held = self._exports_held() - (1 if (self.checkpoints / self.export_id(n)).is_dir() else 0)
            if held >= self.max_checkpoints:
                skip = True
            else:
                skip = False
                self._inflight += 1
        if skip:
            for s in sets:
                self._record(event="skipped", iteration=n, set=s, policy_version=version, reason="checkpoint_cap",
                             held=held)
            log.warning("eval: iteration %d's %s request skipped: %d eval checkpoints already held", n, "+".join(sets),
                        held)
            return []
        if self.background:
            self.wait_exports()
            self._thread = threading.Thread(target=self._export, args=(n, sets), name=f"eval-export-{n}", daemon=True)
            self._thread.start()
        else:
            self._export(n, sets)
        return sets

    def _export(self, n: int, sets: list[str]) -> None:
        version = policy_version(self.run_id, n)
        eid = self.export_id(n)
        final = self.checkpoints / eid
        tmp = self.checkpoints / (eid + PARTIAL_SUFFIX)
        span = self.spans.start("eval.request", n, export_id=eid, sets=sets) if self.spans is not None else None
        t_export = self.wall()
        try:
            info: dict = {}
            reuse = final.is_dir() and (final / "manifest.json").is_file() and not verify_export(
                final, sha256_file(final / "manifest.json"))
            if reuse:  # exported before a crash that came before its requests: the partner may already read it
                manifest_sha = sha256_file(final / "manifest.json")
                files = json.loads(read_text(final / "manifest.json"))["files"]
            else:
                if final.exists():
                    shutil.rmtree(final)
                if tmp.exists():
                    shutil.rmtree(tmp)
                (tmp / "merged").mkdir(parents=True)
                (tmp / "adapter").mkdir()
                self.point("eval.export_started", n)
                info = self.exporter.export(n, tmp / "merged", tmp / "adapter") or {}
                atomic_write_json(tmp / "adapter" / "aorlta_warning.json",
                                  {"warning": ADAPTER_WARNING, "policy_version": version, "use_instead": "merged/"})
                files = files_manifest(tmp)
                manifest = {"schema": MANIFEST_SCHEMA, "export_id": eid, "iteration": n, "policy_version": version,
                            "files": files, "files_sha256": sha256_json(files)}
                atomic_write_json(tmp / "manifest.json", manifest, canonical=True)
                manifest_sha = sha256_file(tmp / "manifest.json")
                os.replace(tmp, final)
                self.point("eval.export_done", n)
            for s in sets:
                rid = f"{eid}.{s}"
                req = EvalRequest(
                    request_id=rid, run_id=self.run_id, iteration=n, policy_version=version,
                    checkpoint={"path": str(final / "merged"), "format": "hf_merged_bf16",
                                "manifest_sha256": manifest_sha, "base_sha256": self.base_sha256},
                    eval={"set": s, "modes": list(self.modes), "samples_per_item": self.samples_per_item},
                    requested_at=utc_now(self.wall), source=self.source,
                )
                write_request(self.inbox / f"{rid}.json", req)
                self._record(event="requested", iteration=n, set=s, request_id=rid, export_id=eid,
                             policy_version=version, manifest_sha256=manifest_sha, files=len(files),
                             tensors=info.get("tensors"), reused=reuse, t_export_start=t_export,
                             t_export_end=self.wall())
                self.point("eval.request_written", n)
            if span is not None:
                span.end(files=len(files))
        except Exception as exc:  # noqa: BLE001 - the eval being unavailable never stops training
            shutil.rmtree(tmp, ignore_errors=True)
            for s in sets:
                if (n, s) not in self.handled:
                    self._record(event="export_failed", iteration=n, set=s, policy_version=version,
                                 reason=type(exc).__name__)
            self.errors.append(f"iteration {n}: {type(exc).__name__}")
            log.warning("eval: export for iteration %d failed (%s); training continues", n, type(exc).__name__)
            if span is not None:
                span.end(error=type(exc).__name__)
        finally:
            with self._lock:
                self._inflight -= 1

    def wait_exports(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)
            if t.is_alive():
                raise TimeoutError("the eval export did not finish")
            self._thread = None

    # results

    def poll(self, current_iteration: int = 0) -> list[dict]:
        """Read the outbox; record new results, delete checkpoints whose requests all have results. Never raises
        for a bad or unknown result: it is logged once and ignored."""
        self._dirs()
        seen = []
        for path in sorted(self.outbox.glob("*.json")):
            rid = path.name[: -len(".json")]
            if rid in self.resolved or rid in self.ignored:
                continue
            try:
                res = read_result(path)
                if res.request_id != rid:
                    raise EvalSchemaError(f"result file {path.name} holds request {res.request_id!r}")
            except (EvalSchemaError, ValueError, OSError) as exc:
                self._record(event="ignored", iteration=current_iteration, request_id=rid, reason="invalid",
                             detail=type(exc).__name__)
                log.warning("eval: ignored an invalid result file %s (%s)", path.name, type(exc).__name__)
                continue
            req = self.requests.get(rid)
            if req is None:
                self._record(event="ignored", iteration=current_iteration, request_id=rid, reason="unknown_request")
                log.warning("eval: ignored a result for unknown request %s", rid)
                continue
            self._record(event="result", iteration=req["iteration"], request_id=rid, export_id=req["export_id"],
                         policy_version=req["policy_version"], set=req["set"], status=res.status,
                         eval_version=res.eval_version, completed_at=res.completed_at,
                         aggregates_sha256=sha256_hex(canonical_json(res.aggregates)))
            seen.append({"request_id": rid, "status": res.status, "policy_version": req["policy_version"]})
            self._cleanup(req["export_id"])
        self.point("eval.polled", current_iteration)
        return seen

    def _cleanup(self, export_id: str) -> None:
        mine = [rid for rid, r in self.requests.items() if r["export_id"] == export_id]
        if mine and all(rid in self.resolved for rid in mine):
            path = self.checkpoints / export_id
            if path.exists():
                doomed = self.checkpoints / (export_id + ".deleting" + PARTIAL_SUFFIX)
                os.replace(path, doomed)
                shutil.rmtree(doomed)
                n = self.requests[mine[0]]["iteration"]
                self._record(event="deleted", iteration=n, export_id=export_id)

    # resume

    def reconcile(self, last_committed: int, can_export: Callable[[int], bool]) -> list[str]:
        """At resume: drop half-written exports, cut events of uncommitted iterations, and issue what a crash between
        a commit and its request lost. Only the weights the engines hold can be exported (`can_export`), so a due
        request for an older iteration is recorded as skipped."""
        self._dirs()
        for p in list(self.checkpoints.iterdir()):
            if p.name.endswith(PARTIAL_SUFFIX):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        self.ledger.truncate_after(last_committed)
        self._load()
        for p in sorted(self.inbox.glob("*.json")):
            rid = p.name[: -len(".json")]
            if rid not in self.requests and not any(rid == f"{self.export_id(n)}.{s}"
                                                    for n in range(1, last_committed + 1) for s in SETS):
                p.unlink()  # a request for an iteration that was not committed
        exports = {r["export_id"] for r in self.requests.values()}
        committed = {self.export_id(n) for n in range(1, last_committed + 1)}
        for p in list(self.checkpoints.iterdir()):
            if p.is_dir() and p.name not in exports and p.name not in committed:
                shutil.rmtree(p)
        issued = []
        for n in range(1, last_committed + 1):
            missing = [s for s in self.due_sets(n) if (n, s) not in self.handled]
            if not missing:
                continue
            if can_export(n):
                issued += self.after_commit(n)
            else:
                for s in missing:
                    self._record(event="skipped", iteration=n, set=s, policy_version=policy_version(self.run_id, n),
                                 reason="weights_not_held_at_resume")
        return issued

    def iteration_events(self, n: int) -> list[dict]:
        return [e for e in self.events if e.get("iteration") == n]

    def status(self) -> dict:
        return {"requests": len(self.requests), "resolved": len(self.resolved),
                "outstanding": sorted(set(self.requests) - set(self.resolved)),
                "checkpoints_held": self._exports_held(), "ignored": sorted(self.ignored), "errors": list(self.errors)}


# --- a fake partner eval (tests and stand-in runs) ------------------------------------------------------------------------


class FakeEval:
    """Plays the partner job: reads requests from the inbox, checks the checkpoint against its manifest, writes a
    result to the outbox. Aggregates are counts only."""

    def __init__(self, inbox: Path, outbox: Path, *, status: str = "done", eval_version: str = "fake-eval/1",
                 wall: Callable[[], float] = time.time):
        self.inbox, self.outbox = Path(inbox), Path(outbox)
        self.status, self.eval_version, self.wall = status, eval_version, wall

    def pending(self) -> list[Path]:
        return [p for p in sorted(self.inbox.glob("*.json")) if not (self.outbox / p.name).exists()]

    def process(self, limit: int | None = None) -> list[str]:
        done = []
        for path in self.pending()[:limit]:
            req = read_request(path)
            problems = verify_export(Path(req.checkpoint["path"]).parent, req.checkpoint["manifest_sha256"])
            status = self.status if not problems else "failed"
            res = EvalResult(request_id=req.request_id, status=status,
                             aggregates={"verified": not problems, "samples_per_item": req.eval["samples_per_item"],
                                         "modes": len(req.eval["modes"])},
                             eval_version=self.eval_version, completed_at=utc_now(self.wall))
            self.outbox.mkdir(parents=True, exist_ok=True)
            write_result(self.outbox / path.name, res)
            done.append(req.request_id)
        return done
