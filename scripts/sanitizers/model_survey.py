#!/usr/bin/env python3
"""Run waitcheck and ConSan over open-source model kernels for the workload survey.

The nightly ``sanitizers-model-survey`` job runs this once per model inventory in
``recipes/sanitizers/models/``. For each model it:

1. launches every (kernel, scenario) pair the inventory keeps, once, with the
   asset's ``launch.py --mode run`` and a private ``TRITON_CACHE_DIR``. Triton
   specialises on runtime values, so the object it compiles for that launch is
   the object the kernel actually dispatches; it is kept as ``dispatched.hsaco``;
2. scans all of those objects with waitcheck in one ``kernel_list`` sweep,
   written to ``<out>/waitcheck-<model>/``;
3. re-runs one scenario per kernel under ConSan against the same cache, so the
   hook instruments the bytes waitcheck scanned. Kernels are taken in a nightly
   rotation. The per-case reports are merged into ``<out>/consan-<model>/``; a
   case that wrote no report, or whose dispatched object is no longer the one
   waitcheck scanned, is merged as an ``error`` for its kernel.

One time budget covers every launch, waitcheck sweep, and ConSan case. A
launch or ConSan case starts only while it could still finish at its own
timeout. A waitcheck sweep is one process and stops at its own deadline,
before it would spend this model's ConSan wave or the time held for a later
model; on that deadline the sweep's process group is stopped and the timeout
is recorded. Each later model keeps one launch wave, one waitcheck sweep, and
one ConSan wave, so an earlier model running to its timeouts cannot leave the
later one with no ConSan report.

Each output directory also gets ``model_survey.json``: what was launched, what
failed to launch, which ConSan cases ran, were excluded, or did not fit tonight.

The kernel sources come from the ``aorta-model-kernels`` release asset pinned
below by URL and SHA-256. Without ``--asset-dir`` the script downloads it,
refuses it unless the digest matches, and extracts it under ``--work``. Bump
``ASSET_VERSION``, ``ASSET_URL`` and ``ASSET_SHA256`` together with each
inventory's ``asset_version``.

Infrastructure faults (an asset that does not match the inventory, a pair count
that drifted, no object compiled at all) exit non-zero. A sanitizer verdict
never does: the survey is observed-only.

    model_survey.py run --inventory recipes/sanitizers/models/gpt-oss.yaml --out results
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ASSET_SCHEMA = "aorta.model_kernel_asset/1"
ASSET_VERSION = "1"
ASSET_ROOT = "aorta-model-kernels"
ASSET_URL = (
    "https://github.com/ROCm/aorta/releases/download/"
    f"model-kernels-v{ASSET_VERSION}/aorta-model-kernels-v{ASSET_VERSION}.tar.gz"
)
ASSET_SHA256 = "130d63c0636bcbc4ea8321be6dd45df82c951aa1c56ab35002ea4c043b5ea709"
INVENTORY_SCHEMA_VERSION = 1
TARGET = "gfx950"
WARP_SIZE = 64
WARM_TIMEOUT_SECONDS = 300
# One waitcheck sweep, same bound as a ConSan case. Reserved for each later
# model, and the longest a sweep is allowed to run.
SCAN_TIMEOUT_SECONDS = 900
# Returned by ``sweep`` when its deadline fired. Not a process exit status.
SWEEP_TIMED_OUT = -1
# Stop a sweep this early so cleanup still leaves the ConSan hold intact.
_DEADLINE_SLACK_SECONDS = 1.0
_STEM_RE = re.compile(r"[^A-Za-z0-9._-]")
_INVENTORY_KEYS = {
    "schema_version",
    "model",
    "asset_version",
    "specs",
    "roles",
    "expected_pairs",
    "consan",
}
_CONSAN_KEYS = {"timeout_seconds", "exclude"}
_VERDICT_RANK = {"pass": 0, "not_checked": 1, "warn": 2, "error": 3, "fail": 4}
_FAILED_STATES = {"error", "timed_out"}
# ConSan names the reader process in a parse error; one cause across kernels
# should count as one cause.
_READER_RE = re.compile(r"reader \d+ ")


class SurveyError(RuntimeError):
    """An infrastructure fault: the survey cannot produce a trustworthy row."""


@dataclass(frozen=True)
class Inventory:
    model: str
    asset_version: str
    specs: tuple[str, ...]
    roles: frozenset[str]
    expected_pairs: int
    consan_timeout: int
    exclude: dict[str, str]
    path: Path


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SurveyError(f"{name} must be a positive integer, got {value!r}")
    return value


def _str_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise SurveyError(f"{name} must be a non-empty list of strings")
    return tuple(value)


def load_inventory(path: Path) -> Inventory:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise SurveyError(f"{path}: expected a mapping")
    unknown = set(doc) - _INVENTORY_KEYS
    if unknown:
        raise SurveyError(f"{path}: unknown keys {sorted(unknown)}")
    if doc.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise SurveyError(f"{path}: schema_version must be {INVENTORY_SCHEMA_VERSION}")
    model = doc.get("model")
    if not isinstance(model, str) or _STEM_RE.search(model) or not model:
        raise SurveyError(f"{path}: model must be a plain name")
    if model != path.stem:
        raise SurveyError(f"{path}: model {model!r} must match the file name")
    asset_version = doc.get("asset_version")
    if not isinstance(asset_version, str) or not asset_version:
        raise SurveyError(f"{path}: asset_version must be a non-empty string")
    consan = doc.get("consan")
    if not isinstance(consan, dict) or set(consan) - _CONSAN_KEYS:
        raise SurveyError(f"{path}: consan must be a mapping of {sorted(_CONSAN_KEYS)}")
    exclude = consan.get("exclude") or {}
    if not isinstance(exclude, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in exclude.items()
    ):
        raise SurveyError(f"{path}: consan.exclude must map kernel names to reasons")
    return Inventory(
        model=model,
        asset_version=asset_version,
        specs=_str_list(doc.get("specs"), f"{path}: specs"),
        roles=frozenset(_str_list(doc.get("roles"), f"{path}: roles")),
        expected_pairs=_positive_int(doc.get("expected_pairs"), f"{path}: expected_pairs"),
        consan_timeout=_positive_int(consan.get("timeout_seconds"), f"{path}: consan.timeout_seconds"),
        exclude=dict(exclude),
        path=path,
    )


@dataclass(frozen=True)
class Pair:
    spec: str
    kernel: str
    scenario: str

    @property
    def stem(self) -> str:
        return _STEM_RE.sub("_", f"{self.kernel}__{self.scenario}")


def model_pairs(inv: Inventory, model_dir: Path) -> list[Pair]:
    """Every kept (kernel, scenario), in spec order; a kernel's first is its ConSan case."""
    pairs: list[Pair] = []
    for spec in inv.specs:
        entries = json.loads((model_dir / "roofline" / "specs" / spec).read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise SurveyError(f"{inv.model}/{spec}: expected a JSON list")
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("role") not in inv.roles:
                continue
            scenarios = entry.get("scenarios")
            if not entry.get("kernel") or not isinstance(scenarios, dict):
                continue
            pairs.extend(Pair(spec, str(entry["kernel"]), str(s)) for s in scenarios)
    if len(pairs) != inv.expected_pairs:
        raise SurveyError(
            f"{inv.model}: the specs list {len(pairs)} pairs, inventory expects "
            f"{inv.expected_pairs}; update expected_pairs in the same change as the asset"
        )
    return pairs


def extract_asset(tarball: Path, dest: Path) -> Path:
    """Extract the asset tarball, refusing any member outside ``ASSET_ROOT``."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with tarfile.open(tarball, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if (
                not member.isfile() and not member.isdir()
                or not parts
                or parts[0] != ASSET_ROOT
                or Path(member.name).is_absolute()
                or ".." in parts
            ):
                raise SurveyError(f"asset member {member.name!r} is not a plain file under {ASSET_ROOT}/")
        for member in members:
            if member.isfile():
                target = dest / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise SurveyError(f"asset member {member.name!r} has no data")
                target.write_bytes(source.read())
    return dest / ASSET_ROOT


def fetch_asset(work: Path, *, url: str = ASSET_URL, sha256: str = ASSET_SHA256) -> Path:
    """Download the pinned asset into ``work`` and return its extracted root."""
    tarball = work / f"aorta-model-kernels-v{ASSET_VERSION}.tar.gz"
    tarball.parent.mkdir(parents=True, exist_ok=True)
    if not tarball.is_file() or hashlib.sha256(tarball.read_bytes()).hexdigest() != sha256:
        print(f"[{time.strftime('%H:%M:%S')}] downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - pinned https URL
            tarball.write_bytes(response.read())
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != sha256:
        raise SurveyError(f"{url} has SHA-256 {digest}, expected {sha256}; refusing it")
    return extract_asset(tarball, work / "asset")


def visible_gpus() -> list[str]:
    """Device ids this process can use: ``HIP_VISIBLE_DEVICES`` if set, else every device."""
    env = os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def check_asset(asset_dir: Path, inv: Inventory) -> Path:
    manifest_path = asset_dir / "MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SurveyError(f"{manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != ASSET_SCHEMA:
        raise SurveyError(f"{manifest_path}: not an {ASSET_SCHEMA} manifest")
    if str(manifest.get("version")) != inv.asset_version:
        raise SurveyError(
            f"{inv.model}: asset is v{manifest.get('version')}, inventory wants v{inv.asset_version}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SurveyError(f"{manifest_path}: no file digests")
    for rel, digest in files.items():
        if rel == "MANIFEST.json":
            continue
        path = asset_dir / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SurveyError(f"asset file {rel} is missing or does not match MANIFEST.json")
    model_dir = asset_dir / inv.model
    if not (model_dir / "roofline" / "harness" / "launch.py").is_file():
        raise SurveyError(f"asset has no harness for {inv.model}")
    return model_dir


def write_shim(path: Path, *, model_dir: Path, pair: Pair, cache: Path, run_out: Path, python: str) -> None:
    """The one command both the warm-up launch and the ConSan dispatch run."""
    q = shlex.quote
    roofline = model_dir / "roofline"
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"export PYTHONPATH={q(str(model_dir / 'pkg_src'))}\n"
        f"export TRITON_CACHE_DIR={q(str(cache))}\n"
        f"cd {q(str(roofline))}\n"
        f"exec {q(python)} harness/launch.py {q(str(roofline / 'specs' / pair.spec))} "
        f"--kernel {q(pair.kernel)} --scenario {q(pair.scenario)} --mode run "
        f"--arch {TARGET} --warp-size {WARP_SIZE} --iters 1 --out {q(str(run_out))} "
        f"--flat-dir {q(str(model_dir / 'kernels'))}\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@dataclass
class Warmed:
    pair: Pair
    case_dir: Path
    ok: bool = False
    reason: str | None = None
    symbol: str | None = None
    code_object: Path | None = None
    sha256: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.pair.spec,
            "kernel": self.pair.kernel,
            "scenario": self.pair.scenario,
            "ok": self.ok,
            "reason": self.reason,
            "symbol": self.symbol,
            "code_object_sha256": self.sha256,
            "seconds": round(self.seconds, 1),
        }


def cached_object(cache: Path) -> tuple[Path, str]:
    """The single compiled object in a warm cache, and the symbol Triton gave it."""
    objects = sorted(cache.rglob("*.hsaco"))
    if len(objects) != 1:
        raise SurveyError(f"expected one compiled object, found {len(objects)}")
    meta = objects[0].with_suffix(".json")
    try:
        name = json.loads(meta.read_text(encoding="utf-8")).get("name")
    except (OSError, ValueError) as exc:
        raise SurveyError(f"no Triton metadata beside {objects[0].name}: {exc}") from exc
    if not isinstance(name, str) or not name:
        raise SurveyError(f"{meta.name} names no kernel")
    return objects[0], name


def warm_one(
    item: Warmed, *, model_dir: Path, python: str, gpu: str, budget: Budget, reserve: float = 0.0
) -> Warmed:
    if not budget.admit(WARM_TIMEOUT_SECONDS, reserve=reserve):
        item.reason = "not launched: the survey time budget is spent"
        return item
    case = item.case_dir
    shutil.rmtree(case, ignore_errors=True)
    case.mkdir(parents=True)
    shim = case / "launch.sh"
    write_shim(shim, model_dir=model_dir, pair=item.pair, cache=case / "triton-cache", run_out=case / "run", python=python)
    start = time.monotonic()
    proc = subprocess.Popen(
        [str(shim)],
        env={**os.environ, "HIP_VISIBLE_DEVICES": gpu},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=WARM_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _stop_process_group(proc)
        item.reason = f"launch timed out after {WARM_TIMEOUT_SECONDS}s"
        item.seconds = time.monotonic() - start
        return item
    except KeyboardInterrupt:
        _stop_process_group(proc)
        raise
    item.seconds = time.monotonic() - start
    (case / "launch.log").write_text(stdout + stderr, encoding="utf-8")
    if proc.returncode != 0:
        tail = (stderr.strip().splitlines() or ["no output"])[-1]
        item.reason = f"launch exited {proc.returncode}: {tail[:200]}"
        return item
    try:
        obj, symbol = cached_object(case / "triton-cache")
    except SurveyError as exc:
        item.reason = str(exc)
        return item
    kept = case / "dispatched.hsaco"
    shutil.copyfile(obj, kept)
    item.ok, item.symbol, item.code_object = True, symbol, kept
    item.sha256 = hashlib.sha256(kept.read_bytes()).hexdigest()
    return item


def on_gpus(gpus: Sequence[str], items: Sequence[object], work: Callable[[object, str], object]) -> list:
    """Run ``work(item, gpu)`` with at most one item per GPU at a time, keeping order."""
    free: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        free.put(gpu)

    def run(item: object) -> object:
        gpu = free.get()
        try:
            return work(item, gpu)
        finally:
            free.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        return list(pool.map(run, items))


def waitcheck_recipe(model: str, warmed: Sequence[Warmed]) -> str:
    # No entry_offset: with a digest, waitcheck then scans the whole code object
    # rather than one entry, so a non-inlined helper is analysed too.
    kernels = [
        {
            "name": item.symbol,
            "code_object": str(item.code_object),
            "code_object_sha256": item.sha256,
            "code_object_index": 0,
        }
        for item in warmed
        if item.ok
    ]
    recipe = {
        "schema_version": 1,
        "mode": "sanitizer",
        "ticket": f"MODEL-SURVEY-WAITCHECK-{model.upper()}",
        "sanitizer_plan": {
            "target": TARGET,
            "source": {"kind": "kernel_list", "kernels": kernels},
            "scope": {"kind": "kernel"},
            "selection": {"requirement": "top_dispatch_count", "top_n": len(kernels)},
            "sanitizers": ["waitcheck"],
            "policy": {"consan_policy": "strict", "on_missing_backend": "fail"},
            "output": {"report": "sanitizer_report.json"},
        },
    }
    return yaml.safe_dump(recipe, sort_keys=False)


def consan_recipe(model: str, item: Warmed, *, timeout: int) -> str:
    recipe = {
        "schema_version": 1,
        "mode": "sanitizer",
        "ticket": f"MODEL-SURVEY-CONSAN-{model.upper()}",
        "sanitizer_plan": {
            "target": TARGET,
            "source": {
                "kind": "kernel",
                "kernel": {
                    "name": item.symbol,
                    "code_object": str(item.code_object),
                    "code_object_sha256": item.sha256,
                    "code_object_index": 0,
                },
                "consan_command": str(item.case_dir / "launch.sh"),
                "consan_driver": "dispatch",
                "consan_log": True,
            },
            "scope": {"kind": "kernel"},
            "selection": {"requirement": "top_dispatch_count", "top_n": 1},
            "sanitizers": ["consan"],
            "policy": {
                "consan_policy": "strict",
                "on_missing_backend": "fail",
                "timeout_seconds": timeout,
            },
            "output": {"report": "sanitizer_report.json"},
        },
    }
    return yaml.safe_dump(recipe, sort_keys=False)


def rotation(items: Sequence[Warmed], night: dt.date) -> list[Warmed]:
    """The ConSan order for ``night``: a fixed order, started one kernel later each night."""
    ordered = sorted(items, key=lambda item: item.pair.kernel)
    if not ordered:
        return []
    start = night.toordinal() % len(ordered)
    return ordered[start:] + ordered[:start]


@dataclass
class Budget:
    """Admit a case only while it could still finish at its own timeout."""

    seconds: float
    clock: Callable[[], float] = time.monotonic
    _start: float = field(init=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._start = self.clock()

    def admit(self, case_seconds: float, *, reserve: float = 0.0) -> bool:
        """True when this case can finish and still leave ``reserve`` seconds.

        ``reserve`` is wall-clock held for models that have not run yet. Cases
        in one wave share the clock, so the hold is one case timeout, not the
        sum across GPUs.
        """
        with self._lock:
            return self.clock() - self._start + case_seconds + reserve <= self.seconds

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.seconds - (self.clock() - self._start))


def _stop_process_group(proc: subprocess.Popen[str]) -> None:
    """Kill ``proc`` and anything it spawned, then reap it.

    The nightly runner is Linux. A new session lets one signal stop the
    process and the hook children it spawned; this path has no Windows
    process-group equivalent.

    ``ProcessLookupError`` is the race where the group is already gone. Any
    other failure to signal the group is reported, and the direct child is
    stopped on its own.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        print(
            f"killpg({proc.pid}) failed: {exc}; stopping the process directly",
            file=sys.stderr,
        )
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1)


def _sweep_deadline(budget: Budget, *, reserve: float, keep: float) -> float:
    """Seconds a sweep may run without spending ``reserve`` or ``keep``.

    ``keep`` is this model's own ConSan wave. ``reserve`` is the hold for
    later models, including their waitcheck sweep.
    """
    slack = budget.remaining() - reserve - keep - _DEADLINE_SLACK_SECONDS
    return min(float(SCAN_TIMEOUT_SECONDS), slack)


def sweep(
    recipe: Path,
    out_dir: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> int:
    """Run one ``aorta sweep`` and stop its process group at ``timeout``.

    A timeout returns ``SWEEP_TIMED_OUT`` after the group is killed, and the
    sweep log says so. ``timeout`` of ``None`` waits without a deadline.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    with (out_dir / "sweep.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; from aorta.cli import main; sys.exit(main())",
             "sweep", "run", "--recipe", str(recipe), "--output-dir", str(out_dir)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
            start_new_session=True,
            text=True,
        )
        try:
            return proc.wait(timeout=timeout)
        except KeyboardInterrupt:
            _stop_process_group(proc)
            raise
        except subprocess.TimeoutExpired:
            _stop_process_group(proc)
            log.write(
                f"sweep timed out after {timeout:.1f}s; the sweep process group was stopped\n"
            )
            return SWEEP_TIMED_OUT


def error_case_report(item: Warmed, reason: str) -> dict:
    """A per-case ConSan report recording ``reason`` as an error for ``item``'s kernel.

    Used when a case cannot be read as a valid observation, so its kernel stays in
    the merged row as an error instead of silently leaving it.
    """
    from aorta.instrumentation.rocjitsu_sanitizers.models import SanitizerReport

    identity = {
        "name": item.symbol,
        "target": TARGET,
        "code_object": str(item.code_object),
        "code_object_sha256": item.sha256,
        "code_object_index": 0,
        "entry_offset": None,
    }
    check = {
        "sanitizer": "consan", "state": "error", "verdict": "error", "reason": reason,
        "returncode": None, "findings": [], "coverage": [], "backend": {},
        "kernel_results": [{
            "identity": identity, "state": "error", "verdict": "error", "findings": [],
            "reason": reason, "returncode": None, "diagnostics_truncated": False,
        }],
    }
    return SanitizerReport.from_dict({
        "schema": "aorta.sanitizer_report/0.1",
        "target": TARGET,
        "worklist": {
            "schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
            "top_n": 1, "kernel_count": 1,
            "kernels": [{"identity": identity, "total_time_ms": 0.0, "dispatch_count": 1,
                         "sources": ["kernel"]}],
        },
        "checks": [check],
        "overall_verdict": "error",
        "execution_status": "error",
    }).to_dict()


def _with_kernel_result(check: dict, identity: dict) -> dict:
    """A ConSan check, with a per-kernel result even when the hook wrote none."""
    if check.get("kernel_results"):
        return check
    return {
        **check,
        "kernel_results": [
            {
                "identity": identity,
                "state": check["state"],
                "verdict": check["verdict"],
                "findings": check.get("findings") or [],
                "reason": check.get("reason"),
                "returncode": check.get("returncode"),
                "diagnostics_truncated": check.get("diagnostics_truncated") is True,
            }
        ],
    }


def _merge_checks(sanitizer: str, checks: Sequence[dict]) -> dict:
    """One check from the same sanitizer's check in every case report."""
    verdict = max((c["verdict"] for c in checks), key=_VERDICT_RANK.__getitem__)
    failed = [c for c in checks if c["state"] in _FAILED_STATES]
    unchecked = [c for c in checks if c["state"] == "not_checked"]

    def tally(group: Sequence[dict], default: str) -> str:
        counts = collections.Counter(_READER_RE.sub("", c.get("reason") or default) for c in group)
        return ", ".join(f"{n}x {r}" for r, n in counts.most_common())

    reason = None
    if failed:
        state = "error"
        reason = f"{len(failed)} of {len(checks)} kernels did not complete ({tally(failed, 'error')})"
    elif verdict == "not_checked":
        state = "not_checked"
        reason = f"{len(unchecked)} of {len(checks)} kernels not checked ({tally(unchecked, 'not_checked')})"
    else:
        state = "ran"
    backend = checks[0].get("backend") or {}
    return {
        "sanitizer": sanitizer,
        "state": state,
        "verdict": verdict,
        "reason": reason,
        "returncode": None,
        "findings": [f for c in checks for f in c.get("findings") or []],
        "kernel_results": [kr for c in checks for kr in c.get("kernel_results") or []],
        "coverage": [cov for c in checks for cov in c.get("coverage") or []],
        "backend": {k: v for k, v in backend.items() if isinstance(v, str)},
        "diagnostics_truncated": any(c.get("diagnostics_truncated") for c in checks),
    }


def merge_consan(reports: Sequence[dict]) -> dict:
    """One ConSan report per model from its per-kernel reports.

    ConSan takes one code object per run, but the survey shows one row per model. Each
    case report carries a ``consan`` check and the hook's own waitcheck preflight;
    the merge keeps both, combining each across kernels: every kernel's own result
    is kept and the worst verdict wins. A check cannot be ``ran`` with a
    ``not_checked`` verdict, so when the worst ConSan outcome is a kernel with
    nothing to instrument, that check is ``not_checked`` with the count in its
    reason, and the report is ``partial`` beside the preflight that did run.
    """
    from aorta.instrumentation.rocjitsu_sanitizers.models import (
        CheckResult,
        KernelWorklist,
        SanitizerReport,
    )

    by_sanitizer: dict[str, list[dict]] = {}
    for report in reports:
        identity = report["worklist"]["kernels"][0]["identity"]
        for check in report["checks"]:
            if check["sanitizer"] == "consan":
                check = _with_kernel_result(check, identity)
            by_sanitizer.setdefault(check["sanitizer"], []).append(check)
    if "consan" not in by_sanitizer:
        raise SurveyError("no case report carries a consan check")
    names = ["consan"] + sorted(name for name in by_sanitizer if name != "consan")
    kernels = [k for report in reports for k in report["worklist"]["kernels"]]
    worklist = KernelWorklist.from_dict(
        {
            "schema": reports[0]["worklist"]["schema"],
            "requirement": reports[0]["worklist"]["requirement"],
            "top_n": len(kernels),
            "kernel_count": len(kernels),
            "kernels": kernels,
        }
    )
    merged = SanitizerReport(
        target=reports[0]["target"],
        worklist=worklist,
        checks=tuple(
            CheckResult.from_dict(_merge_checks(name, by_sanitizer[name])) for name in names
        ),
    )
    return merged.to_dict()


def _later_reserve(later: Sequence[Inventory]) -> float:
    """Seconds held so each later model can still launch, scan, and run ConSan.

    One launch wave, one waitcheck sweep, and one ConSan wave per later
    inventory. On the nightly's eight GPUs the launch and ConSan waves cover
    all of Qwen3; on fewer devices each is still one case, which is what
    writes the report ``--require`` checks for.
    """

    return float(
        sum(WARM_TIMEOUT_SECONDS + SCAN_TIMEOUT_SECONDS + inv.consan_timeout for inv in later)
    )


def run_model(
    inv: Inventory,
    *,
    asset_dir: Path,
    work: Path,
    out: Path,
    gpus: Sequence[str],
    budget: Budget,
    night: dt.date,
    python: str,
    reserve: float = 0.0,
) -> dict:
    model_dir = check_asset(asset_dir, inv)
    pairs = model_pairs(inv, model_dir)
    root = work / inv.model
    print(f"[{time.strftime('%H:%M:%S')}] {inv.model}: launching {len(pairs)} pairs on {len(gpus)} GPU(s)", flush=True)
    warmed = on_gpus(
        gpus,
        [Warmed(pair, root / "cases" / pair.stem) for pair in pairs],
        lambda item, gpu: warm_one(
            item, model_dir=model_dir, python=python, gpu=gpu, budget=budget, reserve=reserve
        ),
    )
    launched = [w for w in warmed if w.ok]
    if not launched:
        raise SurveyError(f"{inv.model}: no pair compiled; first failure: {warmed[0].reason}")
    print(f"[{time.strftime('%H:%M:%S')}] {inv.model}: {len(launched)}/{len(pairs)} compiled", flush=True)
    launch_record = [w.to_dict() for w in warmed]

    recipe_dir = root / "recipes"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    wc_recipe = recipe_dir / f"waitcheck-{inv.model}.yaml"
    wc_recipe.write_text(waitcheck_recipe(inv.model, warmed), encoding="utf-8")
    wc_out = out / f"waitcheck-{inv.model}"
    print(f"[{time.strftime('%H:%M:%S')}] {inv.model}: waitcheck over {len(launched)} objects", flush=True)
    # Keep this model's ConSan wave as well as the later-model hold. A scan
    # that cannot fit is recorded and skipped; it does not cancel ConSan.
    deadline = _sweep_deadline(budget, reserve=reserve, keep=float(inv.consan_timeout))
    if deadline <= 0:
        wc_rc = SWEEP_TIMED_OUT
        shutil.rmtree(wc_out, ignore_errors=True)
        wc_out.mkdir(parents=True)
        (wc_out / "sweep.log").write_text(
            "waitcheck not started: running it would spend the ConSan time held "
            f"for this model ({inv.consan_timeout}s) or a later one\n",
            encoding="utf-8",
        )
    else:
        wc_rc = sweep(wc_recipe, wc_out, timeout=deadline)
    (wc_out / "model_survey.json").write_text(
        json.dumps({"model": inv.model, "inventory": inv.path.name, "pairs": launch_record,
                    "sweep_returncode": wc_rc, "sweep_timed_out": wc_rc == SWEEP_TIMED_OUT,
                    "sweep_timeout_seconds": None if deadline <= 0 else round(deadline, 1)},
                   indent=2) + "\n",
        encoding="utf-8",
    )

    first_by_kernel: dict[str, Warmed] = {}
    for item in warmed:
        first_by_kernel.setdefault(item.pair.kernel, item)
    excluded = {k: r for k, r in inv.exclude.items() if k in first_by_kernel}
    not_launched = {k: w.reason for k, w in first_by_kernel.items() if not w.ok and k not in excluded}
    candidates = [w for k, w in first_by_kernel.items() if w.ok and k not in excluded]
    order = rotation(candidates, night)
    consan_root = root / "consan"
    skipped: list[str] = []
    lock = threading.Lock()

    def consan_one(item: Warmed, gpu: str) -> dict | None:
        if not budget.admit(inv.consan_timeout, reserve=reserve):
            with lock:
                skipped.append(item.pair.kernel)
            return None
        recipe = recipe_dir / f"consan-{item.pair.stem}.yaml"
        recipe.write_text(consan_recipe(inv.model, item, timeout=inv.consan_timeout), encoding="utf-8")
        case_out = consan_root / item.pair.stem
        before = item.sha256
        rc = sweep(
            recipe, case_out, env={"HIP_VISIBLE_DEVICES": gpu}, timeout=float(inv.consan_timeout)
        )
        try:
            obj, _ = cached_object(item.case_dir / "triton-cache")
            same = hashlib.sha256(obj.read_bytes()).hexdigest() == before
        except SurveyError:
            same = False
        report_path = case_out / "sanitizer_report.json"
        if rc == SWEEP_TIMED_OUT:
            report = error_case_report(
                item,
                f"consan_case_timed_out: aorta sweep exceeded {inv.consan_timeout}s "
                "and its process group was stopped",
            )
            same = False
        else:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report = error_case_report(
                    item, f"consan_case_produced_no_report: aorta sweep exited {rc}"
                )
            else:
                if not same:
                    # The row would compare ConSan on one object with waitcheck on
                    # another, which is the one thing this survey promises not to do.
                    report = error_case_report(
                        item,
                        "consan_object_mismatch: the object ConSan dispatched is not the "
                        "one waitcheck scanned",
                    )
        return {"kernel": item.pair.kernel, "scenario": item.pair.scenario, "returncode": rc,
                "report": report, "same_object_as_waitcheck": same}

    print(f"[{time.strftime('%H:%M:%S')}] {inv.model}: ConSan over up to {len(order)} kernels "
          f"({len(excluded)} excluded, {int(budget.remaining())}s of budget left)", flush=True)
    ran = [r for r in on_gpus(gpus, order, consan_one) if r is not None]
    cs_out = out / f"consan-{inv.model}"
    shutil.rmtree(cs_out, ignore_errors=True)
    cs_out.mkdir(parents=True)
    if ran:
        merged = merge_consan([r["report"] for r in ran])
        (cs_out / "sanitizer_report.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    for result in ran:
        src = consan_root / Pair("", result["kernel"], result["scenario"]).stem
        for log in sorted(src.rglob("*.log")):
            dest = cs_out / "cases" / src.name / log.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(log, dest)
    selection = {
        "model": inv.model,
        "inventory": inv.path.name,
        "night": night.isoformat(),
        "budget_seconds": int(budget.seconds),
        "case_timeout_seconds": inv.consan_timeout,
        "ran": [
            {k: r.get(k) for k in ("kernel", "scenario", "returncode", "same_object_as_waitcheck")}
            | {"verdict": r["report"].get("overall_verdict")}
            for r in ran
        ],
        "not_scheduled_tonight": sorted(skipped),
        "excluded": excluded,
        "not_launched": not_launched,
    }
    (cs_out / "model_survey.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    summary = {
        "model": inv.model,
        "pairs": len(pairs),
        "compiled": len(launched),
        "waitcheck_returncode": wc_rc,
        "waitcheck_timed_out": wc_rc == SWEEP_TIMED_OUT,
        "consan_ran": len(ran),
        "consan_candidates": len(order),
        "consan_not_scheduled": len(skipped),
        "consan_excluded": len(excluded),
    }
    print(f"[{time.strftime('%H:%M:%S')}] {inv.model}: {json.dumps(summary)}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="survey one or more models")
    run.add_argument("--inventory", type=Path, action="append", required=True)
    run.add_argument("--out", type=Path, required=True, help="survey informational results dir")
    run.add_argument("--work", type=Path, help="scratch dir (default: <out>/../model-survey-work)")
    run.add_argument("--asset-dir", type=Path,
                     help="an already-extracted aorta-model-kernels root (default: download the pinned asset)")
    run.add_argument("--gpus", default="auto",
                     help="comma-separated device ids, or auto for every visible device")
    run.add_argument("--budget-seconds", type=float, default=3600.0,
                     help="for every launch, waitcheck sweep, and ConSan case; a launch or "
                     "ConSan case starts only if it can still finish at its own timeout, "
                     "a waitcheck sweep stops before spending that hold, and each later "
                     "model keeps one launch wave, one waitcheck sweep, and one ConSan wave")
    run.add_argument("--night", type=dt.date.fromisoformat, default=dt.datetime.now(dt.timezone.utc).date())
    args = parser.parse_args(argv)

    # The launch shim changes directory before using any of these, so a relative
    # path would resolve against the asset rather than the caller.
    out = args.out.resolve()
    work = (args.work or out.parent / "model-survey-work").resolve()
    budget = Budget(args.budget_seconds)
    try:
        gpus = visible_gpus() if args.gpus == "auto" else [
            g.strip() for g in args.gpus.split(",") if g.strip()
        ]
        if not gpus:
            raise SurveyError("no GPU is visible to this process")
        inventories = [load_inventory(path.resolve()) for path in args.inventory]
        asset_dir = args.asset_dir.resolve() if args.asset_dir else fetch_asset(work)
        for index, inv in enumerate(inventories):
            run_model(
                inv,
                asset_dir=asset_dir,
                work=work,
                out=out,
                gpus=gpus,
                budget=budget,
                night=args.night,
                python=sys.executable,
                reserve=_later_reserve(inventories[index + 1 :]),
            )
    except SurveyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
