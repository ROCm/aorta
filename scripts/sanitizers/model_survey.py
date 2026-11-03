#!/usr/bin/env python3
"""Run waitcheck, ConSan and FpSan over open-source model kernels for the workload survey.

The nightly ``sanitizers-model-survey`` job runs this once per model inventory in
``recipes/sanitizers/models/`` with the default ``--sanitizers waitcheck,consan``;
``sanitizers-fpsan-survey`` runs it with ``--sanitizers fpsan``. For each model it:

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
   waitcheck scanned, is merged as an ``error`` for its kernel;
4. runs every launched (kernel, scenario) under FpSan: one ``aorta sweep`` per
   pair whose ``fpsan_command`` is ``fpsan_compare.py`` comparing the pair with
   its reference (the inventory's ``fpsan`` block). The per-pair reports are
   merged into ``<out>/fpsan-<model>/``, a case that wrote no report as an
   ``error``.

Steps 2-4 run only for the sanitizers ``--sanitizers`` selects; step 1 always
runs. One time budget covers every launch, waitcheck sweep, and sanitizer
case. A launch or case starts only while it could still finish at its own
timeout. After a model's first launch wave, a launch must also leave that
model's ConSan and FpSan waves, so slow launches cannot spend the time that
writes their reports, and a ConSan case must leave its FpSan wave. A waitcheck
sweep is one process and stops at its own deadline, before it would spend
this model's later waves or the time held for a later model; on that deadline
the sweep's process group is stopped and the timeout is recorded. Each later
model keeps one launch wave and one wave per selected sanitizer, so an earlier
model running to its timeouts cannot leave the later one with no report.

Each output directory also gets ``model_survey.json``: what was launched, what
failed to launch, which ConSan or FpSan cases ran, were excluded, or did not
fit tonight.

The kernel sources come from the ``aorta-model-kernels`` release asset pinned
below by URL and SHA-256. Without ``--asset-dir`` the script downloads it,
refuses it unless the digest matches, and extracts it under ``--work``. Bump
``ASSET_VERSION``, ``ASSET_URL`` and ``ASSET_SHA256`` together with each
inventory's ``asset_version``.

Infrastructure faults (an asset that does not match the inventory, a pair count
that drifted, no object compiled at all) exit non-zero. A sanitizer verdict
never does: the survey is observed-only.

    model_survey.py run --inventory recipes/sanitizers/models/gpt-oss.yaml --out results
    model_survey.py run --sanitizers fpsan --inventory recipes/sanitizers/models/gpt-oss.yaml --out results
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
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ASSET_SCHEMA = "aorta.model_kernel_asset/1"
# v2 (adds triton-tutorials) is built but not published: these three pins and
# every inventory's asset_version move together once model-kernels-v2 exists.
# Until then an inventory naming asset_version "2" is refused by check_asset.
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
    "fpsan",
}
_CONSAN_KEYS = {"timeout_seconds", "exclude"}
_FPSAN_KEYS = {"timeout_seconds", "exclude", "reference"}
_REFERENCE_KEYS = {"kernel", "num_warps", "num_stages", "constexprs", "grid"}
SANITIZERS = ("waitcheck", "consan", "fpsan")
# FpSan runs in its own nightly job, so a bare run keeps the waitcheck and ConSan survey.
DEFAULT_SANITIZERS = ("waitcheck", "consan")
FPSAN_COMPARE = Path(__file__).resolve().with_name("fpsan_compare.py")
# Each FpSan case runs two sides in sequence; this much of its timeout is left
# for the driver and the sweep around them.
_FPSAN_SIDE_SLACK_SECONDS = 30
_VERDICT_RANK = {"pass": 0, "not_checked": 1, "warn": 2, "error": 3, "fail": 4}
_FAILED_STATES = {"error", "timed_out"}
# ConSan names the reader process in a parse error; one cause across kernels
# should count as one cause.
_READER_RE = re.compile(r"reader \d+ ")


class SurveyError(RuntimeError):
    """An infrastructure fault: the survey cannot produce a trustworthy row."""


@dataclass(frozen=True)
class FpsanPlan:
    """An inventory's ``fpsan:`` block.

    ``reference`` maps a kernel, or one ``<kernel>/<scenario>``, to how its
    reference differs from the default conservative config
    (``fpsan_compare.py``): another spec entry (``kernel``), launch options,
    constexpr overrides, or a grid. A scenario entry replaces its kernel's.
    """

    timeout: int
    exclude: dict[str, str]
    reference: dict[str, dict]

    def reference_for(self, pair: Pair) -> dict:
        return self.reference.get(
            f"{pair.kernel}/{pair.scenario}", self.reference.get(pair.kernel, {})
        )


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
    fpsan: FpsanPlan | None = None


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SurveyError(f"{name} must be a positive integer, got {value!r}")
    return value


def _reasons(value: object, name: str) -> dict[str, str]:
    value = value or {}
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in value.items()
    ):
        raise SurveyError(f"{name} must map kernel names to reasons")
    return dict(value)


def _reference(value: object, name: str) -> dict:
    if not isinstance(value, dict) or not value or set(value) - _REFERENCE_KEYS:
        raise SurveyError(f"{name} must be a non-empty mapping of {sorted(_REFERENCE_KEYS)}")
    if "kernel" in value and (not isinstance(value["kernel"], str) or not value["kernel"]):
        raise SurveyError(f"{name}.kernel must be a spec kernel name")
    for key in ("num_warps", "num_stages"):
        if key in value:
            _positive_int(value[key], f"{name}.{key}")
    constexprs = value.get("constexprs", {})
    if not isinstance(constexprs, dict) or not all(
        isinstance(k, str) and k and (v is None or isinstance(v, (bool, int, float, str)))
        for k, v in constexprs.items()
    ):
        raise SurveyError(f"{name}.constexprs must map argument names to scalars")
    if "grid" in value:
        grid = value["grid"]
        if not isinstance(grid, list) or len(grid) != 3:
            raise SurveyError(f"{name}.grid must be three positive sizes")
        for size in grid:
            _positive_int(size, f"{name}.grid")
    return dict(value)


def _fpsan_plan(value: object, path: Path) -> FpsanPlan | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - _FPSAN_KEYS:
        raise SurveyError(f"{path}: fpsan must be a mapping of {sorted(_FPSAN_KEYS)}")
    reference = value.get("reference") or {}
    if not isinstance(reference, dict) or not all(isinstance(k, str) and k for k in reference):
        raise SurveyError(f"{path}: fpsan.reference must map kernel or kernel/scenario names to overrides")
    return FpsanPlan(
        timeout=_positive_int(value.get("timeout_seconds"), f"{path}: fpsan.timeout_seconds"),
        exclude=_reasons(value.get("exclude"), f"{path}: fpsan.exclude"),
        reference={
            kernel: _reference(override, f"{path}: fpsan.reference.{kernel}")
            for kernel, override in reference.items()
        },
    )


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
    return Inventory(
        model=model,
        asset_version=asset_version,
        specs=_str_list(doc.get("specs"), f"{path}: specs"),
        roles=frozenset(_str_list(doc.get("roles"), f"{path}: roles")),
        expected_pairs=_positive_int(doc.get("expected_pairs"), f"{path}: expected_pairs"),
        consan_timeout=_positive_int(consan.get("timeout_seconds"), f"{path}: consan.timeout_seconds"),
        exclude=_reasons(consan.get("exclude"), f"{path}: consan.exclude"),
        path=path,
        fpsan=_fpsan_plan(doc.get("fpsan"), path),
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
    proc = _CHILDREN.start(
        [str(shim)],
        env={**os.environ, "HIP_VISIBLE_DEVICES": gpu},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=WARM_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _stop_process_group(proc)
        item.reason = f"launch timed out after {WARM_TIMEOUT_SECONDS}s"
        item.seconds = time.monotonic() - start
        return item
    finally:
        _CHILDREN.forget(proc)
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
    """Run ``work(item, gpu)`` with at most one item per GPU at a time, keeping order.

    ``KeyboardInterrupt`` reaches only this thread, never a worker waiting on
    its subprocess. On an interrupt, queued items are dropped and every process
    group the survey started is stopped before the pool is joined, so the
    interrupt does not wait out the workers' timeouts.
    """
    free: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        free.put(gpu)

    def run(item: object) -> object:
        gpu = free.get()
        try:
            return work(item, gpu)
        finally:
            free.put(gpu)

    pool = ThreadPoolExecutor(max_workers=len(gpus))
    try:
        futures = [pool.submit(run, item) for item in items]
        return [future.result() for future in futures]
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        _CHILDREN.stop_all()
        raise
    finally:
        pool.shutdown(wait=True)


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


def fpsan_reference_args(reference: Mapping[str, object]) -> list[str]:
    """``fpsan_compare.py`` flags for one inventory ``fpsan.reference`` entry."""
    args: list[str] = []
    if "kernel" in reference:
        args += ["--reference-kernel", str(reference["kernel"])]
    for key in ("num_warps", "num_stages"):
        if key in reference:
            args += [f"--reference-{key.replace('_', '-')}", str(reference[key])]
    for name, value in sorted(dict(reference.get("constexprs") or {}).items()):
        args += ["--reference-constexpr", f"{name}={json.dumps(value)}"]
    if "grid" in reference:
        args += ["--reference-grid", ",".join(str(g) for g in reference["grid"])]
    return args


def write_fpsan_shim(
    path: Path,
    *,
    model_dir: Path,
    pair: Pair,
    work: Path,
    record: Path,
    python: str,
    timeout: int,
    reference: Mapping[str, object],
) -> None:
    """The ``fpsan_command`` for one pair: the driver over the asset's own harness."""
    roofline = model_dir / "roofline"
    side_timeout = max(1, (timeout - _FPSAN_SIDE_SLACK_SECONDS) // 2)
    argv = [
        python, str(FPSAN_COMPARE),
        "--launch", str(roofline / "harness" / "launch.py"),
        "--spec", str(roofline / "specs" / pair.spec),
        "--kernel", pair.kernel, "--scenario", pair.scenario,
        "--flat-dir", str(model_dir / "kernels"), "--pkg-src", str(model_dir / "pkg_src"),
        "--arch", TARGET, "--warp-size", str(WARP_SIZE),
        "--work", str(work), "--record", str(record), "--side-timeout", str(side_timeout),
        *fpsan_reference_args(reference),
    ]
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(roofline))}\n"
        f"exec {shlex.join(argv)}\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def fpsan_recipe(model: str, item: Warmed, *, command: Path, timeout: int) -> str:
    recipe = {
        "schema_version": 1,
        "mode": "sanitizer",
        "ticket": f"MODEL-SURVEY-FPSAN-{model.upper()}",
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
                "fpsan_command": str(command),
            },
            "scope": {"kind": "kernel"},
            "selection": {"requirement": "top_dispatch_count", "top_n": 1},
            "sanitizers": ["fpsan"],
            "policy": {
                "consan_policy": "strict",
                "on_missing_backend": "fail",
                "timeout_seconds": timeout,
            },
            "output": {"report": "sanitizer_report.json"},
        },
    }
    return yaml.safe_dump(recipe, sort_keys=False)


def pair_rotation(items: Sequence[Warmed], night: dt.date) -> list[Warmed]:
    """The FpSan order for ``night``: every pair in a fixed order, started one pair later each night."""
    ordered = sorted(items, key=lambda item: (item.pair.kernel, item.pair.scenario))
    if not ordered:
        return []
    start = night.toordinal() % len(ordered)
    return ordered[start:] + ordered[:start]


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


class _Children:
    """Every process group the survey has running, so an interrupt can stop them all."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: set[subprocess.Popen[str]] = set()
        self._stopped = False

    def start(self, argv: Sequence[str], **kwargs: object) -> subprocess.Popen[str]:
        """``Popen`` in a new session; refused once ``stop_all`` has run."""
        with self._lock:
            if self._stopped:
                raise KeyboardInterrupt("the survey was interrupted")
            proc = subprocess.Popen(argv, start_new_session=True, **kwargs)
            self._procs.add(proc)
            return proc

    def forget(self, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            self._procs.discard(proc)

    def stop_all(self) -> None:
        with self._lock:
            self._stopped = True
            procs = list(self._procs)
        for proc in procs:
            _stop_process_group(proc)


_CHILDREN = _Children()


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
        proc = _CHILDREN.start(
            [sys.executable, "-c", "import sys; from aorta.cli import main; sys.exit(main())",
             "sweep", "run", "--recipe", str(recipe), "--output-dir", str(out_dir)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
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
        finally:
            _CHILDREN.forget(proc)


def error_case_report(item: Warmed, reason: str, *, sanitizer: str = "consan") -> dict:
    """A per-case ``sanitizer`` report recording ``reason`` as an error for ``item``'s kernel.

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
        "sanitizer": sanitizer, "state": "error", "verdict": "error", "reason": reason,
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


def _usable_sanitizer_report(path: Path) -> bool:
    """True when ``path`` is a report the dashboard publisher will keep."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    from aorta.instrumentation.rocjitsu_sanitizers.models import SanitizerReport

    try:
        SanitizerReport.from_dict(data)
    except (TypeError, ValueError):
        return False
    return True


def waitcheck_failure_report(items: Sequence[Warmed], reason: str, *, timed_out: bool) -> dict:
    """A WaitCheck report for a scan that never wrote one.

    The dashboard publishes a survey case only when ``sanitizer_report.json``
    loads, and it copies the whole case directory with that report, so the
    sweep log beside it is published as the scan's evidence.
    """
    from aorta.instrumentation.rocjitsu_sanitizers.models import SanitizerReport

    state = "timed_out" if timed_out else "error"
    # Two scenarios of one kernel compile the same object. The worklist rejects
    # that repeated identity; a real WaitCheck sweep dedups it the same way.
    identities = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not item.ok or not item.symbol or not item.sha256 or item.code_object is None:
            continue
        key = (item.sha256, item.symbol)
        if key in seen:
            continue
        seen.add(key)
        identities.append({
            "name": item.symbol,
            "target": TARGET,
            "code_object": str(item.code_object),
            "code_object_sha256": item.sha256,
            "code_object_index": 0,
            "entry_offset": None,
        })
    if not identities:
        raise SurveyError("waitcheck failure report has no launched kernel")
    check = {
        "sanitizer": "waitcheck",
        "state": state,
        "verdict": "error",
        "reason": reason,
        "returncode": None,
        "findings": [],
        "coverage": [],
        "backend": {},
        "kernel_results": [
            {
                "identity": identity,
                "state": state,
                "verdict": "error",
                "findings": [],
                "reason": reason,
                "returncode": None,
                "diagnostics_truncated": False,
            }
            for identity in identities
        ],
    }
    return SanitizerReport.from_dict({
        "schema": "aorta.sanitizer_report/0.1",
        "target": TARGET,
        "worklist": {
            "schema": "aorta.kernel_worklist/0.1",
            "requirement": "top_dispatch_count",
            "top_n": len(identities),
            "kernel_count": len(identities),
            "kernels": [
                {
                    "identity": identity,
                    "total_time_ms": 0.0,
                    "dispatch_count": 1,
                    "sources": ["kernel"],
                }
                for identity in identities
            ],
        },
        "checks": [check],
        "overall_verdict": "error",
        "execution_status": "error",
    }).to_dict()


def _with_kernel_result(check: dict, identity: dict) -> dict:
    """A case's primary check, with a per-kernel result even when its backend wrote none."""
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


def _merge_checks(sanitizer: str, checks: Sequence[dict], backend: Mapping[str, str] | None = None) -> dict:
    """One check from the same sanitizer's check in every case report.

    Without ``backend`` the merged check keeps the first case's backend.
    """
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
    if backend is None:
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
    return merge_cases(reports, "consan")


def merge_fpsan(cases: Sequence[tuple[Pair, dict]]) -> dict:
    """One FpSan report per model from its per-(kernel, scenario) reports.

    Every pair is its own comparison, so each keeps its own kernel result even
    when two scenarios of one kernel compiled the same object, which the
    worklist then lists once. A kernel result cannot say which scenario it
    compared, so each finding's metadata names its ``kernel`` and
    ``scenario``, and the merged check's backend carries, per pair,
    ``<kernel>/<scenario>.verdict``, ``.reason`` when the case gave one, and,
    once the driver's line parsed, ``.reference_digest`` and
    ``.candidate_digest``.
    """
    from aorta.instrumentation.rocjitsu_sanitizers.fpsan_result import RESULT_MARKER

    backend = {"result_marker": RESULT_MARKER, "triton_instrumentation_mode": "fpsan"}
    reports = []
    for pair, report in cases:
        identity = report["worklist"]["kernels"][0]["identity"]
        label = f"{pair.kernel}/{pair.scenario}"
        checks = []
        for check in report["checks"]:
            if check["sanitizer"] == "fpsan":
                findings = [
                    {
                        **finding,
                        "kernel_name": finding.get("kernel_name") or identity["name"],
                        "metadata": {**(finding.get("metadata") or {}),
                                     "kernel": pair.kernel, "scenario": pair.scenario},
                    }
                    for finding in check.get("findings") or []
                ]
                check = _with_kernel_result({**check, "findings": findings}, identity)
                reported = check.get("backend") or {}
                backend[f"{label}.verdict"] = check["verdict"]
                if check.get("reason"):
                    backend[f"{label}.reason"] = str(check["reason"])
                for key in ("reference_digest", "candidate_digest"):
                    if isinstance(reported.get(key), str):
                        backend[f"{label}.{key}"] = reported[key]
            checks.append(check)
        reports.append({**report, "checks": checks})
    return merge_cases(reports, "fpsan", backend=backend)


def merge_cases(reports: Sequence[dict], sanitizer: str, *, backend: Mapping[str, str] | None = None) -> dict:
    """Merge per-case reports whose primary check is ``sanitizer`` into one report.

    Other checks a case carries (ConSan's waitcheck preflight) are merged beside
    it. A kernel identity two cases share is listed once in the worklist; each
    case keeps its own kernel result.
    """
    from aorta.instrumentation.rocjitsu_sanitizers.models import (
        CheckResult,
        KernelIdentity,
        KernelWorklist,
        SanitizerReport,
    )

    by_sanitizer: dict[str, list[dict]] = {}
    for report in reports:
        identity = report["worklist"]["kernels"][0]["identity"]
        for check in report["checks"]:
            if check["sanitizer"] == sanitizer:
                check = _with_kernel_result(check, identity)
            by_sanitizer.setdefault(check["sanitizer"], []).append(check)
    if sanitizer not in by_sanitizer:
        raise SurveyError(f"no case report carries a {sanitizer} check")
    names = [sanitizer] + sorted(name for name in by_sanitizer if name != sanitizer)
    kernels = []
    seen: set[str] = set()
    for report in reports:
        for kernel in report["worklist"]["kernels"]:
            key = KernelIdentity.from_dict(kernel["identity"]).stable_key
            if key not in seen:
                seen.add(key)
                kernels.append(kernel)
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
            CheckResult.from_dict(
                _merge_checks(name, by_sanitizer[name], backend if name == sanitizer else None)
            )
            for name in names
        ),
    )
    return merged.to_dict()


def _wave_holds(inv: Inventory, sanitizers: Sequence[str]) -> tuple[float, float]:
    """This model's ConSan and FpSan wave holds, zero for a sanitizer not selected."""
    consan = float(inv.consan_timeout) if "consan" in sanitizers else 0.0
    fpsan = float(inv.fpsan.timeout) if "fpsan" in sanitizers and inv.fpsan else 0.0
    return consan, fpsan


def _later_reserve(later: Sequence[Inventory], sanitizers: Sequence[str] = DEFAULT_SANITIZERS) -> float:
    """Seconds held so each later model can still launch and run each selected sanitizer.

    One launch wave, plus one waitcheck sweep, one ConSan wave and one FpSan
    wave for whichever of them is selected, per later inventory. On the
    nightly's eight GPUs the launch and ConSan waves cover all of Qwen3; on
    fewer devices each is still one case, which is what writes the report
    ``--require`` checks for.
    """

    scan = SCAN_TIMEOUT_SECONDS if "waitcheck" in sanitizers else 0
    return float(sum(WARM_TIMEOUT_SECONDS + scan + sum(_wave_holds(inv, sanitizers)) for inv in later))


def _copy_case_logs(src: Path, dest: Path) -> None:
    for path in sorted([*src.rglob("*.log"), *src.glob("fpsan_compare.json")]):
        target = dest / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def waitcheck_wave(
    inv: Inventory,
    warmed: Sequence[Warmed],
    *,
    recipe_dir: Path,
    out: Path,
    budget: Budget,
    reserve: float,
    keep: float,
) -> int:
    """Scan every compiled object in one sweep; returns the sweep's return code."""
    launched = [w for w in warmed if w.ok]
    launch_record = [w.to_dict() for w in warmed]
    wc_recipe = recipe_dir / f"waitcheck-{inv.model}.yaml"
    wc_recipe.write_text(waitcheck_recipe(inv.model, warmed), encoding="utf-8")
    wc_out = out / f"waitcheck-{inv.model}"
    print(f"[{time.strftime('%H:%M:%S')}] {inv.model}: waitcheck over {len(launched)} objects", flush=True)
    # Keep this model's later waves as well as the later-model hold. A scan
    # that cannot fit is recorded and skipped; it does not cancel them.
    deadline = _sweep_deadline(budget, reserve=reserve, keep=keep)
    not_started = (
        "running it would spend the sanitizer time held for this model "
        f"({int(keep)}s) or a later one"
    )
    if deadline <= 0:
        wc_rc = SWEEP_TIMED_OUT
        shutil.rmtree(wc_out, ignore_errors=True)
        wc_out.mkdir(parents=True)
        (wc_out / "sweep.log").write_text(f"waitcheck not started: {not_started}\n", encoding="utf-8")
    else:
        wc_rc = sweep(wc_recipe, wc_out, timeout=deadline)
    (wc_out / "model_survey.json").write_text(
        json.dumps({"model": inv.model, "inventory": inv.path.name, "pairs": launch_record,
                    "sweep_returncode": wc_rc, "sweep_timed_out": wc_rc == SWEEP_TIMED_OUT,
                    "sweep_timeout_seconds": None if deadline <= 0 else round(deadline, 1)},
                   indent=2) + "\n",
        encoding="utf-8",
    )
    # A scan that dies before writing a report would otherwise vanish from Tab 2:
    # the publisher keeps a case only when this file loads, and it copies the
    # directory, log included. A report the sweep did write is left alone.
    if not _usable_sanitizer_report(wc_out / "sanitizer_report.json"):
        if wc_rc == SWEEP_TIMED_OUT and deadline <= 0:
            reason = f"waitcheck_not_started: {not_started}"
        elif wc_rc == SWEEP_TIMED_OUT:
            reason = (
                f"waitcheck_timed_out: aorta sweep exceeded {deadline:.1f}s "
                "and its process group was stopped"
            )
        else:
            reason = f"waitcheck_produced_no_report: aorta sweep exited {wc_rc}"
        report = waitcheck_failure_report(
            launched, reason, timed_out=wc_rc == SWEEP_TIMED_OUT
        )
        (wc_out / "sanitizer_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return wc_rc


def consan_wave(
    inv: Inventory,
    warmed: Sequence[Warmed],
    *,
    root: Path,
    recipe_dir: Path,
    out: Path,
    gpus: Sequence[str],
    budget: Budget,
    night: dt.date,
    reserve: float,
) -> dict:
    """One scenario per kernel under ConSan, merged into ``<out>/consan-<model>/``."""
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
        _copy_case_logs(src, cs_out / "cases" / src.name)
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
    return {
        "consan_ran": len(ran),
        "consan_candidates": len(order),
        "consan_not_scheduled": len(skipped),
        "consan_excluded": len(excluded),
    }


def fpsan_wave(
    inv: Inventory,
    warmed: Sequence[Warmed],
    *,
    model_dir: Path,
    root: Path,
    recipe_dir: Path,
    out: Path,
    gpus: Sequence[str],
    budget: Budget,
    night: dt.date,
    python: str,
    reserve: float,
) -> dict:
    """Every launched (kernel, scenario) under FpSan, merged into ``<out>/fpsan-<model>/``.

    Each pair is one ``aorta sweep`` whose ``fpsan_command`` runs
    ``fpsan_compare.py`` against the pair's reference. Pairs are taken in a
    nightly rotation, so a budget that runs out skips different pairs each
    night. The driver's record (what the reference changed, both digests and
    cache checks) is kept beside each case's logs as ``fpsan_compare.json``.
    """
    plan = inv.fpsan
    if plan is None:
        raise SurveyError(f"{inv.model}: the inventory has no fpsan block")
    kernels = {w.pair.kernel for w in warmed}
    excluded = {k: r for k, r in plan.exclude.items() if k in kernels}
    not_launched = {
        f"{w.pair.kernel}/{w.pair.scenario}": w.reason
        for w in warmed
        if not w.ok and w.pair.kernel not in excluded
    }
    order = pair_rotation([w for w in warmed if w.ok and w.pair.kernel not in excluded], night)
    fpsan_root = root / "fpsan"
    skipped: list[str] = []
    lock = threading.Lock()

    def fpsan_one(item: Warmed, gpu: str) -> dict | None:
        if not budget.admit(plan.timeout, reserve=reserve):
            with lock:
                skipped.append(f"{item.pair.kernel}/{item.pair.scenario}")
            return None
        case_dir = item.case_dir / "fpsan"
        shutil.rmtree(case_dir, ignore_errors=True)
        case_dir.mkdir(parents=True)
        shim, record = case_dir / "fpsan.sh", case_dir / "record.json"
        write_fpsan_shim(
            shim, model_dir=model_dir, pair=item.pair, work=case_dir / "work", record=record,
            python=python, timeout=plan.timeout, reference=plan.reference_for(item.pair),
        )
        recipe = recipe_dir / f"fpsan-{item.pair.stem}.yaml"
        recipe.write_text(fpsan_recipe(inv.model, item, command=shim, timeout=plan.timeout), encoding="utf-8")
        case_out = fpsan_root / item.pair.stem
        rc = sweep(recipe, case_out, env={"HIP_VISIBLE_DEVICES": gpu}, timeout=float(plan.timeout))
        if rc == SWEEP_TIMED_OUT:
            report = error_case_report(
                item,
                f"fpsan_case_timed_out: aorta sweep exceeded {plan.timeout}s "
                "and its process group was stopped",
                sanitizer="fpsan",
            )
        else:
            try:
                report = json.loads((case_out / "sanitizer_report.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report = error_case_report(
                    item, f"fpsan_case_produced_no_report: aorta sweep exited {rc}", sanitizer="fpsan"
                )
        try:
            driver = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            driver = {}
        else:
            case_out.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record, case_out / "fpsan_compare.json")
        return {"pair": item.pair, "returncode": rc, "report": report,
                "driver": driver if isinstance(driver, dict) else {}}

    print(f"[{time.strftime('%H:%M:%S')}] {inv.model}: FpSan over up to {len(order)} pairs "
          f"({len(excluded)} kernels excluded, {int(budget.remaining())}s of budget left)", flush=True)
    ran = [r for r in on_gpus(gpus, order, fpsan_one) if r is not None]
    fp_out = out / f"fpsan-{inv.model}"
    shutil.rmtree(fp_out, ignore_errors=True)
    fp_out.mkdir(parents=True)
    if ran:
        merged = merge_fpsan([(r["pair"], r["report"]) for r in ran])
        (fp_out / "sanitizer_report.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    for result in ran:
        stem = result["pair"].stem
        _copy_case_logs(fpsan_root / stem, fp_out / "cases" / stem)
    selection = {
        "model": inv.model,
        "inventory": inv.path.name,
        "night": night.isoformat(),
        "budget_seconds": int(budget.seconds),
        "case_timeout_seconds": plan.timeout,
        "ran": [
            {
                "kernel": r["pair"].kernel,
                "scenario": r["pair"].scenario,
                "returncode": r["returncode"],
                "verdict": r["report"].get("overall_verdict"),
                "reference": r["driver"].get("reference"),
                "driver_error": r["driver"].get("error"),
                "differing_tensors": r["driver"].get("differing_tensors"),
                "seconds": r["driver"].get("seconds"),
                **(r["driver"].get("result") or {}),
            }
            for r in ran
        ],
        "not_scheduled_tonight": sorted(skipped),
        "excluded": excluded,
        "not_launched": not_launched,
    }
    (fp_out / "model_survey.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return {
        "fpsan_ran": len(ran),
        "fpsan_candidates": len(order),
        "fpsan_not_scheduled": len(skipped),
        "fpsan_excluded": len(excluded),
    }


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
    sanitizers: Sequence[str] = DEFAULT_SANITIZERS,
) -> dict:
    model_dir = check_asset(asset_dir, inv)
    pairs = model_pairs(inv, model_dir)
    root = work / inv.model
    print(f"[{time.strftime('%H:%M:%S')}] {inv.model}: launching {len(pairs)} pairs on {len(gpus)} GPU(s)", flush=True)
    items = [Warmed(pair, root / "cases" / pair.stem) for pair in pairs]
    # The first wave holds only the later-model time. Holding this model's
    # sanitizer waves there too would compile nothing on a budget too small
    # for them, and "no pair compiled" stops the survey before later models.
    first_wave = {id(item) for item in items[: len(gpus)]}
    consan_hold, fpsan_hold = _wave_holds(inv, sanitizers)
    keep = consan_hold + fpsan_hold
    warmed = on_gpus(
        gpus,
        items,
        lambda item, gpu: warm_one(
            item,
            model_dir=model_dir,
            python=python,
            gpu=gpu,
            budget=budget,
            reserve=reserve if id(item) in first_wave else reserve + keep,
        ),
    )
    launched = [w for w in warmed if w.ok]
    if not launched:
        raise SurveyError(f"{inv.model}: no pair compiled; first failure: {warmed[0].reason}")
    print(f"[{time.strftime('%H:%M:%S')}] {inv.model}: {len(launched)}/{len(pairs)} compiled", flush=True)

    recipe_dir = root / "recipes"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"model": inv.model, "pairs": len(pairs), "compiled": len(launched)}
    if "waitcheck" in sanitizers:
        wc_rc = waitcheck_wave(
            inv, warmed, recipe_dir=recipe_dir, out=out, budget=budget, reserve=reserve, keep=keep
        )
        summary |= {"waitcheck_returncode": wc_rc, "waitcheck_timed_out": wc_rc == SWEEP_TIMED_OUT}
    if "consan" in sanitizers:
        summary |= consan_wave(
            inv, warmed, root=root, recipe_dir=recipe_dir, out=out, gpus=gpus, budget=budget,
            night=night, reserve=reserve + fpsan_hold,
        )
    if "fpsan" in sanitizers:
        summary |= fpsan_wave(
            inv, warmed, model_dir=model_dir, root=root, recipe_dir=recipe_dir, out=out, gpus=gpus,
            budget=budget, night=night, python=python, reserve=reserve,
        )
    print(f"[{time.strftime('%H:%M:%S')}] {inv.model}: {json.dumps(summary)}", flush=True)
    return summary


def parse_sanitizers(text: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(name.strip() for name in text.split(",") if name.strip()))
    unknown = sorted(set(names) - set(SANITIZERS))
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated subset of {','.join(SANITIZERS)}, got {text!r}"
        )
    return tuple(name for name in SANITIZERS if name in names)


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
                     help="for every launch, waitcheck sweep, ConSan case and FpSan case; a "
                     "launch or case starts only if it can still finish at its own timeout, "
                     "a launch after a model's first wave also leaves that model's "
                     "ConSan and FpSan waves, a waitcheck sweep stops before spending that "
                     "hold, a ConSan case leaves this model's FpSan wave, and each later "
                     "model keeps one launch wave and one wave per selected sanitizer")
    run.add_argument("--sanitizers", type=parse_sanitizers, default=DEFAULT_SANITIZERS,
                     help=f"comma-separated subset of {','.join(SANITIZERS)} (default: "
                     f"{','.join(DEFAULT_SANITIZERS)}); fpsan needs an fpsan block in every inventory")
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
        if "fpsan" in args.sanitizers:
            missing = [inv.model for inv in inventories if inv.fpsan is None]
            if missing:
                raise SurveyError(f"fpsan was selected but {missing} have no fpsan block")
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
                reserve=_later_reserve(inventories[index + 1 :], args.sanitizers),
                sanitizers=args.sanitizers,
            )
    except SurveyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
