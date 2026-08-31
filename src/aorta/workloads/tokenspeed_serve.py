"""TokenSpeed online-serving benchmark workload.

Phase 1 of the TokenSpeed integration reached the engine through the probe path
(``mode: probe`` + ``_subprocess``), which carries a *verdict* but no metrics:
``ts_serve_probe.sh`` proves a model comes up and generates, and that is all a
probe cell can say. Serving performance -- TTFT, TPOT, ITL, throughput -- has
nowhere to go on that path, because only a workload class can populate
``WorkloadResult.metrics``, which is what ``aorta sweep run`` aggregates into
``matrix.json``'s ``metrics_summary`` and renders in ``perf.md``.

This workload closes that gap. It does not implement a load generator: TokenSpeed
ships its own (``tokenspeed bench serve``, the same harness AMD publishes its
numbers with), so reimplementing one here would measure a different benchmark
than upstream and drift from it. Instead this class owns the *orchestration* --
resolve config, launch the container, forward the cell's mitigation env, parse
what the harness exported, decide a verdict -- and delegates the actual
measurement to upstream.

Division of labour:

    ``tokenspeed_serve`` (host, this file)
        Config validation, script staging, ``docker run``, env forwarding,
        JSON parsing, aggregation across steps, verdict, optional perf gates.

    ``tokenspeed/ts_bench_serve.sh`` (in container)
        Start one server, poll readiness, run ``tokenspeed bench serve`` N
        times, audit each exported JSON, tear the server down.

Two properties are worth calling out because they are the ones that make results
trustworthy rather than merely present:

**The silent-pass guard.** ``tokenspeed bench serve`` exits 0 regardless of how
many requests failed -- ``metrics.failed`` is reported, never returned. A cell
where the engine refused every request would otherwise be indistinguishable from
a clean one, and worse, would still publish TTFT/throughput numbers computed from
whatever trickle succeeded. Both layers therefore re-read the exported JSON: the
script fails the step (exit 55), and :meth:`run` independently re-checks
``completed``/``failed`` per step so the guard survives someone running the
script by hand or editing its exit codes.

**Mitigation env actually reaching the engine.** The dispatcher resolves a cell's
mitigations into ``config["_aorta_trial_env"]`` and applies them to this
process's environment -- but the engine runs in a container, which does not
inherit them. Forwarding uses the platform's :func:`aorta.run.docker_env_flags`
helper, so an ``hsa_no_scratch_reclaim`` cell genuinely differs from ``none``
instead of quietly benchmarking the same configuration twice.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from aorta.run import docker_env_flags
from aorta.workloads._base import Workload, WorkloadResult

log = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).parent / "tokenspeed"
_BENCH_SCRIPT = "ts_bench_serve.sh"

# Digest-pinned, and genuinely so: a registry can retarget a date tag, so
# `:nightly-20260714` would let the default change underneath a blessed baseline
# while still reading as pinned. This digest is the content every measured number
# in docs/tokenspeed-serving.md was taken against.
#
# The tag it resolved from, for anyone reading a `docker pull` line:
# lightseekorg/tokenspeed-amd:nightly-20260714
_DEFAULT_IMAGE = (
    "lightseekorg/tokenspeed-amd"
    "@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78"
)
_DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

_DEFAULT_WORK_DIR = "/tmp/ts-work-serve"
_DEFAULT_NUM_PROMPTS = 64
_DEFAULT_INPUT_LEN = 1024
_DEFAULT_OUTPUT_LEN = 128
_DEFAULT_NUM_WARMUPS = 1
_DEFAULT_REQUEST_RATE = "inf"
_DEFAULT_SEED = 0
_DEFAULT_PERCENTILE_METRICS = "ttft,tpot,itl,e2el"
_DEFAULT_METRIC_PERCENTILES = "50,90,99"
_DEFAULT_READY_TIMEOUT_SEC = 900
_DEFAULT_BENCH_TIMEOUT_SEC = 1800
_DEFAULT_TEARDOWN_GRACE_SEC = 45
_DEFAULT_SHM_SIZE = "16g"

# Mirrors the `require_uint` ceilings in ts_bench_serve.sh. Day-long sanity rails
# rather than policy -- they catch a millisecond value passed as seconds -- but
# the host has to enforce the same ones, or a recipe the container will reject
# takes a node first and then fails as a workload error instead of a config one.
_MAX_READY_TIMEOUT_SEC = 86400
_MAX_TEARDOWN_GRACE_SEC = 3600
# Mirrors ts_bench_serve.sh: the grace must contain the gateway drain it derives,
# and below 5 no positive drain fits inside it.
_MIN_TEARDOWN_GRACE_SEC = 5

# How long the `docker run` client gets to exit after SIGTERM before it is
# killed. It is a client, not the workload: it has nothing to flush but its own
# pipes, and every path that stops it is already an abnormal one.
_CLIENT_TERMINATE_GRACE_SEC = 5

# Removal passes when the client could not be reaped, which is only possible
# while `Popen` is still running: such a client may create the container just
# after the first pass finds nothing. Three passes a second apart, because this
# runs on the way out of a process that is already dying -- long enough to cover
# a daemon that is slow to register the container, short enough that a supervisor
# waiting for the process to die is not left wondering.
_CLEANUP_REMOVE_ATTEMPTS = 3
_CLEANUP_REMOVE_RETRY_SEC = 1.0

# A few GiB of driver and runtime overhead survives any run, so the check is
# against growth past a margin rather than against zero. The leak this exists for
# is two orders of magnitude larger -- 256 GB on a 309 GB card.
_VRAM_LEAK_THRESHOLD_BYTES = 8 * 1024**3
# Generous against the ~30-45s observed for a tensor-parallel teardown, because
# the failure this prevents is expensive (the next cell dies during startup) and
# the wait costs nothing when there is nothing to wait for -- the common case
# exits on the first sample.
_VRAM_RELEASE_TIMEOUT_SEC = 300
_VRAM_RELEASE_POLL_SEC = 5

_DEFAULT_DATASET = "random"
# `random` generates its own prompts, so it is the only one that needs nothing
# staged; `sharegpt` measures against real conversation lengths, which is what
# makes a serving number comparable to a published one.
_SUPPORTED_DATASETS = frozenset({"random", "sharegpt"})
# Read-only, and under its own path rather than beside the exports: /ts-out is
# writable by the container, and a dataset the run could rewrite is not evidence.
_CONTAINER_DATASET_PATH = "/ts-data/dataset.json"

# Margin added on top of (readiness + steps x bench) when deriving the default
# host-side docker timeout: covers image pull, container start, and teardown.
_TIMEOUT_MARGIN_SEC = 600

_GPU_KFD_NODE = Path("/dev/kfd")

_STARTUP_RE = re.compile(r"^TS_BENCH_METRIC: server_startup_sec=(\d+)", re.MULTILINE)

# Emitted by ts_bench_serve.sh as each measured step begins, so the host can
# distinguish "the measured phase started" from "the measured phase produced a
# parseable export". Warmup steps deliberately do not emit it.
_MEASURED_STEP_START_RE = re.compile(r"^TS_BENCH_STEP_START: \d+", re.MULTILINE)

# Exit codes ts_bench_serve.sh uses, mapped to the reason recorded in
# ``failure_details``. Kept in lockstep with the script's header comment.
_EXIT_REASONS: dict[int, str] = {
    50: "server_exited_during_startup",
    51: "readiness_timeout",
    52: "health_generate_unhealthy",
    53: "bench_step_failed",
    54: "result_json_unusable",
    55: "served_request_shortfall",
    64: "usage_error",
}

_KNOWN_KEYS = frozenset(
    {
        "image",
        "model",
        "served_model_name",
        "tokenizer",
        "num_prompts",
        "warmup_steps",
        "gateway_startup_timeout_sec",
        "input_len",
        "output_len",
        "dataset",
        "dataset_path",
        "max_concurrency",
        "request_rate",
        "num_warmups",
        "ignore_eos",
        "seed",
        "percentile_metrics",
        "metric_percentiles",
        "ready_timeout_sec",
        "bench_timeout_sec",
        "teardown_grace_sec",
        "timeout_sec",
        "serve_args",
        "bench_args",
        "work_dir",
        "hf_home",
        "hf_token_env",
        "shm_size",
        "hip_visible_devices",
        "exclusive_gpus",
        "docker_args",
        "run_as_current_user",
        "keep_work_dir",
        "gates",
        "network",
        "port",
        "control_port",
        "hf_offline",
    }
)
_RESERVED_KEYS = frozenset({"steps"})

# Keys the bench export echoes back from its own arguments rather than measures.
# Dropped from the aggregate because this class already publishes the
# authoritative value for each, and a duplicate only widens `perf.md`'s metric
# table with a column that cannot vary within a cell.
#
# `completed` and `failed` are here for a sharper reason: they are per-step
# request counters, and the generic aggregation below would publish them as
# *means* alongside the `completed_total` / `failed_total` sums this class adds
# deliberately. A run of three 32-prompt steps then reports `completed: 32` and
# `completed_total: 96` side by side in the performance table, which reads as a
# discrepancy rather than as two units, and the mean is the one that hides a
# single bad step among good ones -- the exact thing the sums exist to avoid.
# They stay in the per-step detail, where a shortfall is attributable to a step.
_EXPORT_ECHO_KEYS = frozenset(
    {
        "num_prompts",
        "max_concurrency",
        "burstiness",
        "completed",
        "failed",
    }
)

# The host/container contract. `_container_env` takes the union of this and the
# values the workload actually set, because neither alone is enough: computing it
# catches anything added to `_container_env` later without a second list to
# update, while this set covers the keys whose configured value is *absence* --
# an unbounded `max_concurrency` sets no TS_MAX_CONCURRENCY, so nothing computed
# from `env` would reserve it, and the default configuration would be the one
# left unguarded.
#
# A test asserts every name here is one the workload sets under some
# configuration, so a key cannot be reserved here by mistake and quietly forbid a
# legitimate mitigation.
#
# Each of these is either read back by this class after the run -- to locate
# exports, to audit served-request counts, to report the cell's configuration --
# or determines where the container writes.
#
# Note this is not a blanket "reject every TS_*": a mitigation setting a knob the
# workload does not itself set is legitimate, and is the point of the matrix.
_PROTOCOL_ENV_KEYS = frozenset(
    {
        "TS_BENCH_STEPS",
        "TS_BENCH_WARMUP_STEPS",
        "TS_NUM_PROMPTS",
        "TS_INPUT_LEN",
        "TS_OUTPUT_LEN",
        "TS_NUM_WARMUPS",
        "TS_IGNORE_EOS",
        # Absent when unbounded, which is the default -- so this is the one key
        # the computed set cannot reach on a default configuration.
        "TS_MAX_CONCURRENCY",
        "TS_SEED",
        "TS_DATASET",
        # Absent for `random`, which is the default, and points at a host mount
        # the container cannot create for itself -- so overriding it would send
        # the bench at a path that does not exist, or at an unmounted one.
        "TS_DATASET_PATH",
        "TS_MODEL",
        "TS_SERVED_MODEL_NAME",
        "TS_TOKENIZER",
        "TS_OUT_DIR",
        "TS_RUN_TOKEN",
        # Absent when the corresponding field is empty, which is the default --
        # the same hole `TS_MAX_CONCURRENCY` had. `serve_args` and `bench_args`
        # are validated on the host (the drain-timeout bound among them) and
        # then passed through, so a mitigation supplying them would run flags
        # nothing checked, and `--drain-timeout 600` would be back.
        "TS_SERVE_ARGS",
        "TS_BENCH_ARGS",
        # Absent when `hip_visible_devices` is unset, which is the default, and
        # it decides more than the run: `_owned_gpu_indices` reads the recipe's
        # value to decide which GPUs this trial may be blamed for leaking. A
        # mitigation setting it would have the container use one set of devices
        # while the host attributed VRAM against another.
        "HIP_VISIBLE_DEVICES",
        "TS_PORT",
        "TS_CONTROL_PORT",
        "TS_PERCENTILE_METRICS",
        "TS_METRIC_PERCENTILES",
        "TS_REQUEST_RATE",
        "TS_READY_TIMEOUT",
        "TS_BENCH_TIMEOUT",
        "TS_TEARDOWN_GRACE",
        "TS_GATEWAY_STARTUP_TIMEOUT",
    }
)

# Optional per-trial perf gates. The platform has no metric-threshold gate for
# triage workloads (CI baselines gate *after* the fact, nightly-only), so a
# recipe that wants "fail the cell if TTFT regresses" needs the workload to
# enforce it. Keys are gate names; values are (metric, comparison).
_GATE_SPECS: dict[str, tuple[str, str]] = {
    "max_median_ttft_ms": ("median_ttft_ms", "max"),
    "max_p99_ttft_ms": ("p99_ttft_ms", "max"),
    "max_median_tpot_ms": ("median_tpot_ms", "max"),
    "max_p99_tpot_ms": ("p99_tpot_ms", "max"),
    "max_median_itl_ms": ("median_itl_ms", "max"),
    # p99, for both of these, is the half worth gating. The gateway delivers
    # several tokens per SSE chunk, so most recorded inter-token gaps are ~0 and
    # `median_itl_ms` sits near zero while the real stalls land in the tail --
    # `scripts/ci/eval_lib.py` says the same thing about the nightly allowlist.
    #
    # That observation rules out arming an ITL ceiling *automatically*, the way
    # the nightly does, by taking a margin around a blessed baseline: 0.0 x 1.25
    # is 0.0 and the gate fires on the first non-zero sample. It says nothing
    # against a gate a recipe writes out as an absolute number, which is what
    # these are -- `max_p99_itl_ms: 50` is a stated bound on tail stalls, and it
    # is unaffected by where the median happens to sit. Leaving it out meant the
    # one summary the docs recommend was the one a recipe could not gate on.
    "max_p99_itl_ms": ("p99_itl_ms", "max"),
    "max_median_e2el_ms": ("median_e2el_ms", "max"),
    "max_p99_e2el_ms": ("p99_e2el_ms", "max"),
    "min_output_throughput": ("output_throughput", "min"),
    "min_total_token_throughput": ("total_token_throughput", "min"),
    "min_request_throughput": ("request_throughput", "min"),
}


@dataclass
class _StepRecord:
    """One parsed ``tokenspeed bench serve`` export."""

    step: int
    path: Path
    doc: dict[str, Any]
    scalars: dict[str, float] = field(default_factory=dict)


def _is_scalar(value: Any) -> bool:
    """True for a real, finite, non-bool number.

    ``bool`` is excluded because ``isinstance(True, int)`` holds and a flag
    aggregated as a mean is meaningless. Non-finite values are excluded because
    ``matrix.json`` is JSON and ``NaN``/``inf`` do not round-trip through strict
    JSON readers.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _step_detail(record: _StepRecord) -> dict[str, Any]:
    """One step's scalars, with the export's integers still integers.

    ``scalars`` is float-typed because what is done with it is arithmetic --
    means, sums, gate comparisons -- but the same dict is also published per
    step, and there ``completed: 32.0`` is a counter that has been made to look
    like a measurement. The type comes back from the parsed document, which
    still has it; ``type(...) is int`` rather than ``isinstance`` because a JSON
    boolean is an ``int`` to ``isinstance``.
    """
    detail: dict[str, Any] = {"step": record.step}
    for key, value in record.scalars.items():
        original = record.doc.get(key)
        detail[key] = original if type(original) is int else value
    return detail


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


class TokenSpeedServeWorkload(Workload):
    """Benchmark a TokenSpeed serving endpoint and report serving metrics.

    Runs one containerised server per trial and drives ``tokenspeed bench
    serve`` against it ``steps`` times, so the recipe's ``steps`` is the number
    of measured bench repetitions rather than the number of servers started --
    weight load dominates a short bench, so re-serving per step would mostly
    measure model load.

    ``workload_config`` keys:
        image: OCI image reference (default: pinned TokenSpeed nightly).
            An environment registry entry's ``docker`` field wins over this,
            matching the tier-hint convention.
        model: HF model id to serve (default ``"Qwen/Qwen3-0.6B"``).
        served_model_name: name the bench asks for (default: ``model``).
        tokenizer: tokenizer id for the bench (default: ``model``).
        num_prompts: requests per bench step (default ``64``).
        input_len: random-dataset ISL in tokens (default ``1024``).
        output_len: random-dataset OSL in tokens (default ``128``).
        max_concurrency: in-flight request cap (default: unbounded).
        request_rate: arrival rate in req/s, or ``"inf"`` to submit all at
            once (default ``"inf"``).
        num_warmups: untimed warmup requests inside each bench step
            (default ``1``).
        warmup_steps: whole bench steps run and discarded before the measured
            ones (default ``1``). This is what absorbs Triton JIT
            compilation; ``num_warmups`` does not, because it warms requests
            within an invocation rather than the compile cache.
        gateway_startup_timeout_sec: orchestrator budget for the gateway to
            reach readiness (default: ``ready_timeout_sec``). TokenSpeed's own
            default is 60s, which a cold start exceeds.
        ignore_eos: hold OSL fixed so every cell does the same work
            (default ``True``). ``False`` is only accepted for
            ``dataset: sharegpt``: the bench CLI forces it back on for the
            random dataset after parsing, so the setting could not have taken
            effect. Use ``bench_args: ["--extra-body", '{"ignore_eos":
            false}']`` to reach it on ``random``.
        seed: dataset/sampling seed (default ``0``).
        percentile_metrics: which metrics get percentiles
            (default ``"ttft,tpot,itl,e2el"``).
        metric_percentiles: which percentiles (default ``"50,90,99"``).
        ready_timeout_sec: readiness deadline; must cover a cold-cache model
            download (default ``900``).
        bench_timeout_sec: per-step bench deadline (default ``1800``).
        teardown_grace_sec: SIGTERM grace for the server (default ``45``).
        timeout_sec: whole-container deadline (default: derived from the
            readiness, per-step and teardown budgets plus a margin).
        serve_args: extra ``tokenspeed serve`` args (list or string).
        bench_args: extra ``tokenspeed bench serve`` args (list or string).
        work_dir: node-local scratch root for scripts, HF cache and exported
            JSON (default ``"/tmp/ts-work-serve"``). Must be node-local: an
            NFS home under root-squash cannot be bind-mounted. Scripts and
            exports go to a per-uid ``<work_dir>/u<uid>`` so two users on one
            node do not collide.
        hf_home: HF cache directory (default ``<work_dir>/u<uid>/hf``, i.e.
            per-uid). Persisting it across runs is what keeps a big model from
            re-downloading. To share one cache between users of a node, point
            this at a directory an administrator pre-populated: a cache is only
            shareable if later users can write it, and a world-writable model
            cache is something any local user can pre-populate.
        hf_token_env: name of a host env var holding an HF token, forwarded
            for gated models (default ``"HF_TOKEN"``; skipped when unset).
        hf_offline: serve strictly from the pre-populated HF cache
            (default ``False``). For nodes with no egress.
        network: container ``--network`` (default ``"host"``). Host
            networking is what gives the container a route to the HF Hub on a
            node whose docker bridge has IPv4 forwarding disabled.
        port: gateway port for the OpenAI ``/v1`` surface, or ``"auto"``
            (default) to pick a free one. Under host networking this port
            binds on the host, so ``"auto"`` is what keeps two users of the
            same node from colliding.
        control_port: control port owning ``/health``, or ``"auto"``
            (default: ``port + 1`` when free).
        shm_size: container ``--shm-size`` (default ``"16g"``).
        hip_visible_devices: value for ``HIP_VISIBLE_DEVICES``
            (default: unset, i.e. all GPUs). A visibility filter, not an
            allocation -- see ``exclusive_gpus``.
        exclusive_gpus: assert that no other job shares this node's GPUs
            (default ``False``). Only with this set does unreleased device
            memory after teardown fail the trial; otherwise it is logged as an
            observation, because a co-tenant's allocation looks identical.
        docker_args: extra ``docker run`` args (list or string).
        run_as_current_user: run the container as the calling uid:gid
            (default ``True``) so exported JSON and the HF cache stay
            deletable by the caller.
        keep_work_dir: keep per-trial exports after cleanup
            (default ``True``).
        gates: optional perf gates that fail the trial, e.g.
            ``{max_median_ttft_ms: 500, min_output_throughput: 1000}``.
            Supported keys: see ``_GATE_SPECS``.
    """

    name: ClassVar[str] = "tokenspeed_serve"

    # One container holding one server plus a load generator. Two trials
    # sharing a node would contend for the same GPUs and the same gateway port,
    # which shows up as throughput noise rather than as an error, so trials must
    # not overlap. In-process isolation is enough: this class holds no CUDA/HIP
    # state itself -- everything GPU-touching lives in the container.
    trial_isolation_default: ClassVar[Literal["in_process", "process"]] = "in_process"
    trial_isolation_supported: ClassVar[frozenset[str]] = frozenset({"in_process", "process"})

    # ---------------------------------------------------------------- config

    def _validated_config(self) -> None:
        for key in self.config:
            if key in _KNOWN_KEYS or key in _RESERVED_KEYS or key.startswith("_aorta_"):
                continue
            log.warning("tokenspeed_serve: ignoring unknown workload_config key %r", key)

        cfg = self.config

        # An environment registry entry pinning ``docker`` is the more specific
        # statement (it is what ``aorta env`` recorded for the run), so it wins
        # over the recipe's workload_config image.
        env_descriptor = cfg.get("_aorta_environment") or {}
        env_image = env_descriptor.get("docker") if isinstance(env_descriptor, dict) else None
        self._image = str(env_image or cfg.get("image") or _DEFAULT_IMAGE)

        self._model = str(cfg.get("model") or _DEFAULT_MODEL)
        self._served_model_name = str(cfg.get("served_model_name") or self._model)
        self._tokenizer = str(cfg.get("tokenizer") or self._model)

        self._dataset, self._dataset_path = self._validated_dataset()
        # The copy under the work root that the container actually mounts, set
        # by _stage_dataset() during setup.
        self._staged_dataset: Path | None = None

        self._num_prompts = self._positive_int("num_prompts", _DEFAULT_NUM_PROMPTS)
        self._input_len = self._positive_int("input_len", _DEFAULT_INPUT_LEN)
        self._output_len = self._positive_int("output_len", _DEFAULT_OUTPUT_LEN)
        self._num_warmups = self._non_negative_int("num_warmups", _DEFAULT_NUM_WARMUPS)
        self._seed = self._non_negative_int("seed", _DEFAULT_SEED)
        # Default 1, not 0: the first bench invocation against a fresh server
        # pays Triton JIT compilation and runs several times slower than the
        # rest, so a default of 0 would hand every caller a mean step time
        # dominated by a compile that has nothing to do with serving speed.
        self._warmup_steps = self._non_negative_int("warmup_steps", 1)

        # ``steps`` arrives from the recipe's top-level field, injected by the
        # dispatcher. Guard it here: subprocess with a zero-step loop would
        # produce no JSON and be misreported as a parse failure.
        # Via the shared coercion so `steps: true` and `steps: 1.9` are rejected
        # rather than quietly run as one step, the same as every other int field.
        self._steps = self._positive_int("steps", 1) if cfg.get("steps") is not None else 1

        max_conc = cfg.get("max_concurrency")
        if max_conc is None:
            self._max_concurrency: int | None = None
        else:
            try:
                self._max_concurrency = self._positive_int("max_concurrency", 1)
            except ValueError as exc:
                # The bound is _positive_int's; only the way out of it is local,
                # since 0 is the spelling someone reaches for to mean unbounded
                # and this field expresses that by being absent.
                raise ValueError(f"{exc} (omit it for unbounded)") from exc

        self._request_rate = self._validated_request_rate(
            cfg.get("request_rate", _DEFAULT_REQUEST_RATE)
        )

        self._ignore_eos = self._bool("ignore_eos", True)
        # `random` pins the output length whatever the argv says. `tokenspeed
        # bench serve` sets `ignore_eos = True` for the random dataset on an
        # OpenAI-compatible backend *after* parsing, which is after it has
        # honoured `--disable-ignore-eos` -- so neither omitting `--ignore-eos`
        # nor passing the disable flag reaches the request. Accepting `false`
        # would publish `ignore_eos: false` on a trial that served a pinned
        # length: the reported configuration is not the one that ran, which is
        # the mislabelled pass the owned-flag guards exist to prevent. The
        # payload route does work -- `extra_body` is applied over the forced
        # value, and `--extra-body` is not a flag this workload reserves -- so
        # the message names it rather than only refusing.
        if self._dataset == _DEFAULT_DATASET and not self._ignore_eos:
            raise ValueError(
                "tokenspeed_serve: ignore_eos: false cannot take effect with "
                "dataset: random. The bench CLI forces EOS to be ignored for "
                "that dataset after parsing its arguments, so the trial would "
                "report a setting the run did not have. Use dataset: sharegpt, "
                "or ask for it in the request payload with bench_args: "
                "[\"--extra-body\", '{\"ignore_eos\": false}']."
            )

        self._run_as_current_user = self._bool("run_as_current_user", True)
        self._keep_work_dir = self._bool("keep_work_dir", True)

        self._percentile_metrics = str(cfg.get("percentile_metrics") or _DEFAULT_PERCENTILE_METRICS)
        self._metric_percentiles = str(cfg.get("metric_percentiles") or _DEFAULT_METRIC_PERCENTILES)
        self._validate_percentiles(self._metric_percentiles)

        # Ceilings mirror ts_bench_serve.sh's `require_uint` bounds, so a recipe
        # the container would reject fails here instead of after taking a node.
        self._ready_timeout = self._bounded_int(
            "ready_timeout_sec", _DEFAULT_READY_TIMEOUT_SEC, maximum=_MAX_READY_TIMEOUT_SEC
        )
        self._bench_timeout = self._positive_int("bench_timeout_sec", _DEFAULT_BENCH_TIMEOUT_SEC)
        self._teardown_grace = self._bounded_int(
            "teardown_grace_sec",
            _DEFAULT_TEARDOWN_GRACE_SEC,
            maximum=_MAX_TEARDOWN_GRACE_SEC,
            # The grace has to contain a gateway drain derived from it, and
            # below this there is no positive drain that fits inside it.
            minimum=_MIN_TEARDOWN_GRACE_SEC,
        )
        # Derive rather than hardcode: a 20B model on a cold HF cache spends
        # minutes in readiness alone, and a fixed default would kill the
        # container mid-download and report it as a timeout.
        derived_timeout = (
            self._ready_timeout
            + (self._steps + self._warmup_steps) * self._bench_timeout
            + self._teardown_grace
            + _TIMEOUT_MARGIN_SEC
        )
        # The orchestrator's own gateway budget defaults to 60s and kills a cold
        # start well before our readiness deadline; default it to ours so the
        # outer timeout is the binding one.
        self._gateway_startup_timeout = self._positive_int(
            "gateway_startup_timeout_sec", self._ready_timeout
        )
        self._timeout = self._positive_int("timeout_sec", derived_timeout)
        if self._timeout <= self._ready_timeout:
            raise ValueError(
                f"tokenspeed_serve: timeout_sec ({self._timeout}) must exceed "
                f"ready_timeout_sec ({self._ready_timeout}); the container "
                "would be killed before the server could finish starting."
            )

        self._serve_args = self._arg_list("serve_args")
        self._validate_drain_timeout()
        self._bench_args = self._arg_list("bench_args")
        self._docker_args = self._arg_list("docker_args")
        # Resolved here rather than beside `hf_home` below, because the guard on
        # the next line needs it: `_secret_env_names` reads it, and a recipe
        # naming another variable makes that name the credential on this node.
        self._hf_token_env = str(cfg.get("hf_token_env") or "HF_TOKEN")
        # Checked here as well as at argv-build time so a recipe naming an owned
        # flag fails before a node is occupied. The protocol floor and the secret
        # names are both available now -- the latter unconditionally, which is
        # the point of `_secret_env_names` not depending on the host having a
        # token. Only the keys whose *value* this run computes (TS_PORT and the
        # rest, which need the resolved run token) have to wait for
        # `_docker_argv`, and `_PROTOCOL_ENV_KEYS` already reserves their names.
        #
        # Without the secret half here, `docker_args: ["-e", "HF_TOKEN=..."]`
        # passed validation and `--dry-run` and was only refused inside
        # `_docker_argv`, after the trial had taken a GPU node -- so the recipe
        # that cannot work failed late, and the credential it was trying to put
        # into a world-readable argv was not named until then.
        self._reject_owned_docker_args(owned_env=self._secret_env_names())

        self._shm_size = str(cfg.get("shm_size") or _DEFAULT_SHM_SIZE)
        hip_devices = cfg.get("hip_visible_devices")
        self._hip_visible_devices = None if hip_devices is None else str(hip_devices)
        # Opt-in, and it has to be: nothing the workload can read establishes
        # exclusivity. Defaulting it on would fail healthy trials on shared nodes
        # for a co-tenant's allocation.
        self._exclusive_gpus = self._bool("exclusive_gpus", False)

        # Host networking by default. The server and the load generator both run
        # inside this container and talk over 127.0.0.1, so they do not need it
        # -- pulling weights from the HF Hub does. A cluster node with IPv4
        # forwarding disabled gives a bridged container no route out at all, and
        # the failure arrives as `LocalEntryNotFoundError` from
        # `snapshot_download`, which reads like a bad model id rather than a
        # missing network.
        self._network = str(cfg.get("network") or "host")
        self._hf_offline = self._bool("hf_offline", False)
        self._port_request = cfg.get("port", "auto")
        self._control_port_request = cfg.get("control_port", "auto")
        for label, value in (
            ("port", self._port_request),
            ("control_port", self._control_port_request),
        ):
            if isinstance(value, str) and value != "auto":
                raise ValueError(
                    f'tokenspeed_serve: {label} must be an int or "auto", ' f"got {value!r}"
                )
            # Anything that is not `"auto"` has to be an actual int, checked
            # before the range test rather than coerced by it. `int()` accepted
            # two shapes that bind a port the recipe did not ask for: a float
            # truncates, so `port: 8000.9` ran on 8000 while the recipe said
            # otherwise, and `port: null` -- a natural way to write "unset" in
            # YAML -- reached `int(None)` and surfaced as a TypeError with no
            # mention of which field caused it. `bool` is excluded because it is
            # an `int` subclass, so `port: true` would otherwise mean port 1.
            if not isinstance(value, str) and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(
                    f'tokenspeed_serve: {label} must be an int or "auto", got '
                    f"{value!r} ({type(value).__name__}). Omit the field for "
                    'an automatically chosen free port, or set it to "auto".'
                )
            # 1024, not 1, to match `ts_bench_serve.sh`: the container runs
            # unprivileged and cannot bind a reserved port, so the script rejects
            # anything below 1024 outright. Accepting 1..1023 here meant such a
            # recipe passed `setup()` and was then guaranteed to fail with the
            # script's exit 64 -- after the trial had occupied a node, and
            # reported as a workload failure rather than as the configuration
            # error it is.
            if not isinstance(value, str) and not 1024 <= int(value) <= 65535:
                raise ValueError(
                    f"tokenspeed_serve: {label} ({value!r}) must be in 1024..65535. "
                    "Ports below 1024 are privileged and the container runs "
                    "unprivileged, so ts_bench_serve.sh rejects them."
                )

        # Two services cannot bind one address, so this configuration cannot
        # come up -- and it would fail as a readiness timeout during bring-up,
        # which reads as a slow or broken engine rather than as the recipe error
        # it is. Auto-resolution already keeps the pair distinct.
        if (
            not isinstance(self._port_request, str)
            and not isinstance(self._control_port_request, str)
            and int(self._port_request) == int(self._control_port_request)
        ):
            raise ValueError(
                "tokenspeed_serve: port and control_port are both "
                f"{int(self._port_request)}; the gateway and the control "
                "endpoint are separate listeners and cannot share a port. "
                'Set one of them to "auto", or give them different values.'
            )

        # `work_dir` is the shared root; the scratch this trial writes to is a
        # per-uid directory under it. A fixed path in /tmp is owned by whoever
        # got there first, at that user's umask, so the second user on the node
        # failed creating `scripts/` inside it -- a permission error with no
        # connection to anything in their recipe -- and where the mode did
        # permit writing, `keep_work_dir: false` deleted the other user's
        # exports mid-run. Every recipe here names the same `/tmp/ts-work-serve`,
        # so scoping only the default would have left that unfixed.
        self._work_root = Path(str(cfg.get("work_dir") or _DEFAULT_WORK_DIR)).resolve()
        self._work_dir = self._work_root / f"u{os.getuid()}"
        # Per-uid, like the rest of the scratch, and *not* shared by default.
        #
        # Sharing it was tried and withdrawn. A cache is only shareable if later
        # users can write it, and making the root world-writable does not
        # achieve that: huggingface_hub creates `hub`, `.locks` and each model
        # directory at the creating user's umask, so the second user still fails
        # on the first cache miss -- and a world-writable model cache is
        # something any local user can pre-populate with entries a later run
        # would then load. Neither half is worth the download it saves.
        #
        # Sharing a pre-warmed cache is still the right thing on a busy node; it
        # just has to be deliberate. Point `hf_home` at a directory an
        # administrator populated, where read-only is the intended mode rather
        # than an accident of who got there first.
        hf_home = cfg.get("hf_home")
        self._hf_home = Path(str(hf_home)).resolve() if hf_home else self._work_dir / "hf"

        self._gates = self._validated_gates()

    def _validated_dataset(self) -> tuple[str, Path | None]:
        """Resolve ``dataset`` and, for ``sharegpt``, the file to bench against.

        ``sharegpt`` is the reason this needs validating rather than forwarding.
        Given no ``--dataset-path`` the bench CLI downloads ShareGPT from the Hub
        on first use, which is wrong here in three separate ways: the recipe is
        no longer reproducible (the URL is not pinned and the file is not
        content-addressed), a bridged container often has no route to the Hub at
        all, and the download lands inside the measured window -- so a cell would
        report the first trial as slower for a reason that is not the engine.

        So the host resolves the file and mounts it. ``dataset_path`` is required
        for ``sharegpt`` and must exist before the trial starts, which turns "the
        dataset was not staged" from a mid-run failure into a config error.
        """
        name = str(self.config.get("dataset") or _DEFAULT_DATASET).strip()
        if name not in _SUPPORTED_DATASETS:
            raise ValueError(
                f"tokenspeed_serve: dataset ({name!r}) must be one of "
                f"{', '.join(sorted(_SUPPORTED_DATASETS))}"
            )

        raw_path = self.config.get("dataset_path")
        if name == "random":
            if raw_path:
                # The bench CLI raises "Cannot use 'random' dataset with
                # --dataset-path". Catching it here names the field instead.
                raise ValueError(
                    "tokenspeed_serve: dataset_path is meaningless for "
                    "dataset: random, which generates its prompts from "
                    "input_len/output_len. Set dataset: sharegpt to bench "
                    "against a file."
                )
            return name, None

        if not raw_path:
            raise ValueError(
                "tokenspeed_serve: dataset: sharegpt requires dataset_path "
                "pointing at a ShareGPT JSON file on the host. Without it the "
                "bench CLI downloads the dataset itself, which is not "
                "reproducible, needs Hub access from inside the container, and "
                "puts the download inside the measured window."
            )
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                f"tokenspeed_serve: dataset_path ({path}) is not a file. It must "
                "exist before the trial starts -- otherwise this fails after the "
                "model has loaded, and reads as a bench failure."
            )
        return name, path

    def _coerced_int(self, key: str, default: int) -> int:
        """Accept an integer, or a string spelling one, and nothing else.

        A bare ``int()`` accepted two kinds of malformed recipe and quietly ran a
        *different* load rather than failing: ``true`` (``bool`` is an ``int``
        subclass) and ``1.9`` (truncated) both mean one prompt. A recipe that
        cannot be read the way it was written must not be executed the way it was
        not -- a cell reporting `num_prompts: 1` for a value someone wrote as 1.9
        is a mislabelled result, which is the failure mode this whole file is
        organised against.
        """
        raw = self.config.get(key, default)
        if isinstance(raw, bool):
            raise ValueError(
                f"tokenspeed_serve: {key} must be an integer, got the boolean "
                f"{raw!r}; `bool` is an int subclass in Python, so this would "
                "silently run as 1 or 0."
            )
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(
                f"tokenspeed_serve: {key} ({raw!r}) must be a whole number; "
                "truncating it would run a different load than the recipe asks for."
            )
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tokenspeed_serve: {key} ({raw!r}) must be an integer") from exc
        return value

    def _positive_int(self, key: str, default: int) -> int:
        value = self._coerced_int(key, default)
        if value < 1:
            raise ValueError(f"tokenspeed_serve: {key} ({value}) must be >= 1")
        return value

    def _non_negative_int(self, key: str, default: int) -> int:
        value = self._coerced_int(key, default)
        if value < 0:
            raise ValueError(f"tokenspeed_serve: {key} ({value}) must be >= 0")
        return value

    def _bounded_int(self, key: str, default: int, *, maximum: int, minimum: int = 1) -> int:
        """A positive int the container will also accept.

        ``ts_bench_serve.sh`` bounds these, so a value outside its range passed
        ``setup()``, occupied a GPU node, and then exited 64 inside the container
        -- reported as a workload failure rather than as the configuration error
        it is, with the range that was violated visible only in the container log.
        """
        value = self._positive_int(key, default)
        if value > maximum or value < minimum:
            raise ValueError(
                f"tokenspeed_serve: {key} ({value}) must be between {minimum} "
                f"and {maximum}; ts_bench_serve.sh rejects anything outside "
                "that, so this would fail inside the container after occupying "
                "a node."
            )
        return value

    def _bool(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"tokenspeed_serve: {key} must be a bool, got {type(value).__name__}")
        return value

    def _validated_request_rate(self, value: Any) -> str:
        """Normalise ``request_rate`` to the token the bench CLI accepts.

        Only *positive* infinity means "unlimited". A blanket non-finite check
        would fold NaN and -inf into it, so a typo'd rate would silently become
        the most aggressive setting available -- and the cell would still look
        like it ran the load the recipe asked for.
        """
        if isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity"}:
            return "inf"
        # bool is a subclass of int, so float(True) is 1.0: YAML
        # `request_rate: true` would otherwise be accepted as one request per
        # second and run a load nobody asked for. Same coercion already rejected
        # for the integer fields.
        if isinstance(value, bool):
            raise ValueError(
                f"tokenspeed_serve: request_rate ({value!r}) must be a number "
                'or "inf", not a boolean'
            )
        # An int too large for a float raises here rather than overflowing, and
        # the raw message ("int too large to convert to float") does not say
        # which key was wrong.
        try:
            rate = float(value)
        except OverflowError as exc:
            raise ValueError(
                f"tokenspeed_serve: request_rate ({value!r}) is too large to be "
                'a rate; use a positive number, or "inf" to submit every '
                "request at once"
            ) from exc
        if math.isnan(rate) or math.isinf(rate):
            # Only the string spellings above mean "unlimited". An infinite
            # *float* does not, because by the time it arrives here there is no
            # way to tell which one it was: YAML reads `.inf` and `1.0e999` as
            # the same value, and so does `float("1e999")` -- so accepting it
            # promoted a typo'd finite rate to the heaviest load the harness can
            # generate, while the trial went on reporting the rate the recipe
            # asked for. The quoted token costs the deliberate case one pair of
            # quotes and makes the accident impossible.
            raise ValueError(
                f"tokenspeed_serve: request_rate ({value!r}) is not a usable "
                'rate; use a positive number, or the quoted string "inf" to '
                "submit every request at once (an unquoted infinite float is "
                "indistinguishable from a finite rate that overflowed)"
            )
        if rate <= 0:
            raise ValueError(
                f"tokenspeed_serve: request_rate ({value!r}) must be > 0 or "
                '"inf" (submit every request at once)'
            )
        return repr(rate)

    @staticmethod
    def _validate_percentiles(spec: str) -> None:
        for token in spec.split(","):
            token = token.strip()
            if not token:
                raise ValueError(
                    f"tokenspeed_serve: metric_percentiles ({spec!r}) has an " "empty entry"
                )
            try:
                pct = float(token)
            except ValueError as exc:
                raise ValueError(
                    f"tokenspeed_serve: metric_percentiles ({spec!r}) entry "
                    f"{token!r} is not a number"
                ) from exc
            if not 0 < pct < 100:
                raise ValueError(
                    f"tokenspeed_serve: metric_percentiles ({spec!r}) entry "
                    f"{token!r} must be in (0, 100)"
                )

    def _arg_list(self, key: str) -> list[str]:
        value = self.config.get(key)
        if value is None:
            return []
        if isinstance(value, str):
            # shlex, not split(): the string form is the documented one, and
            # `--foo "bar baz"` under split() became three arguments, silently
            # changing the invocation and defeating the boundary checks below --
            # which see tokens, not what the user wrote.
            try:
                items = shlex.split(value)
            except ValueError as exc:
                raise ValueError(
                    f"tokenspeed_serve: {key} ({value!r}) is not a parseable "
                    f"command line: {exc}. Quote it as a shell would, or pass a "
                    "list, where each entry is one argument and no quoting is "
                    "involved."
                ) from exc
        elif isinstance(value, (list, tuple)):
            items = [str(item) for item in value]
        else:
            raise ValueError(
                f"tokenspeed_serve: {key} must be a string or list, " f"got {type(value).__name__}"
            )
        # These cross into the container as a JSON array and are decoded there
        # from a NUL-separated stream, NUL being the one byte an argument cannot
        # contain. An entry carrying one would be split at it into two
        # arguments, so it is rejected on the host as well as in the script --
        # here it names the offending key on the CPU gate rather than failing
        # after a node has been occupied.
        for index, item in enumerate(items):
            if "\0" in item:
                raise ValueError(
                    f"tokenspeed_serve: {key}[{index}] contains a NUL byte, "
                    "which no argument can carry"
                )
        return items

    def _validated_gates(self) -> dict[str, float]:
        raw = self.config.get("gates")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"tokenspeed_serve: gates must be a mapping, got {type(raw).__name__}")
        gates: dict[str, float] = {}
        for key, value in raw.items():
            if key not in _GATE_SPECS:
                raise ValueError(
                    f"tokenspeed_serve: unknown gate {key!r}; supported gates "
                    f"are {sorted(_GATE_SPECS)}"
                )
            # Checked before float(), which maps True to 1.0 -- so a YAML typo
            # like `max_median_ttft_ms: true` installed a real 1 ms gate that
            # every run breaches, instead of the finite-number error promised
            # just below.
            if isinstance(value, bool):
                raise ValueError(
                    f"tokenspeed_serve: gate {key!r} bound ({value!r}) must be "
                    "a finite positive number, not a boolean"
                )
            bound = float(value)
            if not math.isfinite(bound) or bound <= 0:
                raise ValueError(
                    f"tokenspeed_serve: gate {key!r} bound ({value!r}) must be "
                    "a finite positive number"
                )
            gates[key] = bound
        return gates

    # The running `docker run` client, so the signal handler can reap it before
    # removing the container. Class-level defaults rather than an __init__: the
    # base class owns construction, and every path that reads these tolerates
    # `None` (no client running).
    _docker_client: subprocess.Popen[str] | None = None
    # Reentrant, and that is the whole point. A Python signal handler runs in the
    # main thread -- the same thread that holds this lock while spawning and
    # publishing the client -- so with a plain Lock a SIGTERM arriving in that
    # window deadlocked the handler against the code it interrupted, and the
    # container it was there to remove survived until someone sent SIGKILL.
    _docker_client_lock = threading.RLock()
    # Set for the width of the spawn, so the handler can tell "no client" from
    # "a client may exist that I cannot see yet". Publication is not atomic with
    # process creation: `Popen` returns, and only then is the object assigned.
    _docker_client_spawning = False

    # ---------------------------------------------------------------- setup

    def setup(self) -> None:
        self._validated_config()

        if shutil.which("docker") is None:
            raise RuntimeError(
                "tokenspeed_serve: 'docker' not on PATH. This workload runs the "
                "TokenSpeed engine in its published container; install Docker or "
                "run on a node that has it."
            )

        # Fail in setup(), not run(): a benchmark with no GPU produces no step
        # times, which the matrix would report as a did-not-run without ever
        # saying why.
        if not os.access(_GPU_KFD_NODE, os.R_OK | os.W_OK):
            raise RuntimeError(
                f"tokenspeed_serve: no accessible ROCm GPU ({_GPU_KFD_NODE} is "
                "not readable+writable). TokenSpeed's Gluon kernels target "
                "gfx950/gfx1250; run this on such a node."
            )

        # Stage the in-container script node-locally. The package may live on an
        # NFS home, and under root-squash `docker run -v /home/<user>/...` fails
        # with a permission error on the mount point itself -- so copying into
        # work_dir is what makes this runnable on a cluster at all, not just a
        # convenience.
        self._scripts_dir = self._work_dir / "scripts"
        self._out_dir = self._work_dir / "out"
        # Create the cache directories host-side rather than letting the
        # container do it: as a non-root user the container can create paths
        # *under* the mount but the mount root must already exist and be owned
        # by the caller.
        # The shared root has to be enterable and writable by every user of the
        # node, or the second one cannot create their own `u<uid>` beneath a
        # root the first created at a 022 umask. Same reasoning as /tmp itself,
        # including the sticky bit -- which stops one user removing another's
        # scratch, with the exception that decides the check below it: the
        # *owner* of a sticky directory may rename or remove any entry in it
        # whoever owns that entry. Whoever ran first owns this root, so the bit
        # binds every user except that one, and _assert_trusted_work_root is
        # what declines to trust a root belonging to a stranger.
        #
        # Only attempted on creation: an existing root is the administrator's
        # (or the first user's) to set, and silently widening it would be its own
        # surprise.
        try:
            existed = self._work_root.exists()
            self._work_root.mkdir(parents=True, exist_ok=True)
            if not existed:
                # mkdir's mode argument is masked by the umask, which is the
                # very thing being worked around here, so set it afterwards.
                # Suppressed on failure: losing the race to another user's
                # setup() leaves their mode in place, which is the same
                # outcome as finding the root already there.
                with contextlib.suppress(OSError):
                    os.chmod(self._work_root, 0o1777)
        except OSError as exc:
            raise RuntimeError(
                f"tokenspeed_serve: cannot create {self._work_root}: {exc}. "
                "work_dir must be a node-local, writable path (an NFS home "
                "under root-squash cannot be bind-mounted into a container)."
            ) from exc

        self._assert_trusted_work_root()
        self._assert_own_scratch()

        for directory in (
            self._scripts_dir,
            self._out_dir,
            self._hf_home,
            self._out_dir / "home" / ".cache",
            self._out_dir / "triton-cache",
            self._out_dir / "torchinductor",
        ):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"tokenspeed_serve: cannot create {directory}: {exc}. "
                    "work_dir must be a node-local, writable path (an NFS home "
                    "under root-squash cannot be bind-mounted into a container)."
                ) from exc

        source = _SCRIPTS_DIR / _BENCH_SCRIPT
        if not source.is_file():
            raise RuntimeError(
                f"tokenspeed_serve: packaged script missing at {source}. The "
                "wheel must ship workloads/tokenspeed/*.sh (see pyproject "
                "package-data)."
            )
        # Content-addressed, not a fixed name. `scripts/ts_bench_serve.sh` is
        # shared by every concurrent run from this uid and was overwritten in
        # place, so a second sweep could truncate or replace the file after this
        # trial had syntax-checked it and before its container read the bind
        # mount -- and two different checkouts could have the host validate one
        # script while the container ran the other. Naming by digest makes the
        # staged file immutable for the life of its content: a concurrent run
        # with the same script writes the same bytes to the same path, and one
        # with different bytes writes elsewhere.
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        staged = self._scripts_dir / f"{Path(_BENCH_SCRIPT).stem}.{digest}.sh"
        if not staged.exists():
            # Written via a temporary and renamed, so a concurrent run never
            # sees a partially written script under the final name.
            tmp = staged.with_name(f".{staged.name}.{os.getpid()}")
            shutil.copyfile(source, tmp)
            tmp.chmod(0o755)
            os.replace(tmp, staged)
        self._staged_script = staged

        # Syntax-check the staged copy rather than trusting it: a truncated copy
        # or an edit staged by hand would otherwise fail deep inside the
        # container, where the error reads as a container problem.
        check = subprocess.run(["bash", "-n", str(staged)], capture_output=True, text=True)
        if check.returncode != 0:
            raise RuntimeError(
                f"tokenspeed_serve: staged script {staged} failed 'bash -n':\n"
                f"{check.stderr.strip()}"
            )

        self._staged_dataset = self._stage_dataset()

    def _stage_dataset(self) -> Path | None:
        """Copy the ShareGPT file under the work root and mount that copy.

        ``dataset_path`` is checked with *this* process's credentials, but the
        bind mount is resolved by the docker daemon, which is a different reader.
        The advertised setup is an NFS home with the node-local work root that
        exists precisely because of it: there, a dataset the recipe author can
        read is squashed to nobody for the daemon, so the mount arrives empty or
        the container fails on it -- after the model has loaded, which is the
        mid-run failure ``_validated_dataset`` exists to convert into a config
        error. Staging moves the read to where the daemon is root.

        Content-addressed for the same reason the script is: the copy is
        immutable for the life of its bytes, so a concurrent run either writes
        the same bytes to the same path or writes elsewhere, and a second sweep
        over an unchanged dataset re-uses the copy instead of paying for it.
        """
        if self._dataset_path is None:
            return None
        source = self._dataset_path
        digest = hashlib.sha256()
        try:
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise RuntimeError(f"tokenspeed_serve: cannot read dataset_path {source}: {exc}") from exc

        datasets_dir = self._work_dir / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        staged = datasets_dir / f"{source.stem[:40]}.{digest.hexdigest()[:16]}{source.suffix}"
        if not staged.exists():
            size_mb = source.stat().st_size / (1024 * 1024)
            log.info(
                "tokenspeed_serve: staging dataset %s (%.0f MiB) at %s", source, size_mb, staged
            )
            tmp = staged.with_name(f".{staged.name}.{os.getpid()}")
            try:
                shutil.copyfile(source, tmp)
                tmp.chmod(0o644)
                os.replace(tmp, staged)
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"tokenspeed_serve: cannot stage dataset_path {source} at "
                    f"{staged}: {exc}"
                ) from exc
        return staged

    # ------------------------------------------------------------------ run

    def _container_env(self) -> dict[str, str]:
        """Env for the container: TS_* knobs, then the cell's mitigations.

        Mitigations are merged last, but "last" here settles precedence between
        a mitigation and *anything else a mitigation may legitimately set* --
        not between a mitigation and the knobs below. A mitigation naming a key
        this workload owns is rejected outright rather than allowed to win: the
        host would keep auditing and reporting its own value while the container
        ran the other one. The owned set is everything in ``env``, plus
        ``_PROTOCOL_ENV_KEYS`` for the keys whose configured value is absence,
        plus the secret names. So of the ``TS_*`` namespace, only what this
        workload does not own is a mitigation's to set -- in practice
        ``TS_DRAIN_TIMEOUT``, which ``ts_bench_serve.sh`` reads, bounds against
        the teardown grace, and this class never sets. Everything else a
        mitigation carries (engine and runtime variables, which is what the
        matrix mostly varies) is untouched by the guard and lands here.

        Anything the workload does set is therefore configured through
        ``workload_config``, which is the only route that keeps the host's
        expectations, its audit and its reported configuration in step with the
        run.
        """
        env: dict[str, str] = {
            "TS_MODEL": self._model,
            "TS_SERVED_MODEL_NAME": self._served_model_name,
            "TS_TOKENIZER": self._tokenizer,
            "TS_BENCH_STEPS": str(self._steps),
            "TS_BENCH_WARMUP_STEPS": str(self._warmup_steps),
            "TS_GATEWAY_STARTUP_TIMEOUT": str(self._gateway_startup_timeout),
            "TS_DATASET": self._dataset,
            "TS_NUM_PROMPTS": str(self._num_prompts),
            "TS_NUM_WARMUPS": str(self._num_warmups),
            "TS_REQUEST_RATE": self._request_rate,
            "TS_SEED": str(self._seed),
            "TS_IGNORE_EOS": "1" if self._ignore_eos else "0",
            "TS_PERCENTILE_METRICS": self._percentile_metrics,
            "TS_METRIC_PERCENTILES": self._metric_percentiles,
            "TS_READY_TIMEOUT": str(self._ready_timeout),
            "TS_BENCH_TIMEOUT": str(self._bench_timeout),
            "TS_TEARDOWN_GRACE": str(self._teardown_grace),
            "TS_OUT_DIR": "/ts-out",
            "TS_RUN_TOKEN": self._run_token,
            "TS_PORT": str(self._port),
            "TS_CONTROL_PORT": str(self._control_port),
            # Every cache location is redirected into the mount: as a non-root
            # user the container's default ``/root`` paths are not writable, and
            # Triton masks an unwritable cache as the wildly misleading "Triton
            # is not supported on the current platform".
            "HF_HOME": "/hf-cache",
            "HF_HUB_CACHE": "/hf-cache/hub",
            "TRITON_CACHE_DIR": "/ts-out/triton-cache",
            "HOME": "/ts-out/home",
            "XDG_CACHE_HOME": "/ts-out/home/.cache",
            # TORCHINDUCTOR_CACHE_DIR must be set explicitly, not just pointed
            # somewhere writable: torch's ``cache_dir()`` only computes a
            # default when this is unset, and that default calls
            # ``getpass.getuser()``. Under ``--user <uid>`` the uid has no
            # passwd entry in the image, so the call raises
            # ``KeyError: getpwuid(): uid not found`` at *import* of
            # ``torch._dynamo`` -- which surfaces as the engine dying during
            # startup with a traceback that never mentions users or uids.
            "TORCHINDUCTOR_CACHE_DIR": "/ts-out/torchinductor",
        }
        if self._run_as_current_user:
            # Belt and braces for the same uid-has-no-passwd-entry problem:
            # ``getpass.getuser()`` prefers these over the passwd database, so
            # anything else in the stack that calls it also stays working.
            user = _host_username()
            env["USER"] = user
            env["LOGNAME"] = user
        # ISL/OSL are `random`-only: the bench CLI maps them onto
        # --random-input-len/--random-output-len, and the script drops them for
        # any other dataset. Forwarding them anyway left a ShareGPT container
        # holding a shape it would not use, which is the kind of thing a reader
        # of the env dump reasonably believes.
        if self._dataset == _DEFAULT_DATASET:
            env["TS_INPUT_LEN"] = str(self._input_len)
            env["TS_OUTPUT_LEN"] = str(self._output_len)
        if self._dataset_path is not None:
            env["TS_DATASET_PATH"] = _CONTAINER_DATASET_PATH
        if self._max_concurrency is not None:
            env["TS_MAX_CONCURRENCY"] = str(self._max_concurrency)
        # JSON, not a space-joined string. The recipe documents these as lists,
        # and joining them threw the boundaries away: one item containing a
        # space arrived as two arguments, and a `*` or `?` in a value was
        # glob-expanded against the container's filesystem on the way in. The
        # script decodes these back into a bash array.
        if self._serve_args:
            env["TS_SERVE_ARGS"] = json.dumps(self._serve_args)
        if self._bench_args:
            env["TS_BENCH_ARGS"] = json.dumps(self._bench_args)
        if self._hip_visible_devices is not None:
            env["HIP_VISIBLE_DEVICES"] = self._hip_visible_devices

        # Gated models need a token, and it is deliberately *not* put in here.
        # Everything in this mapping becomes `-e KEY=VALUE` in the docker
        # client's argv, which /proc/<pid>/cmdline exposes to every user on the
        # node for as long as the trial runs. Keeping the token out of the
        # recipe and out of the logs is not enough on its own if the value ends
        # up there. See _docker_argv and _secret_env for how it is passed.
        if self._hf_offline:
            # For nodes with no egress: serve strictly from the pre-populated
            # cache and fail loudly rather than hanging on a doomed download.
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"

        trial_env = self.config.get("_aorta_trial_env") or {}
        if trial_env:
            # A mitigation may set anything the workload does not itself depend
            # on -- that is the point of the matrix, and a mitigation silently
            # losing to a workload default would make two cells identical while
            # reporting them as different.
            #
            # What it may not do is redefine a value this workload itself set,
            # because only the container would learn about the new one.
            # `TS_NUM_PROMPTS=999` would have the script request 999 while
            # `_build_result` still audited against the recipe's 32, so a
            # perfectly healthy cell would fail its served-request audit;
            # `TS_RUN_TOKEN` would have the script write exports under a name the
            # host's glob does not match, and the cell would fail for finding no
            # export at all; `TS_MAX_CONCURRENCY=1` would change the load while
            # `_aggregate` still reported the configured 8, so the cell would
            # *pass*, mislabelled -- the worst of the three, because nothing
            # fails and the number is wrong.
            #
            # The owned set is everything already in `env`, computed rather than
            # listed: a hand-maintained list is exactly how TS_MAX_CONCURRENCY,
            # TS_NUM_WARMUPS and TS_IGNORE_EOS came to be missing from it, and
            # any value added above from here on is covered without anyone
            # remembering to update a second place.
            #
            # Computing it is necessary but not sufficient, because for some keys
            # the workload's setting *is* absence. `max_concurrency` defaults to
            # unbounded, which it expresses by not setting TS_MAX_CONCURRENCY at
            # all -- so a computed-only set leaves the default configuration
            # unprotected while the configured one is guarded, and a mitigation
            # setting TS_MAX_CONCURRENCY=1 there runs the container capped while
            # `_aggregate` reports max_concurrency: None. That is the mislabelled
            # pass again, reached only on the default. Unioning the documented
            # floor closes it: those keys are reserved whether or not this run
            # happens to carry a value for them.
            # The secret names are reserved here too, and unconditionally rather
            # than only when a token is present. Everything in `env` is rendered
            # by `docker_env_flags` into `-e NAME=value` in the docker client's
            # argv, so a mitigation carrying HF_TOKEN would put a credential in
            # /proc/<pid>/cmdline for the life of the trial -- walking straight
            # past the by-name path that exists to keep it out of there.
            owned = set(env) | _PROTOCOL_ENV_KEYS | self._secret_env_names()
            collisions = sorted(set(trial_env) & owned)
            if collisions:
                raise ValueError(
                    "tokenspeed_serve: mitigation(s) for this cell set "
                    f"{', '.join(collisions)}, which this workload sets itself "
                    "as part of its contract with ts_bench_serve.sh. Overriding "
                    "them here would desynchronise the host's expectations, its "
                    "audit, and its reported configuration from the run that "
                    "actually happened. Set the corresponding workload_config "
                    "field instead (e.g. num_prompts, steps, request_rate, "
                    "max_concurrency)."
                )
            env.update(trial_env)
        return env

    def _validate_drain_timeout(self) -> None:
        """An explicit ``--drain-timeout`` still has to fit inside the grace.

        The script derives the drain from ``teardown_grace_sec`` precisely so
        teardown cannot SIGKILL a gateway mid-drain -- that is the delayed
        VRAM release this workload spends a poll loop waiting out. But
        ``--drain-timeout`` is a *serve* flag, so it is not in the bench-flag
        guard, and passing it in ``serve_args`` replaced the derived value with
        an unchecked one: ``60`` against the default 45s grace put every
        teardown back in the failure the derivation exists to avoid.

        Rejected here as well as in the script so a recipe that cannot work
        fails before it takes a node, and mirrors the bound the container
        enforces.
        """
        flag = "--drain-timeout"

        def is_flag(word: str) -> bool:
            # Any spelling argparse resolves to this option, abbreviations
            # included: `--drain` bypassed an exact-match test entirely, so the
            # derived value was appended, the caller's won as the last
            # occurrence, and the bound was never applied. Ambiguity cannot be
            # judged without the server's option list, so any long-option prefix
            # counts -- the same fail-closed choice the bench-flag guard makes.
            name = word.split("=", 1)[0]
            return name.startswith("--") and len(name) > 2 and flag.startswith(name)

        value: str | None = None
        expect_value = False
        for arg in self._serve_args:
            if expect_value:
                # An option where a value should be means the earlier
                # `--drain-timeout` never got one. Accepting it and moving on
                # validated a later occurrence while the malformed one still
                # reached the server, which then failed during startup and was
                # classified as a slow or broken engine -- after taking a node.
                if arg.startswith("-"):
                    raise ValueError(
                        "tokenspeed_serve: serve_args has --drain-timeout "
                        f"followed by {arg!r}, which is another option rather "
                        "than a value."
                    )
                value = arg
                expect_value = False
            elif is_flag(arg):
                if "=" in arg:
                    value = arg.split("=", 1)[1]
                else:
                    expect_value = True
        if expect_value:
            raise ValueError(
                "tokenspeed_serve: serve_args ends with --drain-timeout and no value"
            )
        if value is None:
            return
        # The same unsigned decimal the container's require_uint accepts, and
        # nothing else. `int()` also takes `"+5"`, `" 5 "`, `"5_0"` and the
        # zero-padded `"08"` -- all of which pass here and then exit 64 inside
        # the container, where the failure reads as a script problem on a recipe
        # the host had already approved. Rejected rather than normalized, for the
        # reason require_uint gives: `08` more likely means 8 than 010 octal, and
        # the recipe should say which.
        if not re.fullmatch(r"0|[1-9][0-9]*", value):
            raise ValueError(
                f"tokenspeed_serve: serve_args --drain-timeout ({value!r}) must "
                "be an integer number of seconds, written as plain digits with "
                "no sign, whitespace or leading zero -- the in-container check "
                "rejects those spellings, so accepting them here would only "
                "move the failure into the container."
            )
        drain = int(value)
        if not 1 <= drain < self._teardown_grace:
            raise ValueError(
                f"tokenspeed_serve: serve_args --drain-timeout ({drain}) must be "
                f"at least 1 and less than teardown_grace_sec "
                f"({self._teardown_grace}). Teardown sends SIGTERM and waits "
                f"{self._teardown_grace}s before SIGKILL, so a drain that long "
                "or longer is cut off partway -- which strands GPU memory into "
                "the next trial, the failure the derived default avoids. Raise "
                "teardown_grace_sec if the gateway genuinely needs longer."
            )

    def _assert_trusted_work_root(self) -> None:
        """Refuse a shared scratch root owned by another unprivileged user.

        The root is created ``1777`` so every user of a node can make their own
        ``u<uid>`` beneath it, and the sticky bit is what stops one user removing
        another's entry -- with one exception that matters here: the *owner* of a
        sticky directory may rename or remove anything in it regardless of who
        owns it. Whoever ran first owns the root, so a co-tenant who got there
        first could swap this trial's scratch for a directory of their own after
        :meth:`_assert_own_scratch` had already approved it, and the mounts, the
        exports and the audit that reads them would follow.

        Trusted therefore means a root this run does not have to take a
        stranger's word for: root-owned, because an administrator provisioned it,
        or owned by this uid. Anything else is refused with the remedy, rather
        than used and hoped about -- the check that would otherwise "pass" is one
        whose result another user can change afterwards.
        """
        try:
            info = os.lstat(self._work_root)
        except OSError as exc:
            raise RuntimeError(
                f"tokenspeed_serve: cannot stat {self._work_root}: {exc}"
            ) from exc

        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(
                f"tokenspeed_serve: {self._work_root} is not a directory. The "
                "scratch root is refused rather than followed: on a shared node "
                "another user can create that name, and a symlink would send "
                "this trial's mounts and exports somewhere of their choosing."
            )
        if info.st_uid not in (0, os.getuid()):
            raise RuntimeError(
                f"tokenspeed_serve: {self._work_root} is owned by uid "
                f"{info.st_uid}, which is neither root nor this user "
                f"({os.getuid()}). The owner of a sticky directory can rename "
                "or remove entries in it whoever owns them, so that user could "
                "replace this trial's scratch directory after it has been "
                "checked. Set work_dir to a path you own (for example "
                f"{self._work_root}-$USER), or have an administrator create "
                f"{self._work_root} root-owned with mode 1777 so it is shareable "
                "without being anyone's to rearrange."
            )

    def _assert_own_scratch(self) -> None:
        """Refuse a per-uid scratch directory that is not ours.

        Making the root ``1777`` so every user can create their own subdirectory
        opened a hole of its own: the sticky bit restricts *deleting* another
        user's entry, not *creating* a name. A uid is trivially discoverable, so
        a co-tenant could pre-create ``u<victim>`` as a symlink and have the
        victim's exports, HF writes and container mounts land wherever they
        chose -- or simply read them afterwards.

        Created with ``mkdir`` here rather than as part of the batch below,
        because the check is only meaningful against the directory this run will
        actually use: an existing entry must be a real directory, owned by this
        uid, and not group- or world-writable.
        """
        try:
            self._work_dir.mkdir(mode=0o700, exist_ok=True)
        except FileExistsError:
            pass
        except OSError as exc:
            raise RuntimeError(
                f"tokenspeed_serve: cannot create {self._work_dir}: {exc}"
            ) from exc

        try:
            info = os.lstat(self._work_dir)
        except OSError as exc:
            raise RuntimeError(
                f"tokenspeed_serve: cannot stat {self._work_dir}: {exc}"
            ) from exc

        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(
                f"tokenspeed_serve: {self._work_dir} is not a directory. The "
                "scratch root is shared and world-writable by design, so this "
                "path is refused rather than followed -- another user can create "
                "a name there, and a symlink would send this trial's exports and "
                "mounts somewhere of their choosing. Remove it, or set work_dir "
                "to a directory only you can write."
            )
        if info.st_uid != os.getuid():
            raise RuntimeError(
                f"tokenspeed_serve: {self._work_dir} is owned by uid "
                f"{info.st_uid}, not {os.getuid()}. Another user created this "
                "trial's scratch directory; refusing to write exports the audit "
                "will later read into it."
            )
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(
                f"tokenspeed_serve: {self._work_dir} is group- or world-writable "
                f"(mode {stat.S_IMODE(info.st_mode):04o}). Exports written there "
                "can be replaced between the run and the audit that reads them."
            )

    def _secret_env_names(self) -> set[str]:
        """Names that may only ever travel by reference, never by value.

        Reserved whether or not this run carries a token: the hazard is a
        *mitigation* or ``docker_args`` supplying one, which does not depend on
        the host having the variable set. ``hf_token_env`` is included as well
        as ``HF_TOKEN``, since a recipe pointing at another variable means that
        name is the credential on this node.
        """
        return {"HF_TOKEN", self._hf_token_env}

    def _secret_env(self) -> dict[str, str]:
        """Values forwarded by name rather than by value.

        Anything here is passed as a bare ``-e NAME``, so docker picks it up
        from its own environment instead of carrying it in argv. The distinction
        matters because ``/proc/<pid>/cmdline`` is world-readable while
        ``/proc/<pid>/environ`` is not, and the docker client lives for the
        whole trial.

        Read from the host environment under ``hf_token_env``, so the value
        never appears in a recipe either.
        """
        token = os.environ.get(self._hf_token_env)
        return {"HF_TOKEN": token} if token else {}

    def _docker_argv(self, env: dict[str, str]) -> list[str]:
        argv = [
            "docker",
            "run",
            "--rm",
            # Named so a timeout is recoverable: --rm only fires when the
            # container exits, and a host-side timeout kills the client rather
            # than the container. See _force_remove_container.
            "--name",
            self._container_name(),
            "--device",
            "/dev/kfd",
            "--device",
            "/dev/dri",
            # Both groups, matching host_launch.sh and harvest_code_objects.py.
            # Passing the device through is not sufficient under --user: render
            # nodes are commonly owned by `render` rather than `video`, so
            # `video` alone leaves /dev/dri/renderD* unopenable and the failure
            # surfaces as an unhelpful device or HIP init error.
            "--group-add",
            "video",
            "--group-add",
            "render",
            "--security-opt",
            "seccomp=unconfined",
            "--network",
            self._network,
            # No `--ipc host`, which is what makes this take effect. Docker
            # applies ShmSize only when it creates the /dev/shm mount itself:
            # under host IPC the container gets the host's mount and --shm-size
            # is silently ignored, so `shm_size` was a setting that did nothing
            # and the 16g-vs-256g comparison in docs/tokenspeed-serving.md
            # compared one host mount with itself.
            #
            # Nothing needed host IPC to begin with. Every TokenSpeed process --
            # orchestrator, engine, scheduler, gateway -- is forked inside this
            # one container and already shares its private IPC namespace, which
            # is the same reasoning harvest_code_objects.py records for its own
            # --shm-size. Host IPC would additionally expose node-wide shared
            # memory and semaphores to a third-party image for no gain.
            "--shm-size",
            self._shm_size,
            "-v",
            f"{self._scripts_dir}:/ts-scripts:ro",
            "-v",
            f"{self._out_dir}:/ts-out",
            "-v",
            f"{self._hf_home}:/hf-cache",
        ]
        if self._staged_dataset is not None:
            # The staged copy, not the path the recipe named: see
            # _stage_dataset. Read-only, because the dataset defines what was
            # measured and a run that could rewrite it would make its own
            # results unfalsifiable.
            argv += ["-v", f"{self._staged_dataset}:{_CONTAINER_DATASET_PATH}:ro"]
        if self._run_as_current_user:
            # Without this the container writes the HF cache and the exported
            # JSON as root, and the next run (or the caller's cleanup) cannot
            # delete them -- the exact EPERM the Phase 1 harvest path hit.
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        argv += docker_env_flags(env)
        # Passed by name, with no value: docker reads it from its own
        # environment, which we populate when spawning the client. That keeps
        # the token out of the argv, where it would be world-readable via
        # /proc/<pid>/cmdline, and confines it to /proc/<pid>/environ, which is
        # readable only by the owning uid and root.
        for name in self._secret_env():
            argv += ["-e", name]
        self._reject_owned_docker_args(owned_env=set(env) | self._secret_env_names())
        argv += self._docker_args
        argv += [
            "--entrypoint",
            "bash",
            self._image,
            # The digest-named copy this trial staged and syntax-checked, not
            # the generic name -- which a concurrent run from the same uid could
            # have replaced in between.
            f"/ts-scripts/{self._staged_script.name}",
        ]
        return argv

    # docker takes the last occurrence of most repeated options, and
    # ``docker_args`` is spliced in after the generated ones -- so an extra flag
    # does not merely add to the invocation, it replaces what this class relies
    # on. The consequences do not look like configuration errors:
    #
    #   --name other   the container runs under a name _force_remove_container
    #                  does not know, so a timed-out trial leaks a live
    #                  TokenSpeed holding the GPU -- silently undoing the
    #                  orphan-cleanup guarantee this class advertises
    #   -v ...:/ts-out the exports land somewhere the host does not glob, so a
    #                  completed run reports no results
    #   --entrypoint   the bench script never runs, and the trial fails with
    #                  whatever the replacement did instead
    #   -e TS_RUN_TOKEN
    #                  the same desynchronisation the mitigation guard rejects,
    #                  arriving by a route that bypasses it entirely
    #
    # Rejected rather than reordered, for the reason ts_bench_serve.sh rejects
    # its own owned flags: reordering would make the setting silently ineffective
    # instead of silently destructive, which is not much better.
    #   --ipc / --shm-size
    #                  TokenSpeed's scheduler sizes its shared memory against
    #                  these; a smaller later value fails at load time with an
    #                  error about shared memory, not about docker_args
    #   --rm=false     every completed container is left behind, so the node
    #                  fills up over a sweep rather than in any single trial
    _OWNED_DOCKER_FLAGS = frozenset(
        {
            "--name",
            "--entrypoint",
            "-v",
            "--volume",
            "--mount",
            "--env-file",
            "--network",
            "--net",
            "--user",
            "-u",
            "--ipc",
            "--shm-size",
            "--rm",
            "--device",
            "--group-add",
            "--security-opt",
            # Detaching breaks every assumption after the `docker run` call:
            # it returns 0 immediately, so the trial reports success with no
            # exports while the container keeps benchmarking and holding the
            # GPU -- and because nothing raised or timed out, no cleanup path
            # runs. This workload needs an attached client to supervise the
            # lifecycle at all.
            "--detach",
            "-d",
        }
    )

    # `-d` also travels inside a combined short cluster (`-dit`), which no amount
    # of splitting on `=` would reveal.
    _OWNED_SHORT_FLAG_LETTERS = frozenset({"d"})

    # A short cluster holds boolean flags only up to the first option that takes
    # a value: in `-emodel=x` the `e` consumes the remainder, so the letters
    # after it are a *value*, not flags. Scanning the whole word for owned
    # letters read that value as a cluster and rejected `-emodel=...` -- a legal
    # env-var spelling -- claiming it set `-d`. These are docker's value-taking
    # short options; a letter outside the set is boolean and keeps the scan going.
    _SHORT_OPTS_WITH_VALUE = frozenset("evuplwmach")

    def _docker_option(self, arg: str) -> tuple[str, str | None]:
        """The option a ``docker_args`` token sets, and any value attached to it.

        One parse for every spelling docker accepts, because reading the token
        three ways -- prefix match, split on ``=``, cluster scan -- left gaps
        between them that each admitted an owned option under a spelling none of
        the three claimed:

        * ``-e=NAME=value``. Docker takes the ``=`` as separator, so the value is
          ``NAME=value``; slicing from index 2 kept the ``=`` and extracted an
          empty name, and an empty name matches nothing, so the protocol guard
          waved ``-e=TS_MAX_CONCURRENCY=1`` through. The reported cap and the one
          the container ran under then disagreed.
        * ``-ieTS_RUN_TOKEN=x`` and ``-iv/tmp:/ts-out``. The value-taking option
          need not lead the cluster. The old scan stopped at ``e``/``v`` without
          looking at what followed, and the attached-prefix match only ever
          looked at position 0 -- so the same override that is refused spelled
          ``-e ...`` or ``-v ...`` was accepted with a boolean letter in front.

        Returning the option and its value together means each caller below sees
        the same token the same way. ``None`` for the value is the spaced form,
        where docker takes the next argument.
        """
        if arg.startswith("--"):
            flag, sep, value = arg.partition("=")
            return flag, (value if sep else None)
        if not arg.startswith("-") or len(arg) < 2:
            return arg, None
        for index, letter in enumerate(arg[1:], start=1):
            if not letter.isalpha():
                break
            if letter in self._SHORT_OPTS_WITH_VALUE:
                # The remainder is this option's value in either spelling docker
                # accepts: `-eNAME=v` and `-e=NAME=v` are the same thing, and the
                # single leading `=` is the separator, not part of the value.
                rest = arg[index + 1 :]
                if rest.startswith("="):
                    rest = rest[1:]
                return f"-{letter}", (rest or None)
            if letter in self._OWNED_SHORT_FLAG_LETTERS:
                return f"-{letter}", None
        return arg, None

    def _reject_owned_env(self, name: str, owned_env: set[str] | None) -> None:
        """Refuse an environment variable this workload sets itself.

        Unioned with the declared floor for the same reason the mitigation check
        is: ``owned_env`` can only name keys that are *present*, and an unbounded
        ``max_concurrency`` sets no TS_MAX_CONCURRENCY -- so ``docker_args:
        ["-e", "TS_MAX_CONCURRENCY=1"]`` on the default configuration ran the
        container capped while the host reported ``max_concurrency: None``.
        Fixing that for mitigations and not here just moved the same hole one
        field sideways.

        A secret name gets its own message: there is no workload_config field
        that carries a token, so pointing the caller at one would be advice that
        cannot be followed.
        """
        if not name:
            return
        if name in self._secret_env_names():
            raise ValueError(
                f"tokenspeed_serve: docker_args may not set {name}; it is a "
                "credential, and this workload already forwards it. `-e "
                "NAME=value` would put the value in the docker client's argv, "
                "which /proc/<pid>/cmdline exposes to every user on the node "
                "for the life of the trial. Export it in the environment you "
                "run aorta from, where it is forwarded by name instead, and "
                "use hf_token_env to name a different variable."
            )
        if name in (owned_env or set()) | _PROTOCOL_ENV_KEYS:
            raise ValueError(
                f"tokenspeed_serve: docker_args may not set {name}; this "
                "workload sets it as part of its contract with "
                "ts_bench_serve.sh, and overriding it here would bypass the "
                "same check that rejects it in a mitigation. Set the "
                "corresponding workload_config field instead."
            )

    def _reject_owned_docker_args(self, *, owned_env: set[str] | None = None) -> None:
        """Refuse ``docker_args`` that would displace a generated option."""
        expect_env_value = False
        for arg in self._docker_args:
            if expect_env_value:
                # This token is the preceding `-e`'s value, so it is data even if
                # it reads like a flag: docker would set a variable named `--name`
                # rather than renaming the container.
                expect_env_value = False
                self._reject_owned_env(arg.split("=", 1)[0], owned_env)
                continue

            flag, value = self._docker_option(arg)
            if flag in self._OWNED_DOCKER_FLAGS:
                raise ValueError(
                    f"tokenspeed_serve: docker_args may not set {flag}; this "
                    "workload sets it and depends on the value (container "
                    "naming for orphan cleanup, the mount layout the audit "
                    "reads, and the entrypoint). Use the corresponding "
                    "workload_config field -- network, run_as_current_user, "
                    "work_dir, hf_home -- instead."
                )
            # Only a name the workload sets is a problem; anything else is a
            # legitimate knob, which is what docker_args exists to pass.
            if flag in ("-e", "--env"):
                if value is None:
                    expect_env_value = True
                    continue
                self._reject_owned_env(value.split("=", 1)[0], owned_env)

    def run(self) -> WorkloadResult:
        # Token-qualify this trial's exports. ``$$`` inside the container is
        # always 1 under a fresh PID namespace, so the container cannot mint a
        # distinct tag itself; every trial in a matrix would write the same
        # filename and only the last would survive.
        self._run_token = f"{os.getpid()}-{time.monotonic_ns()}"
        # Resolved per trial, not in setup(): with `--network host` the gateway
        # binds on the host, so a port held by an unrelated process (or by the
        # previous trial's server still draining) must not fail the cell.
        # The gateway avoids an explicitly configured control port. Resolving it
        # blind meant the mixed case -- `port: auto` with an explicit
        # `control_port` -- could hand the gateway that very port, and the
        # equality check downstream would then reject a configuration that was
        # entirely valid. Most likely precisely when the explicit value sits in
        # the ephemeral range, which is where someone picking "a high free port"
        # would put it.
        reserved: set[int] = set()
        if not isinstance(self._control_port_request, str):
            reserved.add(int(self._control_port_request))
        self._port = _resolve_port(self._port_request, avoid=reserved)
        self._control_port = _resolve_port(
            self._control_port_request, avoid={self._port}, near=self._port + 1
        )
        env = self._container_env()
        argv = self._docker_argv(env)
        # Log the image and token, never the argv: docker_env_flags embeds raw
        # env values, which may include an HF token.
        log.info(
            "tokenspeed_serve: running %s model=%s steps=%d token=%s",
            self._image,
            self._model,
            self._steps,
            self._run_token,
        )

        vram_before = _vram_used_by_gpu()
        start = time.monotonic()
        timed_out = False
        # Every cleanup path runs *inside* the termination context, not in an
        # `except` clause outside it. Cleanup takes over a minute in the worst
        # case -- the terminate grace, the pipe drain and `docker rm -f` -- and
        # with the handlers already restored a SIGTERM arriving in that window
        # killed the process outright and stranded the container, which is the
        # exact failure this context exists to prevent.
        with self._remove_container_on_termination():
            # One removal per run, taken in the `finally` below so it covers
            # every exit from this block. Retried only when the client could not
            # be confirmed reaped; see _force_remove_container.
            remove_attempts = 1
            try:
                # Popen rather than run(), so the signal handler has a client to
                # reap. Killing the client before removing the container is what
                # closes the race between the two: otherwise a signal arriving
                # while `docker run` is still starting removes nothing, and the
                # client goes on to create the container after we have exited.
                with self._docker_client_lock:
                    self._docker_client_spawning = True
                    try:
                        self._docker_client = subprocess.Popen(
                            argv,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            # Carries the values referenced by the bare `-e NAME`
                            # flags in argv. Inherits the rest, since docker needs
                            # its own environment (DOCKER_HOST, PATH) to work at
                            # all.
                            env={**os.environ, **self._secret_env()},
                        )
                    finally:
                        self._docker_client_spawning = False
                client = self._docker_client
                stdout, stderr = client.communicate(timeout=self._timeout)
                exit_code = client.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = None
                # `communicate(timeout=)` leaves the client running, so it is
                # killed here and drained -- which also yields whatever it had
                # written.
                #
                # Killing the client does not stop the container: it belongs to
                # the daemon, and `--rm` only fires once it exits. Left alone, a
                # timed-out cell would hand the next one a live TokenSpeed still
                # holding the GPU and the gateway port -- so the next cell fails
                # too, for a reason that is nowhere in its own logs.
                if not self._terminate_docker_client():
                    # A terminate that could not confirm the client is gone
                    # leaves a process that may still create the container after
                    # a removal has reported "No such container" -- the same
                    # orphan race the signal path retries to close, reached from
                    # the timeout instead.
                    remove_attempts = _CLEANUP_REMOVE_ATTEMPTS
                stdout, stderr = self._drain_docker_client()
            except BaseException:
                # KeyboardInterrupt, and anything else raised out of the call.
                # The runner's SIGTERM does *not* arrive here -- see
                # _remove_container_on_termination, which is what handles it. The
                # original exception still propagates as the failure.
                #
                # This path can be entered *from inside* the Popen above, where
                # no client has been published yet and terminate cannot promise
                # one does not exist, so it takes the same retry.
                if not self._terminate_docker_client():
                    remove_attempts = _CLEANUP_REMOVE_ATTEMPTS
                raise
            finally:
                # Unconditionally, including after a client that exited on its
                # own. A completed `communicate()` proves the *client* is gone,
                # not the container: if the client lost its connection to the
                # daemon, or was OOM-killed, or the daemon restarted under it, it
                # returns a nonzero exit while the named container keeps serving
                # and holding the GPUs. Removal was reached only from the timeout
                # and exception paths, so that container survived into the next
                # cell of the sweep -- and `--rm` cannot help, because it fires
                # when the container exits, which is the thing that did not
                # happen. host_launch.sh takes the same unconditional approach on
                # EXIT.
                #
                # Cheap where there is nothing to do: `docker rm -f` on an
                # already-removed container is one call reporting "No such
                # container", logged at debug and treated as the ordinary case.
                self._force_remove_container(attempts=remove_attempts)
                with self._docker_client_lock:
                    self._docker_client = None
        container_elapsed = time.monotonic() - start

        records = self._collect_step_records()
        outstanding_vram, vram_attributed = self._await_vram_release(vram_before)
        # After the drain wait, not before it. That wait blocks this cell for up
        # to _VRAM_RELEASE_TIMEOUT_SEC and the GPU is unusable for the whole of
        # it, so stopping the clock at container exit reported a trial that
        # occupied the node for minutes longer than its own `elapsed_sec` -- and
        # a sweep's budget, summed from these, came out short by the drain time
        # of every cell. The container-only figure is still what benchmark
        # duration means, so it stays, as a metric.
        elapsed = time.monotonic() - start
        return self._build_result(
            records=records,
            exit_code=exit_code,
            timed_out=timed_out,
            elapsed=elapsed,
            container_elapsed=container_elapsed,
            stdout=stdout,
            stderr=stderr,
            leaked_vram=outstanding_vram,
            leaked_vram_attributed=vram_attributed,
        )

    def _owned_gpu_indices(self) -> set[int] | None:
        """GPUs this trial could have allocated on, or ``None`` for "any".

        ``hip_visible_devices`` narrows *which* devices the growth may be
        watched on -- the container could not have touched anything else. It says
        nothing about who else could: ``HIP_VISIBLE_DEVICES`` is a visibility
        filter on this process tree, not an allocation, so a co-tenant may be on
        the same physical GPU throughout. Whether the growth is this trial's to
        be blamed for is a separate question, answered by ``exclusive_gpus``.

        Entries that are not plain indices (a UUID form is also accepted by
        HIP) yield ``None`` rather than a partial set, since a partial set
        would silently narrow the check to whichever entries happened to parse.
        """
        raw = self._hip_visible_devices
        if raw is None:
            return None
        indices: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if not part.isdigit():
                return None
            indices.add(int(part))
        return indices or None

    def _await_vram_release(self, before: dict[int, int]) -> tuple[dict[int, int], bool]:
        """Block until the GPUs this trial used are actually free again.

        ``docker run`` returning is not the same event as the device memory
        coming back. With ``--tensor-parallel-size 2`` on gfx950, measured after
        a *passing* cell: the container is gone, nothing holds /dev/kfd, and one
        GPU still reports 256 GB of its 309 GB in use. It clears on its own after
        roughly 30-45s -- only rank 0's device is released promptly.

        Nothing in the trial that caused this looks wrong, which is what makes it
        worth handling here. aorta starts the next cell immediately, so the cost
        lands there instead: a tensor-parallel cell run straight after another
        one dies during startup with an out-of-memory error, naming a device it
        never chose and a model that fits comfortably. That is exactly how the
        `tp4` cell of `tokenspeed-serve-gptoss-tp.yaml` first failed.

        So this waits rather than reporting, and only reports what is still held
        once waiting has stopped helping.

        A before/after delta says memory grew; it does not say who allocated it.
        On a shared node another tenant starting a job mid-trial produces the
        same delta, and blaming it here would fail a healthy cell and tell the
        operator to reset somebody else's GPU. So growth is only *attributed*
        when something outside this workload says the GPUs were its own:
        ``exclusive_gpus: true``, which is the recipe author asserting a
        scheduler allocation or a dedicated machine.

        ``hip_visible_devices`` does not carry that meaning and no longer implies
        it. It is a visibility filter on this process tree, so it narrows which
        devices are *watched* -- the container could not have allocated elsewhere
        -- while leaving the same device open to a co-tenant. Reading it as an
        exclusive assignment made a healthy trial fail for somebody else's job.

        Without that evidence the wait still happens, because waiting helps the
        next cell whoever owns the memory, but the caller is told the result is
        unattributed and reports it as an observation rather than as this
        trial's failure.

        Returns ``(outstanding, attributed)``.
        """
        if not before:
            return {}, False

        owned = self._owned_gpu_indices()
        attributed = self._exclusive_gpus
        watched = before if owned is None else {i: b for i, b in before.items() if i in owned}
        if not watched:
            return {}, attributed

        deadline = time.monotonic() + _VRAM_RELEASE_TIMEOUT_SEC
        outstanding: dict[int, int] = {}
        while True:
            after = _vram_used_by_gpu()
            outstanding = {
                index: after[index] - used_before
                for index, used_before in watched.items()
                if index in after and after[index] - used_before >= _VRAM_LEAK_THRESHOLD_BYTES
            }
            if not outstanding or time.monotonic() >= deadline:
                break
            log.info(
                "tokenspeed_serve: waiting for %d GPU(s) to release memory after " "teardown (%s)",
                len(outstanding),
                ", ".join(
                    f"GPU {i}: {b / 1024**3:.0f} GiB" for i, b in sorted(outstanding.items())
                ),
            )
            time.sleep(_VRAM_RELEASE_POLL_SEC)
        if outstanding and not attributed:
            log.warning(
                "tokenspeed_serve: %d GPU(s) still hold memory after teardown (%s), but "
                "nothing establishes that this trial owned them exclusively, so the "
                "growth cannot be attributed to it -- a co-tenant's new allocation "
                "produces the same delta, even on a device HIP_VISIBLE_DEVICES "
                "restricted this container to. Not failing the trial; set "
                "exclusive_gpus: true when the node's GPUs really are this job's "
                "alone (a scheduler allocation, or a machine you have to yourself) "
                "to make this check authoritative.",
                len(outstanding),
                ", ".join(
                    f"GPU {i}: {b / 1024**3:.0f} GiB" for i, b in sorted(outstanding.items())
                ),
            )
        return outstanding, attributed

    def _missing_core_metrics(self, record: _StepRecord) -> list[str]:
        """Core metrics a measured step must actually carry.

        Auditing only ``completed``/``failed`` leaves a hole: an export holding
        just those two counters passes, and with the shipped recipes' empty gate
        set the cell goes green having measured nothing -- no duration, no TTFT,
        no throughput. A step that served every request but produced no numbers
        is not a successful measurement, it is an unusable export.

        Checked against the mean/median/throughput family rather than the
        percentiles, because those are emitted unconditionally while ``p50``,
        ``p90`` and ``p99`` depend on ``percentile_metrics`` and
        ``metric_percentiles`` -- requiring them here would fail a cell for a
        legitimate recipe choice.
        """
        required = [
            "duration",
            "output_throughput",
            "request_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
        ]
        # TPOT is an inter-token quantity, so it is only defined once a request
        # emits a second token. At one output token there are no intervals to
        # average and an absent or zero value is correct, not a fault.
        #
        # Which source answers that depends on whether the configuration
        # actually determines the output length. `random` does: the recipe pins
        # the length and EOS is ignored -- forced on by the bench CLI and
        # required to be so by validation above -- which holds it there, so
        # `output_len` is exact. The `_ignore_eos` half of the condition is
        # therefore redundant today and kept as the statement of what the branch
        # depends on.
        #
        # ShareGPT does not. It takes its lengths from the conversations and the
        # bench CLI never sees `output_len` at all, and with `ignore_eos: false`
        # -- which only ShareGPT can express -- the model stops whenever it emits
        # an EOS token, which for a short prompt can be immediately, so every
        # request may produce exactly one token. Deciding from `output_len` there
        # makes a step's validity hinge on a number the run never saw, and
        # rejects a correct export for not carrying a metric that was genuinely
        # undefined.
        #
        # The export is the only source that knows: more output tokens than
        # completed requests means at least one request emitted a second token.
        if self._dataset == _DEFAULT_DATASET and self._ignore_eos:
            tpot_defined = self._output_len > 1
        else:
            # Required rather than best-effort, so "we could not tell" is never
            # a reason to stop asking for TPOT.
            required.append("total_output_tokens")
            completed = record.doc.get("completed")
            total_output = record.doc.get("total_output_tokens")
            tpot_defined = (
                _is_scalar(total_output)
                and type(completed) is int
                and completed > 0
                and float(total_output) > completed
            )
        if tpot_defined:
            required += ["mean_tpot_ms", "median_tpot_ms"]

        # Every one of these must be strictly positive, not merely finite. A
        # step that served requests took time, produced tokens and had a
        # latency, so a zero or negative value is a broken measurement rather
        # than a slow one -- and a negative latency is worse than useless,
        # because it makes a `max_*` gate read as an improvement. A zero-length
        # step is the same problem seen from the other side: whatever throughput
        # it reported cannot have been measured.
        missing: list[str] = []
        for name in required:
            value = record.doc.get(name)
            if not _is_scalar(value) or float(value) <= 0:
                missing.append(name)
        return missing

    def _container_name(self) -> str:
        """The name this trial's container runs under.

        Derived from the run token, which is already per-trial, so two cells on
        one node never collide. Sanitised because docker only accepts
        ``[a-zA-Z0-9][a-zA-Z0-9_.-]*``.
        """
        suffix = re.sub(r"[^a-zA-Z0-9_.-]", "-", self._run_token)
        return f"aorta-ts-serve-{suffix}"

    @contextlib.contextmanager
    def _remove_container_on_termination(self):
        """Remove the container if this process is asked to terminate.

        An ``except BaseException`` around the ``docker run`` call does not cover
        SIGTERM: under Python's default disposition the interpreter terminates
        without raising anything, so no handler in the ``try`` block ever runs.
        SIGTERM is also precisely what the runner sends when a sweep is
        cancelled or a budget expires -- so the case most likely to strand a
        container was the one not actually covered.

        The container has to be removed by *someone*: it is daemon-owned, so it
        survives this process either way, and ``--rm`` only fires once it exits.
        A stranded TokenSpeed holds the GPU and the gateway port, and the next
        cell then fails for a reason recorded nowhere in its own logs.

        The handler removes the container, restores the previous disposition and
        re-raises the signal at itself, so the exit status still reports death by
        signal rather than a normal exit -- a supervisor reading 143 must keep
        reading 143.

        Signals can only be installed from the main thread; when called from a
        worker this degrades to the surrounding exception handling and says so,
        rather than raising and turning a cleanup nicety into a trial failure.
        """
        if threading.current_thread() is not threading.main_thread():
            log.debug(
                "tokenspeed_serve: not on the main thread; relying on exception "
                "handling for container cleanup"
            )
            yield
            return

        previous: dict[int, Any] = {}

        def handler(signum: int, _frame: Any) -> None:
            log.warning(
                "tokenspeed_serve: received signal %d; removing container %s " "before exiting",
                signum,
                self._container_name(),
            )
            # The client first, then the container. A single removal here raced
            # container creation: the handler is armed while `docker run` is
            # starting, so a signal arriving after the client was spawned but
            # before the daemon had created the named container got "No such
            # container", killed Python, and left the client alive to finish
            # creating it -- an orphan holding the GPU, produced by the cleanup
            # path. Reaping the client closes the window, because nothing is
            # left that could still create the container.
            #
            # Except in one window: a signal arriving while `Popen` itself is
            # running finds no client object to reap, because the object is only
            # assigned once the call returns. The removal is retried there, so a
            # container created moments later by a process this one cannot name
            # is still caught.
            reaped = self._terminate_docker_client()
            self._force_remove_container(
                attempts=1 if reaped else _CLEANUP_REMOVE_ATTEMPTS,
            )
            # SIG_DFL, not the previous disposition. Restoring the previous one
            # first looks tidier but does not guarantee the process dies:
            # under `nohup` SIGHUP is inherited as SIG_IGN, so the re-sent
            # signal was ignored and this handler simply returned into the
            # benchmark -- with the container already removed underneath it.
            # A supervisor then saw neither death by signal nor a usable run.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        # SIGHUP as well: a sweep started from a shell that goes away is the same
        # orphan by a different route.
        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                previous[sig] = signal.signal(sig, handler)
            except (OSError, ValueError):  # pragma: no cover - platform-specific
                pass
        try:
            yield
        finally:
            for sig, prev in previous.items():
                try:
                    signal.signal(sig, prev)
                except (OSError, ValueError):  # pragma: no cover
                    pass

    def _terminate_docker_client(self) -> bool:
        """Stop the ``docker run`` client and reap it.

        Ordered before container removal on every cleanup path, because the
        client is the only thing that can still *create* the container. Removing
        first and killing second left a window -- signal arrives, removal
        reports "No such container" because the daemon has not made it yet, and
        the surviving client then creates it after this process is gone.

        Best-effort and silent about a client that has already exited, which is
        the common case: this runs on paths where the run has usually finished
        or failed on its own.

        Returns whether the caller can be sure no client survives: either none
        was published *and* none was mid-spawn, or the published one is now
        reaped. It is ``False`` only in the narrow window where ``Popen`` has
        been entered but has not yet returned the object to assign -- there a
        process may exist that this one cannot name, so a single removal is not
        enough on its own.
        """
        with self._docker_client_lock:
            client = self._docker_client
            spawning = self._docker_client_spawning
        if client is None:
            return not spawning
        if client.poll() is not None:
            return True
        try:
            client.terminate()
            try:
                client.wait(timeout=_CLIENT_TERMINATE_GRACE_SEC)
            except subprocess.TimeoutExpired:
                client.kill()
                client.wait(timeout=_CLIENT_TERMINATE_GRACE_SEC)
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - defensive
            log.warning("tokenspeed_serve: could not stop the docker client: %s", exc)
            return False
        return True

    def _drain_docker_client(self) -> tuple[str, str]:
        """Whatever the client wrote before it was stopped."""
        with self._docker_client_lock:
            client = self._docker_client
        if client is None:
            return "", ""
        try:
            stdout, stderr = client.communicate(timeout=_CLIENT_TERMINATE_GRACE_SEC)
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
            return "", ""
        return _as_text(stdout), _as_text(stderr)

    def _force_remove_container(self, *, attempts: int = 1) -> None:
        """Stop and remove a container the docker client no longer supervises.

        ``docker rm -f`` covers the running and already-exited cases in one call,
        so there is no window between a stop and a remove. Best-effort and
        logged rather than raised: the caller is already reporting the real
        failure, and a cleanup error must not replace it with a less useful one.

        Reached from three directions: a host-side timeout, an exception out of
        the ``docker run`` call, and the signal handler installed by
        :meth:`_remove_container_on_termination` (SIGTERM does not raise, so the
        exception path alone would miss it). Nothing can help against SIGKILL,
        which no handler survives.

        ``attempts`` is for the one case a single removal cannot cover: a client
        that could not be reaped because it was mid-spawn may create the
        container *after* the first removal reported "No such container". Later
        passes catch that, and stop as soon as a pass actually removes something.
        """
        name = self._container_name()
        for attempt in range(1, max(attempts, 1) + 1):
            if self._remove_container_once(name) or attempt == attempts:
                return
            log.debug(
                "tokenspeed_serve: no container %s on attempt %d of %d; a client "
                "this process could not name may still be creating it",
                name,
                attempt,
                attempts,
            )
            time.sleep(_CLEANUP_REMOVE_RETRY_SEC)

    def _remove_container_once(self, name: str) -> bool:
        """One ``docker rm -f`` pass. True when it removed a container."""
        try:
            proc = subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("tokenspeed_serve: could not remove container %s: %s", name, exc)
            return False
        if proc.returncode == 0:
            log.info("tokenspeed_serve: removed orphaned container %s", name)
            return True
        # A nonzero exit is how docker reports most of what can go wrong here --
        # daemon unreachable, permission denied -- and none of it raises. Ignoring
        # it meant the one outcome worth knowing about, a container that is still
        # running and still holding the GPU, produced no output at all while the
        # docstring promised cleanup failures were logged.
        stderr = (proc.stderr or "").strip()
        if "No such container" in stderr:
            # Expected on the ordinary path: `--rm` already removed it, or the
            # trial failed before the container was ever created. Not a failure
            # to clean up, so not a warning -- there is nothing left to leak.
            log.debug("tokenspeed_serve: no container %s to remove", name)
            return False
        log.warning(
            "tokenspeed_serve: could not remove container %s (docker rm -f exited "
            "%d): %s -- it may still be running and holding the GPU; remove it "
            "with `docker rm -f %s` before the next run",
            name,
            proc.returncode,
            stderr or "(no stderr)",
            name,
        )
        return False

    def _collect_step_records(self) -> list[_StepRecord]:
        """Parse whatever the bench exported for this trial.

        Globbed rather than reconstructed from the step count so a partial run
        still yields the steps that did complete -- those are the data points
        that show *where* a degrading cell fell over.
        """
        records: list[_StepRecord] = []
        pattern = f"bench.{self._run_token}.step*.json"
        for path in sorted(self._out_dir.glob(pattern), key=lambda p: _step_index(p.name)):
            try:
                with path.open(encoding="utf-8") as fh:
                    doc = json.load(fh)
            # UnicodeDecodeError is neither an OSError nor a JSONDecodeError, so
            # a truncated or non-UTF-8 export used to propagate out of here and
            # abort the workload -- losing the steps that did parse, and
            # reporting a crash where the script's own result_json_unusable
            # verdict is the accurate one. A corrupt export is exactly the case
            # this handler exists for.
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                log.warning("tokenspeed_serve: unreadable export %s: %s", path, exc)
                continue
            if not isinstance(doc, dict):
                log.warning(
                    "tokenspeed_serve: export %s is %s, expected an object",
                    path,
                    type(doc).__name__,
                )
                continue
            scalars = {k: float(v) for k, v in doc.items() if _is_scalar(v)}
            records.append(
                _StepRecord(step=_step_index(path.name), path=path, doc=doc, scalars=scalars)
            )
        return records

    def _build_result(
        self,
        *,
        records: list[_StepRecord],
        exit_code: int | None,
        timed_out: bool,
        elapsed: float,
        container_elapsed: float | None = None,
        stdout: str,
        stderr: str,
        leaked_vram: dict[int, int] | None = None,
        leaked_vram_attributed: bool = False,
    ) -> WorkloadResult:
        failure_details: list[dict[str, Any]] = []

        # Computed here rather than beside the WorkloadResult, because the
        # incomplete-steps detail below has to stay inside this range: a `step`
        # is read back as an iteration index, and an index past the last
        # iteration is out of range wherever it is consumed.
        started_steps = len(_MEASURED_STEP_START_RE.findall(stdout))
        observed_steps = max(started_steps, len(records))

        # Only reached after waiting past the point where waiting helps -- see
        # _await_vram_release. Reported as a failure rather than a warning
        # because the alternative is a green cell that quietly breaks the next
        # one: nothing in this trial's own numbers looks wrong, and the cost
        # lands later as an out-of-memory error naming a device nobody chose.
        #
        # Only when the growth is this trial's to claim, though, which means
        # `exclusive_gpus: true` and nothing less. A before/after delta cannot
        # tell this workload's leftover memory from a co-tenant's new
        # allocation -- not even on a device HIP_VISIBLE_DEVICES restricted this
        # container to, since that filters what this process tree sees rather
        # than reserving anything. Failing the cell on the strength of it would
        # punish a healthy run for someone else's job, and point
        # `rocm-smi --gpureset` at their device. _await_vram_release logs the
        # unattributed case.
        for index, grown in sorted((leaked_vram or {}).items() if leaked_vram_attributed else []):
            failure_details.append(
                {
                    "reason": "gpu_memory_not_reclaimed",
                    "detail": (
                        f"GPU {index} still holds {grown / 1024**3:.1f} GiB more "
                        "than before this trial, "
                        f"{_VRAM_RELEASE_TIMEOUT_SEC}s after the container "
                        "exited. A tensor-parallel teardown normally clears in "
                        "30-45s; this did not. The next cell on this GPU will "
                        "fail during startup with an out-of-memory error that "
                        "does not mention tensor parallelism. Reclaim with "
                        f"`rocm-smi --gpureset -d {index}`."
                    ),
                    "gpu": index,
                    "leaked_bytes": grown,
                }
            )

        if timed_out:
            failure_details.append(
                {
                    "reason": "container_timeout",
                    "detail": (
                        f"docker run exceeded timeout_sec={self._timeout}; "
                        f"{len(records)}/{self._steps} bench steps exported"
                    ),
                }
            )
        elif exit_code != 0:
            failure_details.append(
                {
                    "reason": _EXIT_REASONS.get(exit_code or -1, "container_failed"),
                    "detail": f"ts_bench_serve.sh exited {exit_code}",
                    "exit_code": exit_code,
                    "stderr_tail": _tail(stderr),
                    # stdout as well, because that is where the script says what
                    # went wrong: every TS_BENCH_FAIL line is printed, not raised.
                    # Keeping only stderr was survivable while a failure also
                    # produced no records -- the `no_bench_export` branch below
                    # carries stdout -- but a later step failing after earlier
                    # ones exported leaves that branch unreached, and the trial
                    # then reported an exit code with the message that explains
                    # it thrown away.
                    "stdout_tail": _tail(stdout),
                }
            )

        if not records:
            failure_details.append(
                {
                    "reason": "no_bench_export",
                    "detail": (
                        "no parseable bench JSON for this trial; see stdout for "
                        "the bring-up phase that failed"
                    ),
                    "stdout_tail": _tail(stdout),
                }
            )
        elif len(records) < self._steps:
            exported = {record.step for record in records}
            # The earliest step with no export, one-based like every other
            # `step` here. Without it this detail named no step at all, so
            # _first_failure_step fell through to its "measured work ran, point
            # at 0" default and reported the first failure at step 1 even when
            # steps 1 and 2 had exported cleanly and step 3 was the one that
            # died. The index is the whole value of the field, and it is
            # knowable: the exports say which numbers arrived.
            missing = next(
                (step for step in range(1, self._steps + 1) if step not in exported),
                None,
            )
            detail: dict[str, Any] = {
                "reason": "incomplete_steps",
                "detail": (f"{len(records)} of {self._steps} bench steps exported a " "result"),
                # Same reasoning: which step stopped and why is on stdout.
                "stdout_tail": _tail(stdout),
            }
            if missing is not None:
                detail["detail"] += f"; earliest missing step {missing}"
                # The index is held inside the steps observed to run. When the
                # script announced the step it then died in -- the ordinary case
                # -- that is the missing step itself and nothing is capped. When
                # it died announcing nothing, the last step known to have run is
                # as far as an iteration index can honestly point, while the
                # text above still names the export that never arrived.
                if observed_steps:
                    detail["step"] = min(missing, observed_steps)
            failure_details.append(detail)

        # Re-audit the served-request counts here as well as in the script. The
        # script owns the fast verdict, but this class must not publish TTFT and
        # throughput computed from a run that dropped requests just because
        # someone changed an exit code -- the numbers would look plausible and
        # be wrong.
        for record in records:
            completed = record.doc.get("completed")
            failed = record.doc.get("failed")
            # `type(...) is not int` rather than `isinstance`, because `bool` is a
            # subclass of `int` in Python and `json` decodes `true`/`false` to it.
            # With `num_prompts: 1` -- a legitimate configuration --
            # `completed: true, failed: false` compares equal to 1 and 0, so an
            # export with meaningless counters would satisfy the audit outright.
            if type(completed) is not int or type(failed) is not int:
                failure_details.append(
                    {
                        "reason": "result_json_unusable",
                        "step": record.step,
                        "detail": (
                            f"completed={completed!r} failed={failed!r} in " f"{record.path.name}"
                        ),
                    }
                )
                continue
            # `failed != 0`, not `failed > 0`: the contract is that none failed,
            # and a negative count is not a run that did better than that -- it
            # is an export that cannot be believed. Reading it as "no failures"
            # would let `completed == num_prompts, failed == -1` through with the
            # metrics computed from whatever produced the -1.
            if failed != 0 or completed != self._num_prompts:
                failure_details.append(
                    {
                        "reason": "served_request_shortfall",
                        "step": record.step,
                        "detail": (
                            f"completed={completed} failed={failed} "
                            f"expected={self._num_prompts}"
                        ),
                    }
                )
            missing = self._missing_core_metrics(record)
            if missing:
                failure_details.append(
                    {
                        "reason": "result_json_unusable",
                        "step": record.step,
                        "detail": (
                            "export carries no usable measurement for "
                            f"{', '.join(missing)} in {record.path.name}"
                        ),
                    }
                )

        metrics = self._aggregate(records, stdout=stdout)
        if container_elapsed is not None:
            metrics["container_elapsed_sec"] = container_elapsed
        # Gates are only meaningful against a measurement. With no export there
        # is nothing to compare, and reporting `gate_metric_missing` here would
        # point the reader at percentile_metrics -- a recipe problem -- when the
        # actual story is the bring-up failure already recorded above.
        if records:
            failure_details.extend(self._check_gates(metrics))

        # `main_work_started` is about the measured phase *beginning*, not about
        # it succeeding -- see WorkloadResult in workloads/_base.py. A parsed
        # export is sufficient evidence but not necessary: a step that ran and
        # then wrote truncated or non-UTF-8 JSON is dropped by
        # _collect_step_records, and reporting did-not-run for it would classify
        # a benchmark failure as a setup failure and hide it from the triage
        # matrix entirely. So the script announces each measured step as it
        # starts, and either signal counts.
        #
        # With no export and no marker, nothing was measured -- the bring-up
        # failures, exits 50-52 and 64, are what land here -- and the matrix
        # correctly reports did-not-run rather than folding a non-measurement in
        # as a data point.
        main_work_started = bool(records) or started_steps > 0

        step_times_ms = [
            record.scalars["duration"] * 1000.0
            for record in records
            if "duration" in record.scalars
        ]

        passed = not failure_details
        return WorkloadResult(
            passed=passed,
            failure_count=len(failure_details),
            first_failure_iteration=(
                _first_failure_step(failure_details, main_work_started=main_work_started)
                if failure_details
                else None
            ),
            failure_details=failure_details,
            # Steps the script announced starting, which is at least the number
            # that produced a parseable export. The two differ exactly when a
            # step ran and its export was corrupt or missing, and in that case
            # `total_iterations=len(records)` was 0 while
            # `first_failure_iteration` was 0 -- an index outside its own range.
            # `executed_iterations` stays on parsed records: that is the count
            # of steps that produced a result, which is what a consumer
            # averaging step_times_ms needs.
            total_iterations=observed_steps,
            step_times_ms=step_times_ms,
            elapsed_sec=elapsed,
            metrics=metrics,
            main_work_started=main_work_started,
            executed_iterations=len(records),
            configured_iterations=self._steps,
        )

    def _aggregate(self, records: list[_StepRecord], *, stdout: str) -> dict[str, Any]:
        """Mean each scalar across steps; keep per-step detail alongside.

        Scalars are discovered from the export rather than listed here so a
        recipe changing ``metric_percentiles`` (which renames the ``p*`` keys)
        does not silently drop its own metrics.
        """
        metrics: dict[str, Any] = {
            "image": self._image,
            "model": self._model,
            "served_model_name": self._served_model_name,
            "num_prompts": self._num_prompts,
            "dataset": self._dataset,
            # Only for `random`, which is the one dataset these describe. Under
            # ShareGPT the lengths come from the conversations and the bench CLI
            # is never told these values, so publishing them labelled a result
            # with a shape it did not run -- and a matrix mixing the two
            # datasets would compare those labels as if they meant the same
            # thing. `None` reads as "not applicable" in the trial JSON and is
            # skipped by the perf aggregate.
            "input_len": self._input_len if self._dataset == _DEFAULT_DATASET else None,
            "output_len": self._output_len if self._dataset == _DEFAULT_DATASET else None,
            "max_concurrency": self._max_concurrency,
            "request_rate": self._request_rate,
            "num_warmups": self._num_warmups,
            "ignore_eos": self._ignore_eos,
            "bench_steps": self._steps,
            "warmup_steps": self._warmup_steps,
            # Strings, deliberately. The matrix aggregates every numeric metric
            # into `perf.md`, and a port rendered as a mean ("44,634.000")
            # occupies a column in the performance table while meaning nothing.
            # Identifiers belong in the trial JSON, not in a perf aggregate.
            "gateway_port": str(self._port),
            "control_port": str(self._control_port),
            # Recorded unconditionally, including on failure: the engine's own
            # log is where a bring-up failure explains itself, and a trial that
            # exported no metrics is exactly when someone needs to find it.
            "server_log": str(self._out_dir / f"bench-server.{self._run_token}.log"),
        }

        startup = _STARTUP_RE.search(stdout)
        if startup:
            metrics["server_startup_sec"] = float(startup.group(1))

        if not records:
            return metrics

        keys: list[str] = []
        for record in records:
            for key in record.scalars:
                if key not in keys and key not in _EXPORT_ECHO_KEYS:
                    keys.append(key)
        # A scalar aggregate is published only when *every* measured step
        # supplied the metric. Averaging over whichever steps happen to carry a
        # key silently changes what the number means: a `max_p99_tpot_ms` gate
        # over three steps, with `p99_tpot_ms` present in one, would evaluate
        # that single step's value while reading as a three-step aggregate --
        # and a gate passing on a third of the evidence is worse than one that
        # reports the metric missing, which is what the caller was promised.
        # Partial values stay visible in the per-step detail below.
        partial: list[str] = []
        for key in keys:
            values = [r.scalars[key] for r in records if key in r.scalars]
            if len(values) == len(records):
                metrics[key] = _mean(values)
            elif values:
                partial.append(key)
        if partial:
            # Named rather than dropped, so "the gate found no metric" can be
            # traced to the export that omitted it.
            metrics["partial_metrics"] = sorted(partial)

        # Sums, not means, for the audit counters: "how many requests did this
        # trial actually serve" is the question, and a mean hides a single bad
        # step among good ones.
        # `type(...) is int` for the same reason as the audit above: a JSON
        # boolean is an `int` to `isinstance`, and summing it would report
        # `completed_total: 1` for an export that never counted anything.
        metrics["completed_total"] = sum(
            r.doc["completed"] for r in records if type(r.doc.get("completed")) is int
        )
        metrics["failed_total"] = sum(
            r.doc["failed"] for r in records if type(r.doc.get("failed")) is int
        )

        # Alias to the name AORTA's CI gating allowlist already knows, so a
        # nightly baseline can gate serving throughput without the allowlist
        # having to learn TokenSpeed's spelling. Only this one aliases cleanly:
        # TTFT is not prefill latency (it includes queueing), so aliasing it to
        # ``prefill_latency_ms`` would gate a differently-defined quantity.
        if "output_throughput" in metrics:
            metrics["tokens_per_sec"] = metrics["output_throughput"]

        # Per-step detail lands in the trial JSON. The matrix aggregates only
        # scalars, so this list is carried without being summarised -- which is
        # what makes step-to-step variance recoverable after the fact.
        metrics["steps"] = [_step_detail(r) for r in records]
        metrics["result_files"] = [str(r.path) for r in records]
        return metrics

    def _check_gates(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for gate, bound in self._gates.items():
            metric_name, comparison = _GATE_SPECS[gate]
            observed = metrics.get(metric_name)
            if not _is_scalar(observed):
                failures.append(
                    {
                        "reason": "gate_metric_missing",
                        "detail": (
                            f"gate {gate} needs metric {metric_name}, which the "
                            "bench did not report (check percentile_metrics / "
                            "metric_percentiles)"
                        ),
                        "gate": gate,
                    }
                )
                continue
            breached = float(observed) > bound if comparison == "max" else float(observed) < bound
            if breached:
                failures.append(
                    {
                        "reason": "perf_gate_breached",
                        "detail": (
                            f"{metric_name}={float(observed):.4g} breaches " f"{gate}={bound:.4g}"
                        ),
                        "gate": gate,
                        "metric": metric_name,
                        "observed": float(observed),
                        "bound": bound,
                    }
                )
        return failures

    # -------------------------------------------------------------- cleanup

    def cleanup(self) -> None:
        # The container is ``--rm`` and tears its own server down on every path
        # that reaches here (the one path that does not -- a host timeout -- is
        # handled in run() via _force_remove_container), so cleanup is only
        # about the exported artifacts. Kept by default: they are the raw
        # evidence behind the metrics, and a failed trial's export is exactly
        # what someone will want to read.
        #
        # Read with getattr because the dispatcher calls cleanup() even when
        # setup() raised, and most of what setup() rejects is rejected before
        # this attribute is assigned. Direct access then raised AttributeError
        # out of cleanup, which the dispatcher logged as a cleanup warning
        # alongside the real configuration error -- two failures reported for
        # one cause, the second of them describing nothing that happened. The
        # default is keep, matching the field's own default: a run that never
        # started has nothing of its own to delete anyway.
        if getattr(self, "_keep_work_dir", True):
            return
        token = getattr(self, "_run_token", None)
        if not token:
            return
        for path in self._out_dir.glob(f"bench*.{token}.*"):
            try:
                path.unlink()
            except OSError as exc:
                log.warning("tokenspeed_serve: cannot remove %s: %s", path, exc)


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _resolve_port(request: Any, *, avoid: set[int] | None = None, near: int | None = None) -> int:
    """Resolve a ``port`` / ``control_port`` config value to a concrete port.

    An explicit request is honoured verbatim -- if an operator pinned a port,
    silently serving on a different one would make the run untraceable. ``auto``
    prefers ``near`` (so the control port keeps its conventional ``port + 1``
    relationship and logs stay readable) and otherwise asks the kernel for an
    ephemeral port.

    There is an unavoidable bind-check-then-use race here, but the alternative
    is a fixed port that collides with any other user of the node -- a race we
    would lose every time rather than rarely.
    """
    avoid = avoid or set()
    if request != "auto":
        return int(request)
    if near is not None and near not in avoid and 1 <= near <= 65535:
        if _port_is_free(near):
            return near
    # The ephemeral fallback has to respect `avoid` too. Resolving the control
    # port happens after the gateway's probe socket is already closed, so the
    # kernel is free to hand back that very port -- and the pair would then be
    # equal, which ts_bench_serve.sh rejects. A fully valid `auto` configuration
    # would fail as a usage error, intermittently, which is the worst way for
    # this to show up.
    #
    # Sockets are held open across the loop so a retry cannot be handed the port
    # the previous attempt just released, then all are closed together.
    held: list[socket.socket] = []
    try:
        for _ in range(20):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            held.append(sock)
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            if port not in avoid:
                return port
    finally:
        for sock in held:
            sock.close()
    raise RuntimeError(
        "tokenspeed_serve: could not find an ephemeral port outside "
        f"{sorted(avoid)} after 20 attempts"
    )


def _vram_used_by_gpu() -> dict[int, int]:
    """Bytes of VRAM in use per GPU, or ``{}`` if that cannot be determined.

    Read from the amdgpu sysfs nodes rather than by shelling out to rocm-smi:
    this runs on the hot path around every trial, sysfs needs no subprocess and
    no PATH assumption, and a node without the attribute simply drops out of the
    comparison instead of failing the trial. An empty result disables the leak
    check, which is the right default -- a missing measurement is not evidence of
    a leak.
    """
    cards: list[tuple[int, int]] = []
    for path in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")):
        try:
            card = int(path.parent.parent.name[len("card") :])
            cards.append((card, int(path.read_text().strip())))
        except (OSError, ValueError):
            continue
    # Keyed by position, not by the DRM card number. Those are not the indices
    # anyone reading the report has in mind: on this gfx950 node the cards
    # enumerate 0, 8, 16, ... , so reporting "GPU 8" would name a device that
    # does not exist as far as rocm-smi is concerned -- and the `--gpureset -d`
    # hint alongside it would be wrong.
    return {position: used for position, (_card, used) in enumerate(sorted(cards))}


def _host_username() -> str:
    """Best-effort host username for the container's ``USER``/``LOGNAME``.

    Only ever used to satisfy ``getpass.getuser()`` inside the container, so a
    placeholder is perfectly serviceable when the host itself cannot resolve a
    name for the uid (the same NSS gap this works around).
    """
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError):
        return f"aorta-{os.getuid()}"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _tail(text: str, limit: int = 2000) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def _step_index(name: str) -> int:
    match = re.search(r"\.step(\d+)\.json$", name)
    return int(match.group(1)) if match else 0


def _first_failure_step(
    failure_details: list[dict[str, Any]], *, main_work_started: bool
) -> int | None:
    """The earliest failing step, as a zero-based iteration index.

    ``step`` is one-based everywhere it is human-facing -- it comes from the
    export filename (``bench.<token>.step1.json``) and appears in log lines and
    failure details, where counting from one is what a reader expects. But
    ``WorkloadResult.first_failure_iteration`` is an *iteration index* and the
    rest of the codebase keeps it in ``0..total_iterations-1``, so returning the
    filename's number put a single-step failure at 1 with one iteration
    recorded: out of range, and off by one against every other workload.
    """
    steps = [d["step"] for d in failure_details if isinstance(d.get("step"), int)]
    if steps:
        return max(0, min(steps) - 1)
    # Not every failure names a step. A perf-gate breach is computed from the
    # aggregate across steps, and a bench step can fail before it identifies
    # itself. Returning None there says "no failure was observed", which
    # contradicts both the WorkloadResult contract (_base.py) and the
    # failure_details sitting beside it. Once measured work has run, index 0 is
    # the best-effort answer the rest of the codebase gives (hrx_perf.py), and
    # it is always in range because main_work_started implies an iteration.
    if main_work_started:
        return 0
    # Nothing ran, so there is genuinely no iteration to point at -- the
    # bring-up failures land here.
    return None


__all__ = ["TokenSpeedServeWorkload"]
