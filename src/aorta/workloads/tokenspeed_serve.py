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

import json
import logging
import math
import os
import re
import shutil
import socket
import subprocess
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

# Margin added on top of (readiness + steps x bench) when deriving the default
# host-side docker timeout: covers image pull, container start, and teardown.
_TIMEOUT_MARGIN_SEC = 600

_GPU_KFD_NODE = Path("/dev/kfd")

_STARTUP_RE = re.compile(r"^TS_BENCH_METRIC: server_startup_sec=(\d+)", re.MULTILINE)

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
_EXPORT_ECHO_KEYS = frozenset({"num_prompts", "max_concurrency", "burstiness"})

# The host/container contract. Every one of these is either used by this class
# after the run -- to locate exports, to audit served-request counts, to report
# the cell's configuration -- or determines where the container writes. A cell's
# mitigation overlay is rejected if it sets one, because the container would act
# on the new value while the host kept reasoning about the old one, and the
# resulting failure would read as an engine fault.
#
# Deliberately not a blanket "reject every TS_*": a mitigation setting a knob the
# workload does not itself depend on is legitimate, and this list is the set the
# workload actually reads back.
_PROTOCOL_ENV_KEYS = frozenset(
    {
        "TS_BENCH_STEPS",
        "TS_BENCH_WARMUP_STEPS",
        "TS_NUM_PROMPTS",
        "TS_INPUT_LEN",
        "TS_OUTPUT_LEN",
        "TS_MODEL",
        "TS_SERVED_MODEL_NAME",
        "TS_TOKENIZER",
        "TS_OUT_DIR",
        "TS_RUN_TOKEN",
        "TS_PORT",
        "TS_CONTROL_PORT",
        "TS_PERCENTILE_METRICS",
        "TS_METRIC_PERCENTILES",
        "TS_REQUEST_RATE",
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
    "max_median_e2el_ms": ("median_e2el_ms", "max"),
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
            (default ``True``).
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
            NFS home under root-squash cannot be bind-mounted.
        hf_home: HF cache directory (default ``<work_dir>/hf``). Persisting
            this across runs is what keeps a big model from re-downloading.
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
            (default: unset, i.e. all GPUs).
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
        steps = cfg.get("steps")
        self._steps = int(steps) if steps is not None else 1
        if self._steps < 1:
            raise ValueError(f"tokenspeed_serve: steps ({self._steps}) must be >= 1")

        max_conc = cfg.get("max_concurrency")
        if max_conc is None:
            self._max_concurrency: int | None = None
        else:
            self._max_concurrency = int(max_conc)
            if self._max_concurrency < 1:
                raise ValueError(
                    "tokenspeed_serve: max_concurrency "
                    f"({self._max_concurrency}) must be >= 1 (omit it for unbounded)"
                )

        self._request_rate = self._validated_request_rate(
            cfg.get("request_rate", _DEFAULT_REQUEST_RATE)
        )

        self._ignore_eos = self._bool("ignore_eos", True)
        self._run_as_current_user = self._bool("run_as_current_user", True)
        self._keep_work_dir = self._bool("keep_work_dir", True)

        self._percentile_metrics = str(cfg.get("percentile_metrics") or _DEFAULT_PERCENTILE_METRICS)
        self._metric_percentiles = str(cfg.get("metric_percentiles") or _DEFAULT_METRIC_PERCENTILES)
        self._validate_percentiles(self._metric_percentiles)

        self._ready_timeout = self._positive_int("ready_timeout_sec", _DEFAULT_READY_TIMEOUT_SEC)
        self._bench_timeout = self._positive_int("bench_timeout_sec", _DEFAULT_BENCH_TIMEOUT_SEC)
        self._teardown_grace = self._positive_int("teardown_grace_sec", _DEFAULT_TEARDOWN_GRACE_SEC)
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
        self._bench_args = self._arg_list("bench_args")
        self._docker_args = self._arg_list("docker_args")

        self._shm_size = str(cfg.get("shm_size") or _DEFAULT_SHM_SIZE)
        hip_devices = cfg.get("hip_visible_devices")
        self._hip_visible_devices = None if hip_devices is None else str(hip_devices)

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
            if not isinstance(value, str) and not 1 <= int(value) <= 65535:
                raise ValueError(f"tokenspeed_serve: {label} ({value!r}) must be in 1..65535")

        work_dir = cfg.get("work_dir") or _DEFAULT_WORK_DIR
        self._work_dir = Path(str(work_dir)).resolve()
        hf_home = cfg.get("hf_home")
        self._hf_home = Path(str(hf_home)).resolve() if hf_home else self._work_dir / "hf"
        self._hf_token_env = str(cfg.get("hf_token_env") or "HF_TOKEN")

        self._gates = self._validated_gates()

    def _positive_int(self, key: str, default: int) -> int:
        value = int(self.config.get(key, default))
        if value < 1:
            raise ValueError(f"tokenspeed_serve: {key} ({value}) must be >= 1")
        return value

    def _non_negative_int(self, key: str, default: int) -> int:
        value = int(self.config.get(key, default))
        if value < 0:
            raise ValueError(f"tokenspeed_serve: {key} ({value}) must be >= 0")
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
        rate = float(value)
        if math.isinf(rate) and rate > 0:
            return "inf"
        if math.isnan(rate) or math.isinf(rate):
            raise ValueError(
                f"tokenspeed_serve: request_rate ({value!r}) is not a usable "
                'rate; use a positive number, or "inf" to submit every request '
                "at once"
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
            return value.split()
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise ValueError(
            f"tokenspeed_serve: {key} must be a string or list, " f"got {type(value).__name__}"
        )

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
            bound = float(value)
            if not math.isfinite(bound) or bound <= 0:
                raise ValueError(
                    f"tokenspeed_serve: gate {key!r} bound ({value!r}) must be "
                    "a finite positive number"
                )
            gates[key] = bound
        return gates

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
        staged = self._scripts_dir / _BENCH_SCRIPT
        shutil.copyfile(source, staged)
        staged.chmod(0o755)
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

    # ------------------------------------------------------------------ run

    def _container_env(self) -> dict[str, str]:
        """Env for the container: TS_* knobs, then the cell's mitigations.

        Mitigations are applied last so a cell can override a default knob, and
        because they are the whole point of the matrix -- a mitigation silently
        losing to a workload default would make two cells identical while
        reporting them as different.
        """
        env: dict[str, str] = {
            "TS_MODEL": self._model,
            "TS_SERVED_MODEL_NAME": self._served_model_name,
            "TS_TOKENIZER": self._tokenizer,
            "TS_BENCH_STEPS": str(self._steps),
            "TS_BENCH_WARMUP_STEPS": str(self._warmup_steps),
            "TS_GATEWAY_STARTUP_TIMEOUT": str(self._gateway_startup_timeout),
            "TS_NUM_PROMPTS": str(self._num_prompts),
            "TS_INPUT_LEN": str(self._input_len),
            "TS_OUTPUT_LEN": str(self._output_len),
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
        if self._max_concurrency is not None:
            env["TS_MAX_CONCURRENCY"] = str(self._max_concurrency)
        if self._serve_args:
            env["TS_SERVE_ARGS"] = " ".join(self._serve_args)
        if self._bench_args:
            env["TS_BENCH_ARGS"] = " ".join(self._bench_args)
        if self._hip_visible_devices is not None:
            env["HIP_VISIBLE_DEVICES"] = self._hip_visible_devices

        # Gated models need a token. Forwarded by name from the host env so the
        # value never appears in a recipe or in run artifacts.
        token = os.environ.get(self._hf_token_env)
        if token:
            env["HF_TOKEN"] = token
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
            # What it may not do is redefine the protocol between this host and
            # the container, because only the container would learn about it.
            # `TS_NUM_PROMPTS=999` would have the script request 999 while
            # `_build_result` still audited against the recipe's 32, so a
            # perfectly healthy cell would fail its served-request audit;
            # `TS_RUN_TOKEN` would have the script write exports under a name the
            # host's glob does not match, and the cell would fail for finding no
            # export at all. Both look like engine faults and are neither.
            collisions = sorted(set(trial_env) & _PROTOCOL_ENV_KEYS)
            if collisions:
                raise ValueError(
                    "tokenspeed_serve: mitigation(s) for this cell set "
                    f"{', '.join(collisions)}, which the workload owns as part "
                    "of its contract with ts_bench_serve.sh. Overriding them "
                    "here would desynchronise the host's expectations from the "
                    "run. Set the corresponding workload_config field instead "
                    "(e.g. num_prompts, steps, request_rate)."
                )
            env.update(trial_env)
        return env

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
            "--group-add",
            "video",
            "--security-opt",
            "seccomp=unconfined",
            "--ipc",
            "host",
            "--network",
            self._network,
            "--shm-size",
            self._shm_size,
            "-v",
            f"{self._scripts_dir}:/ts-scripts:ro",
            "-v",
            f"{self._out_dir}:/ts-out",
            "-v",
            f"{self._hf_home}:/hf-cache",
        ]
        if self._run_as_current_user:
            # Without this the container writes the HF cache and the exported
            # JSON as root, and the next run (or the caller's cleanup) cannot
            # delete them -- the exact EPERM the Phase 1 harvest path hit.
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        argv += docker_env_flags(env)
        argv += self._docker_args
        argv += [
            "--entrypoint",
            "bash",
            self._image,
            f"/ts-scripts/{_BENCH_SCRIPT}",
        ]
        return argv

    def run(self) -> WorkloadResult:
        # Token-qualify this trial's exports. ``$$`` inside the container is
        # always 1 under a fresh PID namespace, so the container cannot mint a
        # distinct tag itself; every trial in a matrix would write the same
        # filename and only the last would survive.
        self._run_token = f"{os.getpid()}-{time.monotonic_ns()}"
        # Resolved per trial, not in setup(): with `--network host` the gateway
        # binds on the host, so a port held by an unrelated process (or by the
        # previous trial's server still draining) must not fail the cell.
        self._port = _resolve_port(self._port_request)
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

        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr)
            exit_code = None
            # The timeout kills the docker *client*; the daemon keeps the
            # container running, and `--rm` only fires once it exits. Left alone,
            # a timed-out cell would hand the next one a live TokenSpeed still
            # holding the GPU and the gateway port -- so the next cell fails too,
            # for a reason that is nowhere in its own logs.
            self._force_remove_container()
        except BaseException:
            # Includes KeyboardInterrupt and the SIGTERM the runner sends before
            # escalating: same orphan, same cleanup, and the original exception
            # still propagates as the failure.
            self._force_remove_container()
            raise
        elapsed = time.monotonic() - start

        records = self._collect_step_records()
        return self._build_result(
            records=records,
            exit_code=exit_code,
            timed_out=timed_out,
            elapsed=elapsed,
            stdout=stdout,
            stderr=stderr,
        )

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
        # emits a second token. At output_len 1 there are no intervals to
        # average and an absent or zero value is correct, not a fault.
        if self._output_len > 1:
            required += ["mean_tpot_ms", "median_tpot_ms"]

        missing: list[str] = []
        for name in required:
            value = record.doc.get(name)
            if not _is_scalar(value):
                missing.append(name)
            elif name == "duration" and float(value) <= 0:
                # A zero-length step cannot have measured a throughput, so
                # whatever it reported is an artefact.
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

    def _force_remove_container(self) -> None:
        """Stop and remove a container the docker client no longer supervises.

        ``docker rm -f`` covers the running and already-exited cases in one call,
        so there is no window between a stop and a remove. Best-effort and
        logged rather than raised: the caller is already reporting the real
        failure, and a cleanup error must not replace it with a less useful one.

        This cannot help against SIGKILL, which no handler survives -- but aorta
        sends SIGTERM first, and that is the window this uses.
        """
        name = self._container_name()
        try:
            proc = subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("tokenspeed_serve: could not remove container %s: %s", name, exc)
            return
        if proc.returncode == 0:
            log.info("tokenspeed_serve: removed orphaned container %s", name)

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
            except (OSError, json.JSONDecodeError) as exc:
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
        stdout: str,
        stderr: str,
    ) -> WorkloadResult:
        failure_details: list[dict[str, Any]] = []

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
            failure_details.append(
                {
                    "reason": "incomplete_steps",
                    "detail": (f"{len(records)} of {self._steps} bench steps exported a " "result"),
                }
            )

        # Re-audit the served-request counts here as well as in the script. The
        # script owns the fast verdict, but this class must not publish TTFT and
        # throughput computed from a run that dropped requests just because
        # someone changed an exit code -- the numbers would look plausible and
        # be wrong.
        for record in records:
            completed = record.doc.get("completed")
            failed = record.doc.get("failed")
            if not isinstance(completed, int) or not isinstance(failed, int):
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
            if failed > 0 or completed != self._num_prompts:
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
        # Gates are only meaningful against a measurement. With no export there
        # is nothing to compare, and reporting `gate_metric_missing` here would
        # point the reader at percentile_metrics -- a recipe problem -- when the
        # actual story is the bring-up failure already recorded above.
        if records:
            failure_details.extend(self._check_gates(metrics))

        # An exported step means the benchmark measured something, even if it
        # later failed the served-request audit or a gate -- those are real
        # observations and belong in the matrix. No export means nothing was
        # measured (the bring-up failures, exits 50-52 and 64, are what usually
        # land here), which the matrix reports as did-not-run rather than
        # folding a non-measurement in as a data point.
        main_work_started = bool(records)

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
                _first_failure_step(failure_details) if failure_details else None
            ),
            failure_details=failure_details,
            total_iterations=len(records),
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
            "input_len": self._input_len,
            "output_len": self._output_len,
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
        for key in keys:
            values = [r.scalars[key] for r in records if key in r.scalars]
            if values:
                metrics[key] = _mean(values)

        # Sums, not means, for the audit counters: "how many requests did this
        # trial actually serve" is the question, and a mean hides a single bad
        # step among good ones.
        metrics["completed_total"] = sum(
            int(r.doc["completed"]) for r in records if isinstance(r.doc.get("completed"), int)
        )
        metrics["failed_total"] = sum(
            int(r.doc["failed"]) for r in records if isinstance(r.doc.get("failed"), int)
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
        metrics["steps"] = [{"step": r.scalars.get("step", r.step), **r.scalars} for r in records]
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
        if self._keep_work_dir:
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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    return port


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


def _first_failure_step(failure_details: list[dict[str, Any]]) -> int | None:
    steps = [d["step"] for d in failure_details if isinstance(d.get("step"), int)]
    return min(steps) if steps else None


__all__ = ["TokenSpeedServeWorkload"]
