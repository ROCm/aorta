"""``inference`` workload: eval-only model execution (no backward / optimizer).

Unlike :mod:`aorta.workloads.training` (forward → loss → backward → optimizer
step) this workload runs *inference*: the model is put in ``eval()`` and driven
under ``torch.inference_mode()``. It measures prefill / decode latency and
throughput, checks logits/outputs for NaN/inf, optionally emits a stable logit
checksum for drift detection, and returns a :class:`WorkloadResult`. The recipe
selects the serving mode:

.. code-block:: yaml

    workload: inference
    workload_config:
      mode: offline_batch   # offline_batch|continuous_batch

Launch mode
-----------
``launch_mode = "single_process"`` with ``min_world_size = 1``: ``aorta run``
is invoked once, with no torchrun wrapper and no ``torch.distributed`` init.
A ``distributed: true`` recipe flag for multi-rank inference is intentionally
out of scope for this first public workload (issue #239).

Reuse
-----
Model topologies come from :mod:`aorta.models.repeated_block`
(``RepeatedBlockModel`` — dense for ``decoder_transformer`` /
``encoder_transformer``, top-1 MoE when ``num_experts > 1``); a small local MLP
covers the smallest lifecycle smoke. Synthetic token prompts are generated
inline with a seeded generator (no real prompts / tokenizers / weights).

KV cache
--------
``RepeatedBlockModel`` has no real paged attention. When ``serving.kv_cache``
is true the workload SIMULATES the prefill/decode split by running a full
forward for prefill then re-running with a length-1 input for each decode token
(tensor slicing) — enough to produce separate prefill and decode latency
numbers without implementing real KV caching. With ``kv_cache`` false the decode
step re-runs the full growing sequence each token.

Serving modes
-------------
* ``offline_batch`` (MVP): a fixed batch is prefilled once then decoded for
  ``generate_tokens`` tokens; ``--steps S`` repeats this for S request batches.
* ``continuous_batch``: a small, deterministic simulated-arrival scheduler over
  ``max_active_requests`` concurrent requests; each of ``S`` scheduler ticks
  admits arrivals, runs one batched decode forward, and retires finished
  requests. Public-safe and dependency-free.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from aorta.workloads._base import Workload, WorkloadResult

# torch (and the torch-dependent aorta helpers) are imported lazily so this
# module can be IMPORTED for workload discovery / registration in an
# environment without torch (e.g. a CLI venv that only drives docker-based
# runs). setup() raises a clear error if torch is unavailable at run time.
try:
    import torch
except Exception as exc:  # pragma: no cover - exercised only in torch-less envs
    _DTYPES: dict[str, "torch.dtype"] = {}
    _IMPORT_ERROR: Exception | None = exc
else:
    from aorta.instrumentation.checksum import tensor_checksum
    from aorta.instrumentation.determinism import enable_deterministic
    from aorta.models import BlockConfig, RepeatedBlockModel

    # Accept both the verbose recipe spellings and the short forms used by
    # llm_determinism, mapping to one torch dtype.
    _DTYPES = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    _IMPORT_ERROR = None

log = logging.getLogger(__name__)

_VALID_DTYPE_NAMES = ("bfloat16", "bf16", "float16", "fp16", "float32", "fp32")
_VALID_DEVICES = ("auto", "cuda", "cpu")
_VALID_MODES = ("offline_batch", "continuous_batch")
_VALID_MODEL_KINDS = ("mlp", "encoder_transformer", "decoder_transformer")
_VALID_ARRIVAL = ("fixed",)

# Autoregressive topologies run a prefill + token-by-token decode loop. The
# encoder topology runs a single (prefill-only) forward and has no decode phase.
_AUTOREGRESSIVE_KINDS = ("mlp", "decoder_transformer")


# --------------------------------------------------------------------------- #
# Typed config
# --------------------------------------------------------------------------- #
@dataclass
class RequestSpec:
    """Inference request shape."""

    batch_size: int = 4
    prompt_len: int = 128
    generate_tokens: int = 32

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RequestSpec":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        _reject_unknown(d, known, "request")
        spec = cls(**{k: int(v) for k, v in d.items() if k in known})
        if spec.batch_size < 1:
            raise ValueError(f"request.batch_size must be >= 1, got {spec.batch_size}")
        if spec.prompt_len < 1:
            raise ValueError(f"request.prompt_len must be >= 1, got {spec.prompt_len}")
        if spec.generate_tokens < 0:
            raise ValueError(f"request.generate_tokens must be >= 0, got {spec.generate_tokens}")
        return spec


@dataclass
class ModelSpec:
    """Model topology. ``kind`` selects which in-tree model is built."""

    kind: Literal["mlp", "encoder_transformer", "decoder_transformer"] = "decoder_transformer"
    hidden_size: int = 512
    num_layers: int = 4
    num_heads: int = 8
    ffn_size: int = 2048
    vocab_size: int = 32_000
    # The repeated-block MoE is top-1 only (no top_k); experts are driven purely
    # by ``num_experts`` (>1 selects the MoE path).
    num_experts: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        d = dict(d or {})
        moe = dict(d.pop("moe", {}) or {})
        known = set(cls.__dataclass_fields__)
        _reject_unknown(d, known, "model")
        # ``enabled`` is accepted (silently ignored) — the issue/#239 recipe
        # shape includes ``model.moe.enabled``; topology is driven by kind only.
        _reject_unknown(moe, {"num_experts", "enabled"}, "model.moe")
        _INT_FIELDS = {"hidden_size", "num_layers", "num_heads", "ffn_size", "vocab_size", "num_experts"}
        spec = cls(**{k: (int(v) if k in _INT_FIELDS else v) for k, v in d.items() if k in known})
        if "num_experts" in moe:
            spec.num_experts = int(moe["num_experts"])
        if spec.kind not in _VALID_MODEL_KINDS:
            raise ValueError(f"model.kind must be one of {list(_VALID_MODEL_KINDS)}, got {spec.kind!r}")
        if spec.num_layers < 1:
            raise ValueError(f"model.num_layers must be >= 1, got {spec.num_layers}")
        if spec.num_experts < 1:
            raise ValueError(f"model.num_experts must be >= 1, got {spec.num_experts}")
        if spec.kind in ("encoder_transformer", "decoder_transformer"):
            # Validate num_heads >= 1 before the modulo so a 0 fails cleanly as a
            # ValueError instead of raising ZeroDivisionError.
            if spec.num_heads < 1:
                raise ValueError(f"model.num_heads must be >= 1, got {spec.num_heads}")
            if spec.hidden_size % spec.num_heads != 0:
                raise ValueError(
                    f"model.hidden_size ({spec.hidden_size}) must be divisible by "
                    f"num_heads ({spec.num_heads})"
                )
        return spec


@dataclass
class ContinuousBatchSpec:
    enabled: bool = False
    max_active_requests: int = 8
    arrival_pattern: Literal["fixed"] = "fixed"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContinuousBatchSpec":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        _reject_unknown(d, known, "serving.continuous_batch")
        spec = cls(**{k: v for k, v in d.items() if k in known})
        spec.enabled = _require_bool(spec.enabled, "serving.continuous_batch.enabled")
        spec.max_active_requests = int(spec.max_active_requests)
        if spec.max_active_requests < 1:
            raise ValueError(
                f"serving.continuous_batch.max_active_requests must be >= 1, "
                f"got {spec.max_active_requests}"
            )
        if spec.arrival_pattern not in _VALID_ARRIVAL:
            raise ValueError(
                f"serving.continuous_batch.arrival_pattern must be one of "
                f"{list(_VALID_ARRIVAL)}, got {spec.arrival_pattern!r}"
            )
        return spec


@dataclass
class ServingSpec:
    # Simulated only — RepeatedBlockModel has no real paged KV cache. See the
    # module docstring for what kv_cache actually exercises.
    kv_cache: bool = True
    continuous_batch: ContinuousBatchSpec = field(default_factory=ContinuousBatchSpec)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ServingSpec":
        d = dict(d or {})
        _reject_unknown(d, {"kv_cache", "continuous_batch"}, "serving")
        cb = ContinuousBatchSpec.from_dict(d.get("continuous_batch", {}))
        kv = _require_bool(d.get("kv_cache", True), "serving.kv_cache")
        return cls(kv_cache=kv, continuous_batch=cb)


@dataclass
class ChecksSpec:
    fail_on_nan_logits: bool = True
    fail_on_nonfinite_output: bool = True
    compare_logits_checksum: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChecksSpec":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        _reject_unknown(d, known, "checks")
        return cls(**{k: _require_bool(v, f"checks.{k}") for k, v in d.items() if k in known})


@dataclass
class InferenceConfig:
    """Top-level recipe knobs for :class:`InferenceWorkload`."""

    mode: Literal["offline_batch", "continuous_batch"] = "offline_batch"
    seed: int = 1234
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: str = "bfloat16"
    warmup_steps: int = 1
    steps: int = 4
    request: RequestSpec = field(default_factory=RequestSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    serving: ServingSpec = field(default_factory=ServingSpec)
    checks: ChecksSpec = field(default_factory=ChecksSpec)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InferenceConfig":
        d = dict(d or {})
        request = RequestSpec.from_dict(d.get("request", {}))
        model = ModelSpec.from_dict(d.get("model", {}))
        serving = ServingSpec.from_dict(d.get("serving", {}))
        checks = ChecksSpec.from_dict(d.get("checks", {}))
        scalar_keys = {"mode", "seed", "device", "dtype", "warmup_steps", "steps"}
        _reject_unknown(
            d,
            scalar_keys | {"request", "model", "serving", "checks"},
            "inference",
            reserved=frozenset({"steps"}),
        )
        cfg = cls(
            request=request,
            model=model,
            serving=serving,
            checks=checks,
            **{k: d[k] for k in scalar_keys if k in d and d[k] is not None},
        )
        # A recipe may select continuous batching via either the top-level
        # ``mode`` or the nested ``serving.continuous_batch.enabled`` flag; keep
        # the two consistent so neither silently wins.
        if cfg.serving.continuous_batch.enabled and cfg.mode == "offline_batch":
            cfg.mode = "continuous_batch"
        if cfg.mode == "continuous_batch":
            cfg.serving.continuous_batch.enabled = True
        if cfg.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {list(_VALID_MODES)}, got {cfg.mode!r}")
        if cfg.device not in _VALID_DEVICES:
            raise ValueError(f"device must be one of {list(_VALID_DEVICES)}, got {cfg.device!r}")
        if cfg.dtype not in _VALID_DTYPE_NAMES:
            raise ValueError(f"dtype must be one of {list(_VALID_DTYPE_NAMES)}, got {cfg.dtype!r}")
        if cfg.steps < 1:
            raise ValueError(f"steps must be >= 1, got {cfg.steps}")
        if cfg.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {cfg.warmup_steps}")
        # The continuous-batch scheduler requires at least one decode step per
        # request; reject generate_tokens=0 here rather than silently bumping it.
        if cfg.mode == "continuous_batch" and cfg.request.generate_tokens < 1:
            raise ValueError(
                "request.generate_tokens must be >= 1 for mode=continuous_batch, "
                f"got {cfg.request.generate_tokens}"
            )
        # The continuous scheduler is a decode loop; it only makes sense for
        # autoregressive topologies (the non-autoregressive encoder has no decode
        # phase). Reject the mismatch rather than silently running decode ticks.
        if cfg.mode == "continuous_batch" and not cfg.is_autoregressive:
            raise ValueError(
                "mode=continuous_batch requires an autoregressive model.kind "
                f"(mlp|decoder_transformer), got {cfg.model.kind!r}"
            )
        return cfg

    @property
    def is_autoregressive(self) -> bool:
        return self.model.kind in _AUTOREGRESSIVE_KINDS


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_bool(value: Any, field: str) -> bool:
    """Validate that a recipe field is a real ``bool`` and fail fast otherwise.

    ``bool(...)`` coercion would turn a malformed value (e.g. the string
    ``"false"``) into ``True`` and silently flip behavior/safety checks. This
    mirrors the validate-and-fail-fast boolean handling in
    :mod:`aorta.workloads._subprocess`.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool, got {type(value).__name__}: {value!r}")
    return value


def _reject_unknown(
    d: dict[str, Any],
    known: set[str],
    section: str,
    *,
    reserved: frozenset[str] = frozenset(),
) -> None:
    """Raise ``ValueError`` for any key not in *known* (excluding reserved keys).

    Keys starting with ``_aorta_`` are platform-injected metadata and are
    always silently accepted. *reserved* holds additional exempted keys — used
    only at the top level to pass ``frozenset({"steps"})``, which the dispatcher
    injects into the top-level workload config. Nested sections (request, model,
    serving, checks) do NOT exempt ``steps`` so a typo like
    ``request.steps`` is correctly rejected.
    """
    unknown = sorted(
        k for k in d
        if k not in known and k not in reserved and not k.startswith("_aorta_")
    )
    if unknown:
        raise ValueError(f"unknown {section} key(s): {unknown}")


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (no numpy dependency at the base layer)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _build_mlp(spec: ModelSpec) -> "torch.nn.Module":
    """Smallest lifecycle-smoke topology: embedding → MLP → vocab head.

    Returns ``[B, T, vocab]`` logits so the inference step (argmax over the
    vocab) is identical across all model kinds.
    """
    from torch import nn

    class _MlpModel(nn.Module):
        def __init__(self, vocab: int, hidden: int, num_layers: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            layers: list[nn.Module] = []
            for _ in range(num_layers):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.GELU())
            self.mlp = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, vocab)

        def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
            return self.head(self.mlp(self.embed(input_ids)))

    return _MlpModel(spec.vocab_size, spec.hidden_size, spec.num_layers)


def _build_model(spec: ModelSpec, seq_len: int) -> "torch.nn.Module":
    """Build an unwrapped, eval-ready model for ``spec.kind``.

    ``encoder_transformer`` and ``decoder_transformer`` both map onto
    ``RepeatedBlockModel``. Caveat: that model always applies a CAUSAL
    attention mask, so ``encoder_transformer`` is not truly bidirectional —
    it reuses the same block for an embedding/classification-style single
    forward, per issue #239's "map onto RepeatedBlockModel" guidance.
    """
    if spec.kind == "mlp":
        return _build_mlp(spec)
    block = BlockConfig(
        vocab_size=spec.vocab_size,
        hidden_size=spec.hidden_size,
        ffn_size=spec.ffn_size,
        num_heads=spec.num_heads,
        num_layers=spec.num_layers,
        seq_len=seq_len,
        num_experts=spec.num_experts,
    )
    return RepeatedBlockModel(block)


class _ContinuousRequest:
    """One in-flight request for the continuous-batch scheduler.

    ``ctx`` is a fixed-width rolling context window so active requests can be
    stacked into a single batched decode forward regardless of how many tokens
    they have generated; ``remaining`` decrements once per decoded token.
    """

    __slots__ = ("ctx", "remaining")

    def __init__(self, ctx: "torch.Tensor", remaining: int) -> None:
        self.ctx = ctx
        self.remaining = remaining


# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #
class InferenceWorkload(Workload):
    """Eval-only inference with offline-batch / continuous-batch serving modes.

    ``launch_mode = "single_process"``: ``aorta run`` invokes it once, with no
    torchrun wrapper and no process group. CPU is acceptable for lifecycle /
    unit smoke; CUDA/ROCm is used when available and the result reports the
    actual device.
    """

    name: ClassVar[str] = "inference"
    launch_mode: ClassVar[Literal["single_process", "distributed"]] = "single_process"
    min_world_size: ClassVar[int] = 1

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cfg: InferenceConfig | None = None
        self._device = None
        self._dtype = None
        self._model = None
        self._parameter_count = 0
        self._input_gen = None

    # -- lifecycle ---------------------------------------------------------- #
    def setup(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "InferenceWorkload requires PyTorch, which failed to import: "
                f"{_IMPORT_ERROR!r}"
            )

        cfg = InferenceConfig.from_dict(self.config)
        # ``--steps`` from the dispatcher overrides the recipe/default count of
        # request batches (offline) / scheduler ticks (continuous).
        if self.config.get("steps") is not None:
            cfg.steps = int(self.config["steps"])
            if cfg.steps < 1:
                raise ValueError(f"steps must be >= 1, got {cfg.steps}")
        self._cfg = cfg

        self._device = self._resolve_device(cfg.device)
        self._dtype = _DTYPES[cfg.dtype]

        enable_deterministic(cfg.seed)
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
            torch.cuda.manual_seed(cfg.seed)
        # Deterministic per-run input stream (CPU generator → host-side randint).
        self._input_gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

        model = _build_model(cfg.model, cfg.request.prompt_len)
        self._parameter_count = sum(p.numel() for p in model.parameters())
        model = model.to(device=self._device, dtype=self._dtype)
        model.eval()
        self._model = model

        log.info(
            "InferenceWorkload setup: mode=%s model=%s device=%s dtype=%s params=%d "
            "batch=%d prompt_len=%d generate_tokens=%d kv_cache=%s steps=%d",
            cfg.mode,
            cfg.model.kind,
            self._device,
            cfg.dtype,
            self._parameter_count,
            cfg.request.batch_size,
            cfg.request.prompt_len,
            cfg.request.generate_tokens,
            cfg.serving.kv_cache,
            cfg.steps,
        )

    def run(self) -> WorkloadResult:
        assert self._cfg is not None
        if self._cfg.mode == "continuous_batch":
            return self._run_continuous()
        return self._run_offline()

    def cleanup(self) -> None:
        self._model = None
        self._input_gen = None

    # -- offline batch ------------------------------------------------------ #
    def _run_offline(self) -> WorkloadResult:
        assert self._cfg is not None
        cfg = self._cfg
        req = cfg.request

        t0 = time.perf_counter()
        step_times: list[float] = []
        prefill_times: list[float] = []
        decode_token_times: list[float] = []
        failures: list[dict[str, Any]] = []
        first_failure: int | None = None
        executed = 0
        main_work_started = False
        decoded_tokens = 0
        last_checksum: int | None = None

        # Warmup iterations are run but NOT recorded in the measured timings.
        for _ in range(cfg.warmup_steps):
            self._offline_batch(req, record=None)

        for step in range(cfg.steps):
            main_work_started = True
            rec: dict[str, list[float] | int | None] = {
                "prefill_ms": [],
                "decode_ms": [],
                "checksum": None,
            }
            step_t0 = time.perf_counter()
            # _offline_batch already synchronizes after every forward, so no
            # extra self._sync() is needed here (it would just double the sync).
            logits_finite, has_nan, has_inf = self._offline_batch(req, record=rec)
            step_times.append((time.perf_counter() - step_t0) * 1000.0)
            executed += 1

            prefill_times.extend(rec["prefill_ms"])  # type: ignore[arg-type]
            decode_token_times.extend(rec["decode_ms"])  # type: ignore[arg-type]
            # Count only decode-covered tokens (the first token came from prefill
            # and is not part of decode timing), so tokens_per_sec aligns with
            # decode_ms.
            decode_tokens_per_step = max(req.generate_tokens - 1, 0) if cfg.is_autoregressive else 0
            decoded_tokens += req.batch_size * decode_tokens_per_step
            last_checksum = rec["checksum"]  # type: ignore[assignment]

            problems = self._numeric_problems(cfg.checks, has_nan, has_inf, logits_finite)
            if problems:
                if first_failure is None:
                    first_failure = step
                failures.append({"step": step, "problems": problems})
                log.error("inference numeric check failed at step %d: %s", step, problems)

        elapsed = time.perf_counter() - t0
        passed = not failures
        # Warmup batches ran in a separate (unrecorded) loop, so every entry in
        # step_times is already a measured step — don't slice off warmup_steps.
        timed = step_times

        prefill_latency_ms = sum(prefill_times) / len(prefill_times) if prefill_times else 0.0
        decode_latency_ms = (
            sum(decode_token_times) / len(decode_token_times) if decode_token_times else None
        )
        total_decode_sec = sum(decode_token_times) / 1000.0
        if decode_token_times and total_decode_sec > 0:
            # Decode throughput: generated tokens per decode second.
            tokens_per_sec = decoded_tokens / total_decode_sec
        elif prefill_times:
            # No decode phase (encoder, or generate_tokens=0): report prefill
            # throughput as prompt tokens processed per prefill second.
            tokens_per_sec = (req.batch_size * req.prompt_len * len(prefill_times)) / (
                sum(prefill_times) / 1000.0
            )
        else:
            tokens_per_sec = 0.0

        metrics = self._base_metrics(cfg)
        metrics.update(
            {
                "prefill_latency_ms": prefill_latency_ms,
                "decode_latency_ms": decode_latency_ms,
                "tokens_per_sec": tokens_per_sec,
                "decoded_tokens": decoded_tokens,
                "step_time_p50": _percentile(timed, 50.0),
                "step_time_p99": _percentile(timed, 99.0),
            }
        )
        if cfg.checks.compare_logits_checksum and last_checksum is not None:
            metrics["logits_checksum"] = last_checksum

        log.info(
            "InferenceWorkload %s (offline_batch): %d batch(es), prefill=%.3fms "
            "decode/tok=%s tok/s=%.1f failures=%d",
            "PASSED" if passed else "FAILED",
            cfg.steps,
            prefill_latency_ms,
            f"{decode_latency_ms:.3f}ms" if decode_latency_ms is not None else "n/a",
            tokens_per_sec,
            len(failures),
        )

        return WorkloadResult(
            passed=passed,
            failure_count=len(failures),
            first_failure_iteration=first_failure,
            failure_details=failures,
            total_iterations=cfg.steps,
            step_times_ms=step_times,
            elapsed_sec=elapsed,
            main_work_started=main_work_started,
            executed_iterations=executed,
            configured_iterations=cfg.steps,
            metrics=metrics,
        )

    def _offline_batch(
        self, req: RequestSpec, *, record: dict | None
    ) -> tuple[bool, bool, bool]:
        """Run one request batch (prefill + optional decode loop).

        Returns ``(all_finite, has_nan, has_inf)`` aggregated over every logits
        tensor produced in the batch. When ``record`` is provided, per-phase
        latencies (ms) and the final-step logits checksum are appended to it.
        """
        assert self._cfg is not None
        cfg = self._cfg
        # Skip _finite_flags() (which forces a CUDA .item() sync) entirely when
        # no numeric check is enabled — avoids skewing latency/throughput metrics.
        checks_enabled = cfg.checks.fail_on_nan_logits or cfg.checks.fail_on_nonfinite_output
        all_finite = True
        any_nan = False
        any_inf = False
        last_logits = None

        with torch.inference_mode():
            prompt = self._make_prompt(req)
            # Prefill: full forward over the prompt.
            p0 = time.perf_counter()
            logits = self._model(prompt)
            self._sync()
            if record is not None:
                record["prefill_ms"].append((time.perf_counter() - p0) * 1000.0)
            if checks_enabled:
                fin, nan, inf = self._finite_flags(logits)
                all_finite &= fin
                any_nan |= nan
                any_inf |= inf
            last_logits = logits

            if cfg.is_autoregressive and req.generate_tokens > 0:
                seq = prompt
                next_tok = logits[:, -1:, :].argmax(dim=-1)
                # Prefill already produced the first generated token (next_tok),
                # so only generate_tokens-1 decode forwards remain. The non-kv
                # path is seeded with that first token on the initial iteration.
                for _ in range(req.generate_tokens - 1):
                    if cfg.serving.kv_cache:
                        # Simulated KV cache: feed only the newest token (the
                        # "shorter input" prefill/decode split). RepeatedBlockModel
                        # has no real cache, so this measures the decode-shape
                        # forward cost, not paged attention.
                        dec_in = next_tok
                    else:
                        seq = torch.cat([seq, next_tok], dim=1)
                        dec_in = seq
                    d0 = time.perf_counter()
                    step_logits = self._model(dec_in)
                    self._sync()
                    if record is not None:
                        record["decode_ms"].append((time.perf_counter() - d0) * 1000.0)
                    if checks_enabled:
                        fin, nan, inf = self._finite_flags(step_logits)
                        all_finite &= fin
                        any_nan |= nan
                        any_inf |= inf
                    next_tok = step_logits[:, -1:, :].argmax(dim=-1)
                    last_logits = step_logits

        # Only checksum when explicitly enabled — it forces a device sync and is
        # pure overhead when compare_logits_checksum is off.
        if (
            record is not None
            and last_logits is not None
            and cfg.checks.compare_logits_checksum
        ):
            record["checksum"] = tensor_checksum(last_logits)
        return all_finite, any_nan, any_inf

    # -- continuous batch --------------------------------------------------- #
    def _run_continuous(self) -> WorkloadResult:
        assert self._cfg is not None
        cfg = self._cfg
        req = cfg.request
        cap = cfg.serving.continuous_batch.max_active_requests

        # Build the full configured request set. ``generate_tokens`` is the
        # per-request decode budget (validated >= 1 for this mode).
        def _fresh_queue() -> list[_ContinuousRequest]:
            return [
                _ContinuousRequest(
                    self._make_prompt(RequestSpec(1, req.prompt_len, req.generate_tokens)),
                    req.generate_tokens,
                )
                for _ in range(req.batch_size)
            ]

        # Warmup runs on a SEPARATE scheduler state so it can't drain/retire
        # requests that the measured ticks need (which would corrupt
        # requests_completed / decoded_tokens / timings).
        if cfg.warmup_steps > 0:
            w_queue = _fresh_queue()
            w_active: list[_ContinuousRequest] = []
            for _ in range(cfg.warmup_steps):
                if not w_queue and not w_active:
                    break
                self._continuous_tick(w_active, w_queue, cap, record=None)

        queue: list[_ContinuousRequest] = _fresh_queue()
        active: list[_ContinuousRequest] = []

        t0 = time.perf_counter()
        step_times: list[float] = []
        decode_token_times: list[float] = []
        failures: list[dict[str, Any]] = []
        first_failure: int | None = None
        executed = 0
        main_work_started = False
        completed = 0
        decoded_tokens = 0
        last_checksum: int | None = None

        for step in range(cfg.steps):
            main_work_started = True
            if not active and not queue:
                # All requests drained before the configured tick budget; record
                # an idle tick so step accounting stays consistent.
                step_times.append(0.0)
                executed += 1
                continue

            rec: dict[str, Any] = {"decode_ms": [], "checksum": None, "retired": 0, "tokens": 0}
            step_t0 = time.perf_counter()
            # _continuous_tick already synchronizes after the forward, so an
            # extra self._sync() here would just add a redundant device sync.
            all_finite, has_nan, has_inf = self._continuous_tick(active, queue, cap, record=rec)
            step_times.append((time.perf_counter() - step_t0) * 1000.0)
            executed += 1
            decode_token_times.extend(rec["decode_ms"])
            completed += rec["retired"]
            decoded_tokens += rec["tokens"]
            last_checksum = rec["checksum"]

            problems = self._numeric_problems(cfg.checks, has_nan, has_inf, all_finite)
            if problems:
                if first_failure is None:
                    first_failure = step
                failures.append({"step": step, "problems": problems})
                log.error("inference numeric check failed at tick %d: %s", step, problems)

        # After all ticks: any requests still active or queued were not served
        # within the configured step budget. This would otherwise produce a
        # misleading passed=True with a low requests_completed count.
        remaining_active = len(active)
        remaining_queued = len(queue)
        remaining_requests = remaining_active + remaining_queued
        if remaining_requests:
            # Attribute to the last valid tick (cfg.steps - 1) so
            # first_failure_iteration stays within 0..total_iterations-1,
            # consistent with training.py and other workloads.
            last_tick = cfg.steps - 1
            if first_failure is None:
                first_failure = last_tick
            failures.append(
                {
                    "step": last_tick,
                    "problems": ["incomplete_requests"],
                    "requests_completed": completed,
                    "remaining_active": remaining_active,
                    "remaining_queued": remaining_queued,
                }
            )

        elapsed = time.perf_counter() - t0
        passed = not failures
        # Warmup ran on separate state, so every recorded tick is measured.
        timed = step_times
        decode_latency_ms = (
            sum(decode_token_times) / len(decode_token_times) if decode_token_times else None
        )
        total_decode_sec = sum(decode_token_times) / 1000.0
        tokens_per_sec = decoded_tokens / total_decode_sec if total_decode_sec > 0 else 0.0

        metrics = self._base_metrics(cfg)
        metrics.update(
            {
                "prefill_latency_ms": None,  # folded into per-tick decode in this sim.
                "decode_latency_ms": decode_latency_ms,
                "tokens_per_sec": tokens_per_sec,
                "decoded_tokens": decoded_tokens,
                "max_active_requests": cap,
                "requests_completed": completed,
                "step_time_p50": _percentile(timed, 50.0),
                "step_time_p99": _percentile(timed, 99.0),
            }
        )
        if cfg.checks.compare_logits_checksum and last_checksum is not None:
            metrics["logits_checksum"] = last_checksum

        log.info(
            "InferenceWorkload %s (continuous_batch): %d tick(s), completed=%d "
            "decode/tok=%s tok/s=%.1f failures=%d",
            "PASSED" if passed else "FAILED",
            cfg.steps,
            completed,
            f"{decode_latency_ms:.3f}ms" if decode_latency_ms is not None else "n/a",
            tokens_per_sec,
            len(failures),
        )

        return WorkloadResult(
            passed=passed,
            failure_count=len(failures),
            first_failure_iteration=first_failure,
            failure_details=failures,
            total_iterations=cfg.steps,
            step_times_ms=step_times,
            elapsed_sec=elapsed,
            main_work_started=main_work_started,
            executed_iterations=executed,
            configured_iterations=cfg.steps,
            metrics=metrics,
        )

    def _continuous_tick(self, active, queue, cap, *, record) -> tuple[bool, bool, bool]:
        """One scheduler tick: admit arrivals, batched decode, retire finished."""
        # Fixed arrival: admit AT MOST one queued request per tick (when below
        # capacity), matching arrival_pattern="fixed" in the recipe/docs.
        if queue and len(active) < cap:
            active.append(queue.pop(0))

        if not active:
            return True, False, False

        # Honor the kv_cache knob (mirrors offline): with simulated KV enabled,
        # decode from only the newest token; otherwise feed the full rolling
        # context window. The window is updated either way for the next tick.
        if self._cfg.serving.kv_cache:
            batch = torch.cat([r.ctx[:, -1:] for r in active], dim=0)
        else:
            batch = torch.cat([r.ctx for r in active], dim=0)
        d0 = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(batch)
        self._sync()
        if record is not None:
            record["decode_ms"].append((time.perf_counter() - d0) * 1000.0)
            record["tokens"] += len(active)
            if self._cfg.checks.compare_logits_checksum:
                record["checksum"] = tensor_checksum(logits)

        # Skip _finite_flags() (CUDA .item() sync) when no numeric check is
        # enabled so disabling checks removes the per-tick overhead entirely.
        checks_enabled = (
            self._cfg.checks.fail_on_nan_logits or self._cfg.checks.fail_on_nonfinite_output
        )
        fin, nan, inf = self._finite_flags(logits) if checks_enabled else (True, False, False)

        next_tok = logits[:, -1:, :].argmax(dim=-1)
        still_active = []
        for i, r in enumerate(active):
            # Roll the fixed-width window forward by the freshly decoded token.
            r.ctx = torch.cat([r.ctx[:, 1:], next_tok[i : i + 1]], dim=1)
            r.remaining -= 1
            if r.remaining > 0:
                still_active.append(r)
            elif record is not None:
                record["retired"] += 1
        active[:] = still_active
        return fin, nan, inf

    # -- internals ---------------------------------------------------------- #
    def _resolve_device(self, device_pref: str) -> "torch.device":
        if device_pref == "cpu":
            return torch.device("cpu")
        if device_pref == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but no CUDA/ROCm device is available")
        if device_pref == "cuda" or (device_pref == "auto" and torch.cuda.is_available()):
            # Bind an explicit device index: torch.cuda.set_device rejects an
            # index-less ``torch.device("cuda")`` on ROCm. LOCAL_RANK lets a
            # future launcher pin per-rank GPUs; default to the current device.
            idx = int(os.environ.get("LOCAL_RANK") or torch.cuda.current_device())
            return torch.device("cuda", idx)
        return torch.device("cpu")

    def _make_prompt(self, req: RequestSpec) -> "torch.Tensor":
        ids = torch.randint(
            0, self._cfg.model.vocab_size, (req.batch_size, req.prompt_len), generator=self._input_gen
        )
        return ids.to(self._device)

    def _sync(self) -> None:
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize()

    @staticmethod
    def _finite_flags(logits: "torch.Tensor") -> tuple[bool, bool, bool]:
        """Return ``(all_finite, has_nan, has_inf)`` for a logits tensor."""
        nan = bool(torch.isnan(logits).any().item())
        inf = bool(torch.isinf(logits).any().item())
        return (not nan and not inf), nan, inf

    @staticmethod
    def _numeric_problems(
        checks: ChecksSpec, has_nan: bool, has_inf: bool, all_finite: bool
    ) -> list[str]:
        problems: list[str] = []
        # ``fail_on_nan_logits`` covers both NaN and inf logits (the documented
        # "NaN/inf logit checks"); inf is reported explicitly so it is not
        # silently dropped when ``fail_on_nonfinite_output`` is disabled.
        if checks.fail_on_nan_logits and has_nan:
            problems.append("nan_logits")
        if checks.fail_on_nan_logits and has_inf:
            problems.append("inf_logits")
        if checks.fail_on_nonfinite_output and not all_finite:
            problems.append("non_finite_output")
        return problems

    def _base_metrics(self, cfg: InferenceConfig) -> dict[str, Any]:
        return {
            "mode": cfg.mode,
            "device": str(self._device),
            "dtype": cfg.dtype,
            "model_kind": cfg.model.kind,
            "parameter_count": self._parameter_count,
            "batch_size": cfg.request.batch_size,
            "prompt_len": cfg.request.prompt_len,
            "generate_tokens": cfg.request.generate_tokens,
            "kv_cache": cfg.serving.kv_cache,
        }


__all__ = [
    "InferenceWorkload",
    "InferenceConfig",
    "RequestSpec",
    "ModelSpec",
    "ServingSpec",
    "ContinuousBatchSpec",
    "ChecksSpec",
]
