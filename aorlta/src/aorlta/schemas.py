"""Records shared between components (design Sections 6.1 and 6.2).

Field names, types and frozenness are the doc's. Where the doc lists a field's allowed values in a
comment, the annotation is a Literal of exactly those values, so `from_dict` refuses anything else.
The JSON on disk uses the same field names: `to_dict` and `from_dict` convert, strictly (no unknown
keys, no missing keys, no coercion).

`LegJob` and `ControlSpec` are named by the CheckAdapter protocol in Section 6.2 but not specified
there; their shape is WP5's (the reward service and its adapters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .config import ConfigError, plain, strict_decode

Source = Literal["standin"]
LegState = Literal["clean", "finding", "fail", "unsettled", "skipped_by_plan", "not_run"]
# "standin-gpu" adds aorta ConSan/ASan on a GPU to standin-static (C2.T11, and stand-in runs).
CheckProfile = Literal["standin-static", "standin-gpu", "fake"]
EndReason = Literal[
    "compiled",
    "no_change",  # a parse that finds the kernel left as shown (an explicit no-change answer, or edits that change
    # nothing): nothing is compiled
    "no_code_block",
    "turns_exhausted",
    "context_limit",
    "quarantined",
    "engine_error",
    "logprob_error",
]
SETTLED_STATES = frozenset({"clean", "finding", "fail"})


class SchemaError(ValueError):
    """A record does not match its schema."""


def to_dict(record: Any) -> Any:
    return plain(record)


def from_dict(cls: type, data: Any) -> Any:
    try:
        return strict_decode(cls, data, cls.__name__)
    except ConfigError as exc:
        raise SchemaError(str(exc)) from None


# --- Section 6.1: rollout ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskRef:  # everything the reward depends on besides the source
    contract_id: str
    parent_id: str | None
    parent_healthy: bool | None  # for broke_healthy
    reference_kind: Literal["frozen", "self_written"] | None  # the reward's +1.0 vs +0.5
    task_sha256: str  # hash of the task files the checks read


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    turn0_ids: list[int] | None  # preferred: the prompt source's exact token ids
    messages: list[dict] | None  # [{"role", "content"}], only if ids are not supplied
    prompt_ids_sha256: str  # of the ids actually sent at turn 0
    task: TaskRef
    source: Source


@dataclass
class CompileResult:
    ok: bool
    rc: int
    seconds: float
    errors: list[dict]  # [{"cls", "line", "col", "token"}], structured, before any text is built
    raw_stderr_sha256: str  # raw text kept only in the run's sealed evidence store


@dataclass
class TurnRecord:
    turn: int  # 0, 1, 2
    seed: int
    input_ids_sha256: str
    input_len: int
    response_ids: list[int]  # includes the stop token when the engine stopped on it
    response_logprobs: list[float]  # engine, fp32 log-softmax of raw logits (temperature 1), A3
    finish_reason: dict  # {"type": "stop"|"length", ...} as returned
    cached_tokens: int
    ideal_cached_tokens: int  # the longest prefix this engine has already seen
    t_submit: float
    t_done: float
    compile: CompileResult | None
    feedback_sha256: str | None  # feedback appended after this turn, if any
    leak_check: Literal["pass", "fail", "not_run"]


@dataclass
class ChainRecord:
    chain_id: str
    prompt_id: str
    k: int
    engine_id: int
    policy_version: str
    turns: list[TurnRecord]
    final_turn: int
    end_reason: EndReason


@dataclass(frozen=True)
class TrainingSample:
    """What the trainer receives for one chain: the final turn's exact input ids, its response ids
    and engine log-probs, the chain id, and the group key (iteration, prompt_id)."""

    chain_id: str
    iteration: int
    prompt_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float]

    @property
    def group_key(self) -> tuple[int, str]:
        return (self.iteration, self.prompt_id)


# --- Section 6.2: reward ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    candidate_id: str  # = chain_id
    run_id: str
    iteration: int
    prompt_id: str
    k: int
    policy_version: str
    response_text: str  # the final answer, untouched: the parse gate reads this
    response_tokens: int  # for the length term
    max_new_tokens: int
    task: TaskRef
    check_profile: CheckProfile
    rollout_compile: CompileResult | None  # a hint only; reused only if settings are proven identical


@dataclass
class LegResult:
    leg: str  # "gate.<name>" (incl. "gate.compile"), "ref.native", "ref.edge", "san.consan", "san.waitcheck", "san.asan"
    state: LegState
    reason: str | None  # e.g. "timeout", "worker_died", "control_dirty", "adapter_error", "wrong_output"
    findings: list[dict]  # adapter-native, sanitised (no source text)
    repeats: int  # settled repeats (the k-clean rule)
    attempts: int
    worker_id: str | None
    bracket: dict | None  # {"before": control_run_id, "after": control_run_id, "clean": bool}
    cache_hit: bool
    t_queued: float
    t_start: float | None
    t_end: float | None


@dataclass
class RewardResult:
    candidate_id: str
    reward: float | None
    status: Literal["scored", "unscored", "quarantined"]
    reason: str | None  # why unscored
    parts: dict | None  # {"base", "broke_healthy", "length"}
    reward_fn: dict  # {"name", "version", "code_sha256"}
    legs: dict[str, LegResult]
    full_check: bool  # False if any required leg was skipped by a plan
    plan: dict | None  # the check plan (C3), always logged
    audit: bool  # C3 audit coin
    cache_key: str
    timings: dict  # per stage: queue, run, total


# --- the CheckAdapter protocol's job types (WP5) -----------------------------------------------------


@dataclass(frozen=True)
class LegJob:
    """One run of one leg: for a candidate, or for a trusted control (then `candidate_id` is None and
    `control_id` is set). Retries are new jobs with `attempt` + 1.

    Adapters that read files get them through `package` (the candidate's package directory),
    `inputs` (artifact name -> path, for example the parsed source or the compiled object) and
    `output_dir` (where this job writes its own artifacts). Fakes read only the hashes."""

    job_id: str
    leg: str
    candidate_id: str | None
    control_id: str | None
    source_sha256: str
    task_sha256: str
    attempt: int = 1
    worker_id: str | None = None
    leg_settings: dict[str, Any] = field(default_factory=dict)  # part of the cache key
    t_queued: float | None = None  # UTC wall time the job was queued
    package: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    output_dir: str | None = None


@dataclass(frozen=True)
class ControlSpec:
    """A trusted control an adapter offers for bracketing (Section 6.2, Trusted controls). A control
    run is clean only if its result is `clean`; `inputs` name the files a real control reads."""

    control_id: str
    leg: str
    source_sha256: str
    task_sha256: str
    inputs: dict[str, str] = field(default_factory=dict)
