"""What the orchestrator needs from the components it drives (design Section 6.6).

The orchestrator is written against these small protocols, not against the components' classes, so it can be
tested with fakes and wired to WP2 (rollout), WP5 (reward), WP4 (trainer) and WP3 (sync) by thin adapters:

- `RolloutPort` is WP2's `RolloutManager.run(prompts, iteration, policy_version, on_candidate)`, plus a chain-end
  callback, the generation deadline, and `in_flight()` (requests the engines hold, for the generation barrier).
- `RewardPort` is WP5's `RewardService` (`submit`, `drain`, `reopen`, `stats`), plus `resubmit`, which D7's retry
  needs: `submit` is idempotent by candidate id, so it cannot re-check a candidate's unsettled legs.
- `TrainerPort` is WP4's streaming path (`stream_begin`/`stream_add`/`stream_finish`), plus `discard` and state
  save/load for checkpoints. `add_group` reports the group's advantages, so episodes carry the trainer's numbers
  rather than a second computation of them.
- `SyncPort` is WP3's `publish` (merge -> write -> reload -> landing check), plus the start-of-run round trip and
  the resume restore.

Every record that crosses a port is one of the shared schemas (`aorlta.schemas`). Nothing here holds prompt,
feedback or source text beyond what those records already carry in memory (Section 6.10).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..ids import sha256_text
from ..schemas import Candidate, ChainRecord, PromptRecord, RewardResult, TrainingSample


class RunStopped(RuntimeError):
    """A condition under which Section 6.6 stops the run. `detail` holds ids, counts and timings only."""

    def __init__(self, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = dict(detail or {})


class BarrierViolation(RuntimeError):
    """A reload was attempted while the engines still held requests (Section 6.6 step 5, A7)."""


class Clock(Protocol):
    def now(self) -> float:  # monotonic seconds
        ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


class PromptPool(Protocol):
    source: str

    def records(self) -> Sequence[PromptRecord]:
        ...


@dataclass(frozen=True)
class ChainEnd:
    """One chain's end, as the rollout reports it: the record, the candidate handed to the reward service (None for
    `engine_error` and `quarantined` chains) and the training sample (None unless the chain may train)."""

    record: ChainRecord
    candidate: Candidate | None
    sample: TrainingSample | None


class RolloutPort(Protocol):
    def start(self, selftest_prompts: Sequence[PromptRecord]) -> dict:
        """Start the engines (read-back and self-test, Section 6.1); refuse to start on any failure."""
        ...

    def run(self, prompts: Sequence[PromptRecord], iteration: int, policy_version: str, *,
            on_candidate: Callable[[Candidate], Any], on_chain: Callable[[ChainEnd], Any],
            deadline: float) -> Any:
        """Generate the iteration's P x K chains. `on_candidate` is called the moment a chain's final answer exists;
        `on_chain` at each chain's end (an adapter may call it for every chain at generation end). Chains still
        running at `deadline` (in the shared clock) end `engine_error`. Returns an object with `chains`,
        `candidates`, `samples`, `metrics` and `stop_reasons` (WP2's RolloutResult), optionally `deadline_hit` and
        `outstanding` (chain ids still running at the deadline)."""
        ...

    def in_flight(self) -> int:
        """Requests the engines hold now (submitted and not answered, including self-test requests)."""
        ...

    def shutdown(self) -> None:
        ...


class RewardPort(Protocol):
    legs: tuple[str, ...]  # every leg a result can carry (the profile's): the score report counts each

    def submit(self, cand: Candidate) -> Future:
        ...

    def resubmit(self, cand: Candidate) -> Future:
        """Check this candidate's unsettled legs again, once (D7); settled legs are reused, not rerun."""
        ...

    def drain(self, deadline: float) -> Any:
        """Stop accepting; at `deadline` (shared clock) every leg still pending becomes unsettled(timeout). Returns an
        object with `outstanding_at_deadline` ([{candidate_id, leg, where}]) and `hit_deadline` (WP5's DrainReport)."""
        ...

    def reopen(self) -> None:
        ...

    def stats(self) -> Any:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class GroupMember:
    chain_id: str
    k: int
    record: ChainRecord
    sample: TrainingSample | None
    result: RewardResult | None  # None: the chain handed over no candidate (engine_error, quarantined)


@dataclass(frozen=True)
class GroupInput:
    """One complete group: every one of its K chains has ended and every candidate has its final result."""

    iteration: int
    prompt_id: str
    members: tuple[GroupMember, ...]  # in k order

    @property
    def chains(self) -> dict[str, ChainRecord]:
        return {m.chain_id: m.record for m in self.members}


@dataclass(frozen=True)
class GroupOutcome:
    """What the trainer made of one group (WP4's GroupPlan, reduced to what the records need)."""

    prompt_id: str
    scored: int
    mean_reward: float | None
    dropped: str | None  # e.g. "fewer_than_2_scored": the group trains nothing
    advantages: dict[str, float]  # chain_id -> advantage, for every scored member of a kept group
    trained: tuple[str, ...]  # chain ids whose samples were accumulated (before the step's gap gate)
    excluded: dict[str, str] = field(default_factory=dict)  # chain_id -> reason, for members that do not train


class TrainerPort(Protocol):
    def begin(self, iteration: int) -> None:
        ...

    def add_group(self, group: GroupInput) -> GroupOutcome:
        """Accumulate one complete group's gradient (streaming, Section 6.4)."""
        ...

    def finish(self) -> dict:
        """All-reduce, one optimizer step, the step checks. Returns the step report (WP4's: `loss`, `grad_norm`,
        `counts`, `excluded` (gap-gated chain ids), `gap`, `checks`, ...). Raises on a failed check, before
        anything is published."""
        ...

    def discard(self) -> None:
        """Drop the accumulated gradient (an iteration that is not trained)."""
        ...

    def save_state(self, directory: Path) -> None:
        """Write the LoRA latent, Adam moments and RNG states into `directory` (which exists and is empty)."""
        ...

    def load_state(self, directory: Path) -> None:
        ...

    def state_digest(self) -> str:
        """sha256 identifying the trained state (latent and optimizer), for checkpoints and tests."""
        ...


class SyncPort(Protocol):
    def round_trip(self) -> dict:
        """Perturb -> every probe changes -> restore -> every probe identical (Section 6.5 step 6)."""
        ...

    def restore(self, version: int) -> dict:
        """At resume: merge the trainer's (just loaded) latent, write it, reload every engine, landing check."""
        ...

    def publish(self, version: int, *, in_flight: Callable[[], int]) -> dict:
        """Merge -> write -> reload -> landing check for policy version `version`. Returns a report with `verdicts`
        ({engine: LANDED|NOOP}), `changed`, `coverage`, `changed_frac`, `seconds`, `slot`. Raises on
        NOT_LANDED or MISMATCH. `in_flight` must be 0 at the reload (the generation barrier)."""
        ...

    def held_slot(self) -> str | None:
        ...


class ExporterPort(Protocol):
    def export(self, version: int, merged_dir: Path, adapter_dir: Path) -> dict:
        """Write the eval checkpoint of policy version `version` (Section 6.7): a merged bf16 Hugging Face directory
        into `merged_dir` and the PEFT adapter (A, B in fp32) into `adapter_dir`. Both directories exist."""
        ...


def describe_error(exc: BaseException, source: str) -> str:
    """An exception for logs and run records. A component's message may carry text from the data it was handling,
    so for a run on any source but the stand-in's (Section 6.10 item 3) only the class name and a hash of the message
    are kept."""
    message = str(exc)
    if source == "standin":
        return f"{type(exc).__name__}: {message}"[:500]
    return f"{type(exc).__name__} (message sha256 {sha256_text(message)[:16]})"
