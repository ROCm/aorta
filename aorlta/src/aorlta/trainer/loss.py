"""C4's objective (design Section 6.4): groups, advantages, normalisers and the per-token loss.

The group rules:
- A_i = r_i - mean(r_j over the scored samples of i's group), with no std division.
- A group with fewer than 2 scored samples is dropped.
- Unscored and quarantined samples never enter a group mean, a batch or a normaliser.

Every other knob is a parameter of 6.4's loss table.
The definitions follow Miles at `2f5fcd890` (`compute_policy_loss`, `vanilla_tis_function`, `compute_approx_kl`,
`get_sum_of_sample_mean`, `loss_function`), except where TokenSpeed forces a difference (TIS below).

Aggregation. For the samples S of one optimizer step, with n_i response tokens and per-token loss l_it:

    sample_mean        L = (1/N) sum_i (1/n_i) sum_t l_it          Miles' default (sum of sample means / batch)
    token_mean         L = sum_it l_it / sum_i n_i                  Miles' --calculate-per-token-loss
    prompt_token_mean  L = (1/G) sum_g sum_{i in g} sum_t l_it / sum_{i in g} n_i      ScaleRL's prompt level
    constant(L_max)    L = (1/(N L_max)) sum_it l_it                Dr. GRPO
    sample_sum         L = (1/N) sum_i sum_t l_it                   the old trainer

N is the number of trained samples, G the number of trained groups. Every form is c_i * l_it with
c_i = u_i / D: u_i (`sample_coefficient`) is known as soon as sample i's group is complete, and D (`normaliser`)
needs the whole step's totals. So the streaming path accumulates sum u_i l_it group by group and divides by D once
at the end, and the result is the batched loss exactly (C4.T1, C4.T7).

A sample the gap gate excludes keeps its reward in its group's mean and its counts in N, G and the token total;
it only loses its gradient (`keep = 0`). The normalisers therefore depend on which samples are scored, never on
the trainer's numerics, as the old trainer's `total = len(samples)` did.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch

from aorlta.config import REWARD_MAX, REWARD_MIN, TrainerSection
from aorlta.ids import IdError, parse_chain_id
from aorlta.rollout.engine_pool import MAX_LOGPROB
from aorlta.schemas import RewardResult, TrainingSample

PG_LOSSES = ("ppo_clip", "reinforce", "cispo")
AGGREGATIONS = ("sample_mean", "token_mean", "prompt_token_mean", "constant", "sample_sum")
GAP_STATISTICS = ("max_abs", "mean_abs", "median_abs", "p99_abs")
_CONSTANT_RE = re.compile(r"constant\(([1-9][0-9]*)\)")


class TrainerInputError(ValueError):
    """The trainer's inputs contradict each other; the run stops rather than guessing (Section 5, item 7)."""


# --- settings ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LossSettings:
    """The loss parameters of Section 6.4's table. `from_trainer` reads them from the run config."""

    pg_loss: str = "ppo_clip"
    eps_clip_low: float = 0.2
    eps_clip_high: float = 0.2
    old_logprobs: str = "trainer_recompute"
    tis: float | None = None  # off, or the cap C
    aggregation: str = "sample_mean"
    kl_beta: float = 0.0
    loss_temperature: float = 0.7
    steps_per_iteration: int = 1
    lm_head_fp32: bool = False
    max_generation_gap: float = 2.0
    max_off_policy_fraction: float = 0.02
    min_scored_per_group: int = 2
    gap_statistic: str = "max_abs"  # which SampleGap statistic the gate compares with max_generation_gap

    def __post_init__(self) -> None:
        if self.pg_loss not in PG_LOSSES:
            raise TrainerInputError(f"pg_loss: expected one of {PG_LOSSES}, got {self.pg_loss!r}")
        if self.old_logprobs not in ("trainer_recompute", "engine"):
            raise TrainerInputError(f"old_logprobs: expected trainer_recompute or engine, got {self.old_logprobs!r}")
        if not (math.isfinite(self.loss_temperature) and self.loss_temperature > 0):
            raise TrainerInputError(f"loss_temperature: must be finite and > 0, got {self.loss_temperature!r}")
        if self.old_logprobs == "engine" and self.loss_temperature != 1.0:
            # A3: TokenSpeed returns log-softmax of the raw logits, so its log-probs are pi_old only at T = 1 (D14).
            raise TrainerInputError(
                "old_logprobs 'engine' needs loss_temperature 1.0: the engine's log-probs are at temperature 1 (A3), "
                f"so they cannot serve as pi_old at T {self.loss_temperature}"
            )
        if not (0 < self.eps_clip_low and 0 < self.eps_clip_high):
            raise TrainerInputError("eps_clip_low and eps_clip_high must be > 0")
        if self.eps_clip_low > 1:
            raise TrainerInputError(f"eps_clip_low: the ratio's lower bound 1 - eps must be >= 0, got {self.eps_clip_low}")
        if self.tis is not None and not (math.isfinite(self.tis) and self.tis > 0):
            raise TrainerInputError(f"tis: the cap must be finite and > 0, got {self.tis!r}")
        if not (math.isfinite(self.kl_beta) and self.kl_beta >= 0):
            raise TrainerInputError(f"kl_beta: must be finite and >= 0, got {self.kl_beta!r}")
        if self.steps_per_iteration < 1:
            raise TrainerInputError(f"steps_per_iteration: must be >= 1, got {self.steps_per_iteration}")
        if not (math.isfinite(self.max_generation_gap) and self.max_generation_gap > 0):
            raise TrainerInputError(f"max_generation_gap: must be finite and > 0, got {self.max_generation_gap!r}")
        if not 0 <= self.max_off_policy_fraction <= 1:
            raise TrainerInputError(f"max_off_policy_fraction: must be in [0, 1], got {self.max_off_policy_fraction}")
        if self.min_scored_per_group < 2:
            raise TrainerInputError("min_scored_per_group: an advantage needs two scored samples")
        if self.gap_statistic not in GAP_STATISTICS:
            raise TrainerInputError(f"gap_statistic: expected one of {GAP_STATISTICS}, got {self.gap_statistic!r}")
        self.aggregation_kind()  # validates the aggregation string

    @classmethod
    def from_trainer(cls, trainer: TrainerSection, *, min_scored_per_group: int = 2) -> "LossSettings":
        if trainer.loss_temperature is None:
            raise TrainerInputError("trainer.loss_temperature is unset; load the run config so it takes the "
                                    "rollout temperature (Section 6.4, Numerics)")
        return cls(
            pg_loss=trainer.pg_loss, eps_clip_low=trainer.eps_clip_low, eps_clip_high=trainer.eps_clip_high,
            old_logprobs=trainer.old_logprobs, tis=trainer.tis, aggregation=trainer.aggregation,
            kl_beta=trainer.kl_beta, loss_temperature=trainer.loss_temperature,
            steps_per_iteration=trainer.steps_per_iteration, lm_head_fp32=trainer.lm_head_fp32,
            max_generation_gap=trainer.max_generation_gap, max_off_policy_fraction=trainer.max_off_policy_fraction,
            min_scored_per_group=min_scored_per_group, gap_statistic=trainer.gap_statistic,
        )

    def aggregation_kind(self) -> tuple[str, int | None]:
        """('constant', L_max) for constant(L_max), else (name, None)."""
        m = _CONSTANT_RE.fullmatch(self.aggregation)
        if m:
            return "constant", int(m.group(1))
        if self.aggregation in AGGREGATIONS and self.aggregation != "constant":
            return self.aggregation, None
        raise TrainerInputError(f"aggregation: expected one of sample_mean, token_mean, prompt_token_mean, "
                                f"constant(<L_max>), sample_sum; got {self.aggregation!r}")


# --- groups and advantages --------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnSpan:
    """Where turn t's reply sits in the final sequence (prompt + response): [start, start + length)."""

    turn: int
    start: int
    length: int
    engine_logprobs: tuple[float, ...]


@dataclass(frozen=True)
class TrainSample:
    """One sample the step trains on: the final turn's exact input ids and response, and its advantage.
    `turns` (from the chain record, when given) locates every turn's reply, the final one included; it is used
    for the gap by turn (C4.T10), never for the loss."""

    chain_id: str
    group: tuple[int, str]
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    engine_logprobs: tuple[float, ...]
    reward: float
    advantage: float
    turns: tuple[TurnSpan, ...] = ()

    @property
    def n_response(self) -> int:
        return len(self.response_ids)

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)

    def ids(self) -> list[int]:
        return [*self.prompt_ids, *self.response_ids]


@dataclass(frozen=True)
class GroupPlan:
    """One group of one iteration: who was scored, its mean reward, and the samples it trains."""

    key: tuple[int, str]
    members: int  # chains seen in the group (samples or results)
    scored: int
    mean_reward: float | None
    samples: tuple[TrainSample, ...]  # empty when the group is dropped
    dropped: str | None
    excluded: dict = field(default_factory=dict)  # reason -> count, for members that do not train

    @property
    def n_response_tokens(self) -> int:
        return sum(s.n_response for s in self.samples)


def _group_of(chain_id: str) -> tuple[int, str]:
    try:
        key = parse_chain_id(chain_id)
    except IdError as exc:
        raise TrainerInputError(f"{chain_id!r} is not a chain id: {exc}") from None
    return (key.iteration, key.prompt_id)


def _check_result(r: RewardResult) -> float | None:
    if r.status == "scored":
        ok = isinstance(r.reward, (int, float)) and not isinstance(r.reward, bool)
        if not ok or not math.isfinite(r.reward) or not REWARD_MIN <= r.reward <= REWARD_MAX:
            raise TrainerInputError(f"{r.candidate_id}: status 'scored' with reward {r.reward!r}, not a number in "
                                    f"[{REWARD_MIN}, {REWARD_MAX}]")
        return float(r.reward)
    if r.status in ("unscored", "quarantined"):
        if r.reward is not None:
            raise TrainerInputError(f"{r.candidate_id}: status {r.status!r} but reward {r.reward!r}")
        return None
    raise TrainerInputError(f"{r.candidate_id}: unknown status {r.status!r}")


def _check_sample(s: TrainingSample) -> None:
    if not s.response_ids:
        raise TrainerInputError(f"{s.chain_id}: empty response; the engine always returns a token")
    if not s.prompt_ids:
        raise TrainerInputError(f"{s.chain_id}: empty prompt")
    if (s.iteration, s.prompt_id) != _group_of(s.chain_id):
        raise TrainerInputError(f"{s.chain_id}: group key {(s.iteration, s.prompt_id)} does not match the chain id")


def logprobs_usable(s: "TrainSample") -> bool:
    """The engine log-probs line up with the response and are finite log-probabilities, as the rollout accepts them
(the gap gate needs them)."""
    lp = s.engine_logprobs
    return len(lp) == len(s.response_ids) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x <= MAX_LOGPROB
        for x in lp)


def turn_spans(sample: TrainingSample, chain, *, rerendered: bool = False) -> tuple[TurnSpan, ...]:
    """Every turn's reply located in the final sequence, from the chain's TurnRecords; refuses a chain whose
    records do not reproduce the sample's ids (Appendix C: ids_{t+1} starts with ids_t + reply_t). A `rerendered`
    chain (`repair_turn_ids: rerender`: the template drops earlier think blocks and trims earlier replies) keeps the
    final turn's check and leaves out an earlier turn whose reply is not in the final sequence as sampled."""
    full = [*sample.prompt_ids, *sample.response_ids]
    if chain.chain_id != sample.chain_id or not chain.turns:
        raise TrainerInputError(f"{sample.chain_id}: chain record {chain.chain_id!r} does not match")
    spans = []
    for tr in chain.turns:
        start, reply = tr.input_len, list(tr.response_ids)
        if full[start:start + len(reply)] != reply or len(tr.response_logprobs) != len(reply):
            if rerendered and tr is not chain.turns[-1]:
                continue
            raise TrainerInputError(f"{sample.chain_id}: turn {tr.turn}'s reply is not at {start} in the sequence")
        spans.append(TurnSpan(tr.turn, start, len(reply), tuple(tr.response_logprobs)))
    last = spans[-1]
    if last.start != len(sample.prompt_ids) or last.length != len(sample.response_ids):
        raise TrainerInputError(f"{sample.chain_id}: the final turn is not the sample's response")
    return tuple(spans)


def build_groups(samples: Iterable[TrainingSample], results: Iterable[RewardResult],
                 *, min_scored: int = 2, chains: dict | None = None, rerendered: bool = False) -> list[GroupPlan]:
    """Join samples with their reward results and form groups (key: iteration, prompt_id).

    Every scored result counts in its group's mean, also one whose chain has no TrainingSample (a
    `logprob_error` chain is scored but never trained, Section 6.1). A sample without a result is unscored,
    never a default reward. `chains` (chain_id -> ChainRecord), when given, adds the turn spans (`rerendered`: see
    `turn_spans`). Groups come back sorted by key.
    """
    if min_scored < 2:
        raise TrainerInputError("min_scored must be >= 2")
    by_chain: dict[str, TrainingSample] = {}
    for s in samples:
        _check_sample(s)
        if s.chain_id in by_chain:
            raise TrainerInputError(f"{s.chain_id}: two TrainingSamples for one chain")
        by_chain[s.chain_id] = s
    res: dict[str, RewardResult] = {}
    for r in results:
        if r.candidate_id in res:
            raise TrainerInputError(f"{r.candidate_id}: two RewardResults for one candidate")
        res[r.candidate_id] = r
    keys: dict[tuple[int, str], list[str]] = {}
    for cid in sorted(set(by_chain) | set(res)):
        keys.setdefault(_group_of(cid), []).append(cid)

    plans = []
    for key in sorted(keys):
        scored: list[tuple[str, float]] = []
        excluded: Counter = Counter()
        for cid in keys[key]:
            r = res.get(cid)
            reward = _check_result(r) if r is not None else None
            if r is None:
                excluded["no_reward_result"] += 1
            elif reward is None:
                excluded[r.status] += 1
            else:
                scored.append((cid, reward))
                if cid not in by_chain:
                    excluded["scored_without_sample"] += 1
        if len(scored) < min_scored:
            plans.append(GroupPlan(key, len(keys[key]), len(scored), None, (), f"fewer_than_{min_scored}_scored",
                                   dict(excluded)))
            continue
        mean = math.fsum(r for _, r in scored) / len(scored)
        trained = []
        for cid, reward in scored:
            s = by_chain.get(cid)
            if s is None:
                continue
            trained.append(TrainSample(
                chain_id=cid, group=key, prompt_ids=tuple(s.prompt_ids), response_ids=tuple(s.response_ids),
                engine_logprobs=tuple(s.response_logprobs), reward=reward, advantage=reward - mean,
                turns=turn_spans(s, chains[cid], rerendered=rerendered) if chains is not None and cid in chains else (),
            ))
        plans.append(GroupPlan(key, len(keys[key]), len(scored), mean, tuple(trained), None, dict(excluded)))
    return plans


# --- normalisers --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StepTotals:
    """What the global normaliser of one optimizer step reads: trained samples, their response tokens, groups."""

    samples: int
    tokens: int
    groups: int

    @classmethod
    def of(cls, groups: Iterable[GroupPlan]) -> "StepTotals":
        gs = [g for g in groups if g.samples]
        return cls(sum(len(g.samples) for g in gs), sum(g.n_response_tokens for g in gs), len(gs))


def sample_coefficient(settings: LossSettings, sample: TrainSample, group_tokens: int) -> float:
    """u_i: the part of sample i's weight that its own group fixes (see the module docstring)."""
    kind, l_max = settings.aggregation_kind()
    if kind == "sample_mean":
        return 1.0 / sample.n_response
    if kind == "prompt_token_mean":
        return 1.0 / group_tokens
    if kind == "constant":
        return 1.0 / l_max
    return 1.0  # token_mean, sample_sum


def normaliser(settings: LossSettings, totals: StepTotals) -> float:
    """D: the step's global normaliser. Never a micro-batch's or a rank's own counts (C4.T1)."""
    kind, _ = settings.aggregation_kind()
    d = {"sample_mean": totals.samples, "constant": totals.samples, "sample_sum": totals.samples,
         "token_mean": totals.tokens, "prompt_token_mean": totals.groups}[kind]
    if d <= 0:
        raise TrainerInputError(f"the step has no trained {kind} normaliser ({totals}); nothing to train")
    return float(d)


# --- the per-token objective ---------------------------------------------------------------------------


def pg_token_loss(settings: LossSettings, lp: torch.Tensor, old_lp: torch.Tensor, adv: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token policy-gradient loss and the clip indicator. `lp` (at T_loss) carries the gradient; `old_lp`
    and `adv` are detached. With `old_logprobs` trainer_recompute and one step per iteration, old_lp = lp
    exactly, the ratio is 1 and all three losses give the gradient -A d(lp)."""
    old_lp, adv = old_lp.detach(), adv.detach()
    lo, hi = 1.0 - settings.eps_clip_low, 1.0 + settings.eps_clip_high
    if settings.pg_loss == "reinforce":
        return -adv * lp, torch.zeros_like(lp)
    ratio = torch.exp(lp - old_lp)
    if settings.pg_loss == "ppo_clip":  # Miles compute_policy_loss
        l1 = -ratio * adv
        l2 = -ratio.clamp(lo, hi) * adv
        return torch.maximum(l1, l2), (l2 > l1).to(lp.dtype)
    coef = ratio.clamp(lo, hi).detach()  # cispo: a clipped, detached importance weight on the REINFORCE term
    return -coef * adv * lp, (coef != ratio.detach()).to(lp.dtype)


def tis_weights(settings: LossSettings, old_raw: torch.Tensor, engine_raw: torch.Tensor) -> torch.Tensor:
    """Truncated importance weights clamp(exp(old - engine), 0, C), detached (Miles vanilla_tis_function).

    Both sides are raw (temperature 1) log-probs, because those are all the engine returns (A3); Miles compares
    at the rollout temperature."""
    return torch.exp(old_raw.detach() - engine_raw.detach()).clamp(0.0, settings.tis)


def kl_k3(lp: torch.Tensor, ref_lp: torch.Tensor) -> torch.Tensor:
    """Schulman's k3 estimate of KL(pi || pi_ref): exp(r) - r - 1 with r = log pi_ref - log pi; >= 0 per token."""
    r = ref_lp.detach() - lp
    return torch.exp(r) - r - 1.0


def token_loss(settings: LossSettings, lp: torch.Tensor, old_lp: torch.Tensor, adv: torch.Tensor, *,
               tis_w: torch.Tensor | None = None, ref_lp: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """The full per-token loss: pg (times TIS weights when on) plus kl_beta * k3 when on."""
    pg, clipped = pg_token_loss(settings, lp, old_lp, adv)
    if settings.tis is not None:
        if tis_w is None:
            raise TrainerInputError("TIS is on but no weights were given")
        pg = pg * tis_w
    loss = pg
    kl = None
    if settings.kl_beta > 0:
        if ref_lp is None:
            raise TrainerInputError("kl_beta > 0 but no reference log-probs were given")
        kl = kl_k3(lp, ref_lp)
        loss = loss + settings.kl_beta * kl
    return loss, {"clipped": clipped.detach(), "kl": None if kl is None else kl.detach()}


# --- the sampler/trainer gap ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleGap:
    """Per-sample |trainer raw log-prob - engine raw log-prob| over the response tokens (the gap gate's input)."""

    chain_id: str
    tokens: int
    max_abs: float
    mean_abs: float
    median_abs: float
    p99_abs: float
    frac_gt_1: float
    mean_signed: float  # mean of (trainer - engine)

    @property
    def statistic(self) -> float:
        """The gate's default statistic, as Section 6.4 defines it: the largest per-token gap."""
        return self.max_abs

    def value(self, statistic: str) -> float:
        """One of GAP_STATISTICS by name (the gate reads `LossSettings.gap_statistic` through this)."""
        if statistic not in GAP_STATISTICS:
            raise TrainerInputError(f"unknown gap statistic {statistic!r}")
        return getattr(self, statistic)


def sample_gap(chain_id: str, trainer_raw: torch.Tensor, engine_raw: Sequence[float]) -> SampleGap:
    t = trainer_raw.detach().double().cpu().reshape(-1)
    e = torch.tensor(list(engine_raw), dtype=torch.float64)
    if t.numel() != e.numel() or t.numel() == 0:
        raise TrainerInputError(f"{chain_id}: {t.numel()} trainer log-probs against {e.numel()} engine log-probs")
    d = t - e
    a = d.abs()
    q = torch.quantile(a, torch.tensor([0.5, 0.99], dtype=torch.float64)).tolist()
    return SampleGap(chain_id, int(a.numel()), float(a.max()), float(a.mean()), q[0], q[1],
                     float((a > 1.0).double().mean()), float(d.mean()))


def gap_keeps(settings: LossSettings, gap: SampleGap | None) -> bool:
    """Whether a sample trains: its engine log-probs are usable and its `gap_statistic` is within the bound."""
    if gap is None:
        return False
    v = gap.value(settings.gap_statistic)
    return math.isfinite(v) and v <= settings.max_generation_gap


# --- the per-sample reference (C4.T1) ---------------------------------------------------------------------


def reference_loss(settings: LossSettings, groups: Sequence[GroupPlan], logprobs_of, *,
                   keep=lambda s: True, backward: bool = True) -> float:
    """The step's loss computed one sample at a time, with weights c_i from the definitions above.

    `logprobs_of(sample) -> (lp_T, lp_raw)` scores one sample on its own (no packing). With `backward` the
    gradient of the whole step accumulates into the model's .grad. Only trainer_recompute with one step is
    covered, which is the configuration C4.T1 tests.
    """
    if settings.old_logprobs != "trainer_recompute" or settings.steps_per_iteration != 1:
        raise TrainerInputError("the per-sample reference covers trainer_recompute with one step per iteration")
    if settings.kl_beta > 0 or settings.tis is not None:
        raise TrainerInputError("the per-sample reference does not cover KL or TIS")
    gs = [g for g in groups if g.samples]
    d = normaliser(settings, StepTotals.of(gs))
    total = 0.0
    for g in gs:
        for s in g.samples:
            if not keep(s):
                continue
            lp, _ = logprobs_of(s)
            adv = torch.full_like(lp, s.advantage)
            loss_tok, _ = token_loss(settings, lp, lp.detach(), adv)
            term = loss_tok.sum() * (sample_coefficient(settings, s, g.n_response_tokens) / d)
            if backward:
                term.backward()
            total += float(term.detach())
    return total
