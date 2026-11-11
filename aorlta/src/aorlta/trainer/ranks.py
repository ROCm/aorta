"""C4 over R ranks (design Section 6.4, Batching and Streaming gradient accumulation).

Each rank holds a replica of the policy and an identical AdamW state. Groups, not samples, go to ranks (a group
trains as soon as its K samples are scored). A rank accumulates, group by group, the *unnormalised* sums
sum u_i l_it into its LoRA gradients, plus the counts the global normaliser D needs. Once per step:

1. the flat fp32 LoRA gradient (32.5 M values, 130 MB) is all-reduced by hand, not through DDP hooks, because
   ranks run different numbers of micro-batches; the per-rank counts and check inputs are all-gathered;
2. every rank divides by the same D, runs the pre-step checks, clips, steps and refreshes W_eff (`merged.refresh`,
   the function whose output C5 writes), then runs the post-step checks, including that all replicas agree.

A failed check raises (`StepRefused` before the step, `TrainerCheckFailed` after it) before anything is
published. `train_iteration` is the batched path, and the only one that takes `steps_per_iteration` > 1; the
streaming path is `stream_begin` / `stream_add` / `stream_finish`, with one step per iteration.

Over R > 1 ranks every step carries an assignment ledger: one `Assignment` (seq, group key, rank, samples, tokens)
per trained group, which the coordinator broadcasts to every rank as it assigns the group. Before the step, every
rank checks the same gathered facts (`groups_trained_once`): all ranks hold the same ledger, and every ledger group
was trained exactly once, on its own rank, so the step's global totals are the ledger's. After the step,
`replicas_identical` also compares an exact checksum of every rank's W_eff, so the tensors C5 writes from rank 0
are the ones every rank's forward runs on (C4.T3).
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import torch

from aorlta.config import TrainerSection
from aorlta.trainer import checks as C
from aorlta.trainer import kernels
from aorlta.trainer.batching import DEFAULT_CHUNK_TOKENS, forward_microbatch, head_logprobs, pack
from aorlta.trainer.loss import (
    GroupPlan,
    LossSettings,
    StepTotals,
    TrainerInputError,
    TrainSample,
    gap_keeps,
    logprobs_usable,
    normaliser,
    sample_coefficient,
    sample_gap,
    tis_weights,
    token_loss,
)

# C4.T1's gradient agreement limit, R ranks against one, by the trained dtype. 0.99999 was set for fp32. In bf16 on
# GPU each process's kernels round their own way (4 ranks against 1: cosine 0.9999795 for the normal spread), so the
# bf16 limit is 0.99998.
RANK_AGREEMENT_COSINE = {"float32": 0.99999, "bfloat16": 0.99998}


def rank_agreement_cosine(dtype: Any) -> float:
    """The limit for `dtype`, a torch dtype or its name. A dtype with no agreed limit is refused."""
    name = str(dtype).removeprefix("torch.")
    if name not in RANK_AGREEMENT_COSINE:
        raise ValueError(f"no rank agreement limit for {name}: only {sorted(RANK_AGREEMENT_COSINE)}")
    return RANK_AGREEMENT_COSINE[name]


def ranks_agree(cosine: float, dtype: Any) -> bool:
    """R ranks' gradient agrees with one rank's when their cosine reaches the limit for the trained dtype."""
    return cosine >= rank_agreement_cosine(dtype)


def engine_tensor(s, usable: bool) -> torch.Tensor:
    """A sample's engine log-probs for the ratio, at most 0: the rollout accepts up to MAX_LOGPROB above it, from
    rounding a probability of 1 (zeros for a sample without usable ones, which is excluded)."""
    return torch.tensor(s.engine_logprobs if usable else [0.0] * s.n_response, dtype=torch.float32).clamp(max=0.0)


# --- the collective -----------------------------------------------------------------------------------


class World(Protocol):
    rank: int
    size: int

    def all_reduce_sum_(self, t: torch.Tensor) -> None: ...

    def broadcast_(self, t: torch.Tensor, src: int = 0) -> None: ...

    def all_gather_object(self, obj: Any) -> list[Any]: ...


class LocalWorld:
    """One rank, no communication."""

    rank, size = 0, 1

    def all_reduce_sum_(self, t: torch.Tensor) -> None:
        return None

    def broadcast_(self, t: torch.Tensor, src: int = 0) -> None:
        return None

    def all_gather_object(self, obj: Any) -> list[Any]:
        return [obj]


class DistWorld:
    """torch.distributed: gloo on CPU, RCCL (the nccl backend) on GPU."""

    def __init__(self, group=None):
        import torch.distributed as dist

        if not dist.is_initialized():
            raise TrainerInputError("torch.distributed is not initialised")
        self._dist, self.group = dist, group
        self.rank, self.size = dist.get_rank(group), dist.get_world_size(group)

    def all_reduce_sum_(self, t: torch.Tensor) -> None:
        self._dist.all_reduce(t, op=self._dist.ReduceOp.SUM, group=self.group)

    def broadcast_(self, t: torch.Tensor, src: int = 0) -> None:
        self._dist.broadcast(t, src=src, group=self.group)

    def all_gather_object(self, obj: Any) -> list[Any]:
        out: list[Any] = [None] * self.size
        self._dist.all_gather_object(out, obj, group=self.group)
        return out


def broadcast_lora(model, world: World) -> None:
    """Give every rank rank 0's LoRA latent. Call it after PEFT has initialised the adapter and before the
    merged-weight forward caches W_eff: PEFT draws A from each process's own random state, so the ranks do not
    otherwise start from the same weights (a replica check caught exactly that)."""
    params = C.LoraTensors.of(model).params()
    flat = torch.cat([p.detach().reshape(-1).float() for p in params])
    world.broadcast_(flat, 0)
    offset = 0
    with torch.no_grad():
        for p in params:
            p.copy_(flat[offset:offset + p.numel()].view_as(p))
            offset += p.numel()


# --- groups to ranks ---------------------------------------------------------------------------------


def group_tokens(g: GroupPlan) -> int:
    return sum(s.length for s in g.samples)


def assign_groups(groups: Sequence[GroupPlan], ranks: int) -> list[list[GroupPlan]]:
    """Longest-processing-time first by forward tokens; deterministic (ties by key, then lowest rank)."""
    if ranks < 1:
        raise TrainerInputError("ranks must be >= 1")
    out: list[list[GroupPlan]] = [[] for _ in range(ranks)]
    load = [0] * ranks
    for g in sorted((g for g in groups if g.samples), key=lambda g: (-group_tokens(g), g.key)):
        r = min(range(ranks), key=lambda i: (load[i], i))
        out[r].append(g)
        load[r] += group_tokens(g)
    return out


class OnlineAssigner:
    """Streaming: each group, as it completes, goes to the rank with the fewest tokens still queued on it, then the
    fewest tokens so far, then the lowest rank. `done` tells it that a rank has trained a group; without it the
    queued and assigned tokens are the same, and the rule is the fewest tokens so far."""

    def __init__(self, ranks: int):
        if ranks < 1:
            raise TrainerInputError("ranks must be >= 1")
        self.load = [0] * ranks
        self.queued = [0] * ranks

    def assign(self, g: GroupPlan) -> int:
        r = min(range(len(self.load)), key=lambda i: (self.queued[i], self.load[i], i))
        n = group_tokens(g)
        self.load[r] += n
        self.queued[r] += n
        return r

    def done(self, rank: int, tokens: int) -> None:
        self.queued[rank] -= tokens


class Assignment(NamedTuple):
    """One ledger entry: the `seq`-th group assigned in this step, and the rank that trains it."""

    seq: int
    key: tuple[int, str]
    rank: int
    samples: int
    tokens: int  # response tokens: the normaliser's token count

    @classmethod
    def of(cls, seq: int, g: GroupPlan, rank: int) -> "Assignment":
        return cls(seq, tuple(g.key), rank, len(g.samples), g.n_response_tokens)


def batched_ledger(per_rank: Sequence[Sequence[GroupPlan]]) -> list[Assignment]:
    """The ledger of an `assign_groups` split: groups in key order, each with the rank that holds it."""
    owner = {g.key: (r, g) for r, gs in enumerate(per_rank) for g in gs}
    return [Assignment.of(i, owner[k][1], owner[k][0]) for i, k in enumerate(sorted(owner))]


def assignment_check(ledgers: Sequence[Sequence], trained: Sequence[Sequence], totals: StepTotals) -> dict:
    """`groups_trained_once`, evaluated by every rank on the same gathered facts: every rank received the same
    ledger, its seqs run 0..n-1 and its keys are distinct; every ledger group was trained exactly once, on the rank
    the ledger names, and no rank trained a group outside it; so the step's totals are the ledger's."""
    ref = [Assignment(*a) for a in ledgers[0]]
    same = all([Assignment(*a) for a in lg] == ref for lg in ledgers)
    seqs_in_order = [a.seq for a in ref] == list(range(len(ref)))
    distinct = len({a.key for a in ref}) == len(ref)
    want = Counter((a.key, a.rank) for a in ref)
    got = Counter((tuple(key), r) for r, keys in enumerate(trained) for key in keys)
    missing, extra = want - got, got - want
    expected = StepTotals(sum(a.samples for a in ref), sum(a.tokens for a in ref), len(ref))
    passed = (same and seqs_in_order and distinct and not missing and not extra and totals == expected)
    return {"groups": len(ref), "ledgers_identical": same, "seqs_in_order": seqs_in_order, "keys_distinct": distinct,
            "missing": [list(x) for x in sorted(missing)][:5], "extra": [list(x) for x in sorted(extra)][:5],
            "totals": totals.__dict__, "ledger_totals": expected.__dict__,
            "per_rank": [len(lst) for lst in trained], "passed": passed}


def weights_checksum(merged) -> tuple | None:
    """Every W_eff tensor the merged forward holds, by name, with its dtype, shape and xxh64 digest
    (`checks.digest`): equal bits give equal checksums on any rank, and any other bits (a permutation too) a
    different one. None when the forward holds no weights (a test stub)."""
    weights = getattr(merged, "weights", None)
    if not weights:
        return None
    return tuple((key, *C.digest(weights[key])) for key in sorted(weights))


def split_minibatches(groups: Sequence[GroupPlan], steps: int) -> list[list[GroupPlan]]:
    """`steps` contiguous slices of the trained groups, as equal in samples as whole groups allow: a group goes to the
    slice its first sample falls in, except that the slices fill in order and each keeps at least one group."""
    gs = [g for g in groups if g.samples]
    if steps > len(gs):
        raise TrainerInputError(f"steps_per_iteration {steps} exceeds the {len(gs)} trained groups")
    total = sum(len(g.samples) for g in gs)
    out: list[list[GroupPlan]] = [[] for _ in range(steps)]
    k = seen = 0
    for i, g in enumerate(gs):
        if i:
            k = min(k + 1, max(k, steps - len(gs) + i, (seen * steps) // total))
        out[k].append(g)
        seen += len(g.samples)
    return out


# --- the start-up warm-up ------------------------------------------------------------------------------

WARMUP_KEY = (0, "warmup")


def warmup_group(ids: Sequence[Sequence[int]], micro_batch_tokens: int, max_new_tokens: int) -> GroupPlan:
    """A synthetic group shaped like a step's micro-batches, for `RankTrainer.warmup`. Four samples of about 50%,
    30%, 15% and 10% of `micro_batch_tokens` pack into one row of three sequences (the packed, varlen path) whose
    length is not a multiple of 16, and one row alone whose length is (Triton specialises integer arguments on
    that); this holds for any `micro_batch_tokens` of at least 256. Each reply is half its sample, at most
    `max_new_tokens`. The tokens are `ids` (real token sequences, such as the landing probes) read in a cycle."""
    flat = [int(t) for seq in ids for t in seq]
    if not flat:
        raise TrainerInputError("the warm-up needs token ids")
    m = micro_batch_tokens
    lengths = [m // 2 + 1, (3 * m) // 10 + 1, (3 * m) // 20 + 1, max(16, (m // 10) // 16 * 16)]
    if sum(lengths[:3]) % 16 == 0:
        lengths[0] += 1
    samples, offset = [], 0
    for i, n in enumerate(lengths):
        seq = [flat[(offset + j) % len(flat)] for j in range(n)]
        offset += n
        r = max(1, min(max_new_tokens, n // 2))
        samples.append(TrainSample(chain_id=f"warmup-{i}", group=WARMUP_KEY, prompt_ids=tuple(seq[:n - r]),
                                   response_ids=tuple(seq[n - r:]), engine_logprobs=(-1.0,) * r, reward=0.0,
                                   advantage=1.0 if i % 2 == 0 else -1.0))
    return GroupPlan(WARMUP_KEY, len(samples), len(samples), 0.0, tuple(samples), None)


# --- the optimizer -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimSettings:
    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0  # 0 = off

    @classmethod
    def from_trainer(cls, t: TrainerSection) -> "OptimSettings":
        return cls(t.lr, tuple(t.betas), t.eps, t.weight_decay, t.grad_clip)


def clip_coefficient(grad_norm: float, grad_clip: float) -> float:
    """Megatron's (Miles') global clip: max_norm / (norm + 1e-6), when the norm exceeds max_norm."""
    if grad_clip > 0 and grad_norm > grad_clip:
        return grad_clip / (grad_norm + 1e-6)
    return 1.0


# --- one rank ----------------------------------------------------------------------------------------


class RankTrainer:
    """One rank's trainer: the model (with the merged-weight forward installed), the AdamW state, the checks."""

    def __init__(self, model, merged, loss: LossSettings, optim: OptimSettings, *, world: World | None = None,
                 micro_batch_tokens: int = 65536, device="cpu", amp: bool = True,
                 chunk_tokens: int = DEFAULT_CHUNK_TOKENS, gap_gate: bool = True,
                 control: C.BaseControl | None = None):
        if merged is None or not callable(getattr(merged, "refresh", None)):
            raise TrainerInputError("the trainer needs the merged-weight forward (an object with refresh())")
        self.model, self.merged, self.loss, self.optim = model, merged, loss, optim
        self.world = world or LocalWorld()
        self.micro_batch_tokens, self.device, self.amp = micro_batch_tokens, device, amp
        self.chunk_tokens, self.gap_gate = chunk_tokens, gap_gate
        self.lora = C.LoraTensors.of(model)
        self.params = self.lora.params()
        self.packed = kernels.honours_packing(model)
        # Built once: Adam moments persist across iterations (zeroed moments every step make Adam signSGD).
        self.optimizer = torch.optim.AdamW(self.params, lr=optim.lr, betas=optim.betas, eps=optim.eps,
                                           weight_decay=optim.weight_decay)
        problems = C.structural_problems(model, self.lora, self.optimizer)
        if problems:
            raise TrainerInputError("; ".join(problems[:5]))
        self.control = control or C.BaseControl.capture(model, self.lora)
        start = ([C.digest(p) for p in self.params], [C.digest(t) for t in self.control.tensors.values()])
        if any(s != start for s in self.world.all_gather_object(start)):
            raise TrainerInputError("the ranks do not start from the same LoRA latent and base weights "
                                    "(call broadcast_lora before installing the merged forward)")
        if self.world.size > 1:  # every target's W_eff, not only the control's samples: what each forward runs on
            w_start = weights_checksum(merged)
            if any(w != w_start for w in self.world.all_gather_object(w_start)):
                raise TrainerInputError("the ranks do not start from the same merged weights W_eff")
        self.steps_done = 0
        self._streaming = False
        self._reset()

    # --- accumulation ---

    def _rows(self, samples: Sequence[TrainSample]) -> list:
        """The micro-batches of `samples`: packed, or one sequence each where the model's linear attention would run
        a packed row as one sequence (`kernels.honours_packing`)."""
        return pack(samples, self.micro_batch_tokens, max_sequences=None if self.packed else 1)

    def _reset(self) -> None:
        for p in self.params:
            p.grad = None
        self.counts = {"samples": 0, "tokens": 0, "groups": 0, "compared": 0, "excluded": 0, "logprob_error": 0,
                       "microbatches": 0, "forward_tokens": 0, "clipped_tokens": 0.0, "loss_tokens": 0}
        self.loss_sum = 0.0
        self.gaps: list = []
        self.excluded_ids: list[str] = []
        self.over_bound_ids: list[str] = []  # samples the gate's bound refuses, whether or not it is enforced
        self.taudit = C.TemperatureAudit(self.loss.loss_temperature)
        self.seconds = {"forward": 0.0, "backward": 0.0}
        self.trained_groups: list[tuple] = []  # the key of every group accumulated here since the last reset
        self._ledger: list[Assignment] | None = None  # this step's assignment ledger, as this rank received it

    def _add(self, groups: Sequence[GroupPlan], old: dict | None, *, across_groups: bool) -> None:
        """Accumulate these groups. The batched path packs all their samples together (sorted by length);
        streaming packs each group on its own, since a group trains as soon as it is complete."""
        self.model.train()
        live = [g for g in groups if g.samples]
        u = {}
        for g in live:
            self.trained_groups.append(tuple(g.key))
            self.counts["samples"] += len(g.samples)
            self.counts["tokens"] += g.n_response_tokens
            self.counts["groups"] += 1
            u.update({s.chain_id: sample_coefficient(self.loss, s, g.n_response_tokens) for s in g.samples})
        batches = [[s for g in live for s in g.samples]] if across_groups else [list(g.samples) for g in live]
        for samples in batches:
            for mb in self._rows(samples):
                self._microbatch(mb, u, old)

    def _microbatch(self, mb, u: dict, old: dict | None) -> None:
        L = self.loss
        t0 = time.monotonic()
        lp, raw, h = forward_microbatch(self.model, mb, device=self.device, temperature=L.loss_temperature,
                                        amp=self.amp, lm_head_fp32=L.lm_head_fp32, chunk_tokens=self.chunk_tokens)
        dev = lp.device
        keep, engine = [], []
        for s, sl in zip(mb.samples, mb.sample_slices()):
            usable = logprobs_usable(s)
            gap = sample_gap(s.chain_id, raw[sl], s.engine_logprobs) if usable else None
            self.counts["compared"] += 1
            if gap is not None:
                self.gaps.append(gap)
            else:
                self.counts["logprob_error"] += 1
            within = gap_keeps(L, gap)
            if not within:
                self.over_bound_ids.append(s.chain_id)
            k = within if self.gap_gate else True  # report-only: measured and logged, nothing excluded
            if (L.tis is not None or L.old_logprobs == "engine") and not usable:
                k = False  # no engine log-probs to weight or ratio against
            if not k:
                self.counts["excluded"] += 1
                self.excluded_ids.append(s.chain_id)
            keep.append(k)
            engine.append(engine_tensor(s, usable))
        engine_raw = torch.cat(engine).to(dev)
        # The temperature check: the loss's log-probs against log_softmax(z / T_loss) recomputed from the same
        # hidden states, on the first chunk (the same GEMM shape as the loss's own first chunk).
        c = min(self.chunk_tokens, mb.loss_tokens)
        rec = head_logprobs(self.model, h[:c].detach(), mb.targets[:c], device=self.device,
                            temperature=L.loss_temperature, amp=self.amp, lm_head_fp32=L.lm_head_fp32)
        self.taudit.add(lp[:c], rec, raw[:c])
        if old is not None and L.old_logprobs == "trainer_recompute":
            old_t = torch.cat([old[s.chain_id][0] for s in mb.samples]).to(dev)
            old_raw = torch.cat([old[s.chain_id][1] for s in mb.samples]).to(dev)
        else:
            old_t, old_raw = lp.detach(), raw
        if L.old_logprobs == "engine":
            old_t = engine_raw  # T_loss is 1 here (LossSettings refuses anything else)
        idx = mb.token_sample.to(dev)
        # An excluded sample's tokens are left out of the objective, not weighted by zero: its ratio against the
        # engine can overflow, and zero times inf is NaN.
        kept = torch.tensor(keep, dtype=torch.bool, device=dev)[idx]
        adv = torch.tensor([s.advantage for s in mb.samples], dtype=torch.float32, device=dev)[idx]
        coef = torch.tensor([u[s.chain_id] for s in mb.samples], dtype=torch.float32, device=dev)[idx]
        tis_w = tis_weights(L, old_raw, engine_raw) if L.tis is not None else None
        ref = None
        if L.kl_beta > 0:
            with torch.no_grad(), self.model.disable_adapter():  # the reference: W_eff = W
                ref, _, _ = forward_microbatch(self.model, mb, device=self.device, temperature=L.loss_temperature,
                                               amp=self.amp, lm_head_fp32=L.lm_head_fp32,
                                               chunk_tokens=self.chunk_tokens)
        loss_tok, aux = token_loss(L, lp[kept], old_t[kept], adv[kept], tis_w=None if tis_w is None else tis_w[kept],
                                   ref_lp=None if ref is None else ref[kept])
        local = (loss_tok * coef[kept]).sum()
        t1 = time.monotonic()
        local.backward()
        self.seconds["forward"] += t1 - t0
        self.seconds["backward"] += time.monotonic() - t1
        self.loss_sum += float(local.detach())
        self.counts["microbatches"] += 1
        self.counts["forward_tokens"] += mb.tokens
        self.counts["loss_tokens"] += int(kept.sum())  # the tokens in the objective, which clipped_tokens counts
        self.counts["clipped_tokens"] += float(aux["clipped"].sum())

    def _old_logprobs(self, groups: Sequence[GroupPlan]) -> dict:
        """pi_old for later mini-batches: the sampling weights' log-probs, computed before the first step."""
        out = {}
        self.model.eval()
        with torch.no_grad():
            for g in groups:
                for mb in self._rows(g.samples):
                    lp, raw, _ = forward_microbatch(self.model, mb, device=self.device,
                                                    temperature=self.loss.loss_temperature, amp=self.amp,
                                                    lm_head_fp32=self.loss.lm_head_fp32,
                                                    chunk_tokens=self.chunk_tokens)
                    for s, sl in zip(mb.samples, mb.sample_slices()):
                        out[s.chain_id] = (lp[sl].detach(), raw[sl].detach())
        return out

    # --- the step ---

    def _flat_grads(self) -> torch.Tensor:
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float()
                          for p in self.params])

    def _local_stats(self) -> dict:
        t = self.taudit
        return {"counts": dict(self.counts), "loss_sum": self.loss_sum,
                "taudit": [t.compared, t.mismatched, t.max_abs_vs_recomputed, t.max_abs_vs_raw],
                "gaps": list(self.gaps), "excluded_ids": list(self.excluded_ids),
                "over_bound_ids": list(self.over_bound_ids), "seconds": dict(self.seconds),
                "trained_groups": list(self.trained_groups),
                "ledger": None if self._ledger is None else [tuple(a) for a in self._ledger]}

    def _assignment(self, stats: Sequence[dict], totals: StepTotals) -> dict:
        """`groups_trained_once` from the gathered stats. Without a ledger it holds only on one rank, which trains
        every group it is given, and only if no group was accumulated twice."""
        ledgers = [s["ledger"] for s in stats]
        if all(lg is None for lg in ledgers):
            keys = Counter(tuple(k) for s in stats for k in s["trained_groups"])
            twice = sorted(k for k, n in keys.items() if n > 1)
            return {"ledger": None, "ranks": len(stats), "trained_twice": [list(k) for k in twice][:5],
                    "passed": len(stats) == 1 and not twice}
        if any(lg is None for lg in ledgers):
            return {"ledger": "not on every rank", "passed": False}
        return assignment_check(ledgers, [s["trained_groups"] for s in stats], totals)

    def _gap_gate_report(self, compared: int, over_bound: Sequence[str]) -> dict:
        """The gate's verdict on this step's samples under its bounds and its statistic (`trainer.gap_statistic`),
        enforced or not (Section 6.4)."""
        fraction = len(over_bound) / compared if compared else (1.0 if over_bound else 0.0)
        return {"mode": "enforce" if self.gap_gate else "report_only", "statistic": self.loss.gap_statistic,
                "max_generation_gap": self.loss.max_generation_gap,
                "max_off_policy_fraction": self.loss.max_off_policy_fraction, "compared": compared,
                "over_bound": len(over_bound), "over_bound_ids": sorted(over_bound), "fraction": fraction,
                "would_refuse": not (compared > 0 and fraction <= self.loss.max_off_policy_fraction)}

    def stream_report(self) -> dict:
        """What the open streamed step has measured, without stepping: the gap and the gate's verdict (for an
        iteration that takes no step, such as one where every group has zero advantage)."""
        if not self._streaming:
            raise TrainerInputError("stream_begin was not called")
        stats = self.world.all_gather_object(self._local_stats())
        counts = {k: sum(s["counts"][k] for s in stats) for k in stats[0]["counts"]}
        gaps = [g for s in stats for g in s["gaps"]]
        over = [i for s in stats for i in s["over_bound_ids"]]
        return {"counts": counts, "gap": C.gap_summary(gaps), "gaps": gaps,
                "gap_gate": self._gap_gate_report(counts["compared"], over)}

    def _finish(self) -> dict:
        t_start = time.monotonic()
        flat = self._flat_grads()
        self.world.all_reduce_sum_(flat)
        stats = self.world.all_gather_object(self._local_stats())
        t_reduce = time.monotonic() - t_start
        counts = {k: sum(s["counts"][k] for s in stats) for k in stats[0]["counts"]}
        totals = StepTotals(counts["samples"], counts["tokens"], counts["groups"])
        d = normaliser(self.loss, totals)
        flat.div_(d)
        loss_value = math.fsum(s["loss_sum"] for s in stats) / d
        grad_norm = float(torch.linalg.vector_norm(flat, dtype=torch.float64))
        taudit = C.TemperatureAudit(self.loss.loss_temperature)
        for s in stats:
            a = C.TemperatureAudit(self.loss.loss_temperature, *s["taudit"])
            taudit.merge(a)
        gaps = [g for s in stats for g in s["gaps"]]
        report: dict[str, Any] = {
            "step": self.steps_done + 1, "ranks": self.world.size, "totals": totals.__dict__, "normaliser": d,
            "aggregation": self.loss.aggregation, "loss": loss_value, "grad_norm": grad_norm, "counts": counts,
            "excluded": sorted(i for s in stats for i in s["excluded_ids"]), "gap": C.gap_summary(gaps),
            "gaps": gaps, "seconds_by_rank": [s["seconds"] for s in stats], "reduce_sec": t_reduce,
            "gap_gate": self._gap_gate_report(counts["compared"], [i for s in stats for i in s["over_bound_ids"]]),
            "packed": self.packed,
        }
        pre_checks = C.pre_step_checks(grad_norm=grad_norm, loss=loss_value, compared=counts["compared"],
                                       excluded=counts["excluded"],
                                       max_fraction=self.loss.max_off_policy_fraction,
                                       temperature_check=taudit.result())
        if not self.gap_gate:
            pre_checks["off_policy_fraction"] = dict(pre_checks["off_policy_fraction"], gate="off", passed=True)
        pre_checks["groups_trained_once"] = self._assignment(stats, totals)
        report["checks"] = dict(pre_checks)
        bad = C.failed(pre_checks, C.GATING_PRE)
        if bad:
            self._reset()
            raise C.StepRefused(bad, report)
        coef = clip_coefficient(grad_norm, self.optim.grad_clip)
        if coef != 1.0:
            flat.mul_(coef)
        report["clip_coefficient"] = coef
        self.last_flat_grad = flat  # the gradient AdamW consumes (C4.T1, C4.T7 compare it)
        offset = 0
        for p in self.params:
            n = p.numel()
            p.grad = flat[offset:offset + n].view_as(p).to(p.dtype).clone()
            offset += n
        pre = C.PreStep.take(self.lora, self.merged)
        t0 = time.monotonic()
        self.optimizer.step()
        merge = self.merged.refresh()
        report["step_sec"] = time.monotonic() - t0
        self.steps_done += 1
        fp = [C.digest(p) for p in self.params]
        t_sum = time.monotonic()
        w_sum = weights_checksum(self.merged) if self.world.size > 1 else None
        replicas = self.world.all_gather_object((fp, w_sum))
        report["checksum_sec"] = time.monotonic() - t_sum
        latent_same = all(r[0] == replicas[0][0] for r in replicas)
        w_eff_same = all(r[1] == replicas[0][1] for r in replicas)
        post = C.post_step_checks(pre, self.lora, self.control, self.merged,
                                  weight_decay=self.optim.weight_decay,
                                  replicas_identical=latent_same and w_eff_same)
        post["replicas_identical"].update(ranks=self.world.size, latent=latent_same, w_eff=w_eff_same,
                                          w_eff_compared=w_sum is not None)
        report["checks"].update(post)
        stats_obj = getattr(merge, "stats", None)
        report["merge"] = stats_obj.as_dict() if hasattr(stats_obj, "as_dict") else None
        bad = C.failed(post, C.GATING_POST)
        self._reset()
        if bad:
            raise C.TrainerCheckFailed(bad, report)
        return report

    # --- the two paths ---

    def train_iteration(self, groups: Sequence[GroupPlan]) -> list[dict]:
        """The batched path: every trained group of the iteration, split into `steps_per_iteration` steps."""
        if self._streaming:
            raise TrainerInputError("a streamed step is open")
        S = self.loss.steps_per_iteration
        minis = split_minibatches(groups, S)
        splits = [assign_groups(m, self.world.size) for m in minis]
        mine = [split[self.world.rank] for split in splits]
        old = self._old_logprobs([g for m in mine[1:] for g in m]) if S > 1 else None
        reports = []
        for k, m in enumerate(mine):
            self._reset()
            self._add(m, old if k > 0 else None, across_groups=True)
            self._ledger = batched_ledger(splits[k])  # every rank computes the same split
            reports.append(self._finish())
        return reports

    def stream_begin(self) -> None:
        if self.loss.steps_per_iteration != 1:
            raise TrainerInputError("streaming accumulation needs steps_per_iteration 1 (Section 6.4)")
        self._reset()
        self._streaming = True

    def stream_add(self, group: GroupPlan) -> None:
        """Accumulate one complete group's gradient on this rank (call it on the rank the group went to)."""
        if not self._streaming:
            raise TrainerInputError("stream_begin was not called")
        self._add([group], None, across_groups=False)

    def stream_finish(self, ledger: Sequence[Assignment] | None = None) -> dict:
        """The step. Over R > 1 ranks `ledger` is the step's assignments as this rank received them (every rank
        passes its own copy); `groups_trained_once` refuses the step unless all ranks hold the same one and trained
        it exactly. One rank may pass none: it trained every group it was given."""
        if not self._streaming:
            raise TrainerInputError("stream_begin was not called")
        if ledger is None and self.world.size > 1:
            raise TrainerInputError(f"a streamed step over {self.world.size} ranks needs the assignment ledger")
        self._streaming = False
        self._ledger = None if ledger is None else [Assignment(*a) for a in ledger]
        return self._finish()

    def stream_discard(self) -> None:
        """Drop an open streamed step's accumulated gradient: an iteration that is not trained."""
        self._reset()
        self._streaming = False

    def warmup(self, group: GroupPlan) -> dict:
        """One streamed step's forward and backward on `group` (see `warmup_group`), then the gradient is dropped:
        no optimizer step and no W_eff refresh, so the latent, the Adam moments and the merged weights are as
        before. It pays the first step's one-time costs (the kernels' compilation) before any real step."""
        if self._streaming:
            raise TrainerInputError("a streamed step is open")
        self._reset()
        # The group's engine log-probs are made up: gated, every sample would leave the objective, and the loss would
        # compile on no token.
        gate, self.gap_gate = self.gap_gate, False
        try:
            self._add([group], None, across_groups=False)
            return {"counts": dict(self.counts), "seconds": dict(self.seconds)}
        finally:
            self.gap_gate = gate
            self._reset()
