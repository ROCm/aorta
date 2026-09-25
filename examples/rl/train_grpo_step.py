#!/usr/bin/env python3
"""GRPO over multi-step probe episodes, with every weight update checked.

The goal is not a good model. It is to show that the loop closes -- rollouts
from the current policy, an event reward per episode, a policy-gradient step,
a checkpoint -- and to hand over before and after checkpoints whose update is
**verified rather than inferred**.

The route, and why this one
===========================
Local HuggingFace ``transformers`` on one GPU, with generation and the
optimiser step **in the same process against the same model object**. The
alternative -- a serving engine generating rollouts and a separate trainer
stepping the weights -- needs a weight transport from trainer to engine after
every step, and a transport that reports success without moving anything is
indistinguishable, from the outputs alone, from one that works. Here the policy
that generates *is* the object the optimiser steps, so there is nothing to ship.

⚠ **This is not slime**, which is the framework decision recorded in
``docs/rl-post-training-decisions.md``. It is a plain GRPO step, deliberately
small, that exercises the environment, the reward and the checks end to end.

What an iteration does
======================
1. **Rollouts from the current policy.** For each scenario, ``--group``
   episodes advance in lockstep through ``episode_env.rollout_scenario``; every
   live episode of a scenario is advanced in one batched ``generate`` call.
2. **One advantage per episode**, the episode's summed event reward normalised
   within its scenario's group, repeated on every step of the episode
   (REINFORCE's trajectory gradient). See ``episode_env`` for why not per step.
3. **The loss**, per (prompt, completion) sample: ``-(A / N) * sum log pi``
   over completion tokens, plus ``(kl_beta / N) * KL`` to a frozen reference
   when ``--kl-beta`` is set, where ``N`` is the number of samples.
4. **Adam**, with moments built once outside the loop so they accumulate. The
   tempting shortcut -- Adam's first step from zeroed moments, every iteration
   -- is signSGD: it moves every element by the full ``lr`` whatever the
   gradient, which is a far larger step than the learning rate suggests.
5. **The checks** (below), and only then a checkpoint.

The KL term
===========
Schulman's k3 estimator of ``KL(policy || reference)``, ``exp(r) - r - 1`` with
``r = log pi_ref - log pi``: unbiased, lower variance than the naive log-ratio,
and non-negative per token, so the penalty cannot pay the policy for drifting.
The reference is the base model, frozen, in bf16 on its own device and only
ever run under ``no_grad``, so neither its precision nor its placement can
touch the policy's update. KL to a reference was chosen over an entropy bonus
because it is the standard GRPO objective and the reference costs nothing.

⚠ The KL share of the loss is a good alarm for a coefficient that is far too
strong and a poor predictor for tuning one: KL per token grows as the policy
drifts, so a coefficient right at iteration 5 can be too weak by iteration 40.

The checks
==========
Every iteration, before a checkpoint is written:

``trained_tensors_are_finite``
    No trained tensor holds a NaN or an inf after the step. A finite gradient
    does not guarantee a finite update, and a non-finite weight reads as
    "moved" to the check below.
``every_trained_tensor_moved``
    A per-tensor float64 fingerprint (sum, sum of squares, sum of absolutes)
    before and after the step; any trained tensor whose fingerprint is
    unchanged did not receive the update.
``frozen_control_unchanged``
    ``model.embed_tokens`` is frozen as a negative control -- the largest single
    tensor -- and must be bit-identical, byte for byte, to the copy taken at
    load: an order-insensitive fingerprint would pass a permuted tensor. Without a frozen set,
    "only what the optimiser touched changed" has nothing to contrast against,
    so the run refuses to start without one.
``gradient_is_finite_and_nonzero``
    A zero or NaN gradient norm means there is no update to verify.

A failed check **stops the run before that iteration's checkpoint is written**,
so ``checkpoint-last`` is always the last iteration that passed, and the exit
status is non-zero. Each checkpoint is written beside its name and swapped in
by rename (:func:`publish_checkpoint`), so a crash mid-save never leaves a
partial tree under that name. ``checkpoint-best`` holds the weights that
*sampled* the best-scoring iteration's rollouts -- that iteration's starting
weights, saved before its update -- because those are the weights the reward
measured.

Advisory, recorded but not gating: ``step_descends_the_gradient``, the cosine
between the realised delta and ``-grad`` on an audit subset of eight tensors.
Adam's per-coordinate normalisation does not guarantee a positive cosine, so a
negative value is not a defect; it is, in practice, the earliest sign that the
KL term outweighs the policy gradient, and it shows up before the reward
curve flattens.

After the last iteration, the ``checkpoint-last`` tree is reloaded from disk in
fp32 and the audit tensors compared with the in-memory weights. The
elementwise check of the whole pair -- including a ceiling on how far Adam can
move any element -- is ``verify_checkpoint_delta.py``, which needs no GPU.

Robustness: non-finite sampling rows
====================================
:class:`FiniteLogits` runs last in the logits-processor chain and turns a NaN
in the sampling distribution into a recorded event rather than letting it reach
``torch.multinomial``, which rejects it with an asynchronous device-side assert
that surfaces far from its cause. Every repair is counted and printed.

Chaining
========
``--init-from`` starts from a checkpoint instead of the base model and does not
rewrite ``checkpoint-pre``; ``--iteration-offset`` numbers the iterations so a
chained run's ``wire.jsonl`` concatenates into one series. Each link writes to
its own ``--out``: a directory already holding a run's artifacts is refused. ⚠ The Adam moments
are **not** carried across a chain -- they are two more copies of the model and
are not written to disk -- so each link restarts with a first step of exactly
``lr`` per element. That is a real discontinuity and belongs beside any chained
curve.

Usage
-----

    python examples/rl/train_grpo_step.py --corpus-root <corpus-dir> \\
        --out <run-dir> --iterations 4 --kl-beta 1e-3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import episode_env  # noqa: E402
from episode_env import Sample  # noqa: E402

ADV_EPS = 1e-4  # the GRPO normaliser's epsilon
DDOF = 1

#: A failed check, or a run that completed no iteration.
EXIT_FAILED = 1
#: Refused to start: the configuration could not produce a verifiable update.
EXIT_REFUSED = 3

#: Frozen as the negative control. See the module docstring.
FROZEN_PREFIXES = ("model.embed_tokens",)

#: The checks that gate a checkpoint. `step_descends_the_gradient` is advisory.
GATING_CHECKS = (
    "every_trained_tensor_moved",
    "trained_tensors_are_finite",
    "frozen_control_unchanged",
    "gradient_is_finite_and_nonzero",
)


def advantages(values: list[float]) -> tuple[list[float], float, float]:
    """GRPO's within-group normalisation: ``(r - mean) / (sd + eps)``.

    A group of one has no spread, so its advantage is zero rather than a
    division by an undefined standard deviation.
    """
    mean = sum(values) / len(values)
    if len(values) > DDOF:
        sd = (sum((v - mean) ** 2 for v in values) / (len(values) - DDOF)) ** 0.5
    else:
        sd = 0.0
    return [(v - mean) / (sd + ADV_EPS) for v in values], mean, sd


def chat_prompt(tok: Any, user: str) -> str:
    """The templated prompt. ``enable_thinking=False`` is not optional.

    Qwen3 is a thinking model by default and would spend the token budget
    inside ``<think>``; the reply would then fail to parse for a reason that
    has nothing to do with the policy.
    """
    return tok.apply_chat_template(
        [{"role": "system", "content": episode_env.SYSTEM},
         {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class FiniteLogits:
    """Turn a non-finite sampling distribution into a recorded event.

    ⚠ **Runs LAST**, because ``generate`` appends custom processors after the
    built-in ones, so ``-inf`` entries are the *expected* output of top-p
    filtering and must not be treated as damage. NaN and ``+inf`` are: softmax
    over a row holding ``+inf`` is ``inf / inf`` = NaN, so a ``+inf`` beside
    finite logits crashes ``multinomial`` exactly as a NaN does. So is a row
    with no finite entry at all, which top-p cannot produce on its own.

    The repair is the least inventive one available. A NaN or ``+inf`` becomes
    ``-inf``, a token that will not be drawn. ``+inf`` is not read as "certain":
    a logit that overflowed is damage, and promoting it to probability 1 would
    make the guard choose the token. A row with nothing finite left is ended at
    EOS rather than sampled from a flattened distribution, because flattening
    makes the guard choose the token and the policy is supposed to.

    ⚠ Read the counts as (decode step x sequence) pairs, not as completions: a
    sequence whose forward pass has gone non-finite stays in the batch after it
    is ended and contributes a hit on every remaining decode step of the call.
    The parse fraction is the honest measure of how much text was touched.
    """

    def __init__(self, eos_token_id: int | None) -> None:
        self.eos_token_id = eos_token_id
        self.nan_steps = 0
        self.nan_rows = 0
        self.dead_rows = 0

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        nan = torch.isnan(scores) | (scores == float("inf"))
        if bool(nan.any()):
            self.nan_steps += 1
            self.nan_rows += int(nan.any(dim=-1).sum())
            scores = torch.where(nan, torch.full_like(scores, float("-inf")), scores)
        dead = ~torch.isfinite(scores).any(dim=-1)
        if bool(dead.any()) and self.eos_token_id is not None:
            self.dead_rows += int(dead.sum())
            scores = scores.clone()
            scores[dead, :] = float("-inf")
            scores[dead, self.eos_token_id] = 0.0
        return scores


class Completion(str):
    """A decoded reply that still carries the token IDs that were sampled.

    The policy gradient has to be taken over the action that was drawn, and
    the action is the token sequence, not its text: decode followed by encode
    is not guaranteed to give the same IDs back (a non-canonical segmentation
    re-encodes canonically, and a skipped special token disappears), so
    re-tokenising the text can apply the gradient to a sequence the policy did
    not sample. A ``str`` subclass rather than a new field, because the
    environment passes the reply through unchanged -- the same object reaches
    ``Sample.completion`` -- so the IDs travel with the text without the
    torch-free environment learning about tokens. ``sample_loss`` uses them
    when present and re-tokenises only a plain ``str``.
    """

    token_ids: tuple[int, ...]

    def __new__(cls, text: str, token_ids: Any) -> Completion:
        obj = super().__new__(cls, text)
        obj.token_ids = tuple(int(t) for t in token_ids)
        return obj


def sampled_ids(generated: list[int], stop_ids: set[int]) -> list[int]:
    """The IDs a sequence actually sampled: through its first stop token.

    ``generate`` pads a finished sequence out to the batch's longest, so
    everything after the first stop token is padding, not policy output. The
    stop token itself is kept: ending the reply is a decision the policy made
    and it is part of the action. A sequence with no stop token hit the length
    cap and every token in it was sampled.
    """
    for index, token in enumerate(generated):
        if token in stop_ids:
            return generated[: index + 1]
    return list(generated)


def _stop_ids(model: Any, tok: Any) -> set[int]:
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    stops = set(configured) if isinstance(configured, (list, tuple)) else {configured}
    stops.add(tok.eos_token_id)
    return {int(t) for t in stops if t is not None}


def generate(model: Any, tok: Any, prompts: list[str], args: Any) -> list[Completion]:
    """Sample one completion per prompt, in left-padded batches of ``gen_batch``.

    Batching across *different* prompts is what makes episodes affordable, and
    left padding is what makes slicing the generated half at the padded prompt
    width correct for a ragged batch. Each reply is a :class:`Completion`, so
    the loss is taken over the sampled IDs rather than a re-tokenisation.
    """
    guard = FiniteLogits(tok.eos_token_id)
    stops = _stop_ids(model, tok)
    out_texts: list[Completion] = []
    for start in range(0, len(prompts), args.gen_batch):
        chunk = prompts[start : start + args.gen_batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad(), torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16):
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                logits_processor=[guard],
            )
        width = enc["input_ids"].shape[1]
        for seq in out:
            ids = sampled_ids(seq[width:].tolist(), stops)
            out_texts.append(Completion(tok.decode(ids, skip_special_tokens=True), ids))
    if guard.nan_steps or guard.dead_rows:
        # Printed, never swallowed: a repaired row is partly the guard's text
        # rather than the policy's, so the count travels with any number
        # computed from these completions.
        print(f"  [logits-guard] repaired {guard.nan_rows} NaN row(s) across "
              f"{guard.nan_steps} decode step(s); {guard.dead_rows} row(s) had no "
              f"finite logit and were ended at EOS", flush=True)
    return out_texts


def episode_rollouts(
    model: Any,
    tok: Any,
    scenarios: list[episode_env.Scenario],
    args: Any,
    policy: Any,
    *,
    log_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[Sample], list[dict[str, Any]]]:
    """Multi-step episodes. Everything except generation lives in the env.

    ``episode_env.rollout_scenario`` imports no ``torch``, so the environment,
    the scoring, the flattening and the group statistics are tested on a CPU;
    this wrapper contributes the one thing that needs the device. Prompts are
    chat-templated here because the template is a property of the model being
    trained, and the environment is model-agnostic.
    """
    groups: list[dict[str, Any]] = []
    samples: list[Sample] = []
    wire: list[dict[str, Any]] = []
    for scenario in scenarios:
        group, scenario_samples, scenario_wire = episode_env.rollout_scenario(
            scenario,
            args.group,
            policy,
            lambda users: generate(model, tok, [chat_prompt(tok, u) for u in users], args),
            advantage_fn=advantages,
            log_root=log_root,
            log_first=args.log_episodes,
        )
        groups.append(group)
        samples.extend(scenario_samples)
        wire.extend(scenario_wire)
        print(f"    [rollout] {scenario.scenario_id:<22} reward {group['reward_mean']:+.4f} "
              f"steps/ep {group['mean_steps']:.2f} conv {group['converged_rate']:.2f}",
              flush=True)
    return groups, samples, wire


def sample_loss(
    model: Any,
    tok: Any,
    sample: Sample,
    total: int,
    *,
    device: str,
    reference: Any = None,
    reference_device: str | None = None,
    kl_beta: float = 0.0,
) -> tuple[Any, dict[str, float]] | None:
    """The loss for one (prompt, completion) sample, or None if it has none.

    ``-(A / total) * sum(log pi(completion))`` plus, with a reference,
    ``(kl_beta / total) * sum(k3)``. ``total`` is the number of samples, so an
    episode of depth *d* contributes *d* terms under one advantage and the loss
    scale does not jump when episodes get longer. The completion is scored as
    the IDs that were sampled when it is a :class:`Completion`, and re-tokenised
    only when it is plain text. An empty completion
    contributes nothing. A zero advantage -- every sample of a group whose
    rewards are all equal -- contributes nothing *only when there is no KL
    term*: the KL penalty does not depend on the advantage, and skipping it
    would leave exactly the flat groups with no anchor to the reference.
    """
    with_kl = reference is not None and kl_beta > 0.0
    if sample.advantage == 0.0 and not with_kl:
        return None
    prompt_ids = tok(chat_prompt(tok, sample.prompt), return_tensors="pt")["input_ids"]
    sampled = getattr(sample.completion, "token_ids", None)
    if sampled is not None:
        comp_ids = torch.tensor([list(sampled)], dtype=prompt_ids.dtype)
    else:
        comp_ids = tok(sample.completion, return_tensors="pt",
                       add_special_tokens=False)["input_ids"]
    if comp_ids.shape[1] == 0:
        return None
    ids = torch.cat([prompt_ids, comp_ids], dim=1).to(device)
    logits = model(input_ids=ids).logits[:, :-1, :].float()
    targets = ids[:, 1:]
    picked = torch.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    comp_lp = picked[:, prompt_ids.shape[1] - 1 :]
    loss = -(sample.advantage / total) * comp_lp.sum()
    stats = {"nll": float(-comp_lp.sum().item()), "kl_sum": 0.0, "kl_tokens": 0.0,
             "pg_abs": float(abs((sample.advantage / total) * comp_lp.sum().item()))}
    if with_kl:
        ref_device = reference_device or device
        with torch.no_grad():
            ref_logits = reference(input_ids=ids.to(ref_device)).logits[:, :-1, :].float()
            ref_lp = torch.log_softmax(ref_logits, dim=-1).gather(
                2, targets.unsqueeze(-1).to(ref_device)
            ).squeeze(-1)[:, prompt_ids.shape[1] - 1 :].to(comp_lp.device)
        logratio = ref_lp - comp_lp
        kl = torch.exp(logratio) - logratio - 1.0
        loss = loss + (kl_beta / total) * kl.sum()
        stats["kl_sum"] = float(kl.sum().item())
        stats["kl_tokens"] = float(kl.numel())
    return loss, stats


def fingerprint(t: Any) -> tuple[float, float, float]:
    """(sum, sum of squares, sum of absolutes) in float64: did this tensor move.

    Three moments rather than one, because a change that preserves the sum --
    a permutation, a zero-sum perturbation -- does not preserve all three.
    """
    f = t.detach().double()
    return float(f.sum().item()), float(f.pow(2).sum().item()), float(f.abs().sum().item())


def bit_identical(a: Any, b: Any) -> bool:
    """Whether two tensors hold exactly the same bits.

    The frozen control's claim is "bit-identical", and neither a fingerprint
    nor ``torch.equal`` makes it: three aggregate moments are unchanged by a
    permutation, and ``torch.equal`` treats ``-0.0`` as ``0.0`` and NaN as
    unequal to itself. Comparing the raw bytes makes it exactly. A tensor that
    is not contiguous is copied first, which only the check pays for.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(
        torch.equal(
            a.detach().contiguous().view(torch.uint8),
            b.detach().contiguous().view(torch.uint8),
        )
    )


def update_checks(
    fp_pre: dict[str, tuple[float, float, float]],
    fp_post: dict[str, tuple[float, float, float]],
    frozen_identical: dict[str, bool],
    grad_norm: float,
    cosine: float,
) -> dict[str, dict[str, Any]]:
    """The per-iteration checks, as data. Pure, so every arm is testable.

    A frozen set that is empty fails ``frozen_control_unchanged`` rather than
    passing it vacuously: "the control did not move" and "there was no
    control" must not read the same.

    ``trained_tensors_are_finite`` is its own check because "moved" cannot see
    it: a finite gradient with an update that overflows leaves NaN or inf in
    the weights, a NaN fingerprint compares unequal to everything, and the
    tensor reads as having moved. It is read off the post-step fingerprints --
    a sum over a tensor holding a NaN or an inf is itself non-finite -- so it
    costs no second pass over the weights.
    """
    unmoved = sorted(n for n in fp_pre if fp_pre[n] == fp_post.get(n))
    non_finite = sorted(
        n for n, moments in fp_post.items() if not all(math.isfinite(m) for m in moments)
    )
    frozen_moved = sorted(n for n, same in frozen_identical.items() if not same)
    finite = grad_norm == grad_norm and grad_norm not in (float("inf"), float("-inf"))
    return {
        "every_trained_tensor_moved": {
            "tensors": len(fp_pre),
            "unmoved": len(unmoved),
            "unmoved_examples": unmoved[:5],
            "passed": bool(fp_pre) and not unmoved,
        },
        "trained_tensors_are_finite": {
            "non_finite": len(non_finite),
            "non_finite_examples": non_finite[:5],
            "passed": bool(fp_post) and not non_finite,
        },
        "frozen_control_unchanged": {
            "tensors": sorted(frozen_identical),
            "moved": frozen_moved,
            "passed": bool(frozen_identical) and not frozen_moved,
        },
        "gradient_is_finite_and_nonzero": {
            "grad_norm": grad_norm,
            "passed": finite and grad_norm > 0.0,
        },
        "step_descends_the_gradient": {
            "cosine_delta_vs_negative_grad": cosine,
            "passed": cosine > 0.0,
            "advisory": True,
        },
    }


#: What a run writes into ``--out``. Any of them already there means the
#: directory belongs to another run.
RUN_ARTIFACTS = ("wire.jsonl", "train-log.json", "checkpoint-pre", "checkpoint-last",
                 "checkpoint-best", "episodes",
                 "checkpoint-last.partial", "checkpoint-last.previous",
                 "checkpoint-best.partial", "checkpoint-best.previous")


def publish_checkpoint(model: Any, tok: Any, dest: Path) -> None:
    """Write a checkpoint so ``dest`` is only ever a complete tree.

    ``save_pretrained`` into ``dest`` itself overwrites it file by file, so a
    preemption or a full disk mid-save leaves a partial tree under the name
    that promises the last verified iteration. The new tree is written to
    ``<dest>.partial`` and swapped in by rename; the old one is kept as
    ``<dest>.previous`` until the swap completes. A crash leaves either the
    old ``dest``, or no ``dest`` and a complete ``<dest>.previous`` -- never a
    partial tree named ``dest``.
    """
    partial = dest.with_name(dest.name + ".partial")
    previous = dest.with_name(dest.name + ".previous")
    if partial.exists():
        shutil.rmtree(partial)
    model.save_pretrained(partial, safe_serialization=True)
    tok.save_pretrained(partial)
    if dest.exists():
        if previous.exists():
            shutil.rmtree(previous)
        dest.rename(previous)
    partial.rename(dest)
    if previous.exists():
        shutil.rmtree(previous)


def update_best(best: dict[str, Any], reward_mean: float, iteration: int,
                model: Any, tok: Any, dest: Path) -> bool:
    """Publish ``model`` as ``checkpoint-best`` if its rollouts scored best so far.

    Called after an iteration's rollouts and **before** its optimiser step,
    so ``checkpoint-best`` holds the weights that earned the reward it is
    ranked by: those that sampled iteration ``iteration``, i.e. the start of
    that iteration. Saving after the step, as this once did, kept weights one
    update past the ones measured -- never evaluated, and possibly worse.
    """
    if not reward_mean > best["reward_mean"]:
        return False
    publish_checkpoint(model, tok, dest)
    best.update(reward_mean=reward_mean, iteration=iteration,
                weights="as sampled at the start of this iteration, before its update")
    return True


def check_output_dir(out: Path) -> str | None:
    """A refusal message if ``--out`` already holds another run's artifacts.

    ``wire.jsonl`` is appended to row by row and ``train-log.json`` rewritten,
    so reusing a directory would merge two runs' rows under duplicate
    (iteration, scenario, episode) keys -- which every reader of the wire then
    groups together -- while the log and the checkpoints describe only the
    second run. A chained link therefore gets its **own** ``--out``;
    ``--iteration-offset`` is what makes the wires of successive links
    concatenate into one series, so there is no legitimate append into an
    existing directory to allow. An empty directory, or one holding other
    files, is fine.
    """
    present = sorted(name for name in RUN_ARTIFACTS if (out / name).exists())
    if present:
        return (
            f"{out} already holds {present} from another run; give this run its own "
            "--out (a chained link too -- --iteration-offset lines the wires up)"
        )
    return None


def validate_args(args: argparse.Namespace) -> str | None:
    """A refusal message for a configuration that cannot produce a checked update."""
    if args.iterations < 1:
        return "--iterations must be >= 1"
    if args.group < 2:
        return "--group must be >= 2: a group of one has no advantage to learn from"
    # Each of these reaches the model only after it has loaded: `--gen-batch 0`
    # is `range(..., step=0)`, and `--max-episode-steps 0` or
    # `--max-new-tokens 0` produce episodes with nothing to learn from.
    for flag, value, floor in (("--max-episode-steps", args.max_episode_steps, 1),
                               ("--max-new-tokens", args.max_new_tokens, 1),
                               ("--gen-batch", args.gen_batch, 1),
                               ("--log-episodes", args.log_episodes, 0),
                               ("--iteration-offset", args.iteration_offset, 0)):
        if value < floor:
            return f"{flag} must be >= {floor}"
    if not (math.isfinite(args.lr) and args.lr > 0):
        return "--lr must be finite and > 0"
    if not (math.isfinite(args.adam_eps) and args.adam_eps > 0):
        return "--adam-eps must be finite and > 0"
    if not (math.isfinite(args.min_parse_frac) and 0.0 <= args.min_parse_frac <= 1.0):
        # NaN would make `parse_fraction < min_parse_frac` always false and
        # silently disable the collapse stop.
        return "--min-parse-frac must be finite and in [0, 1]"
    if not (math.isfinite(args.budget_sec) and args.budget_sec > 0):
        return "--budget-sec must be finite and > 0"
    if not (math.isfinite(args.kl_beta) and args.kl_beta >= 0):
        return "--kl-beta must be finite and >= 0; a negative coefficient pays the policy to drift"
    if not (math.isfinite(args.clip) and args.clip >= 0):
        # A negative clip takes the clipping branch for every positive norm and
        # makes the coefficient negative, reversing every gradient -- an ascent
        # step the advisory cosine check would report but not stop.
        return "--clip must be finite and >= 0 (0 disables clipping)"
    if not 0.0 < args.top_p <= 1.0:
        return "--top-p must be in (0, 1]"
    if not (math.isfinite(args.temperature) and args.temperature > 0):
        return "--temperature must be finite and > 0; sampling at 0 has no spread to normalise"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--corpus-root", default=None,
                        help=f"directory holding the archived matrices "
                             f"(default: ${episode_env.CORPUS_ROOT_ENV})")
    parser.add_argument("--scenarios", default="",
                        help="comma-separated scenario ids to keep; an unknown id is "
                             "an error, not a silent drop")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--group", type=int, default=24,
                        help="episodes per scenario per iteration")
    parser.add_argument("--max-episode-steps", type=int, default=8,
                        help="AgentPolicy.max_iterations, i.e. proposal cycles per "
                             "episode; 8 is the shipped default")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--betas", default="0.9,0.999")
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--clip", type=float, default=0.0,
                        help="global gradient-norm clip; 0 disables it")
    parser.add_argument("--kl-beta", type=float, default=0.0,
                        help="coefficient on KL to the frozen reference; 0 disables it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-device", default="cuda:1",
                        help="the frozen reference lives on its own device so it "
                             "costs the policy no memory")
    parser.add_argument("--param-dtype", choices=("bfloat16", "float32"), default="float32",
                        help="float32 keeps the served weights identical to the "
                             "optimiser's, so no delta is lost to rounding between "
                             "the step and the checkpoint")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="off by default. When on it is non-reentrant: reentrant "
                             "checkpointing severs the backward graph into the "
                             "decoder layers, so only the tensors outside them move")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--gen-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--min-parse-frac", type=float, default=0.25,
                        help="stop if the share of replies that parse as JSON falls "
                             "below this: the earliest sign the policy is collapsing")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="start from this checkpoint instead of the base model, "
                             "and do not rewrite checkpoint-pre. Adam moments are not "
                             "carried across (see the module docstring)")
    parser.add_argument("--iteration-offset", type=int, default=0,
                        help="number iterations from here, so a chained run's "
                             "wire.jsonl concatenates into one series")
    parser.add_argument("--log-episodes", type=int, default=2,
                        help="write a real agent_log.jsonl for this many episodes per "
                             "scenario per iteration; wire.jsonl holds every reply")
    parser.add_argument("--budget-sec", type=float, default=4800.0,
                        help="stop starting new iterations past this; checkpoint-last "
                             "is written after every iteration that passes its checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    refusal = validate_args(args) or check_output_dir(args.out)
    if refusal:
        print(f"[refused] {refusal}", file=sys.stderr)
        return EXIT_REFUSED

    # "x": exclusive create, before anything else is written. check_output_dir
    # is only a preflight; two runs can both pass it, and whichever loses this
    # create stops here instead of writing a checkpoint-pre over the other's.
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        wire = (args.out / "wire.jsonl").open("x", encoding="utf-8")
    except FileExistsError:
        print(f"[refused] {args.out} already holds ['wire.jsonl']: another run took it "
              "after the preflight", file=sys.stderr)
        return EXIT_REFUSED
    code = EXIT_FAILED
    try:
        code = _train(args, wire)
        return code
    finally:
        empty = wire.tell() == 0
        wire.close()
        if code == EXIT_REFUSED and empty:
            # Ours by the exclusive create, and nothing was written: a refused
            # run should not leave --out refusing the next attempt.
            (args.out / "wire.jsonl").unlink()


def _train(args: argparse.Namespace, wire: Any) -> int:  # noqa: C901 - one linear training loop
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from aorta.agent.policy import AgentPolicy

    # The candidate filter warns once per dropped name, which is right for an
    # operator and thousands of lines per iteration here; every drop is in the
    # episode's own record.
    logging.getLogger("aorta.agent.llm").setLevel(logging.ERROR)

    started = time.time()
    torch.manual_seed(args.seed)
    policy = AgentPolicy(max_iterations=args.max_episode_steps)
    only = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    scenarios = episode_env.load_corpus(args.corpus_root, only=only)
    print(f"[corpus] {len(scenarios)} scenarios, {len(scenarios[0].offered)} offered "
          f"candidates, G={args.group}", flush=True)
    for scenario in scenarios:
        print(f"  [corpus] {scenario.scenario_id:<22} {len(scenario.grid.verdicts):>3} cells  "
              f"resolvers={sorted(scenario.resolvers) or '(unresolvable)'}", flush=True)

    print(f"[load] {args.init_from or args.model} in {args.param_dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.init_from) if args.init_from else args.model,
        torch_dtype=getattr(torch, args.param_dtype),
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.config.use_cache = True

    trained: dict[str, torch.Tensor] = {}
    frozen: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if name.startswith(FROZEN_PREFIXES):
            p.requires_grad_(False)
            frozen[name] = p
        else:
            p.requires_grad_(True)
            trained[name] = p
    print(f"[params] trained {len(trained)} tensors / "
          f"{sum(p.numel() for p in trained.values()) / 1e9:.3f} B; frozen {sorted(frozen)}",
          flush=True)
    if not frozen:
        print("[refused] no frozen tensor, so 'only what the optimiser touched' has no "
              "control", file=sys.stderr)
        return EXIT_REFUSED
    # One exact copy of the frozen control, taken before any step, on the same
    # device: the control is the largest single tensor, and comparing it where
    # it lives costs one extra copy of it rather than a transfer per iteration.
    frozen_reference = {n: p.detach().clone() for n, p in frozen.items()}

    # Eight tensors spanning depth, kept exactly so the realised delta can be
    # compared with the gradient that produced it.
    all_trained = sorted(trained)
    audit_names = all_trained[:: max(1, len(all_trained) // 8)][:8]

    reference = None
    if args.kl_beta > 0.0:
        print(f"[load] frozen reference on {args.reference_device}", flush=True)
        reference = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="sdpa",
        ).to(args.reference_device)
        reference.eval()
        reference.config.use_cache = False
        for rp in reference.parameters():
            rp.requires_grad_(False)

    if args.init_from is None:
        pre_dir = args.out / "checkpoint-pre"
        print(f"[ckpt] writing PRE -> {pre_dir}", flush=True)
        model.save_pretrained(pre_dir, safe_serialization=True)
        tok.save_pretrained(pre_dir)

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    betas = tuple(float(x) for x in args.betas.split(","))
    opt = torch.optim.Adam(list(trained.values()), lr=args.lr, betas=betas,
                           eps=args.adam_eps, weight_decay=0.0)

    log: dict[str, Any] = {
        "route": "local transformers, one GPU, in-process generation and optimiser step",
        "model": args.model,
        "param_dtype": args.param_dtype,
        "optimiser": "adam",
        "lr": args.lr,
        "betas": list(betas),
        "clip": args.clip,
        "kl_beta": args.kl_beta,
        "group": args.group,
        "max_episode_steps": args.max_episode_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "iteration_offset": args.iteration_offset,
        "init_from": str(args.init_from) if args.init_from else None,
        "scenarios": [s.scenario_id for s in scenarios],
        # What the run was scored against, so a later --replay can refuse a
        # corpus whose archives changed under the same scenario IDs.
        "corpus": {s.scenario_id: s.digest for s in scenarios},
        "trained_tensors": len(trained),
        "frozen_tensors": sorted(frozen),
        "iterations": [],
    }
    best = {"reward_mean": float("-inf"), "iteration": None}
    status = 0

    def write_log() -> None:
        log["elapsed_sec"] = round(time.time() - started, 1)
        (args.out / "train-log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    try:
        for it in range(args.iteration_offset + 1, args.iteration_offset + args.iterations + 1):
            if time.time() - started > args.budget_sec:
                print(f"[budget] {args.budget_sec}s spent; not starting iteration {it}",
                      flush=True)
                break
            row: dict[str, Any] = {"iteration": it}
            t_it = time.time()

            # ---- rollouts, from the CURRENT policy ------------------------
            model.config.use_cache = True
            model.gradient_checkpointing_disable()
            model.eval()
            # Release the previous iteration's gradients BEFORE generating
            # rather than at the next backward pass: holding a full fp32
            # gradient across the rollout phase is memory that generation
            # needs, and episode prompts are long and batched ragged.
            for p in trained.values():
                p.grad = None
            torch.cuda.empty_cache()
            groups, samples, wire_rows = episode_rollouts(
                model, tok, scenarios, args, policy,
                log_root=(args.out / "episodes" / f"it{it:03d}") if args.log_episodes else None,
            )
            row["groups"] = groups
            for record in wire_rows:
                wire.write(json.dumps({"iteration": it, **record}) + "\n")
            wire.flush()
            row["reward_mean"] = round(sum(g["reward_mean"] for g in groups) / len(groups), 4)
            # Thresholded rather than `sd > 0`: floating-point residue in the
            # mean made an exactly flat group read as having spread.
            row["groups_with_real_spread"] = sum(g["advantage_spread"] > 1e-6 for g in groups)
            # Denominated in STEPS: a group is `group` episodes carrying many
            # more replies, and dividing by episodes would report a fraction
            # above 1.0 and silently disarm the collapse guard.
            row["steps_total"] = sum(g["steps_total"] for g in groups)
            row["parse_fraction"] = round(
                sum(g["parsed_json"] for g in groups) / max(row["steps_total"], 1), 4
            )
            print(f"  [it{it}] mean reward {row['reward_mean']:+.4f}  groups with spread "
                  f"{row['groups_with_real_spread']}/{len(groups)}  parse fraction "
                  f"{row['parse_fraction']:.3f}", flush=True)
            if row["parse_fraction"] < args.min_parse_frac:
                print(f"[collapse] parse fraction {row['parse_fraction']:.3f} below "
                      f"{args.min_parse_frac}: stopping", file=sys.stderr, flush=True)
                row["verdict"] = "policy collapse"
                log["iterations"].append(row)
                status = EXIT_FAILED
                break

            # Before the step: the reward was earned by the weights that
            # sampled these rollouts, so those are the weights to keep.
            update_best(best, row["reward_mean"], it, model, tok, args.out / "checkpoint-best")
            log["best"] = dict(best)

            # ---- the policy-gradient step ---------------------------------
            model.config.use_cache = False
            if args.grad_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            model.train()
            total = len(samples)
            sums = {"nll": 0.0, "kl_sum": 0.0, "kl_tokens": 0.0, "pg_abs": 0.0}
            for sample in samples:
                result = sample_loss(
                    model, tok, sample, total, device=args.device, reference=reference,
                    reference_device=args.reference_device, kl_beta=args.kl_beta,
                )
                if result is None:
                    continue
                loss, stats = result
                loss.backward()
                for key in sums:
                    sums[key] += stats[key]
                del loss

            sq = torch.zeros((), dtype=torch.float64, device=args.device)
            for p in trained.values():
                if p.grad is not None:
                    sq += p.grad.detach().double().pow(2).sum()
            grad_norm = float(sq.sqrt().item())
            coef = args.clip / (grad_norm + 1e-12) if args.clip and grad_norm > args.clip else 1.0
            row.update(
                grad_norm=grad_norm,
                clip_coefficient=coef,
                mean_completion_nll=round(sums["nll"] / max(total, 1), 4),
                mean_kl_per_token=(round(sums["kl_sum"] / sums["kl_tokens"], 6)
                                   if sums["kl_tokens"] else None),
                # On the loss's own scale the two terms are `total * pg_abs`
                # and `kl_beta * kl_sum`, since each per-sample term divides
                # by `total` once.
                kl_share_of_loss=(
                    round(args.kl_beta * sums["kl_sum"]
                          / (total * sums["pg_abs"] + args.kl_beta * sums["kl_sum"]), 6)
                    if sums["kl_tokens"] and (sums["pg_abs"] or sums["kl_sum"]) else None
                ),
            )
            print(f"  [it{it}] grad norm {grad_norm:.6g} clip coef {coef:.6g}"
                  + (f"  KL/token {row['mean_kl_per_token']:.6f} share "
                     f"{row['kl_share_of_loss']:.5f}" if row["mean_kl_per_token"] is not None
                     else ""), flush=True)

            # ---- Adam, and the audit ---------------------------------------
            fp_pre = {n: fingerprint(t) for n, t in trained.items()}
            audit_pre = {n: trained[n].detach().clone() for n in audit_names}
            audit_grad = {
                n: (trained[n].grad.detach().clone() if trained[n].grad is not None
                    else torch.zeros_like(trained[n]))
                for n in audit_names
            }
            gradient_ok = grad_norm == grad_norm and grad_norm > 0.0
            if gradient_ok:
                if coef != 1.0:
                    for p in trained.values():
                        if p.grad is not None:
                            p.grad.mul_(coef)
                opt.step()
            fp_post = {n: fingerprint(t) for n, t in trained.items()}
            # Against the copy taken at load, byte for byte: the control must
            # never move, so every iteration is checked against the start.
            frozen_identical = {
                n: bit_identical(t, frozen_reference[n]) for n, t in frozen.items()
            }
            dot = nrm_d = nrm_g = 0.0
            for n in audit_names:
                d = (trained[n].detach() - audit_pre[n]).double()
                g = audit_grad[n].double()
                dot += float((d * -g).sum().item())
                nrm_d += float(d.pow(2).sum().item())
                nrm_g += float(g.pow(2).sum().item())
            cosine = dot / ((nrm_d**0.5 * nrm_g**0.5) + 1e-30)
            row["checks"] = update_checks(fp_pre, fp_post, frozen_identical,
                                          grad_norm, cosine)
            for key, check in row["checks"].items():
                mark = "PASS" if check["passed"] else ("warn" if check.get("advisory") else "FAIL")
                print(f"  [it{it}] check {key}: {mark}", flush=True)
            failed = [k for k in GATING_CHECKS if not row["checks"][k]["passed"]]
            row["elapsed_sec"] = round(time.time() - t_it, 1)
            if failed:
                # Before the checkpoint, so checkpoint-last is always the last
                # iteration whose update was verified.
                print(f"[check] {failed} failed at iteration {it}; not writing its "
                      f"checkpoint and stopping", file=sys.stderr, flush=True)
                row["verdict"] = f"checks failed: {failed}"
                log["iterations"].append(row)
                status = EXIT_FAILED
                break

            # Disk policy: pre, last and best only -- a checkpoint per
            # iteration costs a model's size each and adds no information.
            publish_checkpoint(model, tok, args.out / "checkpoint-last")
            log["best"] = dict(best)
            log["last_verified_iteration"] = it
            log["iterations"].append(row)
            write_log()
            print(f"  [it{it}] done in {row['elapsed_sec']}s", flush=True)

        # ---- the reload reproduces checkpoint-last ------------------------
        if log.get("last_verified_iteration") is not None:
            post_dir = args.out / "checkpoint-last"
            want = {n: trained[n].detach().float().cpu() for n in audit_names}
            del model
            torch.cuda.empty_cache()
            # fp32, because that is what is on disk: reloading in bf16 would
            # compare a cast copy against fp32 originals and could not match.
            reloaded = dict(AutoModelForCausalLM.from_pretrained(
                post_dir, torch_dtype=torch.float32, local_files_only=True
            ).named_parameters())
            rows = [{"tensor": n,
                     "max_abs_diff": float((reloaded[n].detach().float().cpu() - want[n])
                                           .abs().max())}
                    for n in audit_names]
            passed = all(r["max_abs_diff"] == 0.0 for r in rows)
            log["reload_reproduces_last"] = {"audited_tensors": rows, "passed": passed}
            print(f"  [verify] reload of checkpoint-last: {'PASS' if passed else 'FAIL'}",
                  flush=True)
            if not passed:
                status = EXIT_FAILED
    finally:
        write_log()

    print(f"\niterations completed: {len(log['iterations'])}")
    for row in log["iterations"]:
        print(f"  it{row['iteration']}: reward {row.get('reward_mean'):+.4f}  "
              f"parse {row.get('parse_fraction')}  grad norm "
              f"{row.get('grad_norm', float('nan')):.6g}  {row.get('verdict', '')}")
    if log.get("last_verified_iteration") is None:
        return EXIT_FAILED
    return status


if __name__ == "__main__":
    sys.exit(main())
