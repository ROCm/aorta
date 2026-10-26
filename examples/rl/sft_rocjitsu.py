#!/usr/bin/env python3
"""Supervised warm start on demonstrations, with the GRPO trainer's model, numerics and loss.

Every demonstration step is a ``(prompt, completion)`` sample scored by
:func:`train_grpo_step.sample_loss` with advantage +1 and no KL term, so the
objective is the mean over samples of the completion's log-likelihood, computed
under the sampler's numerics. The completion is the reply's tokens followed by
the end-of-turn token, as the policy samples it: ending the reply is part of the
action. ``model.embed_tokens`` stays frozen, the same control the GRPO trainer
keeps, so ``verify_checkpoint_delta.py`` audits this step like any other.

    sft_rocjitsu.py --demos demos.jsonl --corpus-root <corpus> --out <dir> \\
        --init-from <fp32 base> [--epochs 2 --lr 1e-5 --batch 16 --grad-checkpointing]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blas_backend  # noqa: E402
import rocjitsu_env as rj  # noqa: E402
import train_grpo_step as tr  # noqa: E402
from episode_env import Sample  # noqa: E402


def end_of_turn_id(tok: Any) -> int:
    """The token the policy ends a reply with: ``<|im_end|>`` for Qwen's chat template."""
    candidate = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(candidate, int) and candidate != getattr(tok, "unk_token_id", None):
        return candidate
    return int(tok.eos_token_id)


ADAM_BETAS = (0.9, 0.999)


def demo_samples(rows: list[dict[str, Any]], tok: Any) -> list[Sample]:
    """Each demonstration step as a :class:`Sample` whose completion carries its token IDs and the end of turn."""
    stop = end_of_turn_id(tok)
    out = []
    for r in rows:
        ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        out.append(Sample(scenario_id=r["scenario_id"], prompt=r["prompt"],
                          completion=tr.Completion(r["completion"], list(ids) + [stop]), advantage=1.0,
                          episode=r.get("episode", 0), step=r.get("step", 1)))
    return out


def batches(items: list[Any], size: int, rng: random.Random) -> list[list[Any]]:
    order = list(items)
    rng.shuffle(order)
    return [order[i : i + size] for i in range(0, len(order), size)]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demos", type=Path, required=True)
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--split", default="train", help="the split whose system prompt the demonstrations used")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16, help="samples per optimiser step")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--param-dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=20261005)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if (a.epochs < 1 or a.batch < 1 or not (math.isfinite(a.lr) and a.lr > 0)
            or not (math.isfinite(a.clip) and a.clip >= 0)):
        print("[refused] --epochs and --batch must be >= 1, --lr finite and > 0, and --clip finite and >= 0 "
              "(0 disables it)", file=sys.stderr)
        return tr.EXIT_REFUSED
    if a.out.exists() and any(a.out.iterdir()):
        print(f"[refused] {a.out} is not empty", file=sys.stderr)
        return tr.EXIT_REFUSED
    system = rj.system_for(rj.load_corpus(a.corpus_root, split=a.split))
    system_sha = hashlib.sha256(system.encode()).hexdigest()
    stats_path = a.demos.with_suffix(".stats.json")
    recorded = json.loads(stats_path.read_text()).get("system_sha256") if stats_path.exists() else None
    if recorded != system_sha:
        print(f"[refused] the demonstrations were rendered under system prompt {recorded}, this run's is "
              f"{system_sha}", file=sys.stderr)
        return tr.EXIT_REFUSED
    rows = [json.loads(line) for line in a.demos.read_text().splitlines() if line.strip()]
    if not rows:
        print(f"[refused] {a.demos} holds no demonstration; nothing to train on", file=sys.stderr)
        return tr.EXIT_REFUSED
    steps = a.epochs * math.ceil(len(rows) / a.batch)
    try:
        tr.optimiser_bound(a.lr, steps, *ADAM_BETAS)
    except ValueError as exc:
        print(f"[refused] {steps} optimiser steps ({a.epochs} epoch(s) of {len(rows)} rows in batches of {a.batch}) "
              f"is a run verify_checkpoint_delta.py cannot bound: {exc}", file=sys.stderr)
        return tr.EXIT_REFUSED
    a.out.mkdir(parents=True, exist_ok=True)
    backend = blas_backend.apply()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(a.init_from) if a.init_from else a.model,
                                                 torch_dtype=getattr(torch, a.param_dtype),
                                                 local_files_only=True, attn_implementation="sdpa").to(a.device)
    trained = {}
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith(tr.FROZEN_PREFIXES))
        if p.requires_grad:
            trained[name] = p
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    if not frozen_before:
        print(f"[refused] no parameter matches the frozen control {tr.FROZEN_PREFIXES}, so the weight audit "
              "would check nothing", file=sys.stderr)
        return tr.EXIT_REFUSED
    if a.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.Adam(list(trained.values()), lr=a.lr, betas=ADAM_BETAS, eps=1e-8, weight_decay=0.0)

    samples = demo_samples(rows, tok)
    rng = random.Random(a.seed)
    log: dict[str, Any] = {"demos": str(a.demos), "samples": len(samples), "epochs": a.epochs, "lr": a.lr,
                           "batch": a.batch, "clip": a.clip, "param_dtype": a.param_dtype, "system_sha256": system_sha,
                           "init_from": str(a.init_from) if a.init_from else a.model, "blas_backend": backend,
                           "steps": []}
    started, step = time.time(), 0
    for epoch in range(1, a.epochs + 1):
        for batch in batches(samples, a.batch, rng):
            for p in trained.values():
                p.grad = None
            nll = tokens = 0.0
            for s in batch:
                loss, stats = tr.sample_loss(model, tok, s, len(batch), device=a.device, system=system)
                loss.backward()
                nll += stats["nll"]
                tokens += len(s.completion.token_ids)
                del loss
            sq = sum(float(p.grad.detach().double().pow(2).sum()) for p in trained.values() if p.grad is not None)
            norm = sq ** 0.5
            if not math.isfinite(norm) or norm == 0.0:
                print(f"[refused] step {step + 1}: gradient norm {norm}", file=sys.stderr)
                return tr.EXIT_FAILED
            if a.clip and norm > a.clip:
                for p in trained.values():
                    if p.grad is not None:
                        p.grad.mul_(a.clip / norm)
            opt.step()
            step += 1
            row = {"step": step, "epoch": epoch, "samples": len(batch), "nll_per_token": round(nll / max(tokens, 1), 5),
                   "grad_norm": round(norm, 4), "elapsed_sec": round(time.time() - started, 1)}
            log["steps"].append(row)
            if step % 10 == 0 or step == 1:
                print(f"  [sft] {json.dumps(row)}", flush=True)
    for n, p in model.named_parameters():
        if n in frozen_before and not tr.bit_identical(p, frozen_before[n]):
            print(f"[refused] the frozen control {n} moved", file=sys.stderr)
            return tr.EXIT_FAILED
    log["optimizer_steps"] = step
    log["elapsed_sec"] = round(time.time() - started, 1)
    tr.publish_checkpoint(model, tok, a.out / "checkpoint-sft")
    tr.publish_snapshot(model, tok, a.out / "snapshot-sft")
    (a.out / "sft-log.json").write_text(json.dumps(log, indent=1))
    print(f"[sft] done: {step} optimiser steps over {len(samples)} samples in {log['elapsed_sec']}s; "
          f"first nll/token {log['steps'][0]['nll_per_token']}, last {log['steps'][-1]['nll_per_token']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
