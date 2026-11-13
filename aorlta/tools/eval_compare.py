#!/usr/bin/env python3
"""Compare two score runs: a base model and a trained checkpoint, on the same held-out prompts.

Run one score config twice (`orchestrator.mode: score`), once on the base model and once with `model.adapter_dir`
set to the trained adapter (`tools/export_adapter.py`), each under its own `run.root` (a site file sets both), then:

  eval_compare.py --base BASE_RUN_DIR --trained TRAINED_RUN_DIR [--tokenizer MODEL_DIR] [--samples 6] [--out DIR]

It prints the pass rate (the share of samples whose reward claims clean: a correct answer), the mean reward and the
scored fraction of each, and the per-prompt change, and writes `<out>/eval-compare.json` and `<out>/eval-compare.md`.
The markdown holds, for the prompts that changed most, one sample of each model: its reward, its checks' states and,
with `--tokenizer`, its final reply decoded. It refuses two runs that did not score the same prompts with the same
sampling settings, since their numbers would not be comparable. An unscored sample is never counted as passing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


class Refused(ValueError):
    pass


def read_report(run_dir: Path) -> dict:
    path = Path(run_dir) / "score" / "report.jsonl"
    if not path.is_file():
        raise Refused(f"{run_dir}: no score/report.jsonl (is it a finished score run?)")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    run = next((r for r in rows if r.get("kind") == "run"), None)
    total = next((r for r in rows if r.get("kind") == "total"), None)
    prompts = {r["prompt_id"]: r for r in rows if r.get("kind") == "prompt"}
    if run is None or total is None or not prompts:
        raise Refused(f"{path}: no run, total or prompt rows")
    return {"run": run, "total": total, "prompts": prompts}


def episodes(run_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted((Path(run_dir) / "iterations").glob("iter-*/episodes.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            out.setdefault(e["prompt_id"], []).append(e)
    return out


def headline(rep: dict) -> dict:
    t = rep["total"]
    return {"samples": t["samples"], "scored": t["scored"], "scored_fraction": t["scored_fraction"],
            "pass_rate": t["clean"] / t["samples"] if t["samples"] else 0.0, "clean": t["clean"],
            "mean_reward": (t.get("reward") or {}).get("mean"),
            "model": rep["run"].get("model", {}).get("kind")}


def compare(base_dir: Path, trained_dir: Path) -> dict:
    base, trained = read_report(base_dir), read_report(trained_dir)
    if set(base["prompts"]) != set(trained["prompts"]):
        raise Refused("the two runs did not score the same prompts")
    sb, st = base["run"].get("sampling"), trained["run"].get("sampling")
    if sb != st:
        raise Refused(f"the two runs sampled differently: {sb} against {st}")
    rows = []
    for pid in sorted(base["prompts"]):
        b, t = base["prompts"][pid], trained["prompts"][pid]
        rb, rt = (b.get("reward") or {}).get("mean"), (t.get("reward") or {}).get("mean")
        rows.append({"prompt_id": pid, "base_pass": b["clean"] / b["samples"], "trained_pass": t["clean"] / t["samples"],
                     "base_reward": rb, "trained_reward": rt,
                     "delta_pass": t["clean"] / t["samples"] - b["clean"] / b["samples"]})
    hb, ht = headline(base), headline(trained)
    return {"base": {"run_dir": str(base_dir), **hb}, "trained": {"run_dir": str(trained_dir), **ht},
            "delta": {"pass_rate": ht["pass_rate"] - hb["pass_rate"],
                      "mean_reward": (None if hb["mean_reward"] is None or ht["mean_reward"] is None
                                      else ht["mean_reward"] - hb["mean_reward"])},
            "prompts": rows, "sampling": sb,
            "improved": sum(r["delta_pass"] > 0 for r in rows), "worse": sum(r["delta_pass"] < 0 for r in rows)}


def sample_text(ep: dict, tok) -> str:
    ids = ep["turns"][-1]["response_ids"]
    if tok is None:
        return f"({len(ids)} reply tokens; pass --tokenizer to decode)"
    return tok.decode(ids, skip_special_tokens=True).strip()


def markdown(result: dict, base_dir: Path, trained_dir: Path, tok, samples: int) -> str:
    b, t, d = result["base"], result["trained"], result["delta"]
    fmt = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    dr = "-" if d["mean_reward"] is None else f"{d['mean_reward']:+.3f}"
    lines = ["# Base against trained, on the same held-out prompts", "",
             "| | base | trained | change |", "|---|---|---|---|",
             f"| pass rate | {b['pass_rate']:.3f} ({b['clean']}/{b['samples']}) | {t['pass_rate']:.3f} "
             f"({t['clean']}/{t['samples']}) | {d['pass_rate']:+.3f} |",
             f"| mean reward | {fmt(b['mean_reward'])} | {fmt(t['mean_reward'])} | {dr} |",
             f"| scored fraction | {b['scored_fraction']:.3f} | {t['scored_fraction']:.3f} | |", "",
             f"Prompts improved: {result['improved']}; worse: {result['worse']}; of {len(result['prompts'])}.", "",
             "## Samples, for the prompts that changed most", ""]
    eb, et = episodes(base_dir), episodes(trained_dir)
    picked = sorted(result["prompts"], key=lambda r: (-abs(r["delta_pass"]), r["prompt_id"]))[:samples]
    for r in picked:
        lines.append(f"### {r['prompt_id']}: pass {r['base_pass']:.2f} -> {r['trained_pass']:.2f}")
        for name, eps in (("base", eb), ("trained", et)):
            got = eps.get(r["prompt_id"]) or []
            if not got:
                continue
            e = got[0]
            lines += ["", f"**{name}**: reward {fmt(e.get('reward'))}, {e['status']}, checks {e.get('legs')}", "",
                      "```text", sample_text(e, tok)[:3000], "```"]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval_compare.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", type=Path, required=True, help="the base model's score run directory")
    ap.add_argument("--trained", type=Path, required=True, help="the trained model's score run directory")
    ap.add_argument("--tokenizer", default=None, help="a model directory whose tokenizer decodes the sample replies")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args(argv)
    try:
        result = compare(args.base, args.trained)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval-compare.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (args.out / "eval-compare.md").write_text(markdown(result, args.base, args.trained, tok, args.samples) + "\n",
                                              encoding="utf-8")
    b, t = result["base"], result["trained"]
    print(f"pass rate {b['pass_rate']:.3f} -> {t['pass_rate']:.3f} ({result['delta']['pass_rate']:+.3f}); "
          f"mean reward {b['mean_reward']} -> {t['mean_reward']}; improved {result['improved']}, worse "
          f"{result['worse']} of {len(result['prompts'])} prompts; written to {args.out}/eval-compare.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
