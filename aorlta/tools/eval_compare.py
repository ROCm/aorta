#!/usr/bin/env python3
"""Compare two score runs: a base model and a trained checkpoint, on the same held-out prompts.

Run one score config twice (`orchestrator.mode: score`), once on the base model and once with `model.adapter_dir`
set to the trained adapter (`tools/export_adapter.py`), each under its own `run.root` (a site file sets both), then:

  eval_compare.py --base BASE_RUN_DIR --trained TRAINED_RUN_DIR [--tokenizer MODEL_DIR] [--samples 6] [--out DIR]

It prints the pass rate (the share of samples whose reward claims clean: a correct answer), the mean reward and the
scored fraction of each, and the per-prompt change, and writes `<out>/eval-compare.json` and `<out>/eval-compare.md`.
The markdown holds, for the prompts that changed most, one sample of each model: its reward, its checks' states and,
with `--tokenizer`, its final reply decoded. It refuses two runs that did not score the same prompts in the same
order, from the same base model (every base file's sha256 as the run plans hashed it, and the served or merged
weights' hash), with the same sampling settings, batch layout, reward (the profile's content, not its name, the reward
function's version and code hash, the reward settings, and each check pool's adapter and version, which names a Q&A
judge), task hashes, first prompts and seeds, reward code and plugin files, since their numbers would not be
comparable, and a run whose run plan is missing or whose committed episode records do not cover every sample of its
report, since those checks would have nothing to read. An unscored sample is never counted as passing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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


COMMITTED = re.compile(r"iter-\d+")  # a batch is committed when iterations/iter-<n>/ appears; iter-<n>.partial is not


def episodes(run_dir: Path) -> dict[str, list[dict]]:
    """The committed episode records, by prompt."""
    out: dict[str, list[dict]] = {}
    for f in sorted((Path(run_dir) / "iterations").glob("iter-*/episodes.jsonl")):
        if not COMMITTED.fullmatch(f.parent.name):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            out.setdefault(e["prompt_id"], []).append(e)
    return out


def paired_episodes(base_dir: Path, trained_dir: Path, base: dict, trained: dict) -> tuple[dict, dict]:
    """Both runs' committed episodes, refused unless each report's every sample has exactly one record (its
    iteration and k) and the two runs' records pair one to one. A record with no turns (a chain that ended before its
    first reply) is legal."""
    runs = []
    for run_dir, rep in ((base_dir, base), (trained_dir, trained)):
        eps = episodes(run_dir)
        if not eps:
            raise Refused(f"{run_dir}: no committed episode records (iterations/iter-<n>/episodes.jsonl)")
        if extra := sorted(set(eps) - set(rep["prompts"])):
            raise Refused(f"{run_dir}: episode records for prompts its report does not list: {extra[:3]}")
        keys = {}
        for pid, row in rep["prompts"].items():
            got = [(e["iteration"], e["k"]) for e in eps.get(pid, [])]
            if len(got) != len(set(got)):
                raise Refused(f"{run_dir}: {pid} has more than one episode record for a sample")
            if len(got) != row.get("samples"):
                raise Refused(f"{run_dir}: {pid} has {len(got)} committed episode records; its report counts "
                              f"{row.get('samples')} samples")
            keys[pid] = set(got)
        runs.append((eps, keys))
    (eb, kb), (et, kt) = runs
    for pid in base["prompts"]:
        if kb[pid] != kt[pid]:
            raise Refused(f"{pid}: the two runs' samples do not pair (iteration, k): "
                          f"{sorted(kb[pid] ^ kt[pid])[:3]} are in one run only")
    return eb, et


def fingerprints(base_dir: Path, trained_dir: Path, base: dict, trained: dict) -> None:
    """Refuse two runs whose names agree but whose inputs or scoring differ: the paired samples' first prompt (its
    token hash, so a changed template shows) and seed, the reward function's version and code hash on every scored
    sample, and the plugin files the run plans hashed (a task's own checks and reward live there). A run with no
    run plan, or whose episode records do not cover its report, is refused: there is nothing to check it against."""
    pb, pt = (_plugin_hashes(d) for d in (base_dir, trained_dir))
    eb, et = paired_episodes(base_dir, trained_dir, base, trained)
    for pid in sorted(eb):
        first = {(e["k"], e["iteration"]): (e["turns"][0]["prompt_ids_sha256"], e["turns"][0]["seed"])
                 for e in eb[pid] if e.get("turns")}
        for e in et[pid]:
            got = first.get((e["k"], e["iteration"]))
            if e.get("turns") and got is not None and got != (e["turns"][0]["prompt_ids_sha256"],
                                                               e["turns"][0]["seed"]):
                raise Refused(f"{pid} sample {e['k']}: the two runs' first prompts or seeds differ")
    rb, rt = ({(e.get("reward_version"), e.get("reward_code_sha256")) for eps in run.values() for e in eps
               if e.get("status") == "scored"} for run in (eb, et))
    if rb and rt and rb != rt:
        raise Refused(f"the two runs were scored by different reward code: {sorted(rb)} against {sorted(rt)}")
    if pb != pt:
        raise Refused(f"the two runs loaded different plugin files: {pb} against {pt}")


def _plugin_hashes(run_dir: Path) -> dict:
    path = Path(run_dir) / "run_plan.json"
    if not path.is_file():
        raise Refused(f"{run_dir}: no run_plan.json, so its plugin files cannot be checked")
    plan = json.loads(path.read_text(encoding="utf-8"))
    return {k: (v or {}).get("sha256") for k, v in sorted((plan.get("plugins") or {}).items())}


def reward_content(run_dir: Path, rep: dict) -> dict:
    """What scored the run, by content and not by name: the profile (its stages and legs, pools, requirements, leg
    and adapter settings, clean repeats), the reward function's name, version and code hash, the run's reward
    settings, and each check pool's adapter and version. The profile's name is left out: two names for one profile
    score alike, and one name for two profiles does not."""
    r = rep["run"].get("reward") or {}
    if not all(isinstance(r.get(k), dict) for k in ("profile_content", "reward_fn_identity", "settings", "checkers")):
        raise Refused(f"{run_dir}: its report does not record what scored it (the reward profile's content, the "
                      f"reward function's version and the checkers)")
    return {"profile": {k: v for k, v in r["profile_content"].items() if k != "name"},
            "reward_fn": r["reward_fn_identity"], "checkers": r["checkers"],
            "settings": {k: v for k, v in r["settings"].items() if k != "profile"}}


def differing(a: dict, b: dict, path: str = "") -> list[str]:
    """The dotted paths at which two JSON trees differ, as far down as both are objects."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return [path or "(all)"] if a != b else []
    out = []
    for k in sorted(set(a) | set(b)):
        out += differing(a.get(k), b.get(k), f"{path}.{k}" if path else k)
    return out


def files_digest(files: dict) -> str:
    """sha256 over the files' names and sha256s, as the score report's `files_sha256` takes it (canonical JSON)."""
    return hashlib.sha256(json.dumps(dict(sorted(files.items())), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


SNAPSHOT = re.compile(r"/snapshots/([0-9a-f]{7,64})(?:/|$)")  # a Hugging Face snapshot directory's revision


def base_model(run_dir: Path, rep: dict) -> tuple[dict, str | None]:
    """The base model a run started from: the name and sha256 of every file of `model.base_dir` (the weights, the
    config, the tokenizer), as its run plan hashed them, and the snapshot revision its path names, if any. A run that
    served the base as it is (`merged_dir`) must have served exactly those top-level files; one that merged an adapter
    was checked against them when it merged."""
    if not (Path(run_dir) / "run_plan.json").is_file():
        raise Refused(f"{run_dir}: no run_plan.json, so its base model cannot be checked")
    plan = json.loads((Path(run_dir) / "run_plan.json").read_text(encoding="utf-8"))
    files = {k.partition("/")[2]: v.get("sha256") for k, v in (plan.get("extra_files") or {}).items()
             if k.startswith("base/")}
    if not files:
        raise Refused(f"{run_dir}: its run plan hashes no base model files, so its base model cannot be checked")
    model = rep["run"].get("model") or {}
    if model.get("kind") == "merged_dir":
        top = {n: h for n, h in files.items() if "/" not in n and n != "merge-manifest.json"}
        if model.get("files_sha256") != files_digest(top):
            raise Refused(f"{run_dir}: the model it served (files sha256 {str(model.get('files_sha256'))[:16]}) is "
                          f"not the base its run plan hashed ({files_digest(top)[:16]})")
    elif model.get("kind") == "base_plus_adapter":
        served = (json.loads((Path(run_dir) / "score" / "model.json").read_text(encoding="utf-8"))
                  .get("model") or {}) if (Path(run_dir) / "score" / "model.json").is_file() else {}
        if not (model.get("merge") or {}).get("tensor_digest") or \
                files_digest(served.get("files") or {}) != model.get("files_sha256"):
            raise Refused(f"{run_dir}: its merged weights' hash is missing or does not match the files it recorded")
    else:
        raise Refused(f"{run_dir}: its report names no model kind it can be checked by ({model.get('kind')})")
    paths = [str(v.get("path", "")) for k, v in (plan.get("extra_files") or {}).items() if k.startswith("base/")]
    revisions = {m.group(1) for p in paths if (m := SNAPSHOT.search(p))}
    return files, (revisions.pop() if len(revisions) == 1 else None)


def same_base(base_dir: Path, trained_dir: Path, base: dict, trained: dict) -> None:
    """Refuse two runs that did not start from the same base model: a trained adapter is compared with the base it
    was trained on, not with another revision or another set of weights."""
    (fb, rev_b), (ft, rev_t) = base_model(base_dir, base), base_model(trained_dir, trained)
    if fb != ft:
        changed = sorted(n for n in set(fb) | set(ft) if fb.get(n) != ft.get(n))
        revs = f" (revision {rev_b} against {rev_t})" if rev_b and rev_t and rev_b != rev_t else ""
        raise Refused(f"the two runs did not start from the same base model{revs}: {len(changed)} files differ, "
                      f"{', '.join(changed[:5])}")


def headline(rep: dict) -> dict:
    t = rep["total"]
    return {"samples": t["samples"], "scored": t["scored"], "scored_fraction": t["scored_fraction"],
            "pass_rate": t["clean"] / t["samples"] if t["samples"] else 0.0, "clean": t["clean"],
            "mean_reward": (t.get("reward") or {}).get("mean"),
            "model": rep["run"].get("model", {}).get("kind")}


def compare(base_dir: Path, trained_dir: Path) -> dict:
    base, trained = read_report(base_dir), read_report(trained_dir)
    if list(base["prompts"]) != list(trained["prompts"]):
        raise Refused("the two runs did not score the same prompts in the same order")
    for key in ("sampling", "options"):  # options: P and the passes, which decide each prompt's seeds
        sb, st = base["run"].get(key), trained["run"].get(key)
        if sb != st:
            raise Refused(f"the two runs differ in {key}: {sb} against {st}")
    rb, rt = reward_content(base_dir, base), reward_content(trained_dir, trained)
    if rb != rt:
        raise Refused(f"the two runs differ in reward: {', '.join(differing(rb, rt))}")
    same_base(base_dir, trained_dir, base, trained)
    for pid, b in base["prompts"].items():
        t = trained["prompts"][pid]
        for key in ("task_sha256", "samples"):
            if b.get(key) != t.get(key):
                raise Refused(f"{pid}: the two runs differ in {key}: {b.get(key)} against {t.get(key)}")
    fingerprints(base_dir, trained_dir, base, trained)
    sb = base["run"].get("sampling")
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
    if not ep.get("turns"):  # a chain that ended before its first reply (engine_error) is a legal episode
        return f"(no reply: the chain ended {ep.get('end_reason')} before its first turn)"
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
            e = next((x for x in got if x.get("turns")), got[0])  # a sample with a reply, if there is one
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
