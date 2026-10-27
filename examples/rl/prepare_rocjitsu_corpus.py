#!/usr/bin/env python3
"""Build a rocjitsu corpus directory: the archive files plus a ``split.json``.

  prepare_rocjitsu_corpus.py --source <campaign-dir> --out <corpus-dir> \\
      [--ops waitcnt_weaken,waitcnt_late,barrier_late] \\
      [--test-kernels k_scan,k_stencil,k_wavesum] [--clean-frac 0.1]

The archive files are symlinked, not copied, and ``split.json`` records each
source file's SHA-256 so a later reader can tell the archive changed.

The split is **by kernel**: every faulty scenario whose fault is in a
``--test-kernels`` kernel is held out, so the test set asks for locations in
kernels the policy was never rewarded for. Clean controls carry no kernel; each
split gets its own at ``--clean-frac`` of that split, as the prototype's corpus
did (10%).

A source may also carry its own split (``--split-file``): a JSON object with
``train`` and ``test`` lists of scenario ids, a JSON list of
``{"scenario_id"|"site_id", "split"}`` rows, or a JSONL of such rows. The
scenarios it names are still filtered to usable classes and to ``--ops``, and
every refusal is printed with its count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

#: Faults with no trace in the listing: a drop leaves the kernel's only
#: ``s_nop``, which a disassembly-only shortcut finds 89% of the time.
TRACE_FREE_OPS = ("waitcnt_weaken", "waitcnt_late", "barrier_late")
USABLE_CLASSES = ("manifest", "latent_detected")
DEFAULT_TEST_KERNELS = ("k_scan", "k_stencil", "k_wavesum")

#: Where each archive file is looked for, relative to the source, in order.
CANDIDATES = {
    "sites.json": ("sites.json", "suite/sites/sites.json", "sites/sites.json"),
    "results.jsonl": ("results.jsonl", "results_clean.jsonl", "analysis/results_clean.jsonl",
                      "suite/full/results.jsonl"),
    "suite.dis": ("suite.dis", "disasm.txt", "analysis/suite.dis", "suite/sites/suite.dis"),
    "analysis.json": ("analysis.json", "analysis/analysis.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_files(source: Path) -> dict[str, Path]:
    found, missing = {}, []
    for name, rels in CANDIDATES.items():
        hit = next((source / r for r in rels if (source / r).is_file()), None)
        if hit is None:
            missing.append(f"{name} (looked for {list(rels)})")
        else:
            found[name] = hit
    if missing:
        raise SystemExit(f"{source}: missing archive file(s): {missing}")
    return found


def check_destinations(out: Path, files: dict[str, Path], args: argparse.Namespace) -> dict[str, Path]:
    """Each archive file's resolved target, once no output entry can clobber an input.

    Runs before anything under ``out`` is removed or written. An archive link
    is replaced in place, so its entry must not be a source's own entry or
    real file (``--out`` equal to, or holding, the source's files); an
    existing link to a source is fine and is what a rerun replaces.
    ``split.json`` replaces its entry too (``write_split``), and is refused
    while its real path is an input. The check alone would not be enough:
    the archive links are replaced after it runs, so a ``split.json`` link
    to an archive entry would point at a source by the time it is written.
    """
    targets = {name: path.resolve(strict=True) for name, path in files.items()}
    entry = lambda p: p.parent.resolve() / p.name  # noqa: E731 -- the directory entry, final link not followed
    inputs = {entry(p) for p in files.values()} | set(targets.values())
    inputs |= {Path(p).resolve() for p in (args.split_file, args.train_ids, args.test_ids)
               if p and Path(p).exists()}
    clashes = [str(out / name) for name in files if entry(out / name) in inputs]
    if (out / "split.json").resolve() in inputs:
        clashes.append(str(out / "split.json"))
    if clashes:
        raise SystemExit(f"--out {out} would overwrite input file(s) {clashes}; choose an --out "
                         f"outside --source {args.source}")
    return targets


def write_split(out: Path, text: str) -> None:
    """Write ``out/split.json`` by renaming a new file over its entry, so a link there is never followed.

    The file gets the mode a plain write would (``0o666`` less the umask), not
    ``mkstemp``'s ``0o600``, so a corpus others read stays readable.
    """
    fd, temporary = tempfile.mkstemp(dir=out, prefix=".split.json.", suffix=".partial")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            umask = os.umask(0)
            os.umask(umask)
            os.fchmod(handle.fileno(), 0o666 & ~umask)
            handle.write(text)
        os.replace(temporary, out / "split.json")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_split_file(path: Path) -> dict[str, str]:
    """``{scenario_id: "train"|"test"}`` from any of the accepted shapes."""
    text = path.read_text()
    rows: list[Any]
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = None
    if isinstance(doc, dict) and "train" in doc and "test" in doc:
        return {**{s: "train" for s in doc["train"]}, **{s: "test" for s in doc["test"]}}
    rows = doc if isinstance(doc, list) else [json.loads(x) for x in text.splitlines() if x.strip()]
    out = {}
    for row in rows:
        sid = row.get("scenario_id") or row.get("site_id")
        split = row.get("split")
        if not sid or split not in ("train", "test"):
            raise SystemExit(f"{path}: a row has no scenario id or no train/test split: {row}")
        out[sid] = split
    return out


def read_ids(path: Path) -> list[str]:
    """Scenario ids from a JSON list (of ids or rows), a JSONL of rows, or one id per line."""
    text = path.read_text()
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = None
    if isinstance(doc, list):
        items = doc
    elif text.lstrip().startswith("{"):
        items = [json.loads(x) for x in text.splitlines() if x.strip()]
    else:
        items = [x.strip() for x in text.splitlines() if x.strip() and not x.startswith("#")]
    out = []
    for item in items:
        sid = item if isinstance(item, str) else (item.get("scenario_id") or item.get("site_id"))
        if not sid:
            raise SystemExit(f"{path}: an entry has no scenario id: {item}")
        out.append(sid)
    return out


def select_ops(spec: str, rows: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    """``--ops``: a comma list, or ``trace-free`` -- every op none of whose rows has a trace.

    Every op must be a barrier or a wait operator, because the answer classes
    are ``barrier|waitcnt|none``: an op of another kind would be scored with a
    class it does not have.
    """
    if spec == "trace-free":
        by_op: dict[str, set[bool]] = {}
        for r in rows.values():
            if r.get("op") and r.get("op") != "none":
                by_op.setdefault(r["op"], set()).add(bool(r.get("trace", r["op"].endswith("_drop"))))
        ops = tuple(sorted(op for op, traces in by_op.items() if traces == {False}))
    else:
        ops = tuple(o for o in spec.split(",") if o)
    odd = [o for o in ops if not o.startswith(("barrier_", "waitcnt_"))]
    if odd:
        raise SystemExit(f"op(s) {odd} are neither barrier_* nor waitcnt_*; the answer classes cannot score them")
    if not ops:
        raise SystemExit(f"--ops {spec!r} selects no operator")
    return ops


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source)
    files = find_files(source)
    analysis = json.loads(files["analysis.json"].read_text())
    rows = {r["site_id"]: r for r in analysis["rows"]}
    archived = set()
    with open(files["results.jsonl"], encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rid = json.loads(line)["site_id"]
                if not rid.startswith("BASELINE:"):
                    archived.add(rid)
    ops = select_ops(args.ops, rows)
    refused: Counter[str] = Counter()
    usable = []
    for sid, row in rows.items():
        if row.get("class") not in USABLE_CLASSES:
            refused[f"class {row.get('class')}"] += 1
        elif row.get("op") not in ops:
            refused[f"op {row.get('op')} not in --ops"] += 1
        elif sid not in archived:
            refused["no archived record"] += 1
        else:
            usable.append(sid)
    if args.train_ids or args.test_ids:
        if not (args.train_ids and args.test_ids) or args.split_file:
            raise SystemExit("--train-ids and --test-ids go together, and not with --split-file")
        given = {**{s: "train" for s in read_ids(Path(args.train_ids))},
                 **{s: "test" for s in read_ids(Path(args.test_ids))}}
        args.split_file = f"{args.train_ids} + {args.test_ids}"
    elif args.split_file:
        given = read_split_file(Path(args.split_file))
    if args.split_file:
        unknown = sorted(set(given) - set(rows) - {s for s in given if s.startswith("CLEAN:")})
        if unknown:
            print(f"[split] {len(unknown)} id(s) in --split-file are not in analysis.json: {unknown[:5]}",
                  file=sys.stderr)
        train = [s for s in usable if given.get(s) == "train"]
        test = [s for s in usable if given.get(s) == "test"]
        unsplit = [s for s in usable if s not in given]
        if unsplit:
            refused["usable but not named by --split-file"] += len(unsplit)
        rule = f"from {args.split_file}"
        test_kernels = sorted({rows[s]["kernel"] for s in test})
    else:
        test_kernels = [k for k in args.test_kernels.split(",") if k]
        train = [s for s in usable if rows[s]["kernel"] not in test_kernels]
        test = [s for s in usable if rows[s]["kernel"] in test_kernels]
        rule = f"by kernel: test = faults in {test_kernels}"
    if not train or (not test and not args.allow_empty_test):
        raise SystemExit(f"the split leaves {len(train)} train and {len(test)} test fault(s); both must be "
                         f"non-empty (rule: {rule}; pass --allow-empty-test for a train-only corpus)")
    overlap = {rows[s]["kernel"] for s in train} & {rows[s]["kernel"] for s in test}
    if overlap and not args.allow_kernel_overlap:
        raise SystemExit(f"train and test share fault kernel(s) {sorted(overlap)}; a split by kernel "
                         f"must not (pass --allow-kernel-overlap to accept it)")
    order = lambda ids: sorted(ids, key=lambda s: (rows[s]["kernel"], rows[s]["op"], s))  # noqa: E731

    def clean(n_faulty: int, name: str) -> list[str]:
        n = max(1, round(args.clean_frac * n_faulty / (1 - args.clean_frac))) if args.clean_frac else 0
        return [f"CLEAN:{name}:{i}" for i in range(n)]

    train, test = order(train), order(test)
    train_ids, test_ids = train + clean(len(train), "train"), test + clean(len(test), "test")
    out = Path(args.out)
    targets = check_destinations(out, files, args)
    out.mkdir(parents=True, exist_ok=True)
    for name, target in targets.items():
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)

    def counts(ids: list[str]) -> dict[str, Any]:
        faulty = [s for s in ids if not s.startswith("CLEAN:")]
        return {"faulty": len(faulty), "clean": len(ids) - len(faulty),
                "by_kernel": dict(sorted(Counter(rows[s]["kernel"] for s in faulty).items())),
                "by_op": dict(sorted(Counter(rows[s]["op"] for s in faulty).items())),
                "distinct_answers": len({(rows[s]["kernel"], rows[s]["kernel_offset"]) for s in faulty})}

    spec = {
        "rule": rule,
        "ops": list(ops),
        "usable_classes": list(USABLE_CLASSES),
        "test_kernels": test_kernels,
        "clean_frac": args.clean_frac,
        "source": str(source.resolve()),
        "files": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in files.items()},
        "refused": dict(refused),
        "counts": {"train": counts(train_ids), "test": counts(test_ids)},
        "train": train_ids,
        "test": test_ids,
    }
    write_split(out, json.dumps(spec, indent=1) + "\n")
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="a campaign directory (v1 layout or flat)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ops", default=",".join(TRACE_FREE_OPS),
                        help="comma list, or 'trace-free' for every op with no trace in analysis.json")
    parser.add_argument("--test-kernels", default=",".join(DEFAULT_TEST_KERNELS))
    parser.add_argument("--split-file", default=None, help="use the source's own train/test split")
    parser.add_argument("--train-ids", default=None, help="the source's train ids (with --test-ids)")
    parser.add_argument("--test-ids", default=None, help="the source's test ids (with --train-ids)")
    parser.add_argument("--clean-frac", type=float, default=0.1)
    parser.add_argument("--allow-kernel-overlap", action="store_true")
    parser.add_argument("--allow-empty-test", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 <= args.clean_frac < 1.0:
        parser.error("--clean-frac must be in [0, 1)")
    spec = build(args)
    print(json.dumps({k: spec[k] for k in ("rule", "ops", "test_kernels", "refused", "counts")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
