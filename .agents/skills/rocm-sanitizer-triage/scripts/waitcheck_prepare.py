#!/usr/bin/env python3
"""Prepare exact-entry Waitcheck recipes or aggregate their saved reports."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import waitcheck_report as reader

SCHEMA = "waitcheck-exact-entry-plan-v1"
TEMPLATE = Path(__file__).resolve().parents[1] / "assets/waitcheck-exact-entry.yaml"


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def digest(path):
    from aorta.instrumentation.rocjitsu_sanitizers.selection import sha256_code_object

    if path.is_symlink():
        raise ValueError(f"Expected a regular file, found a symlink: {path}")
    value = sha256_code_object(path)
    if value is None:
        raise ValueError(f"Expected a readable, nonempty, regular file: {path}")
    return value


def number(value, field):
    from aorta.workloads.tokenspeed.harvest_code_objects import _entry_offset

    if type(value) not in (int, str) or not re.fullmatch(r"0[xX][0-9a-fA-F]+|[0-9]+", str(value)):
        raise ValueError(f"Missing or invalid {field}: {value!r}")
    parsed = _entry_offset(value)
    if parsed is None or parsed.bit_length() > 64:
        raise ValueError(f"{field} cannot be represented as an unsigned 64-bit integer")
    return parsed


def list_objects(args):
    if args.object:
        if args.pattern or not args.kernel:
            raise ValueError("--object requires --kernel and does not take --pattern")
        if args.object.is_symlink() or not args.object.is_file():
            raise ValueError(f"Object must be a regular file: {args.object}")
        return [args.object.resolve(strict=True)]
    if args.kernel:
        raise ValueError(
            "--directory selects every listed entry for the target; use --object for one kernel"
        )
    if args.directory.is_symlink() or not args.directory.is_dir():
        raise ValueError(f"Expected an object directory: {args.directory}")
    directory = args.directory.resolve(strict=True)
    if args.output.resolve().is_relative_to(directory):
        raise ValueError("Place the fresh output directory outside the input directory")
    patterns = args.pattern or ["*.hsaco"]
    if any(not pattern or "/" in pattern or "\\" in pattern for pattern in patterns):
        raise ValueError("--pattern must be a filename pattern, e.g. '*.hsaco'")

    def fail(error):
        raise error

    objects = []
    for base, directories, files in os.walk(directory, onerror=fail):
        for name in directories:
            if (Path(base) / name).is_symlink():
                raise ValueError(
                    f"Cannot establish coverage through directory symlink: {Path(base) / name}"
                )
        for name in files:
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
                obj = Path(base) / name
                if obj.is_symlink() or not obj.is_file():
                    raise ValueError(f"Matched object is not a regular file: {obj}")
                objects.append(obj)
    if not objects:
        raise ValueError(f"No files in {directory} match {patterns}")
    return sorted(objects)


def run_listing(command, receipts, label):
    """Keep the command and both streams on disk, including failed attempts."""
    output = receipts / f"{label}.stdout"
    write_json(receipts / f"{label}.argv.json", command)
    with output.open("xb") as stdout, (receipts / f"{label}.stderr").open("xb") as stderr:
        try:
            proc = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=15
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            write_json(receipts / f"{label}.status.json", {"returncode": None, "error": str(exc)})
            raise ValueError(
                f"{label} listing failed on {command[1]}; see {receipts}: {exc}"
            ) from exc
    write_json(receipts / f"{label}.status.json", {"returncode": proc.returncode})
    if proc.returncode:
        raise ValueError(f"{label} listing exited {proc.returncode}; see {receipts}")
    return reader.bounded(output).decode("utf-8")


def inspect_object(tool, obj, target, receipts):
    counts = {}
    for line in run_listing(
        [str(tool), str(obj), "--list-code-objects"], receipts, "images"
    ).splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"(gfx[0-9a-f]+):\s*([0-9]+)", line.strip())
        if not match or match[1] in counts:
            raise ValueError(f"Unsupported or duplicate image count: {line!r}")
        counts[match[1]] = int(match[2])
    if not counts.get(target):
        raise ValueError(f"No {target} image listed for {obj}; available counts: {counts}")

    listing = run_listing(
        [str(tool), str(obj), "--target", target, "--list-kernels"], receipts, "kernels"
    )
    images, selectors, entries = {}, set(), []
    for line in listing.splitlines():
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=reader.unique_object)
        if (
            not isinstance(row, dict)
            or row.get("schema") != "rj-waitcheck-corpus-inventory-v1"
            or row.get("input") != str(obj)
            or row.get("target") != target
        ):
            raise ValueError(f"Inventory schema, input or target differs for {obj}")
        image = number(row.get("code_object_index"), "code_object_index")
        if image >= counts[target]:
            raise ValueError(f"Image {image} exceeds listed {target} image count")
        if row.get("kind") == "code-object":
            if image in images:
                raise ValueError(f"Duplicate image record {image}")
            images[image] = number(row.get("kernel_count"), "kernel_count")
        elif row.get("kind") == "kernel":
            entry = number(row.get("kernel_entry"), "kernel_entry")
            if (
                "kernel_entry_hex" in row
                and number(row["kernel_entry_hex"], "kernel_entry_hex") != entry
            ):
                raise ValueError("Decimal and hexadecimal entry offsets disagree")
            name = row.get("kernel_name")
            if not isinstance(name, str) or not name.strip() or (image, entry) in selectors:
                raise ValueError(f"Missing name or duplicate entry selector in {obj}")
            selectors.add((image, entry))
            entries.append(
                {
                    "name": name,
                    "target": target,
                    "code_object": str(obj),
                    "code_object_index": image,
                    "entry_offset": entry,
                }
            )
        else:
            raise ValueError(f"Unsupported listing record kind: {row.get('kind')!r}")
    observed = Counter(entry["code_object_index"] for entry in entries)
    if len(images) != counts[target] or any(
        observed[index] != count for index, count in images.items()
    ):
        raise ValueError(f"Image or kernel counts disagree across the listings for {obj}")
    if not entries:
        raise ValueError(f"No kernel entries listed for {obj}")
    return entries


def prepare(args):
    import yaml

    from aorta.instrumentation.rocjitsu_sanitizers.recipe import load_sanitizer_recipe

    if not re.fullmatch(r"gfx[0-9a-f]+", args.target):
        raise ValueError("Supply a listed target architecture such as gfx950")
    objects = list_objects(args)
    tool = args.waitcheck.resolve(strict=True)
    if not tool.is_file() or not os.access(tool, os.X_OK):
        raise ValueError(f"Waitcheck must be executable: {tool}")
    tool_hash = digest(tool)
    if args.waitcheck_sha256 and args.waitcheck_sha256.lower() != tool_hash:
        raise ValueError("Waitcheck SHA-256 differs from --waitcheck-sha256")
    if args.output.is_symlink():
        raise ValueError(f"Output must be a fresh directory: {args.output}")
    dest = args.output.resolve()
    dest.mkdir(mode=0o700)
    objects_record, selected = [], []
    for idx, obj in enumerate(objects, 1):
        receipts = dest / f"inventory-{idx:04d}"
        receipts.mkdir()
        before = digest(obj)
        rows = inspect_object(tool, obj, args.target, receipts)
        if digest(obj) != before or digest(tool) != tool_hash:
            raise ValueError(f"Input or Waitcheck changed during inventory: {obj}")
        if args.kernel:
            rows = [row for row in rows if row["name"] == args.kernel]
            if len(rows) != 1:
                raise ValueError(
                    f"Expected one exact match for {args.kernel!r} in {obj}; found {len(rows)}"
                )
        for row in rows:
            row["code_object_sha256"] = before
        selected.extend(rows)
        objects_record.append(
            {"path": str(obj), "sha256": before, "entries": len(rows), "inventory": receipts.name}
        )

    recipe = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    plan = recipe["sanitizer_plan"]
    plan["target"] = args.target
    entries = []
    for idx, identity in enumerate(selected, 1):
        key = f"entry-{idx:04d}"
        entry_dir = dest / key
        entry_dir.mkdir()
        plan["source"]["kernel"] = {k: v for k, v in identity.items() if k != "target"}
        path = entry_dir / "recipe.yaml"
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(recipe, stream, sort_keys=False)
        load_sanitizer_recipe(path)
        entries.append({"id": key, "identity": identity, "recipe_sha256": digest(path)})
    if list_objects(args) != objects:
        raise ValueError("The set of requested objects changed during inventory")
    if digest(tool) != tool_hash or any(
        digest(Path(row["path"])) != row["sha256"] for row in objects_record
    ):
        raise ValueError("An inventoried input changed before the plan was finalized")
    manifest = dest / "manifest.json"
    write_json(
        manifest,
        {
            "schema": SCHEMA,
            "target": args.target,
            "checker": str(tool),
            "checker_sha256": tool_hash,
            "scope": str(args.directory.resolve()) if args.directory else str(objects[0]),
            "patterns": (args.pattern or ["*.hsaco"]) if args.directory else [],
            "objects": objects_record,
            "entries": entries,
        },
    )
    print(f"Prepared {len(entries)} exact-entry recipe(s). Manifest: {manifest}")
    if len(entries) == 1:
        print(f"Recipe: {dest / entries[0]['id'] / 'recipe.yaml'}")


def result_row(plan, item):
    import yaml

    key = item["id"]
    directory = plan / key
    report = directory / "scan/sanitizer_report.json"
    log = directory / "scan/waitcheck/waitcheck-0.log"
    row = {"id": key, "expected_identity": item["identity"], "report": str(report)}
    if directory.is_symlink() or directory.resolve().parent != plan:
        raise ValueError(f"Invalid entry directory: {directory}")
    try:
        recipe = directory / "recipe.yaml"
        if digest(recipe) != item["recipe_sha256"]:
            raise ValueError("Prepared recipe fingerprint mismatch")
        # Compare literal paths so an archived Linux plan can be read on another host.
        prepared = yaml.safe_load(reader.bounded(recipe))["sanitizer_plan"]
        if dict(prepared["source"]["kernel"], target=prepared["target"]) != item["identity"]:
            raise ValueError("Manifest identity differs from its prepared recipe")
        if not report.resolve().is_relative_to(directory) or not log.resolve().is_relative_to(
            directory
        ):
            raise ValueError("Report or log path leaves prepared entry directory")
        if not report.exists() and not report.is_symlink():
            return dict(row, outcome="missing_report")
        evidence = reader.bounded(report)
        raw = json.loads(evidence.decode("utf-8-sig"), object_pairs_hook=reader.unique_object)
        # Give the official reader precisely the bytes used for identity and status below.
        with tempfile.TemporaryDirectory(prefix="waitcheck-coverage-") as tempdir:
            snapshot = Path(tempdir) / "report.json"
            snapshot.write_bytes(evidence)
            raw_log = log if log.exists() or log.is_symlink() else None
            _, reader_exit = reader.summarize(snapshot, raw_log)
        actual = raw["worklist"]["kernels"][0]["identity"]
        check = raw["checks"][0]
        entry = next(iter(check["kernel_results"]), {})
        row.update(
            reported_identity=actual,
            reported_verdict=raw["overall_verdict"],
            execution=raw["execution_status"],
            check_state=check["state"],
            check_reason=check.get("reason"),
            entry_state=entry.get("state"),
            entry_reason=entry.get("reason"),
            native_exit=entry.get("returncode"),
            check_verdict=check["verdict"],
            entry_verdict=entry.get("verdict"),
            retained_findings=len(entry.get("findings", [])) if entry else None,
            reader_exit=reader_exit,
            raw_log=str(raw_log) if raw_log else None,
        )
        if actual != item["identity"]:
            return dict(row, outcome="identity_mismatch")
        if reader_exit:
            return dict(row, outcome="evidence_conflict")
        if raw["execution_status"] == "complete" and check["state"] == entry.get("state") == "ran":
            return dict(row, outcome="reported_" + raw["overall_verdict"])
        return dict(row, outcome="incomplete")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RecursionError,
        IndexError,
        yaml.YAMLError,
    ) as exc:
        return dict(row, outcome="unusable_evidence", reason=str(exc))


def aggregate(args):
    plan = args.plan.resolve(strict=True)
    manifest = json.loads(
        reader.bounded(plan / "manifest.json"), object_pairs_hook=reader.unique_object
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or not manifest.get("entries")
    ):
        raise ValueError("Expected a completed Waitcheck preparation manifest")
    entries = manifest["entries"]
    objects = manifest["objects"]
    counts = Counter(item["identity"]["code_object"] for item in entries)
    expected = {item["path"]: item["entries"] for item in objects}
    if len(expected) != len(objects) or counts != expected:
        raise ValueError("The manifest's object and entry totals disagree")
    rows = []
    for index, item in enumerate(entries, 1):
        if item["id"] != f"entry-{index:04d}":
            raise ValueError("Duplicate, missing or invalid entry ID in manifest")
        rows.append(result_row(plan, item))
    counts = Counter(row["outcome"] for row in rows)
    write_json(
        args.output,
        {
            "requested_entries": len(entries),
            "outcome_counts": dict(counts),
            "entries": rows,
            "scope": "Saved static reports for the selected compiled entries. "
            "Application execution and numerical correctness are outside this evidence.",
        },
    )
    print(f"Coverage written for {len(entries)} entries: {args.output}")


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    build = commands.add_parser(
        "prepare", help="Inventory compiled objects and write exact-entry recipes"
    )
    inputs = build.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--object", type=Path)
    inputs.add_argument("--directory", type=Path)
    build.add_argument("--kernel", help="Exact entry name for single-object selection")
    build.add_argument(
        "--pattern", action="append", help="Recursive directory filename match; default *.hsaco"
    )
    build.add_argument("--target", required=True)
    build.add_argument("--waitcheck", required=True, type=Path)
    build.add_argument(
        "--waitcheck-sha256", help="Expected checker fingerprint from a trusted receipt"
    )
    build.add_argument("--output", type=Path, required=True, help="Fresh plan directory")
    coverage = commands.add_parser(
        "aggregate", help="Summarize saved reports for the prepared entries"
    )
    coverage.add_argument("--plan", type=Path, required=True)
    coverage.add_argument("--output", type=Path, required=True, help="Fresh coverage JSON file")
    args = cli.parse_args(argv)
    try:
        if args.action == "prepare":
            prepare(args)
        else:
            aggregate(args)
    except (OSError, ValueError, KeyError, TypeError, RecursionError, ImportError) as exc:
        print(f"Cannot {args.action} Waitcheck evidence: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
