#!/usr/bin/env python3
"""Build the labelled RL corpus from a tree of real sanitizer runs.

Turns `sanitizer_report.json` artifacts into JSONL that `triage_reward.py
--corpus` and `proposal_reward.py --corpus` read directly, so a generated
corpus needs no conversion pass before it can be scored.

Two things this file is deliberate about.

**One example is one scenario, not one finding.** The two-wave LDS race
reproducer emits 64 findings, and they are 64 lanes of the same race: same
code, same severity, same instruction pair, differing only in lane mask and LDS
byte range. Counting those as 64 examples would inflate the corpus 64-fold and
train a model on one race repeated. `distinct_evidence` therefore dedupes on
the tuple that identifies a race *site*, and the per-scenario counts are
reported separately from the raw finding count so the inflation stays visible.

**Ground truth comes from the committed baselines, not from the run.** A
scenario named in `fixtures/expected/verdict_baselines.json` carries the
expected verdict alongside the observed one, plus an explicit agreement flag.
Agreement is decided by the gate's own `_compare_case`, imported from
`scripts/sanitizers/compare_verdict_baselines.py` rather than restated here, so
it covers the whole contract -- `overall_verdict`, `execution_status`, the
per-sanitizer verdicts and the `finding_shape` substrings. Comparing only the
top-level verdict would let a report keep its verdict while losing the evidence
it is supposed to cite, and still be recorded as agreeing.
When the two disagree the example is still emitted -- it is evidence of a tool
defect, and dropping it would hide exactly the thing worth reporting -- but it
is flagged `ground_truth.agrees = false`, and `triage_reward.load_corpus` skips
those rows unless asked for them (`--include-disagreements`). The flag on its
own was not enough: nothing read it, so the advertised no-conversion path scored
against the observed verdict on precisely the scenarios where the observed
verdict is known to be wrong, and a policy that reproduced the defect was
rewarded for it. Emit-and-flag is the right build-time decision; acting on the
flag is the consumer's job and now happens by default.

The verdict label itself is produced by `triage_reward.label_sanitizer_report`,
which routes it through `SanitizerReport.from_dict`. That is the seam that
rejects a report whose stored verdict contradicts its own checks, so corpus rot
fails loudly here rather than silently later.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The gate's own comparison, imported rather than restated. Agreement here has
# to mean what it means to `scripts/sanitizers/compare_verdict_baselines.py`,
# or a report can keep its top-level verdict while its execution status, a
# per-sanitizer verdict or a cited finding shape regresses, be recorded as
# agreeing, and then hand that regressed evidence to a reward as ground truth.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "sanitizers"))

from aorta.instrumentation.rocjitsu_sanitizers.models import SanitizerReport  # noqa: E402
from compare_verdict_baselines import _compare_case  # noqa: E402
from triage_reward import label_sanitizer_report  # noqa: E402

CORPUS_SCHEMA = "aorta.rl_corpus/0.1"

# Which workload family a case directory belongs to. Recorded on every example
# even though there are only three families today: the detector and finding
# vocabulary is workload-independent, but which failures co-occur is not, so a
# corpus that cannot be split by family cannot be checked for balance. Adding
# the field now is free; backfilling it onto archived artifacts is not.
WORKLOAD_FAMILIES: dict[str, str] = {
    "consan-clean": "synthetic_hip_lds",
    "consan-racy": "synthetic_hip_lds",
    "consan-lds-dispatch": "synthetic_hip_lds",
    "waitcheck-lds-dispatch": "synthetic_hip_lds",
    "consan-tiny": "synthetic_hip_vecadd",
    "waitcheck-tiny": "synthetic_hip_vecadd",
    "consan-gemm": "tensile_gemm_object",
    "waitcheck-gemm": "tensile_gemm_object",
    "waitcheck": "tensile_gemm_object",
    # The committed survey reports use their own directory names, and they are
    # the same three families, so one map covers both trees.
    "gemm_f32_consan": "tensile_gemm_object",
    "gemm_f32_waitcheck": "tensile_gemm_object",
    "lds_reduce_consan": "synthetic_hip_lds",
    "lds_reduce_waitcheck": "synthetic_hip_lds",
    "tiny_vecadd_consan": "synthetic_hip_vecadd",
    "tiny_vecadd_waitcheck": "synthetic_hip_vecadd",
}

# Case directory -> the key it is gated against in verdict_baselines.json.
# Mirrors scripts/sanitizers/compare_verdict_baselines.py rather than
# re-deriving it, so the corpus and the nightly gate agree on what ground truth
# means for these three cases.
BASELINE_KEYS: dict[str, str] = {
    "waitcheck": "waitcheck_gemm",
    "consan-clean": "consan_clean",
    "consan-racy": "consan_racy",
}


class DuplicateScenario(Exception):
    """Two reports would publish the same corpus id."""


@dataclass
class Scenario:
    """One sanitizer run: one report, one verdict, one or more findings."""

    # Two names, and the split is load-bearing.
    #
    # `case` is the leaf directory, and it is a *lookup key*: `WORKLOAD_FAMILIES`
    # and `BASELINE_KEYS` are keyed on it, as is the gate's `_compare_case`. It
    # is not unique across a tree, and must not be made unique, or those lookups
    # stop resolving.
    #
    # `scenario_id` is the identifier the corpus publishes, and it has to be
    # unique: `run_e2e` keys both its label map and its GRPO groups on it, so
    # two scenarios sharing an id means one report scored against the other's
    # label -- mislabelled training data that nothing downstream can detect,
    # because both halves are individually well-formed.
    case: str
    scenario_id: str
    report_path: Path
    doc: dict[str, Any]
    workload_family: str
    baseline_key: str | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)


def _findings(check: dict[str, Any]) -> list[dict[str, Any]]:
    """Every finding on a check, whether attached directly or per kernel.

    ConSan writes the same finding objects to both places, so a naive
    concatenation doubles the count -- the racy reproducer reads as 128
    findings when it produced 64. Identical records are therefore collapsed
    here, before the site-level dedup, so the raw count means "findings the
    tool emitted" rather than "times a finding was written down".
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates = list(check.get("findings") or [])
    for kernel_result in check.get("kernel_results") or []:
        candidates.extend(kernel_result.get("findings") or [])
    for finding in candidates:
        key = json.dumps(finding, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(finding)
    return out


def _race_site(finding: dict[str, Any]) -> tuple:
    """The tuple that identifies a distinct finding *site*.

    For a ConSan race the instruction pair is what makes two findings the same
    defect; lane mask and LDS byte range make them different lanes of it.

    Waitcheck locates a hazard differently, and reading it through the ConSan
    keys collapsed the whole check to a single site. Its findings carry
    ``entry_offset: null`` and a constant ``code`` of ``wait_hazard``, and put
    the producer and consumer offsets in ``metadata.context_1`` and
    ``context_2`` -- none of which the ConSan keys touch, so every finding
    produced an identical tuple. Measured against the committed survey report,
    ``gemm_f32_waitcheck``'s 32 distinct hazards became 1 site and 31 evidence
    locations were dropped before anything downstream could see them.

    The two families' keys are disjoint -- Waitcheck has no ``first_inst``,
    ConSan no ``context_1`` -- so the added fields are ``None`` on a ConSan
    finding and its sites partition exactly as before.
    """
    meta = finding.get("metadata") or {}
    return (
        finding.get("sanitizer"),
        finding.get("code"),
        finding.get("severity"),
        finding.get("code_object"),
        finding.get("entry_offset"),
        meta.get("first_inst"),
        meta.get("second_inst"),
        meta.get("kind"),
        meta.get("context_1"),
        meta.get("context_2"),
    )


def _kernel_names(check: dict[str, Any]) -> list[str]:
    """Kernel attribution for a check, from wherever it is actually present.

    `Finding.kernel_name` is null in practice, so attribution has to be read
    from the kernel result identity and the backend's selected kernel. Both are
    recorded so a consumer can see which level the name came from.
    """
    names: list[str] = []
    for kernel_result in check.get("kernel_results") or []:
        name = (kernel_result.get("identity") or {}).get("name")
        if name and name not in names:
            names.append(name)
    backend_kernel = (check.get("backend") or {}).get("selected_kernel")
    if backend_kernel and backend_kernel not in names:
        names.append(backend_kernel)
    return names


def _field_population(scenario: Scenario) -> dict[str, Any]:
    """How many findings actually populate each attribution field.

    Reported because the archived corpus was limited by exactly this: a finding
    that names no kernel and no entry offset supports a verdict but not an
    attribution, and a corpus cannot be assessed without knowing which.
    """
    tracked = ("kernel_name", "entry_offset", "code_object", "severity", "code")
    counts = dict.fromkeys(tracked, 0)
    total = 0
    for check in scenario.checks:
        for finding in _findings(check):
            total += 1
            for name in tracked:
                if finding.get(name) not in (None, ""):
                    counts[name] += 1
    return {"findings": total, "populated": counts}


def collect(root: Path) -> list[Scenario]:
    """Every sanitizer report under a results tree, as a scenario.

    Refuses a tree in which two reports share a leaf directory name, rather
    than emitting both. `rglob` accepts an arbitrary tree, so a results root
    holding two runs -- `<run>/<case>/sanitizer_report.json`, the ordinary shape
    for an archived sweep -- produced two scenarios with the same id. `run_e2e`
    keys its label map and its GRPO groups on that id, so one report was scored
    against the other report's label: mislabelled training data, and
    undetectable downstream because each half is well-formed on its own.

    Refused rather than disambiguated, and the reason is id *stability*. A
    path-derived id would vary with where `--results` points -- the same report
    under `survey/` and under `survey/reports/` would get different ids -- and
    these ids key every recorded measurement. Keeping the leaf name means every
    existing id stays byte-identical no matter which root a rebuild uses, and
    the collision becomes a loud refusal naming both paths. Two runs of one case
    are also two scenarios the family and baseline lookups cannot tell apart,
    so which of them is which is the operator's call, not this function's.
    """
    scenarios: list[Scenario] = []
    seen: dict[str, Path] = {}
    for path in sorted(root.rglob("sanitizer_report.json")):
        case = path.parent.name
        scenario_id = case
        if scenario_id in seen:
            raise DuplicateScenario(
                f"two reports share the scenario id {scenario_id!r}: "
                f"{seen[scenario_id]} and {path}. The corpus id keys the label "
                "map and the GRPO groups in run_e2e, so emitting both would "
                "score one report against the other's label. Build them into "
                "separate corpora, or rename one case directory."
            )
        seen[scenario_id] = path
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipped {path}: unreadable ({exc})", file=sys.stderr)
            continue
        if not isinstance(doc, dict):
            # `[]`, `"partial"`, `null` and `0` are all valid JSON and none of
            # them is a report. Without this the document went into `Scenario`
            # unexamined and `doc.get("checks")` raised `AttributeError` three
            # lines down -- which no caller catches, so ONE truncated artifact
            # anywhere under `--results` aborted the entire build with a
            # traceback rather than costing the corpus one scenario.
            #
            # Skipped on the same terms as an unparseable one, because it is
            # the same fault: a file that is not a report. `json.JSONDecodeError`
            # only means the bytes were not JSON, and the bytes being JSON was
            # never the question.
            print(
                f"  skipped {path}: not a report ({type(doc).__name__} at the "
                "top level, expected an object)",
                file=sys.stderr,
            )
            continue
        scenarios.append(
            Scenario(
                case=case,
                scenario_id=scenario_id,
                report_path=path,
                doc=doc,
                workload_family=WORKLOAD_FAMILIES.get(case, "unknown"),
                baseline_key=BASELINE_KEYS.get(case),
                checks=list(doc.get("checks") or []),
            )
        )
    return scenarios


def _baseline_problems(scenario: Scenario, expected: dict[str, Any]) -> list[str]:
    """Every way this report departs from its committed baseline, or ``[]``.

    Delegates to the gate's ``_compare_case``, so the four things the baseline
    contract actually covers -- ``overall_verdict``, ``execution_status``, the
    per-sanitizer verdicts and the ``finding_shape`` substrings -- are all
    checked. Comparing only the top-level verdict, which is what this did
    first, calls a report that lost its cited evidence "agreeing".

    An ungated scenario returns ``[]``, and the caller turns that into
    ``agrees: None`` rather than ``True``: uncovered is not the same as
    checked-and-matching, and most of the corpus is uncovered.
    """
    if not expected:
        return []
    try:
        report = SanitizerReport.from_dict(scenario.doc)
    except (ValueError, KeyError, TypeError) as exc:
        # `label_sanitizer_report` has already parsed this document by the time
        # we are called, so reaching here means the two disagree about it --
        # which is itself a disagreement worth recording rather than crashing on.
        return [f"report did not load for comparison: {type(exc).__name__}: {exc}"]
    return _compare_case(scenario.baseline_key or scenario.case, report, expected)


def triage_example(
    scenario: Scenario, baselines: dict[str, Any], run_meta: dict[str, Any]
) -> dict[str, Any] | None:
    """One triage example, labelled through aorta's own report model."""
    try:
        label = label_sanitizer_report(scenario.doc, source=str(scenario.report_path))
    except (ValueError, KeyError, TypeError) as exc:
        print(f"  rejected {scenario.report_path}: {exc}", file=sys.stderr)
        return None

    sites: list[tuple] = []
    evidence: list[dict[str, Any]] = []
    for check in scenario.checks:
        for finding in _findings(check):
            site = _race_site(finding)
            if site in sites:
                continue
            sites.append(site)
            evidence.append(
                {
                    "sanitizer": finding.get("sanitizer") or check.get("sanitizer"),
                    "code": finding.get("code"),
                    "severity": finding.get("severity"),
                    "code_object": finding.get("code_object"),
                    "entry_offset": finding.get("entry_offset"),
                    "kernel_name": finding.get("kernel_name"),
                    "kernel_names_from_identity": _kernel_names(check),
                    "metadata": finding.get("metadata") or {},
                }
            )

    expected = baselines.get(scenario.baseline_key or "", {})
    problems = _baseline_problems(scenario, expected)
    ground_truth = {
        "source": "verdict_baselines.json" if expected else "none",
        "baseline_key": scenario.baseline_key,
        "expected_verdict": expected.get("overall_verdict"),
        "expected_execution_status": expected.get("execution_status"),
        "observed_verdict": label.verdict,
        "observed_execution_status": scenario.doc.get("execution_status"),
        "agrees": (problems == []) if expected else None,
        # Named, not just counted. "This report disagrees with its baseline" is
        # not actionable; "check 'consan' verdict='pass', expected 'fail'" is,
        # and it is the difference between a row a reader can triage and one
        # they have to re-derive.
        "disagreements": problems,
        "by_construction": scenario.case in {"consan-racy", "consan-clean"},
    }

    return {
        "schema": CORPUS_SCHEMA,
        "kind": "triage",
        "example_id": f"triage:{scenario.scenario_id}",
        "scenario_id": scenario.scenario_id,
        "workload_family": scenario.workload_family,
        "label": label.as_dict(),
        "checks": [
            {
                "sanitizer": check.get("sanitizer"),
                "verdict": check.get("verdict"),
                "findings": len(_findings(check)),
                "kernel_names": _kernel_names(check),
            }
            for check in scenario.checks
        ],
        "distinct_evidence": evidence,
        "finding_counts": {
            "raw": sum(len(_findings(c)) for c in scenario.checks),
            "distinct_sites": len(sites),
        },
        "field_population": _field_population(scenario),
        "ground_truth": ground_truth,
        "provenance": {"report": str(scenario.report_path), **run_meta},
    }


# The proposal ladder grades contract validity -- parseable, right schema, valid
# category, registered name, still-available name -- which is a property of the
# model's output and not of the run. So these proposals are synthesised, but
# each is synthesised *against a real scenario*: the candidate set and the
# evidence are the ones that scenario would really have put in front of the
# model. That keeps the tier-4 and tier-5 cases honest, because "unregistered"
# and "already tried" are judged against the real registry and the real state.
PROPOSAL_VARIANTS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("valid", "a registered, still-available name", {}),
    ("hallucinated_name", "an unregistered name that reads like a real one",
     {"next_mitigations": ["rccl_p2p_disable"]}),
    ("invalid_category", "a category outside AUTOPSY_CATEGORIES",
     {"category": "lds_race"}),
    ("already_tried", "a registered name that is no longer available",
     {"next_mitigations": ["__TRIED__"]}),
    ("mistyped_stop", "stop as a string, which from_dict coerces",
     {"stop": "yes"}),
)


def proposal_examples(
    scenario: Scenario, label_verdict: str, run_meta: dict[str, Any]
) -> list[dict[str, Any]]:
    """The proposal-contract examples for one scenario."""
    candidates = ["hsa_no_sdma", "hip_launch_blocking", "amd_log_level_4", "none"]
    tried = ["hsa_no_sdma"]
    available = [c for c in candidates if c not in tried and c != "none"]

    out: list[dict[str, Any]] = []
    for variant, detail, override in PROPOSAL_VARIANTS:
        body: dict[str, Any] = {
            "category": "illegal_mem",
            "hypothesis": (
                f"{scenario.case}: sanitizer verdict {label_verdict}; "
                f"suspect kernel-level memory ordering"
            ),
            "next_mitigations": [available[0]],
            "confidence": 0.6,
            "stop": False,
        }
        body.update(override)
        if body.get("next_mitigations") == ["__TRIED__"]:
            body["next_mitigations"] = [tried[0]]
        out.append(
            {
                "schema": CORPUS_SCHEMA,
                "kind": "proposal",
                "example_id": f"proposal:{scenario.scenario_id}:{variant}",
                "scenario_id": scenario.scenario_id,
                "workload_family": scenario.workload_family,
                "variant": variant,
                "detail": detail,
                "proposal": {
                    "name": f"{scenario.scenario_id}:{variant}",
                    "raw": json.dumps(body),
                    "candidates": candidates,
                    "tried": tried,
                },
                "provenance": {"report": str(scenario.report_path), **run_meta},
            }
        )
    return out


#: The files a corpus is made of. `manifest.json` is written last on the
#: success path and is what :func:`_holds_a_corpus` keys on.
CORPUS_FILES = ("triage.jsonl", "proposal.jsonl", "manifest.json")


def _holds_a_corpus(out: Path) -> bool:
    """Whether ``out`` already holds a corpus this script produced.

    The manifest is the key rather than the directory existing, so `--out`
    pointed at a directory that is not a corpus is never mistaken for one --
    and so nothing this script did not write is ever removed by
    :func:`discard_corpus`.
    """
    return (out / "manifest.json").is_file()


def discard_corpus(out: Path) -> bool:
    """Remove the corpus at ``out``, if there is one. Returns whether there was.

    The fail-closed guarantee this script claims -- a build that refuses to
    publish leaves nothing for a later step to consume -- held only for a
    first build. On a rebuild the refusal came *before* `mkdir`, which was the
    whole argument, and left the previous `triage.jsonl`, `proposal.jsonl` and
    manifest exactly where a trainer looks for them. The command exited 1 and
    the corpus on disk was still readable, still well-formed, and now stale:
    the worst of the shapes this file keeps arguing against, because it is
    indistinguishable from a good corpus at the point of use.

    Deleting is the destructive option and it is the consistent one. A
    successful build already replaces all three files unconditionally, so this
    directory's previous contents were forfeit the moment the command was run;
    the failure path was the only one pretending otherwise. Scoped to the
    files above for the same reason :func:`_holds_a_corpus` exists -- `--out`
    given someone's home directory removes three names that are not there.
    """
    if not _holds_a_corpus(out):
        return False
    for name in CORPUS_FILES:
        (out / name).unlink(missing_ok=True)
    return True


def publish(out: Path, payload: dict[str, str]) -> None:
    """Write the corpus beside ``out``, then swap it in.

    Writing in place published a torn corpus on any failure after the first
    `open`: `triage.jsonl` from this build beside a `proposal.jsonl` and a
    manifest from the last one, with the exit code saying the build failed and
    the directory saying otherwise. The scenario ids line up well enough for
    `run_e2e` to key its label map on them, so the mismatch is not detectable
    downstream -- it is just wrong.

    Two renames on one filesystem: the previous corpus moves aside and the
    staged one takes its place. The window where `out` does not exist is
    between them, which no amount of care removes without a real transaction;
    what it does remove is the window where `out` exists and is half of two
    builds. A failed second rename puts the previous corpus back.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(f".{out.name}.staging-{os.getpid()}")
    previous = out.with_name(f".{out.name}.previous-{os.getpid()}")
    for scratch in (staging, previous):
        shutil.rmtree(scratch, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        for name, text in payload.items():
            (staging / name).write_text(text, encoding="utf-8")
        if out.exists():
            out.rename(previous)
        try:
            staging.rename(out)
        except OSError:
            if previous.exists():
                previous.rename(out)
            raise
    finally:
        for scratch in (staging, previous):
            shutil.rmtree(scratch, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="tree of sanitizer run outputs")
    parser.add_argument("--baselines", type=Path, required=True,
                        help="fixtures/expected/verdict_baselines.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="corpus output directory")
    parser.add_argument("--run-meta", type=Path, default=None,
                        help="JSON of provenance shared by every example")
    args = parser.parse_args(argv)

    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))
    run_meta = (
        json.loads(args.run_meta.read_text(encoding="utf-8"))
        if args.run_meta and args.run_meta.exists()
        else {}
    )

    scenarios = collect(args.results)
    if not scenarios:
        print(f"no sanitizer reports under {args.results}", file=sys.stderr)
        if discard_corpus(args.out):
            print(f"  removed the previous corpus at {args.out}", file=sys.stderr)
        return 1

    triage: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    rejected = 0
    for scenario in scenarios:
        example = triage_example(scenario, baselines, run_meta)
        if example is None:
            rejected += 1
            continue
        triage.append(example)
        proposals.extend(
            proposal_examples(scenario, example["label"]["verdict"], run_meta)
        )

    # The empty-corpus case, which the "no reports" check above does not cover.
    # `collect` finding nothing is a wrong `--results` path and already fails;
    # finding reports that every one of which `triage_example` rejects is a
    # different fault -- a baselines file that does not match this run, a
    # schema change, a results tree from another workload -- and it used to
    # write two empty `.jsonl` files, a manifest reading `"scenarios": 0`, and
    # exit 0.
    #
    # That is the shape this repo keeps finding: nothing was learned and the
    # output says so in a field nobody reads, while the exit code says the
    # build succeeded. Downstream it is worse than a crash, because an empty
    # corpus is a *valid* corpus -- training on it is a no-op run that looks
    # like a run, and `recipe_reward`'s novelty gate refuses an empty corpus
    # root for exactly this reason one layer over.
    #
    # Refused before anything is written, so a failed build leaves no corpus
    # for a later step to find and mistake for a good one -- and `discard_corpus`
    # is what extends that from "no directory is created" to "no corpus is
    # left", which are the same sentence only on a machine that has never run
    # this command before.
    if not triage:
        print(
            f"all {len(scenarios)} discovered report(s) were rejected, so there "
            f"is nothing to publish; refusing to write an empty corpus to "
            f"{args.out}. Check --baselines matches this results tree.",
            file=sys.stderr,
        )
        if discard_corpus(args.out):
            print(
                f"  removed the previous corpus at {args.out}: this build "
                "refused to publish, so anything still there is stale and "
                "would train as if it were this run's",
                file=sys.stderr,
            )
        return 1

    families: dict[str, int] = {}
    verdicts: dict[str, int] = {}
    disagreements: list[str] = []
    raw_findings = 0
    distinct_sites = 0
    for example in triage:
        families[example["workload_family"]] = (
            families.get(example["workload_family"], 0) + 1
        )
        verdicts[example["label"]["verdict"]] = (
            verdicts.get(example["label"]["verdict"], 0) + 1
        )
        raw_findings += example["finding_counts"]["raw"]
        distinct_sites += example["finding_counts"]["distinct_sites"]
        if example["ground_truth"]["agrees"] is False:
            disagreements.append(example["scenario_id"])

    manifest = {
        "schema": CORPUS_SCHEMA,
        "scenarios": len(triage),
        "examples": {"triage": len(triage), "proposal": len(proposals),
                     "total": len(triage) + len(proposals)},
        "rejected_reports": rejected,
        "findings": {"raw": raw_findings, "distinct_sites": distinct_sites},
        "workload_families": families,
        "verdicts": verdicts,
        "ground_truth_disagreements": disagreements,
        "run_meta": run_meta,
    }
    publish(
        args.out,
        {
            "triage.jsonl": "".join(json.dumps(row) + "\n" for row in triage),
            "proposal.jsonl": "".join(json.dumps(row) + "\n" for row in proposals),
            "manifest.json": json.dumps(manifest, indent=2) + "\n",
        },
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
