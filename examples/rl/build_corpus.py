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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


@dataclass
class Scenario:
    """One sanitizer run: one report, one verdict, one or more findings."""

    case: str
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
    defect; lane mask and LDS byte range make them different lanes of it. The
    metadata keys are absent on a Waitcheck finding, where code plus offset is
    already the site, so this degrades to that.
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
    """Every sanitizer report under a results tree, as a scenario."""
    scenarios: list[Scenario] = []
    for path in sorted(root.rglob("sanitizer_report.json")):
        case = path.parent.name
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipped {path}: unreadable ({exc})", file=sys.stderr)
            continue
        scenarios.append(
            Scenario(
                case=case,
                report_path=path,
                doc=doc,
                workload_family=WORKLOAD_FAMILIES.get(case, "unknown"),
                baseline_key=BASELINE_KEYS.get(case),
                checks=list(doc.get("checks") or []),
            )
        )
    return scenarios


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
    ground_truth = {
        "source": "verdict_baselines.json" if expected else "none",
        "baseline_key": scenario.baseline_key,
        "expected_verdict": expected.get("overall_verdict"),
        "expected_execution_status": expected.get("execution_status"),
        "observed_verdict": label.verdict,
        "observed_execution_status": scenario.doc.get("execution_status"),
        "agrees": (
            expected.get("overall_verdict") == label.verdict if expected else None
        ),
        "by_construction": scenario.case in {"consan-racy", "consan-clean"},
    }

    return {
        "schema": CORPUS_SCHEMA,
        "kind": "triage",
        "example_id": f"triage:{scenario.case}",
        "scenario_id": scenario.case,
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
                "example_id": f"proposal:{scenario.case}:{variant}",
                "scenario_id": scenario.case,
                "workload_family": scenario.workload_family,
                "variant": variant,
                "detail": detail,
                "proposal": {
                    "name": f"{scenario.case}:{variant}",
                    "raw": json.dumps(body),
                    "candidates": candidates,
                    "tried": tried,
                },
                "provenance": {"report": str(scenario.report_path), **run_meta},
            }
        )
    return out


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

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("triage.jsonl", triage), ("proposal.jsonl", proposals)):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

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
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
