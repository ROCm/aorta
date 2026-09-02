#!/usr/bin/env python3
"""Print sanitizer findings to stdout so the Watch agent can see them.

Watch reads a job's log. A sanitizer verdict lands in sanitizer_report.json, so
a hazard like a missing s_waitcnt never reaches the log at all and Watch reports
the job healthy — even though `watch_config.yaml` lists that exact hazard as an
expectation it should catch. Echoing the findings after the sweep puts them in
the stream Watch is actually reading.

Runs at the end of the batch job and must never fail it: any error here is
printed and swallowed, because a broken summary is not a reason to lose a run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MAX_FINDINGS = 5


def main() -> int:
    if len(sys.argv) < 2:
        print("[sanitizer] no report path given")
        return 0

    report = Path(sys.argv[1])
    if not report.is_file():
        print(f"[sanitizer] no report at {report}")
        return 0

    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[sanitizer] could not read {report}: {exc}")
        return 0

    verdict = data.get("overall_verdict")
    status = data.get("execution_status")
    print(f"[sanitizer] overall verdict={verdict} status={status} target={data.get('target')}")

    for check in data.get("checks") or []:
        findings = check.get("findings") or []
        name = check.get("sanitizer")
        print(
            f"[sanitizer] {name}: verdict={check.get('verdict')} "
            f"state={check.get('state')} findings={len(findings)}"
        )
        reason = check.get("reason")
        if reason:
            print(f"[sanitizer] {name}: reason={reason}")
        for finding in findings[:MAX_FINDINGS]:
            message = str(finding.get("message") or finding.get("code") or "").strip()
            if message:
                print(f"[sanitizer] {name} finding: {message[:400]}")
        if len(findings) > MAX_FINDINGS:
            print(f"[sanitizer] {name}: {len(findings) - MAX_FINDINGS} further finding(s) omitted")

    # Watch's expectations phrase the hard failure as "sanitizer guardrail not
    # clean", which the aorta CLI only raises for fail or error. A warn is a real
    # finding but not a guardrail failure, so it gets its own wording rather than
    # borrowing one that overstates it.
    if verdict in {"fail", "error"}:
        print(f"[sanitizer] sanitizer guardrail not clean: overall_verdict={verdict}")
    elif verdict == "warn":
        print(f"[sanitizer] sanitizer reported a warning verdict: overall_verdict={verdict}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - must never fail the batch job
        print(f"[sanitizer] summary failed: {exc}")
        sys.exit(0)
