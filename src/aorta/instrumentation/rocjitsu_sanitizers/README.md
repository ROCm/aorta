# RocJITsu kernel sanitizers

Public, typed primitives for selecting important GPU kernels and applying
RocJITsu guardrails without importing customer-derived artifacts.

## Phase 0 capability

Available now as a Python library:

- normalize Magpie kernel summaries and generic dispatch CSVs;
- deterministically select `top_time` or `top_dispatch_count`;
- deduplicate stable kernel identities;
- run standalone Waitcheck on one exact code-object entry at a time;
- run the valid exact-entry Waitcheck CLI and retain its raw log;
- parse upstream `rj-waitcheck-diagnostic-v1` JSONL for corpus workflows;
- parse Record/Replay output from the combined Waitcheck + ConSan hook;
- preserve per-code-object ConSan coverage and fail closed on timeout, backend
  failure, missing verdicts, or incomplete coverage;
- write and strictly reload experimental `aorta.sanitizer_report/0.1` JSON.

Not available yet:

- recipe/dispatcher execution of `sanitizer_plan`;
- automatic Magpie/TraceLens execution;
- automatic RocJITsu provisioning;
- top-K ConSan execution.

The last item is intentionally fail-closed. Current RocJITsu ConSan has no
documented kernel allowlist, so wrapping a model would instrument every
supported code object it loads. A top-K ConSan request therefore returns
`not_checked` with `worklist_scope_unsupported`; it never silently falls back to
whole-application instrumentation.

## Static Waitcheck example

```python
from pathlib import Path
import hashlib

from aorta.instrumentation.rocjitsu_sanitizers import (
    KernelIdentity,
    KernelObservation,
    SelectionRequirement,
    run_sanitizers,
    select_kernels,
)

artifact = Path("/tmp/my_kernel.hsaco")
observations = [
    KernelObservation(
        identity=KernelIdentity(
            name="my_kernel",
            target="gfx950",
            code_object=str(artifact),
            code_object_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            entry_offset=0x120,
        ),
        total_time_ms=12.5,
        dispatch_count=40,
        sources=("user",),
    )
]
worklist = select_kernels(
    observations,
    requirement=SelectionRequirement.TOP_TIME,
    top_n=1,
)
report = run_sanitizers(
    worklist,
    target="gfx950",
    sanitizers=("waitcheck", "consan"),
    output_dir=Path("./sanitizer-out"),
    waitcheck_binary=Path("/path/to/rj_waitcheck"),
)
```

Waitcheck requires an exact final-code identity: code-object path plus kernel
entry offset, with code-object index when the container has multiple device
images. Fuzzy filename matching is deliberately not supported.

## Verdict and health rules

- Waitcheck structured hazard → `warn`.
- ConSan attributed race → `fail`.
- Backend timeout, launch failure, unexpected exit, missing verdict, or
  incomplete coverage → `error`.
- Missing backend or unsupported worklist scoping → `not_checked`.
- `pass` only means a requested backend ran healthily and produced no finding;
  Record/Replay's bounded snapshot is not proof that a program is race-free.

The report keeps finding severity and execution completeness separate. For
example, a Waitcheck warning plus scoped ConSan `not_checked` produces
`overall_verdict: warn` and `execution_status: partial`.

## Planned scoped ConSan flow

1. Profile the uninstrumented model/module through Magpie or TraceLens.
2. Resolve top-K observations to stable code-object/kernel identities.
3. Rerun the normal application under the combined hook.
4. Pass the identities through a new RocJITsu allowlist so ConSan instruments
   only the resolved worklist.

This keeps real framework state, arguments, streams, collectives, and fused/JIT
kernels intact without requiring generic kernel replay.
