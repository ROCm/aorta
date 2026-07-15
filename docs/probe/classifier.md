# `aorta probe` — Five-Tier Classifier Reference (Issue #188, Phase 2)

This document is the **stable detector-ID reference** for the
`aorta probe` Phase 2 classifier. Each `result.json::failure_detectors_fired`
or `warn_detectors_fired` entry is a string from one of the tiers below.
Built-in (`tier1:` / `tier2:` / `tier3:` / `tier4:`) and custom
(`custom:<id>`) detectors are listed as peers in every consumer
(matrix.md, matrix.json, `aorta bundle`).

The classifier is invoked from `SubprocessWorkload.run()` post-exit; see
`src/aorta/probe/classifier/__init__.py` for the entry point.

---

## Tier 1 — Process Detectors

Source: `src/aorta/probe/classifier/tier1_process.py`.

| ID | Fires when |
|---|---|
| `tier1:exit_nonzero` | `proc.returncode != 0` and no signal fired |
| `tier1:sigsegv` | `returncode == -signal.SIGSEGV` |
| `tier1:sigabrt` | `returncode == -signal.SIGABRT` |
| `tier1:sigbus` | `returncode == -signal.SIGBUS` |
| `tier1:timeout` | `Popen.wait(timeout=recipe.timeout_per_trial)` raised `TimeoutExpired` |
| `tier1:coredump` | Any `core.*` (or bare `core`) file exists directly under `<trial_dir>/` post-exit |
| `tier1:exec_failed` | The wrapped command never launched — `Popen` raised ENOENT / EACCES / ENOEXEC |

Encounter order: when the command never launched, `exec_failed` fires
alone and suppresses every other detector (including coredump).
Otherwise (the command launched) the order is timeout > signal >
exit_nonzero, with coredump appended after whichever of those fired.
The verdict resolver preserves this order.

`tier1:timeout` and `tier1:exec_failed` are **error detectors** (issue
#230): a trial whose only fired detectors are these resolves to
`verdict = "error"` (no valid observation), not `"fail"`. They land in
`error_detectors_fired`, not `failure_detectors_fired`. See
[Verdict Precedence](#verdict-precedence).

**Coredump caveat.** The dispatcher does **not** force
`cwd=<trial_dir>` on the workload's `Popen` -- the user's `--`
command often references files by relative path and a forced cwd
would silently break the repro. With the kernel-default
`core_pattern` (`core`/`core.<pid>` next to the process's cwd), core
files therefore land in `aorta probe`'s invocation cwd, **not** the
trial dir, and `tier1:coredump` will not fire on a real segfault.
To wire coredump detection, set
`/proc/sys/kernel/core_pattern` to an absolute template that
interpolates the trial dir (e.g.
`/path/to/run/core.%e.%p` and a sidecar collector), or set
`ulimit -c unlimited` and have the workload write its own core
artifact into `$AORTA_PROBE_TRIAL_DIR`.

## Tier 2 — Hang Monitor

Source: `src/aorta/probe/classifier/tier2_hang.py`.

| ID | Fires when |
|---|---|
| `tier2:hang` | Two-of-three predicate (stdout silent for `hang_window_sec`, GPU idle per `amd-smi monitor` GFX% < 5, `/proc/<pid>/io` rchar+wchar delta = 0) AND elapsed > `hang_grace_period_at_start`. The GPU-idle leg gracefully degrades to "always False" when `amd-smi` is missing/unparseable, so the predicate collapses to 2-of-2 (stdout + IO) on hosts without ROCm telemetry. |

Defaults (overridable per recipe top-level key in `mode: probe`):

- `hang_window_sec`: 30 seconds
- `hang_grace_period_at_start`: 60 seconds

A `HangMonitor` thread runs alongside `proc.wait()`, polling at most
every 5 seconds. The first time the predicate trips, the monitor flips
its `hang_detected` flag. The workload reads the flag post-exit.

## Tier 3 — Kernel + GPU Detectors

Source: `src/aorta/probe/classifier/tier3_kernel.py`.

`dmesg` scan IDs:

| ID | Regex anchor |
|---|---|
| `tier3:amdgpu_reset` | `amdgpu: GPU reset` |
| `tier3:sdma_timeout` | `SDMA semaphore timeout \| SDMA hang` |
| `tier3:vm_l2_fault` | `VM_L2_PROTECTION_FAULT` |
| `tier3:xgmi_link_error` | `XGMI.*(?:error\|fail\|link down)` |
| `tier3:pcie_aer_fatal` | `AER:?\s+Fatal` |

`amd-smi` snapshot diff IDs:

| ID | Fires when | Severity |
|---|---|---|
| `tier3:vram_growth` | `vram_used_mib(post) - vram_used_mib(pre) >= 256 MiB` | **warn** |
| `tier3:thermal_throttle` | `thermal_throttle_count(post) > thermal_throttle_count(pre)` | fail |

> **`tier3:vram_growth` is advisory (warn), not a failure.** The probe samples
> WHOLE-GPU VRAM at only two points (pre/post the opaque subprocess) and cannot
> attribute the delta to the trial's own process on a multi-tenant host, so it
> never flips the verdict to `fail` — it lands in `warn_detectors_fired`. The
> kernel-fault `dmesg` IDs above stay hard failures. The set of advisory IDs is
> `TIER3_WARN_DETECTOR_IDS` in `tier3_kernel.py`. A recipe can suppress the
> check entirely (no warn) with `tier3_vram_growth: false`.

**Fail-soft**: missing `dmesg` / `amd-smi` binaries log
`tier3 disabled: <reason>` exactly **once per `aorta probe`
invocation** (FR 2.11) and the classifier continues with Tiers 1+2+4.
The single-warning rule is enforced by the per-invocation
`Tier3State` instance — the runner owns one and threads it into each
trial.

**Live amd-smi polling.** `poll_amd_smi` runs
`amd-smi monitor --csv --gfx --vram-usage` and parses the documented
CSV columns: `VRAM_USED` (summed across GPUs) and `GFX%` (max across
GPUs). The CSV path is preferred over `amd-smi metric --json`
because the column layout is stable across ROCm 6.x / 7.x; the JSON
shape has been observed to differ between point releases (e.g.
socket-vs-partition layouts on MI300). `thermal_throttle_count` is
NOT computed in the live path -- `monitor` only exposes current %
time in throttle, not a cumulative counter -- so
`tier3:thermal_throttle` fires only via the fake-shim test path
today. A future enhancement can plumb a per-trial throttle counter
through `amd-smi metric --violation --json`.

`AORTA_PROBE_AMDSMI_FAKE=vram=<MiB>,throttle=<n>[,util=<pct>]` is a
test-only shim that bypasses the real `amd-smi` invocation; unit
tests in `tests/probe/classifier/test_tier3.py` use it to exercise
the diff logic and the GPU-idle leg of the Tier 2 two-of-three
predicate without a GPU.

## Tier 4 — Built-in Pattern Library

Source: `src/aorta/probe/classifier/tier4_patterns.py`.

Version: `BUILTIN_PATTERN_VERSION = "1"` (exposed by
`aorta probe --list-patterns --version`).

| ID | Description | Sample |
|---|---|---|
| `tier4:python_traceback` | Python traceback header line | `Traceback (most recent call last):` |
| `tier4:hip_error` | HIP error code | `hipError_OutOfMemory` |
| `tier4:cuda_error` | CUDA error code | `cudaErrorIllegalAddress` |
| `tier4:rocm_error` | ROCm error-code marker | `Error code: 1` |
| `tier4:nccl_rccl_error` | NCCL/RCCL collective error | `NCCL error: ...` |
| `tier4:collective_timeout` | Torch distributed watchdog timeout | `Watchdog caught collective operation timeout` |
| `tier4:nan_signature` | Training-loss NaN signature | `loss is NaN` |

Each pattern ships a sibling fixture log at
`tests/probe/fixtures/tier4_logs/<detector-suffix>.txt` -- `.txt`
rather than `.log` so the project's `.gitignore` `*.log` rule never
silently swallows a new fixture, and the basename is the detector ID
with the `tier4:` prefix stripped (e.g. `cuda_error.txt` for
`tier4:cuda_error`). Adding a new pattern requires:

1. Add to `_BUILTIN_PATTERNS` in `tier4_patterns.py`.
2. Add fixture under `tests/probe/fixtures/tier4_logs/<suffix>.txt`.
3. Bump `BUILTIN_PATTERN_VERSION`.
4. Update this document.

Catastrophic-backtracking mitigation: every Tier 4 `re.search` runs
against a **10 MiB-capped window** (`MAX_LOG_BYTES`). Logs longer
than the cap are scanned in successive windows that overlap by
**4 KiB** (`_WINDOW_OVERLAP_BYTES`, capped at half the window) so
matches straddling a window seam — typically the multi-line shape
of `Traceback (most recent call last):` plus its body — still
fire. The same overlapping-window scheme is used by Tier 5
(`tier5_custom._iter_windows`).

## Tier 5 — Custom Patterns

Source: `src/aorta/probe/classifier/tier5_custom.py`.

Detector IDs are `custom:<id>` where `<id>` is the recipe-declared
pattern ID. Compile-validated at recipe load. Optional `condition`
fields are sandbox-validated at recipe load via
`aorta.probe.sandbox.validate_and_compile` — see
[sandbox.md](sandbox.md).

`on_match` semantics:

| Value | Effect |
|---|---|
| `fail` | Contributes to `failure_detectors_fired`. Default. |
| `warn` | Contributes to `warn_detectors_fired`. Does **not** change verdict. |
| `info` | Populates `capture{}` only. |

`required_for_pass: true` (only valid with `on_match: fail`): if the
pattern does **not** fire, the verdict resolver injects
`meta:missing_pass_signal` and `verdict = "fail"`.

## Meta Detectors

| ID | Source |
|---|---|
| `meta:missing_pass_signal` | Verdict resolver (`src/aorta/probe/classifier/verdict.py`). Synthesised when a `required_for_pass` pattern did not fire. A genuine **failure**. |
| `meta:env_file_validation_failed` | `SubprocessWorkload` when a `probe.env` (`env_passthrough_mode: file`) bundle is rejected before launch. An **error** (config/infra; the command never ran). |

## Verdict Precedence

Implemented in `src/aorta/probe/classifier/verdict.py::resolve`. The
verdict is three-way (issue #230): **`fail` > `error` > `pass`**.

1. **`fail`** — the event of interest manifested. Any Tier 1–4 detector
   (other than the error detectors below) OR any `on_match: fail` custom
   pattern fires. A `required_for_pass: true` pattern that did NOT fire
   adds `meta:missing_pass_signal` and is a `fail`.
2. **`error`** — the trial produced **no valid observation**: its only
   fired detectors are error detectors
   (`ERROR_DETECTOR_IDS = {tier1:timeout, tier1:exec_failed}`, plus the
   `meta:env_file_validation_failed` path). An infra crash, a launch
   failure, or a timeout the hang monitor did **not** recognise as a hang.
   Excluded from the matrix event-rate denominator.
3. **`pass`** — nothing fired.

Precedence is **fail > error > pass**: a timeout that the monitor *does*
recognise as a hang fires both `tier1:timeout` (error) and `tier2:hang`
(fail) → the hang wins → `fail`. Only when error detectors fired and no
failure detector did is the verdict `error`.

`failure_detectors_fired` and `error_detectors_fired` each list the
matching fired detectors in encounter order
(T1 → T2 → T3 → T4 → custom-fail → `meta:*`).

`on_match: warn` contributes only to `warn_detectors_fired`;
`on_match: info` only to `capture`. Neither changes the verdict.

## `result.json` Shape

```jsonc
{
  "verdict": "pass" | "fail" | "error",
  "exit_code": int,
  "walltime_sec": float,
  "peak_vram_mib": int | null,
  "argv": [string, ...],
  "cell_name": string,
  "trial_index": int,
  "failure_detectors_fired": [string, ...],
  "error_detectors_fired": [string, ...],   // issue #230 — infra-error signals
  "warn_detectors_fired": [string, ...],
  "capture": {string: (string | float | int)},
  "tier_durations_ms": {"tier1": float, "tier2": float, "tier3": float, "tier4": float, "tier5": float},

  // Phase 3: cell env bundle (scrubbed by aorta bundle redaction):
  "env": {string: string},

  // Phase 1 keys still present (subset of Phase 2 shape):
  "env_passthrough_mode": "inherit" | "file",
  "timed_out": bool
}
```

## Ownership

Per Open Question #1 + #5 in the rubric, the AORTA team
(`@mycpuorg` per `CODEOWNERS`) owns the Tier 4 pattern library.
Every version bump requires:

- Update to `_BUILTIN_PATTERNS` in `tier4_patterns.py`.
- New fixture log under `tests/probe/fixtures/tier4_logs/`.
- Bump of `BUILTIN_PATTERN_VERSION`.
- This document updated.
- A security-reviewer sign-off (sandbox-touching changes only).

## See Also

- [sandbox.md](sandbox.md) — `condition` expression whitelist.
- [usage.md](usage.md) — CLI walkthrough + `--list-patterns` flag.
- `tests/probe/classifier/` — unit tests for every tier.
