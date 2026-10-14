# Sanitizers Nightly · gfx950

> ⚠️ **Stale** — latest sanitizer nightly run `38397494175` (2026-10-14 19:41:55 UTC) did not complete successfully (cancelled); the data below may be stale. [view failed run](https://github.com/ROCm/aorta/actions/runs/38397494175)

Run `2026-10-14T194155-38397494175` · commit `bf117600ca07` · 2026-10-14 19:41:55 UTC

❌ **INCOMPLETE** — 3/3 sanitizer report(s) are missing

Observed `WARN` or `FAIL` verdicts may be expected positive-control outcomes. Baseline status is the regression-health signal.

| Recipe | Backend | Baseline status | Observed | Expected | Execution | Findings | Coverage |
|---|---|---|---|---|---|--:|---|
| daily-waitcheck-gemm | waitcheck (static) | ❌ **Report missing** | `—` | `warn` | ❌ **missing** | 0 | — |
| daily-consan-clean | consan (dynamic) | ❌ **Report missing** | `—` | `pass` | ❌ **missing** | 0 | — |
| daily-consan-racy | consan (dynamic) | ❌ **Report missing** | `—` | `fail` | ❌ **missing** | 0 | — |

Two views below: **Expected behavior (guardrails)** (baseline-checked, the gate) and **Workload survey (observed-only)** (non-gating).

## Expected behavior (guardrails) · Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — ❌ **Report missing**</summary>

Observed sanitizer verdict: `—`

Observation: report missing

</details>

<details><summary><b>daily-consan-clean</b> — ❌ **Report missing**</summary>

Observed sanitizer verdict: `—`

Observation: report missing

</details>

<details><summary><b>daily-consan-racy</b> — ❌ **Report missing**</summary>

Observed sanitizer verdict: `—`

Observation: report missing

</details>

## Workload survey (observed-only)

How real GPU kernels behave under AMD's sanitizers — **waitcheck** (static `s_waitcnt` wait-count scan) and **ConSan** (dynamic data-race check); where both produced a report the kernel is shown under each, and a scan that was skipped or whose report is missing still appears, marked report missing with no verdict. **No expected-behavior comparison on this tab**; an `error` / `fail` / `warn` here is an observation of how the kernel behaved, not a regression. Each case lists a copy-paste command to reproduce the run.

Surveyed 3 kernels across 6 sanitizer runs — 2 pass · 1 warn · 3 error

| Kernel | waitcheck | ConSan | Findings | Note |
|---|---|---|--:|---|
| gemm | `warn` | `error` | 32 | combined_hook_exit_1 |
| lds dispatch | `pass` | `error` | 0 | combined_hook_exit_1 |
| tiny | `pass` | `error` | 0 | combined_hook_exit_1 |

<details><summary><b>consan-gemm</b> — observed `error`</summary>

Observation: consan error; reason combined_hook_exit_1; gemm_f32_ss: combined_hook_exit_1; preflight error

Reason: `combined_hook_exit_1 — gemm_f32_ss: combined_hook_exit_1`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `gemm_f32_ss` | 1 | `error` | 0 | `consan_gemm_f32.hsaco` | `57c5d8efa4` | combined_hook_exit_1 |

</details>

<details><summary><b>consan-lds-dispatch</b> — observed `error`</summary>

Observation: consan error; reason combined_hook_exit_1; lds_reduce: combined_hook_exit_1; preflight error

Reason: `combined_hook_exit_1 — lds_reduce: combined_hook_exit_1`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-lds-dispatch.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `lds_reduce` | 1 | `error` | 0 | `lds.hsaco` | `fced371e5b` | combined_hook_exit_1 |

</details>

<details><summary><b>consan-tiny</b> — observed `error`</summary>

Observation: consan error; reason combined_hook_exit_1; tiny_vecadd: combined_hook_exit_1; preflight error

Reason: `combined_hook_exit_1 — tiny_vecadd: combined_hook_exit_1`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-tiny.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `tiny_vecadd` | 1 | `error` | 0 | `tiny.hsaco` | `e959fc4b34` | combined_hook_exit_1 |

</details>

<details><summary><b>waitcheck-gemm</b> — observed `warn`</summary>

Observation: waitcheck warn; 32 finding(s) (wait_hazard)

Finding: `consan_gemm_f32.hsaco:gfx950[0]:.text+0x293c: missing s_waitcnt lgkmcnt(14) before use of v20`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm-object.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `gemm_f32_ss` | 1 | `warn` | 32 | `consan_gemm_f32.hsaco` | `57c5d8efa4` | — |

</details>

<details><summary><b>waitcheck-lds-dispatch</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-lds-dispatch.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `lds_reduce` | 1 | `pass` | 0 | `lds.hsaco` | `fced371e5b` | — |

</details>

<details><summary><b>waitcheck-tiny</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-tiny.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `tiny_vecadd` | 1 | `pass` | 0 | `tiny.hsaco` | `e959fc4b34` | — |

</details>

## History / trend

| Run | Commit | daily-waitcheck-gemm | daily-consan-clean | daily-consan-racy | Gate |
|---|---|---|---|---|---|
| 2026-10-14T194155-38397494175 | `bf117600ca07` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | Incomplete |
| 2026-10-12T142744-38250297996 | `bf117600ca07` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-11T141417-38139192341 | `224d19f4a371` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-10T141315-38051823495 | `224d19f4a371` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-09T141446-37929907291 | `224d19f4a371` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-08T141657-37776980403 | `224d19f4a371` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-07T142601-37621140131 | `cab08a1abc4c` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-06T141219-37463222089 | `1f59e2040535` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-05T163143-37309782694 | `b306c32cf2b8` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-10-05T161430-37207730800 | `77d4fbb48a5d` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | Incomplete |
| 2026-10-03T153513-37127429854 | `77d4fbb48a5d` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-02T141417-37006721646 | `77d4fbb48a5d` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-10-01T142332-36861864709 | `87939a437aa4` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-30T143716-36715153798 | `87939a437aa4` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-29T125525-36568258934 | `e45a9ae778df` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-09-28T132908-36422119599 | `f32e64cf324f` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-09-28T131255-36318835944 | `f32e64cf324f` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | Incomplete |
| 2026-09-26T141847-36241711085 | `f32e64cf324f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-25T142218-36134840458 | `2f68c26d995d` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-24T142227-35998938187 | `fe7f92211b62` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-23T142606-35860346694 | `1680309d023b` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-22T142217-35726994798 | `c9e41ba7dff2` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-21T142115-35599335281 | `ef044c5474dc` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-20T141859-35510413832 | `684b4600dbb5` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-19T141910-35442659743 | `684b4600dbb5` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-18T142430-35344430257 | `684b4600dbb5` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-17T162056-35237696462 | `d2fa5f1645c2` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-17T132440-35221004771 | `73409f79a56a` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-09-17T112050-35210422045 | `73409f79a56a` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-09-17T074922-35189763335 | `73409f79a56a` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
