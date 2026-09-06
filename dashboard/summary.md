# Sanitizers Nightly · gfx950

> ⚠️ **Stale** — latest sanitizer nightly run `34032862308` (2026-09-06 12:38:28 UTC) did not complete successfully (failure); the data below may be stale. [view failed run](https://github.com/ROCm/aorta/actions/runs/34032862308)

Run `2026-09-06T123828-34032862308` · commit `4b4553ef14af` · 2026-09-06 12:38:28 UTC

❌ **REGRESSION** — investigate 1/3 sanitizer outcomes that do not match their baselines

Observed `WARN` or `FAIL` verdicts may be expected positive-control outcomes. Baseline status is the regression-health signal.

| Recipe | Backend | Baseline status | Observed | Expected | Execution | Findings | Coverage |
|---|---|---|---|---|---|--:|---|
| daily-waitcheck-gemm | waitcheck (static) | ❌ **Unexpected outcome** | `error` | `warn` | ❌ **error** | 32 | — |
| daily-consan-clean | consan (dynamic) | ✅ **Expected outcome** | `pass` | `pass` | complete | 0 | 0/0, 2/2 |
| daily-consan-racy | consan (dynamic) | ✅ **Expected outcome** | `fail` | `fail` | complete | 64 | 0/0, 2/2 |

Two views below: **Expected behavior (guardrails)** (baseline-checked, the gate) and **Workload survey (observed-only)** (non-gating).

## Expected behavior (guardrails) · Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — ❌ **Unexpected outcome**</summary>

Observed sanitizer verdict `error` · expected `warn`
Observation: waitcheck error; reason worklist_not_fully_checked; 32 finding(s) (wait_hazard)
backend `rj_waitcheck` `a70945fb1135` · selection `top_dispatch_count` top-3 · 3 kernel(s) · execution ❌ **error**

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `gemm_NT_M256_N4096_K1024` | 479 | `error` | 0 | `sol_126578.hsaco` | `57c5d8efa4` |
| `gemm_NT_M128_N4096_K1280` | 471 | `—` | 0 | `sol_175415.hsaco` | `57c5d8efa4` |
| `gemm_TT_M64_N64_K1280` | 440 | `warn` | 32 | `sol_137678.hsaco` | `aeb46fded1` |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| waitcheck | `wait_hazard` | warning | 32 | sol_137678.hsaco:gfx950[0]:.text+0x4a4: missing s_waitcnt lgkmcnt(0) before def of s45 |

</details>

<details><summary><b>daily-consan-clean</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `pass` · expected `pass`
Observation: consan pass; preflight pass
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `consan_lds_race` | 1 | `pass` | 0 | `—` | `—` |

</details>

<details><summary><b>daily-consan-racy</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `fail` · expected `fail`
Observation: consan fail; 64 finding(s) (1); preflight pass
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `consan_lds_race_2wave` | 1 | `fail` | 64 | `—` | `—` |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| consan | `1` | race | 64 | [rocjitsu-dbi-hooks] ConSan MOI auto replay diagnostic reader=99634626596784 index=0 kind=1 code_object=fnv1a64:3deb93883c6c6fa8 report_generation=2 generation… |

</details>

## Workload survey (observed-only)

How real GPU kernels behave under AMD's sanitizers — **waitcheck** (static `s_waitcnt` wait-count scan) and **ConSan** (dynamic data-race check); where both produced a report the kernel is shown under each, and a scan that was skipped or whose report is missing still appears, marked report missing with no verdict. **No expected-behavior comparison on this tab**; an `error` / `fail` / `warn` here is an observation of how the kernel behaved, not a regression. Each case lists a copy-paste command to reproduce the run.

Surveyed 3 kernels across 6 sanitizer runs — 3 pass · 3 error

| Kernel | waitcheck | ConSan | Findings | Note |
|---|---|---|--:|---|
| gemm | `error` | `error` | 0 | consan_strict_load_rejection |
| lds dispatch | `pass` | `pass` | 0 | — |
| tiny | `pass` | `error` | 0 | combined_hook_exit_86 |

<details><summary><b>consan-gemm</b> — observed `error`</summary>

Observation: consan error; reason consan_strict_load_rejection; preflight error

Reason: `consan_strict_load_rejection`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `gemm_f32_ss` | 1 | `error` | 0 | `consan_gemm_f32.hsaco` | `57c5d8efa4` |

</details>

<details><summary><b>consan-lds-dispatch</b> — observed `pass`</summary>

Observation: consan pass; preflight pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-lds-dispatch.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `lds_reduce` | 1 | `pass` | 0 | `lds.hsaco` | `d17178926f` |

</details>

<details><summary><b>consan-tiny</b> — observed `error`</summary>

Observation: consan error; reason combined_hook_exit_86; preflight error

Reason: `combined_hook_exit_86`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-tiny.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `tiny_vecadd` | 1 | `error` | 0 | `tiny.hsaco` | `76d69af357` |

</details>

<details><summary><b>waitcheck-gemm</b> — observed `error`</summary>

Observation: waitcheck error; reason worklist_not_fully_checked

Reason: `worklist_not_fully_checked`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm-object.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `gemm_f32_ss` | 1 | `error` | 0 | `consan_gemm_f32.hsaco` | `57c5d8efa4` |

</details>

<details><summary><b>waitcheck-lds-dispatch</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-lds-dispatch.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `lds_reduce` | 1 | `pass` | 0 | `lds.hsaco` | `d17178926f` |

</details>

<details><summary><b>waitcheck-tiny</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-tiny.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `tiny_vecadd` | 1 | `pass` | 0 | `tiny.hsaco` | `76d69af357` |

</details>

## History / trend

| Run | Commit | daily-waitcheck-gemm | daily-consan-clean | daily-consan-racy | Gate |
|---|---|---|---|---|---|
| 2026-09-06T123828-34032862308 | `4b4553ef14af` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-05T123756-33965800908 | `4b4553ef14af` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-04T124232-33872660689 | `4b4553ef14af` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-03T124320-33755195031 | `90dae9376bd3` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-03T103956-33743980591 | `90dae9376bd3` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-02T124713-33629901482 | `30dcc055cba8` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-01T125503-33507543444 | `d11092a603d9` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-31T125804-33391758880 | `c57a1e4b0720` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-30T125306-33311286343 | `c57a1e4b0720` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-29T124958-33252284140 | `e61f455a130b` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-28T164558-33189127590 | `e61f455a130b` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-27T162009-33089985218 | `e61f455a130b` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-26T124411-32967422099 | `bed2771d52dd` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-25-32846400271 | `905b1f9e3e16` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-24-32725903878 | `78d1ae686dc3` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-23-32638584704 | `78d1ae686dc3` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-22-32572213077 | `78d1ae686dc3` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-21-32480768201 | `78d1ae686dc3` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-20-32373146952 | `78d1ae686dc3` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-20-32367664704 | `b4be48fdd6dd` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-19-32251405821 | `b4be48fdd6dd` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-18-32135585222 | `b4be48fdd6dd` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-17-32028590251 | `b4be48fdd6dd` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-16-31946286609 | `b4be48fdd6dd` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-15-31883858931 | `b4be48fdd6dd` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-14-31800000546 | `68c980aa449f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-13-31718502485 | `68c980aa449f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-13-31714026084 | `68c980aa449f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-13-31699735119 | `f1660c492df6` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-12-31615014466 | `f1660c492df6` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
