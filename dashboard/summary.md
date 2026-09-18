# Sanitizers Nightly · gfx950

Run `2026-09-18T142430-35344430257` · commit `684b4600dbb5` · 2026-09-18 14:24:30 UTC

✅ **HEALTHY** — 3/3 sanitizer outcomes match their baselines

Observed `WARN` or `FAIL` verdicts may be expected positive-control outcomes. Baseline status is the regression-health signal.

| Recipe | Backend | Baseline status | Observed | Expected | Execution | Findings | Coverage |
|---|---|---|---|---|---|--:|---|
| daily-waitcheck-gemm | waitcheck (static) | ✅ **Expected outcome** | `warn` | `warn` | complete | 64 | — |
| daily-consan-clean | consan (dynamic) | ✅ **Expected outcome** | `pass` | `pass` | complete | 0 | 0/0, 2/2 |
| daily-consan-racy | consan (dynamic) | ✅ **Expected outcome** | `fail` | `fail` | complete | 1 | 0/0, 2/2 |

Two views below: **Expected behavior (guardrails)** (baseline-checked, the gate) and **Workload survey (observed-only)** (non-gating).

## Expected behavior (guardrails) · Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `warn` · expected `warn`
Observation: waitcheck warn; 64 finding(s) (wait_hazard)
backend `rj_waitcheck` `e1d2e5451e02` · selection `top_dispatch_count` top-3 · 3 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `gemm_NT_M256_N4096_K1024` | 479 | `warn` | 32 | `sol_126578.hsaco` | `57c5d8efa4` | — |
| `gemm_NT_M128_N4096_K1280` | 471 | `warn` | 0 | `sol_175415.hsaco` | `57c5d8efa4` | same code object as gemm_NT_M256_N4096_K1024; scanned once |
| `gemm_TT_M64_N64_K1280` | 440 | `warn` | 32 | `sol_137678.hsaco` | `aeb46fded1` | — |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| waitcheck | `wait_hazard` | warning | 64 | sol_126578.hsaco:gfx950[0]:.text+0x1804: missing s_waitcnt lgkmcnt(14) before use of v42 |

</details>

<details><summary><b>daily-consan-clean</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `pass` · expected `pass`
Observation: consan pass; preflight pass
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `consan_lds_race` | 1 | `pass` | 0 | `—` | `—` | — |

</details>

<details><summary><b>daily-consan-racy</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `fail` · expected `fail`
Observation: consan fail; 1 finding(s) (sampled_conflict); preflight pass
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `consan_lds_race_2wave` | 1 | `fail` | 1 | `—` | `—` | — |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| consan | `sampled_conflict` | race | 1 | [rocjitsu-dbi-hooks] ConSan conflict reader=105316173954704 first_index=2 second_index=13 first_kind=2 second_kind=1 first_owner=0 second_owner=1 epoch=0 gener… |

</details>

## Workload survey (observed-only)

How real GPU kernels behave under AMD's sanitizers — **waitcheck** (static `s_waitcnt` wait-count scan) and **ConSan** (dynamic data-race check); where both produced a report the kernel is shown under each, and a scan that was skipped or whose report is missing still appears, marked report missing with no verdict. **No expected-behavior comparison on this tab**; an `error` / `fail` / `warn` here is an observation of how the kernel behaved, not a regression. Each case lists a copy-paste command to reproduce the run.

Surveyed 3 kernels across 6 sanitizer runs — 3 pass · 1 warn · 2 error

| Kernel | waitcheck | ConSan | Findings | Note |
|---|---|---|--:|---|
| gemm | `warn` | `error` | 32 | combined_hook_timeout |
| lds dispatch | `pass` | `pass` | 0 | — |
| tiny | `pass` | `error` | 0 | combined_hook_exit_86 |

<details><summary><b>consan-gemm</b> — observed `error`</summary>

Observation: consan error; reason combined_hook_timeout; gemm_f32_ss: combined_hook_timeout; preflight error

Reason: `combined_hook_timeout — gemm_f32_ss: combined_hook_timeout`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `gemm_f32_ss` | 1 | `error` | 0 | `consan_gemm_f32.hsaco` | `57c5d8efa4` | combined_hook_timeout |

</details>

<details><summary><b>consan-lds-dispatch</b> — observed `pass`</summary>

Observation: consan pass; preflight pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-lds-dispatch.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `lds_reduce` | 1 | `pass` | 0 | `lds.hsaco` | `41bcc7371e` | — |

</details>

<details><summary><b>consan-tiny</b> — observed `error`</summary>

Observation: consan error; reason combined_hook_exit_86; tiny_vecadd: combined_hook_exit_86; preflight error

Reason: `combined_hook_exit_86 — tiny_vecadd: combined_hook_exit_86`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-tiny.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `tiny_vecadd` | 1 | `error` | 0 | `tiny.hsaco` | `8b4c3cb62f` | combined_hook_exit_86 |

</details>

<details><summary><b>waitcheck-gemm</b> — observed `warn`</summary>

Observation: waitcheck warn; 32 finding(s) (wait_hazard)

Finding: `consan_gemm_f32.hsaco:gfx950[0]:.text+0x1804: missing s_waitcnt lgkmcnt(14) before use of v42`

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
| `lds_reduce` | 1 | `pass` | 0 | `lds.hsaco` | `41bcc7371e` | — |

</details>

<details><summary><b>waitcheck-tiny</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-tiny.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 | Detail |
|---|--:|---|--:|---|---|---|
| `tiny_vecadd` | 1 | `pass` | 0 | `tiny.hsaco` | `8b4c3cb62f` | — |

</details>

## History / trend

| Run | Commit | daily-waitcheck-gemm | daily-consan-clean | daily-consan-racy | Gate |
|---|---|---|---|---|---|
| 2026-09-18T142430-35344430257 | `684b4600dbb5` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-17T162056-35237696462 | `d2fa5f1645c2` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-09-17T132440-35221004771 | `73409f79a56a` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-09-17T112050-35210422045 | `73409f79a56a` | ✅ **Match**<br>Observed: `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-09-17T074922-35189763335 | `73409f79a56a` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-09-16T131107-35095734136 | `73409f79a56a` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | ❌ **Report missing**<br>Observed: `—` | Incomplete |
| 2026-09-15T130837-34968668435 | `73409f79a56a` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ❌ **Mismatch**<br>Observed: `error`; expected `pass` | ❌ **Mismatch**<br>Observed: `error`; expected `fail` | Regression |
| 2026-09-14T124757-34843407133 | `73409f79a56a` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-13T124242-34756892631 | `73409f79a56a` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-12T124242-34693445702 | `73409f79a56a` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-11T124542-34598790075 | `73409f79a56a` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-10T124939-34476548600 | `6a8c70da0add` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-09T124656-34350894756 | `6a8c70da0add` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-08T124853-34225959457 | `d84bea127a2e` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
| 2026-09-07T124249-34121807780 | `d5a8bba0383e` | ❌ **Mismatch**<br>Observed: `error`; expected `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Regression |
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
