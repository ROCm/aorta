# Sanitizers Nightly · gfx950

Run `2026-08-27T162009-33089985218` · commit `e61f455a130b` · 2026-08-27 16:20:09 UTC

✅ **HEALTHY** — 3/3 sanitizer outcomes match their baselines

Observed `WARN` or `FAIL` verdicts may be expected positive-control outcomes. Baseline status is the regression-health signal.

| Recipe | Backend | Baseline status | Observed | Expected | Execution | Findings | Coverage |
|---|---|---|---|---|---|--:|---|
| daily-waitcheck-gemm | waitcheck (static) | ✅ **Expected outcome** | `warn` | `warn` | complete | 64 | — |
| daily-consan-clean | consan (dynamic) | ✅ **Expected outcome** | `pass` | `pass` | complete | 0 | 0/0, 2/2 |
| daily-consan-racy | consan (dynamic) | ✅ **Expected outcome** | `fail` | `fail` | complete | 64 | 0/0, 2/2 |

Two views below: **Expected behavior (guardrails)** (baseline-checked, the gate) and **Workload survey (observed-only)** (non-gating).

## Expected behavior (guardrails) · Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `warn` · expected `warn`
Observation: waitcheck warn; 64 finding(s) (wait_hazard)
backend `rj_waitcheck` `a70945fb1135` · selection `top_dispatch_count` top-3 · 3 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `gemm_NT_M256_N4096_K1024` | 479 | `warn` | 32 | `sol_126578.hsaco` | `5bd40b78d1` |
| `gemm_NT_M128_N4096_K1280` | 471 | `—` | 0 | `sol_175415.hsaco` | `5bd40b78d1` |
| `gemm_TT_M64_N64_K1280` | 440 | `warn` | 32 | `sol_137678.hsaco` | `7ea836fe29` |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| waitcheck | `wait_hazard` | warning | 64 | sol_126578.hsaco:gfx950[0]:.text+0x28f8: missing s_waitcnt lgkmcnt(14) before use of v88 |

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
| consan | `1` | race | 64 | [rocjitsu-dbi-hooks] ConSan MOI auto replay diagnostic reader=17189248 index=0 kind=1 code_object=fnv1a64:7db8350dfcf3fedf report_generation=2 generation=2 epo… |

</details>

## Workload survey (observed-only)

How real GPU kernels behave under AMD's sanitizers — **waitcheck** (static `s_waitcnt` wait-count scan) and **ConSan** (dynamic data-race check); where both produced a report the kernel is shown under each, and a scan that was skipped or whose report is missing still appears, marked report missing with no verdict. **No expected-behavior comparison on this tab**; an `error` / `fail` / `warn` here is an observation of how the kernel behaved, not a regression. Each case lists a copy-paste command to reproduce the run.

Surveyed 3 kernels across 6 sanitizer runs — 3 pass · 1 warn · 2 error

| Kernel | waitcheck | ConSan | Findings | Note |
|---|---|---|--:|---|
| gemm | `warn` | `error` | 32 | consan_strict_load_rejection |
| lds dispatch | `pass` | `pass` | 0 | — |
| tiny | `pass` | `error` | 0 | combined_hook_exit_86 |

<details><summary><b>consan-gemm</b> — observed `error`</summary>

Observation: consan error; reason consan_strict_load_rejection; preflight error

Reason: `consan_strict_load_rejection`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `gemm_f32_ss` | 1 | `error` | 0 | `consan_gemm_f32.hsaco` | `5bd40b78d1` |

</details>

<details><summary><b>consan-lds-dispatch</b> — observed `pass`</summary>

Observation: consan pass; preflight pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-lds-dispatch.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `lds_reduce` | 1 | `pass` | 0 | `lds.hsaco` | `d00809b2b8` |

</details>

<details><summary><b>consan-tiny</b> — observed `error`</summary>

Observation: consan error; reason combined_hook_exit_86; preflight error

Reason: `combined_hook_exit_86`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-consan-tiny.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `tiny_vecadd` | 1 | `error` | 0 | `tiny.hsaco` | `682927c30b` |

</details>

<details><summary><b>waitcheck-gemm</b> — observed `warn`</summary>

Observation: waitcheck warn; 32 finding(s) (wait_hazard)

Finding: `consan_gemm_f32.hsaco:gfx950[0]:.text+0x28f8: missing s_waitcnt lgkmcnt(14) before use of v88`

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm-object.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `gemm_f32_ss` | 1 | `warn` | 32 | `consan_gemm_f32.hsaco` | `5bd40b78d1` |

</details>

<details><summary><b>waitcheck-lds-dispatch</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-lds-dispatch.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `lds_reduce` | 1 | `pass` | 0 | `lds.hsaco` | `d00809b2b8` |

</details>

<details><summary><b>waitcheck-tiny</b> — observed `pass`</summary>

Observation: waitcheck pass

Reproduce: `aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-tiny.yaml`

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `tiny_vecadd` | 1 | `pass` | 0 | `tiny.hsaco` | `682927c30b` |

</details>

## History / trend

| Run | Commit | daily-waitcheck-gemm | daily-consan-clean | daily-consan-racy | Gate |
|---|---|---|---|---|---|
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
| 2026-08-12-31598906719 | `5fbbb5c04dc4` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-12-31596128350 | `261893abafd3` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-11-31513370892 | `d80f57250f7f` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-11-31511121124 | `fc0a3ff54748` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Failed |
| 2026-08-11-31508140809 | `1663ad70a9a2` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Failed |
| 2026-08-11-31490825323 | `d1ba9de1aea2` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-10-31387689897 | `7f3ba90830b9` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
| 2026-08-10-31386669577 | `7f3ba90830b9` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
