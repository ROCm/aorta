# Sanitizers Nightly · gfx950

Run `2026-08-10-31386669577` · commit `7f3ba90830b9` · 2026-08-10

✅ **HEALTHY** — 3/3 sanitizer outcomes match their baselines

Observed `WARN` or `FAIL` verdicts may be expected positive-control outcomes. Baseline status is the regression-health signal.

| Recipe | Backend | Baseline status | Observed | Expected | Execution | Findings | Coverage |
|---|---|---|---|---|---|--:|---|
| daily-waitcheck-gemm | waitcheck (static) | ✅ **Expected outcome** | `warn` | `warn` | complete | 32 | — |
| daily-consan-clean | consan (dynamic) | ✅ **Expected outcome** | `pass` | `pass` | complete | 0 | 0/0, 2/2 |
| daily-consan-racy | consan (dynamic) | ✅ **Expected outcome** | `fail` | `fail` | complete | 64 | 0/0, 2/2 |

## Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `warn` · expected `warn`
backend `rj_waitcheck` `8431040601e7` · selection `top_dispatch_count` top-3 · 3 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `gemm_NT_M256_N4096_K1024` | 479 | `warn` | 32 | `sol_126578.hsaco` | `80c7c264ad` |
| `gemm_NT_M128_N4096_K1280` | 471 | `—` | 0 | `sol_175415.hsaco` | `80c7c264ad` |
| `gemm_TT_M64_N64_K1280` | 440 | `—` | 0 | `sol_137678.hsaco` | `80c7c264ad` |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| waitcheck | `wait_hazard` | warning | 32 | sol_126578.hsaco:gfx950[0]:.text+0x12d610: missing s_waitcnt lgkmcnt(0) before def of s45 |

</details>

<details><summary><b>daily-consan-clean</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `pass` · expected `pass`
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `consan_lds_race` | 1 | `pass` | 0 | `—` | `—` |

</details>

<details><summary><b>daily-consan-racy</b> — ✅ **Expected outcome**</summary>

Observed sanitizer verdict `fail` · expected `fail`
backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution complete

| Kernel | Dispatch | Observed sanitizer verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `consan_lds_race_2wave` | 1 | `fail` | 64 | `—` | `—` |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| consan | `1` | race | 64 | [rocjitsu-dbi-hooks] ConSan MOI auto replay diagnostic reader=32453664 index=0 kind=1 code_object=fnv1a64:ee4f81229a69f3ed report_generation=2 generation=2 epo… |

</details>

## History / trend

| Run | Commit | daily-waitcheck-gemm | daily-consan-clean | daily-consan-racy | Gate |
|---|---|---|---|---|---|
| 2026-08-10-31386669577 | `7f3ba90830b9` | ✅ **Match**<br>Observed: `warn` | ✅ **Match**<br>Observed: `pass` | ✅ **Match**<br>Observed: `fail` | Healthy |
