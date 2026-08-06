# Sanitizers Nightly · gfx950

Run `run local-20260806161917` · commit `5bc10d527471` · 2026-08-06T16:19:32+00:00

✅ **PASS** — all verdicts match baselines

| Recipe | Backend | Verdict | Baseline | Execution | Findings | Coverage |
|---|---|---|---|---|--:|---|
| daily-waitcheck-gemm | waitcheck (static) | `warn` ✅ | `warn` | complete | 32 | — |
| daily-consan-clean | consan (dynamic) | `pass` ✅ | `pass` | complete | 0 | 0/0, 2/2 |
| daily-consan-racy | consan (dynamic) | `fail` ✅ | `fail` | complete | 64 | 0/0, 2/2 |

## Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — <code>warn</code></summary>

backend `rj_waitcheck` `472fcf288714` · selection `top_dispatch_count` top-3 · 3 kernel(s) · execution `complete`

| Kernel | Dispatch | Verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `gemm_NT_M256_N4096_K1024` | 479 | `warn` | 32 | `sol_126578.hsaco` | `93f09ae670` |
| `gemm_NT_M128_N4096_K1280` | 471 | `—` | 0 | `sol_175415.hsaco` | `93f09ae670` |
| `gemm_TT_M64_N64_K1280` | 440 | `—` | 0 | `sol_137678.hsaco` | `93f09ae670` |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| waitcheck | `wait_hazard` | warning | 32 | sol_126578.hsaco:gfx950[0]:.text+0x100eb8: missing s_waitcnt lgkmcnt(0) before def of s45 |

</details>

<details><summary><b>daily-consan-clean</b> — <code>pass</code></summary>

backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution `complete`

| Kernel | Dispatch | Verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `consan_lds_race` | 1 | `pass` | 0 | `—` | `—` |

</details>

<details><summary><b>daily-consan-racy</b> — <code>fail</code></summary>

backend `—` · selection `top_dispatch_count` top-1 · 1 kernel(s) · execution `complete`

| Kernel | Dispatch | Verdict | Findings | Code object | SHA-256 |
|---|--:|---|--:|---|---|
| `consan_lds_race_2wave` | 1 | `fail` | 64 | `—` | `—` |

| Sanitizer | Code | Severity | Count | Example |
|---|---|---|--:|---|
| consan | `1` | race | 64 | [rocjitsu-dbi-hooks] ConSan MOI auto replay diagnostic reader=13927600 index=0 kind=1 code_object=fnv1a64:6e7736ba22da0c89 report_generation=2 generation=2 epo… |

</details>

## History / trend

| Run | Commit | daily-waitcheck-gemm | daily-consan-clean | daily-consan-racy | Gate |
|---|---|---|---|---|---|
| run local-20260806161917 | `5bc10d527471` | `warn` | `pass` | `fail` | green |
