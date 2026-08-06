# Sanitizers Nightly · gfx950

> ⚠️ **Stale** — latest sanitizer nightly run `31116518267` did not complete successfully (cancelled); the data below may be stale. [view failed run](https://github.com/ROCm/aorta/actions/runs/31116518267)

Run `run 31116518267` · commit `4d410a42b61a` · 2026-08-06T15:41:52+00:00

❌ **FAIL** — verdict mismatch vs baselines

| Recipe | Backend | Verdict | Baseline | Execution | Findings | Coverage |
|---|---|---|---|---|--:|---|
| daily-waitcheck-gemm | waitcheck (static) | `—` ❌ (want warn) | `warn` | missing | 0 | — |
| daily-consan-clean | consan (dynamic) | `—` ❌ (want pass) | `pass` | missing | 0 | — |
| daily-consan-racy | consan (dynamic) | `—` ❌ (want fail) | `fail` | missing | 0 | — |

## Kernel details

<details><summary><b>daily-waitcheck-gemm</b> — <code>—</code></summary>

report missing

</details>

<details><summary><b>daily-consan-clean</b> — <code>—</code></summary>

report missing

</details>

<details><summary><b>daily-consan-racy</b> — <code>—</code></summary>

report missing

</details>

## History / trend

| Run | Commit | daily-waitcheck-gemm | daily-consan-clean | daily-consan-racy | Gate |
|---|---|---|---|---|---|
| run 31116518267 | `4d410a42b61a` | `—` | `—` | `—` | red |
