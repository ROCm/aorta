# Workload survey (Tab 2) — public-safe generic-GEMM survey

This directory populates the **Workload survey (observed-only)** tab (Tab 2) of
the sanitizer dashboard rendered by
`scripts/sanitizers/gen_sanitizer_dashboard.py --survey`. It realizes the #367
scope update ("populate aorta-internal kernels on Tab 2") in a strictly
de-branded, public-safe form.

> **These fixtures are not what the published dashboard shows.** The nightly
> (`.github/workflows/sanitizers-nightly.yml`) renders Tab 2 from
> `--informational-results-dir`, i.e. live output of the `daily-consan-*` /
> `daily-waitcheck-*` recipes, and never passes `--survey`. This directory is the
> `--survey` path: local rendering, unit tests, and the de-branded reproduction
> recipes. Read the verdicts below as *recorded fixtures*, not as current CI
> state — and see the staleness note under the table.

Tab 2 is **observed-only**: it shows kernels drawn from multiple workloads with
no expected/baseline comparison. A `warn`, `error`, or `not_checked` here is an
observation, **never** a regression — it does not affect the guardrail gate
(Tab 1).

## What Tab 2 shows

Each kernel is run under **both** sanitizers — `waitcheck` (static ISA scan) and
`ConSan` (dynamic) — giving six survey cases:

| Kernel | Workload label | Sanitizer | Observed verdict |
|---|---|---|---|
| `hipblaslt_gemm_f32_nt_128x128` | `hipblaslt:gemm_f32` | waitcheck (static) | `warn` (many `wait_hazard` findings) |
| `hipblaslt_gemm_f32_nt_128x128` | `hipblaslt:gemm_f32` | ConSan (dynamic) | `error` (`consan_strict_load_rejection`) |
| `tiny_vecadd` | `synthetic:vecadd` | waitcheck (static) | `pass` |
| `tiny_vecadd` | `synthetic:vecadd` | ConSan (dynamic) | `error` (fails closed, exit 86) |
| `lds_reduce` | `synthetic:lds_reduce` | waitcheck (static) | `pass` |
| `lds_reduce` | `synthetic:lds_reduce` | ConSan (dynamic) | `error` (fails closed, exit 86) — **stale, now passes upstream** |

Showing an `error`/`warn` here is intended — Tab 2 records what the sanitizers
observed, including fail-closed behavior on heavy production code objects.

### Staleness of the recorded ConSan verdicts (2026-08-27)

The three `*_consan` fixtures were recorded on rocjitsu `db0c47df` and have not
been regenerated since. They have not all drifted the same way, and one has not
drifted at all — read the fixture you care about rather than the date.
Regenerating needs a gfx950 host (see "Regenerating" below); until then:

* **`lds_reduce_consan` is wrong in verdict.** Its exit 86 was
  [ROCm/rocm-systems#9972](https://github.com/ROCm/rocm-systems/issues/9972)
  (zero captured records), fixed in `15275dad`. The equivalent nightly lane now
  observes a clean `pass` (`access=5/5`, `barrier=2/2`,
  `dynamic_complete=true`).
* **`gemm_f32_consan` is right in verdict but wrong in cause.** It still records
  `consan_strict_load_rejection` / exit 92, which is what CI reports — but the
  underlying rejection is no longer the overlapping-anchor defect
  ([#10378](https://github.com/ROCm/rocm-systems/issues/10378), fixed). It is now
  the patched-image growth ceiling, because the extracted object grew from
  15.5 MB to ~183 MiB with ROCm 7.2.4. See
  [`docs/sanitizers/consan-gemm-patched-image-growth-cap.md`](../../../docs/sanitizers/consan-gemm-patched-image-growth-cap.md).
* **`tiny_vecadd_consan` is still accurate.** `tiny_vecadd` has no MOI-admissible
  sites (`access=0/0`, `applicable=false`, "no MOI report sites"), so strict
  require-records fails closed at exit 86 by design. Measured 2026-08-27: giving
  it a *dispatching* driver does not change this — the only difference is the
  message ("1 auto MOI report buffer(s)" instead of "0"), never the verdict.

## Layout

```
recipes/sanitizers/survey/
├── README.md                      # this file
├── generic_gemm_survey.json       # the --survey spec (committed, generated)
├── generic-gemm-survey.yaml       # reproduction recipe: hipBLASLt GEMM, both sanitizers
├── tiny-vecadd-survey.yaml        # reproduction recipe: tiny_vecadd control, both sanitizers
├── lds-reduce-survey.yaml         # reproduction recipe: lds_reduce control, both sanitizers
└── reports/<case>/sanitizer_report.json   # committed, scrubbed recorded outputs
```

The spec's `report_path` points at the committed fixtures so the dashboard loads
real recorded data at render time. The spec's `report_rel` is the *published*
drill-down link (`survey/<case>/sanitizer_report.json`), which the publish step
co-locates next to `index.html` — mirroring the guardrail `runs/<id>/`
co-location.

## Rendering

```bash
python scripts/sanitizers/gen_sanitizer_dashboard.py \
    --results-dir <guardrail-run-dir> \
    --baselines recipes/sanitizers/fixtures/expected/verdict_baselines.json \
    --survey recipes/sanitizers/survey/generic_gemm_survey.json \
    --out-dir dashboard
```

## Regenerating

* **The spec** (`generic_gemm_survey.json`) is generated deterministically from
  the committed report fixtures — re-running reproduces it byte-for-byte:

  ```bash
  python scripts/sanitizers/gen_survey_spec.py \
      --reports-dir recipes/sanitizers/survey/reports \
      --out recipes/sanitizers/survey/generic_gemm_survey.json
  ```

* **The reports** are recorded GPU outputs, not fabricated. Each committed
  fixture directory holds exactly **one** sanitizer's report
  (`reports/<kernel>_waitcheck/` and `reports/<kernel>_consan/`), so the recorded
  layout is *per sanitizer*, not one combined report per kernel. To regenerate
  them on a gfx950 host, run the reproduction recipe (`*-survey.yaml`) **once per
  sanitizer** — set `sanitizers:` to a single entry (`[waitcheck]`, then
  `[consan]`) so each run emits a single-sanitizer `sanitizer_report.json` that
  maps to one fixture directory. (Running both sanitizers in one plan, as the
  recipe lists them for provenance, produces a single combined report whose
  `checks[]` you would then have to split into the per-sanitizer fixtures.) Then
  scrub any absolute paths / labels before committing (see the policy below).

## De-branded / public-safe policy (CLAUDE.md rule #4)

This lives in the **public** `ROCm/aorta` repo, so it carries **no** customer,
project-codename, org-label, ticket, or NDA identifiers.

* The GEMM lane scans the **generic public hipBLASLt gfx950 f32 Tensile object**
  (`TensileLibrary_*SB*gfx950.co` content, the same kind
  `scripts/sanitizers/prepare_gemm_isa.py` prepares). Only the *label* was ever
  customer-branded; the scanned ISA itself is generic public content.
* `tiny_vecadd` and `lds_reduce` are ordinary, independently-authored synthetic
  repros with no customer association.
* All committed data is **scrubbed**: private absolute run-area paths are
  replaced with generic relative paths (e.g.
  `survey_isa/hipblaslt_gemm_f32.hsaco`); customer/codename/ticket names are
  removed; kernels are named generically. The `aorta.sanitizer_report/0.1`
  schema is preserved so the generator and `summarize_case` parse the fixtures.
* `tests/sanitizers/test_survey_generic_gemm.py` enforces a scrub guard: a
  forbidden-substring denylist (customer/codename/org tokens, private absolute
  paths, internal run/user identifiers) must not appear in the committed spec,
  the fixtures, or the rendered dashboard output.
