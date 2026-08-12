# Workload survey (Tab 2) — public-safe generic-GEMM survey

This directory populates the **Workload survey (observed-only)** tab (Tab 2) of
the sanitizer dashboard rendered by
`scripts/sanitizers/gen_sanitizer_dashboard.py --survey`. It realizes the #367
scope update ("populate aorta-internal kernels on Tab 2") in a strictly
de-branded, public-safe form.

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
| `lds_reduce` | `synthetic:lds_reduce` | ConSan (dynamic) | `error` (fails closed, exit 86) |

Showing an `error`/`warn` here is intended — Tab 2 records what the sanitizers
observed, including fail-closed behavior on heavy production code objects.

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

* **The reports** are recorded GPU outputs, not fabricated. To regenerate them on
  a gfx950 host, run the reproduction recipes (`*-survey.yaml`) with the aorta
  sanitizer runner; each runs both sanitizers over its kernel and emits a
  `sanitizer_report.json`. Then scrub any absolute paths / labels before
  committing (see the policy below).

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
