# Reproduce `waitcheck`

Sanitizer case `waitcheck` from run `2026-08-11-31490825323` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. Its logs, recipe copy and inputs were pruned for falling outside the nightly's log-retention window. See `index.html` for every file actually published here.

## Run

- Commit: `d1ba9de1aea20b0c960c92821f590fb3e8eaaa47`
- Date: 2026-08-11
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/31490825323

## Observed

- Verdict: `warn`
- Execution: `complete`
- Findings: 32

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout d1ba9de1aea20b0c960c92821f590fb3e8eaaa47
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa -- the per-transpose f32 GEMM code objects (sol_<n>.hsaco): `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `python scripts/sanitizers/prepare_gemm_isa.py --csv recipes/sanitizers/fixtures/gemm_shapes_unique.csv --out recipes/sanitizers/fixtures/isa --top-n 3`. Extracted from the shipped Tensile libraries, never compiled from a .hip source.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/isa` | `—` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:sol_126578.hsaco` | `80c7c264ad7c7a7156b575dc1732947d808fcc2a70d4d4ed8e336d174748934f` |
| `code_object:sol_137678.hsaco` | `80c7c264ad7c7a7156b575dc1732947d808fcc2a70d4d4ed8e336d174748934f` |
| `code_object:sol_175415.hsaco` | `80c7c264ad7c7a7156b575dc1732947d808fcc2a70d4d4ed8e336d174748934f` |
| `path` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/bin/rj_waitcheck` |
| `sha256` | `5c02eee0055cfa17524ceebece462936da2fcccbc629dad82cd1c388ec0909e3` |

## Files here

See `index.html` for the browsable list. `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
