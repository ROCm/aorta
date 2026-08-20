# Reproduce `consan-clean`

Sanitizer case `consan-clean` from run `2026-08-17-32028590251` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `b4be48fdd6dd7e12056a6141c93377feed9f1688`
- Date: 2026-08-17
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/32028590251

## Observed

- Verdict: `pass`
- Execution: `complete`
- Findings: 0

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout b4be48fdd6dd7e12056a6141c93377feed9f1688
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/bin/consan_lds_race -- a host repro binary built from fixtures/repro/consan_lds_race.hip: `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -O1 -g recipes/sanitizers/fixtures/repro/consan_lds_race.hip -o recipes/sanitizers/fixtures/bin/consan_lds_race`.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running. `aorta` sets `HSA_TOOLS_LIB`, `HSA_TOOLS_DISABLE_REGISTER`, `RJ_CONSAN_MODE`, `RJ_CONSAN_POLICY` for you from the recipe's policy block.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-consan-clean.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/bin/consan_lds_race` | `759c96c9016e8c7596c3f38b5948c0fc7f3dade8cff5b45c5e06ead7e92389b8` |

## Recorded digests

| Key | Value |
|---|---|
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/consan_lds_race` |
| `command_sha256` | `759c96c9016e8c7596c3f38b5948c0fc7f3dade8cff5b45c5e06ead7e92389b8` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `00ef00a3a202d4bc79202d370fab4337c5f152f0d80d3e9e9c0833cd63d3fc97` |
| `selected_identity_sha256` | `c309b9bffe94b6e47f08114af5391a49a508c70461bbd6d90b40356298da06f0` |
| `selected_kernel` | `consan_lds_race` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
