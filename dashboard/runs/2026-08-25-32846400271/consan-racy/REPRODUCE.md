# Reproduce `consan-racy`

Sanitizer case `consan-racy` from run `2026-08-25-32846400271` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `905b1f9e3e16865eb1c6362e72846ffc391ec901`
- Date: 2026-08-25
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/32846400271
- Container image: `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e`
- rocjitsu bundle: `4227d40fb5b4ea76273589c56dac069af08b7aab` (https://github.com/ROCm/rocm-systems/actions/runs/32745941408)

## Observed

- Verdict: `error`
- Execution: `error`
- Findings: 0
- Reason: `combined_hook_exit_86`

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 905b1f9e3e16865eb1c6362e72846ffc391ec901
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/bin/consan_lds_race_2wave -- a host repro binary built from fixtures/repro/consan_lds_race_2wave.hip: `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -O1 -g recipes/sanitizers/fixtures/repro/consan_lds_race_2wave.hip -o recipes/sanitizers/fixtures/bin/consan_lds_race_2wave`.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running. `aorta` sets `HSA_TOOLS_LIB`, `HSA_TOOLS_DISABLE_REGISTER`, `RJ_CONSAN_MODE`, `RJ_CONSAN_POLICY` for you from the recipe's policy block.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-consan-racy.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/bin/consan_lds_race_2wave` | `9cb8228032743efcf05ec805cee056eff5933e34e44e9f85b6a48d6c3814110c` |

## Recorded digests

| Key | Value |
|---|---|
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/consan_lds_race_2wave` |
| `command_sha256` | `9cb8228032743efcf05ec805cee056eff5933e34e44e9f85b6a48d6c3814110c` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `17d54c42f5483e01a19413187775d7f173651b99ff23f603fb033225eb1e9964` |
| `selected_identity_sha256` | `1452dfe001bd4d91c933990dac33f91555d78b4ddc919ba7cb35ae2a691eacde` |
| `selected_kernel` | `consan_lds_race_2wave` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
