# Reproduce `consan-racy`

Sanitizer case `consan-racy` from run `2026-09-03T124320-33755195031` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `90dae9376bd3a09531478a72752ccb204e36fc5c`
- Date: 2026-09-03 12:43:20 UTC
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/33755195031
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `f92da4cb3c5b3612db3752a36f4f3d0d3e9ff768` (https://github.com/ROCm/rocm-systems/actions/runs/33641388794)

## Observed

- Verdict: `fail`
- Execution: `complete`
- Findings: 64

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 90dae9376bd3a09531478a72752ccb204e36fc5c
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/bin/consan_lds_race_2wave -- a host repro binary built from fixtures/repro/consan_lds_race_2wave.hip: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -O1 -g recipes/sanitizers/fixtures/repro/consan_lds_race_2wave.hip -o recipes/sanitizers/fixtures/bin/consan_lds_race_2wave`.

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
| `fixtures/bin/consan_lds_race_2wave` | `169a27c22fc6a799c14ab717b82171172ab8325fb0aa9e103b822db344ffbeec` |

## Recorded digests

| Key | Value |
|---|---|
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/consan_lds_race_2wave` |
| `command_sha256` | `169a27c22fc6a799c14ab717b82171172ab8325fb0aa9e103b822db344ffbeec` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `228b3dee0b4315534682fe2f86e95998ef0a5e70c0174b4fd06f3504bb82f871` |
| `selected_identity_sha256` | `1452dfe001bd4d91c933990dac33f91555d78b4ddc919ba7cb35ae2a691eacde` |
| `selected_kernel` | `consan_lds_race_2wave` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
