# Reproduce `consan-racy`

Sanitizer case `consan-racy` from run `2026-09-16T131107-35095734136` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `73409f79a56ac10ae5374b419e5737221daadc34`
- Date: 2026-09-16 13:11:07 UTC
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/35095734136
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `3ddf9dee36e2943cf0c6f548a8735553c183aea5` (https://github.com/ROCm/rocm-systems/actions/runs/34919210236)

## Observed

- Verdict: `error`
- Execution: `error`
- Findings: 0
- Reason: `consan_output_parse_error: reader 108641345055616 access site count mismatch`

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 73409f79a56ac10ae5374b419e5737221daadc34
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
| `hook_sha256` | `176f11c07cf5e310ffe2207132bde42aceee44eb86454635ba2ff7ced6cd2cc5` |
| `selected_identity_sha256` | `1452dfe001bd4d91c933990dac33f91555d78b4ddc919ba7cb35ae2a691eacde` |
| `selected_kernel` | `consan_lds_race_2wave` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
