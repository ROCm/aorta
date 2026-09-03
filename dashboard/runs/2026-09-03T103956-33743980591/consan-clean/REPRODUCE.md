# Reproduce `consan-clean`

Sanitizer case `consan-clean` from run `2026-09-03T103956-33743980591` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `90dae9376bd3a09531478a72752ccb204e36fc5c`
- Date: 2026-09-03 10:39:56 UTC
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/33743980591
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `f92da4cb3c5b3612db3752a36f4f3d0d3e9ff768` (https://github.com/ROCm/rocm-systems/actions/runs/33641388794)

## Observed

- Verdict: `pass`
- Execution: `complete`
- Findings: 0

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 90dae9376bd3a09531478a72752ccb204e36fc5c
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/bin/consan_lds_race -- a host repro binary built from fixtures/repro/consan_lds_race.hip: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -O1 -g recipes/sanitizers/fixtures/repro/consan_lds_race.hip -o recipes/sanitizers/fixtures/bin/consan_lds_race`.

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
| `fixtures/bin/consan_lds_race` | `6a418d14a630011b94fae38a5acac5cfda2258b58c8627af27d60df7dad78660` |

## Recorded digests

| Key | Value |
|---|---|
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/consan_lds_race` |
| `command_sha256` | `6a418d14a630011b94fae38a5acac5cfda2258b58c8627af27d60df7dad78660` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `228b3dee0b4315534682fe2f86e95998ef0a5e70c0174b4fd06f3504bb82f871` |
| `selected_identity_sha256` | `c309b9bffe94b6e47f08114af5391a49a508c70461bbd6d90b40356298da06f0` |
| `selected_kernel` | `consan_lds_race` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
