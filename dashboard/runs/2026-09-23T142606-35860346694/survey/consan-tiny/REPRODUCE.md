# Reproduce `consan-tiny`

Sanitizer case `consan-tiny` from run `2026-09-23T142606-35860346694` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `1680309d023be4843a3fe9cd176a014eafbe059c`
- Date: 2026-09-23 14:26:06 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/35860346694
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `a3a161beb9effe72bfce30d85d383b8be18b1e5c` (https://github.com/ROCm/rocm-systems/actions/runs/35620132711)

## Observed

- Verdict: `error`
- Execution: `error`
- Findings: 0
- Reason: `combined_hook_exit_86`

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 1680309d023be4843a3fe9cd176a014eafbe059c
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa/tiny.hsaco -- a gfx code object built from fixtures/kernels/tiny_vecadd.hip: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `hipcc --genco --offload-arch=gfx950 recipes/sanitizers/fixtures/kernels/tiny_vecadd.hip -o tmp.o`; `if head -c 24 tmp.o | grep -qF __CLANG_OFFLOAD_BUNDLE__; then clang-offload-bundler --type=o --unbundle --input=tmp.o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output=recipes/sanitizers/fixtures/isa/tiny.hsaco; else cp tmp.o recipes/sanitizers/fixtures/isa/tiny.hsaco; fi && rm -f tmp.o`. The conditional matters: --genco emits a raw code object on some ROCm builds and a clang-offload bundle on others, and the recorded digest is of the unbundled object.
- fixtures/bin/consan_tiny_load -- a host repro binary built from fixtures/kernels/consan_load.hip: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -DOBJECT="\"$(pwd)/recipes/sanitizers/fixtures/isa/tiny.hsaco\"" recipes/sanitizers/fixtures/kernels/consan_load.hip -o recipes/sanitizers/fixtures/bin/consan_tiny_load`. The -DOBJECT define is required: it is how the binary learns which code object to load.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running. `aorta` sets `HSA_TOOLS_LIB`, `HSA_TOOLS_DISABLE_REGISTER`, `RJ_CONSAN_MODE`, `RJ_CONSAN_PRESET`, `RJ_CONSAN_POLICY` for you from the recipe's policy block.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-consan-tiny.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/isa/tiny.hsaco` | `2c1ac5e57654a02a2b234a66dcbcdcbc054978983e999a12ec0fad413b9904df` |
| `fixtures/bin/consan_tiny_load` | `d110a697d5866e8aedc0b2475ff3437719e1de59836432436ee11862bdaf4f75` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:tiny.hsaco` | `2c1ac5e57654a02a2b234a66dcbcdcbc054978983e999a12ec0fad413b9904df` |
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/consan_tiny_load` |
| `command_sha256` | `d110a697d5866e8aedc0b2475ff3437719e1de59836432436ee11862bdaf4f75` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `3ecd2de3786db4fd785c8ff1c2ae9939fa5bf5d28b928c409f678b9597b7fa95` |
| `selected_identity_sha256` | `eac462821a1c9976b136e7ad9ce9288915913c36964924a87f7c78ed89ec5a14` |
| `selected_kernel` | `tiny_vecadd` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
