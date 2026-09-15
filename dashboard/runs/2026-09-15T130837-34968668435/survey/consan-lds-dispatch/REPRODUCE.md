# Reproduce `consan-lds-dispatch`

Sanitizer case `consan-lds-dispatch` from run `2026-09-15T130837-34968668435` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `73409f79a56ac10ae5374b419e5737221daadc34`
- Date: 2026-09-15 13:08:37 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/34968668435
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `3ddf9dee36e2943cf0c6f548a8735553c183aea5` (https://github.com/ROCm/rocm-systems/actions/runs/34919210236)

## Observed

- Verdict: `error`
- Execution: `error`
- Findings: 0
- Reason: `consan_output_parse_error: reader 99252414605888 access site count mismatch`

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 73409f79a56ac10ae5374b419e5737221daadc34
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa/lds.hsaco -- a gfx code object built from fixtures/kernels/lds_reduce.hip: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `hipcc --genco --offload-arch=gfx950 recipes/sanitizers/fixtures/kernels/lds_reduce.hip -o tmp.o`; `if head -c 24 tmp.o | grep -qF __CLANG_OFFLOAD_BUNDLE__; then clang-offload-bundler --type=o --unbundle --input=tmp.o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output=recipes/sanitizers/fixtures/isa/lds.hsaco; else cp tmp.o recipes/sanitizers/fixtures/isa/lds.hsaco; fi && rm -f tmp.o`. The conditional matters: --genco emits a raw code object on some ROCm builds and a clang-offload bundle on others, and the recorded digest is of the unbundled object.
- fixtures/bin/lds_dispatch -- a host repro binary built from fixtures/kernels/lds_dispatch.hip: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -DLDS_HSACO="\"$(pwd)/recipes/sanitizers/fixtures/isa/lds.hsaco\"" recipes/sanitizers/fixtures/kernels/lds_dispatch.hip -o recipes/sanitizers/fixtures/bin/lds_dispatch`. The -DLDS_HSACO define is required: it is how the binary learns which code object to load.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running. `aorta` sets `HSA_TOOLS_LIB`, `HSA_TOOLS_DISABLE_REGISTER`, `RJ_CONSAN_MODE`, `RJ_CONSAN_POLICY` for you from the recipe's policy block.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-consan-lds-dispatch.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/isa/lds.hsaco` | `b19dc7d3f3a175f87ff5d5d979aac4ef71a89466965eaa998af2ec60901eb2b3` |
| `fixtures/bin/lds_dispatch` | `ded4e7b336e6fea0b95c7f59f219c26dcd3a19ec83cbcdd5e8e5df30090ba708` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:lds.hsaco` | `b19dc7d3f3a175f87ff5d5d979aac4ef71a89466965eaa998af2ec60901eb2b3` |
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/lds_dispatch` |
| `command_sha256` | `ded4e7b336e6fea0b95c7f59f219c26dcd3a19ec83cbcdd5e8e5df30090ba708` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `176f11c07cf5e310ffe2207132bde42aceee44eb86454635ba2ff7ced6cd2cc5` |
| `selected_identity_sha256` | `ff125747c94d3ce6688b2c92be80142e1da8f56fe8c921ca307b4f3106484ae8` |
| `selected_kernel` | `lds_reduce` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
