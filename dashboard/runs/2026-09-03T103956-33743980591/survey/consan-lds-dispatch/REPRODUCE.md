# Reproduce `consan-lds-dispatch`

Sanitizer case `consan-lds-dispatch` from run `2026-09-03T103956-33743980591` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `90dae9376bd3a09531478a72752ccb204e36fc5c`
- Date: 2026-09-03 10:39:56 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
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
| `fixtures/isa/lds.hsaco` | `774cd97b0faad0f31a3c42586e4a831730493ff1f1eef600f1d72586be7be24b` |
| `fixtures/bin/lds_dispatch` | `ded4e7b336e6fea0b95c7f59f219c26dcd3a19ec83cbcdd5e8e5df30090ba708` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:lds.hsaco` | `774cd97b0faad0f31a3c42586e4a831730493ff1f1eef600f1d72586be7be24b` |
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/lds_dispatch` |
| `command_sha256` | `ded4e7b336e6fea0b95c7f59f219c26dcd3a19ec83cbcdd5e8e5df30090ba708` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `228b3dee0b4315534682fe2f86e95998ef0a5e70c0174b4fd06f3504bb82f871` |
| `selected_identity_sha256` | `3aed36a81b047b3813920a1efb93870ce8ca7d25f5457cd59c9508d6203f51b8` |
| `selected_kernel` | `lds_reduce` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
