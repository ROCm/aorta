# Reproduce `consan-lds-dispatch`

Sanitizer case `consan-lds-dispatch` from run `2026-08-19-32251405821` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `b4be48fdd6dd7e12056a6141c93377feed9f1688`
- Date: 2026-08-19
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/32251405821

## Observed

- Verdict: `error`
- Execution: `error`
- Findings: 0
- Reason: `combined_hook_exit_86`

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout b4be48fdd6dd7e12056a6141c93377feed9f1688
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa/lds.hsaco -- a gfx code object built from fixtures/kernels/lds_reduce.hip: `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `hipcc --genco --offload-arch=gfx950 recipes/sanitizers/fixtures/kernels/lds_reduce.hip -o tmp.o`; `if head -c 24 tmp.o | grep -qF __CLANG_OFFLOAD_BUNDLE__; then clang-offload-bundler --type=o --unbundle --input=tmp.o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output=recipes/sanitizers/fixtures/isa/lds.hsaco; else cp tmp.o recipes/sanitizers/fixtures/isa/lds.hsaco; fi && rm -f tmp.o`. The conditional matters: --genco emits a raw code object on some ROCm builds and a clang-offload bundle on others, and the recorded digest is of the unbundled object.
- fixtures/bin/lds_dispatch -- a host repro binary built from fixtures/kernels/lds_dispatch.hip: `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -DLDS_HSACO="\"$(pwd)/recipes/sanitizers/fixtures/isa/lds.hsaco\"" recipes/sanitizers/fixtures/kernels/lds_dispatch.hip -o recipes/sanitizers/fixtures/bin/lds_dispatch`. The -DLDS_HSACO define is required: it is how the binary learns which code object to load.

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
| `fixtures/isa/lds.hsaco` | `4b06ebb114f0ef640d68ecb89f549247a2ec1dfeae4f56ecd0e3a3b3fd5ea086` |
| `fixtures/bin/lds_dispatch` | `aa1dd66638cc40ce635e79f89d946134607a034098a724d18ba25845b1d973d7` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:lds.hsaco` | `4b06ebb114f0ef640d68ecb89f549247a2ec1dfeae4f56ecd0e3a3b3fd5ea086` |
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/lds_dispatch` |
| `command_sha256` | `aa1dd66638cc40ce635e79f89d946134607a034098a724d18ba25845b1d973d7` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `00ef00a3a202d4bc79202d370fab4337c5f152f0d80d3e9e9c0833cd63d3fc97` |
| `selected_identity_sha256` | `a85373841d5b2ed18d907b5548c1f8681997f7c9d2a53d2f465d224b9e2efac6` |
| `selected_kernel` | `lds_reduce` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
