# Reproduce `consan-tiny`

Sanitizer case `consan-tiny` from run `2026-08-13-31714026084` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. Its logs, recipe copy and inputs were pruned for falling outside the nightly's log-retention window. See `index.html` for every file actually published here.

## Run

- Commit: `68c980aa449f656a870923831f63a8aa34c145a3`
- Date: 2026-08-13
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/31714026084

## Observed

- Verdict: `error`
- Execution: `error`
- Findings: 0
- Reason: `combined_hook_exit_86`

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 68c980aa449f656a870923831f63a8aa34c145a3
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa/tiny.hsaco -- a gfx code object built from fixtures/kernels/tiny_vecadd.hip: `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `hipcc --genco --offload-arch=gfx950 recipes/sanitizers/fixtures/kernels/tiny_vecadd.hip -o tmp.o`; `if head -c 24 tmp.o | grep -qF __CLANG_OFFLOAD_BUNDLE__; then clang-offload-bundler --type=o --unbundle --input=tmp.o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output=recipes/sanitizers/fixtures/isa/tiny.hsaco; else cp tmp.o recipes/sanitizers/fixtures/isa/tiny.hsaco; fi && rm -f tmp.o`. The conditional matters: --genco emits a raw code object on some ROCm builds and a clang-offload bundle on others, and the recorded digest is of the unbundled object.
- fixtures/bin/consan_tiny_load -- a host repro binary built from fixtures/kernels/consan_load.hip: `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -DOBJECT="\"$(pwd)/recipes/sanitizers/fixtures/isa/tiny.hsaco\"" recipes/sanitizers/fixtures/kernels/consan_load.hip -o recipes/sanitizers/fixtures/bin/consan_tiny_load`. The -DOBJECT define is required: it is how the binary learns which code object to load.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running. `aorta` sets `HSA_TOOLS_LIB`, `HSA_TOOLS_DISABLE_REGISTER`, `RJ_CONSAN_MODE`, `RJ_CONSAN_POLICY` for you from the recipe's policy block.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-consan-tiny.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/isa/tiny.hsaco` | `2fc658e22ab8a8375294dd165fa65a45f02231e03d29074d45c6552aab5b0cca` |
| `fixtures/bin/consan_tiny_load` | `074dcf75bba651806f358097cf95f68c4a437da5c16d986cee016d8f9ee8c3bb` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:tiny.hsaco` | `2fc658e22ab8a8375294dd165fa65a45f02231e03d29074d45c6552aab5b0cca` |
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/consan_tiny_load` |
| `command_sha256` | `074dcf75bba651806f358097cf95f68c4a437da5c16d986cee016d8f9ee8c3bb` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `bb799f4bacf3938be04a8eda7766d410e8a4d3d4f2051d2b151a3ec24051a14b` |
| `selected_identity_sha256` | `bc26435e0939ad83867b5db6759e78987a0d5aaa260ec6bf266e6dc082216957` |
| `selected_kernel` | `tiny_vecadd` |

## Files here

See `index.html` for the browsable list. `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
