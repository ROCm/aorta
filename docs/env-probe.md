# Env Probe

Capture a versioned, schema-stable snapshot of the trial environment so
that cross-environment comparison becomes a `jq` diff instead of a
multi-day investigation.

The same code path is used three ways:

* **CLI**: `aorta env probe -o env.json` -- on-demand, by an operator.
* **B1 (per-trial runner)**: calls `collect_env()` once per trial; the
  snapshot is embedded in `TrialResult.env`.
* **B2 (matrix runner)**: calls `collect_env()` once per matrix start
  (host scope) and once per `--environment-axis` value (container
  scope).

## Why

Numerical / correctness investigations on GPU stacks routinely lose
weeks to **implicit environment state** -- a library swap between
docker images, an HSA tunable that differs between hosts, a different
hipBLASLt build pulled into one conda env vs another. Attaching a
complete machine-readable environment snapshot to every trial result
makes those confounds visible the moment they happen, instead of after
the fact.

The class of bug that motivated this is **GEMM kernel library drift
across environments** -- most commonly hipBLASLt. The schema captures
its commit, package version, library hash, and Tensile kernel
fingerprint as first-class fields, so two trials whose hipBLASLt
identities differ become trivially diffable.

## Quick start

```bash
# Default: writes env.json in the current directory
aorta env probe

# Custom path; parent dirs are created if missing
aorta env probe -o runs/exp1/env.json

# Pretty-print + spot-check key fields
jq . runs/exp1/env.json
jq '.hipblaslt.rocm_release_tweak' runs/exp1/env.json   # schema 1.1: was .commit in 1.0
jq '.partial' runs/exp1/env.json

# Diff two snapshots from different docker images
diff <(jq -S . env_a.json) <(jq -S . env_b.json)
```

## CLI

```text
Usage: aorta env probe [OPTIONS]

  Capture trial-environment state to env.json (issue #147).

Options:
  -o, --output FILE        Path to write env.json (ignored with
                           --summary / --field).  [default: env.json]
  -v, --verbose            After the brief, also print the full
                           snapshot JSON to stdout.
  --summary                Print only the one-screen brief and exit
                           (skip JSON write). Use for quick eyeballing
                           without producing an artifact.
  --field DOTTED.PATH      Print one snapshot field as JSON and exit
                           (skip file write). Example: --field
                           pytorch_build.ninja_hipcc.targets.ck_sdpa
                           .use_defines_present.USE_ROCM_CK_SDPA. For
                           keys containing '.' (e.g. 'libaotriton_v2
                           .so'), use jq on a full snapshot.
  --buck-target TEXT       Buck2 label to introspect for library
                           identity. When given, the snapshot's
                           library_introspection list is populated
                           from `buck2 audit dependencies <label>
                           --transitive --json`. Ignored if buck2
                           isn't on PATH.
  --buck-timeout INTEGER   Per-call timeout (seconds) for `buck2
                           audit dependencies`.  [default: 10]
  --help                   Show this message and exit.
```

`--summary` and `--field` are mutually exclusive output modes; both
short-circuit the JSON write entirely. Pair `aorta env probe -o
env.json` with a follow-up `aorta env probe --field …` when you
need both an archived snapshot and a scripted lookup.

The CLI is a thin wrapper. It calls `collect_env()`, writes the JSON,
and prints a multi-line per-block brief (~18 lines on a populated host).
After the brief, any `partial_reasons` entries are echoed inline so the
operator can act on them without `jq`'ing the JSON. A closing
`[PARTIAL, N reason(s)]` (or `[OK]`) marker repeats the probe state at
end-of-output. Sample:

```text
Wrote env probe to /tmp/env.json (schema_version=1.5) [PARTIAL]
  runtime:   baremetal / python=venv
  build_sys: none
  rocm:      7.2.1 (dev: None)
  hip:       7.2.53211-e1a6bc5663 (amd)
  hipblaslt: 1.2.2 rocm_release_tweak=dabb6df2b9
  rocblas:   5.2.0 rocm_release_tweak=dabb6df2b9
  miopen:    3.5.1 rocm_release_tweak=dabb6df2b9
  rccl:      2.27.7 (code=22707)
  gpu_arch:  ['gfx942'] (counts={'gfx942': 8})
  host:      kernel=5.15.0-174-generic machine=x86_64  glibc=2.35
  ck:        system=1.2.0/23d531c8  ck_tile=yes  libtorch_hip=4067 ck:: syms
  tensile:   kernel_db=filenames-sha256:743a8d…  [Tensile pip pkg: (not installed); build-time tool, normal]
  triton:    3.5.1+rocm7.2.1.gita272dfa8
  fbgemm:    in PyTorch: USE_FBGEMM=True USE_FBGEMM_GENAI=True  [fbgemm_gpu pip pkg: (not installed); separate from torch's vendored copy]
  aiter:     (not installed) [aiter pip pkg; optional ROCm inference lib]
  aotriton:  bundled=0.11.1 present=True images_dir=True  [AOTRITON_INSTALLED_PREFIX=(unset)]
  rdhc:      unavailable (system_health=null)
  python:    3.12.13 | pytorch: 2.9.1+rocm7.2.1.gitff65f5bc
  torch build: git_commit=ff65f5bc install=source | submodules(git): composable_kernel=23d531c8 aiter=9a469a60 fbgemm=8c1f8d2b
  torch flags: gpu_archs=[gfx942]  USE_ROCM=ON USE_CUDA=OFF USE_NCCL=ON USE_MKL=OFF USE_MKLDNN=ON USE_FBGEMM=yes USE_FBGEMM_GENAI=no USE_FLASH_ATTENTION=yes USE_MEM_EFF_ATTENTION=yes USE_ROCM_CK_SDPA=yes USE_ROCM_CK_GEMM=no DISABLE_AOTRITON=no FLASH_NAMESPACE=pytorch_flash
  torch syms:  pytorch_flash::=142 mha_fwd_aot=4 mha_fwd_ck=4 _efficient_attention=18 aotriton::=72 ck_tile::FmhaFwd=1820 ck_tile::FmhaBwd=1240 ck_tile::BlockFmha=890 ck_tile::TileFmha=320 group_gemm_ck=12 aiter::=0  |  libaotriton_v2.so=yes  |  -DUSE_ROCM_CK_SDPA=yes -DUSE_ROCM_CK_GEMM=no
  flags:       FLASH_ATTN=on CK_SDPA=on AOTRITON=on MEM_EFF=on
  cmake cache: 32 allowlisted entries from /work/pytorch/build/CMakeCache.txt
  ninja hipcc: c10_hip=18D archs=[gfx942] torch_cpu=14D archs=[?] torch_hip=42D archs=[gfx942]
  aiter hsa:   aiter_meta/hsa:gfx942=1180.co/3a7b9e0f aiter_meta/hsa:gfx950=420.co/c1d2e8a4
  sdpa:        flash=on mem_eff=on math=on cudnn=off

Partial reasons:
  - system_health: rdhc exited 1 (stderr: sudo: a password is required)
  - rocm.version_dev: /opt/rocm/.info/version-dev missing, empty, or unreadable

[PARTIAL, 2 reason(s)]
```

The sample shows a **source install** (`install=source`) so the new
`cmake cache` / `ninja hipcc` lines have populated data. On a wheel
install (`install=wheel`) those two lines render `(unavailable --
wheel install or build/CMakeCache.txt missing)` /
`(unavailable -- wheel install or build/build.ninja missing)`
respectively, and `pytorch_build.submodule_commits` adds a
`wheel install -- direct SHAs not recoverable; <github URL>` partial
reason instead of the resolved per-submodule SHAs shown here.

`[PARTIAL]` indicates at least one probe fell back to `None` -- the
snapshot is still complete (every key present), it just records what
was missing in `partial_reasons`. Run with `-v`/`--verbose` to also
dump the full snapshot JSON to stdout (useful for remote operators
who want to copy-paste without reading `env.json` from disk).

## Library API

```python
from aorta.instrumentation.environment import collect_env, EnvSnapshot

snapshot: EnvSnapshot = collect_env()   # NEVER raises

# Embed in a trial result (B1 pattern)
trial_result = {
    "trial_id": "...",
    "passed": True,
    "metrics": {...},
    "env": snapshot.to_dict(),
}
write_json(trial_result_path, trial_result)

# Reconstruct later (post-mortem, comparison tools)
loaded = read_json(trial_result_path)
env = EnvSnapshot.from_dict(loaded["env"])

# Surface the partial state to the user without aborting
if snapshot.partial:
    log.warning("env probe partial: %s", snapshot.partial_reasons)

# Multi-line human summary, same as the CLI prints (~18 lines on a
# populated host -- one labelled cell per top-level block).
print(snapshot.summary())
```

`collect_env()` is the **only** public entrypoint. It NEVER raises:
each probe is fail-soft, and the orchestrator body has a top-level
`try/except Exception` plus a disaster-recovery helper for any genuinely
unexpected failure. Callers always get back a valid, fully-shaped
`EnvSnapshot` object.

## env.json schema

| Top-level key | Type | Source | Notes |
| --- | --- | --- | --- |
| `schema_version` | `str` | constant | Currently `"1.5"`. See the changelog comment in `src/aorta/instrumentation/environment.py` next to the `SCHEMA_VERSION` constant for the field-by-field history. |
| `captured_at` | `str` | `datetime` | ISO-8601 UTC with trailing `Z` |
| `partial` | `bool` | computed | `True` if any probe fell back |
| `partial_reasons` | `list[str]` | per-probe | one human-readable line per fallback |
| `system_health` | `dict \| null` | `rdhc --quick --json` (subprocess) | verbatim parsed JSON; `null` when rdhc absent / sudo unavailable / timeout / malformed |
| `rocm` | `dict[str, str \| null]` | `/opt/rocm/.info/version{,_dev}`, `/sys/module/amdgpu/version` | `version`, `version_dev`, `kmd_version` |
| `hip` | `dict[str, str \| null]` | `hipconfig --version/--platform/--compiler/--runtime/--cpp_config` | five subprocesses; `--version` and `--platform` cannot be combined (no delimiter) |
| `hipblaslt` | `dict` | header parse + `sha256(libhipblaslt.so)` + sorted-filenames hash of `lib/hipblaslt/library/*` | `rocm_release_tweak` (NOT a per-hipBLASLt commit -- it's the ROCm release identifier shared across every library in a release; see note below), `package_version`, `lib_hash`, `kernel_db_revision`, `applied_prs: {}` |
| `rocblas` | `dict` | header parse + `sha256(librocblas.so)` + sorted-filenames hash of `lib/rocblas/library/*` | Same shape as `hipblaslt`. Header lives at `include/rocblas/internal/rocblas-version.h`. |
| `miopen` | `dict` | header parse + `sha256(libMIOpen.so)` + sorted-filenames hash of `share/miopen/db/*.txt` | `rocm_release_tweak`, `package_version`, `lib_hash`, `kernel_db_revision`. MIOpen drives convolution kernels on ROCm; kernel-DB drift changes which conv kernel runs. |
| `rccl` | `dict` | header parse for `NCCL_VERSION_CODE` + `sha256(librccl.so)` | `version_code` (raw int, e.g. `22707`), `version` (decoded `"2.27.7"`), `lib_hash`. RCCL is AMD's NCCL-compatible collectives library. |
| `gpu_arch` | `dict` | `rocm_agent_enumerator` subprocess (no `/dev/kfd` access typically required) | `agent_count`, `gfx_targets` (sorted unique), `agent_arch_counts` (per-arch distribution -- captures both homogeneous and mixed-arch boxes). |
| `host` | `dict` | `os.uname()` + `os.confstr("CS_GNU_LIBC_VERSION")` | `kernel_release`, `kernel_version`, `machine`, `glibc_version`. Kernel + glibc drift is the #1 confound for compiled-against-vs-runtime issues with C++ extensions. |
| `composable_kernel` | `dict` | header at `include/ck/version.h` + `nm -D` of `libtorch_hip.so` piped through `c++filt` + `torch.__config__.show()` flag scan | Two sub-blocks (`system: {version, commit, ck_tile_present}`, `pytorch_bundled: {present, symbol_count}`) plus top-level `pytorch_use_ck_sdpa` / `pytorch_use_ck_gemm` booleans (build-time flags baked into the wheel; NOT runtime env vars). System and bundled CK can drift independently. |
| `tensile` | `dict` | optional `import Tensile` + sorted-filenames hash over the union of hipBLASLt + rocBLAS kernel DBs | `package_version` (usually `null` outside builders), `kernel_db_combined_hash` |
| `triton` | `dict` | `import triton; triton.__version__` | `package_version`. ROCm Triton fork bakes the source commit into `__version__`. |
| `fbgemm` | `dict` | optional `import fbgemm_gpu` + parse of `torch.__config__.show()` for `-DUSE_FBGEMM*` defines | `package_version`, `pytorch_use_fbgemm`, `pytorch_use_fbgemm_genai`. The two booleans capture the build-time decision baked into the PyTorch wheel even when `fbgemm_gpu` isn't a separate pip package. |
| `aiter` | `dict` | `import aiter; aiter.__version__` + `importlib.metadata.version("amd_aiter" \| "aiter")` + scan of `aiter_meta/hsa/<gfx>/` (or `$AORTA_PYTORCH_SRC/third_party/aiter/hsa/`) | `package_version`, `package_dist_name` (which PyPI dist matched: `amd_aiter` is the canonical AMD-internal ROCm/PyTorch image dist; `aiter` is the upstream name), `commit` (parsed from the setuptools_scm `+g<sha>` local-version segment, matches the image tag's `aiter-<sha>` label), `hsa_tree` (issue #176 -- per-arch fingerprint of pre-built `.co` kernel binaries: file_count, co_count, deterministic combined_sha256). Most installs record `null` for everything; absence is silent. |
| `aotriton` | `dict` | scan of `<torch>/lib/libaotriton_v2.so*` filenames + `sha256` of the resolved file + presence of `<torch>/lib/aotriton.images/` + `$AOTRITON_INSTALLED_PREFIX` | Default ROCm Flash Attention backend. Bundled in the wheel via `cmake/External/aotriton.cmake` (NOT a `third_party/` git submodule). Fields: `bundled_present`, `bundled_version`, `bundled_lib_hash`, `bundled_images_dir_present`, `installed_prefix`. CK is the alternative backend (toggled via `TORCH_ROCM_FA_PREFER_CK=1`). |
| `runtime_context` | `dict` | `/.dockerenv`, `/run/.containerenv`, `$SINGULARITY_NAME`, `/proc/1/cgroup`, `sys.prefix`, `$CONDA_DEFAULT_ENV` | `type`, `python_env`, `venv_path`, `conda_env_name` |
| `docker` | `dict \| null` | `$AORTA_DOCKER_IMAGE` / `$AORTA_DOCKER_DIGEST` env vars + `/proc/self/cgroup` | `null` on baremetal; image+digest provided by the launcher (the only reliable way from inside a container) |
| `env_vars` | `dict[str, str \| null]` | explicit canonical list (currently 31 names; see `CANONICAL_ENV_VARS` in `environment.py` for the live set) | GPU scoping + HSA / runtime + GPU queue / codegen + NCCL/RCCL + FBGEMM + MIOpen + SDPA backend selection + GEMM backend preference + hipBLASLt autotune + PyTorch / inductor. Build-time cmake flags (`USE_ROCM_CK_SDPA`, `USE_ROCM_CK_GEMM`, `USE_FBGEMM*`) are NOT in this list -- they're surfaced under their respective library blocks instead, parsed from `torch.__config__.show()`. |
| `python_version` | `str` | `platform.python_version()` | always populated |
| `pytorch_version` | `str \| null` | optional `import torch` (no CUDA/HIP context init) | `null` when torch absent |
| `pytorch_build` | `dict` | `torch.version.{git_version,hip,cuda,debug}` + install-kind detection + optional `git -C <src>/third_party/<sub> rev-parse HEAD` + parse of `torch.__config__.show()` + `nm -D libtorch_hip.so \| c++filt` symbol grep + scan of `<torch>/lib/` + parse of `<source>/build/CMakeCache.txt` + stream of `<source>/build/build.ninja` (modern `enable_language(HIP)` path) or walk of `<source>/build/**/<target>.dir/**/*.hip.o.cmake` (legacy `FindHIP.cmake` fallback) | `git_commit` is the linchpin -- pins every vendored submodule deterministically. Sub-blocks: `flags` (raw `build_settings`, `cxx_defines`, `cxx_flags_raw`, `cuda_flags_raw`, `gpu_arch_list`), `build_flags` (issue #170 stable 17-key parsed bool/str/None subset, with `CAFFE2_USE_MIOPEN` aliased to `USE_MIOPEN`), `binary_introspection` (`libtorch_hip_symbol_counts`, `torch_lib_bundled`, `cxx_flags_use_defines` -- pure facts, no ON/OFF inference), `cmake_cache` (source/editable installs only -- allowlisted entries from CMakeCache.txt), `ninja_hipcc` (source/editable installs only -- per-target HIPCC defines + codegen flags + offload archs; `_parser` discriminates the two parser strategies). See "PyTorch source-tree submodule probing" below. |
| `build_system` | `dict` | `buck2 --version` + `buck2 root` + `hg id -i` / `git rev-parse HEAD` | Always present. `{"kind": "buck2", "buck2_version": str, "repo_root": str, "revision": str \| null}` when buck2 is on PATH AND we are demonstrably inside a Buck checkout (both `buck2 --version` and `buck2 root` succeed); `buck2_version` and `repo_root` are guaranteed populated, only `revision` may be `null`. `{"kind": "none"}` in every other case, including the dominant "buck2 is installed but cwd is not inside a Buck checkout" scenario. Added in schema 1.3 for issue #163 (A1.2a) so consumers can branch on Buck2 vs. system-package environments. See "Running inside a Buck environment" below. |
| `library_introspection` | `list[dict]` | `buck2 audit dependencies <target> --transitive --json` (only when `--buck-target` is supplied) | Always present. Empty `[]` outside Buck mode. In Buck mode, one entry per matched library: `{"name", "source": "buck", "revision", "target"}`. The matched library set lives in `KNOWN_LIBRARY_PATTERNS` in `src/aorta/instrumentation/buck_introspect.py`. Added in schema 1.4 for issue #163 (A1.2b). |
| `library_introspection_alternates` | `list[dict]` | synthesised from A1's per-library blocks when a Buck match overlaps | Always present. Empty `[]` outside Buck mode and when no Buck-matched library is also captured by A1. Each entry mirrors the unified shape with `source: "package"` and pulls `revision` / `package_version` / `lib_hash` from the matching A1 block (e.g. `hipblaslt`). Added in schema 1.4 for issue #163 (A1.2b). |
| `pytorch_sdpa` | `dict` | `torch.backends.cuda.{flash,mem_efficient,math,cudnn}_sdp_enabled()` | `backends_enabled` dict, one bool per SDPA backend + per-getter `null` when missing on older torch. Runtime state, NOT compile-time -- combine with `pytorch_build.binary_introspection.libtorch_hip_symbol_counts` for the full "compiled in AND enabled" picture. Added in schema 1.5 for issue #176 (PR #177). |

`runtime_context.type` is one of `"docker" | "podman" | "singularity" | "baremetal"`. Adding values is a schema change.

`runtime_context.python_env` is one of `"venv" | "conda" | "system"`.

## Installing RDHC

RDHC (ROCm Deployment Health Check) is a system-level tool maintained by
the ROCm team. Aorta wraps it but does **not** vendor it -- if it is
absent the env probe still produces a complete snapshot, just with
`system_health: null` and a `partial_reasons` entry pointing here.

### Why it isn't a Python `requirements.txt` dependency

`rdhc` is **not a PyPI package** -- it ships as part of the
[`rocm-systems`](https://github.com/ROCm/rocm-systems/tree/main/projects/rocm-core/rdhc)
repository, installed alongside the ROCm platform via system package
managers. Even if it were available on PyPI, hard-pinning it would
break aorta on:

* non-ROCm hosts (NVIDIA, CPU-only CI runners)
* ROCm docker images that strip `rocm-systems` to keep the image small
* Apple silicon laptops used for analysis-only work

The fail-soft contract (`partial=True` + a clear reason) is therefore
the design, not a workaround. This page is the canonical install path
for operators who want full `system_health` coverage.

### Install on Ubuntu / Debian

If you already have the ROCm apt repo configured (the standard install
path documented at <https://rocm.docs.amd.com/projects/install-on-linux>):

```bash
sudo apt install rocm-core rocm-systems
which rdhc        # /opt/rocm/bin/rdhc or /usr/bin/rdhc
sudo -n rdhc --quick --json /tmp/rdhc.json && jq '.rdhc_version' /tmp/rdhc.json
```

### Install on RHEL / CentOS / Rocky / SLES

```bash
sudo dnf install rocm-core rocm-systems    # or `zypper` on SLES
which rdhc
```

### Install from source (any distro, including stripped containers)

```bash
git clone https://github.com/ROCm/rocm-systems
cd rocm-systems/projects/rocm-core/rdhc
# Follow the project's README for build + install. Typically:
sudo make install
```

### Install rdhc's Python deps into the **system** Python

`rdhc` is a Python script (`#!/usr/bin/env python3`) that imports
`prettytable` and `PyYAML`. The probe runs it via
`sudo -n -E rdhc --quick --json <tmp>`. Under sudo, `secure_path` in
`/etc/sudoers` overrides the calling shell's `PATH` (and `-E` does NOT
override `secure_path` for the `PATH` variable specifically), so
`#!/usr/bin/env python3` resolves to **system** `python3`
(`/usr/bin/python3`), NOT whatever venv or conda env aorta itself runs
in. Installing `prettytable` into your venv has zero effect on the
subprocess that rdhc actually runs in.

Install rdhc's deps into the system Python where they'll be visible:

```bash
# rdhc ships its own requirements.txt with the apt/dnf package
sudo pip3 install -r /opt/rocm/share/rdhc/requirements.txt

# Verify rdhc can now import everything it needs
/opt/rocm/bin/rdhc --quick --json /tmp/rdhc_check.json && echo OK
```

If `pip3 install` is blocked by PEP 668 ("externally-managed
environment"), you have two options that keep rdhc on system python:
either pass `--break-system-packages` (acceptable for these two
packages -- `prettytable` and `PyYAML` are well-behaved) or install
distro packages where versions are recent enough (`apt install
python3-prettytable python3-yaml`; check `prettytable>=3.14.0` -- on
Ubuntu 22.04 the apt version is too old, use pip).

### Ubuntu 24.04 + venv: known sudo `-E` PATH gotcha

On Ubuntu 24.04, the default sudoers `secure_path` is even stricter
about `PATH` preservation than on 22.04, and `sudo -E` does NOT
preserve the venv's `PATH` even when `aorta env probe` is invoked from
inside one. Symptom: `system_health: rdhc exited 1` with stderr like
`prettytable not installed`, even though both rdhc and prettytable
appear available in the calling shell.

Two fixes:

1. **Install the deps into system Python (recommended)** -- per the
   section above. Once they're in `/usr/lib/python3/...`, sudo's PATH
   reset doesn't matter.
2. Replace `-E` with `--preserve-env=PATH` if you want the venv's
   `python3` to be the one rdhc runs in. Aorta does not do this
   automatically yet (planned follow-up); to override, wrap rdhc in a
   small script and point the sudoers rule at the wrapper.

### Configure passwordless sudo for `rdhc`

`aorta env probe` runs `sudo -n -E rdhc --quick --json <tmp>`. The `-n`
flag means **never prompt** -- if sudo would require a password, the
probe records `system_health: rdhc exited 1 (no stderr; likely sudo-n
unavailable)` and continues with `partial=True`. Two ways to fix:

1. **Recommended (per-tool sudoers entry).** Drop a file in
   `/etc/sudoers.d/` so only `rdhc` is passwordless, not all of sudo:

   ```bash
   sudo visudo -f /etc/sudoers.d/aorta-rdhc
   ```

   Contents:

   ```text
   # Allow members of the `aorta-users` group to run rdhc without a
   # password. Adjust the group / user and the binary path to match
   # your install.
   %aorta-users ALL=(ALL) NOPASSWD: /opt/rocm/bin/rdhc, /opt/rocm/bin/rdhc.py, /usr/bin/rdhc, /usr/bin/rdhc.py
   ```

   Then `sudo usermod -aG aorta-users $USER` (and re-login). Verify:

   ```bash
   sudo -n -E rdhc --quick --json /tmp/check.json && echo OK
   ```

2. **Inside docker images you build yourself.** Add `rdhc` to the
   image and configure passwordless sudo as part of the build:

   ```dockerfile
   RUN apt-get update && apt-get install -y rocm-core rocm-systems sudo \
    && echo "%aorta-users ALL=(ALL) NOPASSWD: /opt/rocm/bin/rdhc, /usr/bin/rdhc" \
       > /etc/sudoers.d/aorta-rdhc \
    && groupadd aorta-users
   ```

### Verify the env probe picks it up

```bash
aorta env probe -o /tmp/env.json
jq '.system_health.rdhc_version' /tmp/env.json   # expect: a version string, NOT null
jq '.partial_reasons[]' /tmp/env.json | grep -i rdhc   # expect: nothing
```

If `partial_reasons` still contains an `rdhc` entry, the message is the
ground truth -- it tells you exactly what failed (PATH, sudo-n,
timeout, malformed JSON). The `(see docs/env-probe.md#installing-rdhc)`
hint at the end of those reasons points back at this page.

## Schema changelog

Mirrors the in-code comment at `SCHEMA_VERSION` in
`src/aorta/instrumentation/environment.py`. Recorded here so consumers
tracking schema evolution don't have to read source.

### `1.5` (current)

Adds source/editable-install build introspection plus a runtime SDPA
backend probe (PR #177, issue #176). This entry collapses what was
originally two separate version bumps on the PR #177 branch (1.2 ->
1.3 source-introspection plus 1.3 -> 1.4 legacy-FindHIP fallback)
because main shipped its own 1.3 (build_system) and 1.4
(library_introspection) in parallel via PR #164 / PR #165. All five
additive surfaces below are net-new on top of main's 1.4.

* `pytorch_build.cmake_cache` -- parsed
  `<source>/build/CMakeCache.txt` for source / editable installs.
  `entries` is a sorted dict of `<NAME>: {type, value}` filtered by
  an allowlist of name prefixes (`USE_`, `CK_`, `AITER_`, `FLASH_`,
  `HIPBLAS`, `DISABLE_`, `AOTRITON`, `ROCM_`, `HIP_PLATFORM`,
  `HIP_RUNTIME`, `HIP_COMPILER`, `HIP_VERSION`, `PYTORCH_ROCM_ARCH`,
  `TORCH_BUILD_VERSION`, `BUILD_TYPE`, `CMAKE_BUILD_TYPE`). Wheel
  installs render `entries: null` and `_source_file: null` --
  absence is the documented common case, no partial reason.
* `pytorch_build.ninja_hipcc` -- per-target HIPCC defines + codegen
  flags + offload archs. Two parser strategies in one block,
  discriminated by `_parser`:
    * `"ninja_defines"` -- streamed parse of
      `<source>/build/build.ninja` for the modern
      `enable_language(HIP)` build shape. Identifies targets via the
      `-D<target>_EXPORTS` token cmake appends per shared-lib
      target. Streamed line-by-line (build.ninja can be 350+ MB on a
      fully-built tree).
    * `"legacy_findhip_per_source"` -- fallback walk of
      `<source>/build/**/<target>.dir/**/*.hip.o.cmake` driver
      scripts when the ninja-only scan returns `targets: {}` (common
      on ROCm/PyTorch Jenkins images that still use the legacy
      `FindHIP.cmake` flow). Parses `set(HIP_HIPCC_FLAGS …)` /
      `set(HIP_CLANG_FLAGS …)` cmake-list values;
      `_legacy_scripts_scanned` reports the read count.
  Both parsers report the same per-target shape (`defines`,
  `use_defines_present`, `codegen_flags_present`, `offload_archs`).
  Targets reported: `torch_hip`, `torch_cpu`, `c10_hip`, `ck_sdpa`
  (CK-backed SDPA backend; owns `USE_ROCM_CK_SDPA`,
  `CK_TILE_FMHA_*`, `FLASHATTENTION_DISABLE_*` -- statically linked
  into `libtorch_hip.so` so its flags don't appear in the wheel's
  host-side `CXX_FLAGS`), `mslk` (Multi-Stream Layer Kernels).
  Wheel installs render `targets: null`.
* `aiter.hsa_tree` -- per-arch fingerprint of aiter's pre-built HSA
  `.co` kernel binaries. Per arch: `file_count`, `co_count`,
  deterministic `combined_sha256` over sorted `(relpath, sha256)`
  pairs. Three search roots:
  `importlib.util.find_spec("aiter_meta")`, the sibling
  `aiter_meta` dir, and `$AORTA_PYTORCH_SRC/third_party/aiter/hsa`.
  Returns `null` when no tree is locatable -- silent absence (most
  installs lack it).
* `pytorch_sdpa` (new top-level) -- `backends_enabled:
  {flash_sdp_enabled, mem_efficient_sdp_enabled, math_sdp_enabled,
  cudnn_sdp_enabled}`. Pure Python attribute lookups on
  `torch.backends.cuda` -- no GPU work, no HIP context init. Per-
  getter `null` when the function is missing on older torch
  (distinguishable from True/False).

Backwards-compat notes:

* `pytorch_sdpa` is a new top-level dataclass field with a default
  factory, so a 1.4 reader running `EnvSnapshot.from_dict()` on a
  1.5 snapshot silently drops the unknown key, and a 1.5 reader
  loading a pre-1.5 snapshot gets the dataclass-default
  `backends_enabled` (all-None) instead of a missing field.
* Every new nested key (`pytorch_build.cmake_cache`,
  `pytorch_build.ninja_hipcc` plus its `_parser` /
  `_legacy_scripts_scanned` keys, `aiter.hsa_tree`) lives under
  existing top-level dicts. 1.4 consumers indexing these directly
  on a 1.4 snapshot get `KeyError`, not `None` -- use `.get(key)`
  or guard on `schema_version`.
* The `ninja_hipcc.targets` extension (`ck_sdpa` / `mslk`) is a
  silent addition. Consumers iterating `targets` see additional
  rows on 1.5 snapshots but the iteration shape is unchanged.
  Hard-coding the pre-1.5 three-name list still works; you just
  don't see the new SDPA-relevant data.

### `1.4`

Additive change (issue #163, A1.2b):

* New top-level `library_introspection` (list[dict], default `[]`)
  and `library_introspection_alternates` (list[dict], default `[]`).
  Both always present. Populated only when `collect_env(buck_target=...)`
  is invoked (or `aorta env probe --buck-target ...`); each transitive
  dep label of the Buck target that matches one of the patterns in
  `KNOWN_LIBRARY_PATTERNS` (see `src/aorta/instrumentation/buck_introspect.py`)
  yields one entry of shape `{"name", "source": "buck", "revision",
  "target"}`. When a library matched via Buck is also captured by
  A1's existing per-library blocks (`hipblaslt`, `rocblas`, `miopen`,
  `rccl`), the A1-side identifiers are synthesised into a parallel
  entry placed in `library_introspection_alternates` so consumers can
  compare the buck label vs. the system-package version/lib_hash.
  Outside Buck mode both lists stay empty; A1's existing per-library
  top-level blocks remain authoritative.

Schema 1.4 is forward- and backward-compatible: an env.json produced
by 1.4 contains every 1.3 key unchanged. Loading a 1.1 / 1.2 / 1.3
env.json into the 1.4 `EnvSnapshot.from_dict()` succeeds because all
new fields default to empty (lists or `{"kind": "none"}` per their
type).

### `1.3`

Additive change (issue #163, A1.2a):

* New top-level `build_system` block, always present. Captures Buck2
  presence + `buck2 --version` + `buck2 root` + source revision (`hg
  id -i`, falling back to `git rev-parse HEAD`). The detector is
  strict about what counts as `kind=buck2`: both `buck2 --version`
  AND `buck2 root` must succeed, so the buck2 shape's `buck2_version`
  and `repo_root` are guaranteed populated (only `revision` may be
  `null`). Every other case -- buck2 absent, buck2 broken, or "buck2
  is installed but cwd is not inside a Buck checkout" (where `buck2
  root` exits non-zero) -- collapses to `{"kind": "none"}`. The bump
  from 1.2 -> 1.3 isn't strictly required for an additive field, but
  it makes the new key easy to detect in consumer pipelines.

Schema 1.3 is forward- AND backward-compatible:

* An env.json produced by 1.3 contains every 1.2 key unchanged.
* Loading a 1.1 / 1.2 env.json (no `build_system` key) into the 1.3
  `EnvSnapshot.from_dict()` succeeds: the missing field is back-filled
  with `{"kind": "none"}` (the only honest default -- the producer
  did not run the build_system probe at all). This mirrors the
  existing tolerance for missing `partial_reasons`.

### `1.2`

Top-level-key-additive -- every new field lives under existing
top-level dicts (`pytorch_build`, `aiter`), so 1.1 readers loading
a 1.2 snapshot via `EnvSnapshot.from_dict(...)` do NOT raise and
existing top-level access still works. Note however that the new
**nested** keys (`pytorch_build.flags`, `pytorch_build.build_flags`,
`pytorch_build.binary_introspection`, `aiter.package_dist_name`,
`aiter.commit`) are NOT backfilled on 1.1 snapshots -- a consumer
indexing them on a 1.1 snapshot gets a `KeyError`, not `None`. Use
`.get(key)` or guard on `schema_version` if you read these from
historical snapshots.

* `pytorch_build.flags` -- structured raw introspection from
  `torch.__config__.show()`: `build_settings` (KEY=VALUE dict from the
  `Build settings:` block), `cxx_defines` (`-D<NAME>[=<value>]` tokens
  parsed out of `CXX_FLAGS`), verbatim `cxx_flags_raw` /
  `cuda_flags_raw`, and `gpu_arch_list` from
  `torch.cuda.get_arch_list()`.
* `pytorch_build.build_flags` (issue #170) -- stable 17-key parsed
  subset projected from `flags`. Boolean-like values (ON/OFF/TRUE/
  FALSE/1/0, any case) become bools; non-boolean values like
  `BUILD_TYPE=Release` stay strings; flags absent from
  `__config__.show()` render as `null`. `CAFFE2_USE_MIOPEN` is treated
  as an alias for `USE_MIOPEN` (the Caffe2-era spelling some ROCm
  builds still emit). The brief gains a compact one-liner:
  `flags: FLASH_ATTN=on CK_SDPA=on AOTRITON=on MEM_EFF=on`.
* `pytorch_build.binary_introspection` -- `nm | c++filt` substring
  counts on `libtorch_hip.so` (`pytorch_flash::`, `ck_tile::FmhaFwd`,
  `aotriton::`, `aiter::`, ...), bundled-lib presence in
  `<torch>/lib/` (`libaotriton_v2.so`), and presence of specific
  `-DUSE_*` defines in `CXX_FLAGS`. Pure facts -- a non-zero count
  proves a code path is compiled in, but a zero count does NOT prove
  the cmake option was OFF (linker stripping). The CK pytorch-bundled
  probe and binary_introspection share one `nm` dump per
  `collect_env()` call via `_HipSymbolDumpCache`.
* `aiter` -- gained `package_dist_name` (which PyPI dist matched:
  `amd_aiter` is the AMD-internal ROCm/PyTorch image dist; `aiter` is
  the upstream name) and `commit` (parsed from the setuptools_scm
  `+g<sha>` local-version segment, matches the image tag's
  `aiter-<sha>` label).

### `1.1`

Renames (non-additive -- bumped from `1.0`):

* `hipblaslt.commit` -> `hipblaslt.rocm_release_tweak`
* `rocblas.commit` -> `rocblas.rocm_release_tweak`
* `miopen.commit` -> `miopen.rocm_release_tweak`
  (the value is the ROCm-release-shared identifier, not a per-library
  upstream commit; the old name misled consumers)
* `hipblaslt.tensile_yaml_revision` -> `hipblaslt.kernel_db_revision`
* `rocblas.tensile_yaml_revision` -> `rocblas.kernel_db_revision`
  (matches `miopen.kernel_db_revision`; modern hipBLASLt/rocBLAS ship
  `.dat` not `.yaml`, so the old name was inaccurate too)

`env_vars` removals (env_vars is an explicit allowlist; removing is a
breaking change):

* `USE_ROCM_CK_SDPA` and `USE_ROCM_CK_GEMM` -- these are build-time
  cmake flags, NOT runtime env vars. Setting them in the workload's
  environment does nothing. Replaced with
  `composable_kernel.pytorch_use_ck_sdpa` and `pytorch_use_ck_gemm`
  booleans, parsed from `torch.__config__.show()`.

Additive changes (no bump strictly required, recorded here for
visibility):

* New top-level blocks: `rocblas`, `composable_kernel`, `tensile`,
  `triton`, `fbgemm`, `aiter`, `aotriton`, `miopen`, `rccl`,
  `gpu_arch`, `host`, `pytorch_build`.
* `env_vars` gained 22 entries (was 13 in 1.0; now 31): GPU scoping
  (`HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
  `HSA_OVERRIDE_GFX_VERSION`), launch (`HIP_LAUNCH_BLOCKING`),
  PyTorch ROCm arch + inductor (`PYTORCH_ROCM_ARCH`), MIOpen
  (`MIOPEN_SYSTEM_DB_PATH`, `MIOPEN_USER_DB_PATH`,
  `MIOPEN_DEBUG_DISABLE_FIND_DB`, `MIOPEN_FIND_MODE`), SDPA backend
  (`TORCH_ROCM_FA_PREFER_CK`,
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`), GEMM backend +
  autotune (`TORCH_BLAS_PREFER_HIPBLASLT`,
  `TORCH_HIPBLASLT_TUNING_FILE`,
  `TORCH_HIPBLASLT_TUNING_OVERRIDE_FILE`), and NCCL/RCCL
  (`NCCL_P2P_LEVEL`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`,
  `RCCL_MSCCL_ENABLE`).
* `host.glibc_version` strips the redundant `"glibc "` prefix from
  the value (the field name carries the unit). On 1.0 this was
  `"glibc 2.35"`; on 1.1 it's `"2.35"`.

### `1.0` (initial release)

Original probe blocks: `system_health`, `rocm`, `hip`, `hipblaslt`,
`runtime_context`, `docker`, `env_vars`, `python_version`,
`pytorch_version`. 13 canonical env vars. See git history for the
original A1 PR (#152).

## Field-naming notes

### `*.rocm_release_tweak` is a release identifier, not a per-library commit

The `hipblaslt`, `rocblas`, and `miopen` blocks each have a
`rocm_release_tweak` field parsed from `<LIB>_VERSION_TWEAK` defines
in their respective headers. **Despite the name suggesting "git
tweak", AMD sets these macros to the ROCm release identifier** -- so
in any given ROCm release, `hipblaslt.rocm_release_tweak ==
rocblas.rocm_release_tweak == miopen.rocm_release_tweak`. It is NOT a
per-library upstream commit SHA.

For per-library binary-level drift detection, use:

* `<lib>.lib_hash` -- changes any time the binary changes, even
  within the same release tweak (catches local rebuilds, debug
  variants, cherry-picked patches).
* `<lib>.kernel_db_revision` (hipblaslt/rocblas) /
  `miopen.kernel_db_revision` -- catches kernel-DB drift independent
  of the lib binary.
* `<lib>.applied_prs` -- a forward-compat slot for explicit PR
  detectors when a specific patch warrants tracking.

The `composable_kernel.system.commit` field IS a real upstream commit
(40-char SHA), because CK ships its own `CK_COMMIT_ID` define populated
from the upstream submodule.

### `hip.version` vs `pytorch_build.hip_version` are deliberately both captured

* `hip.version` -- the **system-installed** HIP version (from
  `hipconfig --version`). What ROCm is installed on the host.
* `pytorch_build.hip_version` -- the **compile-time** HIP version
  PyTorch was built against (from `torch.version.hip`). What the wheel
  expects.

These will be equal on a host where torch was built against the
installed HIP. They diverge -- and reveal a real bug class -- when
someone runs a wheel built against HIP 7.1 on a host with HIP 7.2
installed (the wheel may load but dispatch to mismatched API surfaces).
Capturing both is intentional.

## PyTorch source-tree submodule probing

`pytorch_build.git_commit` (always available on any installed PyTorch)
is the lookup key that pins every vendored `third_party/` submodule
deterministically -- including AMD-relevant ones like `composable_kernel`,
`aiter`, and `fbgemm`. The probe captures it from
`torch.version.git_version`.

To go further and capture the actual bound submodule SHAs in-process,
point the probe at a PyTorch source tree:

```bash
# Explicit (recommended): tell aorta where the checkout lives
AORTA_PYTORCH_SRC=/path/to/pytorch aorta env probe -o env.json
```

When set, the probe runs `git -C $AORTA_PYTORCH_SRC/third_party/<sub>
rev-parse HEAD` for each entry in `CANONICAL_PYTORCH_SUBMODULES` and
records the SHA in `pytorch_build.submodule_commits.<sub>`.
`pytorch_build.submodule_commits._source = "git"` records the
provenance.

For pip-installed wheels (the common case), the probe falls back
gracefully: every `submodule_commits.<sub>` is `null`, and a single
`partial_reasons` line emits a copy-pasteable URL template:

```
pytorch_build.submodule_commits: wheel install -- direct SHAs not
recoverable; resolve via
github.com/pytorch/pytorch/tree/<git_commit>/third_party/<name>
(set AORTA_PYTORCH_SRC=<src> to enable in-process probing)
```

`<git_commit>` is the captured `pytorch_build.git_commit` (substituted
in if known). The operator can paste the URL into a browser to see the
exact bound commit for any submodule on the GitHub tree.

**Auto-detection** also covers two cases without `AORTA_PYTORCH_SRC`:

* **Editable installs** (`pip install -e /path/to/pytorch`): detected
  via the `direct_url.json` marker in `torch-<version>.dist-info/`.
  `install_kind = "editable"`.
* **Source-shadowed imports** (running `python -c "import torch"` from
  inside a checkout where the local `torch/` dir wins over the wheel):
  detected by walking up from `torch.__file__` for a sibling `.git` +
  `third_party/`. `install_kind = "source"`.

Adding a new submodule to track is a deliberate three-step change
(mirrors `CANONICAL_ENV_VARS`):

1. Add to `CANONICAL_PYTORCH_SUBMODULES` in
   `src/aorta/instrumentation/environment.py`.
2. Update `TestPytorchBuildBlockShape::test_canonical_submodules_constant_is_stable`
   AND `test_submodule_commits_keys_stable` in the test file.
3. Justify in your PR -- which submodule, why it materially affects
   trial results.

## Fail-soft contract

`collect_env()` never raises. When something can't be captured:

1. The corresponding field is set to `None`.
2. A short reason like `"system_health: rdhc not on PATH"` or
   `"hipblaslt.commit: /opt/rocm/include/hipblaslt/hipblaslt-version.h
   missing, empty, or unreadable"` is appended to `partial_reasons`.
3. `partial` is set to `True`.
4. The run continues.

This is the difference between "I ran the matrix, env probe failed,
now I have nothing" and "I ran the matrix, env probe is honest about
what it couldn't capture, here are the artifacts."

Documented absences DO NOT trigger `partial`:

* `docker == None` on baremetal (no container, nothing to record).
* `env_vars[X] == None` when the variable is unset (the documented
  contract -- the consumer can tell unset apart from `""`).
* `runtime_context.venv_path == None` when not inside a venv.

## Performance

| Environment | Wall time |
| --- | --- |
| Baremetal host (no rdhc), warm cache | ~2.2 s |
| Inside a docker image | ~3 s |

Most of the time goes to the bundled-CK probe, which runs `nm -D
--defined-only` over `libtorch_hip.so` (~400 MB on a typical ROCm wheel)
and pipes the output through `c++filt`. Cold-cache nm over a
several-hundred-MB binary can stretch to several seconds; the per-tool
budget for nm/c++filt is `NM_TIMEOUT_SEC = 30 s` (vs `SHORT_TIMEOUT_SEC
= 5 s` for the smaller subprocesses) so the probe stays inside the
< 15 s overall target on contended I/O.

Without the bundled-CK probe (e.g. when binutils is stripped from the
container, or for a CPU-only PyTorch wheel), the probe completes in
under 0.5 s.

No GPU compute. `import torch` will `dlopen` the HIP runtime libraries
(so `pmap` shows them in the process), but verified via `rocprofv3
--hip-trace`: zero HIP API calls and zero kernel dispatches.

## Container detection precedence

Container `type` is resolved in this order, first match wins:

1. `/.dockerenv` exists -> `"docker"`.
2. `/run/.containerenv` exists -> `"podman"`.
3. `$SINGULARITY_NAME` is set OR cgroup contains `singularity` ->
   `"singularity"`.
4. Cgroup fallback: `/proc/1/cgroup` contains `singularity` ->
   `"singularity"`; then `docker` -> `"docker"`; then `podman` ->
   `"podman"`.
5. Otherwise -> `"baremetal"`.

Singularity wins over docker/podman in the cgroup fallback so a
Singularity instance whose underlying cgroup happens to be docker-shim
shaped is not misclassified.

## Where the data comes from

If you need to verify a value or diagnose a missing one, the per-field
sources are:

| Field | Read from |
| --- | --- |
| `rocm.version` | `/opt/rocm/.info/version` |
| `rocm.version_dev` | `/opt/rocm/.info/version-dev` (often empty on developer builds) |
| `rocm.kmd_version` | `/sys/module/amdgpu/version` (kernel module sysfs) |
| `hip.*` | `hipconfig --version` / `--platform` / `--compiler` / `--runtime` / `--cpp_config` |
| `hipblaslt.rocm_release_tweak` | `HIPBLASLT_VERSION_TWEAK` define in `/opt/rocm/include/hipblaslt/hipblaslt-version.h` |
| `hipblaslt.package_version` | `HIPBLASLT_VERSION_{MAJOR,MINOR,PATCH}` defines in the same header |
| `hipblaslt.lib_hash` | `sha256(/opt/rocm/lib/libhipblaslt.so)` resolved through symlinks |
| `hipblaslt.kernel_db_revision` | sha256 of sorted filenames of `*.yaml`/`*.dat`/`*.co` under `/opt/rocm/lib/hipblaslt/library/` |
| `rocblas.rocm_release_tweak` | `ROCBLAS_VERSION_TWEAK` define in `/opt/rocm/include/rocblas/internal/rocblas-version.h` |
| `rocblas.package_version` | `ROCBLAS_VERSION_{MAJOR,MINOR,PATCH}` defines in the same header |
| `rocblas.lib_hash` | `sha256(/opt/rocm/lib/librocblas.so)` resolved through symlinks |
| `rocblas.kernel_db_revision` | sha256 of sorted filenames of `*.yaml`/`*.dat`/`*.co` under `/opt/rocm/lib/rocblas/library/` |
| `miopen.rocm_release_tweak` | `MIOPEN_VERSION_TWEAK` define in `/opt/rocm/include/miopen/version.h` |
| `miopen.package_version` | `MIOPEN_VERSION_{MAJOR,MINOR,PATCH}` defines in the same header |
| `miopen.lib_hash` | `sha256(/opt/rocm/lib/libMIOpen.so)` resolved through symlinks |
| `miopen.kernel_db_revision` | sha256 of sorted filenames of `*.txt` under `/opt/rocm/share/miopen/db/` (changes when conv-kernel set changes; the `MIOPEN_SYSTEM_DB_PATH` env var overrides this directory at runtime) |
| `rccl.version_code` / `rccl.version` | `NCCL_VERSION_CODE` define in `/opt/rocm/include/rccl/rccl.h`, decoded into `MAJOR.MINOR.PATCH` |
| `rccl.lib_hash` | `sha256(/opt/rocm/lib/librccl.so)` resolved through symlinks |
| `gpu_arch.*` | `rocm_agent_enumerator` subprocess (one gfx-target per detected GPU on stdout); `gfx000` placeholder filtered out |
| `host.kernel_release` / `kernel_version` / `machine` | `os.uname()` |
| `host.glibc_version` | `os.confstr("CS_GNU_LIBC_VERSION")` with the redundant `"glibc "` prefix stripped (so the value is the bare version string, e.g. `"2.35"`); returns `null` on non-glibc systems like musl / macOS |
| `composable_kernel.pytorch_use_ck_sdpa` / `.pytorch_use_ck_gemm` | substring search in `torch.__config__.show()` for `-DUSE_ROCM_CK_SDPA` / `-DUSE_ROCM_CK_GEMM`. Build-time flags baked into the PyTorch wheel; setting these as runtime env vars does NOT change behavior. `False` is meaningful (built without the CK SDPA/GEMM path -- dispatches to AOTriton / non-CK rocBLAS instead). |
| `composable_kernel.system.version` | `CK_VERSION_{MAJOR,MINOR,PATCH}` defines in `/opt/rocm/include/ck/version.h` |
| `composable_kernel.system.commit` | `CK_COMMIT_ID` define in the same header (full 40-char SHA) |
| `composable_kernel.system.ck_tile_present` | existence of `/opt/rocm/include/ck_tile/core/config.hpp` |
| `composable_kernel.pytorch_bundled.symbol_count` | `nm -D --defined-only` of `<torch>/lib/libtorch_hip.so` piped through `c++filt`, counting lines containing `ck::`. `null` when torch absent, lib missing, or binutils stripped from the container. |
| `tensile.package_version` | `import Tensile; Tensile.__version__` (rare; build-time tool) |
| `tensile.kernel_db_combined_hash` | sorted-filenames sha256 over the union of the hipBLASLt + rocBLAS kernel DBs (each filename namespaced by parent dir basename) |
| `triton.package_version` | `import triton; triton.__version__` (ROCm fork puts source commit into the version string) |
| `fbgemm.package_version` | `import fbgemm_gpu; fbgemm_gpu.__version__` (commonly `null` -- FBGEMM is vendored inside PyTorch) |
| `fbgemm.pytorch_use_fbgemm` / `fbgemm.pytorch_use_fbgemm_genai` | substring search in `torch.__config__.show()` for the `-DUSE_FBGEMM` and `-DUSE_FBGEMM_GENAI` defines. `False` is meaningful (built-without); `null` only when torch is absent. |
| `aiter.package_version` | `import aiter; aiter.__version__` |
| `aotriton.bundled_version` | parsed from `<torch>/lib/libaotriton_v2.so.MAJOR.MINOR.PATCH` filename (highest-versioned wins) |
| `aotriton.bundled_lib_hash` | `sha256(<torch>/lib/libaotriton_v2.so*)` resolved through symlinks |
| `aotriton.bundled_images_dir_present` | existence of `<torch>/lib/aotriton.images/` |
| `aotriton.installed_prefix` | value of `$AOTRITON_INSTALLED_PREFIX` (operator override pointing PyTorch at a system AOTriton install; `null` for the default bundled-wins case) |
| `pytorch_build.git_commit` | `torch.version.git_version` (always available on installed torch) |
| `pytorch_build.hip_version` / `.cuda_version` / `.debug` | `torch.version.{hip,cuda,debug}` |
| `pytorch_build.install_kind` | `"wheel"` (default), `"editable"` (PEP 660 `direct_url.json` + `dir_info.editable=True`), `"source"` (`AORTA_PYTORCH_SRC` env var or walked-up `.git`+`third_party/`), `"unknown"` (torch import failed) |
| `pytorch_build.source_path` | The detected/configured PyTorch source root; `null` for wheel installs |
| `pytorch_build.submodule_commits.{composable_kernel,aiter,fbgemm}` | `git -C <source>/third_party/<name> rev-parse HEAD` when a source tree is detected; otherwise `null` and a `partial_reasons` line emits the GitHub URL template for manual lookup |
| `pytorch_build.submodule_commits._source` | Provenance tag: `"git"` when populated from a source tree, `null` for wheel installs |
| `pytorch_build.ninja_hipcc._source_file` | Path to `<source>/build/build.ninja` when at least one parser ran; `null` for wheel installs / build.ninja absent |
| `pytorch_build.ninja_hipcc._parser` | `"ninja_defines"` when modern `enable_language(HIP)` ninja rules matched; `"legacy_findhip_per_source"` when the per-source `*.hip.o.cmake` fallback ran; `null` on wheel installs or when no parser succeeded |
| `pytorch_build.ninja_hipcc._legacy_scripts_scanned` | int count of `*.hip.o.cmake` driver scripts read by the legacy-FindHIP fallback; `null` on the modern path or wheel installs |
| `pytorch_build.ninja_hipcc.targets.{torch_hip,torch_cpu,c10_hip}` | Modern path: streamed parse of `build.ninja` `DEFINES = …` / `FLAGS = …` lines under `HIP_COMPILER__*` rules, attributed by `-D<target>_EXPORTS`. Legacy path: walked `<source>/build/**/<target>.dir/**/*.hip.o.cmake` scripts, parsed `set(HIP_HIPCC_FLAGS …)` / `set(HIP_CLANG_FLAGS …)` cmake-list values |
| `pytorch_build.ninja_hipcc.targets.ck_sdpa` | Same parsing pipeline as the libtorch targets; surfaces the CK-backed SDPA backend's compile-time flags (`USE_ROCM_CK_SDPA`, `CK_TILE_FMHA_*`, `FLASHATTENTION_DISABLE_*`, `CK_USE_*`) that are otherwise invisible because ck_sdpa is statically linked into `libtorch_hip.so` and its flags don't propagate to the wheel's host-side `CXX_FLAGS` |
| `pytorch_build.ninja_hipcc.targets.mslk` | Same parsing pipeline; surfaces Multi-Stream Layer Kernels compile-time flags |
| `system_health` | `sudo -n -E rdhc --quick --json <tempfile>` |
| `docker.image` / `docker.digest` | `$AORTA_DOCKER_IMAGE` / `$AORTA_DOCKER_DIGEST` env vars set by the launcher |
| `docker.container_id` | `/proc/self/cgroup` |

Path constants live in
[`src/aorta/instrumentation/environment.py`](../src/aorta/instrumentation/environment.py)
under `# Filesystem locations`. Tests in
`tests/instrumentation/test_environment.py::TestPathConstants`
structurally enforce that all of them are absolute and that the set
is stable.

## How to add a new env var

See [the module README](../src/aorta/instrumentation/README.md#how-to-add-a-new-env-var-to-canonical_env_vars).
Short version: edit `CANONICAL_ENV_VARS`, update
`test_canonical_var_names_stable`, document why in your PR.

## Running inside a Buck environment

As of schema 1.3 (issue #163, A1.2a), the env probe detects whether
it's running inside a [Buck2](https://buck2.build/) build environment
and records the result in the top-level `build_system` block. As of
schema 1.4 (issue #163, A1.2b), it can additionally introspect a
target's transitive deps to populate the `library_introspection`
list. As of `aorta env recipe --format buck` (issue #163, A1.2c), the
captured `library_introspection` can be re-emitted as a best-effort
BUCK file fragment for cross-environment handoff.

Quick check:

```bash
# Outside any Buck repo
aorta env probe -o /tmp/env.json
jq '.build_system' /tmp/env.json
# -> {"kind": "none"}

# Inside a Buck2 checkout (e.g., the open-source examples at
#  https://github.com/facebook/buck2/tree/main/examples)
cd ~/buck2-examples/python/hello_world
aorta env probe -o /tmp/env.json
jq '.build_system' /tmp/env.json
# -> {"kind": "buck2", "buck2_version": "buck2 ...",
#     "repo_root": "/.../hello_world", "revision": "<sha>"}
```

The detector wraps `buck2 --version`, `buck2 root`, and a revision
lookup (`hg id -i` then `git rev-parse HEAD`); none of these
subprocesses are vendored. The `kind=buck2` shape is only emitted
when both `buck2 --version` and `buck2 root` succeed -- so the field
unambiguously means "we are demonstrably running inside a functional
Buck2 checkout", not merely "the `buck2` binary happens to be
installed somewhere on PATH". The dominant non-Buck case
(developer laptop with vendored buck2 but cwd outside any Buck repo)
therefore reports `{"kind": "none"}`, and the revision lookup never
runs against an unrelated working directory. If only the revision
lookup fails (no VCS at the Buck root), the dict is still populated
with `revision: null`.

### Buck-aware library introspection (`--buck-target`)

```bash
# Outside Buck mode the two introspection lists stay empty.
aorta env probe -o /tmp/env.json
jq '.library_introspection' /tmp/env.json              # -> []
jq '.library_introspection_alternates' /tmp/env.json   # -> []

# Inside a Buck checkout, point the probe at a top-level target. We
# wrap `buck2 audit dependencies <target> --transitive --json` and
# match each transitive dep label against KNOWN_LIBRARY_PATTERNS.
aorta env probe --buck-target //myproj:training_main -o /tmp/env.json
jq '.library_introspection' /tmp/env.json
# -> [
#   {"name":"hipblaslt","source":"buck","revision":"<repo-sha>","target":"//.../hipblaslt_lib"},
#   {"name":"rccl","source":"buck","revision":"<repo-sha>","target":"//.../rccl_lib"},
#   {"name":"pytorch","source":"buck","revision":"<repo-sha>","target":"//pytorch:torch"},
#   ...
# ]

# When a Buck-matched library is also captured by an A1 per-library
# block (hipblaslt, rocblas, miopen, rccl), an alternate entry is
# synthesised from the A1 block so consumers can compare the buck
# label against the system-package version/lib_hash:
jq '.library_introspection_alternates[] | select(.name=="hipblaslt")' /tmp/env.json
# -> {"name":"hipblaslt","source":"package","revision":"...","package_version":"...","lib_hash":"..."}
```

`--buck-timeout` (default 10 s) caps the audit subprocess. A timeout,
non-zero exit, or unparseable JSON degrades gracefully: both lists are
returned empty and the failure is recorded in `partial_reasons`. The
match patterns currently cover `hipblaslt`, `rccl`, `pytorch`, and
`rocm` runtime; new libraries are added by appending to
`KNOWN_LIBRARY_PATTERNS` in `src/aorta/instrumentation/buck_introspect.py`.

### Emitting a Buck recipe (`aorta env recipe --format buck`)

Issue #163 (A1.2c) ships a read-only emitter that turns an env.json
back into a BUCK file fragment for cross-environment handoff:

```bash
aorta env probe --buck-target //myproj:training_main -o /tmp/env.json
aorta env recipe --format buck /tmp/env.json > BUCK.aorta-recipe
head -n 20 BUCK.aorta-recipe
# # ============================================================================
# # AUTO-GENERATED BY `aorta env recipe --format buck` -- BEST-EFFORT, NOT EXACT.
# # env.json captures observed state, not a complete build recipe. Internal
# # targets, host-coupled driver state, mounted source trees, local patches,
# # and private toolchains are not recoverable from observation. Use as a
# # starting point.
# # ============================================================================
# # build_system: kind=buck2 buck2_version=buck2 ... repo_root=... revision=<sha>
# # 3 buck-introspected libraries:
#
# # original_target = //third-party/rocm:hipblaslt
# prebuilt_cxx_library(
#     name = "hipblaslt",
#     version = "<repo-sha>",
#     # NOTE: emitted as prebuilt_cxx_library because env.json
#     # captures the resolved binary, not the source-build recipe.
#     ...
# )
```

The emitter is text generation only: it does NOT invoke `buck2 build`,
vendor any Buck rules / macros / rule libraries, or attempt to
reconstruct internal targets, host-coupled driver state, mounted
source trees, local patches, or private toolchains. Those are not
recoverable from observation by construction; the loud header
documents the limitation so consumers don't mistake the fragment
for a complete build recipe.

Each entry in `library_introspection` whose `source == "buck"` becomes
one `prebuilt_cxx_library(...)` call pinning the captured `revision`
as a `version` attribute. Entries with `source` set to `"pkg-config"`
or `"elf"` (from A1's existing library introspection path) are
skipped -- they have no Buck target to point at -- but any
`library_introspection_alternates` entries are appended as a trailing
comment block so a diff against a non-Buck reference snapshot can
spot the merged-away identifiers.

`--format dockerfile` is reserved on the CLI surface but exits with a
"not yet implemented" error; the actual implementation tracks
separately. Future formats slot in via the same `--format` flag.

## Testing the probe locally

```bash
# Assuming aorta is installed in your venv
aorta env probe -o /tmp/env.json
jq '.partial' /tmp/env.json
jq '.partial_reasons' /tmp/env.json

# Force the no-rdhc fallback (rdhc isn't on PATH)
PATH= aorta env probe -o /tmp/env_no_rdhc.json
jq '.system_health' /tmp/env_no_rdhc.json    # null
jq '.hipblaslt.commit' /tmp/env_no_rdhc.json # still populated

# Compare across docker images (manual until `aorta env matrix` lands)
docker run --rm <image-A> aorta env probe -o /workspace/env_a.json
docker run --rm <image-B> aorta env probe -o /workspace/env_b.json
diff <(jq -S . env_a.json) <(jq -S . env_b.json)
```

## Output modes

`aorta env probe` has three output modes. Pick whichever matches what
you're actually doing — the JSON artifact, a quick eyeball, or one
field for a script.

| Mode | When to use | What it does |
| --- | --- | --- |
| Default | Producing an artifact to keep, diff, or attach to a trial result | Writes the full JSON to `env.json` (or `-o <path>`) and prints the brief to stdout |
| `--summary` | "I just want to look at the build" — no archival need | Prints only the one-screen brief; **does not write JSON** |
| `--field DOTTED.PATH` | Scripting a one-value lookup | Prints just that field's value as JSON (type preserved); **does not write JSON** |

```bash
# Default mode (artifact + brief)
aorta env probe -o /tmp/env.json

# Quick eyeball -- no file written
aorta env probe --summary

# One-field lookup, JSON-typed output (a bool prints as `true`, a
# string prints as `"foo"`, a missing optional prints as `null`)
aorta env probe --field schema_version
aorta env probe --field pytorch_build.git_commit
aorta env probe --field pytorch_build.ninja_hipcc.targets.ck_sdpa.use_defines_present.USE_ROCM_CK_SDPA
```

When `--field` can't resolve the path, it surfaces a one-line error
listing the keys that *are* available at the parent level — so
typos and renames are self-correcting:

```bash
$ aorta env probe --field pytorch_build.cmke_cache
Error: Key 'cmke_cache' not found at 'pytorch_build'. Available keys:
  binary_introspection, build_flags, cmake_cache, cuda_version, debug,
  flags, git_commit, hip_version, install_kind, ninja_hipcc (+ 2 more)
```

`--field` supports simple dotted paths. For keys that themselves
contain a `.` (the only one in the current schema is
`"libaotriton_v2.so"` under `torch_lib_bundled`), use `jq` on a full
snapshot instead.

## jq cookbook

Copy-paste recipes for the common questions. All assume you have a
snapshot file at `/tmp/env.json` (or two snapshots: `good.json` and
`bad.json` for diffs).

### Is the build configured the way I think it is?

```bash
# All three vantages on USE_ROCM_CK_SDPA -- cmake-time, HIPCC-time,
# and symbol-presence. They should all agree.
jq '{
  cmake:   .pytorch_build.cmake_cache.entries.USE_ROCM_CK_SDPA.value,
  hipcc:   .pytorch_build.ninja_hipcc.targets.ck_sdpa.use_defines_present.USE_ROCM_CK_SDPA,
  sym_fwd: .pytorch_build.binary_introspection.libtorch_hip_symbol_counts."ck_tile::FmhaFwd",
  sym_bwd: .pytorch_build.binary_introspection.libtorch_hip_symbol_counts."ck_tile::FmhaBwd"
}' /tmp/env.json

# Same for AOTriton -- different libs and symbol families.
jq '{
  cmake:        .pytorch_build.cmake_cache.entries.USE_AOTRITON.value,
  bundled_lib:  .aotriton.bundled_present,
  bundled_dir:  .aotriton.bundled_images_dir_present,
  bundled_sym:  .pytorch_build.binary_introspection.libtorch_hip_symbol_counts."aotriton::",
  mha_fwd_aot:  .pytorch_build.binary_introspection.libtorch_hip_symbol_counts.mha_fwd_aot
}' /tmp/env.json
```

### Which SDPA backends are compiled in AND enabled at runtime?

```bash
# Compiled in (build-time, derived from symbol counts)
jq '.pytorch_build.binary_introspection.libtorch_hip_symbol_counts
    | with_entries(.value = (.value > 0))' /tmp/env.json

# Enabled at runtime (torch.backends.cuda.<...>_sdp_enabled())
jq '.pytorch_sdpa.backends_enabled' /tmp/env.json
```

### What HIPCC defines + codegen flags did `ck_sdpa` compile with?

```bash
# Just the flags the SDPA-NaN triage cares about (yes/no per flag)
jq '.pytorch_build.ninja_hipcc.targets.ck_sdpa.use_defines_present' /tmp/env.json

# Codegen flags (denormal-flush, ffast-math, ffp-contract, ...)
jq '.pytorch_build.ninja_hipcc.targets.ck_sdpa.codegen_flags_present' /tmp/env.json

# GPU offload archs
jq '.pytorch_build.ninja_hipcc.targets.ck_sdpa.offload_archs' /tmp/env.json

# Which parser ran (build.ninja vs legacy FindHIP fallback) and how
# many scripts the fallback walked
jq '.pytorch_build.ninja_hipcc | {_parser, _legacy_scripts_scanned}' /tmp/env.json
```

### What's actually in the GEMM kernel libraries?

```bash
# hipBLASLt identity -- the four fields that motivated this whole
# probe (a hipBLASLt swap is the #1 source of GEMM drift).
jq '.hipblaslt | {rocm_release_tweak, package_version, lib_hash, kernel_db_revision}' /tmp/env.json

# Same for rocBLAS, MIOpen, RCCL
jq '{rocblas, miopen, rccl}' /tmp/env.json
```

### Diff two snapshots

```bash
# Full diff -- gold standard
diff <(jq -S . good.json) <(jq -S . bad.json)

# Just the SDPA-relevant compile state (much smaller diff)
diff \
  <(jq -S '.pytorch_build.ninja_hipcc.targets' good.json) \
  <(jq -S '.pytorch_build.ninja_hipcc.targets' bad.json)

# Just env_vars (most common runtime-state difference)
diff \
  <(jq -S '.env_vars' good.json) \
  <(jq -S '.env_vars' bad.json)

# Just GEMM library identities (most common build-time difference)
diff \
  <(jq -S '{hipblaslt, rocblas, miopen, rccl}' good.json) \
  <(jq -S '{hipblaslt, rocblas, miopen, rccl}' bad.json)

# Just submodule commits
diff \
  <(jq -S '.pytorch_build.submodule_commits' good.json) \
  <(jq -S '.pytorch_build.submodule_commits' bad.json)
```

### Where can things have silently failed?

```bash
# Was anything missing? List every block that didn't probe cleanly.
jq '{partial, partial_reasons}' /tmp/env.json

# Just the action items -- one reason per line for grep / wc
jq -r '.partial_reasons[]' /tmp/env.json
```

### One-liner answers for triage hotline questions

```bash
# "What ROCm + HIP versions is this build against?"
jq '{rocm: .rocm.version, hip: .hip.version, gfx: .gpu_arch.gfx_targets}' /tmp/env.json

# "Which submodule commits is this PyTorch built from?"
jq '.pytorch_build.submodule_commits' /tmp/env.json

# "Was the build done from source or installed as a wheel?"
jq -r '.pytorch_build.install_kind' /tmp/env.json

# "Is aiter installed, and which arches are its HSA blobs covering?"
jq '.aiter | {
  package_dist_name,
  commit,
  archs: (
    (.hsa_tree // {}) | to_entries
    | (first // null)
    | (if . then (.value | keys) else [] end)
  )
}' /tmp/env.json
```

### Build a SDPA-NaN triage one-pager from one snapshot

```bash
jq '{
  build: {
    rocm:           .rocm.version,
    hip:            .hip.version,
    pytorch_commit: .pytorch_build.git_commit,
    install:        .pytorch_build.install_kind,
    parser:         .pytorch_build.ninja_hipcc._parser
  },
  sdpa_compile_in: .pytorch_build.binary_introspection.libtorch_hip_symbol_counts
    | with_entries(.value = (.value > 0)),
  sdpa_runtime_enabled: .pytorch_sdpa.backends_enabled,
  ck_sdpa_flags: (.pytorch_build.ninja_hipcc.targets.ck_sdpa // {}
    | {defines: .use_defines_present, codegen: .codegen_flags_present, archs: .offload_archs}),
  numerics_relevant_env: (.env_vars
    | {HSA_XNACK, HSA_KERNARG_POOL_SIZE, HSA_NO_SCRATCH_RECLAIM,
       AMDGCN_USE_BUFFER_OPS, DISABLE_TF32,
       TORCH_ROCM_FA_PREFER_CK, TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL})
}' /tmp/env.json
```

## Out of scope (deferred to P1)

* `aorta env matrix` (multi-docker fan-out)
* `aorta env diff env_a.json env_b.json`
* GPU kernel introspection (anything that runs HIP work)
* Workload config (`AMP_DTYPE`, `MODEL_DTYPE`, ...) -- captured by
  `aorta run` in the trial result (Task B1), not by env probe

## See also

* Module reference: [`src/aorta/instrumentation/README.md`](../src/aorta/instrumentation/README.md)
* Issue with the full acceptance criteria: [#147](https://github.com/ROCm/aorta/issues/147)
