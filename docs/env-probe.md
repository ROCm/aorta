# Env Probe

Capture a versioned, schema-stable snapshot of the trial environment so
that cross-environment comparison becomes a `jq` diff instead of a
multi-day investigation.

For Buck2 workloads, including the required client/workload snapshot pair, see
[Buck2 Env-Probe Workflow](buck2-env-probe-workflow.md).

The same code path is used three ways:

* **CLI**: `aorta env probe -o env.json` -- on-demand, by an operator.
* **Application API**: `capture_to("env.json", ...)` -- in-process capture for
  Buck `.par` applications that own `__main__`.
* **B1 (per-trial runner)**: calls `collect_env()` once per trial; the
  snapshot is embedded in `TrialResult.env`.
* **B2 (matrix runner)**: captures the host environment once per matrix. For
  an isolated environment, it asks the workload wrapper to run the probe
  inside that environment and promotes the resulting snapshot; the core
  dispatcher does not launch the container. See
  [Environment snapshots](../recipes/README.md#environment-snapshots).

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

### Catalog vs. runtime trace -- what `env probe` does and doesn't capture

The **catalog** (what `env probe` captures, under `tensile_catalog` /
`miopen_catalog` / `rocfft_catalog`) identifies the *recipe book*: which
hipBLASLt/rocBLAS Tensile, MIOpen, or rocFFT kernel/solution menu is
installed on disk. It does **not** tell you which recipe the runtime
actually picked for a given GEMM/conv/FFT -- the per-call solution ID
(`#381881` / `#368xxx` you see in workload logs); that requires a
runtime trace (a separate capability, deferred to the runtime
follow-up).

Kernel selection happens in two layers:

1. **The frozen menu (layer 1, captured here).** At build/install time
   the library bakes a solution library / logic files into / alongside
   the `.so` -- the full menu of recipes plus a decision tree. This menu
   does not change at runtime; it is files on disk, fully visible without
   running anything. The `*_catalog` blocks fingerprint **and enumerate**
   it.
2. **The runtime pick (layer 2, NOT captured here).** The actual choice
   happens per-call at runtime, querying live device properties -- so a
   driver/firmware/KFD change can flip the pick even with an identical
   `.so`. Out of scope for the no-compute probe.

**Why this matters:** two hosts can report the *same* hipBLASLt commit
(e.g. the historically-broken `7e32d53eb1`) yet ship a **materially
different** Tensile install -- a vendor build can patch or select
different kernels even when nominally pointing at that commit. A bare
version/commit string would call these "the same"; the deepened
`tensile_catalog.<lib>.menu` per-file content hashes make the difference
**diffable from the installed files alone**. (The solution-ID half --
which kernel each host's runtime actually selected -- belongs to the
runtime follow-up.)

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
                           from a bound `buck2 cquery 'deps(%s)'
                           <label> --json`
                           (each matched entry carries both a
                           stripped `target` and the raw
                           `configured_target`, schema 1.6). Ignored
                           if buck2 isn't on PATH.
  --buck-option TYPE=VALUE Exact ordered Buck input: mode=VALUE,
                           config=KEY=VALUE, or modifier=VALUE.
                           Repeat in the workload command's order.
  --buck-mode-file PATH    Grouped convenience form for mode files.
  --buck-config KEY=VALUE  Grouped convenience form for -c values.
  --buck-modifier TEXT     Grouped convenience form for -m values.
  --buck-default-context   Confirm that the target uses Buck's
                           default invocation context. Mutually
                           exclusive with the three explicit context
                           options above.
  --buck-timeout INTEGER   Per-call timeout (seconds, must be >= 1)
                           for `buck2 cquery 'deps(...)'`.
                           [default: 10]
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
Wrote env probe to /tmp/env.json (schema_version=1.13) [PARTIAL]
  runtime:   baremetal / python=venv  container_detected=no  probe=direct  ns=mnt:1a2b3c…
  build_sys: none
  buck ctx:  status=not_requested source=none fingerprint=-
  rocm:      7.2.1 (dev: None)
  hip:       7.2.53211-e1a6bc5663 (amd)
  hipblaslt: 1.2.2 rocm_release_tweak=dabb6df2b9
  rocblas:   5.2.0 rocm_release_tweak=dabb6df2b9
  miopen:    3.5.1 rocm_release_tweak=dabb6df2b9
  rccl:      2.27.7 (code=22707) net_plugin=external [librccl-net.so]
  nics:      broadcom(fw=232.0.219.16/pkg 232.1.196.16 links=8/8)  cx7(fw=28.36.1010 (FB_0000000038) links=0/0)
  gpu_arch:  ['gfx942'] (counts={'gfx942': 8})
  host:      kernel=5.15.0-174-generic machine=x86_64  glibc=2.35
  ck:        system=1.2.0/23d531c8  ck_tile=yes  libtorch_hip=4067 ck:: syms
  tensile:   kernel_db=filenames-sha256:743a8d…  [Tensile pip pkg: (not installed); build-time tool, normal]
  catalog:   tensile[hb=content-sha256:1a2b3c… rb=content-sha256:4d5e6f…]  miopen=content-sha256:7890ab…  rocfft=absent
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
from aorta.instrumentation.environment import capture_to, collect_env, EnvSnapshot

snapshot: EnvSnapshot = collect_env()   # NEVER raises

# Buck .par/application path: capture this process and write env.json.
# Filesystem errors raise; collection itself remains fail-soft.
workload_snapshot = capture_to(
    "env.workload.json",
    probe_invocation="buck2_run",
)

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

`collect_env()` is the in-memory entrypoint and NEVER raises: each probe is
fail-soft, and the orchestrator body has a disaster-recovery helper for any
genuinely unexpected failure. `capture_to()` uses the same collection path and
returns the same `EnvSnapshot`, then writes it as UTF-8 JSON. It raises only
when the output directory/file cannot be created or validated.

## env.json schema

| Top-level key | Type | Source | Notes |
| --- | --- | --- | --- |
| `schema_version` | `str` | constant | Currently `"1.16"`. See the changelog comment in `src/aorta/instrumentation/environment.py` next to the `SCHEMA_VERSION` constant for the field-by-field history. |
| `captured_at` | `str` | `datetime` | ISO-8601 UTC with trailing `Z` |
| `partial` | `bool` | computed | `True` if any probe fell back |
| `partial_reasons` | `list[str]` | per-probe | one human-readable line per fallback |
| `system_health` | `dict \| null` | `rdhc --quick --json` (subprocess) | verbatim parsed JSON; `null` when rdhc absent / sudo unavailable / timeout / malformed |
| `rocm` | `dict[str, str \| null]` | `<rocm-root>/.info/version{,-dev}`, `/sys/module/amdgpu/version`, then the TheRock manifest / pip metadata / `torch.__version__` | `version`, `version_dev`, `kmd_version`, plus schema 1.16 attribution: `root`, `lib_root`, `root_source` (`ROCM_PATH`/`ROCM_HOME`/`opt_rocm`/`import:_rocm_sdk_core`/`none`), `layout` (`classic`/`wheel`), `version_source` (`version_file`/`therock_manifest`/`pip:<dist>`/`torch`/null). The root is resolved rather than hardcoded to `/opt/rocm` (issue #381) so a wheel-layout install reads correctly; `root_source: "none"` is what distinguishes "no ROCm install found" from "install found but it has no version file". |
| `therock` | `dict` | `<rocm-root>/share/therock/therock_manifest.json` | Schema 1.16. Build provenance for a wheel-layout (TheRock) install: `status` (`present`/`absent`), `manifest_path`, `rocm_version`, `rocm_package_version`, `the_rock_commit`, `github_run_id`, `github_job`, `gemm_libraries_commit`, `submodules` (`[{name, url, pin_sha, patches}]`). **Documented absence**: a classic `/opt/rocm` install ships no manifest, so `status:"absent"` there carries no `partial_reason`. Where present it is strictly richer than the header parse -- full 40-char upstream pins vs a truncated tweak, plus the applied-patch list, which is the only signal that a built library is *not* just what its pinned commit contains. |
| `amdgpu_driver` | `dict` | `dpkg-query`/`rpm` (package), `modinfo amdgpu` (module), `/sys/module/amdgpu/version` (reused from `rocm.kmd_version`), `/dev/kfd` + `/sys/class/kfd` (existence) | Schema 1.10. `scope` (constant `"host_kernel"`), `status` (`"present"`/`"absent"`), `package_name` (stable candidate family `amdgpu-dkms`/`amdgpu-kmod`/null), `package_version`, `package_full_name` (complete canonical identity — apt `name=version` or rpm NVRA — capturing kernel-suffixed package names such as `amdgpu-kmod-<kernel>-<driver-version>.<arch>` that `package_name` alone drops; the single field two host snapshots diff on), `package_manager` (`"dpkg"`/`"rpm"`/null), `module_version`, `module_srcversion`, `kmd_version`, `kfd_device_present`, `kfd_sysfs_present`. **HOST-KERNEL SCOPE applies only to `kfd_device_present`/`kfd_sysfs_present`/`kmd_version`**: the amdgpu KMD + KFD live in the host kernel a container *shares*, so these three fields report the **host's** driver even from inside a container — complementary to `rocm`/`hip` which read the container's `/opt/rocm` userspace. **`package_name`/`package_version`/`package_full_name`/`package_manager` and `module_version`/`module_srcversion` are NOT host-guaranteed** — dpkg/rpm and `modinfo` read the *current filesystem's* package DB and `/lib/modules`, so from inside a container these reflect the container's own (typically driver-less) view even while the three fields above still show the host's driver as present. **Documented absence**: a GPU-less host → `status:"absent"` with NO `partial`. Only a *conflict* — amdgpu module loaded (`modinfo` metadata present) but no dpkg/rpm package resolvable — records a `partial_reason`; `/dev/kfd` alone never does, since a passthrough container legitimately has the host's KFD node without a container-local package. Package lookup is a portable, **glob-capable** dpkg-then-rpm query parsed in Python (no shell pipe), so kernel-suffixed RPM names are matched. |
| `hip` | `dict[str, str \| null]` | `hipconfig --version/--platform/--compiler/--runtime/--cpp_config` | five subprocesses; `--version` and `--platform` cannot be combined (no delimiter) |
| `hipblaslt` | `dict` | header parse + `sha256(libhipblaslt.so)` + sorted-filenames hash of `lib/hipblaslt/library/*` (and `library/<gfx>/*` on the wheel layout) + the TheRock manifest | `rocm_release_tweak` (NOT a per-hipBLASLt commit -- it's the ROCm release identifier shared across every library in a release; see note below), `package_version`, `lib_hash`, `kernel_db_revision`, `applied_prs: {}`, and schema 1.16 `upstream_commit` / `upstream_commit_matches_tweak`. `upstream_commit` is the full 40-char `rocm-libraries` pin from the manifest -- ROCm consolidated hipBLASLt and rocBLAS into that monorepo, so it is now their upstream identity. It matters most on a wheel image, where the version header is a devel-only file that runtime images omit: without it the hipBLASLt provenance would read `null` exactly where the NaN escalations need it. `null` on a classic install is expected, not a fallback. `upstream_commit_matches_tweak` cross-checks the two sources by prefix at the tweak's own length (the tweak is a short hash of unspecified width -- measured 10 chars on ROCm 7.2.4), and is `null` whenever only one source is present. |
| `rocblas` | `dict` | header parse + `sha256(librocblas.so)` + sorted-filenames hash of `lib/rocblas/library/*` (and `library/<gfx>/*`) + the TheRock manifest | Same shape as `hipblaslt`, including `upstream_commit`, which is the same `rocm-libraries` pin. Header lives at `include/rocblas/internal/rocblas-version.h`. |
| `miopen` | `dict` | header parse + `sha256(libMIOpen.so)` + sorted-filenames hash of `share/miopen/db/*.txt` | `rocm_release_tweak`, `package_version`, `lib_hash`, `kernel_db_revision`. MIOpen drives convolution kernels on ROCm; kernel-DB drift changes which conv kernel runs. |
| `rccl` | `dict` | header parse for `NCCL_VERSION_CODE` + `sha256(librccl.so)` + resolve+hash of `NCCL_NET_PLUGIN` + best-effort `sha256` of `librccl-anp.so`/`librccl-net.so` in the lib dir | `version_code` (raw int, e.g. `22707`), `version` (decoded `"2.27.7"`), `lib_hash`, `net_plugin_mode` (`"external"`/`"internal"`/`"unknown"`), `plugin_path`, `plugin_lib_hash`, `anp_lib_hash`, `net_lib_hash`. RCCL is AMD's NCCL-compatible collectives library. **`plugin_path`/`plugin_lib_hash` are the authoritative net-plugin signal**: `NCCL_NET_PLUGIN` is resolved to a real `.so` (absolute path, or bare name found on `LD_LIBRARY_PATH` then the rccl lib dir) and *that* file is hashed -- this is how a real AMD-ANP deployment ships the plugin (`librccl-net.so` under a user-build tree, not `librccl-anp.so` in `/opt/rocm/lib`). `net_plugin_mode` is `"external"` when `NCCL_NET_PLUGIN` resolves, `"internal"` when it is unset/empty (built-in net-ib), and `"unknown"` when it is set but unresolvable (misconfigured launcher -- this records a `partial_reason`; the unset case does not). `anp_lib_hash`/`net_lib_hash` are a best-effort scan of the rccl lib dir for the packaged-install case; `null` when absent (documented absence, no `partial`). |
| `gpu_arch` | `dict` | `rocm_agent_enumerator` subprocess (no `/dev/kfd` access typically required) | `agent_count`, `gfx_targets` (sorted unique), `agent_arch_counts` (per-arch distribution -- captures both homogeneous and mixed-arch boxes). |
| `nics` | `dict` | `lspci` presence gate + `ethtool -i` + `ibv_devices` + `rdma link` (Tier-1, sudo-free); AINIC adds `nicctl` via `sudo -n` (Tier-2) | Multi-vendor RoCE NIC/fabric stack keyed by vendor (`ainic`/`broadcom`/`cx7`), schema 1.7 (issue #202). Each vendor: `present` (bool, from `lspci -d <id>`). When present: `driver_version`, `firmware`, `pkg_version` (Broadcom's `<fw>/pkg <pkg>` split out; `null` when absent or equal to `firmware`), `rdma_devices` (list), `links` (`[{device, state, netdev}]`). **RDMA devices and links are bound to a vendor by their kernel driver** (resolved via the sysfs `device/driver` symlink — `ionic`/`bnxt_en`/`mlx5_core`), NOT by device-name prefix, because device names vary by host (e.g. `ionic_0` vs `rdma3`). AINIC-only Tier-2: `nicctl_version`, `card` (`asic`/`host_sw`/`firmware`/`uuid`), `profile` (`device_config`/`sriov`), `dcqcn` (`enabled`/`token_bucket_size`/`ai_rate`/`hai_rate`/`cnp_dscp`; the DCQCN query targets the first resolved AINIC RDMA device). **Documented absence**: vendor absent from `lspci` -> `{"present": false}`, no `partial`; present with zero RDMA devices is valid. AINIC Tier-2 output layouts are tolerant-parsed and pending confirmation against real `nicctl` capture. |
| `host` | `dict` | `os.uname()` + `os.confstr("CS_GNU_LIBC_VERSION")` | `kernel_release`, `kernel_version`, `machine`, `glibc_version`. Kernel + glibc drift is the #1 confound for compiled-against-vs-runtime issues with C++ extensions. |
| `composable_kernel` | `dict` | header at `include/ck/version.h` + `nm -D` of `libtorch_hip.so` piped through `c++filt` + `torch.__config__.show()` flag scan | Two sub-blocks (`system: {version, commit, ck_tile_present}`, `pytorch_bundled: {present, symbol_count}`) plus top-level `pytorch_use_ck_sdpa` / `pytorch_use_ck_gemm` booleans (build-time flags baked into the wheel; NOT runtime env vars). System and bundled CK can drift independently. |
| `tensile` | `dict` | optional `import Tensile` + sorted-filenames hash over the union of hipBLASLt + rocBLAS kernel DBs | `package_version` (usually `null` outside builders), `kernel_db_combined_hash` |
| `tensile_catalog` | `dict` | reuses the `hipblaslt`/`rocblas`/`tensile` captures + a new per-file enumeration of `lib/{hipblaslt,rocblas}/library/*` | **The "recipe book" -- installed-library identity, NOT the runtime-selected solution ID.** Top-level `doc` + `status`. Per-library `hipblaslt`/`rocblas` carry the existing `package_version`/`lib_hash`/`kernel_db_revision` (no regression) plus a `menu` sub-block: `status`, `reason`, `dir`, `logic_file_count` (`.dat`/`.yaml`), `file_count`, `gfx_arch_coverage`, `files` (per-file `{name,size,suffix,is_logic,sha256}`), and `combined_content_hash`. `combined` mirrors the `tensile` block. **`files` is `null` by default (compact mode)** — the per-file lists dominate env.json size (~25k lines on a populated host), so the default probe drops them while keeping every count and `combined_content_hash` (a diff still detects a changed catalog); run `aorta env probe --extended` to retain the full per-file list for localizing *which* file changed. Partial-not-silent: a dir that can't be located/listed is `status:"partial"` (distinct from a present-but-empty menu, which is `ok` with `logic_file_count:0`). Added in schema 1.9 (issue #54); compact default added in schema 1.10. |
| `miopen_catalog` | `dict` | reuses the `miopen` capture + a per-file enumeration of the MIOpen db dir (`share/miopen/db/`) | **Recipe book for MIOpen's convolution databases** -- same scoping as `tensile_catalog`. `doc` + `status`, the reused `package_version`/`lib_hash`/`kernel_db_revision` (no regression), `db_dir` + `db_dir_source` (`default` or `MIOPEN_SYSTEM_DB_PATH`), `env_overrides`, and a `menu`. `logic_file_count` counts the selectable DBs (`.kdb`/`.fdb.txt`/`.db.txt`/`.db`); `.ktn.model`/`.tn.model` heuristic models are catalog members but not logic. **`menu.files` is `null` by default (compact mode)** — same size rationale as `tensile_catalog`; use `aorta env probe --extended` to retain the per-file list. Added in schema 1.9 (issue #54 follow-up); compact default added in schema 1.10. |
| `rocfft_catalog` | `dict` | optional `rocfft_kernel_cache.db` under `$ROCFFT_RTC_SYS_CACHE_PATH` or `/opt/rocm/lib/[rocfft/]` | Fingerprint of rocFFT's **optional** AOT kernel cache -- the READ-ONLY system cache, not the read-write runtime user cache. rocFFT usually compiles at runtime, so absence is normal: a *probed* "no cache" result is `status:"absent"` with **no** partial reason. (The default/backfill/disaster shape is `status:"partial"` with a "not captured" reason instead, so a snapshot that never probed is distinguishable from one that probed and found nothing.) `doc` + `status` (`ok`/`absent`/`partial`) + `env_overrides` (both `ROCFFT_RTC_SYS_CACHE_PATH` -- the override this catalog actually resolves against -- and `ROCFFT_RTC_CACHE_PATH`, recorded for visibility only since it names the mutable, workload-dependent user cache, not installed identity; both recorded even when no cache is found so a configured-but-empty override is visible) + `kernel_cache` (`present`, `path`, `source`, `size`, `sha256`, `reason`). Added in schema 1.9 (issue #54 follow-up). |
| `triton` | `dict` | `import triton; triton.__version__` (+ commit parse) | `package_version`, `commit` (schema 1.8). ROCm Triton fork bakes the source commit into `__version__` (e.g. `3.5.1+rocm7.2.1.gita272dfa8` -> `commit: "a272dfa8"`); fb builds versioned `+fb` carry no SHA -> `commit: null`. |
| `torchrec` | `dict` | `find_spec("torchrec")` + loader-based text read of the resolved package's `version.py` + separate distribution metadata | `package_version`, `commit`, `source_version`, `source_commit`, `distribution_version`. Schema 1.15 keeps import-target and dist-info identities separate. The primary identity follows the import target: source identity wins; if its version is unreadable, metadata is promoted only when that distribution's `RECORD` owns the resolved module/package. Unrelated or unprovable metadata remains separate and adds `partial_reasons`; a source-only SHA remains `source_commit` rather than forming a contradictory primary identity without a version. `.par`/zip resources are read through the loader without importing torchrec or touching CUDA. PEP 420 namespaces are recognized before and after import and count as absent unless unrelated metadata also exists, in which case that metadata is retained separately with a partial reason. |
| `fbgemm` | `dict` | optional `import fbgemm_gpu` (+ commit parse) + parse of `torch.__config__.show()` for `-DUSE_FBGEMM*` defines | `package_version`, `commit` (schema 1.8 -- best-effort git SHA from a setuptools_scm `+g<sha>` local-version segment or a `git_version`/`__commit__` module attr; `null` when fbgemm_gpu is vendored-in-torch rather than separately installed, or carries no SHA), `pytorch_use_fbgemm`, `pytorch_use_fbgemm_genai`. The two booleans capture the build-time decision baked into the PyTorch wheel even when `fbgemm_gpu` isn't a separate pip package. |
| `aiter` | `dict` | `import aiter; aiter.__version__` + `importlib.metadata.version("amd_aiter" \| "aiter")` + scan of `aiter_meta/hsa/<gfx>/` (or `$AORTA_PYTORCH_SRC/third_party/aiter/hsa/`) | `package_version`, `package_dist_name` (which PyPI dist matched: `amd_aiter` is the canonical AMD-internal ROCm/PyTorch image dist; `aiter` is the upstream name), `commit` (parsed from the setuptools_scm `+g<sha>` local-version segment, matches the image tag's `aiter-<sha>` label), `hsa_tree` (issue #176 -- per-arch fingerprint of pre-built `.co` kernel binaries: file_count, co_count, deterministic combined_sha256). Most installs record `null` for everything; absence is silent. |
| `aotriton` | `dict` | scan of `<torch>/lib/libaotriton_v2.so*` filenames + `sha256` of the resolved file + presence of `<torch>/lib/aotriton.images/` + `$AOTRITON_INSTALLED_PREFIX` | Default ROCm Flash Attention backend. Bundled in the wheel via `cmake/External/aotriton.cmake` (NOT a `third_party/` git submodule). Fields: `bundled_present`, `bundled_version`, `bundled_lib_hash`, `bundled_images_dir_present`, `installed_prefix`. CK is the alternative backend (toggled via `TORCH_ROCM_FA_PREFER_CK=1`). |
| `runtime_context` | `dict` | `/.dockerenv`, `/run/.containerenv`, `$SINGULARITY_NAME`, `/proc/1/cgroup`, `sys.prefix`, `$CONDA_DEFAULT_ENV` | `type`, `python_env`, `venv_path`, `conda_env_name`. **`type` only ever returns `docker`/`podman`/`singularity`/`baremetal`** — an unnamed sandbox (RE worker, containerd k8s pod) falls through to `baremetal`; use `container_detected` (below) for the runtime-agnostic "am I isolated?" answer. |
| `container_detected` | `bool` | named-runtime match + `/proc/self/ns/mnt` vs `/proc/1/ns/mnt` (private mount ns) + container/k8s tokens in `/proc/self/cgroup` | Schema 1.11. Runtime-**agnostic** isolation smoke test: `true` on *any* sandbox signal even when the runtime can't be named. Fixes the `runtime_context.type == "baremetal"` false negative for RE-workers / k8s pods. `container_detected:true` + `type:"baremetal"` is the honest reading of an unnamed sandbox. Fail-soft; `false` means "no isolation signal observed," not "definitely bare metal." |
| `execution_context` | `dict` | `--execution-context` CLI flag (self-declared) | Schema 1.11. `probe_invocation` (`"direct"` \| `"buck2_run"` \| `"buck2_action"`; `"direct"` by default) records **how the probe was launched** — set `buck2_action` when running the probe as a Buck2 action so it captures the executor's env, not the invoking shell's. `likely_execution_platform` (`str \| null`) is reserved for phase-2 Buck2 work and is always `null` today. See the design note `docs/env-probe-container-execution-context.md`. The flag also **warns to stderr** when a non-`direct` context is claimed but `container_detected` is `false` and neither `$AORTA_RE_IMAGE` nor `$AORTA_DOCKER_IMAGE` is set (you may have probed the wrong place). |
| `probe_namespace` | `str \| null` | `/proc/self/ns/mnt` (primary) or `/proc/self/ns/cgroup` (fallback), hashed with `/proc/sys/kernel/random/boot_id` | Schema 1.12. **Mismatch-only namespace observation**, not a durable identity. Value forms: `mnt:<sha256[:16]>` or `cgroup-ns:<sha256[:16]>`; when `boot_id` is unavailable, the source token remains hashed and the prefix becomes `mnt-local:` / `cgroup-ns-local:`. Different values with the same prefix prove a different boot or namespace observation. Equal values are advisory because Linux can recycle non-initial namespace inode numbers after teardown. Different source prefixes are not directly comparable, and `*-local:` values are not cross-host comparable. Missing boot identity, fallback use, and total failure add `partial_reasons`; `null` means neither namespace handle was readable. |
| `docker` | `dict \| null` | `$AORTA_DOCKER_IMAGE` / `$AORTA_DOCKER_DIGEST` env vars + `/proc/self/cgroup` | `null` on baremetal; image+digest provided by the launcher (the only reliable way from inside a container) |
| `env_vars` | `dict[str, str \| null]` | explicit canonical list (currently 137 names, generated from `ENV_KNOB_REGISTRY` in `env_knobs.py`; the count is asserted by `test_docs_env_var_count_matches_registry`, so it cannot go stale) | GPU scoping + HSA / runtime + GPU queue / codegen + NCCL/RCCL + AINIC net-plugin/fabric tuning + gfx950 fence-ordering knob + FBGEMM + MIOpen + SDPA backend selection + GEMM backend preference + hipBLASLt autotune + PyTorch / inductor. Build-time cmake flags (`USE_ROCM_CK_SDPA`, `USE_ROCM_CK_GEMM`, `USE_FBGEMM*`) are NOT in this list -- they're surfaced under their respective library blocks instead, parsed from `torch.__config__.show()`. |
| `python_version` | `str` | `platform.python_version()` | always populated |
| `pytorch_version` | `str \| null` | optional `import torch` (no CUDA/HIP context init) | `null` when torch absent |
| `pytorch_build` | `dict` | `torch.version.{git_version,hip,cuda,debug}` + install-kind detection + optional `git -C <src>/third_party/<sub> rev-parse HEAD` + parse of `torch.__config__.show()` + `nm -D libtorch_hip.so \| c++filt` symbol grep + scan of `<torch>/lib/` + parse of `<source>/build/CMakeCache.txt` + stream of `<source>/build/build.ninja` (modern `enable_language(HIP)` path) or walk of `<source>/build/**/<target>.dir/**/*.hip.o.cmake` (legacy `FindHIP.cmake` fallback) | `git_commit` is the linchpin -- pins every vendored submodule deterministically. Sub-blocks: `flags` (raw `build_settings`, `cxx_defines`, `cxx_flags_raw`, `cuda_flags_raw`, `gpu_arch_list`), `build_flags` (issue #170 stable 17-key parsed bool/str/None subset, with `CAFFE2_USE_MIOPEN` aliased to `USE_MIOPEN`), `binary_introspection` (`libtorch_hip_symbol_counts`, `torch_lib_bundled`, `cxx_flags_use_defines` -- pure facts, no ON/OFF inference), `cmake_cache` (source/editable installs only -- allowlisted entries from CMakeCache.txt), `ninja_hipcc` (source/editable installs only -- per-target HIPCC defines + codegen flags + offload archs; `_parser` discriminates the two parser strategies). See "PyTorch source-tree submodule probing" below. |
| `build_system` | `dict` | `buck2 --version` + `buck2 root` + `hg id -i` / `git rev-parse HEAD` | Always present. `{"kind": "buck2", "buck2_version": str, "repo_root": str, "revision": str \| null}` when buck2 is on PATH AND we are demonstrably inside a Buck checkout (both `buck2 --version` and `buck2 root` succeed); `buck2_version` and `repo_root` are guaranteed populated, only `revision` may be `null`. `{"kind": "none"}` in every other case, including the dominant "buck2 is installed but cwd is not inside a Buck checkout" scenario. Added in schema 1.3 for issue #163 (A1.2a) so consumers can branch on Buck2 vs. system-package environments. See "Running inside a Buck environment" below. |
| `buck_invocation` | `dict` | typed Buck context + bound `buck2 cquery 'deps(%s)' <target> --json` | Schema 1.13, always present. `status` distinguishes `not_requested`, `buck_not_detected`, `success`, and `failure`; `target` records the requested label. `context_source` is `none`, `unspecified`, `default_confirmed`, or `explicit`. Ordered `mode_files`, ordered `config_keys` (**never config values**), ordered `modifiers`, `option_order`, and `context_fingerprint` (`sha256:<hex>` over the full ordered context values) make the client-side query context comparable without disclosing raw overrides. `configured_root_target` retains the configured root label when cquery returns one; `comparison` is currently always `not_compared`. This is configured-graph invocation provenance, not execution placement: it does not answer the design note's Open Q1/Q2, populate `likely_execution_platform`, or prove where an action ran. |
| `library_introspection` | `list[dict]` | bound `buck2 cquery 'deps(%s)' <target> --json` (only when `--buck-target` is supplied) | Always present. Empty `[]` outside Buck mode. In Buck mode, one entry per matched library: `{"name", "source": "buck", "revision", "target", "configured_target"}`. `target` is the canonical Buck label (stable across daemon restarts); `configured_target` preserves the raw cquery output including its per-run configuration suffix (`(prelude//platforms:default#<hash>)`) for forensics. The matched library set lives in `KNOWN_LIBRARY_PATTERNS` in `src/aorta/instrumentation/buck_introspect.py`. Added in schema 1.4 for issue #163 (A1.2b); migrated from `buck2 audit dependencies` to `buck2 cquery` and split `target` / `configured_target` in schema 1.6 (PR #187). Query binding replaced target interpolation in schema 1.13 without changing entry behavior. |
| `library_introspection_alternates` | `list[dict]` | synthesised from A1's per-library blocks when a Buck match overlaps | Always present. Empty `[]` outside Buck mode and when no Buck-matched library is also captured by A1. Each entry mirrors the unified shape with `source: "package"` and pulls `revision` / `package_version` / `lib_hash` from the matching A1 block (e.g. `hipblaslt`). Added in schema 1.4 for issue #163 (A1.2b). |
| `pytorch_sdpa` | `dict` | `torch.backends.cuda.{flash,mem_efficient,math,cudnn}_sdp_enabled()` | `backends_enabled` dict, one bool per SDPA backend + per-getter `null` when missing on older torch. Runtime state, NOT compile-time -- combine with `pytorch_build.binary_introspection.libtorch_hip_symbol_counts` for the full "compiled in AND enabled" picture. Added in schema 1.5 for issue #176 (PR #177). |

**Buck/monorepo native-lib recovery (schema 1.8).** The lib-on-disk
probes (`composable_kernel.pytorch_bundled`, `aotriton`, and
`pytorch_build.binary_introspection`) normally locate torch's native
libraries at `<torch.__file__>/../lib`. When torch is a Buck target
(e.g. `root//framework:runtime`) its Python package is materialised into a
link-tree but the C++ runtime is `dlopen`'d from a separate
build-artifact directory, so `<torch>/lib/libtorch_hip.so` does not
exist and those fields previously came back `null`. Because the probe
runs in-process with torch imported, the libraries ARE mapped into the
process: the probe now falls back to `/proc/self/maps` to recover the
real lib directory, populating the bundled-CK symbol count, the AOTriton
bundle fields, and the `libtorch_hip_symbol_counts` / `torch_lib_bundled`
facts for Buck builds. No field shapes change; the fallback is Linux-only
and silently inert elsewhere. Mappings the kernel has marked
`" (deleted)"` (the backing file was unlinked after being `dlopen`'d,
e.g. a build-artifact directory cleaned up post-load) are skipped
rather than treated as a match, since that path no longer exists on
disk.

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

### `1.16` (current)

Layout-agnostic ROCm discovery (issue #381). ROCm ships in two on-disk shapes
— a classic system install rooted at `/opt/rocm`, and TheRock's Python wheel
(ROCm 7.14+, and what `rocm/pytorch:latest` already is) which has no
`/opt/rocm` at all. Every ROCm path was hardcoded to the former, so on the
latter the probe degraded *silently*: `rocm.version` null, and the hipBLASLt
commit — core evidence in the NaN escalations — lost. Paths are now derived
from a resolved root.

One new top-level key (`therock`) and four changes to existing blocks; a 1.15
snapshot loaded through `from_dict()` gains them as their empty/`null` shape.

* **`rocm` gains attribution**: `root`, `lib_root`, `root_source`, `layout`,
  `version_source`. A null version used to be unattributable — "found an
  install with no version file" and "found no install at all" produced
  identical output. `root_source: "none"` now says which one happened.
* **`rocm.version` gains a fallback chain** past `.info/version`: the TheRock
  manifest, then pip metadata (`rocm` / `rocm-sdk-core`), then torch's
  `+rocm` local version segment. Ordered most to least authoritative, and
  `version_source` records which one answered.
* **New `therock` block** — full 40-char upstream submodule pins,
  `the_rock_commit`, `github_run_id`, and applied patches. `status:"absent"`
  on a classic install is a documented absence, not a partial.
* **`hipblaslt` / `rocblas` gain `upstream_commit`** (the full
  `rocm-libraries` pin the version header truncates) and
  `upstream_commit_matches_tweak`.

Note on the cross-check: the header tweak is a short hash of *unspecified*
width — measured 10 characters on ROCm 7.2.4, and documented upstream as
7–12 — so the comparison is a prefix match at the tweak's own length rather
than a fixed 8. Today the two sources never coexist on a real image (classic
has headers but no manifest; the runtime wheel has a manifest but no
headers), so the field is `null` in practice and exists for the devel-wheel
case and as a mis-parse guard.

Values on a classic install are otherwise unchanged: every derived path
resolves to the literal it replaced, and the Tensile fingerprint of a flat
kernel directory is byte-identical. The kernel-database scan now also reads
one level of `gfx*`-named subdirectories, which is how the wheel layout ships
Tensile (`library/gfx950/...`); nested files are recorded as `<arch>/<name>`.

### `1.15`

recom-NaN follow-up: backend / Stream-K selectors, numeric-check knobs, and
TorchRec identity fixes. `1.14` already shipped, so these changes to what a
snapshot emits get their own version rather than silently redefining it. No
new top-level keys — `env_vars` gains entries and `torchrec` gains accuracy plus
three sub-keys (`source_version`, `source_commit`, `distribution_version`); a
1.14 snapshot loaded through `from_dict()` gains them as `null`.

* GEMM env-var coverage is now decided by following each knob's upstream
  `getenv` site to a call site, rather than by reading its name, so the whole
  behaviour-changing class is captured instead of a subset. Two snapshots with
  different registered GEMM environment settings will now expose those
  *configuration* differences. This does not prove that identical snapshots
  selected the same runtime solution or kernel: the probe records declared
  configuration and never observes the backend, solution ID, kernel or launch
  geometry a run actually chose, which can still differ through device state,
  driver behaviour or library heuristics. That is the same layer-1 / layer-2
  boundary described under *"Catalog vs. runtime trace"* above.

  | Class | Added in 1.15 |
  | --- | --- |
  | Library / ExtOp loading | `HIPBLASLT_EXT_OP_LIBRARY_PATH`, `HIPBLASLT_PRELOAD_KERNELS` |
  | Backend routing | `ROCBLAS_USE_HIPBLASLT_BATCHED`, `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH` |
  | Numeric path | `ROCBLAS_INTERNAL_FORCE_VALU_FOR_DGEMM` |
  | Allocator / workspace | `ROCBLAS_DEVICE_MEMORY_SIZE`, `ROCBLAS_INTERNAL_TRSM_REG_KERNEL_MEM_LIMIT` |
  | Solution selection | `TENSILE_SOLUTION_SELECTION_METHOD`, `TENSILE_EXPERIMENTAL_SELECTION`, `TENSILE_TAM_SELECTION_ENABLE`, `TENSILE_NAIVE_SEARCH`, `TENSILE_METRIC`, `TENSILE_PREDICTION_LIB`, `TENSILE_GRIDBASED_KDTREE`, `TENSILE_GRIDBASED_BATCH_EXP` |
  | Origami / GridBased selection | `ANALYTICAL_GEMM_HEURISTICS`, `ANALYTICAL_GEMM_HEURISTICS_VARIANCE`, `GRIDBASED_TOPSOLS` |
  | Stream-K | `TENSILE_STREAMK_DYNAMIC_GRID`, `TENSILE_STREAMK_FIXED_GRID`, `TENSILE_STREAMK_MAX_CUS`, `TENSILE_STREAMK_DATA_PARALLEL`, `TENSILE_STREAMK_DYNAMIC_WGM`, `TENSILE_STREAMK_FULL_TILES`, `TENSILE_STREAMK_GRID_MULTIPLIER` |
  | Workgroup mapping / StaggerU | `TENSILE_FIXED_WGM`, `TENSILE_FIXED_WGMXCC`, `TENSILE_FIXED_WGMXCCCHUNK`, `TENSILE_DISABLE_STAGGERU`, `TENSILE_FIXED_STAGGERU`, `TENSILE_FIXED_STAGGERU_MAPPING`, `TENSILE_FIXED_STAGGERU_STRIDE_SHIFT` |
  | Skips work | `TENSILE_DB2` (bit 0 `skipKernelLaunch`, bit 1 `skipInitKernelLaunch`) |
  | Forward-compat (absent from the reference build) | `TENSILE_STREAMK5_FORCE_MODE`, `TENSILE_STREAMK_TILES`, `TENSILE_STREAMK_SPLIT` |

  Also captured, and classified `gemm_diagnostics` because the only call site
  found was a print or a client-side report: `HIPBLASLT_LOG_{FILE,LEVEL,MASK}`,
  `HIPBLASLT_BENCH_PERF` and `HIPBLASLT_BENCH_PERF_ALL` (these fill
  `hipblasltClientPerformanceArgs` and nothing else),
  `HIPBLASLT_BENCH_PRINT_COMMAND`, `HIPBLASLT_ENABLE_MARKER`,
  `TENSILE_ENABLE_MARKER`, `ROCBLAS_LAYER`, `ROCBLAS_LOG_*_PATH`,
  `ROCBLAS_VERBOSE_{HIPBLASLT,TENSILE}_ERROR`, `ANALYTICAL_GEMM_DEBUG`,
  `ORIGAMI_LOG_FILE`,
  `TENSILE_DB` (every bit it sets is a `print*`), `TENSILE_ADAPTIVE_GEMM_LOG`,
  `TENSILE_AUTO_GSU_ALGO` (each guards a lone `std::cout`),
  `TENSILE_SOLUTION_SELECTION_TRACE`, and `TENSILE_BENCHMARK`
  (`Debug::getBenchmark()` has no call site in the reference libraries at all).

  Earlier `1.15` drafts *excluded* these. They are captured now because the
  contract is comprehensive capture with classification recorded, not capture
  filtered by classification: recording a variable does not claim it affected
  execution, it preserves the declared environment. The classification survives
  in the manifest's `category` / `consumer` fields, where a triager can weigh it
  without it having decided what the snapshot preserves. The exact-string audit
  also rejects command templates such as `ROCBLAS_API_BENCH -f gemmt -r`;
  whitespace tokenization must never invent a variable from client help text.

  **Which library these statements are about.** Presence and absence above are
  stated against the *reference build* — hipBLASLt 1.4.70002 + rocBLAS
  5.0.70002 on ROCm 7.0.2, the build the escalation runs. Every ROCm release
  ships a different subset of these knobs: ROCm 7.2.3, for example, has no
  `HIPBLASLT_CHECK_NUMERICS*` and no `TENSILE_FIXED_STAGGERU*`, and it does
  have `HIPBLASLT_BENCH_PERF_ALL`. The captured list is therefore a superset
  keyed to the build under investigation, not a claim about every ROCm.

  **`null` means unset, not unsupported.** Each registered variable records its
  raw exported value, or `null` when the variable is not set in the probed
  process. `_capture_env_vars()` is `os.environ.get` over the manifest and
  nothing more, so this block does not determine whether the installed library
  supports the variable, consumed it, or was affected by it — an exported knob is
  recorded verbatim even when the local library has never heard of it, which is
  what keeps the list portable across ROCm releases. Nothing here contributes a
  `partial_reasons` entry.
* The `HIPBLASLT_CHECK_NUMERICS` family (`_SCAN_EVERY` / `_SCAN_FROM` /
  `_SCAN_UNTIL` / `_STOP_ON_FIRST`) and `ROCBLAS_CHECK_NUMERICS` are now
  **captured**. 1.14 wrongly classed them as behaviour-neutral logging:
  in-library numeric checking launches extra scan kernels and adds
  synchronization, which can hide or expose a timing race even when the final
  arithmetic is unchanged. The rocBLAS variant is the load-bearing one on the
  stock 1.4.0 image, where the repro's GEMMs fall back to rocBLAS — capturing
  only the hipBLASLt half would capture the half that is not running.
* **`torchrec` records two identities separately.** `find_spec` and
  `importlib.metadata` answer different questions, and in a Buck /
  PYTHONPATH-over-pip process they can resolve to different installs — so the
  snapshot no longer merges them:
  * `source_version` / `source_commit` — the code Python would actually import,
    read from `version.py` inside the resolved import target.
  * `distribution_version` — what a `*.dist-info` on `sys.path` claims, which may
    belong to a different copy.
  * `package_version` / `commit` stay the primary identity for consumers wanting
    one answer, and follow the **import target**, because a snapshot should
    describe what would run. `commit` therefore always comes from the same
    install as the version beside it, never from the other one's metadata.

  A disagreement is reported in `partial_reasons` rather than reconciled
  silently — two torchrecs on one path is a fact about the process. Versions are
  compared after PEP 440 normalisation, so `1.4.0-1` and `1.4.0.post1` are not
  treated as a conflict; and because an unparseable version falls back to
  "different", an incomplete normaliser can only add a reason, never discard an
  identity.

  `version.py` is read **through the loader**, so a Buck `.par` (zipimport)
  resolves its identity instead of reporting `null` while claiming the file was
  unreadable — that deployment is the one this feature exists for. Only the
  package's own `submodule_search_locations` are consulted, so a single-module
  `torchrec.py` no longer adopts an unrelated sibling `version.py` and publish
  another project's SHA as torchrec's. Neither `find_spec` nor a metadata read
  failing is folded into "absent": each records its own reason, and a readable
  `version.py` identity survives a metadata failure.
* `torchrec.commit` prefers the **full 40-char** build SHA read as text from
  the installed `torchrec/version.py` (`git_version`, written by TorchRec's own
  `setup.py`). A release wheel such as `1.4.0` carries no SHA in its version
  string, so 1.14 reported `null` even though the SHA was on disk — which made
  two different builds of `1.4.0` indistinguishable in a diff. The file is
  parsed as text, never imported, so the no-`torch.cuda` contract holds.
  The version-string parse remains the fallback and now also recognises
  TorchRec's source-build **bare-hex** local segment (`1.8.0a0+0123abc` ->
  `0123abc`), guarded against setuptools_scm's dirty-tree marker
  `+d<YYYYMMDD>` (every character of which is valid hex) so the build *date*
  is never published as a commit SHA.
* A torchrec that is **importable but has no dist-info** (Buck / PYTHONPATH
  link-tree) is resolved from `version.py` via `importlib.util.find_spec`
  instead of being reported absent; only an install with neither dist-info nor
  `version.py` records the present-but-version-unknown `partial_reasons`
  entry. A PEP 420 **namespace** portion — a bare `torchrec/` directory
  anywhere on `sys.path` — is loader-less and counts as absent, so it can no
  longer flip a clean snapshot to `partial`.
* `summary()` now prints a `torchrec:` line, matching its sibling package
  blocks.

### `1.14`

Recsys-stack package identity + recom-NaN environment knobs. Additive — one
new top-level block plus new `env_vars` entries; older snapshots and direct
constructors remain compatible.

* New top-level **`torchrec`** block `{package_version, commit}`, defaulted so
  older snapshots round-trip. TorchRec is the sharded-embedding / training-
  pipeline library layered on the already-captured fbgemm and is the core
  library of the MI350X recom-repro workload, so its version is load-bearing
  for that escalation. Same shape and fail-soft contract as `triton`.
* Expanded **`CANONICAL_ENV_VARS`** with the hipBLASLt / rocBLAS / Tensile GEMM
  numeric + kernel-path knobs that flip *which* library, *which* kernels, and
  (for xf32/TF32) the numeric path itself run the GEMMs:
  `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32`, `ROCBLAS_USE_HIPBLASLT`,
  `HIPBLASLT_TENSILE_LIBPATH`, `ROCBLAS_TENSILE_LIBPATH`,
  `HIPBLASLT_USE_ROCROLLER` (+ `HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL`),
  `HIPBLASLT_TUNING_OVERRIDE_FILE`, `ROCBLAS_DEFAULT_ATOMICS_MODE`,
  `ROCBLAS_INTERNAL_FP16_ALT_IMPL` (+ `_RNZ`), `ROCBLAS_STREAM_ORDER_ALLOC`,
  and `TENSILE_SOLUTION_INDEX`; plus `HSA_TOOLS_DISABLE_REGISTER` and
  `LD_LIBRARY_PATH` (which `.so` actually loads).

  Note: 1.14 also claimed the `HIPBLASLT_CHECK_NUMERICS*` knobs were excluded
  as behaviour-neutral. That was wrong and is corrected in 1.15, which
  captures them.

### `1.13`

Buck configured-graph invocation provenance. Additive — one new top-level
block, defaulted so older snapshots and direct constructors remain compatible.

* New top-level **`buck_invocation`** block. Its `status` distinguishes no
  request, Buck not detected, cquery success, and cquery failure. It records
  the requested target, context source (`none` / `unspecified` /
  `default_confirmed` / `explicit`), ordered mode files, ordered config
  **keys only**, ordered modifiers, cross-option `option_order`, an aggregate
  `sha256:` fingerprint over the full ordered context values, the configured
  root target when available, and `comparison: "not_compared"`.
* New repeatable CLI option `--buck-option TYPE=VALUE` preserves exact
  cross-option order for `mode`, `config`, and `modifier` inputs. Grouped
  convenience options remain available when the normal mode / `-c` / `-m`
  order is sufficient. Context options require `--buck-target` and cannot be
  mixed with `--buck-default-context`. There is no shell or arbitrary
  passthrough string.
* Cquery now binds the target through `deps(%s)` plus a separate target argv
  entry. Target text is never interpolated into Buck query syntax.
* `--buck-target` without explicit context or `--buck-default-context` still
  runs for backwards compatibility, but records `context_source:
  "unspecified"` and a clear partial reason. A request when
  `build_system.kind != "buck2"` is similarly partial and records
  `status: "buck_not_detected"`.
* Raw Buck config values are never persisted. The key is visible and the
  full value participates only in the aggregate fingerprint, so value drift
  remains detectable without disclosure.
* Scope remains deliberately narrow: this block describes the client-side
  configured-graph query. It does **not** answer execution-context Open Q1 or
  Q2, populate `execution_context.likely_execution_platform`, or prove actual
  local/remote execution placement.

### `1.12`

Container & execution-context visibility, phase 2 (namespace). Additive —
one new top-level field, defaulted so older readers don't raise. See the
design note `docs/env-probe-container-execution-context.md`.

* New top-level **`probe_namespace`** (`str | null`). A mismatch-only
  namespace observation derived from `/proc/self/ns/mnt`, or
  `/proc/self/ns/cgroup` as a fallback, hashed with the per-boot `boot_id`
  (`/proc/sys/kernel/random/boot_id`) → `mnt:<hash[:16]>` /
  `cgroup-ns:<hash[:16]>`. Different same-kind values prove different
  observations; equality is advisory because Linux can recycle namespace
  inode numbers after teardown. When `boot_id` is unavailable the token is
  still hashed and emitted as `mnt-local:` / `cgroup-ns-local:`; it is not
  cross-host comparable. Missing boot identity, fallback use, and total
  failure add partial reasons. `null` means neither namespace handle was
  readable. Defaulted `null` + `from_dict()` backfill so pre-1.12 snapshots
  round-trip.

### `1.11`

Container & execution-context visibility, phase 1. Additive — two new
top-level fields, both defaulted so older readers don't raise. See the
design note `docs/env-probe-container-execution-context.md`.

* New top-level **`container_detected`** (`bool`). A runtime-**agnostic**
  isolation smoke test: `true` on any generic sandbox signal (a named
  runtime match, a private mount namespace via `/proc/self/ns/mnt` vs
  `/proc/1/ns/mnt`, or a container/k8s token in `/proc/self/cgroup`).
  Fixes the long-standing false negative where an RE worker or containerd
  k8s pod — which have none of the `docker`/`podman`/`singularity` markers
  `runtime_context.type` knows — reported as `"baremetal"`, the opposite of
  the truth. `container_detected:true` + `type:"baremetal"` is the honest
  reading of an unnamed sandbox. Fail-soft; defaulted `false`.
* New top-level **`execution_context`** block: `probe_invocation`
  (`"direct"`/`"buck2_run"`/`"buck2_action"`, **self-declared** via the new
  `aorta env probe --execution-context` flag; `"direct"` by default) and
  `likely_execution_platform` (`null` today; reserved for phase-2 Buck2
  work). Defaulted via `_empty_execution_context()`.
* New CLI flag **`--execution-context`** (also accepted by
  `python -m aorta.instrumentation._probe_main`). Captures nothing new — it
  labels the snapshot and **warns to stderr** when a non-`direct` context
  is claimed but `container_detected` is `false` and neither
  `$AORTA_RE_IMAGE` nor `$AORTA_DOCKER_IMAGE` is set (the "you probed the
  host shell, not where the job ran" guardrail). Never a hard error.
* **Remaining (phase 2, Buck2 / remote-execution):**
  `likely_execution_platform` population and the `$AORTA_RE_IMAGE` launcher
  convention are deferred pending on-cluster empirics — see the design
  note's Open Questions. (The `probe_namespace` observation landed in schema
  1.12, above.)

### `1.10`

Host/container runtime split, part 1: KFD/AMDGPU kernel-mode-driver
identity. Additive — one new top-level block, defaulted so older readers
don't raise.

* New top-level **`amdgpu_driver`** block, always present (defaulted via
  `_empty_amdgpu_driver()`). **Host-kernel scope — three fields only**:
  `kfd_device_present`, `kfd_sysfs_present`, and `kmd_version` reflect the
  host kernel a container shares, so they report the *host's* driver even
  when the probe runs inside a container — the deliberate counterpart to
  `rocm`/`hip`, which read the container's `/opt/rocm` userspace. This is
  what lets a single in-process probe separate a "host runtime" signal
  from a "container runtime" signal without a second invocation or
  privileged access.
* **Not host-guaranteed**: `package_name`/`package_version`/
  `package_manager` (dpkg/rpm) and `module_version`/`module_srcversion`
  (`modinfo`) read the *current filesystem's* package database and
  `/lib/modules`. From inside a container these reflect the container's
  own view — typically null, since the amdgpu driver package is normally
  installed only on the host — even while `kfd_device_present` and
  `kmd_version` correctly show the host driver as present. Treat these
  two field groups as filesystem-scoped, not host-scoped.
* Fields: `scope` (constant `"host_kernel"` — describes the block's
  intent, not a per-field guarantee; see the two bullets above), `status`
  (`"present"`/`"absent"`), `package_name` (stable candidate family) /
  `package_version` / `package_full_name` / `package_manager` (portable,
  **glob-capable dpkg-then-rpm** query over `amdgpu-dkms` then
  `amdgpu-kmod`, parsed in Python — no shell pipeline), `module_version`
  / `module_srcversion` (`modinfo amdgpu`), `kmd_version` (reused verbatim
  from `rocm.kmd_version` so the two blocks never disagree),
  `kfd_device_present` / `kfd_sysfs_present` (`/dev/kfd` + `/sys/class/kfd`
  existence checks — no `open()`, no `ioctl`, no GPU compute).
* **`package_full_name`** is the complete canonical package identity — apt
  `name=version` on Debian, or an rpm NVRA such as
  `amdgpu-kmod-<kernel>-<driver-version>.<arch>` on
  RHEL/SLES. Some vendors bake the kernel release into the RPM *package
  name*, which an exact `rpm -q amdgpu-kmod` misses entirely; the query
  uses a glob (`rpm -qa 'amdgpu-kmod*'`) to catch these and reconstructs
  the full identity so two host snapshots can be diffed on one field.
  `package_name` stays the stable family label (`amdgpu-dkms`/`amdgpu-kmod`)
  so it remains a clean grouping key across hosts with different suffixes.
* **Documented absence, not partial**: a GPU-less host records
  `status:"absent"` with an empty `partial_reasons`, mirroring the `nics`
  vendor-absent precedent. The only reason this block ever appends is a
  *same-filesystem conflict*: the amdgpu module is demonstrably loaded
  (`modinfo` metadata present) yet no dpkg/rpm package on the same
  filesystem can be named — an unusual, diagnosable state (out-of-band
  driver install, stripped package DB). `/dev/kfd` alone never triggers a
  reason: it is host-kernel state passed into a container, so a container
  with the host's KFD node but no container-local package is the normal
  case, not a conflict.
* Backwards-compat: `EnvSnapshot.amdgpu_driver` uses a `default_factory`,
  so a ≤1.9 snapshot round-trips via `from_dict()` (back-filled with the
  empty/absent block). A 1.9 reader indexing `amdgpu_driver` on a pre-1.10
  snapshot gets that back-filled block, not a `KeyError`.
* **Compact catalog default + `--extended`.** The default `aorta env probe`
  now drops the per-file `menu.files` lists from `tensile_catalog` and
  `miopen_catalog` (setting them to `null`) — these dominated env.json
  size (~25k lines on a populated host). Every summary field is kept:
  `status`, `dir`, `file_count`, `logic_file_count`, `gfx_arch_coverage`,
  and `combined_content_hash`, so a two-host diff still detects *that* a
  catalog changed. `aorta env probe --extended` (or `collect_env(detail=
  "full")`) restores the complete per-file lists for forensic follow-up
  (localizing *which* recipe/db file changed). The files are hashed either
  way — `combined_content_hash` requires it — so `--extended` only controls
  retention of the per-file breakdown, not probe work. This is a shape
  change to the *default output*, not a schema-field removal: the `files`
  key is still present (as `null`), so readers indexing it are unaffected.
* **Thematic output order.** `env.json` keys (and the CLI brief lines) are
  now grouped by theme — host/kernel, then ROCm runtime + `amdgpu_driver`
  driver, then GPU/fabric hardware, then compute libraries, then build,
  then pytorch — instead of dataclass declaration order, so related
  environment facts sit next to each other for readability and diffing.
  `partial` / `partial_reasons` moved to the **end** of the object (a
  probe-status trailer) so the potentially long reasons list doesn't push
  the identity blocks down. Key *set* is unchanged; only ordering moved.

### `1.9`

Static kernel-catalog "recipe books" for the on-disk, runtime-selected
GPU-compute catalogs (issue #54 + follow-ups). All additive: three new
top-level blocks, each defaulted so older readers don't raise.

* New top-level **`tensile_catalog`** (issue #54): installed-library
  identity for the hipBLASLt/rocBLAS Tensile install. Reuses the existing
  `hipblaslt`/`rocblas`/`tensile` captures (no regression) and deepens
  them with a per-file `menu` enumeration of the frozen solution library
  (`TensileLibrary` / `*.dat` / `*.yaml` / `*.co` / `*.hsaco`): per-file
  content sha256 + size, `logic_file_count`, `gfx_arch_coverage`, and a
  `combined_content_hash`, so a two-host diff localizes *which* logic
  file differs. Explicitly NOT the runtime-selected solution ID
  (`381881`/`368xxx`). Partial-not-silent per menu.
* New top-level **`miopen_catalog`** (issue #54 follow-up): the same
  treatment for MIOpen's databases under `/opt/rocm/share/miopen/db/`
  (`.kdb` kernel-db, `.fdb.txt` find-db, `.db.txt`/`.db` perf-db, plus
  `.ktn.model`/`.tn.model` heuristic nets). Honors `MIOPEN_SYSTEM_DB_PATH`
  (records `db_dir`/`db_dir_source`) and surfaces `MIOPEN_USER_DB_PATH`.
* New top-level **`rocfft_catalog`** (issue #54 follow-up): fingerprints
  rocFFT's *optional* AOT `rocfft_kernel_cache.db` when present; absence
  is the common case (`status:"absent"`, no partial reason). Resolves
  against the read-only `ROCFFT_RTC_SYS_CACHE_PATH` system-cache override
  (or the default `/opt/rocm/lib[/rocfft]/` locations) -- NOT the
  mutable, per-process `ROCFFT_RTC_CACHE_PATH` user cache, which is
  recorded under `env_overrides` for visibility only.

Backwards-compat: all three are new top-level dataclass fields with
default factories, so a 1.7/1.8 reader loading a 1.9 snapshot drops the
unknown keys and a 1.9 reader loading an older snapshot gets the empty
defaults.

### `1.8`

Buck-target torch native-lib recovery + best-effort package commits.
Additive nested keys only (`triton.commit`, `fbgemm.commit`); no
top-level keys or dataclass fields change, so `EnvSnapshot.from_dict`
still round-trips pre-1.8 snapshots.

* Added a `commit` field under the `triton` and `fbgemm` blocks --
  best-effort git SHA parsed from a setuptools_scm `+g<sha>`
  local-version segment, a ROCm/fb fork `.git<sha>` segment, or a
  `git_version`/`__commit__` module attribute that is itself a valid
  7-40 char hex SHA. `null` when no SHA is recoverable (e.g. fb wheels
  versioned `3.5.0+fb`, or `fbgemm_gpu` not separately installed).
  Mirrors the existing `aiter.commit` field.
* Behavioural (not schema-shape) change: the `libtorch_hip.so`-dependent
  probes (`composable_kernel.pytorch_bundled` symbol count, `aotriton.*`,
  `pytorch_build.binary_introspection` symbol counts / `torch_lib_bundled`)
  now locate torch's native lib dir via `/proc/self/maps` when
  `<torch>/lib` is absent, populating those fields for Buck/monorepo
  torch targets (e.g. `root//framework:runtime`) whose C++ runtime is
  `dlopen`'d from a build-artifact dir.

### `1.7`

RCCL net-plugin identity + multi-vendor NIC/RoCE fabric capture
(issue #202). Both additive; the new top-level `nics` block is what
drives the version bump.

* New top-level **`nics`** block keyed by vendor (`ainic`, `broadcom`,
  `cx7`). Each vendor has a Tier-0 `present` gate (`lspci -d
  <vendor>:<device>`); when present, Tier-1 sudo-free fields
  `driver_version` (sysfs `/sys/module/<drv>/version`, falling back to
  `ethtool -i` `version:` -- the sysfs file does not exist for in-tree
  `mlx5_core`/`bnxt_en` on modern kernels), `firmware` (`ethtool -i`
  `firmware-version:`, with Broadcom's `<fw>/pkg <pkg>` form split into
  `firmware` + `pkg_version`), `rdma_devices` (`ibv_devices`), and `links`
  (`rdma link` -> `[{device, state, netdev}]`). RDMA devices/links are
  bound to a vendor by their kernel driver (sysfs `device/driver`
  symlink), not by device-name prefix, since names vary by host
  (`ionic_0` vs `rdma3`). AINIC additionally gets
  Tier-2 `nicctl` fields (`nicctl_version`, `card`, `profile`, `dcqcn`)
  via `sudo -n`. **Documented absence**: a vendor not in `lspci` is
  `{"present": false}` with no `partial`; a present vendor with zero RDMA
  devices (observed on CX7) is valid, not partial. Only an
  expected-but-failed capture (tool times out, sudo denied) records a
  reason. `EnvSnapshot.nics` uses `field(default_factory=dict)` so a
  <=1.6 snapshot round-trips via `from_dict()`.
* `rccl` gained five nested keys: `net_plugin_mode`
  (`"external"`/`"internal"`/`"unknown"`), `plugin_path`,
  `plugin_lib_hash`, `anp_lib_hash`, and `net_lib_hash`. The
  authoritative signal is `plugin_path`/`plugin_lib_hash`: the probe
  resolves `NCCL_NET_PLUGIN` to a real `.so` (absolute path or bare name
  on `LD_LIBRARY_PATH`/lib dir) and hashes it -- matching how real
  AMD-ANP ships the plugin (`librccl-net.so` in a user-build tree).
  `anp_lib_hash`/`net_lib_hash` remain a best-effort lib-dir scan for
  packaged installs. All reuse `_hash_shared_library` (no new hashing
  logic). `net_plugin_mode` is derived (see the field table); the
  `"unknown"` case (env set but unresolvable) records a `partial_reason`,
  every other case is silent. These are nested keys under the existing
  `rccl` dict, so a reader loading an older snapshot must guard with
  `.get(...)`.
* Buck: `KNOWN_LIBRARY_PATTERNS` gained an `"ainic"` key matching
  `:rccl-anp(-lib)` / `:rccl-net(-lib)` targets, so the ANP/net plugin
  surfaces in `library_introspection` like `rccl`.

### `1.6`

Additive change to `library_introspection` (PR #187, issue #183):

* Each `library_introspection[*]` entry now carries two Buck-label
  fields instead of one. `target` is the canonical Buck label with
  the cquery configuration suffix stripped -- stable across daemon
  restarts and the form that round-trips into another
  `buck2 query` / `buck2 build`. `configured_target` preserves the
  raw `buck2 cquery` output including its per-run configuration
  suffix (`(prelude//platforms:default#<hash>)`) for forensics when
  reconciling two probes that diverged on the same source tree.

Bundled with the cquery migration: A1.2b's original implementation
called `buck2 audit dependencies --transitive --json`, which was
removed from open-source buck2 before A1.2b ever ran end-to-end.
PR #187 swapped it for `buck2 cquery 'deps(<target>)' --json`
(buck2 docs' recommended replacement; same configured-graph
semantics as the deprecated `audit dependencies --transitive`).
This is a runtime-behaviour change, not a schema change -- the
emitted `library_introspection` entries take the same shape
either way, but the new `configured_target` field is what motivated
the schema bump.

Backwards-compat:

* 1.5 readers loading a 1.6 snapshot see `configured_target` as an
  unknown nested key inside each entry and silently ignore it
  (entries are plain dicts, not dataclasses, so `from_dict` doesn't
  trip on the new key).
* 1.6 readers loading a 1.5 snapshot get entries without
  `configured_target` and must guard with `.get("configured_target")`
  if they want it.

### `1.5`

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

### `*_catalog` blocks are the recipe book (installed library), not the recipe used

The `tensile_catalog` / `miopen_catalog` / `rocfft_catalog` blocks answer
exactly one question: *is the installed kernel/solution catalog the same
on both hosts?* Every field is **installed-library identity** -- the
frozen on-disk menu. They deliberately do **not** capture the
runtime-selected solution/solver ID (`#381881` / `#368xxx`), which is
chosen per-call at runtime against live device properties and needs a
workload trace (a separate capability).

Two hosts can report the **same** hipBLASLt commit (e.g. `7e32d53eb1`)
and still ship a **different** Tensile install -- a vendor build can
patch or select different kernels even when nominally pointing at that
commit. That is precisely the case `tensile_catalog.<lib>.menu` is built
to expose: the shallow filename fingerprint (`kernel_db_revision`) can
match while the per-file `menu.files[*].sha256` differ. Use the menu's
`combined_content_hash` for a one-shot "are the menus identical?" check,
and the per-file `files` list to localize *which* logic file changed. The
same applies to `miopen_catalog` for convolution databases.

### `composable_kernel.system.commit` is a real upstream commit

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
   `"hipblaslt.rocm_release_tweak:
   /opt/rocm/include/hipblaslt/hipblaslt-version.h not readable"` is
   appended to `partial_reasons`.
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
| `amdgpu_driver.package_version` / `.package_name` / `.package_manager` | `dpkg-query -W -f='${Version}' amdgpu-dkms` (Debian/Ubuntu), falling back to `rpm -q --qf '%{VERSION}-%{RELEASE}' amdgpu-dkms` then `amdgpu-kmod` (RHEL/SLES). First hit wins; `null` when no package manager names the driver. |
| `amdgpu_driver.module_version` / `.module_srcversion` | `modinfo amdgpu` `version:` / `srcversion:` fields (module metadata; does not load the module or touch the GPU). `srcversion` is a source hash — it changes on a rebuild even when `version` doesn't. |
| `amdgpu_driver.kmd_version` | reused verbatim from `rocm.kmd_version` (`/sys/module/amdgpu/version`) — no second read, so the two blocks can never disagree |
| `amdgpu_driver.kfd_device_present` / `.kfd_sysfs_present` | existence of `/dev/kfd` / `/sys/class/kfd` (pure `Path.exists()`, never opened) |
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
| `rccl.plugin_path` / `rccl.plugin_lib_hash` | `NCCL_NET_PLUGIN` resolved to a real regular file (absolute path, or bare name searched on `LD_LIBRARY_PATH` then `/opt/rocm/lib`) and `sha256`'d through symlinks. Both `null` when `NCCL_NET_PLUGIN` is unset or unresolvable. When it resolves but the file can't be read to hash, `plugin_path` is still populated while `plugin_lib_hash` is `null` (and a `partial_reason` is recorded). |
| `rccl.anp_lib_hash` / `rccl.net_lib_hash` | best-effort `sha256(/opt/rocm/lib/librccl-anp.so)` / `sha256(/opt/rocm/lib/librccl-net.so)` for packaged installs; `null` when absent (documented absence, no `partial`) |
| `rccl.net_plugin_mode` | derived: `"external"` when `NCCL_NET_PLUGIN` resolves to a real `.so`; `"internal"` when unset/empty; `"unknown"` when set but unresolvable (records a `partial_reason`) |
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
| `tensile_catalog.<lib>.{package_version,lib_hash,kernel_db_revision}` | taken verbatim from the matching `hipblaslt`/`rocblas` block (no regression) |
| `tensile_catalog.<lib>.menu.files[*].sha256` | `sha256` of each `*.dat`/`*.yaml`/`*.co`/`*.hsaco`/`*.txt` file under `/opt/rocm/lib/<lib>/library/` (content hash, not just filename) |
| `tensile_catalog.<lib>.menu.gfx_arch_coverage` | sorted unique `gfx<arch>` tokens parsed from those filenames (e.g. `TensileLibrary_lazy_gfx942.dat` -> `gfx942`) |
| `tensile_catalog.<lib>.menu.combined_content_hash` | sha256 over the sorted `(filename, content-sha256)` pairs -- one value to eyeball when diffing whole menus |
| `miopen_catalog.{package_version,lib_hash,kernel_db_revision}` | taken verbatim from the `miopen` block (no regression) |
| `miopen_catalog.db_dir` / `.db_dir_source` | the effective MIOpen db dir enumerated -- `$MIOPEN_SYSTEM_DB_PATH` when set (`"MIOPEN_SYSTEM_DB_PATH"`), else `/opt/rocm/share/miopen/db/` (`"default"`) |
| `miopen_catalog.menu.files[*].sha256` | `sha256` of each `.kdb`/`.fdb.txt`/`.db.txt`/`.db`/`.ktn.model`/`.tn.model` file in the db dir (content hash) |
| `miopen_catalog.menu.logic_file_count` | count of the selectable DBs (`.kdb`/`.fdb.txt`/`.db.txt`/`.db`); excludes the `*.model` heuristic nets |
| `rocfft_catalog.kernel_cache.{present,path,source,size,sha256}` | the optional `rocfft_kernel_cache.db` found under `$ROCFFT_RTC_SYS_CACHE_PATH` (`"ROCFFT_RTC_SYS_CACHE_PATH"`) or `/opt/rocm/lib/[rocfft/]` (`"rocm_lib"`); all `null` + `status:"absent"` when no cache exists (the common case). Does **not** resolve against `$ROCFFT_RTC_CACHE_PATH` -- that names the mutable, per-process runtime user cache, not installed identity. |
| `rocfft_catalog.env_overrides.{ROCFFT_RTC_SYS_CACHE_PATH,ROCFFT_RTC_CACHE_PATH}` | the raw values of both vars (or `null`), recorded even when no cache file is found so a configured-but-empty override is diffable. Only `ROCFFT_RTC_SYS_CACHE_PATH` affects `kernel_cache` resolution above; `ROCFFT_RTC_CACHE_PATH` is recorded for visibility only. |
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
Short version: add one `EnvironmentKnob` to `ENV_KNOB_REGISTRY` in
`env_knobs.py` — `CANONICAL_ENV_VARS` is generated from it — then update
`test_canonical_var_names_stable` and document why in your PR.

## Auditing what the manifest covers

`CANONICAL_ENV_VARS` is generated from `ENV_KNOB_REGISTRY`, a manifest that
records, per variable: `name`, `library`, `consumer`, `category`,
`source_reference` and `reference_build`. Entries assert what is known about who
reads a variable. They deliberately do **not** assert that the installed library
supports it, that the process exported it, or that it changed a run.

### The captured inventory

Generated from the manifest, so it is the set the probe actually reads rather than a
second copy of it. Regenerate with `python scripts/audit_env_knobs.py
--emit-docs-table`; `test_docs_knob_inventory_matches_registry` fails if this block and
the manifest disagree. (The `1.15` changelog table above is a different thing on
purpose: it records what that release *changed*, which is history and does not
regenerate.)

<!-- BEGIN GENERATED: env-knob-inventory -->

137 knobs, generated from `ENV_KNOB_REGISTRY` by
`python scripts/audit_env_knobs.py --emit-docs-table`. `library` is the component the variable belongs to. For GEMM knobs present in the reference build, ownership is measured from the shipped shared object's exact string table -- which shows the name, not a call site, so it is ownership and not proof of consumption; forward-compatible absent entries are marked in the manifest and are declared rather than measured. A row's presence records that the variable is preserved in a snapshot -- not that the installed library supports it, that the process exported it, or that it affected a run.

| Category | Library | Variables |
| --- | --- | --- |
| `gpu_scoping` | hip-runtime | `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` |
| `runtime` | hsa-runtime | `HSA_XNACK`, `HSA_KERNARG_POOL_SIZE`, `HSA_NO_SCRATCH_RECLAIM`, `HSA_OVERRIDE_GFX_VERSION`, `HSA_TOOLS_DISABLE_REGISTER` |
| `runtime` | hip-runtime | `GPU_MAX_HW_QUEUES`, `HIP_LAUNCH_BLOCKING` |
| `codegen` | compiler | `AMDGCN_USE_BUFFER_OPS` |
| `gemm_numeric` | pytorch | `DISABLE_TF32` |
| `codegen` | pytorch | `PYTORCH_ROCM_ARCH` |
| `collectives` | rccl | `NCCL_MAX_NCHANNELS`, `NCCL_P2P_LEVEL`, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `RCCL_MSCCL_ENABLE` |
| `fabric` | rccl | `RCCL_AINIC_ROCE`, `NCCL_NET_PLUGIN`, `NCCL_NET`, `RCCL_CTS_OFFLOAD_ENABLED`, `NCCL_IB_GID_INDEX`, `NCCL_IB_ROCE_VERSION_NUM`, `NCCL_IB_TC`, `NCCL_IB_FIFO_TC`, `NCCL_GDR_FLUSH_DISABLE`, `NCCL_GDRCOPY_ENABLE`, `NCCL_IB_USE_INLINE`, `NCCL_IB_PCI_RELAXED_ORDERING`, `NCCL_IB_QPS_PER_CONNECTION`, `NCCL_PXN_DISABLE`, `NCCL_IGNORE_CPU_AFFINITY`, `NCCL_NET_OPTIONAL_RECV_COMPLETION`, `RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING`, `NCCL_IB_TIMEOUT`, `NCCL_IB_SL`, `NCCL_IB_SPLIT_DATA_ON_QPS`, `NCCL_DMABUF_ENABLE`, `NCCL_CUMEM_ENABLE`, `IONIC_LOCKFREE`, `RCCL_DISABLE_RAIL_TREES`, `RCCL_LL128_FORCE_ENABLE`, `NCCL_WORK_FIFO_BYTES`, `RCCL_GFX9_CHEAP_FENCE_OFF` |
| `embedding_backend` | fbgemm | `FBGEMM_NO_JK`, `FBGEMM_TBE_V2`, `FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL`, `FBGEMM_BOUNDS_CHECK_INDICES_V2` |
| `conv_backend` | miopen | `MIOPEN_SYSTEM_DB_PATH`, `MIOPEN_USER_DB_PATH`, `MIOPEN_DEBUG_DISABLE_FIND_DB`, `MIOPEN_FIND_MODE` |
| `attention_backend` | pytorch | `TORCH_ROCM_FA_PREFER_CK`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` |
| `gemm_routing` | pytorch | `TORCH_BLAS_PREFER_HIPBLASLT` |
| `gemm_solution_selection` | pytorch | `TORCH_HIPBLASLT_TUNING_FILE`, `TORCH_HIPBLASLT_TUNING_OVERRIDE_FILE` |
| `framework` | pytorch | `TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE`, `PYTORCH_CUDA_ALLOC_CONF` |
| `loader` | dynamic-loader | `LD_LIBRARY_PATH` |
| `gemm_loading` | hipblaslt | `HIPBLASLT_TENSILE_LIBPATH`, `HIPBLASLT_EXT_OP_LIBRARY_PATH`, `HIPBLASLT_PRELOAD_KERNELS` |
| `gemm_loading` | rocblas | `ROCBLAS_TENSILE_LIBPATH` |
| `gemm_solution_selection` | rocblas | `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH`, `TENSILE_EXPERIMENTAL_SELECTION`, `TENSILE_TAM_SELECTION_ENABLE` |
| `gemm_routing` | rocblas | `ROCBLAS_USE_HIPBLASLT`, `ROCBLAS_USE_HIPBLASLT_BATCHED` |
| `gemm_routing` | hipblaslt | `HIPBLASLT_USE_ROCROLLER`, `HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL` |
| `gemm_solution_selection` | hipblaslt | `HIPBLASLT_TUNING_OVERRIDE_FILE`, `TENSILE_SOLUTION_SELECTION_METHOD`, `TENSILE_PREDICTION_LIB`, `TENSILE_GRIDBASED_KDTREE`, `TENSILE_GRIDBASED_BATCH_EXP`, `ANALYTICAL_GEMM_HEURISTICS`, `ANALYTICAL_GEMM_HEURISTICS_VARIANCE`, `GRIDBASED_TOPSOLS` |
| `gemm_numeric` | hipblaslt | `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32` |
| `gemm_numeric` | rocblas | `ROCBLAS_DEFAULT_ATOMICS_MODE`, `ROCBLAS_INTERNAL_FP16_ALT_IMPL`, `ROCBLAS_INTERNAL_FP16_ALT_IMPL_RNZ`, `ROCBLAS_INTERNAL_FORCE_VALU_FOR_DGEMM` |
| `gemm_workspace` | rocblas | `ROCBLAS_STREAM_ORDER_ALLOC`, `ROCBLAS_DEVICE_MEMORY_SIZE`, `ROCBLAS_INTERNAL_TRSM_REG_KERNEL_MEM_LIMIT` |
| `gemm_solution_selection` | hipblaslt+rocblas | `TENSILE_SOLUTION_INDEX`, `TENSILE_NAIVE_SEARCH`, `TENSILE_METRIC` |
| `gemm_launch_geometry` | hipblaslt+rocblas | `TENSILE_STREAMK_DYNAMIC_GRID`, `TENSILE_STREAMK_FIXED_GRID`, `TENSILE_STREAMK_MAX_CUS`, `TENSILE_STREAMK_FULL_TILES`, `TENSILE_STREAMK_GRID_MULTIPLIER` |
| `gemm_launch_geometry` | hipblaslt | `TENSILE_STREAMK_DATA_PARALLEL`, `TENSILE_STREAMK_DYNAMIC_WGM`, `TENSILE_FIXED_WGM`, `TENSILE_FIXED_WGMXCC`, `TENSILE_FIXED_WGMXCCCHUNK`, `TENSILE_DISABLE_STAGGERU`, `TENSILE_FIXED_STAGGERU`, `TENSILE_FIXED_STAGGERU_MAPPING`, `TENSILE_FIXED_STAGGERU_STRIDE_SHIFT` |
| `gemm_skip_work` | hipblaslt+rocblas | `TENSILE_DB2` |
| `gemm_forward_compat` | hipblaslt | `TENSILE_STREAMK5_FORCE_MODE`, `TENSILE_STREAMK_TILES`, `TENSILE_ADAPTIVE_GEMM_NTAB_ALGO`, `TENSILE_STREAMK_SPLIT` |
| `gemm_numeric_check` | hipblaslt | `HIPBLASLT_CHECK_NUMERICS`, `HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY`, `HIPBLASLT_CHECK_NUMERICS_SCAN_FROM`, `HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL`, `HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST` |
| `gemm_numeric_check` | rocblas | `ROCBLAS_CHECK_NUMERICS` |
| `gemm_diagnostics` | hipblaslt | `ANALYTICAL_GEMM_DEBUG`, `ORIGAMI_LOG_FILE`, `HIPBLASLT_LOG_FILE`, `HIPBLASLT_LOG_LEVEL`, `HIPBLASLT_LOG_MASK`, `HIPBLASLT_BENCH_PERF`, `HIPBLASLT_BENCH_PERF_ALL`, `HIPBLASLT_BENCH_PRINT_COMMAND`, `HIPBLASLT_ENABLE_MARKER`, `TENSILE_ENABLE_MARKER`, `TENSILE_ADAPTIVE_GEMM_LOG`, `TENSILE_AUTO_GSU_ALGO`, `TENSILE_BENCHMARK` |
| `gemm_diagnostics` | rocblas | `ROCBLAS_LAYER`, `ROCBLAS_LOG_PATH`, `ROCBLAS_LOG_TRACE_PATH`, `ROCBLAS_LOG_BENCH_PATH`, `ROCBLAS_LOG_PROFILE_PATH`, `ROCBLAS_VERBOSE_HIPBLASLT_ERROR`, `ROCBLAS_VERBOSE_TENSILE_ERROR`, `TENSILE_SOLUTION_SELECTION_TRACE` |
| `gemm_diagnostics` | hipblaslt+rocblas | `TENSILE_DB` |

<!-- END GENERATED: env-knob-inventory -->

Two different checks keep it honest, and it matters which one proves what:

* **Drift between lists** — `test_canonical_var_names_stable` compares the
  generated names against a hand-written set, so adding a knob must be
  acknowledged. This catches accidental change and nothing more; it cannot show
  that the upstream libraries are covered, because both sides are written by hand.
* **Coverage of the installed libraries** — `scripts/audit_env_knobs.py` reads
  complete NUL-delimited C strings from the shipped hipBLASLt / rocBLAS shared
  objects and diffs them against the manifest:

```bash
python scripts/audit_env_knobs.py                    # audit the local ROCm install
python scripts/audit_env_knobs.py --rocm-lib /opt/rocm-7.0.2/lib --strict
python scripts/audit_env_knobs.py --json audit.json  # machine-readable
```

  `covered` is a manifest knob found in an installed library. `not_present` is a
  manifest knob this ROCm does not ship — expected for forward-compat entries.
  Capture remains independent: an exported value is preserved even when the
  library ignores it; `null` means only that it was unset. `uncovered` is an
  audited name the library exposes but the manifest omits. The mechanically
  discoverable scope is every `HIPBLASLT_*`, `ROCBLAS_*`, and `TENSILE_*` string
  that begins at a NUL boundary; the five non-prefixed Origami/GridBased names
  found by source review are also checked exactly. The NUL-boundary rule is what
  keeps help text from being read as a variable name, and its cost is that a name
  the linker folded into the tail of a longer string (`.rodata` is
  `SHF_MERGE|SHF_STRINGS`) has no standalone copy to find. A future non-prefixed
  `getenv` still requires source review to
  add it to that explicit scope — the binary string table alone cannot distinguish
  arbitrary uppercase constants from environment-variable names. On the exact
  reference build, identified by both resolved basename and pinned SHA-256, the
  audit also checks library ownership and provenance; those checks are skipped on
  other or patched versions where ownership may legitimately differ.
  `--strict` makes any applicable audit error fatal; a missing requested library
  is always a setup error.

  The script resolves libraries through the soname symlink rather than globbing,
  because a ROCm tree can keep a stale `libhipblaslt.so.1.0.70002` beside the
  active `1.4.70002` and a glob picks the wrong one. A runtime-only tree ships no
  bare `.so` devel link, so the fallback reads the `<soname>.<major>` links and
  takes the highest major numerically — lexicographic ordering picked the oldest
  co-installed library, and a single-digit glob missed a two-digit major
  entirely.

The audit needs a ROCm install, so the GPU workflow runs it with `--strict`
whenever the registry, audit script, or their tests change. The CPU gate covers
the comparator logic with fixture libraries in
`tests/instrumentation/test_env_knob_audit.py`.

Provenance is recorded at the granularity it was actually established. GEMM knobs
carry a measured `source_reference` (the shipped library whose string table holds
the name, re-checkable with the script above) and the reference build that
statement is about. Knobs inherited from schema ≤ 1.14 are marked
`INHERITED_UNAUDITED`, because they were never traced to an upstream call site —
naming a source they were not verified against would be worse than admitting the
gap. Tracing those to upstream `getenv` sites is open follow-up work, and the
count of unaudited entries is asserted to be enumerable by
`test_inherited_knobs_are_marked_unaudited_not_given_a_false_source`.

## Running inside a Buck environment

As of schema 1.3 (issue #163, A1.2a), the env probe detects whether
it's running inside a [Buck2](https://buck2.build/) build environment
and records the result in the top-level `build_system` block. As of
schema 1.4 (issue #163, A1.2b), it can additionally introspect a
target's transitive deps to populate the `library_introspection`
list. Schema 1.13 adds the redacted `buck_invocation` block so that
the mode/config/modifier context used for that cquery is no longer
implicit. As of `aorta env recipe --format buck` (issue #163, A1.2c), the
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
jq '.buck_invocation.status' /tmp/env.json              # -> "not_requested"

# Confirm that this target needs no mode/config/modifier overrides.
aorta env probe \
  --buck-target //app:trainer \
  --buck-default-context \
  -o /tmp/env.json

# Or reproduce the exact cross-option order used by the workload.
aorta env probe \
  --buck-target //app:trainer \
  --buck-option mode=root//mode/debug \
  --buck-option config=build.profile=debug \
  --buck-option mode=root//mode/gpu \
  --buck-option config=scheduler.policy=local \
  --buck-option modifier=//constraints:linux \
  --buck-option modifier=//constraints:gfx \
  -o /tmp/env.json

# The implementation binds the label as a separate query argument:
# buck2 cquery @root//mode/debug -c build.profile=debug \
#   @root//mode/gpu -c scheduler.policy=local \
#   -m //constraints:linux -m //constraints:gfx \
#   'deps(%s)' //app:trainer --json
# No shell or free-form passthrough command is involved.
jq '.library_introspection' /tmp/env.json
# -> [
#   {"name":"hipblaslt","source":"buck","revision":"<repo-sha>","target":"//libs:hipblaslt","configured_target":"root//libs:hipblaslt (prelude//platforms:default#abc...)"},
#   {"name":"rccl","source":"buck","revision":"<repo-sha>","target":"//libs:rccl","configured_target":"root//libs:rccl (prelude//platforms:default#abc...)"},
#   ...
# ]

# When a Buck-matched library is also captured by an A1 per-library
# block (hipblaslt, rocblas, miopen, rccl), an alternate entry is
# synthesised from the A1 block so consumers can compare the buck
# label against the system-package version/lib_hash:
jq '.library_introspection_alternates[] | select(.name=="hipblaslt")' /tmp/env.json
# -> {"name":"hipblaslt","source":"package","revision":"...","package_version":"...","lib_hash":"..."}

jq '.buck_invocation' /tmp/env.json
# -> {
#      "status": "success",
#      "target": "//app:trainer",
#      "context_source": "explicit",
#      "mode_files": ["root//mode/debug", "root//mode/gpu"],
#      "config_keys": ["build.profile", "scheduler.policy"],
#      "modifiers": ["//constraints:linux", "//constraints:gfx"],
#      "option_order": ["mode", "config", "mode", "config", "modifier", "modifier"],
#      "context_fingerprint": "sha256:<digest>",
#      "configured_root_target": "root//app:trainer (prelude//platforms:default#...)",
#      "comparison": "not_compared"
#    }
```

`--buck-timeout` (default 10 s) caps the cquery subprocess. A timeout,
non-zero exit, or unparseable JSON degrades gracefully: both lists are
returned empty and the failure is recorded in `partial_reasons`. The
match patterns currently cover `hipblaslt`, `rccl`, `pytorch`, and
`rocm` runtime; new libraries are added by appending to
`KNOWN_LIBRARY_PATTERNS` in `src/aorta/instrumentation/buck_introspect.py`.

If `--buck-target` is given without any explicit context options and without
`--buck-default-context`, the query still runs so older invocations keep
working. The snapshot records `context_source: "unspecified"` and becomes
partial with a reason saying the default context was not confirmed. Likewise,
if `build_system.kind` is not `buck2`, Aorta still attempts the historical
fail-soft cquery path but records `status: "buck_not_detected"` plus a partial
reason; the result cannot look like a fully plausible Buck capture.

Only config keys are serialized. Full `KEY=VALUE` strings are used to invoke
Buck and compute the aggregate context fingerprint, then discarded. Changing
a value or changing order changes the fingerprint, while the raw value never
appears in `buck_invocation`.

This metadata answers “which client-side context configured this cquery?” It
does **not** answer execution-context Open Q1 (native remote-worker markers) or
Open Q2 (selected execution platform), and it does not prove whether any
workload action ran locally, remotely, or from cache. See
[`env-probe-container-execution-context.md`](env-probe-container-execution-context.md).

### Emitting a Buck recipe (`aorta env recipe --format buck`)

Issue #163 (A1.2c) ships a read-only emitter that turns an env.json
back into a BUCK file fragment for cross-environment handoff:

```bash
aorta env probe --buck-target //app:trainer --buck-default-context -o /tmp/env.json
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
jq '.hipblaslt.rocm_release_tweak' /tmp/env_no_rdhc.json # still populated
```

### Probe a Docker image without AORTA installed

The probe must run inside the environment it describes. Use this manual path
for a standalone image capture when AORTA is installed on the host but not in
the image. It is not the execution model for every sweep: a Docker-aware
workload that implements the
[in-container snapshot contract](../recipes/README.md#wrapper-contract-for-in-container-snapshots)
performs the equivalent mount and probe automatically.

Do not mount the host's entire virtual-environment `site-packages` directory.
Putting it first on `PYTHONPATH` can make host PyTorch or TorchRec shadow the
versions in the image, producing a snapshot of the wrong frameworks. Mount
only the `aorta` package and use the dependency-free `_probe_main` entry point.
The image must provide Python 3.10 or newer.

```bash
IMAGE_REF=registry.example/image:tag
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
OUTPUT_DIR=$(pwd)
AORTA_PACKAGE=$(
  python -c \
    'from pathlib import Path; import aorta; print(Path(aorta.__file__).resolve().parent)'
)

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v "$AORTA_PACKAGE":/opt/aorta_src/aorta:ro \
  -v "$OUTPUT_DIR":/out \
  -e AORTA_DOCKER_IMAGE="$IMAGE_REF" \
  -e AORTA_DOCKER_DIGEST="$IMAGE_ID" \
  "$IMAGE_ID" \
  env PYTHONPATH=/opt/aorta_src \
  python3 -m aorta.instrumentation._probe_main /out/env.json
```

`AORTA_PACKAGE` resolves only the public package from the active host
environment, so framework imports still come from the image. The command runs
the resolved immutable image ID while recording both the requested image
reference and that ID in `env.json`. Repeat with another image and compare the
two output files with `diff <(jq -S . env_a.json) <(jq -S . env_b.json)`.

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

### Is the installed kernel catalog (recipe book) the same on two hosts?

```bash
# Tensile menu identity (installed-library identity -- NOT the runtime
# solution pick, which needs a trace).
jq '.tensile_catalog | {status,
  hipblaslt: .hipblaslt.menu.combined_content_hash,
  rocblas:   .rocblas.menu.combined_content_hash}' /tmp/env.json

# The 7e32d53eb1 trap: two hosts can report the SAME hipBLASLt commit
# yet ship a different Tensile install. The shallow filename hash can
# match while the per-file content hashes differ -- compare both.
diff \
  <(jq -S '.tensile_catalog.hipblaslt | {kernel_db_revision, menu: .menu.combined_content_hash, archs: .menu.gfx_arch_coverage}' good.json) \
  <(jq -S '.tensile_catalog.hipblaslt | {kernel_db_revision, menu: .menu.combined_content_hash, archs: .menu.gfx_arch_coverage}' bad.json)

# Localize WHICH Tensile logic file differs (per-file content hashes)
diff \
  <(jq -S '.tensile_catalog.hipblaslt.menu.files' good.json) \
  <(jq -S '.tensile_catalog.hipblaslt.menu.files' bad.json)

# MIOpen convolution db menu identity. Also surfaces which dir was read
# (a MIOPEN_SYSTEM_DB_PATH override points at a different catalog).
jq '.miopen_catalog | {status, db_dir, db_dir_source,
  menu: .menu.combined_content_hash, archs: .menu.gfx_arch_coverage,
  logic: .menu.logic_file_count}' /tmp/env.json

# rocFFT optional AOT kernel cache -- usually absent (that's fine)
jq '.rocfft_catalog | {status, present: .kernel_cache.present,
  path: .kernel_cache.path, sha: .kernel_cache.sha256}' /tmp/env.json

# Did a menu read cleanly, or couldn't we locate/parse it?
jq '{tensile: .tensile_catalog.status, miopen: .miopen_catalog.menu.status,
  rocfft: .rocfft_catalog.status}' /tmp/env.json
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
* **Runtime kernel solution-ID capture** (`381881` / `368xxx`):
  enabling `HIPBLASLT_LOG_MASK` / `CHECK_NUMERICS`, parsing
  `hipblaslt.log`, and `hipblaslt-bench` replay. The `*_catalog` blocks
  capture only the *installed menu* (layer 1); which recipe the runtime
  actually picks (layer 2) needs a workload and belongs to the runtime
  follow-up.
* Workload config (`AMP_DTYPE`, `MODEL_DTYPE`, ...) -- captured by
  `aorta run` in the trial result (Task B1), not by env probe

## See also

* Module reference: [`src/aorta/instrumentation/README.md`](../src/aorta/instrumentation/README.md)
* Issue with the full acceptance criteria: [#147](https://github.com/ROCm/aorta/issues/147)
