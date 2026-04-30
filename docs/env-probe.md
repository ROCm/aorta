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
jq '.hipblaslt.commit' runs/exp1/env.json
jq '.partial' runs/exp1/env.json

# Diff two snapshots from different docker images
diff <(jq -S . env_a.json) <(jq -S . env_b.json)
```

## CLI

```text
Usage: aorta env probe [OPTIONS]

  Capture trial-environment state to env.json (issue #147).

Options:
  -o, --output FILE  Path to write env.json.  [default: env.json]
  --help             Show this message and exit.
```

The CLI is a thin wrapper. It calls `collect_env()`, writes the JSON,
and prints a six-line summary like:

```text
Wrote env probe to /tmp/env.json (schema_version=1.0) [PARTIAL]
  runtime:  baremetal / python=venv [PARTIAL, 3 reason(s)]
  rocm:     7.2.1 (dev: None)
  hip:      7.2.53211-e1a6bc5663 (amd)
  hipblaslt: commit=dabb6df2b9
  rdhc:     unavailable (system_health=null)
  python:   3.12.3 | pytorch: 2.12.0a0+gitf68b851
```

`[PARTIAL]` indicates at least one probe fell back to `None` -- the
snapshot is still complete (every key present), it just records what
was missing in `partial_reasons`.

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

# Six-line human summary, same as the CLI prints
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
| `schema_version` | `str` | constant | `"1.0"` today; bumps on non-additive changes only |
| `captured_at` | `str` | `datetime` | ISO-8601 UTC with trailing `Z` |
| `partial` | `bool` | computed | `True` if any probe fell back |
| `partial_reasons` | `list[str]` | per-probe | one human-readable line per fallback |
| `system_health` | `dict \| null` | `rdhc --quick --json` (subprocess) | verbatim parsed JSON; `null` when rdhc absent / sudo unavailable / timeout / malformed |
| `rocm` | `dict[str, str \| null]` | `/opt/rocm/.info/version{,_dev}`, `/sys/module/amdgpu/version` | `version`, `version_dev`, `kmd_version` |
| `hip` | `dict[str, str \| null]` | `hipconfig --version/--platform/--compiler/--runtime/--cpp_config` | five subprocesses; `--version` and `--platform` cannot be combined (no delimiter) |
| `hipblaslt` | `dict` | header parse + `sha256(libhipblaslt.so)` + sorted-filenames hash of `lib/hipblaslt/library/*` | `commit`, `package_version`, `lib_hash`, `tensile_yaml_revision`, `applied_prs: {}` |
| `runtime_context` | `dict` | `/.dockerenv`, `/run/.containerenv`, `$SINGULARITY_NAME`, `/proc/1/cgroup`, `sys.prefix`, `$CONDA_DEFAULT_ENV` | `type`, `python_env`, `venv_path`, `conda_env_name` |
| `docker` | `dict \| null` | `$AORTA_DOCKER_IMAGE` / `$AORTA_DOCKER_DIGEST` env vars + `/proc/self/cgroup` | `null` on baremetal; image+digest provided by the launcher (the only reliable way from inside a container) |
| `env_vars` | `dict[str, str \| null]` | canonical 13-name list, explicit | HSA + GPU queue + RCCL + FBGEMM + PyTorch |
| `python_version` | `str` | `platform.python_version()` | always populated |
| `pytorch_version` | `str \| null` | optional `import torch` (no CUDA/HIP context init) | `null` when torch absent |

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
| Baremetal host (no rdhc) | ~0.10 s |
| Inside a docker image | ~0.8 s |

Both well inside the <5 s no-rdhc / <15 s with-rdhc targets.

No GPU compute. Verified via `rocprofv3 --hip-trace`: zero HIP API
calls, zero kernel dispatches.

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
| `hipblaslt.commit` | `HIPBLASLT_VERSION_TWEAK` define in `/opt/rocm/include/hipblaslt/hipblaslt-version.h` |
| `hipblaslt.package_version` | `HIPBLASLT_VERSION_{MAJOR,MINOR,PATCH}` defines in the same header |
| `hipblaslt.lib_hash` | `sha256(/opt/rocm/lib/libhipblaslt.so)` resolved through symlinks |
| `hipblaslt.tensile_yaml_revision` | sha256 of sorted filenames of `*.yaml`/`*.dat`/`*.co` under `/opt/rocm/lib/hipblaslt/library/` |
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

## Out of scope (deferred to P1)

* `aorta env matrix` (multi-docker fan-out)
* `aorta env diff env_a.json env_b.json`
* GPU kernel introspection (anything that runs HIP work)
* Composable Kernel / rocBLAS commit + build state -- same pattern as
  the `hipblaslt` block but a different code path; lands when a second
  incident points at one of them
* Workload config (`AMP_DTYPE`, `MODEL_DTYPE`, ...) -- captured by
  `aorta run` in the trial result (Task B1), not by env probe

## See also

* Module reference: [`src/aorta/instrumentation/README.md`](../src/aorta/instrumentation/README.md)
* Issue with the full acceptance criteria: [#147](https://github.com/ROCm/aorta/issues/147)
