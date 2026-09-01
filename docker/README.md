# Docker Setup for Aorta

This directory contains Docker configurations for building and running Aorta training workloads.

## Overview

We provide a unified Docker Compose configuration that supports multiple Dockerfile variants through environment variables. Each user can maintain their own `.env` file (git-ignored) with personalized settings.

## Quick Start

### Method 1: Interactive Setup (Recommended)

Run the setup script to create your `.env` file interactively:

```bash
bash setup-env.sh
docker compose -f docker-compose.build.yaml up -d
```

The script will guide you through:
- Selecting a Dockerfile
- Naming your container
- Configuring volume mounts
- Setting environment variables

### Method 2: Manual Configuration

Copy the example and edit manually:

```bash
cp .env.example .env
# Edit .env with your settings
docker compose -f docker-compose.build.yaml up -d
```

## Available Dockerfiles

The **Stack** column is what each image actually installs, which is not what the
filename suggests: `rocm70_9-1` / `rocm70_2` come from the `amdgpu` installer
package name (`amdgpu-install-internal-7.0_9-1`), which is an installer revision
rather than a ROCm version. Read this column, not the filename, when picking an
image to reproduce a stack against.

| Dockerfile | Stack | Use Case |
|------------|-------|----------|
| `Dockerfile.rocm-latest` | **10.0 / PyTorch 2.13** (newest stack CI validates) — compose default | General development and testing |
| `Dockerfile.rocm70_9-1` | 7.2 / PyTorch 2.9.1 | Older-stack comparison |
| `Dockerfile.rocm70_9-1-shampoo` | 7.0 meta build #19, plus Shampoo optimizer | Shampoo optimizer experiments |
| `Dockerfile.rocm70_2-ubuntu-pytorch` | 7.0.2.1 build #17 on Ubuntu 22.04 | Legacy 7.0.2.x support |
| `Dockerfile.rocm70_2-ubuntu-nan` | 7.0.2.1 build #17, plus NaN debugging | Debugging NaN issues |
| `Dockerfile.rocm-ubuntu-ebpf` | 7.2.0.1 build #5, plus eBPF tracing (bpftrace, bcc) | eBPF-based GPU queue/memory tracing |
| `Dockerfile.ci-gpu` | 10.0 / PyTorch 2.13, pinned by digest | GPU CI on self-hosted runners (see `.env.ci`) |
| `Dockerfile.rocm-canary` | whatever `rocm/pytorch:latest` (or a dispatched override) resolves to, via a `BASE_IMAGE` build arg | Non-gating latest-ROCm canary (see `.env.canary`, issue #382) |

Except for `Dockerfile.rocm-latest`, `Dockerfile.ci-gpu` and
`Dockerfile.rocm-canary`, the images above are
pinned to older ROCm **on purpose** — the version is the thing under test (a
customer's stack, a specific `amdgpu` build, a bisect point). Bumping them would
turn a reproducer into noise. New general-purpose tooling belongs in
`Dockerfile.rocm-latest`.

ROCm **10.0** is the newest production release these images track, as of the
issue #383 stack flip — which moved ROCm, Ubuntu, Python and PyTorch together
(7.2.4 → 10.0, 24.04 → 26.04, py3.12 → py3.14, torch 2.10 → 2.13), because the
ROCm 10 line publishes no torch 2.10 image and so offered no smaller step.

That flip also retires the 7.x version-reading rules that used to live here: ROCm
7.9–7.13 were the technology *preview* stream (a higher number there was not an
upgrade), and 7.14 was the first wheel-based (TheRock) production line. The ROCm
10 `rocm/pytorch` images the two tracking Dockerfiles here use are wheel-layout
(TheRock) builds — in **those images** ROCm lives under `site-packages` and there
is no `/opt/rocm` at all. That is a property of the image, not of the release:
ROCm 10 still ships DEB/RPM/runfile packages that install a classic `/opt/rocm`
tree, so a container built on one of those still has one and is still supported.
Issue #381 made ROCm discovery layout-agnostic, which is what makes either
readable.

**Disk:** budget the *uncompressed* size for these ROCm bases. Docker Hub lists
the ROCm 10 base at around 20.5 GB, but that is the compressed manifest —
`docker images` reports **51.8 GB** once pulled, and the images built on top add
little. Recent ROCm bases all land in the 40–52 GB range, so a host that keeps
the gate image and the canary image at once needs well over 100 GB for these
alone, before any older pinned reproducer images.

`Dockerfile.ci-gpu`, `Dockerfile.rocm-latest` and `Dockerfile.rocm-canary` run
[`rocm_layout_guard.py`](rocm_layout_guard.py) at build time. It accepts either
layout and fails the build only when neither yields a readable ROCm version and
lib directory, so a digest bump onto an unreadable base is loud instead of
silently reporting `null`. Run it inside a container to debug a bad image:

```bash
docker run --rm --entrypoint python <image> /usr/local/share/aorta/rocm_layout_guard.py
```

### Wheel-layout images: bare sonames and the library path

On a wheel-layout `rocm/pytorch` base — what `Dockerfile.rocm-latest`,
`Dockerfile.ci-gpu` and (in practice) `Dockerfile.rocm-canary` build on — two
things are missing that a classic `/opt/rocm` install provides, and both bite
hipcc-built code:

- The base ships the ROCm **runtime** wheels but not `rocm[devel]`, so only
  versioned sonames (`libamdhip64.so.7`) exist. hipcc hands the linker the bare
  `libamdhip64.so`, and Proton `dlopen`s bare names, so both fail. All three
  Dockerfiles rebuild the absent link farm at build time, across
  `_rocm_sdk_core/lib` and `_rocm_sdk_libraries/lib`, deriving the directories
  rather than hardcoding them and skipping the step entirely on a classic base.
  It is 44 links on the digest `Dockerfile.ci-gpu` currently pins; the step is
  generalized rather than a name list, so that count is an observation and not a
  contract. The block is byte-identical in all three files and a test holds it
  that way — the canary in particular is only readable if a canary/gate
  difference is attributable to ROCm rather than to our own images.
- hipcc output on this base carries neither `DT_RPATH` nor `DT_RUNPATH` and the
  base sets no `LD_LIBRARY_PATH`, so a binary that links still dies before `main`
  at exit 127. In CI every launch goes through a workflow step that exports the
  resolved lib dirs. `Dockerfile.rocm-latest` instead writes
  `/etc/profile.d/aorta-rocm-libpath.sh` (also sourced from `/etc/bash.bashrc`,
  since `docker exec -it … bash` is interactive but not a login shell), which
  **appends** those dirs — so a substitution you inherited, such as the custom
  RCCL from `docker-compose.rccl.yaml`, still wins.

A non-interactive `docker exec <container> bash -c '…'` gets neither file. Export
the dirs yourself there, the way the workflows do:

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"
```

That is byte-identical to the line the sanitizer dashboard publishes beside its
rebuild commands, which is where it is kept honest.

`Dockerfile.ci-gpu` deliberately ships no profile script: its launches all come
from workflow steps that export the line above, and the gate is where an image
that quietly depends on a baked environment instead of on the binary's own
`RPATH` would hide a regression (see the `readelf -d` note in that file).
`Dockerfile.rocm-canary` ships none either, for a different reason — it has no
interactive users, and it exists to predict the gate image, so environment the
gate does not have would be exactly the canary/gate difference the link farm was
ported to remove.

## The latest-ROCm canary (`Dockerfile.rocm-canary`)

The one image here whose base is *meant* to move. It exists so a new ROCm
release is noticed early without the merge gate depending on a moving tag —
`.github/workflows/latest-rocm-canary.yml` resolves `rocm/pytorch:latest` to a
concrete digest at job start, passes it in as `BASE_IMAGE`, and records it with
the results, so the tag moves between runs while each run stays reproducible.

The workflow also takes an optional `base_image` `workflow_dispatch` input, which
points the lane at some other tag for a single run. It closes a real gap rather
than adding a convenience: a major ROCm release is published under a versioned
tag well before AMD moves `:latest`, so the lane is blind to it during exactly
the window the early warning is worth something (ROCm 10 was the case in point).
Leave the input blank to follow `:latest` as usual. An override does not make an
unattributable run possible — it goes through the same resolve step, so the run
still records `tag@sha256:…`, and the recorded base image is what tells an
overridden canary row apart from a scheduled one.

`BASE_IMAGE` has no default and `docker-compose.canary.yaml` requires
`CANARY_BASE_IMAGE`, so building it without a resolved digest fails rather than
quietly meaning "whatever `:latest` is right now". Build it the way CI does:

```bash
cd docker
cp .env.canary /tmp/canary.env
echo "CANARY_BASE_IMAGE=rocm/pytorch:latest@sha256:<digest>" >> /tmp/canary.env
docker compose --env-file /tmp/canary.env \
  -f docker-compose.build.yaml -f docker-compose.canary.yaml build
```

This lane is a wheel-layout (TheRock) image in practice, which is why it needed
issue #381 first — before layout-agnostic discovery it would have reported
`rocm: null`.

## CI configuration (`.env.ci`)

GPU CI uses a committed env file (not gitignored) so runs are reproducible:

```bash
docker compose --env-file .env.ci -f docker-compose.build.yaml up -d --build
```

See [`.env.ci`](.env.ci), [`Dockerfile.ci-gpu`](Dockerfile.ci-gpu), and
[`docs/ci-testing-plan.md`](../docs/ci-testing-plan.md) (Phase 2).

### Required

- **`DOCKERFILE`**: Which Dockerfile to build from
- **`CONTAINER_NAME`**: Unique name for your container (avoid conflicts with other users)

### Volume Mounts

- **`AORTA_WORKSPACE`**: Path to aorta workspace (default: `..`)
- **`RCCL_PATH`**: Optional. Leave unset to use the image's RCCL (no YAML edit needed). To use a custom RCCL build, set this and run with `-f docker-compose.rccl.yaml` (see [Using custom RCCL](#using-custom-rccl)).

### Optional

- **`AMDGPU_DRIVER_VARIANT`**: Driver variant for environment_info.json
- **`EXTRA_MOUNT_SRC_*`** / **`EXTRA_MOUNT_DST_*`**: Additional volume mounts

## Example Configurations

### Example 1: Standard Development (image RCCL)

```bash
# .env
DOCKERFILE=Dockerfile.rocm-latest
CONTAINER_NAME=myuser-dev-20260205
AORTA_WORKSPACE=..
# RCCL_PATH unset = use image RCCL
```

Run: `docker compose -f docker-compose.build.yaml up -d`

### Example 2: Shampoo with Custom RCCL

```bash
# .env
DOCKERFILE=Dockerfile.rocm70_9-1-shampoo
CONTAINER_NAME=shampoo-experiment-1
AORTA_WORKSPACE=/apps/username/aorta_work/aorta_1
RCCL_PATH=/apps/username/rccl
```

Run: `docker compose -f docker-compose.build.yaml -f docker-compose.rccl.yaml up -d`

### Example 3: NaN Debugging

```bash
# .env
DOCKERFILE=Dockerfile.rocm70_2-ubuntu-nan
CONTAINER_NAME=debug-nan-issue
AORTA_WORKSPACE=..
AMDGPU_DRIVER_VARIANT=patched
```

### Example 4: eBPF GPU Tracing

```bash
# .env
DOCKERFILE=Dockerfile.rocm-ubuntu-ebpf
CONTAINER_NAME=myuser-ebpf-tracing
AORTA_WORKSPACE=..
```

Inside the container, verify eBPF readiness with `aorta ebpf-info`, then run
workloads with `--ebpf-trace` and/or `--ebpf-memory-trace` flags.

## Using custom RCCL

By default, the container uses the RCCL bundled in the image. You do not need to set or remove any RCCL path in the YAML.

To use a custom RCCL build:

1. Set `RCCL_PATH` in your `.env` to your RCCL build directory.
2. Run with the RCCL override file:

   ```bash
   docker compose -f docker-compose.build.yaml -f docker-compose.rccl.yaml up -d
   ```

The override file adds the RCCL volume and RCCL-related environment variables only when you use it.

## File Structure

```
docker/
├── docker-compose.build.yaml     # Unified compose file (use this!)
├── docker-compose.rccl.yaml      # Optional: use with -f when RCCL_PATH is set
├── docker-compose.yaml           # Image-based compose (alternative)
├── .env.example                  # Template for your .env
├── .env                          # Your personal config (git-ignored)
├── setup-env.sh                  # Interactive setup script
├── Dockerfile.rocm-latest        # ROCm 10.0 / PyTorch 2.13 (compose default)
├── Dockerfile.rocm70_9-1         # ROCm 7.2 / PyTorch 2.9.1
├── Dockerfile.rocm70_9-1-shampoo # ROCm 7.0 meta build #19 + Shampoo
├── Dockerfile.rocm70_2-ubuntu-*  # ROCm 7.0.2.1 build #17
├── Dockerfile.rocm-ubuntu-ebpf   # ROCm 7.2.0.1 build #5 + eBPF tracing tools
└── rccl_test/                    # Separate RCCL testing setup
```

## Common Commands

### Start Container

```bash
docker compose -f docker-compose.build.yaml up -d
```

### Stop Container

```bash
docker compose -f docker-compose.build.yaml down
```

### View Logs

```bash
docker compose -f docker-compose.build.yaml logs -f
```

### Connect to Container

```bash
docker exec -it <your-container-name> bash
```

### Rebuild After Dockerfile Changes

```bash
docker compose -f docker-compose.build.yaml build
docker compose -f docker-compose.build.yaml up -d
```

### View Resolved Configuration

See what environment variables are being used:

```bash
docker compose -f docker-compose.build.yaml config
```

## Tips

1. **Unique Container Names**: Use descriptive, unique names to avoid conflicts with other users on shared systems
   - Good: `username-shampoo-2026-02-05`
   - Bad: `training` (too generic)

2. **Git Ignore**: Your `.env` file is git-ignored, so your personal configuration won't be committed

3. **Environment Override**: You can override any variable at runtime:
   ```bash
   CONTAINER_NAME=test-run docker compose -f docker-compose.build.yaml up
   ```

4. **VSCode Integration**: Use VSCode's "Attach to Running Container" feature for an IDE experience

5. **Multiple Variants**: You can run multiple containers with different Dockerfiles simultaneously by using different container names

## Troubleshooting

### "container name already in use"

Another user or previous run is using that name. Choose a different `CONTAINER_NAME`.

### "No such file or directory" for volumes

Check that paths in your `.env` exist and are accessible:
```bash
ls -la $AORTA_WORKSPACE
ls -la $RCCL_PATH
```

### Changes to .env not taking effect

Stop and restart the container:
```bash
docker compose -f docker-compose.build.yaml down
docker compose -f docker-compose.build.yaml up -d
```

### Need to add more volume mounts

Edit your `.env` and add:
```bash
EXTRA_MOUNT_SRC_1=/path/on/host
EXTRA_MOUNT_DST_1=/path/in/container
```

Then update `docker-compose.build.yaml` to reference them in the volumes section.

## Migration from Old Compose Files

If you were using:
- `docker-compose.rocm70_9-1.yaml` → Use `docker-compose.build.yaml` with `DOCKERFILE=Dockerfile.rocm70_9-1`
- `docker-compose.rocm70_9-1-shampoo.yaml` → Use `docker-compose.build.yaml` with `DOCKERFILE=Dockerfile.rocm70_9-1-shampoo`

These old files are deprecated and will be removed in a future update.

## Related Documentation

- [Getting Started Guide](../docs/getting-started.md)
- [Running Benchmarks](../docs/running-benchmark.md)
- [Profiling Guide](../docs/profiling.md)
