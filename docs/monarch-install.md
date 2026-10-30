# Install and run Monarch

[Monarch](https://github.com/meta-pytorch/monarch) is PyTorch's actor framework for running one Python controller across many GPU processes. Aorta's `tiny_transformer` workload uses it when a recipe sets `launcher: monarch`. Aorta does not install Monarch.

This page covers installing Monarch, checking it on one machine, and the settings a multi-machine run needs. [Tiny transformer](tiny-transformer.md) covers the 16-GPU aorta runs.

## Which install you need

| GPU | How to install |
|---|---|
| NVIDIA (CUDA) | Install the published `torchmonarch` package. See [CUDA](#cuda). |
| AMD (ROCm) | Build Monarch from source against the same ROCm your PyTorch uses. See [ROCm](#rocm). The published `torchmonarch` packages are CUDA builds and do not run on ROCm. |

Install Monarch into the Python environment that runs aorta. Every machine in the run must see that same environment at the same path, for example on a shared filesystem.

## CUDA

```bash
pip install torchmonarch            # latest release
pip install --pre torchmonarch      # nightly
```

Then [check the install](#check-the-install).

## ROCm

This takes about an hour, most of it downloads and the compile. Build on a machine with the GPUs you will run on, and keep the environment on a filesystem with at least 25 GB free. The ROCm PyTorch packages alone are about 20 GB.

The steps below use `uv`, and the verified build used Python 3.11. Install `uv` with `curl -LsSf https://astral.sh/uv/install.sh | sh` if you do not have it.

### 1. Create the environment and install ROCm PyTorch

Pick the PyTorch index for your ROCm version. For example, `https://download.pytorch.org/whl/nightly/rocm10.1` is the nightly ROCm 10.1 channel.

```bash
export MONARCH_ROOT=/shared/path/monarch      # a directory every machine can read
export TORCH_INDEX=https://download.pytorch.org/whl/nightly/rocm10.1
mkdir -p "$MONARCH_ROOT" && cd "$MONARCH_ROOT"

uv venv --python 3.11 venv
source venv/bin/activate
uv pip install --pre torch --index-url "$TORCH_INDEX"
python -c 'import torch; print(torch.version.hip, torch.cuda.get_device_name(0))'
```

`torch.version.hip` must print a version, not `None`, and the device name must be your GPU.

### 2. Install the ROCm development files

Monarch compiles HIP code and links the HIP runtime and RCCL. The build needs the ROCm compiler, headers, and RCCL headers, and they must come from the same ROCm your PyTorch uses. If one process loads two different HIP runtimes, it fails.

If your PyTorch came from the PyTorch index, as in step 1, install the matching development package from that same index:

```bash
uv pip install --pre "rocm[devel]==10.1.0" --index-url "$TORCH_INDEX"
uv pip install numpy
rocm-sdk init
export ROCM_PATH=$(rocm-sdk path --root)
```

Use only the PyTorch index here. Adding `--extra-index-url https://pypi.org/simple` lets `uv` pick an incompatible `rocm-sdk-devel` package from PyPI.

If your PyTorch instead uses a system ROCm installation, set `ROCM_PATH` to that installation, for example `/opt/rocm-<version>`, and skip the install above.

### 3. Point the build at that ROCm only

Save this as `$MONARCH_ROOT/monarch-env.sh`. Source it in every shell that builds Monarch or runs it:

```bash
source "$MONARCH_ROOT/venv/bin/activate"
export ROCM_PATH=${ROCM_PATH:-$(rocm-sdk path --root)}
export ROCM_HOME=$ROCM_PATH HIP_PATH=$ROCM_PATH
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode
export MONARCH_GPU_PLATFORM=rocm
export CC=$ROCM_PATH/lib/llvm/bin/clang CXX=$ROCM_PATH/lib/llvm/bin/clang++
export LIBCLANG_PATH=$ROCM_PATH/lib/llvm/lib
export BINDGEN_EXTRA_CLANG_ARGS="-resource-dir=$($CC -print-resource-dir)"
export RUSTUP_HOME=$MONARCH_ROOT/tools/rustup CARGO_HOME=$MONARCH_ROOT/tools/cargo
export PATH=$ROCM_PATH/bin:$CARGO_HOME/bin:$MONARCH_ROOT/tools/protoc/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# GPU kernel caches on local disk. Many processes writing a shared home
# directory at once can fail with "LLVM ERROR: IO failure on output stream".
export AMD_COMGR_CACHE_DIR=/tmp/$USER/comgr-cache
export TRITON_CACHE_DIR=/tmp/$USER/triton-cache
export MIOPEN_USER_DB_PATH=/tmp/$USER/miopen MIOPEN_CUSTOM_CACHE_DIR=/tmp/$USER/miopen
```

What these settings are for:

- `ROCM_PATH`, `PATH`, and `LD_LIBRARY_PATH` make this ROCm win over any other ROCm on the machine.
- `MONARCH_GPU_PLATFORM=rocm` tells the build to link ROCm. It is required when CUDA is also installed.
- `LIBCLANG_PATH` and `BINDGEN_EXTRA_CLANG_ARGS` let the Rust bindings find ROCm's `libclang` and its built-in headers such as `stddef.h`.
- The cache variables keep GPU kernel caches on each machine's local disk.

### 4. Install Rust nightly and protoc

```bash
source "$MONARCH_ROOT/monarch-env.sh"
mkdir -p "$MONARCH_ROOT/tools" && cd "$MONARCH_ROOT/tools"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup-init.sh
sh rustup-init.sh -y --no-modify-path --default-toolchain nightly --profile minimal

V=$(curl -s https://api.github.com/repos/protocolbuffers/protobuf/releases/latest \
  | grep -o '"tag_name": *"v[^"]*"' | grep -o 'v[0-9.]*')
curl -sSL -o protoc.zip \
  "https://github.com/protocolbuffers/protobuf/releases/download/$V/protoc-${V#v}-linux-x86_64.zip"
mkdir -p protoc && (cd protoc && python3 -m zipfile -e ../protoc.zip . && chmod +x bin/protoc)
```

The build also needs `git`, `cmake`, `ninja`, and the `libibverbs` development headers from your system packages. Monarch pins its own Rust nightly, and `cargo` downloads it on the first build.

### 5. Build Monarch

```bash
source "$MONARCH_ROOT/monarch-env.sh"
cd "$MONARCH_ROOT"
git clone --recurse-submodules https://github.com/meta-pytorch/monarch.git
cd monarch
uv pip install setuptools-rust wheel "setuptools>=77"
uv pip install --no-build-isolation -v . 2>&1 | tee build.log
```

- `--recurse-submodules` is required. The ROCm build runs a tool from a submodule.
- `--no-build-isolation` builds against the ROCm PyTorch already in the environment, instead of downloading another one.
- Use `uv pip`. A `uv` environment has no `pip` module.

Near the top of `build.log`, check for:

```
✓ Building WITH tensor_engine + GPU support
  - ROCm: <the ROCM_PATH you set>
```

If it shows a different ROCm, `monarch-env.sh` was not sourced in that shell. Rebuild after sourcing it.

## Install aorta into the same environment

The aorta command and the Monarch workers run in this environment, so aorta has to be installed in it. For example, from an aorta checkout:

```bash
source "$MONARCH_ROOT/monarch-env.sh"      # ROCm; on CUDA, activate your environment
uv pip install /path/to/aorta
aorta sweep run --recipe /path/to/aorta/recipes/training/tiny-transformer.yaml --dry-run
```

The dry run prints the recipe's plan and does not train. This installs aorta with its own dependencies. It does not reinstall PyTorch, because aorta does not list PyTorch as a dependency.

## Check the install

Run this outside the Monarch source directory, so Python imports the installed package:

```bash
cd /tmp
python -c 'import monarch, torch; from monarch.actor import Actor, endpoint; print("ok", torch.__version__)'
```

On ROCm, also check that Monarch links the HIP runtime from `ROCM_PATH`:

```bash
ldd "$(python -c 'import monarch._rust_bindings as m; print(m.__file__)')" | grep amdhip
```

The `libamdhip64` path must be under `$ROCM_PATH`.

## Run Monarch on one machine

This starts one process per GPU on this machine and calls an actor in each:

```python
# monarch_smoke.py
from monarch.actor import Actor, endpoint, this_host


class Probe(Actor):
    @endpoint
    def identify(self) -> str:
        import os
        import socket

        import torch

        return (
            f"host={socket.gethostname()} pid={os.getpid()} "
            f"gpus_visible={torch.cuda.device_count()}"
        )


procs = this_host().spawn_procs({"gpus": 8})      # set to the GPU count on this machine
actors = procs.spawn("probe", Probe)
for point, line in actors.identify.call().get().items():
    print(point, line)
```

```bash
source "$MONARCH_ROOT/monarch-env.sh"      # ROCm only
python monarch_smoke.py
```

Monarch works on this machine when it prints one line per GPU, each with a different `pid`.

`gpus_visible` can be the full GPU count in every process. Monarch does not always limit each process to one GPU. Code that always uses device 0 would then put every process on the same GPU. Aorta's training loop uses `LOCAL_RANK` as the device index in that case.

## Settings for runs across machines

The Monarch controller starts one Monarch worker process on each machine, and those workers start the training processes. The workers inherit the controller shell's environment. Before running aorta, set these in that shell:

- **The same Python environment on every machine.** Activate the environment, and on ROCm source `monarch-env.sh`. The workers run the same Python path.
- **An IPv4 address the workers can reach.** Aorta binds Monarch's TCP transport to the controller's IPv4 address, which it looks up from this machine's hostname. If that lookup fails or gives an address the workers cannot reach, set `AORTA_CONTROLLER_IP` to the right IPv4 address. Monarch's default local Unix socket does not work across machines.
- **RCCL or NCCL network settings for your cluster.** For example, `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` name the network interface, and `NCCL_IB_HCA` lists the RDMA devices. On ROCm, RCCL warns at startup unless `HSA_NO_SCRATCH_RECLAIM=1` is set.
- **A free worker port.** Monarch workers listen on `MONARCH_PORT`, which defaults to 22222. After a run, that port can stay open for about 20 seconds. Wait that long before the next run, or set a different `MONARCH_PORT`.

## Troubleshooting

| What you see | Cause and fix |
|---|---|
| `attach failed ... reply timed out` | The workers could not reach the controller. Set `AORTA_CONTROLLER_IP` to the controller's IPv4 address. |
| `Address already in use` when the workers start | A previous run still holds `MONARCH_PORT`. Wait about 20 seconds, or set a different `MONARCH_PORT`. |
| `Unable to find libclang` during the build | `LIBCLANG_PATH` is not set. Source `monarch-env.sh`. |
| `'stddef.h' file not found` during the build | `BINDGEN_EXTRA_CLANG_ARGS` is not set. Source `monarch-env.sh`. |
| `rocm[devel]` fails to resolve and mentions PyPI | Install it from the PyTorch index only, without `--extra-index-url`. |
| `LLVM ERROR: IO failure on output stream`, and one process exits before step 0 | The GPU kernel cache is on a shared home directory. Use the local cache variables in `monarch-env.sh`. |
| The controller waits about 2 minutes after one process is killed | A known Monarch issue: [meta-pytorch/monarch#5167](https://github.com/meta-pytorch/monarch/issues/5167). Setting `HYPERACTOR_MESH_SUPERVISION_WATCHDOG_TIMEOUT=20s` shortens the wait. |
| You need the workers' own logs | Monarch writes `/tmp/$USER/monarch_log.log` on each machine. |
| `logging_option(stream_to_client=True) has no effect` | Harmless. Worker output is not streamed to the controller. Set `HYPERACTOR_MESH_ENABLE_LOG_FORWARDING=true` before the run to stream it. |
| `No module named 'aorta'` on a worker, or `Workload 'tiny_transformer' not found` | aorta is not installed in the environment the workers run. See [Install aorta into the same environment](#install-aorta-into-the-same-environment). |
