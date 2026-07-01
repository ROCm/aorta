# Running AORTA workloads on an emulated GPU (mirage + rocjitsu)

Status: **implemented (single-process); multi-rank pending upstream**

## Summary

This adds an **emulator environment axis** to AORTA so a workload or triage
cell can run on a software-emulated GPU — driven by the
[`mirage`](https://github.com/ROCm/rocm-systems/tree/develop/emulation/mirage)
control plane and the [`rocjitsu`](https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu)
software GPU emulator — with **no physical GPU**. The same recipe can then run
in two places:

* `environment: local` (or `{docker: ...}`) → real hardware.
* `environment: emulated-rocjitsu` → emulated GPU (hardware-free dev / CI /
  functional-correctness).

rocjitsu interposes the kernel-mode driver via `LD_PRELOAD`, so an unmodified
ROCm/PyTorch process runs against an emulated device; mirage is the
session/exec control plane that injects that preload. AORTA already threads a
resolved `Environment` into every workload's config (`_aorta_environment`); this
change adds an emulator axis to that descriptor and a small launch backend that
honours it.

## Design

### Environment axis
`Environment` (`registry/types.py`) gains two optional fields, peers of
`docker`/`venv`/`buck_target`:

* `mirage_profile` — name of a mirage profile (which itself encodes the emulator
  backend, topology, and exec mode). Authoritative.
* `emulator` — optional convenience hint naming the backend
  (`"rocjitsu"` / `"hotswap"` / `"noop"`).

Both default `None`, so every existing environment, recipe, and sidecar is
unchanged. The registry (`environments.py`) and the JSON sidecar loader
(`sidecar.py`) accept the new keys; a built-in `emulated-rocjitsu` environment
(`mirage_profile: mi350x`) resolves when the profile exists (scripts create it
on first use). A companion `emulated-rocjitsu-dbt` environment targets
rocjitsu-dbt on physical GPUs. The dispatcher already serialises the whole descriptor into
`_aorta_environment`, so no dispatcher change is needed to carry the new fields.

### Launch backend
`aorta.emulation.mirage_launch` turns an emulated environment into a launch:

* `wrap_argv_for_environment(config, argv)` — when the resolved environment
  carries `mirage_profile`, returns `["mirage", "run", "--profile", <p>, "--",
  *argv]`; otherwise returns the argv unchanged (so non-emulated launches are
  byte-for-byte identical). The mirage binary is resolved from `$MIRAGE_BIN`
  (default `mirage` on `$PATH`); a requested-but-unbuildable emulation raises
  loudly rather than silently running on real hardware.

This module only *constructs* the launch; it spawns nothing — mirroring AORTA's
policy that the platform threads tier hints while wrappers decide how to launch.

### Consumers
* **`aorta probe`** (`workloads/_subprocess.py`): opt-in — when the cell's
  environment is emulated, the opaque user argv is transparently wrapped through
  mirage.
* **In-process triage workloads**: launch the whole `aorta triage run` under
  `mirage run` so the workload's GPU calls hit the emulated device (operator
  pattern). The new single-process `gpu_smoke` workload + `recipes/gpu-smoke-emulated.yaml`
  demonstrate this end-to-end.

### `gpu_smoke` workload
A minimal single-process workload (`min_world_size = 1`) that runs a trivial
CUDA/HIP kernel (`x.add_(1.0)`) and verifies the result. It is the smallest
end-to-end GPU check and is ideal as a hardware-free emulator/CI smoke test.

## Usage

**Reproduction guide:** [docs/emulated-gpu-mirage-setup.md](../emulated-gpu-mirage-setup.md)

```sh
# Build mirage (rocm-systems/emulation/mirage), then from this repo:
export MIRAGE_BIN=/path/to/mirage

# Recommended: vLLM container path (fast torch)
./scripts/emulation/run_mirage_container.sh gpu-smoke
./scripts/emulation/run_mirage_container.sh probe

# rocjitsu-dbt on a machine with GPUs
EMULATOR=rocjitsu-dbt ./scripts/emulation/run_mirage_container.sh gpu-smoke

# Full CLI matrix (12 commands × rocjitsu + rocjitsu-dbt)
./scripts/emulation/run_mirage_matrix.sh
```

Host-only path (slow under rocjitsu CPU emulation):

```sh
mirage profile create mi350x --emulator rocjitsu --agent MI350X --no-input
mirage run --profile mi350x -- \
    aorta triage run --recipe recipes/gpu-smoke-emulated.yaml
```

Prerequisites: `mirage` CLI (`$MIRAGE_BIN`), rocjitsu runtime, and for the
recommended path a ROCm torch inside the vLLM container image.

## Capabilities & limitations

rocjitsu is a **functional** software emulator. What that means for AORTA:

* **Works:** single-GPU enumeration + real GPU kernels under emulation
  (validated: `gpu_smoke` triage recipe runs green on an emulated MI350X/gfx950
  with no physical GPU); multi-GPU *enumeration* (`device_count()` reflects the
  profile).
* **Pending upstream:** multi-rank RCCL collectives (torchrun with ≥2 ranks).
  rocjitsu's daemon is currently single-client, so distributed workloads that
  need a multi-rank process group do not run under emulation yet. This is
  tracked on the emulator side.
* **Out of scope:** the emulator is functional, not cycle/timing-accurate by
  default, and models no NIC/RDMA fabric — so it is a **functional-correctness
  and harness/CI** substrate, not a substitute for at-scale or timing-sensitive
  hardware testing.

## Testing

`tests/emulation/test_mirage_launch.py` covers (no GPU required): the
`Environment` round-trips the new keys, the built-in `emulated-rocjitsu`
resolves, sidecars accept the keys, emulation detection, argv wrapping +
passthrough, `$MIRAGE_BIN` resolution + error paths, and the `SubprocessWorkload`
opt-in wrap.
