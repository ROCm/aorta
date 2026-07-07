# HRX HIP-launch workload

The `hrx` workload runs the HRX HIP-compatibility-layer launch probes so the
HRX runtime can be exercised and A/B-compared (HRX-on vs stock ROCm HIP)
through the normal `aorta run` / `aorta sweep run` workload flows.

It originates from the `ROCm/hrx-system` #156 / #158 / #160 investigation
(see issue #273).

## What it does

Each probe computes `out[i] = in[i] + 100` (with `in = 7`, `out` pre-zeroed)
over a single HIP kernel-launch path and prints a verdict derived from
`out[0]`:

| `out[0]` | verdict | meaning |
|---|---|---|
| `107` | `FULLY_WORKS` | read + write + copies all correct (the only pass) |
| `100` | `INPUT_READ_ZERO` | H2D / input argument broken |
| `0` | `OUTPUT_NOT_WRITTEN` | kernel write never reached the host buffer |
| other | `GARBAGE` | address mismatch |

Select the path with `workload_config.probe`:

| `probe` | HIP path under test |
|---|---|
| `static` | `hipLaunchKernelGGL` (statically registered) |
| `module` (default) | `hipModuleLaunchKernel` (pre-packed `extra` buffer) |
| `graph_add` | `hipGraphAddKernelNode` (`extra`) |
| `graph_setparams` | `hipGraphKernelNodeSetParams` (`extra`) |
| `graph_execsetparams` | `hipGraphExecKernelNodeSetParams` (`extra`) |

## How HRX-on vs HRX-off works

The workload does not decide the runtime. It builds the probe with `hipcc` and
execs it as a **child** process; which `libamdhip64.so` that child loads is an
environment concern set per cell:

- **hrx_off** — no routing; the child uses stock ROCm HIP.
- **hrx_on** — an `LD_PRELOAD` + `LD_LIBRARY_PATH` + `HRX_GPU_DRIVER` bundle
  points the child at HRX's `libamdhip64.so`. `LD_PRELOAD` is honored at
  `exec()`, which is exactly why the probe is a subprocess.

See `recipes/hrx-launch-probe-smoke.yaml` for a ready-to-edit A/B recipe.

## Config keys

| key | default | meaning |
|---|---|---|
| `probe` | `module` | which launch path (table above) |
| `gpu_arch` | `gfx942` | `hipcc --offload-arch` target |
| `hipcc` | `$HIPCC` / `/opt/rocm/bin/hipcc` / PATH | compiler |
| `build_dir` | temp dir | where probe binaries are built |
| `timeout_sec` | `120` | per-run subprocess timeout |
| `keep_build` | `false` | keep `build_dir` after cleanup |

## Prerequisites

A ROCm/`hipcc` toolchain (the workload compiles the probe from vendored
source). On a host without `hipcc`/GPU, `setup()` raises and the cell is
classified as a setup failure / `did_not_run` — it never reports a false
`OUTPUT_NOT_WRITTEN`.

## Quick run

The A/B matrix (both cells, one command) comes from the recipe:

```bash
# The ticket comes from the recipe (ticket: HRX-273-SMOKE); passing --ticket
# too would fail ("--recipe conflicts with --ticket").
aorta sweep run --recipe recipes/hrx-launch-probe-smoke.yaml --output ./triage_results
cat triage_results/HRX-273-SMOKE/hrx/*/matrix.md
```

`aorta run` has no per-workload-config flag, so it uses the default `module`
probe on stock HIP; select a different `probe` or route through HRX via a
recipe's `workload_config` / cell `extra_env` (as the recipe above does):

```bash
# default module probe, stock HIP, single run
aorta run --workload hrx

# route the child through an installed HRX prefix
aorta run --workload hrx --extra-env \
  "HRX_GPU_DRIVER=amdgpu,LD_LIBRARY_PATH=/path/to/hrx-root/lib,LD_PRELOAD=/path/to/hrx-root/lib/libamdhip64.so"
```
