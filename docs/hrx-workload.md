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
source) and an accessible ROCm GPU. `setup()` raises when `hipcc` is missing
**or** when no GPU is reachable (it checks that `/dev/kfd` is readable+writable,
the node HIP needs to initialise a device), so a CPU-only host or a container
started without `--device=/dev/kfd` classifies the cell as a setup failure /
`did_not_run` — it never reports a false `OUTPUT_NOT_WRITTEN`.

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

# HRX performance workload (`hrx_perf`)

Where `hrx` checks *correctness*, the companion `hrx_perf` workload measures
*speed* under HRX vs stock ROCm HIP. It builds a deliberately **big** HIP
benchmark with `hipcc` and execs it, timing each iteration host-side (launch +
`hipDeviceSynchronize`) so the per-step number includes runtime/launch overhead
— the axis on which an alternate HIP runtime can differ.

## Benchmarks

Select with `workload_config.bench`:

| `bench` | kind | what it stresses | throughput metric |
|---|---|---|---|
| `gemm` (default) | compute-bound tiled SGEMM (`C = A*B`, N×N floats) | FLOPs + launch path | `metrics.gflops` |
| `triad` | bandwidth-bound STREAM triad (`a = b + s*c`) | HBM bandwidth + launch path | `metrics.gbps` |

Each bench runs `warmup` untimed then `iters` timed iterations and verifies a
checksum (a wrong result → `PERF_FAIL`, so a bogus runtime can't yield a
meaningless "fast" number).

## How the comparison shows up

The workload reports every timed iteration as `step_times_ms`, so
`aorta sweep run`'s matrix renders:

- **Mean step (ms)** per cell — the mean timed iteration.
- **Confound** — `hrx_on`'s step time as a ratio of the `hrx_off` baseline.
  `speed (+N%)` means HRX was N% slower per iteration; `-` means no meaningful
  slowdown.

Achieved throughput (`gflops` / `gbps`) is in each trial's `result.json`
`metrics` and in `matrix.json`.

### `perf.md` — per-run performance report

Every `aorta sweep run` also writes a `perf.md` next to `matrix.md` /
`matrix.json` in the run directory (no flag needed — it is pure formatting of
data already collected). It carries a per-cell step-timing percentile table
(mean / std / min / max / p50 / p90 / p99 step ms + mean wall clock) and, when
a cell reported numeric `metrics`, a workload-throughput table (`gflops` /
`gbps` / `mean_step_ms`). The same aggregates are in
`matrix.json::cells[*].metrics_summary`. Read it alongside `matrix.md`:

```bash
cat triage_results/HRX-PERF-GEMM/hrx_perf/*/perf.md
```

Example (`hrx_off` baseline; `hrx_on` here shows `did_not_run` because the
recipe still has the placeholder HRX path — fill it in to get a real number):

```
| Cell    | ... | Iters | Mean step (ms) | Confound    |
|---------|-----|-------|----------------|-------------|
| hrx_off | ... | 50/50 | 9.6            | (baseline)  |
| hrx_on  | ... | —     | n/a            | did_not_run |
```

## Config keys

| key | default | meaning |
|---|---|---|
| `bench` | `gemm` | `gemm` or `triad` |
| `gpu_arch` | `gfx942` | `hipcc --offload-arch` target |
| `size` | gemm `4096`, triad `64000000` | matrix dim N (gemm) / element count (triad) |
| `iters` | `50` | timed iterations (reported as per-step times) |
| `warmup` | `10` | untimed warmup iterations |
| `hipcc` | `$HIPCC` / `/opt/rocm/bin/hipcc` / PATH | compiler |
| `build_dir` | temp dir | where bench binaries are built |
| `timeout_sec` | `600` | per-run subprocess timeout |
| `keep_build` | `false` | keep `build_dir` after cleanup |

HRX-on vs HRX-off routing, the `hipcc` build-env sanitization, and the
`LD_PRELOAD` guards (nonexistent path fails setup; ignored preload fails the
run) are identical to the `hrx` workload above.

## Quick run

```bash
# compute-bound A/B (add --strict to fail if a cell errors or never runs)
aorta sweep run --recipe recipes/hrx-perf-gemm.yaml --output ./triage_results --strict
cat triage_results/HRX-PERF-GEMM/hrx_perf/*/matrix.md

# bandwidth-bound A/B
aorta sweep run --recipe recipes/hrx-perf-triad.yaml --output ./triage_results --strict
cat triage_results/HRX-PERF-TRIAD/hrx_perf/*/matrix.md

# single run, stock HIP, defaults (gemm 4096)
aorta run --workload hrx_perf
```
