# TokenSpeed under AORTA

Runs [TokenSpeed](https://github.com/lightseekorg/tokenspeed) — an AMD-optimized
LLM inference engine — under `aorta sweep`, on gfx950.

Everything in **this** document runs on the built-in `_subprocess` path (`aorta
sweep run` with `mode: probe`), which wraps an opaque command and never parses
its argv. That is the point: it shows aorta triaging a third-party engine it
knows nothing about, with no new Python.

The trade-off is that `mode: probe` carries a verdict but no metrics. Serving
*numbers* — TTFT, TPOT, throughput — need a workload class to reach
`WorkloadResult.metrics`, and that is the separate `tokenspeed_serve` workload
documented in [TokenSpeed serving benchmarks](tokenspeed-serving.md).

Two probes run today, in increasing order of usefulness-per-second:

| | What it exercises | Cost per trial | Recipe |
|---|---|---|---|
| **Kernel probe** | TokenSpeed's Gluon/Triton kernels via its own benchmark + numerics harness | ~17 s | `recipes/tokenspeed/tokenspeed-kernel-gemm-smoke.yaml` |
| **Serving probe** | `tokenspeed serve` bring-up: readiness, one completion, teardown | ~5 min, noisy | `recipes/tokenspeed/tokenspeed-serve-probe-smoke.yaml` |

Plus two paths that are not probes:

- harvesting kernel code objects and running them through aorta's **sanitizer**
  pipeline (Waitcheck, ConSan) — see [Sanitizers](#sanitizers);
- the **`tokenspeed_serve` workload**, which benchmarks serving performance —
  see [TokenSpeed serving benchmarks](tokenspeed-serving.md).

The scripts live in
[`src/aorta/workloads/tokenspeed/`](../src/aorta/workloads/tokenspeed/); that
directory's README is the per-file reference. `host_launch.sh` is the only entry
point aorta sees, and `TS_ENTRY` selects which in-container script it runs.

## Prerequisites

**gfx950 only.** TokenSpeed's platform check accepts gfx950 and gfx1250 and
raises on anything else, so MI300X (gfx942) cannot run any of this.

On a SLURM cluster, four things differ from a normal dev box and each one
produces a confusing failure if missed:

1. **Do not run on the head node.** Allocate first:

   ```bash
   salloc --no-shell -p interactive -N1 -t 12:00:00 -J tokenspeed-aorta
   srun --jobid=<id> --overlap bash -lc '...'
   ```

2. **A stale `~/.docker/config.json` breaks every pull.** If it holds a Docker
   Hub PAT that no longer authenticates, docker uses it instead of falling back
   to anonymous — so even `docker pull hello-world` fails with `authentication
   required`. Override it rather than editing the file:

   ```bash
   mkdir -p /tmp/dockercfg-anon && echo '{}' > /tmp/dockercfg-anon/config.json
   DOCKER_CONFIG=/tmp/dockercfg-anon \
     docker pull lightseekorg/tokenspeed-amd:nightly-20260714
   ```

   (The documented `:latest` tag does not exist; `nightly-20260714` and `tml`
   are what is published.)

3. **Nothing under a root-squashed `/home` can be bind-mounted.** The docker
   daemon runs as root, so on a root-squashed NFS export `-v /home/...` fails
   with `mkdir /home/<user>: permission denied`. Scripts, the HF cache, and
   probe output all have to live node-locally — `/tmp/ts-work/...`.
   `host_launch.sh` and `stage_scripts.sh` refuse `/home` paths up front instead
   of letting docker produce that error. Note `/tmp` is per-node: an `rm -rf
   /tmp/...` run on the login node does not clear the compute node's copy, and
   aorta will happily resume a cached run you thought you deleted.

4. **Compute nodes may predate aorta's Python floor (≥3.10).** A venv built on
   the login node also breaks, because its `bin/python` resolves to the node's
   interpreter and then cannot find its own `site-packages`. Provision an
   interpreter that travels with the venv:

   ```bash
   uv python install 3.11
   uv venv --python 3.11 ~/ts-aorta/.venv311
   uv pip install --python ~/ts-aorta/.venv311/bin/python -e path/to/aorta
   ```

Finally, stage the scripts node-locally. Re-run after **any** edit: a stale
staging dir fails every cell in a matrix identically, or — worse — passes while
silently running the previous version of the script.

Run it **on the compute node**, not the login node. `/tmp` is per-node, so
staging from the login node writes a directory the trials never see; if the
compute node already has an older copy, the run succeeds against the stale
scripts and the edit you were testing simply has no effect.

```bash
srun --jobid=$JOBID --overlap \
  bash src/aorta/workloads/tokenspeed/stage_scripts.sh
```

## Three things the container namespace breaks

All of these bit during bring-up and are easy to reintroduce, so they have
regression tests.

**Per-trial filenames cannot use `$$`.** `TS_OUT_DIR` is one host directory
shared by every trial in a matrix, so output files have to be unique per trial.
The obvious `$$` does not work: each `docker run` gets a fresh PID namespace, so
the entry script is always PID 1 and every trial writes the same filename. This
was observed as a 12-trial matrix producing exactly one export file, with the
first eleven trials' evidence overwritten — and the matrix still passed, because
each trial reads back its own file before the next one starts. So it fails
silently, in the direction of losing data. `host_launch.sh` therefore mints
`TS_RUN_TOKEN` *host-side* (where the PID is real) and the entry scripts name
`kernel_bench.<token>.json`, `server.<token>.log`, and
`completion.<token>.json` after it.

**The container must not run as root.** `harvest_code_objects.py` passes
`--user`, because a root-owned Triton cache leaves the calling user unable to
delete its own run area — including the harvest script's own cleanup, so a
*second* harvest fails with `EPERM`. `--user` alone is not enough: Triton calls
`getpass.getuser()`, which falls through to `pwd.getpwuid()` and raises
`KeyError: getpwuid(): uid not found` for a uid absent from the container's
`/etc/passwd`, so `USER` and `HOME` are passed too.

**An unwritable JIT cache is reported as an unsupported GPU.** If a bind-mounted
`TRITON_CACHE_DIR` is not writable by the container user, TokenSpeed fails with:

```
RuntimeError: Triton is not supported on the current platform.
Only NVIDIA CUDA and AMD HIP backends are supported.
```

That message sends you to the GPU, and the GPU is fine. Triton initialises its
AMD driver by compiling a small HIP utility into the cache directory; the real
error is a `PermissionError` from `os.makedirs`, which TokenSpeed swallows with a
bare `except BaseException` in `get_available_device()` and re-raises as the
message above. The usual cause is the mount point not existing on the compute
node, so docker auto-created it as root — remember `/tmp` is per-node, so
creating it on the login node does not create it where the container runs.
`harvest_code_objects.py` write-probes the cache directory up front and says so
plainly instead.

## Kernel probe

The cheap path, and the one to reach for first. It drives TokenSpeed's kernels
through TokenSpeed's own `tokenspeed_kernel.benchmark` / `.numerics` harnesses:
no weights, no server, no readiness polling.

```bash
export TS_IMAGE=lightseekorg/tokenspeed-amd:nightly-20260714
export TS_SCRIPTS_DIR=/tmp/ts-work/scripts
export TS_ENTRY=ts_kernel_probe.sh TS_GPUS=1

aorta sweep run \
  --recipe recipes/tokenspeed/tokenspeed-kernel-gemm-smoke.yaml \
  --mitigations-file recipes/tokenspeed/tokenspeed-kernel-sidecar.json \
  --output-dir /tmp/ts-work/kernel-out --ticket TOKENSPEED-KERNEL \
  -- bash /tmp/ts-work/scripts/host_launch.sh
```

The matrix is **solution × ROCm knob**: rows pin a `gemm.mm` solution (Gluon vs
torch) via the sidecar's `TS_KERNEL_NAME`, columns apply a launch knob
(`hip_launch_blocking`). Since `mode: probe` runs one argv across all cells,
per-cell kernel selection has to travel as environment — hence a sidecar
"mitigation" that is really a kernel selector. `env_passthrough_mode: file` is
mandatory: `docker run` does not inherit `os.environ`, so `inherit` would make
every cell silently run the same default kernel.

Measured on `nightly-20260714`, 4 cells × 3 trials, all 12 passing in 3 m 30 s.
Figures are the median across the three trials at the 4096³ shape; the spread
across trials was under 1.5 µs in every cell.

| Solution | Diagnostic | p50 @ 4096³ | TFLOPs |
|---|---|---|---|
| `gluon_mm_a16w16_gfx950` | none | 107.2 µs | 1282 |
| `gluon_mm_a16w16_gfx950` | `hip_launch_blocking` | 139.6 µs | 984 |
| `torch_mm` | none | 106.9 µs | 1286 |
| `torch_mm` | `hip_launch_blocking` | 131.3 µs | 1047 |

The diagnostic axis is doing real work here: `hip_launch_blocking` costs ~30%
throughput, which is the expected shape of serializing every launch. That is
the axis behaving correctly, not a regression — but it means the two columns are
not comparable as perf numbers, only within a column.

Each trial covers 5 shapes (M ∈ {1, 16, 128, 512, 4096} at N=K=4096). The small-M
shapes are bandwidth-bound (~1.6-1.8 TB/s at M ≤ 16) and the large-M shapes
compute-bound, so a numerics or perf regression that only affects decode-shaped
work shows up in the same trial as one affecting prefill.

Per-shape numbers land on stdout as `TS_KERNEL_METRIC` lines
(`median_latency_us`, `p99_latency_us`, `tflops`, `bandwidth_gb_s`,
`numerics_passed`). They are **not** in `result.json`'s `metrics` dict — the
probe path has no such channel. The `Mean step (ms)` column in `matrix.md` is
whole-container walltime (docker start + JIT + benchmark, ~17 s), not kernel
time; do not read it as a perf signal.

### The verdict does not come from the exit code

`tokenspeed_kernel.benchmark` ends in an unconditional `return 0` — it exits
successfully even when `--verify` finds a numerics mismatch. Trusting it would
turn a wrong-answer kernel into a green cell. So `ts_kernel_probe.sh` re-reads
the exported JSON and fails the trial itself (exit 32) when any record has
`numerics_passed == false`. `tokenspeed_kernel.numerics` does return 1 properly,
which is why `TS_KERNEL_MODE=numerics` can trust its exit code.

`numerics_passed: null` means *not checked*, not failed — the reference
solutions (`torch_mm`) report null because they are what everything else is
compared against.

### Only `gemm.mm` is drivable

Surveyed on `nightly-20260714` with `list_harness_coverage.py` (kept in the
workload directory so this can be re-checked against a newer image): 21 operator
families with 40 registered kernels, of which **1 family can actually be run**.

| Status | Families |
|---|---|
| runnable | `gemm.mm` (9 kernels) |
| no input generator | the other 20, including every attention kernel (`mha_prefill`, `mha_decode_with_kvcache`, `mla_*`, `dsa_*`, `gdn_chunk_prefill`), `moe.apply` (9 kernels), all `quantization.*`, `sampling.argmax`, `embedding.rope*`, `transform.hadamard_transform` |

The harness needs three registries to line up — a kernel, an input generator,
and a shape list — and only `gemm.mm` has all three. `--op
attention.mha_prefill` dies with `KeyError: No standard shapes registered for
attention.mha_prefill. Known: gemm.mm` before any kernel launches. The
generators that *do* exist for `moe.align_block_size` and `quantize.fp8_*` are
keyed to operator names that no registered kernel uses (`moe.apply`,
`quantization.fp8`), so they match nothing either.

This is an upstream TokenSpeed gap, not an aorta one. The recipes name it as its
own detector (`ts_kernel_no_input_generator`, `ts_kernel_no_standard_shapes`) so
pointing one at an unsupported operator is self-diagnosing.

It is **not**, however, the binding constraint on kernel coverage. TokenSpeed's
own pytest suites build these inputs themselves, so they reach what the
benchmark harness cannot — see [Suite probe](#suite-probe). Use the benchmark
harness when you want numbers and the suites when you want coverage.

## Suite probe

The benchmark harness reaches 8 AMD-relevant kernels. Driving TokenSpeed's own
op test suites instead reaches **32 of 37**, because those suites construct
their own inputs rather than going through the input-generator registry.

(38 distinct kernel names are registered, of which `cublaslt_mm_nvfp4` is a
cuBLASLt path and cannot run here, leaving 37. The "40" above counts
family/kernel pairs, and two kernels are registered under two families each.)

```bash
export TS_IMAGE=lightseekorg/tokenspeed-amd:nightly-20260714
export TS_SCRIPTS_DIR=/tmp/ts-work/scripts
export TS_ENTRY=ts_pytest_probe.sh TS_GPUS=0

aorta sweep run \
  --recipe recipes/tokenspeed/tokenspeed-kernel-suites-smoke.yaml \
  --mitigations-file recipes/tokenspeed/tokenspeed-pytest-sidecar.json \
  --output-dir /tmp/ts-work/suites-out --ticket TOKENSPEED-SUITES \
  -- bash /tmp/ts-work/scripts/host_launch.sh
```

The trade-off is that these suites *assert*; they do not measure. There are no
TFLOPs or latency figures here — use the kernel probe for gemm perf. What you
get is a correctness verdict per operator family, and a populated JIT cache that
[the sanitizer path](#sanitizers) can turn into a Waitcheck corpus covering
attention and MoE rather than only gemm.

### What the suites actually cover

Measured with `map_kernel_test_coverage.py`, which wraps
`KernelRegistry.get_for_operator` and runs the suites so a *resolved* kernel can
be told from a *skipped* one. Static inspection cannot answer this, because the
tests parametrize over solutions and skip at run time.

| Route | Kernels | Detail |
|---|---|---|
| Registry-mediated suites | 23 | attention 15, `moe.apply` gluon 4, quantization 3, transform 1 |
| Benchmark harness | 8 | `gemm.mm`, AMD-relevant |
| Direct-import AMD suites | 1 | `gluon_argmax_gfx950` |
| **Reachable** | **32 / 37** | |
| No execution test anywhere | 5 | `triton_*_moe_apply` variants |
| Out of scope | 1 | `cublaslt_mm_nvfp4` (cuBLASLt, NVIDIA) |

Read that tool's `covered` as "resolved through the registry", not "tested". It
under-reports in two ways worth knowing. `tokenspeed-kernel-amd/test/ops`
imports implementations directly and never consults the registry, so kernels it
exercises look uncovered — `gluon_argmax_gfx950` passes 20 tests yet shows as
uncovered. And expert-parallel (`_ep_`) solutions skip when too few GPUs are
visible. The 5 Triton MoE variants are a real gap: `moe.apply::gluon` is the only
MoE solution any test ever requests, and the `_ep_` names appear solely in
`test_kernel_api_selection.py`, which tests selection logic rather than
executing anything.

### These suites skip heavily, and pytest exits 0 when they do

A single file reports hundreds of skips, because NVIDIA-only solutions
(`flashinfer`, `fa4`, `cuda`) are not registered on AMD. That is expected and
harmless. What is not harmless is that pytest exits 0 when *everything* skips or
is deselected, which would turn a cell that proved nothing into a green one. So
`ts_pytest_probe.sh` counts executed tests from the JUnit report and fails the
trial when none ran (`ts_pytest_nothing_executed`, exit 41). Counts come from
the report rather than the terminal summary because the summary wording shifts
between pytest versions.

Measured on `nightly-20260714`, one GPU, 12 cells (6 suites × 2 diagnostics),
all passing in 3 m 29 s. Counts were identical in both columns, so only the
`none` column is shown; the walltime is the in-container pytest time and
excludes container start and JIT.

| Suite | Passed | Skipped | Walltime |
|---|---|---|---|
| `test_attention.py` | 46 | 84 | 21 s |
| `test_attention_dsa.py` | 13 | 0 | 12 s |
| `test_attention_mla.py` | 6 | 0 | 9 s |
| `test_moe_gluon_bf16_gfx950.py` | 9 | 0 | 8 s |
| `test_quantization.py` | 17 | 9 | 5 s |
| `test_sampling_gluon_gfx950.py` | 20 | 0 | 5 s |

`hip_launch_blocking` changed no verdict and no count here, which is the
expected result for assertion suites — it serializes launches, so it would only
change an outcome if the failure were an async-ordering one. It is kept as a
column so that when a suite *does* go red, the two cells immediately separate "the
kernel is wrong" from "the pipelining around it is wrong".

## Serving probe

Bring-up triage only: does the engine come up on this stack, and can it produce
a token. For serving *performance* on the same engine, use the
[`tokenspeed_serve` workload](tokenspeed-serving.md) instead — this probe cannot
report a latency or a throughput.

```bash
export TS_IMAGE=lightseekorg/tokenspeed-amd:nightly-20260714
export TS_SCRIPTS_DIR=/tmp/ts-work/scripts
export TS_MODEL=Qwen/Qwen3-0.6B TS_GPUS=0

aorta sweep run \
  --recipe recipes/tokenspeed/tokenspeed-serve-probe-smoke.yaml \
  --output-dir /tmp/ts-work/probe-out --ticket TOKENSPEED-PHASE0 \
  -- bash /tmp/ts-work/scripts/host_launch.sh
```

`tokenspeed serve` never exits, so handing it directly to aorta would always hit
`timeout_per_trial` and classify as `tier1:timeout` — an *error*, not a verdict.
`ts_serve_probe.sh` gives the probe something that terminates.

**It is an orchestrator, not one server.** `tokenspeed serve` spawns an smg
gateway (the OpenAI-compatible `/v1` surface, on `--port`) in front of a gRPC
engine, plus a TokenSpeed control server that owns `/health` and
`/health_generate` on `--port + 1`. Polling `/health` on the gateway port just
logs 5xx from the gateway. The script passes `--control-port` explicitly so
readiness never rests on that implicit offset, checks readiness on the control
port, and sends the completion to the **gateway** port — the path a real client
takes, so a gateway that is up but not wired to the engine still fails.

Teardown signals the whole process group, with a 45 s grace period. That has to
exceed TokenSpeed's own 30 s gateway drain; at 20 s every teardown escalated to
SIGKILL, leaving the next cell racing to reclaim the KV cache.

Measured: both cells pass, `failure_detectors_fired: []`, ~10-11 min for the
2-cell matrix. Teardown exited cleanly on SIGTERM every time, with no escalation
to SIGKILL.

Startup to `/health` is the dominant cost and it is **noisy**: 189, 276, 285,
291, 316 and 319 seconds across six runs of the same recipe on the same node —
a 1.7× spread with nothing changed between them. So do not treat a slow
bring-up as a signal without repeats, and do not tune `timeout_per_trial` close
to an observed number; the recipe's 1800 s leaves deliberate headroom.

That this is minutes for a *0.6B* model is the surprising part: it is not weight
loading. It is dominated by allocating the default 250 GB KV cache and a 537 GB
pinned host tier, both roughly independent of model size, so a much larger model
is not proportionally slower to bring up.

Known gaps on this path:

- `peak_vram_mib` is `null`. The work happens inside a container under a PID
  aorta is not watching.
- `env.json` is host-scoped and reports `torch not importable`,
  `triton not importable`. ROCm and PyTorch live in the container, so the
  environment aorta records is not the environment the workload ran in.

## Sanitizers

TokenSpeed's canonical kernels run through aorta's **existing** sanitizer
infrastructure — no sanitizer changes were needed.

TokenSpeed's kernels are Gluon/Triton, so they do not exist as committed
binaries — they are JIT-compiled into the Triton cache on first use. aorta's
sanitizer pipeline consumes code objects by path plus a SHA-256 identity
(`source.kind: kernel_list`). `harvest_code_objects.py` bridges the two: run the
kernel once with a clean cache, collect the `.hsaco`, inventory it with
`rj_waitcheck --list-kernels`, and write a ready-to-run recipe.

First provision the RocJITsu binaries (see the
[sanitizer README](../src/aorta/instrumentation/rocjitsu_sanitizers/README.md) —
needs a GitHub token with `actions:read`):

```bash
python download_sanitizer_artifacts.py --dest ./rocjitsu-sanitizers
export ROCJITSU_PREBUILT="$PWD/rocjitsu-sanitizers"
```

Then harvest and run:

```bash
python3 src/aorta/workloads/tokenspeed/harvest_code_objects.py \
  --image lightseekorg/tokenspeed-amd:nightly-20260714 \
  --kernel gluon_mm_a16w16_gfx950 --dtype bf16 --dtype-role a \
  --dest /tmp/ts-work/sanitizer-run --gpus 2 \
  --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck" \
  --docker-config /tmp/dockercfg-anon

aorta sweep run \
  --recipe /tmp/ts-work/sanitizer-run/waitcheck-gluon_mm_a16w16_gfx950.yaml \
  --output-dir /tmp/ts-work/sanitizer-out
```

`ROCJITSU_PREBUILT` has to be exported in the shell that runs `aorta sweep run`,
not only the one that harvests. aorta discovers the backend from it (then
`$ROCJITSU_BUILD`, then `PATH`), and without it the run completes with
`overall verdict: not_checked` and every kernel `not_checked` — which the
guardrail rejects, but which reads as a sanitizer failure rather than a missing
binary.

### Reaching attention and MoE, not just gemm

`--kernel` / `--op` drive the benchmark harness, so they can only ever harvest
`gemm.mm`. `--pytest-suite` drives one of TokenSpeed's own suites instead, which
is the only way to compile the attention, MoE, quantization, sampling and
transform kernels — and therefore the only way to get them under Waitcheck:

```bash
python3 src/aorta/workloads/tokenspeed/harvest_code_objects.py \
  --image lightseekorg/tokenspeed-amd:nightly-20260714 \
  --pytest-suite tokenspeed-kernel/test/ops/test_attention.py \
  --pytest-k mha_prefill \
  --dest /tmp/ts-work/attn-run --gpus 5 \
  --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck" \
  --docker-config /tmp/dockercfg-anon

aorta sweep run \
  --recipe /tmp/ts-work/attn-run/waitcheck-test_attention.yaml \
  --output-dir /tmp/ts-work/attn-sanitizer-out
```

Measured on `nightly-20260714`: that selection harvests **20 code objects and
all 20 pass Waitcheck** (`state: ran`, `verdict: pass`, no findings), covering
`_mha_prefill` ×6, `_mha_prefill_sliding` ×4 and `_fwd_kernel` ×10.

The multiplicity is the point, and it is why staging is content-addressed. The
Triton cache keeps one directory per shape specialization and reuses file names
across them, so ten *different* `_fwd_kernel.hsaco` exist. Staging them by bare
name would overwrite them in turn, leaving a recipe whose digests match nothing
on disk except the last copy — Waitcheck would then reject every earlier entry
for a digest mismatch. Staged names therefore carry a digest prefix
(`_fwd_kernel.9918ec158129.hsaco`), and byte-identical objects collapse to one
identity.

The generated recipe and the harvested objects are a **run area, not something
to commit**: a harvested object is specific to the image, the GPU target, and
the shapes that were compiled, so pinning one in git would assert a provenance
it does not have. Re-harvest instead — it takes seconds.

Result — one `gluon_mm_a16w16_gfx950` benchmark emits two shape-specialized
objects, and both pass Waitcheck. The CLI itself is terse:

```
sanitizer report: /tmp/ts-work/sanitizer-out/sanitizer_report.json
overall verdict: pass (execution complete)
```

The verdict worth reading is in `sanitizer_report.json`, under
`checks[].kernel_results[]`:

```json
{
  "state": "ran",
  "verdict": "pass",
  "returncode": 0,
  "findings": [],
  "identity": {
    "name": "_mfma_lds_mediumm_kernel",
    "target": "gfx950",
    "code_object": ".../code_objects/_mfma_lds_mediumm_kernel.hsaco",
    "code_object_sha256": "0b20ac465fa485c0243028a89c404021b438acefbbc243ec5267060387969aa7"
  }
}
```

Check `state: ran` and a populated `code_object_sha256`, not just the overall
verdict: `not_checked` also renders as a non-failure, so "no findings" and "never
looked" are indistinguishable from the top-level line alone. The digest is what
proves the object analyzed is the object harvested. `checks[].backend` records
the `rj_waitcheck` binary's own SHA-256 alongside it, so a report identifies both
sides of the analysis.

Two caveats on this output. `coverage` is `[]` and `entry_offset` is `null` on the
`kernel_list` path — the digest plus `code_object_index` is the whole identity, so
do not expect per-instruction coverage detail here. And re-harvesting the same
kernel on the same image and node reproduces byte-identical digests, which is
what makes a report comparable across runs.

### ConSan reaches gemm, not attention

ConSan used to be unreachable here: it only runs against a caller-supplied
`source.consan_command`, and the documented loader pattern assumes a
hand-compiled HIP file rather than a Triton object with a Triton launch ABI, so
adding `consan` to a recipe could only ever return
`not_checked: consan_command_not_provisioned`. That was
[#399](https://github.com/ROCm/aorta/issues/399), closed by
[#403](https://github.com/ROCm/aorta/pull/403), which added a generic loader
that resolves a Triton cache entry at run time through ctypes.

`--consan` wires the harvest into it:

```bash
python3 src/aorta/workloads/tokenspeed/harvest_code_objects.py \
    --image lightseekorg/tokenspeed-amd:nightly-20260714 \
    --kernel gluon_mm_a16w16_gfx950 --dtype bf16 --dtype-role a \
    --dest /tmp/ts-work/consan-gemm \
    --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck" \
    --consan

for recipe in /tmp/ts-work/consan-gemm/consan/consan-*.yaml; do
    aorta sweep run --recipe "$recipe" \
        --output-dir "/tmp/ts-work/consan-out/$(basename "$recipe" .yaml)"
done
```

One recipe per code object, not one over the list: ConSan takes exactly one
object per run, and a TokenSpeed kernel compiles to several shape-specialized
objects. `--consan-limit` caps the fan-out, which matters because an attention
harvest yields 20. The shims are emitted with `--copy-object`, lifting each
object and its sidecars out of the Triton cache before the next harvest deletes
it.

On the gemm kernels this works, with full static instrumentation:

```
pass | consan ran/pass | access 208/208 | analysis_complete=True | _mfma_lds_largem_kernel
pass | consan ran/pass | access  77/77  | analysis_complete=True | _mfma_lds_mediumm_kernel
```

**On the attention kernels it does not**, and the reason is upstream rather than
here. All 20 harvested objects discover their sites, report them supported, and
then fail to lower every one of them — 2502 sites, `access_patched=0`,
`lowering_reason=instrumentation_patch_missing`. Worse, the hook counts barrier
sites without itemizing them on that path, so aorta's coverage cross-check fails
closed with `consan_output_parse_error: barrier site count mismatch` and the real
number never reaches the report. Filed as
[#405](https://github.com/ROCm/aorta/issues/405).

Waitcheck is unaffected and passes on all 20, so these kernels are reachable —
ConSan simply cannot instrument them yet.

#### Why the recipes default to `consan_policy: lenient`

`strict` sets `RJ_CONSAN_MOI_REQUIRE_RECORDS`, which demands visible dynamic
records. The loader runs in `load` mode: it loads and instruments the object but
never dispatches it, so there is no dispatch packet and no records, and strict
fails closed with `combined_hook_exit_86` however healthy the run was — measured,
not assumed:

```
ConSan analysis verdict applicable=true static_complete=true dynamic_complete=false
                        access=77/77 barrier=12/12 visible_evidence=0
RJ_CONSAN_MOI_REQUIRE_RECORDS requested, but 1 auto MOI report buffer(s)
contained zero visible records and no kernel dispatch packet was observed
```

So this lane verifies that ConSan can read, patch and analyze the JIT kernels —
static coverage — and nothing about races at run time.

Getting dynamic evidence is not a policy change. `dispatch` mode needs the
argument signature, and Triton does not write one into the metadata for these
kernels (confirmed against real cache entries: `num_warps`, `warp_size` and
`shared` are present, `signature` is not). Supplying one by hand for a kernel
with a 320-byte kernarg segment means reconstructing TokenSpeed's launch, and
synthesized scalars default to zero, which gives a zero-trip kernel that records
nothing anyway. The route that would actually work is ConSan over TokenSpeed's
own dispatches, which needs the RocJITsu entry-point allowlist that
`src/aorta/instrumentation/rocjitsu_sanitizers/README.md` already tracks as
unavailable.

## Tests

`tests/probe/test_tokenspeed_probe.py` — 50 tests, no GPU or container required.
They cover script syntax, the guardrails (NFS refusal, missing entry script,
missing selector), recipe and sidecar wellformedness, per-trial output naming,
that every recipe axis entry resolves through the real mitigation registry, and
that nothing in the recipe directory is gitignored.

Two gates are covered against stubs rather than real kernels, because both guard
against a *silent* pass that the shipped code cannot be made to produce on
demand:

- the **numerics gate**, against a synthetic benchmark export. The shipped
  kernels pass, and a numerics mismatch is precisely the verdict the upstream
  CLI does not report through its exit code, which is why the gate exists.
- the **nothing-executed gate**, against a stub suite of only-skipped and
  deselected tests. pytest exits 0 in both cases, so without it an empty cell
  would be indistinguishable from a verified one.

The pytest-probe tests run the real pytest against a temporary stub workspace,
via a `python3` shim on `PATH` pointing at the current interpreter — the script
calls `python3 -m pytest`, and the bare `python3` on `PATH` is not necessarily
the one running the tests.

```bash
python -m pytest tests/probe/test_tokenspeed_probe.py -q
```

## Not done yet

- **Attention/MoE *performance*.** Correctness and Waitcheck coverage are done
  via [the suite probe](#suite-probe), but numbers still need the benchmark
  harness, which needs input generators — see
  [Only `gemm.mm` is drivable](#only-gemmmm-is-drivable). Note
  `set_input_generator`, `set_standard_shapes` and `set_benchmark_shapes` are
  public, so generators could be registered from a plugin without forking
  TokenSpeed; upstream is the better home.
- **5 Triton MoE variants.** No execution test exists anywhere upstream, so
  neither route reaches them — see
  [What the suites actually cover](#what-the-suites-actually-cover).
- **ConSan on the attention kernels.** Works on gemm; every site fails to lower
  on attention, and the failure is reported as a parse error rather than as
  incomplete coverage. Blocked on
  [#405](https://github.com/ROCm/aorta/issues/405).
- **Dynamic race evidence anywhere.** The loader instruments but does not
  dispatch, so every ConSan lane here is static-coverage only — see
  [Why the recipes default to `consan_policy: lenient`](#why-the-recipes-default-to-consan_policy-lenient).
- **A larger model.** `gpt-oss-20b` is TokenSpeed's canonical AMD benchmark
  model; only Qwen3-0.6B has been run.
