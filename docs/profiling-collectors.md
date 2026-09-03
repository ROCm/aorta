# Profiling Collectors (`rocprof` / `proton`)

Attach a GPU profiler to a command aorta runs, without editing the command.
Both collectors work by rewriting the launch argv — the same seam the mirage
emulator uses — so an opaque `aorta sweep run ... -- <command>` gets profiled
and the payload never learns it is being measured.

This page is the reference: every option, the artifact layout, and the
troubleshooting table. It does not tell you *which* Proton backend to pick —
for that, and for what each one costs you, see
[Choosing a Proton backend](proton-backends.md).

How wide "a command" is differs by collector, and it is the first thing to
check: `rocprof` wraps **anything**, while Proton's default `mode: cli` takes
over a Python *script*, so it attaches only to `python ... <script>.py`, a
bare `pytest`, or `python -m pytest`. Proton's `mode: env` accepts any argv
but is not edit-free — it hands the payload `AORTA_PROTON_*` variables that
the payload itself must act on by calling `proton.start()` / `proton.finalize()`.
See [Proton attach modes](#proton-attach-modes).

Two collectors ship today:

- **`rocprof`** — runs the whole command under `rocprofv3` (ROCm's
  kernel/API tracer). Wraps *any* command: a HIP binary, `torchrun`, a
  shell script. Writes CSV / JSON / pftrace / OTF2 / rocpd artifacts.
- **`proton`** — runs a Python launch under Triton's Proton profiler, which
  attributes GPU time back to the **launch site** (Python frame or shadow
  frame) rather than to a mangled kernel symbol. Writes a `.hatchet` tree.

Both parse their own artifacts after the run and merge a small set of flat
numeric metrics into the trial's `metrics`, so `rocprof_gpu_time_ms` and
friends land in `perf.md` and `matrix.json` with no report-side changes.

## When to use them

This is a **measurement** step, not a detection step. It answers *where the
GPU time went* and *which kernels ran*, for a command you can already run.
It does not classify a failure, and it does not decide whether a mitigation
worked — the sweep verdict does that.

- A mitigation changes wall time and you want to know **which kernel** moved.
- You need a kernel/dispatch trace of a repro you only have as an opaque
  launch command, and you do not want to fork the launch script to add a
  profiler.
- You want a Triton kernel's time attributed to the **Python line that
  launched it** (`proton`), not to `triton_poi_fused_add_0`.
- You want the same capture taken identically across every cell of a sweep,
  next to the trial JSON, so two cells are comparable.

What it is **not**: a passthrough. `--collect rocprof` changes the child's
stderr (see [Troubleshooting](#troubleshooting)) and both collectors add
launch overhead. Do not enable a collector on a timing-sensitive race repro
and then conclude the race is gone.

## Prerequisites

For `rocprof`:

- `rocprofv3` on `$PATH`, or `$ROCPROF_BIN` pointing at it. It ships with
  ROCm. A missing binary is a hard setup failure, not a silent unprofiled
  run.
- Nothing else. The collector is flags-only; no Python dependency.

For `proton`:

- Triton importable **from the interpreter the workload runs under**, since
  the collector attaches as `<python> -m triton.profiler.proton`. Proton
  ships inside Triton; there is no separate install.
- The command must be a Python script launch (or `pytest`) for the default
  `mode: cli`. Anything else needs `mode: env` — see
  [Attach modes](#proton-attach-modes).
- The `proton` / `proton-viewer` console scripts are **not** used by the
  collector, on purpose: on a typical host those shims are shebanged to
  whichever interpreter installed Triton, which is usually not the one the
  workload runs under. `$AORTA_PROTON_PYTHON` overrides the interpreter
  choice when Triton lives somewhere else again.

Verified against ROCm 7.0.2.2 / `rocprofv3` 1.0.1 on gfx950.

## Standalone usage (supported path)

Both collectors are thin wrappers over tools you can run by hand, and doing
so is the fastest way to separate "my payload is broken" from "my profiler
is broken". Run the payload bare first, then under the profiler, then under
aorta.

`rocprofv3` directly:

```bash
rocprofv3 --kernel-trace --stats --output-format csv \
  -d /tmp/rocprof_out -o run -- ./my_gpu_binary --size 512
ls /tmp/rocprof_out
# run_kernel_stats.csv  run_kernel_trace.csv  run_domain_stats.csv
# run_agent_info.csv
```

Proton directly, from an interpreter that has Triton:

```bash
python -m triton.profiler.proton -n /tmp/proton_out/run \
  --context shadow --data tree my_script.py --iters 20
proton-viewer -m time/s /tmp/proton_out/run.hatchet
```

Note the argument shapes differ and the collectors preserve that: `rocprofv3`
takes a `--` separator before the command, Proton's front-end collects the
script and its arguments with `argparse.REMAINDER` and takes none.

### Startup sanity check

Before trusting a capture, confirm artifacts exist. `rocprofv3` writes
**nothing at all** when the profiled command dispatched no GPU work, and
Proton writes no tree when the process never launched a kernel — in both
cases the trial still passes and simply carries no numeric collector
metrics. So an empty collector directory means "no GPU work was seen", not
"the run failed". Collector metrics live under `.result.metrics` in the
per-trial JSON (see [Output](#output) for where that file sits):

```bash
jq -r '.result.metrics | to_entries[]
       | select(.key | startswith("rocprof_"))
       | "\(.key)=\(.value)"' <run_dir>/<cell>/<workload>/trial_d0_m0_t0.json
```

If only `rocprof_artifact_dir` comes back, the profiler attached but nothing
*parseable* came out of it. That is the expected result for a command that
dispatched no kernels, but parsing is fail-soft, so it also covers a
non-CSV `output_format` (the parser reads the CSVs), a partial capture from
a killed run, and a malformed one. Look in the artifact directory the metric
points at before concluding the payload did no GPU work.

## Configuration

Collectors are selected either on the command line or in the recipe.

```bash
# Names only; overrides any recipe-pinned collect:.
aorta sweep run --recipe r.yaml --collect rocprof -- ./my_gpu_binary
aorta sweep run --recipe r.yaml --collect proton -- python train.py
```

`--collect` is *accepted* on both the workload (triage) flow and the
subprocess (probe) flow, but the two argv-wrapping profilers only *measure*
on the subprocess flow — both commands above are subprocess-flow invocations,
which is what the trailing `-- <command>` marks. `wrap_argv_for_collectors`
and `summarize_collectors` are each called from one place,
`SubprocessWorkload`, so on an in-process workload `collect: [rocprof]`
validates and lands in the trial config and then does nothing: no argv wrap,
no artifact directory, no `rocprof_*` metrics. This is the one
requested-but-not-measured path here that does not announce itself, unlike a
missing `rocprofv3`, an unpreparable artifact directory, or a missing `env(1)`,
which all fail the trial's setup. An in-process workload can still opt in by
reading `_aorta_collect` itself, the way `layer_numerics` does.

Note that `--collect rocprof,proton` is rejected: with no options to carry,
the CLI flag gets Proton's default `backend: auto`, which resolves to a
queue-intercepting backend on AMD and so conflicts with `rocprofv3` (see
[Combining collectors](#combining-collectors)).

```yaml
# Recipe, list form -- names only, default options.
collect: [rocprof]
```

```yaml
# Recipe, mapping form -- the only way to set per-collector options.
collect:
  rocprof:
    trace: "kernel,hip"
    output_format: "csv"
    summary_units: "usec"
```

An empty / null value in the mapping form means "enabled, no options":

```yaml
collect:
  rocprof:            # enabled, defaults
```

Precedence and scope:

- `collect:` is **cross-cutting**, not a matrix axis: the same collectors
  run in every cell.
- A cell may set its own `collect:`. Absent inherits the recipe level, a
  present value replaces it for that cell, and `collect: []` disables
  collection for that cell.
- A `mode: probe` recipe accepts the same top-level `collect:` block, but has
  no cell scope to override at — it synthesises its cells from
  `mitigation_axis x diagnostic_axis`, so the recipe-level collectors apply
  to every one of them.
- CLI `--collect` is an operator override applied to every cell; it clears
  per-cell overrides. It selects **names only** — there is no CLI syntax for
  per-collector options, so a run that needs new options needs a recipe. The
  recipe's options for the collectors that survive the override are kept, so
  `--collect rocprof,proton` against a recipe that pins
  `proton: {backend: instrumentation}` runs with that backend (and is
  accepted, because those options are what resolve the conflict below).
  Options for collectors the override dropped are discarded with them.
- Every option is validated at recipe-load time. A typo fails the whole
  recipe up front rather than producing a run with no measurements in it.

### `rocprof` options

| Option | Values | Default | Notes |
|---|---|---|---|
| `trace` | comma- or space-separated: `kernel`, `hip`, `hip_runtime`, `memory_copy`, `rccl`, `marker`, `scratch` | `kernel` | Maps one-to-one onto `--kernel-trace`, `--hip-trace`, `--hip-runtime-trace`, `--memory-copy-trace`, `--rccl-trace`, `--marker-trace`, `--scratch-memory-trace`. Must name at least one domain. `kernel` is the only domain the summary parser aggregates. |
| `output_format` | `csv`, `json`, `pftrace`, `otf2`, `rocpd` | `csv` | Only `csv` yields the numeric metrics below — the parser reads CSV. The others are for out-of-band analysis. |
| `stats` | `1/true/yes/on` or `0/false/no/off` | `true` | Adds `--stats`, i.e. the pre-aggregated `*_kernel_stats.csv`. The parser falls back to summing dispatch spans from the kernel trace whenever the stats CSVs are absent (`stats` off) or yield no usable rows (a capture truncated mid-write, or a column this parser pins renamed by a future rocprofv3). |
| `pmc` | comma- or space-separated counter names | (unset) | Hardware counters. Rendered last before the `--` separator because `--pmc` is variadic. Must name at least one counter. |
| `kernel_include_regex` | regex | (unset) | `--kernel-include-regex`; must be non-empty. |
| `summary_units` | `sec`, `msec`, `usec`, `nsec` | (unset → rocprofv3's own default) | Unit for the human-readable summary file only. Does not change the parsed metrics, which are always ms. |

### `proton` options

The `backend` row below is the schema. For which backend answers which question
— and for the version matrix, the two opposite initialisation contracts, and the
measured sharp edges — see [Choosing a Proton backend](proton-backends.md).

| Option | Values | Default | Notes |
|---|---|---|---|
| `mode` | `cli`, `env` | `cli` | See [Attach modes](#proton-attach-modes). |
| `backend` | `auto`, `rocprofiler`, `roctracer`, `instrumentation`, `cupti` | `auto` | `auto` omits Proton's `-b` and lets Proton pick the backend matching the active runtime. On AMD that resolution is version-dependent — `select_profiler_from_triton_backend('hip')` returns `rocprofiler` on Triton 3.8.0 and newer and `roctracer` on 3.7.1 and earlier; on NVIDIA it is `cupti` either way. It is the only spelling correct on every version and is why it is the default: **naming a backend is a version commitment, and naming `roctracer` is an attach-mode commitment too.** `rocprofiler` is the preferred AMD backend upstream and has been a released backend since Triton 3.8.0 (2026-08-28), whose CLI advertises `-b {cupti,rocprofiler,roctracer,instrumentation}`; 3.7.x and earlier accept only `cupti`/`roctracer`/`instrumentation` and exit with an argparse `invalid choice: 'rocprofiler'` before the payload runs. `roctracer` is the deprecated AMD predecessor, and the one *whole-kernel* AMD backend present in *every* release including the oldest; `instrumentation` the intra-kernel path, also present in every release; `cupti` the NVIDIA one, accepted so a recipe stays portable even though aorta's examples are AMD. Pinning `roctracer` means `mode: env` and a payload that drives Proton itself: on Triton 3.7.x and earlier such a pin captures an empty tree under the default `mode: cli`. Measured on 3.8.0 the same pin captures normally, but the collector still refuses that one pairing on every version, because 3.7.x is what several images in use ship — see [Pinning an explicit AMD backend](#pinning-an-explicit-amd-backend) and [#439](https://github.com/ROCm/aorta/issues/439) for when the refusal goes. The two AMD tracing backends have *opposite* initialisation contracts, so `rocprofiler` is not covered — it configures itself when `libproton` loads, which makes `mode: cli` the ordering upstream prefers for it — see [Pinning an explicit AMD backend](#pinning-an-explicit-amd-backend). `instrumentation` measures *inside* a kernel and publishes **no numeric metrics** — its leaves carry `cycles` / `normalized_cycles`, never `time (<unit>)`, so such a trial reports `proton_artifact_dir` and nothing else. |
| `backend_mode` | `pcsampling`, `periodic_flushing` | (unset) | Proton's per-backend `--mode`. **Requires an explicit (non-`auto`) backend**, because which values are valid depends on which backend runs: `rocprofiler` takes both, `roctracer` only `periodic_flushing`, `cupti` both. Not valid on `backend: instrumentation`, whose modes are the `instrumentation_mode` key, and rejected together with `instrumentation_mode` *or* `granularity` — all three render into Proton's single `--mode` argument, so only one may be set. What the schema accepts is the backend's documented domain, not what a given build implements: `rocprofiler` + `pcsampling` is rejected by Triton 3.8.0 itself with `ValueError: [PROTON] RocprofSDKProfiler: unsupported mode: pcsampling`, because AMD PC sampling landed upstream after the 3.8.0 tag. `roctracer` + `periodic_flushing` is accepted by `start()` on 3.8.0, but **accepted is not the same as working**: on Triton 3.7.1 / ROCm 7.0.2 that pair kills the workload with SIGSEGV (exit 139) once kernels dispatch — verified on two payloads — so leave it unset there. Reaching Proton through `mode: cli` is version-dependent — see the paragraph below the table. |
| `context` | `shadow`, `python` | `shadow` | How a kernel's time is attributed to a calling frame. |
| `data` | `tree`, `trace` | `tree` | `tree` writes the `.hatchet` file the parser reads; `trace` writes a chrome trace instead, so it produces no numeric metrics. |
| `instrumentation_mode` | `default`, `mma`, `pcsampling` | (unset) | **Requires `backend: instrumentation`.** Renders into Proton's single `--mode` argument, so under `mode: cli` it reaches Proton on Triton 3.8.0 and newer and is silently dropped on 3.7.1 and earlier; `mode: env` carries it on every version. See the paragraph below the table. |
| `granularity` | `cta`, `warp`, `warp_2`, `warp_4`, `warp_8`, `warp_group`, `warp_group_2`, `warp_group_4`, `warp_group_8` | (unset) | **Requires `backend: instrumentation`.** Version-dependent under `mode: cli` like `instrumentation_mode`, and unusable on Triton 3.7.1 by either route regardless: the rendered `default:granularity=warp` string fails at kernel exit with `RuntimeError: Only warp granularity is supported for now`. Warp is that backend's own default there, so leave it unset. |
| `hook` | `triton` | (unset) | Renders Proton's `-k triton` under `mode: cli`; exported as `AORTA_PROTON_HOOK` under `mode: env`. Registers Proton's launch hook, which records Triton kernel launch metadata alongside the timing. |

`instrumentation_mode` and `granularity` are rejected with an actionable
error on any other backend rather than accepted-and-ignored: Proton would
silently produce a profile you did not ask for. Together they render into
Proton's single `--mode` argument as
`<instrumentation_mode>:granularity=<granularity>`, with
`instrumentation_mode` defaulting to `default` when only `granularity` is
set. `backend_mode` renders into that same argument, which is why the two
cannot be combined. When none of them is set the collector omits `--mode`
entirely and Proton keeps its own default.

Every `--mode`-bearing option — `backend_mode`, `instrumentation_mode`,
`granularity` — reaches Proton through `mode: cli` **only on Triton 3.8.0 and
newer**. Line 75 of `third_party/proton/proton/proton.py` at the `v3.8.0` tag
reads `start(args.name, context=args.context, data=args.data, backend=backend,
mode=args.mode, hook=args.hook)`; the equivalent line on 3.7.1 has no `mode=`,
so that front-end parses `-m/--mode` and then calls `start()` without it,
dropping the value on the floor. The CLI wrap renders `--mode` anyway rather
than refusing the combination: `validate_options` runs in aorta's own
interpreter, not the workload's, so it cannot know which Triton will execute the
wrap, and refusing would reject a recipe that is correct on the current release.
Under `mode: env` the payload reads `AORTA_PROTON_MODE` and passes it to
`proton.start(mode=...)` itself, which works on every version — that is why the
shipped `amd-*` examples use it.
`granularity` does not survive even that route on Triton 3.7.1: the rendered
`default:granularity=warp` is rejected inside
`libproton.exit_instrumented_op` (the bare `default` and other knobs such as
`buffer_size` are accepted, and the typed
`triton.profiler.mode.Default(granularity="warp")` object works). Warp
granularity is that backend's default on that version anyway, so the
`amd-instrumentation` example sets `instrumentation_mode` and omits
`granularity` deliberately. `hook` is in none of this: every Proton CLI
forwards `-k`, so it behaves identically in either attach mode on every
version.

### Proton attach modes

**`mode: cli` (default)** rewrites the launch:

```
python train.py --steps 10
→ python -m triton.profiler.proton -n <out>/proton \
      --context shadow --data tree train.py --steps 10
```

Proton's front-end **runs a script** (through `runpy.run_path`); it is not a
generic command runner. So the CLI wrap only applies to a Python launch
(`python`, `python3`, `python3.12`, or `pytest`), plus a small set of
no-argument interpreter flags that are kept in front of `-m` where they
belong: `-u`, `-B`, `-E`, `-s`, `-S`, `-O`, `-OO`, `-I`, `-b`, `-q`.

`python -m pytest ...` is accepted and normalised onto the bare
`pytest ...` spelling, because Proton dispatches on the target's basename
and runs it as `pytest.main(args)` — the same call `python -m pytest`
makes. The two spellings therefore produce an identical wrap:

```
pytest -k gemm
python -m pytest -k gemm
→ python -m triton.profiler.proton -n <out>/proton \
      --context shadow --data tree pytest -k gemm
```

**No other `-m <module>` works**, and the difference is Proton's, not
aorta's: `runpy.run_path` resolves a filesystem path, so a module name has
no spelling Proton can execute. `python -m torch.distributed.run ...` is
refused at setup with a message naming the working spellings, rather than
failing later inside Proton. Anything else — a HIP binary, a shell script,
`torchrun`, `python -c`, an interpreter option outside the set above —
raises a `ProtonWrapError` naming `mode: env` as the escape hatch. This is
deliberate: requesting a measurement that cannot be taken is a clean setup
failure, not a silently unprofiled run. `rocprof`, by contrast, wraps
anything.

**`mode: env`** leaves the command's own argv alone and hands it
`AORTA_PROTON_*` variables, for a workload that calls `proton.start()` /
`proton.finalize()` itself (scoped or intra-kernel measurement):

```yaml
collect:
  proton:
    mode: "env"
```

Because the generic collector seam only gets to rewrite argv — it has no
channel into the child's environment — the variables are delivered via an
`env(1)` prefix:

```
./my_launcher.sh
→ env AORTA_PROTON_CONTEXT=shadow AORTA_PROTON_DATA=tree ... \
      ./my_launcher.sh
```

A wrapper-style consumer that owns the child environment directly should
skip the argv seam and call `aorta.instrumentation.proton.build_env(out_dir,
options)`, which returns the same bundle as a plain `dict[str, str]` to
merge into the subprocess env. It never mutates `os.environ`.

| Variable | Meaning |
|---|---|
| `AORTA_PROTON_DIR` | Directory the profile should be written into |
| `AORTA_PROTON_NAME` | Session name (path stem) to pass to `proton.start(name)` |
| `AORTA_PROTON_BACKEND` | `backend` option value; **absent on the default `backend: auto`**, so the workload passes `backend=None` and gets Proton's own selection |
| `AORTA_PROTON_CONTEXT` | `context` option value |
| `AORTA_PROTON_DATA` | `data` option value |
| `AORTA_PROTON_MODE` | Rendered `--mode` value — the intra-kernel pair or `backend_mode`; absent when none of them is set |
| `AORTA_PROTON_HOOK` | `hook` option value, for `proton.start(hook=...)`; absent when unset |

#### Pinning an explicit AMD backend

The two AMD tracing backends look like a pair and are not one. Their
initialisation contracts are **opposite**, so what is safe for one is the
failure mode of the other. Note the incompatibility is in *initialisation
order*, not in the attach modes themselves: `roctracer` is the only backend an
attach mode is forced on, while `rocprofiler` works under either — `mode: cli`
loads `libproton` before the payload, and `mode: env` is equally safe provided
the payload imports Proton before torch, which every shipped payload now does
for an unrelated reason (see [Import order](#import-order)).

| Backend | Configures itself | Runtime state it needs at that moment | Attach mode |
|---|---|---|---|
| `roctracer` | when the profiling session starts | HIP runtime **already** up | `mode: env` — `mode: cli` is refused |
| `rocprofiler` | when `libproton.so` is loaded | HSA **not** yet initialised | either; `mode: cli` is the ordering upstream prefers |

An explicit `backend: roctracer` therefore belongs with `mode: env`, and only
there. Proton's CLI front-end calls its own `_select_backend()` **only when
`-b` is absent**, and that call initialises the Triton HIP driver as a side
effect. `roctracer` records nothing unless it starts *after* the HIP runtime
is up, so naming the backend removes the very step that makes the capture work
— and Proton does not complain.

Measured on the shipped `triton-vecadd` payload (8x gfx950 / MI355X, ROCm
7.0.2, Triton 3.7.1, PyTorch 2.13.0+rocm7.2), 3/3 deterministic:

| Wrap | Result |
|---|---|
| no `-b` (aorta's `backend: auto`) | ~3 KB hatchet, 27 dispatches across 5 kernels |
| `-b roctracer` | 160-byte hatchet, `ROOT` frame with empty metrics |

Both runs exit 0. Ordering alone accounts for the difference: driving Proton
from the Python API with `import torch` deferred past `proton.start()`
reproduces the empty tree (160 bytes), and initialising the driver before
`start()` gives a populated one (~1.6 KB with kernels).

In aorta the consequence is a quiet one: `parse_summary` finds no leaf
metrics in a `ROOT`-only tree, so the trial exits 0 carrying only
`proton_artifact_dir` — no `proton_gpu_time_ms`, no kernel names, nothing to
notice in `perf.md`. The collector therefore refuses that pairing at setup:
`mode: cli` with an explicit `backend: roctracer` raises `ProtonWrapError`
naming `mode: env` as the route.

**This is a Triton 3.7.x-and-earlier defect, and the guard has an expiry.** The
byte counts above are from 3.7.1. On 3.8.0 the same pinned `-b roctracer`
captures normally — measured on gfx950 / ROCm 10 at 3090 bytes, byte-for-byte
what `backend: auto` produces there — so the bug does not reproduce on the newer
stack. The guard stays because 3.7.x is still what several images in use ship
(3.7.1 in `rocm/pytorch:rocm7.14_ubuntu26.04_py3.14_pytorch_release_2.12.0`,
3.6.0 in `rocm/pytorch:latest`), and removing it today would regress those users
into the silent unprofiled trial it exists to prevent.

`test_cli_mode_pin_captures_nothing_only_on_the_versions_the_guard_targets`
asserts *whichever* of the two behaviours the installed Triton has, rather than
skipping on the newer stack, so both halves stay under test: the 3.7.x half is
the original regression, and the 3.8.0 half is what will say the guard has
become a no-op — and what fails loudly if a later Triton reintroduces the
ordering bug after the guard is gone. Removal is tracked in
[#439](https://github.com/ROCm/aorta/issues/439).

**`rocprofiler` is not covered, and must not be.** The evidence for it is of a
different kind from everything above — it is read from upstream's source and
its comments, not measured here — so it is worth being explicit about the
asymmetry. In Triton v3.8.0's
`third_party/proton/csrc/lib/Profiler/RocprofSDK/RocprofSDKProfiler.cpp` the
profiler's constructor calls `rocprofiler::forceConfigure<true>(&protonConfigure)`,
and that constructor is reached from an `__attribute__((constructor)) void
protonRocprofSDKLoadHook()` whose own comment says it "Runs during dlopen of
libproton.so (i.e. `import triton.profiler._C`) … so its constructor — which
calls rocprofiler_force_configure — runs before any Python code executes." The
comment above the `forceConfigure` call gives the reason:

> Deferring until doStart() is unsafe: any code that fully initializes HSA
> beforehand (e.g. triton's HIP driver query at pytest collection time, or a
> torch import chain) causes rocprofiler-sdk 1.2.0 to silently skip
> kernel-dispatch buffer tracing installation on already-existing queues,
> producing an empty dispatch buffer and no per-kernel timing data.

So `rocprofiler` must be configured *before* HSA comes up, which is the exact
inverse of `roctracer`'s requirement. The CLI path loads `libproton` before the
payload runs, so `mode: cli` gives it the ordering upstream prefers; an
env-mode payload that imports torch and only then calls `proton.start()` gives
it the ordering upstream warns against. Refusing the CLI pin would have pushed
operators toward the failure mode rather than away from it.

That claim is **not** measured here, unlike the `roctracer` numbers above, and
the difference in confidence is real: no environment available to this work
could run it. Triton 3.7.1 predates the backend entirely, and the ROCm 10
image that has it cannot `dlopen librocprofiler-sdk.so` (see
[`amd-rocprofiler`](../examples/profiling/proton/amd-rocprofiler/README.md)).
What is documented for `rocprofiler` is upstream's stated contract; what is
documented for `roctracer` is an observed byte count.

Note that this is only about the *attach mode*. `rocprofiler` does install an
HSA queue interceptor, so it is still one of the backends that conflicts with
a simultaneous `rocprof` capture — see [Combining
collectors](#combining-collectors). The two questions are independent.

The `mode: cli` guard therefore covers `roctracer` alone, and the backends
outside it are outside for different reasons rather than one.
`instrumentation` is outside it because it installs no queue
interceptor, and a `-b instrumentation` CLI wrap of a payload that carries
intra-kernel scopes captures them correctly (verified on Triton 3.7.1 — 1738
bytes, both scopes, cycle counts intact). What `mode: cli` costs on that backend
is `--mode` on Triton 3.7.1 and earlier, never the capture — a different and
version-scoped limitation, with `mode: env` as the same remedy; see the
`instrumentation_mode` row above. `cupti` is not an AMD path and is untested
here.

`backend: auto` is unaffected, and this is why it is the default: omitting
`-b` is exactly what keeps Proton's own driver-initialising selection step in
the picture.

A future queue-intercepting backend has to be classified by its own
initialisation contract before it is added to the guard. The collector spells
the refused set out as `frozenset({"roctracer"})` rather than deriving it from
the interception set for that reason: interception and attach-mode ordering
turned out to be different properties, and deriving one from the other is what
produced the wrong answer for `rocprofiler` in the first place.

What `auto` then resolves to is version-dependent, and the artifact does not
record it. On AMD it is `rocprofiler` from Triton 3.8.0 onward and `roctracer`
on 3.7.1 and earlier, so two captures of the same payload taken on two images
can come from two different backends while looking identical. Proton's
`.hatchet` metadata carries only the device type
(`{"HIP": {"0": {"arch": "gfx950", ...}}}`) and never the backend name, and the
collector deliberately publishes no `proton_backend` metric rather than one that
would imply more than it knows. The Triton version in the run's env snapshot is
what disambiguates which backend produced a given set of numbers — read it
before comparing `proton_gpu_time_ms` across runs from different environments.

To pin `roctracer`, use `mode: env` and a payload that calls `proton.start()`
after torch is imported (`rocprofiler` also accepts `mode: cli`). Note this is a
constraint on the `start()` call, not on the import: every Proton payload should
import `triton.profiler` **before** torch, because the reverse order deadlocks
the process at exit on Triton 3.8.0 — see [Import order](#import-order) below:

```yaml
collect:
  proton:
    mode: "env"
    backend: "roctracer"
    context: "shadow"
    data: "tree"
```

[`examples/profiling/proton/amd-roctracer/`](../examples/profiling/proton/amd-roctracer/)
is that recipe with a working payload;
[`amd-instrumentation/`](../examples/profiling/proton/amd-instrumentation/)
and [`amd-rocprofiler/`](../examples/profiling/proton/amd-rocprofiler/) are
the same shape for the other two backends.

#### Import order

**Import `triton.profiler` before `torch`, in every payload, whatever backend
it uses.** On Triton 3.8.0 the reverse order hangs the process forever at
`exit()` — after a perfectly good capture has already been flushed to disk.
These two lines are the entire reproducer:

```python
import torch
import triton.profiler  # process now never exits
```

`libproton.so` calls `rocprofiler_force_configure` from an
`__attribute__((constructor))`, so importing Proton registers it as a
rocprofiler-sdk client. When that registration lands *after* HSA is already up
— importing torch is enough, no GPU work required — the atexit
`rocprofiler::registration::finalize()` takes its registration mutex, invokes
Proton's `protonToolFini`, and that re-enters the same non-recursive mutex on
the same thread. The main thread then sits in `futex_wait` forever.

The failure is unusually easy to misread, because everything the payload was
asked to do succeeded: the kernels ran, the tree is complete, the `.hatchet`
is on disk, and stdout says `PASS`. Only the exit never happens. Under a test
runner or a sweep it reads as a hang with no failing assertion.

This is not a Proton-collector defect, and it is not specific to aorta: any
process on this stack that imports torch before `triton.profiler` is affected.
Filed against both repositories that can fix it, each carrying the two-line
reproducer, the gdb stack and the 3.7.1-clean / 3.8.0-affected boundary:
[triton-lang/triton#11549](https://github.com/triton-lang/triton/issues/11549)
(`protonToolFini` re-enters rocprofiler-sdk from inside a client finalizer) and
[ROCm/rocm-systems#11123](https://github.com/ROCm/rocm-systems/issues/11123)
(`invoke_client_finalizer` holds a non-recursive mutex across that callback).
Neither issue is closed, and no *released* Triton carries a fix — the one that
exists, `triton-lang/triton#11009`, landed on `main` after the `v3.8.0` tag was
cut (see the troubleshooting entry below). So the import order below is a
workaround rather than a cure.
Tracked here as [ROCm/aorta#434](https://github.com/ROCm/aorta/issues/434).

The ordering is free, and it does not fight either backend's contract. What
`roctracer` constrains is when `proton.start()` runs — its interceptor goes in
at session start, so the runtime has to be up by then — not when the module is
imported; measured on gfx950, moving the import above torch leaves the capture
byte-identical on both Triton 3.7.1 and 3.8.0. What `rocprofiler` constrains is
the import itself, and in the same direction. So every payload can import
Proton first and still call `proton.start()` after torch, which is what all
three shipped examples do. The imports are marked
`# isort: skip  # noqa: I001` so the linter does not undo it, and
`test_proton_payloads_import_proton_before_torch` fails the CPU suite if a
future edit reverses one.

### Device selection on AMD

Proton on AMD reads `ROCR_VISIBLE_DEVICES`; for the queue-intercepting
backends it does not honour `HIP_VISIBLE_DEVICES` and rejects it outright.
The collector translates automatically: when the environment the trial runs
with carries `HIP_VISIBLE_DEVICES` and the backend is `auto`, `rocprofiler`
or `roctracer`, the wrap becomes

```
env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=<value> <proton wrap...>
```

and a warning is logged. An explicit `ROCR_VISIBLE_DEVICES` already in the
environment wins — including an explicitly empty one, since an empty device
list means "hide every device" rather than "unset". The same applies to
`HIP_VISIBLE_DEVICES`: Proton rejects it on presence, so an empty value is
translated like any other.

No translation happens on the `instrumentation` or `cupti` backends. If a
translation *is* needed and `env` is not on `$PATH`, the trial fails setup:
argv rewriting is the only channel the collector has for these variables, so
continuing would profile the wrong device or not at all.

### Combining collectors

`rocprof` together with `proton` on backend `auto`, `rocprofiler` or
`roctracer` is **rejected at recipe load**: both install an HSA queue interceptor, and the
second one to attach will fail or report nothing. The error names the two
ways out — `backend: instrumentation` (intra-kernel measurement, no queue
interception), or run the two collectors as separate cells.

Wrap order, when more than one wrap applies:

1. Collectors wrap first, in the order `rocprof`, `proton`, with **rocprof
   outermost**. `rocprof` runs a whole command under the profiler while
   Proton takes over a Python script's execution, so rocprof has to be the
   outer process for the pair to compose at all.
2. The mirage emulation wrap is applied **after** the collectors, so it ends
   up outside them and the profiler runs *inside* the emulated environment:
   `mirage run -- rocprofv3 ... -- <command>`, not the reverse (which would
   profile the emulator's own launcher).

## Output

Collector artifacts land in a per-trial directory the platform creates and
threads into the trial config, one subdirectory per collector:

```
<run_dir>/<cell>/
  trial_0/                          # probe-flow trial artifacts
    stdout.log  stderr.log  result.json
  <workload>/
    trial_d0_m0_t0.json             # the trial JSON; metrics live here
    trial_d0_m0_t0/                 # the collector directory
      rocprof/
        aorta_kernel_stats.csv
        aorta_kernel_trace.csv
        aorta_domain_stats.csv
        aorta_agent_info.csv
        aorta_rocprof_summary.txt
      proton/
        proton.hatchet
```

Two things to internalise about that layout. First, the collector directory
is a **sibling** of the probe flow's `trial_<n>/` artifacts, not inside it —
it hangs off the per-workload subdirectory, keyed by the trial's matrix
coordinates. Second, you never have to reconstruct the path: the absolute
directory is reported as the `rocprof_artifact_dir` / `proton_artifact_dir`
metric, so `jq` it out of the trial JSON and use that.

The directory is created whether or not the payload produced GPU activity,
so the trial tree has the same shape either way. It is created **empty**: a
resumed trial replays onto the same paths, and inheriting the interrupted
attempt's files would make the summary report the previous run's numbers.
`aorta bundle` copies every file under the run directory, so collector
artifacts travel with a bundle automatically.

Being a sibling of `trial_<n>/` does not exempt the collector directory from
retention. Profiler traces are the artifact class a recipe's `retain` block
exists for, so its level prunes this tree too — a sweep with
`retain: {on_pass: none}` keeps captures only for the trials that failed. The
metrics are parsed *before* pruning, so `rocprof_gpu_time_ms` and friends
survive in the trial JSON even at a level that deletes the trace they came
from, and `result.json` lists each pruned file (with a leading `../`, marking
it as coming from the collector tree) under `capture.retention.deleted`.

The probe flow's `trial_<n>/result.json` records the argv that actually ran,
which is the *wrapped* one. So the exact profiler invocation — including any
environment rewriting the wrap did, such as the `HIP_VISIBLE_DEVICES`
translation below — is auditable from the artifact, not just from a log line:

```bash
jq -r '.argv | join(" ")' <run_dir>/<cell>/trial_0/result.json
```

On the file names: the collector passes `rocprofv3 -o aorta` for
determinism, which produces the flat `aorta_*.csv` files above. Run
`rocprofv3` **without** `-o` and it nests its output one level deeper under
a hostname directory, prefixed by PID
(`<hostname>/<pid>_kernel_stats.csv`) — worth knowing when comparing an
aorta capture against a hand-rolled one. The parser globs recursively, so
either shape reads fine.

`aorta_rocprof_summary.txt` is `rocprofv3`'s human-readable summary, routed
to a file via `-S --summary-output-file`. Without that redirect it prints to
stderr, where the probe classifier's stderr detectors would ingest it as
workload output. Note that `--summary-output-file` takes a filename *stem*
relative to `-d`, not a path: `rocprofv3` prefixes it with the `-o` basename
and appends `.txt`, which is where the `aorta_` prefix comes from. Handing it
an absolute path makes it splice that path into the middle of a filename and
create stray directories under `-d`.

### Metrics

Each collector contributes the same five keys under its own prefix.

| Metric | Type | Meaning |
|---|---|---|
| `<c>_kernel_count` | numeric | Total dispatches summed across kernels. Omitted (while the timings are still reported) when the capture gave no trustworthy launch count: an unreadable `Calls` column in a rocprof stats row, or a Proton leaf with no readable `count`. Both are per-kernel aggregates, so there is no safe number to substitute, and a fabricated count would read as measured. |
| `<c>_gpu_time_ms` | numeric | Total GPU time in ms across all kernels |
| `<c>_top_kernel_ms` | numeric | GPU time in ms of the single hottest kernel |
| `<c>_top_kernels` | list | The 5 hottest kernel names, hottest first |
| `<c>_artifact_dir` | string | Absolute path to the collector's output directory |

`<c>` is `rocprof` or `proton`. The three numeric ones are picked up by the
`perf.md` metrics table and aggregated per cell into
`matrix.json::cells[*].metrics_summary` (mean / min / max / n across
trials). The list and string values reach **neither** report — both aggregate
numeric scalars only — and are readable from the per-trial dispatcher JSON at
`.result.metrics`, which is where the `jq` recipes below look for them.

Only `<c>_artifact_dir` is guaranteed. A capture with no kernel data
contributes just that key, and so does a capture whose data the parser does
not speak. Proton's `instrumentation` backend is the standing example of the
second case: a working intra-kernel capture carries `cycles` and
`normalized_cycles` on its leaves, not `time (<unit>)` and not `count`. The
parser keys exclusively on `time (<unit>)`, so an `instrumentation` trial
publishes no numeric metrics at all however healthy the capture is — read the
tree with `proton-viewer -m normalized_cycles` instead.

## Analysis recipes

Read the per-cell numbers straight out of the matrix:

```bash
jq -r '.cells[] | [.name,
                   (.metrics_summary.rocprof_gpu_time_ms.mean // "-"),
                   (.metrics_summary.rocprof_kernel_count.mean // "-")]
       | @tsv' <run_dir>/matrix.json
```

Find the hottest kernels of one trial, and its artifact directory:

```bash
TRIAL=<run_dir>/<cell>/<workload>/trial_d0_m0_t0.json
jq -r '.result.metrics.rocprof_top_kernels[]' "$TRIAL"
ART=$(jq -r '.result.metrics.rocprof_artifact_dir' "$TRIAL")
```

The rocprof CSVs are readable directly — `<stem>_kernel_stats.csv` carries
`Name`, `Calls`, `TotalDurationNs`, `AverageNs`, `Percentage`, `MinNs`,
`MaxNs`, `StdDev`:

```bash
column -s, -t < "$ART"/aorta_kernel_stats.csv | head
```

For the non-CSV formats:

- `output_format: rocpd` gives a queryable SQLite database — join dispatches
  against agent info with plain SQL instead of post-processing CSV.
- `output_format: pftrace` opens in <https://ui.perfetto.dev> for a timeline
  view.

For Proton, list what a profile actually contains before charting it, then
print the metrics you want:

```bash
PROFILE=$(jq -r '.result.metrics.proton_artifact_dir' "$TRIAL")/proton.hatchet
proton-viewer --list "$PROFILE"
proton-viewer -m time/s,normalized_cycles "$PROFILE"
```

Run `proton-viewer` in the **same environment as the capture** — it is part
of Triton, and a host shim bound to a Triton-less interpreter will fail the
same way the `proton` console script does.

## Troubleshooting

**The payload printed `PASS`, the `.hatchet` is on disk, and the process never
exits.** An exit-time deadlock between Proton and rocprofiler-sdk on Triton
3.8.0, triggered by importing `torch` before `triton.profiler`. Nothing is
wrong with the capture — it is complete — so there is no failing assertion and
no error output; the process simply sits in `futex_wait` until something kills
it. Confirm with `gdb -p <pid> -batch -ex 'thread 1' -ex bt`: the main thread
will be in `rocprofiler::registration::finalize()` under `__run_exit_handlers`,
blocked on `get_registration_mutex()`. Fix by importing `triton.profiler`
before torch — see [Import order](#import-order). Tracked in
[ROCm/aorta#434](https://github.com/ROCm/aorta/issues/434).

Two things decide whether an environment is exposed, and both surprise people
trying to confirm it:

- **The unversioned `librocprofiler-sdk.so` must resolve.** A stock
  `rocm/pytorch` wheel image ships only `librocprofiler-sdk.so.1`, so Proton's
  constructor `dlopen` fails, no rocprofiler-sdk client registers, and nothing
  deadlocks — the image is *accidentally* immune. `Dockerfile.ci-gpu`'s
  devel-symlink fixup creates that soname, which is why CI is exposed and a
  bare `docker run` of the same base is not. A classic `/opt/rocm` install
  ships both spellings. **No GPU is needed** to reproduce it.
- **Triton must be 3.8.0 exactly.** 3.7.x predates the constructor;
  `triton-lang/triton#11009` removed the re-entrant call on `main` but landed
  after the `v3.8.0` tag was cut, so 3.8.0 is the affected window. Once the
  installed Triton carries that fix the import order stops being load-bearing
  for this reason (it still is for `rocprofiler`, which needs configuring
  before HSA).

**`rocprofv3 not found`.** Put `rocprofv3` on `$PATH` (it ships with ROCm)
or set `$ROCPROF_BIN` to the binary. `$ROCPROF_BIN` accepts either a bare
name resolved through `$PATH` or a path, which must point at an executable
file. A missing profiler fails the trial's setup deliberately, rather than
running unprofiled and reporting nothing.

**`collect: cannot prepare the <name> artifact directory ...`.** The
collector has nowhere to write — usually a file sitting where the directory
should be, a read-only output path, or a full disk. Like a missing
`rocprofv3` this fails setup rather than running the trial unprofiled. Fix
the `--output` path or drop the collector from the request.

**`proton needs 'env' on $PATH ...`.** The wrap had variables to deliver —
the `AORTA_PROTON_*` bundle in `mode: env`, or a `HIP_VISIBLE_DEVICES`
translation — and argv rewriting is its only channel for them. Install
coreutils in the image, or drop `proton`.

**`proton mode 'cli' needs a Python script launch, got '...'`.** Proton's
front-end executes a script. Either invoke the workload as
`<python> <script.py> ...`, or switch the recipe to `proton: {mode: env}`
and have the workload drive `proton.start()` / `proton.finalize()` itself.
A `torchrun` or `docker run` command will always hit this.

**`proton mode 'cli' cannot wrap 'python -m <module>'`.** Proton runs its
target through `runpy.run_path`, which needs a path; `pytest` is the one
module it also accepts. Pass the module's script by path, or use
`mode: env`. `python -m pytest` does *not* hit this — it is normalised onto
the bare `pytest` spelling.

**`No module named triton`, from the Proton wrap.** The interpreter running
the workload has no Triton. Run inside a container / venv that has it, or
point `$AORTA_PROTON_PYTHON` at an interpreter that does. The collector
uses `-m triton.profiler.proton` rather than the `proton` console script
precisely because the console script's shebang is frequently the wrong
interpreter.

**``RuntimeError: Could not load `lib<something>.so` `` from Proton.** The
environment's ROCm ships only the versioned soname (`libroctracer64.so.4`)
while Proton `dlopen`s the unversioned name. Seen on container images whose
ROCm comes from Python wheels rather than a system install. Add a directory of
unversioned symlinks to `$LD_LIBRARY_PATH`, or run on a host with a system ROCm
install (where both spellings exist).

**Which** library it names follows the backend, so it changed with Triton
3.8.0: `libroctracer64.so` for `roctracer`, and `librocprofiler-sdk.so` for the
`rocprofiler` backend that 3.8.0 added. That matters for `backend: auto`
specifically, because `auto` resolves to `rocprofiler` from 3.8.0 — measured in
aorta's ROCm 10 base image, where the default example fails on
`librocprofiler-sdk.so`. The remedy is the same for either name. Note this says
nothing about whether the backend or its mode is supported; it is purely a
packaging problem.

**`invalid choice: 'rocprofiler'` from Proton, before the payload runs.**
The installed Triton predates the `rocprofiler` backend, which was released in
Triton 3.8.0 (2026-08-28). 3.7.1, 3.7.0 and 3.6.0+rocm7.2.4 all offer
`-b {cupti,roctracer,instrumentation}` and nothing else. Upgrade to 3.8.0 or
newer, or drop the pin and let the default `backend: auto` choose — which on
3.7.1 hands you `roctracer`. See the `backend` row of the
[Proton options](#proton-options) table.
`python -m triton.profiler.proton --help` lists what your build accepts; 3.8.0
derives that list from `libproton.get_available_profilers()`, which there
returns `['cupti', 'rocprofiler', 'roctracer', 'instrumentation']`. An
`AttributeError` from calling that function is the quickest check that a build
predates the backend registry entirely.

**`ValueError: [PROTON] RocprofSDKProfiler: unsupported mode: pcsampling`.**
A different diagnosis from the one above, and worth separating: here the
`rocprofiler` backend *is* present — so you are on Triton 3.8.0 or newer — and
it is the mode that is missing. AMD PC sampling landed upstream after the 3.8.0
tag, so `backend_mode: pcsampling` needs a post-3.8 `main` build.
`backend_mode: periodic_flushing` is accepted on 3.8.0, as is `roctracer` with
the same mode — though see the SIGSEGV entry below before reaching for it.
Backend availability and mode availability are separate questions on this
backend; check which one the error is about before editing the recipe.

**The trial died with exit 139 (SIGSEGV, "dumped core") and no traceback.**
Check whether the recipe sets `backend_mode: periodic_flushing` on
`roctracer`. On Triton 3.7.1 / ROCm 7.0.2 that pair passes Proton's own mode
validation and then segfaults once kernels dispatch, taking the workload with
it and leaving no Python-level diagnostic. Verified on two separate payloads;
dropping the option makes the same run pass. It is reachable only through
`mode: env`, because the collector refuses a `mode: cli` `roctracer` pin for
the unrelated reason above.

The option is left in the schema rather than removed, for the same reason
`granularity` is: the domain mirrors what Proton documents, and the failure is
one build's, not the option's. But unlike `granularity` — which fails with a
catchable `RuntimeError` naming itself — this one gives the operator nothing,
which is why it gets an entry keyed on the exit code.

**A `mode: cli` recipe's `--mode` value had no effect.** Triton 3.7.1 and
earlier parse `-m/--mode` in the Proton front-end and then call `start()`
without it, so `backend_mode` / `instrumentation_mode` / `granularity` are
rendered into the wrap and dropped; 3.8.0 and newer forward the value. The
collector renders the flag on every version because it cannot tell from its own
interpreter which Triton will run the wrap. Switch to `mode: env` if you need
the knob to take effect independently of the installed version — see the
paragraph under the [Proton options](#proton-options) table.

**`proton mode 'cli' cannot pin backend 'roctracer'`, at setup.** `roctracer`
is the only backend this is raised for. The full message states the mechanism:

> Proton's command front-end calls `_select_backend()` only when `-b` is
> absent, and that call is what initialises the GPU runtime, so roctracer's
> interceptor is installed before the first HSA queue exists and records
> nothing: the profile comes back as an empty ROOT frame and the trial
> reports no proton metrics while exiting 0. Use `proton: {mode: env}` with a
> payload that calls `proton.start()` after the runtime is up (see
> `examples/profiling/proton/amd-roctracer`), or leave `backend: auto`, which
> omits `-b` and is unaffected. Note this applies to roctracer only —
> rocprofiler configures itself when libproton loads, so its CLI pin is
> allowed and is the ordering upstream prefers.

See [Pinning an explicit AMD backend](#pinning-an-explicit-amd-backend) for
the measured evidence and for `rocprofiler`'s opposite contract.
`instrumentation` and `cupti` are not refused either: the first does not
interpose on queues, and the second is not an AMD path.

**`RuntimeError: Only warp granularity is supported for now`, raised at
kernel exit from `libproton.exit_instrumented_op`.** Triton 3.7.1's
instrumentation backend rejects the rendered `<mode>:granularity=<value>`
string that the `granularity` option produces, even for `granularity: warp`
— which is that backend's own default. Drop `granularity` from the recipe
and keep `instrumentation_mode` alone. See the `granularity` row of the
[Proton options](#proton-options) table.

**A Proton profile with a ROOT node and no children.** An ordering problem,
and which ordering depends on the backend — the two AMD ones want opposite
things, so check which one ran before acting.

On `roctracer` the interceptor has to go in *after* the HIP runtime is up; a
session that starts before the first HSA queue exists captures nothing and
still exits 0. The default `backend: auto` avoids this by construction,
because Proton's own backend selection initialises the runtime before the
session starts, and an explicit `roctracer` pin skips exactly that step —
which is why the collector refuses it under `mode: cli` and points at
`mode: env` instead. See
[Pinning an explicit AMD backend](#pinning-an-explicit-amd-backend) for the
mechanism and the measured sizes. Under `mode: env` the payload owns the
ordering: import torch (or otherwise touch the GPU) *before*
`proton.start()`.

On `rocprofiler` the fix is the reverse. Upstream's own source comment (see
the same section) warns that anything which fully initialises HSA before
`libproton.so` is loaded — it names a torch import chain — makes
rocprofiler-sdk skip dispatch-buffer tracing on the queues that already exist
and hand back an empty buffer. So an env-mode payload must import
`triton.profiler` *before* torch;
[`amd-rocprofiler/gelu.py`](../examples/profiling/proton/amd-rocprofiler/gelu.py)
does, and says so at the import site. `mode: cli` is not exposed to this at
all, since it loads `libproton` before the payload. This diagnosis is
upstream's, not something reproduced here.

**No `.hatchet` written at all, and the payload exited 0.** The payload
ended by raising `SystemExit` (`raise SystemExit(main())`, or a bare
`sys.exit(0)`) on its success path. Proton runs the script through
`execute_as_main`, which on Triton 3.6.0 catches `Exception`, not
`BaseException`, so a success-path `SystemExit` escapes Proton's own CLI
before `finalize()` writes the profile. Triton 3.7.1 handles it, so this
looks version-specific from the outside. Return normally on success and
exit non-zero only on failure:

```python
if __name__ == "__main__":
    code = main()
    if code:
        raise SystemExit(code)
```

The shipped examples use exactly this shape; `tests/test_examples.py`
pins it for every example payload.

**Trial stderr differs from an unprofiled run.** Expected with
`--collect rocprof`. `rocprofv3` unconditionally writes lines like
`W... simple_timer.cpp:55] [rocprofv3] ...` and `Opened result file: ...`
to stderr. The human-readable summary is redirected to
`rocprof/aorta_rocprof_summary.txt` so the probe classifier's stderr detectors do
not ingest it, but the remaining noise is the profiler's own and cannot be
suppressed from here. Profiling is a measurement mode, not a byte-exact
passthrough — do not use it on a run whose stderr you are diffing.

**No artifacts, but the trial passed.** `rocprofv3` writes no files at all
when the profiled command performed no GPU work (a probe of `/bin/echo`,
for instance). That is a legitimate outcome, so parsing is fail-soft and
you get only `<c>_artifact_dir`. Check the payload actually dispatched a
kernel before suspecting the collector.

**`<c>_artifact_dir` present but no numeric metrics.** Four common causes:
`output_format` is not `csv` (the parser reads CSV only), `data: trace`
rather than `tree` on the Proton side (no `.hatchet` to walk), the payload
dispatched nothing, or `backend: instrumentation`, whose leaves carry
`cycles` / `normalized_cycles` instead of the `time (<unit>)` the parser
keys on — that last one is expected, not a fault, and the tree itself is
fine (see [Metrics](#metrics)). Malformed or partial artifacts degrade the
same way — an opt-in measurement never turns an otherwise-healthy trial into
a failure.

**`collect: <name> requested but no _aorta_collect_dir was threaded into the
trial config (non-writing rank?); skipping.`** The platform only threads the
collector output directory on the artifact-writing rank, so a non-writing
rank has nowhere to put artifacts and is skipped rather than scattering
files into the cwd. Expected on multi-rank launches; profile rank 0.

**Wrong GPU in the Proton profile.** See
[Device selection on AMD](#device-selection-on-amd). Pin with
`ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`.

**A recipe that used to load now fails with a queue-interception error.**
You have `rocprof` and `proton` in the same `collect:` block with a
queue-intercepting Proton backend. Split them into two cells, or use
`backend: instrumentation`.

**Triton wall time dominated by compilation.** Triton JIT-compiles on first
launch. That compile dispatches no kernel, so it does not appear in the
profile but does inflate the trial's wall clock. Give the payload a warmup,
or compare `<c>_gpu_time_ms` rather than wall time.

## Sweep / collector integration

Unlike `layer_numerics`, these two collectors need **no workload opt-in**.
The platform launches them itself in the generic subprocess seam, which is
why an opaque command works:

```bash
aorta sweep run \
  --recipe examples/profiling/rocprof/hip-gemm/recipe.yaml \
  --collect rocprof \
  --output ./profiling_results \
  -- /tmp/hip_gemm 512 20
```

The same names work in a recipe on either flow, and that is the only way to
pass options:

```yaml
collect:
  rocprof:
    trace: "kernel,hip"
```

Two consequences worth internalising:

- **Every trial of every cell gets a capture.** With N cells × M trials you
  get N×M artifact directories. Keep payload sizes and trial counts small
  when a collector is on.
- **Summary *parsing* never changes the verdict — attaching a collector
  can.** Parsing is fail-soft by construction: a missing, partial or malformed
  capture costs you metrics, never the trial. The attach path is deliberately
  the opposite. A collector that cannot run (no `rocprofv3`, a Proton wrap of
  a command its front end cannot execute, no `env(1)` for a device
  translation, an artifact directory that cannot be prepared) fails the
  trial's setup rather than running it unprofiled. A configuration that would
  run and measure nothing is refused there too: a `mode: cli` pin of
  `roctracer` (see
  [Pinning an explicit AMD backend](#pinning-an-explicit-amd-backend)). And a profiler is a real
  process in the trial: its own non-zero exit is the trial's exit code, and
  `rocprof`'s `simple_timer` stderr lines are visible to the classifier's
  stderr detectors (see [Troubleshooting](#troubleshooting)). Profiling is a
  measurement mode, not a transparent one.

Runnable end-to-end examples for both collectors, with payloads and
recipes, live in [`examples/profiling/`](../examples/profiling/README.md).

## Reference: environment variables

Recipe options are the primary interface. These environment variables cover
the cases a recipe cannot express, mostly "the tool is not where the
collector expects it".

| Var | Read by | Purpose | Default |
|---|---|---|---|
| `ROCPROF_BIN` | `rocprof` | Profiler binary: a bare name resolved on `$PATH`, or a path to an executable | `rocprofv3` from `$PATH` |
| `AORTA_PROTON_PYTHON` | `proton` | Interpreter to run Proton's front-end under; needed when Triton lives in a different interpreter than the workload | the workload's own `argv[0]` when it is a Python interpreter; for a bare `pytest`, the interpreter its console script is shebanged to; else the running `sys.executable` |
| `AORTA_PROTON_*` | the workload | Exported *by* `mode: env` for a workload that drives Proton itself; see [Proton attach modes](#proton-attach-modes) | — |

`ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` are not collector knobs, but
the Proton wrap reads and rewrites them — see
[Device selection on AMD](#device-selection-on-amd).

## See also

- [`examples/profiling/README.md`](../examples/profiling/README.md) — seven
  runnable examples (HIP GEMM, torch matmul, Triton vecadd, Triton softmax,
  plus one per pinned AMD Proton backend: roctracer, instrumentation,
  rocprofiler).
- [`recipes/README.md`](../recipes/README.md#schema-rules-full-detail) — the
  `collect:` schema, cell-level overrides, CLI precedence.
- [`docs/layer-numerics.md`](layer-numerics.md) — the other documented
  collector, a per-layer NaN / magnitude logger. Workload opt-in, unlike
  these two.
- [`docs/profiling.md`](profiling.md) — the older manual capture routes and
  the in-workload telemetry these collectors sit alongside.
