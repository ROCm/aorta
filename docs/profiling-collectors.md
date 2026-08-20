# Profiling Collectors (`rocprof` / `proton`)

Attach a GPU profiler to **any command aorta runs**, without editing the
command. Both collectors work by rewriting the launch argv — the same seam
the mirage emulator uses — so an opaque `aorta sweep run ... -- <command>`
gets profiled for free and the payload never learns it is being measured.

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

If only `rocprof_artifact_dir` comes back, the profiler attached but found
no kernel data.

## Configuration

Collectors are selected either on the command line or in the recipe.

```bash
# Names only. Works on both the workload (triage) flow and the
# subprocess (probe) flow, and overrides any recipe-pinned collect:.
aorta sweep run --recipe r.yaml --collect rocprof -- ./my_gpu_binary
aorta sweep run --recipe r.yaml --collect proton -- python train.py
```

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
| `stats` | `1/true/yes/on` or `0/false/no/off` | `true` | Adds `--stats`, i.e. the pre-aggregated `*_kernel_stats.csv`. With `stats` off the parser falls back to summing dispatch spans from the kernel trace. |
| `pmc` | comma- or space-separated counter names | (unset) | Hardware counters. Rendered last before the `--` separator because `--pmc` is variadic. Must name at least one counter. |
| `kernel_include_regex` | regex | (unset) | `--kernel-include-regex`; must be non-empty. |
| `summary_units` | `sec`, `msec`, `usec`, `nsec` | (unset → rocprofv3's own default) | Unit for the human-readable summary file only. Does not change the parsed metrics, which are always ms. |

### `proton` options

| Option | Values | Default | Notes |
|---|---|---|---|
| `mode` | `cli`, `env` | `cli` | See [Attach modes](#proton-attach-modes). |
| `backend` | `auto`, `rocprofiler`, `roctracer`, `instrumentation`, `cupti` | `auto` | `auto` omits Proton's `-b` and lets Proton pick the backend matching the active runtime — `rocprofiler` where rocprofiler-sdk is available, `roctracer` otherwise. It is the only portable spelling and is why it is the default: **naming a backend is a version commitment.** `rocprofiler` is the preferred AMD backend upstream but Triton 3.7.x and earlier accept only `cupti`/`roctracer`/`instrumentation` and exit with an argparse `invalid choice: 'rocprofiler'` before the payload runs. `roctracer` is the deprecated AMD predecessor; `instrumentation` the intra-kernel path; `cupti` the NVIDIA one, accepted so a recipe stays portable even though aorta's examples are AMD. Pin a backend when you need to know exactly which one measured. |
| `context` | `shadow`, `python` | `shadow` | How a kernel's time is attributed to a calling frame. |
| `data` | `tree`, `trace` | `tree` | `tree` writes the `.hatchet` file the parser reads; `trace` writes a chrome trace instead, so it produces no numeric metrics. |
| `instrumentation_mode` | `default`, `mma`, `pcsampling` | (unset) | **Requires `backend: instrumentation`.** |
| `granularity` | `cta`, `warp`, `warp_2`, `warp_4`, `warp_8`, `warp_group`, `warp_group_2`, `warp_group_4`, `warp_group_8` | (unset) | **Requires `backend: instrumentation`.** |

`instrumentation_mode` and `granularity` are rejected with an actionable
error on any other backend rather than accepted-and-ignored: Proton would
silently produce a profile you did not ask for. Together they render into
Proton's single `--mode` argument as
`<instrumentation_mode>:granularity=<granularity>`, with
`instrumentation_mode` defaulting to `default` when only `granularity` is
set. When neither is set the collector omits `--mode` entirely and Proton
keeps its own default.

### Proton attach modes

**`mode: cli` (default)** rewrites the launch:

```
python train.py --steps 10
→ python -m triton.profiler.proton -n <out>/proton \
      --context shadow --data tree train.py --steps 10
```

Proton's front-end **`exec`s a script**; it is not a generic command runner.
So the CLI wrap only applies to a Python launch (`python`, `python3`,
`python3.12`, or `pytest`), plus a small set of no-argument interpreter
flags that are kept in front of `-m` where they belong: `-u`, `-B`, `-E`,
`-s`, `-S`, `-O`, `-OO`, `-I`, `-b`, `-q`. Anything else — a HIP binary, a
shell script, `torchrun`, `python -c`, an interpreter option outside that
set — raises a `ProtonWrapError` naming `mode: env` as the escape hatch.
This is deliberate: requesting a measurement that cannot be taken is a
clean setup failure, not a silently unprofiled run. `rocprof`, by contrast,
wraps anything.

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
| `AORTA_PROTON_MODE` | Rendered `--mode` value; absent when no intra-kernel knob is set |

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
environment wins. No translation happens on the `instrumentation` or
`cupti` backends, and none happens if `env` is not on `$PATH` (a warning
says so, and the profile may target the wrong device).

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
so the trial tree has the same shape either way. `aorta bundle` copies every
file under the run directory, so collector artifacts travel with a bundle
automatically.

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
| `<c>_kernel_count` | numeric | Total dispatches summed across kernels |
| `<c>_gpu_time_ms` | numeric | Total GPU time in ms across all kernels |
| `<c>_top_kernel_ms` | numeric | GPU time in ms of the single hottest kernel |
| `<c>_top_kernels` | list | The 5 hottest kernel names, hottest first |
| `<c>_artifact_dir` | string | Absolute path to the collector's output directory |

`<c>` is `rocprof` or `proton`. The three numeric ones are picked up by the
`perf.md` metrics table and aggregated per cell into
`matrix.json::cells[*].metrics_summary` (mean / min / max / n across
trials). The list and string values are skipped in `perf.md` — it only
aggregates numeric scalars — but are retained in the per-trial JSON.

Only `<c>_artifact_dir` is guaranteed. A capture with no kernel data
contributes just that key.

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

**`rocprofv3 not found`.** Put `rocprofv3` on `$PATH` (it ships with ROCm)
or set `$ROCPROF_BIN` to the binary. `$ROCPROF_BIN` accepts either a bare
name resolved through `$PATH` or a path, which must point at an executable
file. A missing profiler fails the trial's setup deliberately, rather than
running unprofiled and reporting nothing.

**`proton mode 'cli' needs a Python script launch, got '...'`.** Proton's
front-end executes a script. Either invoke the workload as
`<python> <script.py> ...`, or switch the recipe to `proton: {mode: env}`
and have the workload drive `proton.start()` / `proton.finalize()` itself.
A `torchrun` or `docker run` command will always hit this.

**`No module named triton`, from the Proton wrap.** The interpreter running
the workload has no Triton. Run inside a container / venv that has it, or
point `$AORTA_PROTON_PYTHON` at an interpreter that does. The collector
uses `-m triton.profiler.proton` rather than the `proton` console script
precisely because the console script's shebang is frequently the wrong
interpreter.

**`RuntimeError: Could not load \`libroctracer64.so\`` from Proton.** The
environment's ROCm ships only the versioned soname
(`libroctracer64.so.4`) while Proton `dlopen`s the unversioned name. Seen on
container images whose ROCm comes from Python wheels rather than a system
install. Add a directory of unversioned symlinks to `$LD_LIBRARY_PATH`, or
run on a host with a system ROCm install (where both spellings exist).

**`invalid choice: 'rocprofiler'` from Proton, before the payload runs.**
The installed Triton predates the `rocprofiler` backend. Drop the pin and
let the default `backend: auto` choose — that is what it is for. See the
`backend` row of the [Proton options](#proton-options) table.

**A Proton profile with a ROOT node and no children.** Proton's queue
interceptor has to be installed before the first HSA queue is created; a
run that initialises the GPU earlier captures nothing and still exits 0.
The default `backend: auto` avoids this by construction, because Proton's
own backend selection initialises the runtime before the session starts.
An explicit backend pin skips that step, so pin one only when you need to.

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

**`<c>_artifact_dir` present but no numeric metrics.** Three common causes:
`output_format` is not `csv` (the parser reads CSV only), `data: trace`
rather than `tree` on the Proton side (no `.hatchet` to walk), or the
payload dispatched nothing. Malformed or partial artifacts degrade the same
way — an opt-in measurement never turns an otherwise-healthy trial into a
failure.

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
- **A collector never changes the verdict.** Summary parsing is fail-soft by
  construction; the worst case is fewer metrics.

Runnable end-to-end examples for both collectors, with payloads and
recipes, live in [`examples/profiling/`](../examples/profiling/README.md).

## Reference: environment variables

Recipe options are the primary interface. These environment variables cover
the cases a recipe cannot express, mostly "the tool is not where the
collector expects it".

| Var | Read by | Purpose | Default |
|---|---|---|---|
| `ROCPROF_BIN` | `rocprof` | Profiler binary: a bare name resolved on `$PATH`, or a path to an executable | `rocprofv3` from `$PATH` |
| `AORTA_PROTON_PYTHON` | `proton` | Interpreter to run Proton's front-end under; needed when Triton lives in a different interpreter than the workload | the workload's own `argv[0]` when it is a Python interpreter, else the running `sys.executable` |
| `AORTA_PROTON_*` | the workload | Exported *by* `mode: env` for a workload that drives Proton itself; see [Proton attach modes](#proton-attach-modes) | — |

`ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` are not collector knobs, but
the Proton wrap reads and rewrites them — see
[Device selection on AMD](#device-selection-on-amd).

## See also

- [`examples/profiling/README.md`](../examples/profiling/README.md) — four
  runnable examples (HIP GEMM, torch matmul, Triton vecadd, Triton softmax).
- [`recipes/README.md`](../recipes/README.md#schema-rules-full-detail) — the
  `collect:` schema, cell-level overrides, CLI precedence.
- [`docs/layer-numerics.md`](layer-numerics.md) — the other documented
  collector, a per-layer NaN / magnitude logger. Workload opt-in, unlike
  these two.
- [`docs/profiling.md`](profiling.md) — the older manual capture routes and
  the in-workload telemetry these collectors sit alongside.
