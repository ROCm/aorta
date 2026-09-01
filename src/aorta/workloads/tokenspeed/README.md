# TokenSpeed probe scripts

Scripts for running [TokenSpeed](https://github.com/lightseekorg/tokenspeed) —
a third-party AMD-optimized LLM inference engine — under `aorta sweep` on
gfx950. Full guide, measured results, and constraints:
[`docs/tokenspeed.md`](../../../../docs/tokenspeed.md).

Most of this directory is the probe integration: those recipes run on the
built-in `_subprocess` path (`mode: probe`), which wraps an opaque command, so
`host_launch.sh` is the only thing aorta sees and no workload class is involved.

The exception is `ts_bench_serve.sh`, which is driven by the
[`tokenspeed_serve`](../tokenspeed_serve.py) workload class rather than by
`host_launch.sh`. A probe cell cannot report a latency or a throughput —
`mode: probe` has no metrics channel — so serving performance needs a workload
class. See [`docs/tokenspeed-serving.md`](../../../../docs/tokenspeed-serving.md).

**No TokenSpeed source is vendored here.** The engine and its kernels are
consumed only through the published container image
(`lightseekorg/tokenspeed-amd:<tag>`), which these scripts invoke via its public
CLIs (`tokenspeed serve`, `python -m tokenspeed_kernel.benchmark`,
`python -m tokenspeed_kernel.numerics`) and, for the suite probe, the test tree
the image already ships at `/workspace`. Every file below is AORTA-maintained;
none is an upstream drop, so edit them in place.

The two survey tools read TokenSpeed private attributes (`_by_name`,
`_INPUT_GENERATORS`, `_STANDARD_SHAPES`) because the registries expose no public
enumeration. They fail loudly rather than reporting nothing if those move, and
they exist so the coverage claims in the guide can be re-checked against a newer
image instead of being trusted indefinitely.

| file | runs on | purpose |
|---|---|---|
| `host_launch.sh` | host | The opaque command `aorta sweep run` wraps. Turns the cell's `AORTA_ENV_FILE` into `docker run --env-file` and mints the per-trial `TS_RUN_TOKEN`. `TS_ENTRY` picks the in-container script. |
| `ts_serve_probe.sh` | container | Serving bring-up: start `tokenspeed serve`, poll readiness on the control port, issue one completion against the gateway port, tear the process group down. |
| `ts_bench_serve.sh` | container | Serving **benchmark** (not a probe): start one server, run `tokenspeed bench serve` for the configured warmup + measured steps, export one JSON per step, tear down. Audits each export because the bench CLI exits 0 even when every request failed. Launched by the `tokenspeed_serve` workload class, not by `host_launch.sh`. |
| `ts_kernel_probe.sh` | container | Kernel numerics and/or benchmark. Re-reads the exported JSON to reach its own verdict, because the upstream benchmark CLI exits 0 even when `--verify` fails. |
| `ts_pytest_probe.sh` | container | Runs one of TokenSpeed's own op test suites, reaching the families the benchmark harness cannot. Requires at least one executed test, because pytest exits 0 when everything skips. |
| `stage_scripts.sh` | host | Mirrors this directory to a node-local path the docker daemon can read, and syntax-checks it. Required: the daemon runs as root and cannot traverse a root-squashed NFS home. |
| `harvest_code_objects.py` | host | Runs a kernel — via the benchmark harness or `--pytest-suite` — with a clean Triton cache, collects the JIT-compiled `.hsaco`, and emits a `mode: sanitizer` recipe pinning each object by SHA-256. `--pytest-suite` is the only route to attention and MoE code objects. `--consan` additionally emits one loader shim and one single-kernel ConSan recipe per object, via `scripts/sanitizers/triton_consan_loader.py`. |
| `list_harness_coverage.py` | container | Surveys which TokenSpeed operators its own numerics/benchmark harness can actually drive. Substantiates the "only `gemm.mm`" constraint in the guide. |
| `map_kernel_test_coverage.py` | container | Surveys which registered kernels TokenSpeed's own test suites actually exercise, by instrumenting registry lookups while pytest runs. Substantiates the coverage table in the guide. |

Exit codes are a documented interface — the recipes' `custom_patterns`,
`tests/probe/test_tokenspeed_probe.py` and (for `ts_bench_serve.sh`)
`tests/workloads/test_tokenspeed_serve.py` all depend on them. See the header of
each script. `ts_bench_serve.sh` uses the 50-55 band precisely so a triage log
never leaves it ambiguous whether a verdict came from it or from
`ts_serve_probe.sh` (20-23).
