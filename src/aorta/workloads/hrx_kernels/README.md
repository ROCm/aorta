# HRX HIP-launch probe kernels

Vendored source for the `hrx` workload (`aorta.workloads.hrx:HrxWorkload`).

Each probe isolates one HIP kernel-launch path and computes `out[i] = in[i] + 100`
with `in = 7` and `out` pre-zeroed, then prints a `VERDICT=` line derived from
`out[0]`:

| `out[0]` | verdict | meaning |
|---|---|---|
| `107` | `FULLY_WORKS` | read + write + copies all correct |
| `100` | `INPUT_READ_ZERO` | H2D / input argument broken |
| `0` | `OUTPUT_NOT_WRITTEN` | kernel write / output argument never reached host |
| other | `GARBAGE` | address mismatch |

| file | probe id | HIP path under test |
|---|---|---|
| `hrx_probe.cpp` | `static` | `hipLaunchKernelGGL` (statically registered) |
| `hrx_probe_module.cpp` (+ `hrx_probe_module_kernel.cpp`) | `module` | `hipModuleLaunchKernel` (pre-packed `extra` buffer) |
| `hrx_probe_graph_add.cpp` | `graph_add` | `hipGraphAddKernelNode` (`extra`) |
| `hrx_probe_graph_setparams.cpp` | `graph_setparams` | `hipGraphKernelNodeSetParams` (`extra`) |
| `hrx_probe_graph_execsetparams.cpp` | `graph_execsetparams` | `hipGraphExecKernelNodeSetParams` (`extra`) |

`hrx_probe_module_kernel.cpp` is compiled to a standalone code object with
`hipcc --genco` and loaded at runtime by the `module` probe via
`hipModuleLoad`; the other probes embed their kernel and are compiled directly.

These originate from the `ROCm/hrx-system` #156 / #158 / #160 investigation.
