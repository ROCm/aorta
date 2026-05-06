# Vendored bpftrace scripts

These `.bt` scripts are vendored verbatim from the `ebpfaultline` project
(the `bpftrace/` subtree of the AMD-internal `ebpfaultline-main`
distribution; not yet open-sourced as of this PR) and are kept here so
AORTA can ship them as part of its `aorta.ebpf` module without a
runtime dependency on that source tree.

## Source

- Upstream subtree: `ebpfaultline-main/bpftrace/` (AMD internal; replace
  this entry with the canonical URL once the upstream project is
  published).
- License: MIT (see upstream `README.md`).

## Files

| Script | Purpose | Heisenberg risk |
|--------|---------|-----------------|
| `gpu_cont.bt` | Full-fidelity tracer: KFD evict/restore, SVM eviction, BO move, VM flush, PTE storms, ioctl errors, signals. | **High** -- many kprobes can suppress non-deterministic GPU memory races. |
| `gpu_cont_light.bt` | Lightweight: only the three KFD/SVM kprobes plus ioctl errors and signals. | Medium. |
| `gpu_cont_tp_only.bt` | Tracepoints only, no kprobes; recommended default. | **Low**. |
| `gpu_cont_1kprobe.bt` | TP_ONLY plus a single eviction kprobe (experiment). | Medium. |
| `gpu_cont_unrelated_kprobe.bt` | TP_ONLY plus an unrelated `do_sys_openat2` kprobe (control). | Medium. |

## Kernel compatibility

The original scripts were validated on Linux **6.9.0-fbk6** with AMD ROCm.
Kprobe symbols and tracepoint argument layouts can change across kernel
versions; the Python `BpftraceRunner` will surface bpftrace errors via
its stderr capture so issues are visible to the caller.

## Updating

When syncing newer upstream changes, copy the `.bt` files into this
directory verbatim and update this file's `Source` reference.
