Verbatim AddressSanitizer output captured on a gfx950 (MI350) GPU, 2026-10-19,
inside `rocm/dev-ubuntu-22.04:7.0.2` with the ROCm 7.0.2 `*-asan` runtime
packages, from `asan_probe.hip` built with
`hipcc --offload-arch=gfx950:xnack+ -fsanitize=address -shared-libsan -g -O1`.
Each file is the program's stderr for one deliberate defect, unedited apart from
trailing whitespace (stripped for the repo's whitespace hook), so the
parser is tested against what the runtime really prints rather than a guess.

| file | how it was produced |
|---|---|
| `device_heap_buffer_overflow_write.txt` | kernel stores one int past a `hipMalloc` buffer |
| `device_heap_buffer_overflow_read.txt` | kernel loads one int past a `hipMalloc` buffer |
| `device_heap_use_after_free.txt` | kernel stores into a buffer after `hipFree` |
| `device_lds_overflow.txt` | kernel stores one int past a `__shared__` array |
| `host_heap_buffer_overflow.txt` | host code stores one int past a `malloc` buffer |
| `stock_runtime_hostcall_abort.txt` | the write overflow with the STOCK runtime: detected, unreported |
| `uninstrumented_launcher_not_preloaded.txt` | an uninstrumented binary with `lib/asan` on the path |
| `stock_runtime_leak_report.txt` | a correct program, stock runtime, `detect_leaks=1` |
| `xnack_off_invalid_device_function.txt` | the xnack+ build run with `HSA_XNACK=0` |
