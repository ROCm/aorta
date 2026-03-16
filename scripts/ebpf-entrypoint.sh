#!/bin/bash
# Mount debugfs so eBPF tracepoints are visible inside the container.
# Requires: --privileged or --cap-add=SYS_ADMIN --cap-add=BPF --cap-add=PERFMON

mount -t debugfs debugfs /sys/kernel/debug 2>/dev/null

if [ -d /sys/kernel/debug/tracing/events/amdgpu ]; then
    echo "[ebpf-init] debugfs mounted -- amdgpu tracepoints available"
else
    echo "[ebpf-init] WARNING: debugfs mounted but amdgpu tracepoints not found"
    echo "[ebpf-init]   container may need: --privileged or --cap-add=SYS_ADMIN"
fi

exec "$@"
