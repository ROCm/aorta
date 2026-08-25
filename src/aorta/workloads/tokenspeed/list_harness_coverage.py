"""Report which TokenSpeed operators the numerics/benchmark harness can drive.

A kernel being registered is not enough to benchmark it: the harness also needs
an input generator and a shape list for that operator. This script cross-joins
the three registries so a recipe author can see, per operator, whether it is
runnable out of the box, runnable only with explicit --shapes, or not runnable
at all. Run inside the TokenSpeed container.

This is the tool behind the "only gemm.mm is drivable" claim in the workload
README. It is kept so that claim can be re-checked against a newer image rather
than taken on trust -- upstream closing the gap is exactly what would widen the
set of kernels these recipes can cover.

It reads TokenSpeed's registries through private attributes (``_by_name``,
``_INPUT_GENERATORS``, ``_STANDARD_SHAPES``), because nothing public exposes
them. That makes it liable to break on a TokenSpeed upgrade -- but it breaks with
a loud AttributeError rather than a quietly wrong answer, which is the right way
round for a survey tool.

The container needs GPU access even though nothing is launched: importing
``tokenspeed_kernel`` runs the platform probe, which raises
``RuntimeError: tokenspeed-kernel requires an NVIDIA CUDA or AMD ROCm GPU``
without the device flags.

Usage -- mount the staged copy, since the docker daemon cannot read a
root-squashed NFS home (``stage_scripts.sh`` puts this file there):

  docker run --rm --device=/dev/kfd --device=/dev/dri \
      --group-add video --group-add render \
      --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
      -v /tmp/ts-work/scripts:/tools <image> \
      python3 /tools/list_harness_coverage.py
"""

from __future__ import annotations

import json
import sys

from tokenspeed_kernel.numerics import inputs
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels

# Input generators and shape lists are registered as an import side effect of
# each numerics submodule, not by load_builtin_kernels(). Import them all up
# front or the registries look empty and every operator appears unsupported.
_NUMERICS_MODULES = ("gemm", "moe", "quantize")


def _load_numerics_registrations() -> list[str]:
    loaded = []
    for name in _NUMERICS_MODULES:
        try:
            __import__(f"tokenspeed_kernel.numerics.{name}")
            loaded.append(name)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the survey
            loaded.append(f"{name}(FAILED: {type(exc).__name__})")
    return loaded


def main() -> int:
    load_builtin_kernels()
    print(f"numerics modules imported: {', '.join(_load_numerics_registrations())}")
    registry = KernelRegistry.get()

    ops = sorted({(s.family, s.mode) for s in registry._by_name.values()})
    generators = set(inputs._INPUT_GENERATORS)
    standard = set(inputs._STANDARD_SHAPES)
    bench = set(getattr(inputs, "_BENCHMARK_SHAPES", {}))

    report: dict[str, dict[str, object]] = {}
    for family, mode in ops:
        key = (family, mode)
        has_gen = key in generators
        has_shapes = key in standard or key in bench
        if has_gen and has_shapes:
            status = "runnable"
        elif has_gen:
            status = "needs_explicit_shapes"
        else:
            status = "no_input_generator"
        report[f"{family}.{mode}"] = {
            "status": status,
            "input_generator": has_gen,
            "standard_shapes": key in standard,
            "benchmark_shapes": key in bench,
            "kernels": sorted(s.name for s in registry._by_operator[key]),
        }

    order = {"runnable": 0, "needs_explicit_shapes": 1, "no_input_generator": 2}
    for op, info in sorted(report.items(), key=lambda kv: (order[str(kv[1]["status"])], kv[0])):
        kernels = info["kernels"]
        assert isinstance(kernels, list)
        print(f"{str(info['status']):22s} {op:36s} kernels={len(kernels)}")

    counts: dict[str, int] = {}
    for info in report.values():
        counts[str(info["status"])] = counts.get(str(info["status"]), 0) + 1
    print(f"\nsummary: {counts}")

    # The harness registries are keyed independently of the kernel registry, so
    # a generator can exist for an operator that has no kernels (and vice
    # versa). Surface that gap explicitly -- it is the difference between "we
    # can drive this kernel" and "this kernel is merely registered".
    orphan_generators = sorted(f"{f}.{m}" for (f, m) in generators if (f, m) not in set(ops))
    if orphan_generators:
        print(f"input generators with no registered kernel: {orphan_generators}")

    print("---JSON---")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
