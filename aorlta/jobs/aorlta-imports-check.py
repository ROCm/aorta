"""The run image's import pre-flight (`jobs/aorlta-run.sbatch`): every module the orchestrator, the in-process
checkers and the trainer import, in the image they run in, before any engine starts. A module that only imports
inside a function (xxhash in the slot writer, the fast-path kernels in the trainer) would otherwise fail minutes
into setup. Exits non-zero with the list of failures."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import sys

MODULES = (
    "aorlta.orchestrator.run", "aorlta.orchestrator.wiring", "aorlta.orchestrator.score", "aorlta.reward.service",
    "aorlta.reward.remote", "aorlta.reward.factories", "aorlta.sync.writer", "aorlta.sync.landing",
    "aorlta.sync.merge", "aorlta.sync.roundtrip",
    "aorlta.trainer.ranks", "aorlta.trainer.rank_pool", "aorlta.trainer.batching", "aorlta.trainer.kernels",
    "aorlta.measure.timeline",
    "xxhash", "safetensors", "yaml", "click", "peft", "accelerate", "transformers", "fla", "causal_conv1d",
)  # not tokenspeed: tokenspeed-kernel refuses to import without a GPU, and each engine imports it on its own GPU
DISTS = ("torch", "transformers", "peft", "safetensors", "xxhash", "fla-core", "causal-conv1d", "PyYAML", "click")


def main(aorta_src: str, aorta_modules: tuple[str, ...] = ("models", "waitcheck")) -> int:
    bad = {}
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - every failure is listed
            bad[name] = f"{type(exc).__name__}: {exc}"[:300]
    from aorlta.reward.adapters import aorta_import

    for name in aorta_modules:  # the waitcheck leg runs in this image (the GPU legs do not); none: no aorta adapter
        mod, why = aorta_import.load(name, aorta_src)
        if mod is None:
            bad[f"aorta.{name}"] = why
    versions = {}
    for d in DISTS:
        try:
            versions[d] = md.version(d)
        except md.PackageNotFoundError:
            versions[d] = None
    print(json.dumps({"python": sys.version.split()[0], "versions": versions, "aorta": list(aorta_modules),
                      "failed": bad}, sort_keys=True))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], tuple(sys.argv[2:])))
