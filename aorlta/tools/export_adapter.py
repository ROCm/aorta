#!/usr/bin/env python3
"""Export a training run's checkpoint as a PEFT LoRA adapter directory.

A checkpoint (`<run_dir>/ckpt/iter-<n>/`) holds the trainer's fp32 LoRA latent in `trainer/trainer.pt`, named by the
PEFT model's own modules (`<module>.lora_A`, `<module>.lora_B`). This writes them as PEFT saves an adapter:
`adapter_model.safetensors` with the keys `<module>.lora_A.weight` and `<module>.lora_B.weight` (fp32, unchanged
bits), and an `adapter_config.json` for the plain LoRA the trainer trains (r and alpha from
`aorlta.orchestrator.wiring`, no dropout, no bias). The output is what `tools/merge_base_adapter.py` merges into the
base and what a score run's `model.adapter_dir` serves; peft's `PeftModel.from_pretrained` loads it too.

The checkpoint is checked against its MANIFEST.json first (`aorlta.orchestrator.state.verify_checkpoint`), and the
written tensors are read back and compared bit for bit. `export-manifest.json` beside the output records the
checkpoint's iteration and policy version, the sha256 of the input and of every output file, and the run plan's
base directory when the run directory is the checkpoint's grandparent.

  export_adapter.py --ckpt RUN_DIR/ckpt/iter-000003 --out DIR [--base-model NAME_OR_PATH]

Exit codes: 0 written and checked; 1 the read-back check failed; 2 the inputs were refused before anything was
written, or `<out>` appeared while the export ran and is left as it was (a failed read-back, and a checked export
refused this way, are left in `<out>.partial`). A run that used `sync.stochastic_rounding` served weights rounded from this latent with random bits, so a
merge of the export (round to nearest) is close to, not equal to, what its engines served.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from aorlta.atomic import atomic_write_json  # noqa: E402

ADAPTER_CONFIG, ADAPTER_WEIGHTS, MANIFEST_NAME = "adapter_config.json", "adapter_model.safetensors", "export-manifest.json"
SCHEMA = "aorlta.export-adapter/1"
PEFT_PREFIX = "base_model.model."


class Refused(ValueError):
    """The checkpoint is not one this tool can export; nothing was written."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_latent(ckpt: Path) -> tuple[dict, dict[str, tuple[torch.Tensor, torch.Tensor]], int, str]:
    """(the checkpoint's run state, module -> (A, B), steps done, sha256 of the trainer file), after the checkpoint's
    own manifest check. The trainer file is read once, before that check, and the bytes read are what must match the
    manifest's entry for it and what is loaded and hashed: a file replaced while the export runs is refused or never
    read, and is not exported or hashed in place of the checked bytes."""
    from aorlta.ids import sha256_json
    from aorlta.orchestrator.state import MANIFEST_NAME as CKPT_MANIFEST
    from aorlta.orchestrator.state import TRAINER_DIR, CheckpointError, verify_checkpoint
    from aorlta.trainer.rank_pool import TRAINER_STATE, TRAINER_STATE_SCHEMA

    rel = f"{TRAINER_DIR}/{TRAINER_STATE}"
    try:
        raw = (ckpt / rel).read_bytes()
        state = verify_checkpoint(ckpt)
        recorded = json.loads((ckpt / CKPT_MANIFEST).read_text(encoding="utf-8"))
    except (CheckpointError, OSError, ValueError) as exc:
        raise Refused(f"{ckpt}: {exc}") from None
    digest = hashlib.sha256(raw).hexdigest()
    files = recorded.get("files") if isinstance(recorded, dict) else None
    if (not isinstance(files, dict) or recorded.get("files_sha256") != sha256_json(files)
            or recorded.get("iteration") != state.iteration
            or files.get(rel) != {"sha256": digest, "size": len(raw)}):
        raise Refused(f"{ckpt}: {rel} as read is not the file its manifest records")
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("schema") != TRAINER_STATE_SCHEMA:
        raise Refused(f"{ckpt}: trainer state schema {saved.get('schema') if isinstance(saved, dict) else None!r}")
    lora = saved.get("lora")
    if not isinstance(lora, dict) or not lora:
        raise Refused(f"{ckpt}: no LoRA tensors")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, t in lora.items():
        module, dot, ab = name.rpartition(".")
        if not dot or ab not in ("lora_A", "lora_B") or not isinstance(t, torch.Tensor) or t.dim() != 2:
            raise Refused(f"{ckpt}: {name!r} is not a 2-D lora_A or lora_B tensor")
        pairs.setdefault(module, {})[ab] = t
    modules = {}
    for module, slot in sorted(pairs.items()):
        if set(slot) != {"lora_A", "lora_B"}:
            raise Refused(f"{ckpt}: {module} has only {sorted(slot)}")
        if not module.startswith(PEFT_PREFIX):
            raise Refused(f"{ckpt}: {module} is not a PEFT module name ({PEFT_PREFIX}...)")
        modules[module] = (slot["lora_A"], slot["lora_B"])
    return ({"run_id": state.run_id, "iteration": state.iteration, "policy_version": state.policy_version},
            modules, int(saved.get("steps_done", 0)), digest)


def adapter_config(modules: dict, base_model: str | None) -> dict:
    from aorlta.orchestrator.wiring import LORA_ALPHA, LORA_R

    ranks = {a.shape[0] for a, _ in modules.values()} | {b.shape[1] for _, b in modules.values()}
    if ranks != {LORA_R}:
        raise Refused(f"the LoRA ranks are {sorted(ranks)}, not the trainer's r={LORA_R}")
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model, "r": LORA_R,
            "lora_alpha": LORA_ALPHA, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
            "use_dora": False, "use_rslora": False, "modules_to_save": None, "init_lora_weights": True,
            "target_modules": sorted({m.rsplit(".", 1)[-1] for m in modules})}


def run_plan_base(ckpt: Path) -> str | None:
    plan = ckpt.parent.parent / "run_plan.json"
    if not plan.is_file():
        return None
    try:
        return json.loads(plan.read_text(encoding="utf-8"))["config"]["model"]["base_dir"]
    except (ValueError, KeyError, TypeError):
        return None


def export(ckpt: Path, out: Path, base_model: str | None = None) -> dict:
    from safetensors.torch import load_file, save_file

    ckpt, out = Path(ckpt), Path(out)
    if out.exists():
        raise Refused(f"{out} exists")
    state, modules, steps, trainer_sha256 = read_latent(ckpt)
    plan_base = run_plan_base(ckpt)
    cfg = adapter_config(modules, base_model if base_model is not None else plan_base)
    tensors = {}
    for module, (a, b) in modules.items():
        tensors[f"{module}.lora_A.weight"] = a.detach().to("cpu").contiguous()
        tensors[f"{module}.lora_B.weight"] = b.detach().to("cpu").contiguous()
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise Refused(f"{tmp} exists")
    tmp.mkdir(parents=True)
    save_file(tensors, str(tmp / ADAPTER_WEIGHTS), metadata={"format": "pt"})
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp / ADAPTER_WEIGHTS, 0o666 & ~umask)  # safetensors writes it owner-only; an adapter is for sharing
    (tmp / ADAPTER_CONFIG).write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    back = load_file(str(tmp / ADAPTER_WEIGHTS))
    same = sorted(back) == sorted(tensors) and all(
        back[k].dtype == t.dtype and back[k].shape == t.shape
        and torch.equal(back[k].view(torch.uint8), t.view(torch.uint8)) for k, t in tensors.items())
    manifest = {
        "schema": SCHEMA, "pass": same, "checkpoint": {"dir": str(ckpt.resolve()), "state": state, "steps_done": steps,
                                                       "trainer_pt_sha256": trainer_sha256},
        "run_plan_base_dir": plan_base, "modules": len(modules), "tensors": len(tensors),
        "r": cfg["r"], "lora_alpha": cfg["lora_alpha"],
        "files": {n: sha256_file(tmp / n) for n in (ADAPTER_WEIGHTS, ADAPTER_CONFIG)},
    }
    atomic_write_json(tmp / MANIFEST_NAME, manifest)
    if same:  # a failed read-back stays in <out>.partial, never at <out>
        publish(tmp, out)
    return manifest


def publish(tmp: Path, out: Path) -> None:
    """Rename `tmp` to `out`, refusing (with `tmp` left in place) an `out` that appeared since the check at entry.
    On POSIX a rename replaces an empty directory, so `out` is first reserved by creating it, which fails if anything
    is there; the rename then replaces only that reservation, and fails if anything was written into it meanwhile."""
    try:
        out.mkdir()
    except FileExistsError:
        raise Refused(f"{out} appeared during the export; the checked export is left in {tmp}") from None
    try:
        os.rename(tmp, out)
    except OSError as exc:
        try:
            out.rmdir()  # only if still empty: what was written into it is not ours to remove
        except OSError:
            pass
        raise Refused(f"{out} changed during the export ({exc.strerror}); the checked export is left in "
                      f"{tmp}") from None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="export_adapter.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True, help="a checkpoint directory, <run_dir>/ckpt/iter-<n>")
    ap.add_argument("--out", type=Path, required=True, help="the adapter directory to write (must not exist)")
    ap.add_argument("--base-model", default=None,
                    help="base_model_name_or_path for adapter_config.json (default: the run plan's model.base_dir)")
    args = ap.parse_args(argv)
    try:
        manifest = export(args.ckpt, args.out, args.base_model)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: manifest[k] for k in ("pass", "modules", "tensors", "r", "lora_alpha")}))
    return 0 if manifest["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
