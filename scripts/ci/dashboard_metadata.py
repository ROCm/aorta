"""Workload and category labels for the nightly CI dashboard.

Kept as a Python module (not JSON) because the repo gitignores ``*.json``.
"""

from __future__ import annotations

from typing import Any

_COMPARE_NOTE = (
    "Match the aorta, PyTorch, ROCm, and HIP versions shown in the dashboard "
    "header when comparing numbers. Small differences on different hardware are "
    "expected; regressions vs the blessed baseline are what nightly CI flags."
)


def _read_matrix_cmds(artifact_workload: str) -> list[str]:
    """Shell helpers to locate and read the newest matrix for one recipe workload."""
    wl = artifact_workload
    return [
        (
            "find triage_results -mindepth 3 -maxdepth 3 -type d "
            f"-path '*/{wl}/*' -printf '%T@ %p\\n' 2>/dev/null "
            "| sort -rn | head -1 | cut -d' ' -f2-"
        ),
        f"cat triage_results/<TICKET>/{wl}/<timestamp>/matrix.md",
        (
            f"cat triage_results/<TICKET>/{wl}/<timestamp>/matrix.json "
            "| python -m json.tool | head -80"
        ),
    ]


def _install_setup(*, min_gpus: int, distributed: bool = False) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {
            "title": "Clone AORTA and install the nightly wheel channel",
            "commands": [
                "git clone https://github.com/ROCm/aorta.git",
                "cd aorta",
                "pip install --upgrade pip",
                (
                    "pip install --upgrade --pre 'amd-aorta[hw-queue]' "
                    "-f https://github.com/ROCm/aorta/releases/expanded_assets/dev-wheels"
                ),
                "aorta --help",
            ],
        },
        {
            "title": "Confirm ROCm + PyTorch see enough GPUs",
            "commands": [
                (
                    'python -c "import torch; n=torch.cuda.device_count(); '
                    "assert torch.cuda.is_available() and n>0, 'no CUDA/HIP devices'; "
                    f"assert n>={min_gpus}, f'need {min_gpus} GPU(s), have {{n}}'; "
                    "print(f'{n} GPU(s), HIP', torch.version.hip)\""
                ),
            ],
        },
    ]
    if distributed:
        steps.append(
            {
                "title": "Export single-node torchrun env (adjust for multi-node)",
                "commands": [
                    "export NCCL_DEBUG=WARN  # optional, for troubleshooting",
                    "export MASTER_ADDR=127.0.0.1",
                    "export MASTER_PORT=29500",
                    "# Slurm / multi-node: see recipes/README-running-recipes.md",
                ],
            }
        )
    return steps


def _workload_repro(
    *,
    artifact_workload: str,
    prerequisites: list[str],
    recipe: str,
    run_command: str,
    min_gpus: int,
    distributed: bool,
    dry_run: str,
    verify_title: str,
    verify_extra: list[str] | None = None,
    success: str,
    setup_extra: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    verify_cmds = _read_matrix_cmds(artifact_workload)
    if verify_extra:
        verify_cmds.extend(verify_extra)
    setup = _install_setup(min_gpus=min_gpus, distributed=distributed)
    if setup_extra:
        setup.extend(setup_extra)
    return {
        "prerequisites": prerequisites,
        "setup": setup,
        "dry_run": {
            "title": "Validate the recipe YAML (no GPU execution)",
            "command": dry_run,
        },
        "run": {"title": "Run the nightly recipe", "command": run_command},
        "verify": [{"title": verify_title, "commands": verify_cmds}],
        "success_criteria": success,
        "compare_notes": _COMPARE_NOTE,
    }


DASHBOARD_METADATA: dict[str, Any] = {
    "categories": {
        "platform": {
            "label": "Platform",
            "summary": "ROCm and PyTorch load and execute on GPU.",
            "workloads": ["gpu_smoke"],
        },
        "inference": {
            "label": "Inference",
            "summary": "Offline LLM prefill/decode latency and throughput.",
            "workloads": ["inference_offline"],
        },
        "training": {
            "label": "Training",
            "summary": "PyTorch DDP and FSDP training step times.",
            "workloads": [
                "training_ddp",
                "training_ddp_8gpu",
                "training_fsdp",
                "training_fsdp_8gpu",
            ],
        },
        "correctness": {
            "label": "Correctness",
            "summary": "Numerical determinism and distributed race detection.",
            "workloads": [
                "llm_determinism",
                "llm_determinism_8gpu",
                "race",
                "race_8gpu",
            ],
        },
    },
    "workloads": {
        "gpu_smoke": {
            "title": "GPU platform smoke",
            "summary": "Verifies the ROCm/PyTorch stack loads and runs on GPU.",
            "headline_metrics": ["mean_step_time_ms"],
            "recipe": "recipes/ci/gpu-smoke.yaml",
            "min_gpus": 1,
            "run_command": "aorta sweep run --recipe recipes/ci/gpu-smoke.yaml",
            "repro": _workload_repro(
                artifact_workload="gpu_smoke",
                prerequisites=[
                    "One AMD GPU with working ROCm drivers",
                    "Python 3.10+ and a PyTorch build for your ROCm version",
                    "Network access to github.com/ROCm/aorta (clone + dev wheels)",
                ],
                recipe="recipes/ci/gpu-smoke.yaml",
                run_command="aorta sweep run --recipe recipes/ci/gpu-smoke.yaml",
                min_gpus=1,
                distributed=False,
                dry_run="aorta sweep run --recipe recipes/ci/gpu-smoke.yaml --dry-run",
                verify_title="Read the smoke matrix",
                success=(
                    "Rank 0 prints `Wrote matrix to ...` and matrix.md shows every "
                    "cell as pass (or record when no baseline exists yet). "
                    "mean_step_time_ms is the headline metric on the dashboard."
                ),
            ),
        },
        "inference_offline": {
            "title": "Offline LLM inference",
            "summary": "Validates prefill/decode latency, throughput, and logits checksum.",
            "headline_metrics": [
                "tokens_per_sec",
                "prefill_latency_ms",
                "decode_latency_ms",
                "logits_checksum",
            ],
            "recipe": "recipes/inference/example-inference-smoke.yaml",
            "min_gpus": 1,
            "run_command": (
                "aorta sweep run --recipe recipes/inference/example-inference-smoke.yaml"
            ),
            "repro": _workload_repro(
                artifact_workload="inference",
                prerequisites=[
                    "One AMD GPU with enough VRAM for the built-in smoke model",
                    "PyTorch with ROCm (the recipe uses AORTA's RepeatedBlockModel locally)",
                    "Same AORTA wheel channel as nightly when comparing checksums",
                ],
                recipe="recipes/inference/example-inference-smoke.yaml",
                run_command=(
                    "aorta sweep run --recipe recipes/inference/example-inference-smoke.yaml"
                ),
                min_gpus=1,
                distributed=False,
                dry_run=(
                    "aorta sweep run --recipe "
                    "recipes/inference/example-inference-smoke.yaml --dry-run"
                ),
                verify_title="Inspect latency, throughput, and checksum artifacts",
                verify_extra=[
                    "grep -E 'tokens_per_sec|prefill|decode|checksum' "
                    "triage_results/<TICKET>/inference/<timestamp>/matrix.md",
                ],
                success=(
                    "All cells pass against baseline (or record on first capture). "
                    "Headline metrics: tokens_per_sec, prefill/decode latency, "
                    "and logits_checksum must match baseline when graded pass."
                ),
            ),
        },
        "training_ddp": {
            "title": "PyTorch DDP training (2 GPU)",
            "summary": "Distributed data parallel training step time on two GPUs.",
            "headline_metrics": ["step_time_p50", "step_time_p99"],
            "recipe": "recipes/training/example-training-ddp-smoke.yaml",
            "min_gpus": 2,
            "run_command": (
                "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                "sweep run --recipe recipes/training/example-training-ddp-smoke.yaml"
            ),
            "repro": _workload_repro(
                artifact_workload="training",
                prerequisites=[
                    "Two AMD GPUs on one node, visible to a single PyTorch process group",
                    "RCCL usable between the two devices (check PCIe/xGMI topology)",
                    "torchrun on PATH (ships with PyTorch)",
                ],
                recipe="recipes/training/example-training-ddp-smoke.yaml",
                run_command=(
                    "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                    "sweep run --recipe recipes/training/example-training-ddp-smoke.yaml"
                ),
                min_gpus=2,
                distributed=True,
                dry_run=(
                    "aorta sweep run --recipe "
                    "recipes/training/example-training-ddp-smoke.yaml --dry-run"
                ),
                verify_title="Confirm both ranks finished and step times recorded",
                verify_extra=[
                    "grep -E 'step_time_p50|step_time_p99' "
                    "triage_results/<TICKET>/training/<timestamp>/matrix.md",
                ],
                success=(
                    "Both ranks exit cleanly; matrix.md lists step_time_p50/p99 per cell. "
                    "Nightly grades lower step time as better vs blessed baseline."
                ),
            ),
        },
        "training_ddp_8gpu": {
            "title": "PyTorch DDP training (8 GPU)",
            "summary": "Distributed data parallel training step time on eight GPUs.",
            "headline_metrics": ["step_time_p50", "step_time_p99"],
            "recipe": "recipes/training/example-training-ddp-smoke.yaml",
            "min_gpus": 8,
            "run_command": (
                "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                "sweep run --recipe recipes/training/example-training-ddp-smoke.yaml"
            ),
            "repro": _workload_repro(
                artifact_workload="training",
                prerequisites=[
                    "Eight AMD GPUs on one node (nightly reference: single MI350 node)",
                    "torchrun --standalone --nproc_per_node=8 must bind one rank per GPU",
                    "Enough host memory for eight concurrent training workers",
                ],
                recipe="recipes/training/example-training-ddp-smoke.yaml",
                run_command=(
                    "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                    "sweep run --recipe recipes/training/example-training-ddp-smoke.yaml"
                ),
                min_gpus=8,
                distributed=True,
                dry_run=(
                    "aorta sweep run --recipe "
                    "recipes/training/example-training-ddp-smoke.yaml --dry-run"
                ),
                verify_title="Check weak-scaling step times on eight ranks",
                verify_extra=[
                    "grep step_time_p50 triage_results/<TICKET>/training/<timestamp>/matrix.md",
                ],
                success=(
                    "All eight ranks participate; step_time_p50/p99 recorded. "
                    "Compare to the 2-GPU DDP row in the scaling section on the dashboard."
                ),
            ),
        },
        "training_fsdp": {
            "title": "PyTorch FSDP training (2 GPU)",
            "summary": "Fully sharded data parallel training on two GPUs.",
            "headline_metrics": ["step_time_p50", "step_time_p99"],
            "recipe": "recipes/training/example-training-fsdp-smoke.yaml",
            "min_gpus": 2,
            "run_command": (
                "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                "sweep run --recipe recipes/training/example-training-fsdp-smoke.yaml"
            ),
            "repro": _workload_repro(
                artifact_workload="training",
                prerequisites=[
                    "Two AMD GPUs with FSDP-compatible PyTorch build",
                    "Same torchrun/NCCL setup as DDP (FSDP still uses dist.init_process_group)",
                ],
                recipe="recipes/training/example-training-fsdp-smoke.yaml",
                run_command=(
                    "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                    "sweep run --recipe recipes/training/example-training-fsdp-smoke.yaml"
                ),
                min_gpus=2,
                distributed=True,
                dry_run=(
                    "aorta sweep run --recipe "
                    "recipes/training/example-training-fsdp-smoke.yaml --dry-run"
                ),
                verify_title="Read FSDP step-time matrix",
                success=(
                    "matrix.md shows pass/record per cell with step_time_p50/p99. "
                    "Failures usually indicate sharding or RCCL init problems across the two GPUs."
                ),
            ),
        },
        "training_fsdp_8gpu": {
            "title": "PyTorch FSDP training (8 GPU)",
            "summary": "Fully sharded data parallel training on eight GPUs.",
            "headline_metrics": ["step_time_p50", "step_time_p99"],
            "recipe": "recipes/training/example-training-fsdp-smoke.yaml",
            "min_gpus": 8,
            "run_command": (
                "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                "sweep run --recipe recipes/training/example-training-fsdp-smoke.yaml"
            ),
            "repro": _workload_repro(
                artifact_workload="training",
                prerequisites=[
                    "Full eight-GPU node (nightly runs on a single MI350 host)",
                    "FSDP requires stable NCCL/RCCL across all eight ranks",
                ],
                recipe="recipes/training/example-training-fsdp-smoke.yaml",
                run_command=(
                    "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                    "sweep run --recipe recipes/training/example-training-fsdp-smoke.yaml"
                ),
                min_gpus=8,
                distributed=True,
                dry_run=(
                    "aorta sweep run --recipe "
                    "recipes/training/example-training-fsdp-smoke.yaml --dry-run"
                ),
                verify_title="Confirm eight-rank FSDP matrix",
                success=(
                    "All ranks complete; step times appear in matrix.md. "
                    "Use the dashboard scaling table to compare 2-GPU vs 8-GPU FSDP efficiency."
                ),
            ),
        },
        "llm_determinism": {
            "title": "LLM determinism (2 GPU)",
            "summary": (
                "Bit-exact repeatability across ranks; divergence indicates "
                "silent corruption."
            ),
            "headline_metrics": ["ranks_with_divergence"],
            "recipe": "recipes/llm-determinism/example-llm-determinism.yaml",
            "min_gpus": 2,
            "run_command": (
                "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                "sweep run --recipe recipes/llm-determinism/example-llm-determinism.yaml"
            ),
            "repro": _workload_repro(
                artifact_workload="llm_determinism",
                prerequisites=[
                    "Two GPUs — determinism is checked across ranks in one job",
                    "Identical random seeds and deterministic PyTorch ops where required",
                ],
                recipe="recipes/llm-determinism/example-llm-determinism.yaml",
                run_command=(
                    "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                    "sweep run --recipe recipes/llm-determinism/example-llm-determinism.yaml"
                ),
                min_gpus=2,
                distributed=True,
                dry_run=(
                    "aorta sweep run --recipe "
                    "recipes/llm-determinism/example-llm-determinism.yaml --dry-run"
                ),
                verify_title="Look for rank divergence in matrix and trial JSON",
                verify_extra=[
                    "grep -i diverge triage_results/<TICKET>/llm_determinism/<timestamp>/matrix.md",
                    "find triage_results/<TICKET>/llm_determinism/<timestamp>/cells -name 'trial_*.json' | head",
                ],
                success=(
                    "ranks_with_divergence must be 0 for a pass. Any non-zero value "
                    "means silent numerical corruption between ranks and should fail nightly CI."
                ),
            ),
        },
        "llm_determinism_8gpu": {
            "title": "LLM determinism (8 GPU)",
            "summary": "Bit-exact repeatability at full-node scale.",
            "headline_metrics": ["ranks_with_divergence"],
            "recipe": "recipes/llm-determinism/example-llm-determinism.yaml",
            "min_gpus": 8,
            "run_command": (
                "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                "sweep run --recipe recipes/llm-determinism/example-llm-determinism.yaml"
            ),
            "repro": _workload_repro(
                artifact_workload="llm_determinism",
                prerequisites=[
                    "Eight-GPU node — catches determinism bugs that only appear at scale",
                    "Stable RCCL collectives; any rank mismatch fails the workload",
                ],
                recipe="recipes/llm-determinism/example-llm-determinism.yaml",
                run_command=(
                    "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                    "sweep run --recipe recipes/llm-determinism/example-llm-determinism.yaml"
                ),
                min_gpus=8,
                distributed=True,
                dry_run=(
                    "aorta sweep run --recipe "
                    "recipes/llm-determinism/example-llm-determinism.yaml --dry-run"
                ),
                verify_title="Verify zero divergence across all eight ranks",
                success=(
                    "ranks_with_divergence == 0 on every cell. This is a correctness gate, "
                    "not a performance benchmark."
                ),
            ),
        },
        "race": {
            "title": "RCCL race detection (2 GPU)",
            "summary": (
                "Detects timing races and silent data corruption in distributed layers."
            ),
            "headline_metrics": ["layer_checksum_mismatches"],
            "recipe": "recipes/race/race_smoke.yaml",
            "min_gpus": 2,
            "run_command": (
                "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                "sweep run --recipe recipes/race/race_smoke.yaml"
            ),
            "repro": _workload_repro(
                artifact_workload="race",
                prerequisites=[
                    "Two GPUs — race workload stresses concurrent RCCL + layer checksums",
                    "Race workloads use fresh process isolation per trial (see recipe)",
                ],
                recipe="recipes/race/race_smoke.yaml",
                run_command=(
                    "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                    "sweep run --recipe recipes/race/race_smoke.yaml"
                ),
                min_gpus=2,
                distributed=True,
                dry_run="aorta sweep run --recipe recipes/race/race_smoke.yaml --dry-run",
                verify_title="Check layer checksum mismatches in matrix output",
                verify_extra=[
                    "grep -i checksum triage_results/<TICKET>/race/<timestamp>/matrix.md",
                ],
                success=(
                    "layer_checksum_mismatches must be 0. Non-zero values indicate "
                    "detected races or silent corruption in distributed layers."
                ),
                setup_extra=[
                    {
                        "title": "Race-specific: reserve trial master ports on static launchers",
                        "commands": [
                            "export AORTA_TRIAL_MASTER_PORT_BASE=30000",
                            "# Required for srun / static torchrun without elastic agent store",
                        ],
                    },
                ],
            ),
        },
        "race_8gpu": {
            "title": "RCCL race detection (8 GPU)",
            "summary": "Race detection at full-node scale.",
            "headline_metrics": ["layer_checksum_mismatches"],
            "recipe": "recipes/race/race_smoke.yaml",
            "min_gpus": 8,
            "run_command": (
                "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                "sweep run --recipe recipes/race/race_smoke.yaml"
            ),
            "repro": _workload_repro(
                artifact_workload="race",
                prerequisites=[
                    "Eight-GPU node — amplifies timing races vs the 2-GPU smoke",
                    "Export AORTA_TRIAL_MASTER_PORT_BASE on static launchers (see README-running-recipes)",
                ],
                recipe="recipes/race/race_smoke.yaml",
                run_command=(
                    "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                    "sweep run --recipe recipes/race/race_smoke.yaml"
                ),
                min_gpus=8,
                distributed=True,
                dry_run="aorta sweep run --recipe recipes/race/race_smoke.yaml --dry-run",
                verify_title="Confirm zero checksum mismatches at 8-GPU scale",
                success=(
                    "layer_checksum_mismatches == 0 for every cell. "
                    "Any failure warrants inspecting per-trial JSON under cells/."
                ),
                setup_extra=[
                    {
                        "title": "Race-specific: reserve trial master ports on static launchers",
                        "commands": [
                            "export AORTA_TRIAL_MASTER_PORT_BASE=30000",
                        ],
                    },
                ],
            ),
        },
    },
}
