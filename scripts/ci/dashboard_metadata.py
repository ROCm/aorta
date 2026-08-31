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

_TRAINING_SUCCESS = (
    "All ranks complete without error. step_time_p50/p99 appear in perf.md. "
    "Nightly currently records training cells (passed/recording only); "
    "step-time thresholds are not perf-gated in regression_baselines.yaml yet."
)

_METRICS_IN_PERF = (
    'grep -m 10 -E "{pattern}" "$RUN_DIR/perf.md" 2>/dev/null '
    '|| python3 -c "import json,sys; d=json.load(open(sys.argv[1])); '
    'print(json.dumps(d.get(\\\"cells\\\",[]), indent=2)[:2000])" "$RUN_DIR/matrix.json"'
)

def _trial_metric_in_artifacts(*, perf_grep_pattern: str, metric_key: str) -> list[str]:
    """Advisory perf.md grep plus mandatory raw-trial JSON read.

    perf.md aggregates passing trials only, so a mixed pass/fail run can show a
    passing zero while failed trials still carry non-zero correctness metrics.
    """
    return [
        f'grep -m 5 -E "{perf_grep_pattern}" "$RUN_DIR/perf.md" 2>/dev/null || true',
        (
            "python3 - \"$RUN_DIR\" <<'PY'\n"
            "import json\n"
            "from pathlib import Path\n"
            "import sys\n"
            "\n"
            "run = Path(sys.argv[1])\n"
            "doc = json.loads((run / 'matrix.json').read_text(encoding='utf-8'))\n"
            "out = {}\n"
            "for cell in doc.get('cells') or []:\n"
            "    values = []\n"
            "    for raw in cell.get('trial_paths') or []:\n"
            "        path = Path(raw)\n"
            "        if not path.is_absolute() and not path.exists():\n"
            "            path = run / path\n"
            "        if path.is_dir():\n"
            "            path = path / 'result.json'\n"
            "        if not path.is_file():\n"
            "            continue\n"
            "        trial = json.loads(path.read_text(encoding='utf-8'))\n"
            "        metrics = (trial.get('result') or {}).get('metrics') or {}\n"
            f"        if '{metric_key}' in metrics:\n"
            f"            values.append(metrics['{metric_key}'])\n"
            "    out[cell.get('name', '?')] = values or None\n"
            "print(json.dumps(out, indent=2))\n"
            "PY"
        ),
    ]


_DIVERGENCE_IN_ARTIFACTS = _trial_metric_in_artifacts(
    perf_grep_pattern="ranks_with_divergence|diverge",
    metric_key="ranks_with_divergence",
)
_RACE_CHECKSUM_IN_ARTIFACTS = _trial_metric_in_artifacts(
    perf_grep_pattern="layer_checksum_mismatch",
    metric_key="layer_checksum_mismatches",
)


def _repro_output_root(entry_name: str) -> str:
    return f"triage_results/repro/{entry_name}"


def _read_matrix_cmds(repro_root: str) -> list[str]:
    """Shell helpers to locate and read the newest matrix for one repro run."""
    root = repro_root
    return [
        (
            f'RUN_DIR="$(find {root} -mindepth 3 -maxdepth 3 -type d '
            f'-printf \'%T@ %p\\n\' 2>/dev/null '
            f'| sort -rn | head -1 | cut -d\' \' -f2-)"'
        ),
        'if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then echo "No run directory found"; exit 1; fi',
        'echo "Using run directory: $RUN_DIR"',
        'cat "$RUN_DIR/matrix.md"',
        'python3 -m json.tool "$RUN_DIR/matrix.json" | head -80',
        'if [ -f "$RUN_DIR/perf.md" ]; then head -60 "$RUN_DIR/perf.md"; else echo "(no perf.md)"; fi',
        (
            "# Standalone --strict only catches errored or not-run cells; dashboard "
            "pass/record/fail comes from nightly_eval.py separately comparing matrix.json "
            "to config/ci/regression_baselines.yaml"
        ),
    ]


def _install_setup(*, min_gpus: int, distributed: bool = False) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {
            "title": "Check out the dashboard commit and install the matching AORTA wheel",
            "commands": [
                "git clone https://github.com/ROCm/aorta.git",
                "cd aorta",
                "git checkout {{HEAD_SHA}}",
                "python3 -m pip install --upgrade pip",
                (
                    "python3 -m pip install --upgrade --pre "
                    "'amd-aorta[hw-queue]=={{AORTA_VERSION}}' "
                    "-f https://github.com/ROCm/aorta/releases/expanded_assets/dev-wheels"
                ),
                "# If that exact wheel is unavailable, pick the closest dev-wheel build",
                "# and compare `aorta --version` with the dashboard header before reproducing.",
                "aorta --help",
            ],
        },
        {
            "title": "Confirm ROCm + PyTorch see enough GPUs",
            "commands": [
                (
                    'python3 -c "import torch; n=torch.cuda.device_count(); '
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
    entry_name: str,
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
    repro_root = _repro_output_root(entry_name)
    verify_cmds = _read_matrix_cmds(repro_root)
    if verify_extra:
        verify_cmds.extend(verify_extra)
    setup = _install_setup(min_gpus=min_gpus, distributed=distributed)
    if setup_extra:
        setup.extend(setup_extra)
    cmd = run_command.rstrip()
    if "--output-dir" not in cmd:
        cmd = f"{cmd} --output-dir {repro_root}"
    strict_cmd = cmd if cmd.endswith("--strict") else f"{cmd} --strict"
    return {
        "prerequisites": prerequisites,
        "setup": setup,
        "dry_run": {
            "title": "Validate the recipe YAML (no GPU execution)",
            "command": dry_run,
        },
        "run": {
            "title": "Run the recipe with --strict (same flag as nightly CI)",
            "command": strict_cmd,
        },
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
        "serving": {
            "label": "Serving",
            "summary": "Online LLM serving latency (TTFT/TPOT) and token throughput.",
            "workloads": ["tokenspeed_serve_smoke"],
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
                entry_name="gpu_smoke",
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
                    "Rank 0 prints `Wrote matrix to ...`. matrix.md shows every cell "
                    "completed; mean_step_time_ms is recorded in perf.md. Nightly gates "
                    "pass/fail against the blessed gpu_smoke::baseline-local baseline; "
                    "step time is recorded but not performance-thresholded."
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
                entry_name="inference_offline",
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
                    _METRICS_IN_PERF.format(
                        pattern="tokens_per_sec|prefill|decode|checksum"
                    ),
                ],
                success=(
                    "Recipe completes with --strict. logits_checksum is the active "
                    "correctness gate in regression_baselines.yaml; latency and "
                    "throughput are recorded on the dashboard but not perf-gated yet."
                ),
            ),
        },
        "tokenspeed_serve_smoke": {
            "title": "TokenSpeed online serving",
            "summary": (
                "Time to first token, time per output token, and token throughput "
                "from a containerised TokenSpeed server."
            ),
            # Deliberately not the p99s or median_itl_ms: see
            # docs/tokenspeed-gating-rollout.md for which serving metrics are
            # stable enough to read as headline numbers.
            "headline_metrics": [
                "median_ttft_ms",
                "median_tpot_ms",
                "output_throughput",
            ],
            "recipe": "recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml",
            "min_gpus": 1,
            "run_command": (
                "aorta sweep run --recipe "
                "recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml"
            ),
            "repro": _workload_repro(
                entry_name="tokenspeed_serve_smoke",
                prerequisites=[
                    "One gfx950 (MI355X) or gfx1250 GPU — the TokenSpeed image targets these",
                    "A working docker client and daemon: the engine runs in its own container",
                    "A node-local work_dir — an NFS home under root-squash cannot be bind-mounted",
                    "Egress to the Hugging Face Hub, or a pre-populated cache plus hf_offline",
                ],
                recipe="recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml",
                run_command=(
                    "aorta sweep run --recipe "
                    "recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml"
                ),
                min_gpus=1,
                distributed=False,
                dry_run=(
                    "aorta sweep run --recipe "
                    "recipes/tokenspeed/tokenspeed-serve-bench-smoke.yaml --dry-run"
                ),
                verify_title="Inspect serving latency and throughput artifacts",
                verify_extra=[
                    _METRICS_IN_PERF.format(
                        pattern="ttft|tpot|output_throughput|completed_total"
                    ),
                ],
                success=(
                    "Both cells pass with completed_total == num_prompts * steps and "
                    "failed_total == 0. median_ttft_ms / median_tpot_ms / "
                    "output_throughput appear in perf.md. Serving cells are "
                    "record-only until a baseline is blessed — see "
                    "docs/tokenspeed-gating-rollout.md for which metrics get bounds "
                    "and why bring-up time never does (189-379s on one node with "
                    "nothing changed)."
                ),
                setup_extra=[
                    {
                        "title": "Serving-specific: pre-warm the model cache as the running uid",
                        "commands": [
                            "export HF_HOME=/tmp/ts-work-serve/u$(id -u)/hf",
                            "# run_as_current_user defaults to true: a cache populated by a",
                            "# root container leaves the trial failing with PermissionError",
                            "docker pull lightseekorg/tokenspeed-amd@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78",
                        ],
                    },
                ],
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
                entry_name="training_ddp",
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
                    _METRICS_IN_PERF.format(pattern="step_time_p50|step_time_p99"),
                ],
                success=_TRAINING_SUCCESS,
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
                entry_name="training_ddp_8gpu",
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
                    _METRICS_IN_PERF.format(pattern="step_time_p50"),
                ],
                success=(
                    f"{_TRAINING_SUCCESS} Compare 2-GPU vs 8-GPU step_time_p50 in the "
                    "dashboard scaling section."
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
                entry_name="training_fsdp",
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
                verify_title="Read FSDP step-time artifacts",
                verify_extra=[
                    _METRICS_IN_PERF.format(pattern="step_time_p50|step_time_p99"),
                ],
                success=_TRAINING_SUCCESS,
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
                entry_name="training_fsdp_8gpu",
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
                verify_title="Confirm eight-rank FSDP artifacts",
                verify_extra=[
                    _METRICS_IN_PERF.format(pattern="step_time_p50"),
                ],
                success=(
                    f"{_TRAINING_SUCCESS} Use the dashboard scaling table to compare "
                    "2-GPU vs 8-GPU FSDP efficiency."
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
                entry_name="llm_determinism",
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
                verify_title="Confirm ranks_with_divergence from perf.md or matrix.json",
                verify_extra=_DIVERGENCE_IN_ARTIFACTS,
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
                entry_name="llm_determinism_8gpu",
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
                verify_title="Confirm ranks_with_divergence is zero on all eight ranks",
                verify_extra=_DIVERGENCE_IN_ARTIFACTS,
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
                entry_name="race",
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
                verify_extra=_RACE_CHECKSUM_IN_ARTIFACTS,
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
                entry_name="race_8gpu",
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
                verify_extra=_RACE_CHECKSUM_IN_ARTIFACTS,
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
