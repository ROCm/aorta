"""Customer-facing labels for the nightly CI dashboard.

Kept as a Python module (not JSON) because the repo gitignores ``*.json``.
"""

from __future__ import annotations

from typing import Any

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
            "run_command": "aorta sweep run --recipe recipes/ci/gpu-smoke.yaml",
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
            "run_command": (
                "aorta sweep run --recipe recipes/inference/example-inference-smoke.yaml"
            ),
        },
        "training_ddp": {
            "title": "PyTorch DDP training (2 GPU)",
            "summary": "Distributed data parallel training step time on two GPUs.",
            "headline_metrics": ["step_time_p50", "step_time_p99"],
            "run_command": (
                "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                "sweep run --recipe recipes/training/example-training-ddp-smoke.yaml"
            ),
        },
        "training_ddp_8gpu": {
            "title": "PyTorch DDP training (8 GPU)",
            "summary": "Distributed data parallel training step time on eight GPUs.",
            "headline_metrics": ["step_time_p50", "step_time_p99"],
            "run_command": (
                "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                "sweep run --recipe recipes/training/example-training-ddp-smoke.yaml"
            ),
        },
        "training_fsdp": {
            "title": "PyTorch FSDP training (2 GPU)",
            "summary": "Fully sharded data parallel training on two GPUs.",
            "headline_metrics": ["step_time_p50", "step_time_p99"],
            "run_command": (
                "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                "sweep run --recipe recipes/training/example-training-fsdp-smoke.yaml"
            ),
        },
        "training_fsdp_8gpu": {
            "title": "PyTorch FSDP training (8 GPU)",
            "summary": "Fully sharded data parallel training on eight GPUs.",
            "headline_metrics": ["step_time_p50", "step_time_p99"],
            "run_command": (
                "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                "sweep run --recipe recipes/training/example-training-fsdp-smoke.yaml"
            ),
        },
        "llm_determinism": {
            "title": "LLM determinism (2 GPU)",
            "summary": (
                "Bit-exact repeatability across ranks; divergence indicates "
                "silent corruption."
            ),
            "headline_metrics": ["ranks_with_divergence"],
            "run_command": (
                "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                "sweep run --recipe recipes/llm-determinism/example-llm-determinism.yaml"
            ),
        },
        "llm_determinism_8gpu": {
            "title": "LLM determinism (8 GPU)",
            "summary": "Bit-exact repeatability at full-node scale.",
            "headline_metrics": ["ranks_with_divergence"],
            "run_command": (
                "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                "sweep run --recipe recipes/llm-determinism/example-llm-determinism.yaml"
            ),
        },
        "race": {
            "title": "RCCL race detection (2 GPU)",
            "summary": (
                "Detects timing races and silent data corruption in distributed layers."
            ),
            "headline_metrics": ["layer_checksum_mismatches"],
            "run_command": (
                "torchrun --standalone --nproc_per_node=2 $(which aorta) "
                "sweep run --recipe recipes/race/race_smoke.yaml"
            ),
        },
        "race_8gpu": {
            "title": "RCCL race detection (8 GPU)",
            "summary": "Race detection at full-node scale.",
            "headline_metrics": ["layer_checksum_mismatches"],
            "run_command": (
                "torchrun --standalone --nproc_per_node=8 $(which aorta) "
                "sweep run --recipe recipes/race/race_smoke.yaml"
            ),
        },
    },
    "engineer_only_metrics": [
        "rank",
        "world_size",
        "local_world_size",
        "node_count",
        "corruption_details_omitted",
        "parameter_count",
        "num_experts",
        "num_layers",
        "generate_tokens",
        "decoded_tokens",
        "prompt_len",
        "batch_size",
        "eff_batch_size",
        "eff_ffn_size",
        "eff_num_heads",
        "eff_seq_len",
        "declared_h2d_tensor_size",
        "effective_h2d_tensor_size",
        "layers_verified",
        "expected",
        "n",
        "sum",
        "final_loss",
    ],
}
