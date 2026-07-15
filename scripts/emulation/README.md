# Emulation helper scripts

Scripts for running AORTA under the mirage GPU emulator (PR #227).

## Quick start

```bash
export MIRAGE_BIN=/path/to/mirage/build/manylinux/bin/mirage
export MIRAGE_AORTA_IMAGE=docker.io/vllm/vllm-openai-rocm:latest

./scripts/emulation/run_mirage_container.sh gpu-smoke
./scripts/emulation/run_mirage_container.sh probe
./scripts/emulation/run_mirage_container.sh inference-smoke
./scripts/emulation/run_mirage_container.sh training-ddp-smoke
./scripts/emulation/run_mirage_container.sh training-fsdp-smoke
./scripts/emulation/run_mirage_container.sh llm-determinism   # slow under rocjitsu
./scripts/emulation/run_mirage_matrix.sh
```

## Covered workloads

| Command | Workload | Recipe | Notes |
|---------|----------|--------|-------|
| `gpu-smoke` | `gpu_smoke` | `recipes/emulated/gpu-smoke-emulated.yaml` | single-process |
| `probe` | `_subprocess` | `recipes/probe/example-probe-smoke.yaml` | wraps argv in mirage |
| `inference-smoke` | `inference` | `recipes/emulated/inference-smoke-emulated.yaml` | single-process |
| `training-ddp-smoke` | `training` (ddp) | `recipes/emulated/training-ddp-smoke-emulated.yaml` | world_size=1 singleton |
| `training-fsdp-smoke` | `training` (fsdp) | `recipes/emulated/training-fsdp-smoke-emulated.yaml` | world_size=1 singleton |
| `llm-determinism` | `llm_determinism` | `recipes/emulated/llm-determinism-emulated.yaml` | world_size=1; slow under rocjitsu |

The `training` and `llm_determinism` recipes are single-rank singleton smokes
(they exercise the lifecycle + JSON schema, not multi-rank collectives). A real
multi-rank path needs `mirage run --daemon --gpus-per-node N` + `torchrun` and is
a later phase. The matrix runs the single-process cases by default; add the slow
`llm_determinism` case with `INCLUDE_LLM_DET=1` (and a larger `TIMEOUT_SEC`).
