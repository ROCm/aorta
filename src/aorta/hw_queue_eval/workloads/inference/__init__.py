"""
Inference workloads for hardware queue evaluation.

These workloads simulate patterns from LLM inference:
- Speculative decoding with draft/verify models
- Continuous batching with prefill/decode overlap
- Multi-model RAG pipelines
"""

from aorta.hw_queue_eval.workloads.inference.continuous_batch import ContinuousBatchWorkload
from aorta.hw_queue_eval.workloads.inference.rag_pipeline import RAGPipelineWorkload
from aorta.hw_queue_eval.workloads.inference.speculative_decode import SpeculativeDecodeWorkload

__all__ = [
    "SpeculativeDecodeWorkload",
    "ContinuousBatchWorkload",
    "RAGPipelineWorkload",
]
