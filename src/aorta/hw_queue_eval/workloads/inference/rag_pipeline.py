"""
Multi-Model RAG Pipeline Workload

Pattern: Multiple models process different stages of RAG
- Embedding model stream(s): Encode queries/documents
- Retriever stream(s): Vector similarity search
- Reranker stream(s): Score retrieved documents
- Generator stream(s): Produce final response

This tests concurrent execution of different model types.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aorta.hw_queue_eval.workloads.base import InferenceWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class EmbeddingModel(nn.Module):
    """Simple embedding model for encoding."""

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean pooling over sequence
        return self.encoder(x.mean(dim=1))


class RerankerModel(nn.Module):
    """Cross-encoder reranker."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, query: torch.Tensor, docs: torch.Tensor) -> torch.Tensor:
        # query: [batch, hidden]
        # docs: [batch, num_docs, hidden]
        batch, num_docs, hidden = docs.shape
        query_expanded = query.unsqueeze(1).expand(-1, num_docs, -1)
        combined = torch.cat([query_expanded, docs], dim=-1)
        scores = self.scorer(combined).squeeze(-1)
        return scores


class GeneratorModel(nn.Module):
    """Simple generator for response generation."""

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 4):
        super().__init__()
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


@WorkloadRegistry.register
class RAGPipelineWorkload(InferenceWorkload):
    """
    Multi-model RAG pipeline simulation.

    Simulates a RAG system with:
    1. Query embedding (encode user query)
    2. Document retrieval (vector similarity)
    3. Reranking (cross-encoder scoring)
    4. Generation (produce response)

    Stream assignment:
    - Streams 0-1: Embedding model
    - Streams 2-3: Retrieval (vector ops)
    - Streams 4-5: Reranker model
    - Streams 6-7: Generator model
    """

    name = "rag_pipeline"
    description = "Multi-model RAG pipeline"
    category = "inference"
    min_streams = 4
    max_streams = 12
    recommended_streams = 8
    switch_latency_sensitivity = "high"
    memory_requirements_gb = 4.0

    def __init__(
        self,
        hidden_size: int = 768,
        embedding_dim: int = 256,
        vocab_size: int = 32000,
        batch_size: int = 4,
        query_length: int = 64,
        num_docs: int = 100,
        top_k: int = 10,
        generation_length: int = 128,
    ):
        """
        Initialize RAG pipeline workload.

        Args:
            hidden_size: Model hidden size
            embedding_dim: Embedding dimension for retrieval
            vocab_size: Vocabulary size for generator
            batch_size: Batch size
            query_length: Query sequence length
            num_docs: Number of documents in corpus
            top_k: Number of documents to retrieve
            generation_length: Output generation length
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.query_length = query_length
        self.num_docs = num_docs
        self.top_k = top_k
        self.generation_length = generation_length

        self._embedding_model: Optional[EmbeddingModel] = None
        self._reranker_model: Optional[RerankerModel] = None
        self._generator_model: Optional[GeneratorModel] = None

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup models and document corpus."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Create models
        self._embedding_model = EmbeddingModel(
            self.hidden_size, self.embedding_dim
        ).to(device)
        self._embedding_model.eval()

        self._reranker_model = RerankerModel(self.hidden_size).to(device)
        self._reranker_model.eval()

        self._generator_model = GeneratorModel(
            self.hidden_size, self.vocab_size
        ).to(device)
        self._generator_model.eval()

        # Stream assignments
        quarter = max(1, stream_count // 4)
        self._embed_streams = list(range(0, quarter))
        self._retrieve_streams = list(range(quarter, 2 * quarter))
        self._rerank_streams = list(range(2 * quarter, 3 * quarter))
        self._generate_streams = list(range(3 * quarter, stream_count))

        # Ensure at least one stream per stage
        for stream_list in [self._retrieve_streams, self._rerank_streams, self._generate_streams]:
            if not stream_list:
                stream_list.append(0)

        # Query input
        self._query_input = torch.randn(
            self.batch_size, self.query_length, self.hidden_size,
            dtype=torch.float32, device=device
        )
        self._tensors["query_input"] = self._query_input

        # Document corpus (pre-embedded)
        self._doc_embeddings = torch.randn(
            self.num_docs, self.embedding_dim,
            dtype=torch.float32, device=device
        )
        self._tensors["doc_embeddings"] = self._doc_embeddings

        # Document representations for reranking
        self._doc_hidden = torch.randn(
            self.num_docs, self.hidden_size,
            dtype=torch.float32, device=device
        )
        self._tensors["doc_hidden"] = self._doc_hidden

        # Tokens per iteration
        self._tokens_per_iteration = self.batch_size * self.generation_length

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one RAG iteration.

        Pipeline stages run with dependencies:
        1. Embed query
        2. Retrieve documents
        3. Rerank retrieved documents
        4. Generate response
        """
        embed_stream = streams[self._embed_streams[0]]
        retrieve_stream = streams[self._retrieve_streams[0]]
        rerank_stream = streams[self._rerank_streams[0]]
        generate_stream = streams[self._generate_streams[0]]

        # Stage 1: Embed query
        with torch.cuda.stream(embed_stream):
            query_embedding = self._embedding_model(self._query_input)
            # Normalize for similarity
            query_embedding = F.normalize(query_embedding, dim=-1)

        # Stage 2: Retrieve documents (wait for embedding)
        retrieve_stream.wait_stream(embed_stream)

        with torch.cuda.stream(retrieve_stream):
            # Compute similarities
            similarities = torch.matmul(
                query_embedding, self._doc_embeddings.T
            )  # [batch, num_docs]

            # Get top-k
            top_scores, top_indices = torch.topk(similarities, self.top_k, dim=-1)

            # Gather top-k document representations
            # [batch, top_k, hidden_size]
            retrieved_docs = self._doc_hidden[top_indices.flatten()].reshape(
                self.batch_size, self.top_k, self.hidden_size
            )

        # Stage 3: Rerank (wait for retrieval)
        rerank_stream.wait_stream(retrieve_stream)

        with torch.cuda.stream(rerank_stream):
            query_hidden = self._query_input.mean(dim=1)  # [batch, hidden]
            rerank_scores = self._reranker_model(query_hidden, retrieved_docs)

            # Get reranked order
            _, reranked_indices = torch.sort(rerank_scores, dim=-1, descending=True)

            # Reorder documents
            batch_indices = torch.arange(self.batch_size, device=self._device).unsqueeze(1)
            reranked_docs = retrieved_docs[batch_indices, reranked_indices]

        # Stage 4: Generate response (wait for reranking)
        generate_stream.wait_stream(rerank_stream)

        with torch.cuda.stream(generate_stream):
            # Concatenate query and top reranked docs as context
            context = torch.cat([
                self._query_input,
                reranked_docs[:, :3, :].expand(-1, -1, -1).reshape(
                    self.batch_size, -1, self.hidden_size
                )
            ], dim=1)

            # Generate
            output_logits = self._generator_model(context)

    def get_throughput_unit(self) -> str:
        return "queries/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        return (iterations * self.batch_size) / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hidden_size": self.hidden_size,
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "batch_size": self.batch_size,
            "query_length": self.query_length,
            "num_docs": self.num_docs,
            "top_k": self.top_k,
            "generation_length": self.generation_length,
            "stream_assignment": {
                "embed": self._embed_streams,
                "retrieve": self._retrieve_streams,
                "rerank": self._rerank_streams,
                "generate": self._generate_streams,
            },
        }

    def cleanup(self) -> None:
        """Cleanup models."""
        super().cleanup()
        self._embedding_model = None
        self._reranker_model = None
        self._generator_model = None
