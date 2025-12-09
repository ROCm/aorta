"""
Async Data Loading Workload

Pattern: Overlap CPU data loading with GPU preprocessing
- Data transfer streams: H2D copies
- Preprocessing streams: GPU data augmentation
- Compute streams: Model forward/backward

This tests the ability to hide data transfer latency.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from aorta.hw_queue_eval.workloads.base import BaseWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class SimpleConvNet(nn.Module):
    """Simple CNN for data pipeline testing."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


@WorkloadRegistry.register
class AsyncDataLoadWorkload(BaseWorkload):
    """
    Async data loading with GPU preprocessing overlap.

    Simulates a data pipeline where:
    1. Data is transferred from CPU to GPU (H2D)
    2. GPU preprocessing (augmentation, normalization)
    3. Model training/inference

    These stages are pipelined across multiple batches.

    Stream assignment:
    - Streams 0-1: Data transfer (H2D)
    - Streams 2-3: GPU preprocessing
    - Streams 4+: Compute (forward/backward)
    """

    name = "async_dataload"
    description = "Async data loading with GPU preprocessing overlap"
    category = "pipeline"
    min_streams = 4
    max_streams = 12
    recommended_streams = 6
    switch_latency_sensitivity = "medium"
    memory_requirements_gb = 4.0

    def __init__(
        self,
        batch_size: int = 32,
        image_size: int = 224,
        num_prefetch: int = 2,
        num_classes: int = 1000,
    ):
        """
        Initialize async data loading workload.

        Args:
            batch_size: Batch size per pipeline stage
            image_size: Image dimensions (square)
            num_prefetch: Number of batches to prefetch
            num_classes: Number of output classes
        """
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_prefetch = num_prefetch
        self.num_classes = num_classes

        self._model = None
        self._cpu_data: List[torch.Tensor] = []
        self._gpu_buffers: List[torch.Tensor] = []
        self._preprocessed: List[torch.Tensor] = []

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup model, CPU data, and GPU buffers."""
        self._stream_count = stream_count
        self._device = device
        self._is_setup = True

        # Create model
        self._model = SimpleConvNet(num_classes=self.num_classes).to(device)
        self._model.eval()

        # Stream assignments
        third = max(1, stream_count // 3)
        self._transfer_streams = list(range(0, third))
        self._preprocess_streams = list(range(third, 2 * third))
        self._compute_streams = list(range(2 * third, stream_count))

        if not self._preprocess_streams:
            self._preprocess_streams = [0]
        if not self._compute_streams:
            self._compute_streams = [0]

        # Create CPU data (pinned memory for async transfer)
        self._cpu_data = []
        for i in range(self.num_prefetch):
            data = torch.randn(
                self.batch_size, 3, self.image_size, self.image_size,
                dtype=torch.float32
            ).pin_memory()
            self._cpu_data.append(data)

        # GPU buffers for receiving data
        self._gpu_buffers = []
        for i in range(self.num_prefetch):
            buf = torch.empty(
                self.batch_size, 3, self.image_size, self.image_size,
                dtype=torch.float32, device=device
            )
            self._gpu_buffers.append(buf)
            self._tensors[f"gpu_buffer_{i}"] = buf

        # Preprocessed data buffers
        self._preprocessed = []
        for i in range(self.num_prefetch):
            buf = torch.empty(
                self.batch_size, 3, self.image_size, self.image_size,
                dtype=torch.float32, device=device
            )
            self._preprocessed.append(buf)
            self._tensors[f"preprocessed_{i}"] = buf

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one pipelined iteration.

        Pipeline:
        1. Transfer batch N+1 while processing batch N
        2. Preprocess batch N while computing batch N-1
        """
        transfer_stream = streams[self._transfer_streams[0]]
        preprocess_stream = streams[self._preprocess_streams[0]]
        compute_stream = streams[self._compute_streams[0]]

        # Process multiple batches in pipeline
        for batch_idx in range(self.num_prefetch):
            # Stage 1: Async transfer to GPU
            with torch.cuda.stream(transfer_stream):
                self._gpu_buffers[batch_idx].copy_(
                    self._cpu_data[batch_idx], non_blocking=True
                )

            # Stage 2: GPU preprocessing (wait for transfer)
            preprocess_stream.wait_stream(transfer_stream)

            with torch.cuda.stream(preprocess_stream):
                # Simulate preprocessing: normalize, augment
                data = self._gpu_buffers[batch_idx]

                # Normalize (ImageNet-style)
                mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(1, 3, 1, 1)
                normalized = (data - mean) / std

                # Random horizontal flip simulation
                flip_mask = torch.rand(self.batch_size, 1, 1, 1, device=self._device) > 0.5
                flipped = torch.where(flip_mask, normalized.flip(-1), normalized)

                self._preprocessed[batch_idx].copy_(flipped)

            # Stage 3: Compute (wait for preprocessing)
            compute_stream.wait_stream(preprocess_stream)

            with torch.cuda.stream(compute_stream):
                output = self._model(self._preprocessed[batch_idx])

    def get_throughput_unit(self) -> str:
        return "images/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        total_images = iterations * self.batch_size * self.num_prefetch
        return total_images / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "num_prefetch": self.num_prefetch,
            "num_classes": self.num_classes,
            "stream_assignment": {
                "transfer": self._transfer_streams,
                "preprocess": self._preprocess_streams,
                "compute": self._compute_streams,
            },
        }

    def cleanup(self) -> None:
        """Cleanup model and buffers."""
        super().cleanup()
        self._model = None
        self._cpu_data = []
        self._gpu_buffers = []
        self._preprocessed = []
