"""
Compute simulation for the RCCL race condition reproducer.

This module provides pluggable compute patterns (GEMM, attention, etc.)
that can be used to simulate forward/backward passes during testing.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import torch

if TYPE_CHECKING:
    from .config import ReproducerConfig


class BaseCompute(ABC):
    """
    Abstract base class for compute simulation.

    Subclasses must implement:
    - setup(): Allocate weights and buffers
    - forward(): Run forward pass (must use batch_gpu for data dependency)
    - backward(): Run backward pass

    The `parameters` property returns trainable tensors for the optimizer.
    """

    def __init__(self, config: "ReproducerConfig", dtype: torch.dtype):
        """
        Initialize compute simulator.

        Args:
            config: Reproducer configuration.
            dtype: Data type for tensors.
        """
        self.config = config
        self.dtype = dtype
        self._parameters: List[torch.Tensor] = []

    @abstractmethod
    def setup(self, requires_grad: bool = False) -> None:
        """
        Allocate weights and buffers.

        Args:
            requires_grad: Whether to enable gradients on parameters.
        """
        pass

    @abstractmethod
    def forward(self, batch_gpu: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Run forward pass.

        IMPORTANT: Must use batch_gpu to create a data dependency on the H2D
        transfer. This is critical for testing race conditions.

        Args:
            batch_gpu: GPU tensor from H2D transfer.

        Returns:
            Output tensor (for backward pass if using autograd).
        """
        pass

    @abstractmethod
    def backward(
        self,
        forward_output: Optional[torch.Tensor],
        use_autograd: bool = False
    ) -> None:
        """
        Run backward pass.

        Args:
            forward_output: Output from forward pass (for autograd backward).
            use_autograd: If True, use autograd; otherwise simulate manually.
        """
        pass

    @property
    def parameters(self) -> List[torch.Tensor]:
        """Return list of trainable parameters (for optimizer)."""
        return self._parameters


class GEMMCompute(BaseCompute):
    """
    GEMM-based compute simulation (stacked matrix multiplications).

    This simulates a simple neural network with multiple linear layers
    followed by GELU activations.

    Configuration:
        - gemm_size: Matrix size NxN (default: 5120)
        - gemm_layers: Number of GEMM layers (default: 26)
        - include_backward_compute: Whether to run backward (default: True)
    """

    def __init__(self, config: "ReproducerConfig", dtype: torch.dtype):
        super().__init__(config, dtype)
        self.weight_matrices: List[torch.Tensor] = []
        self.activation_buffer: Optional[torch.Tensor] = None
        self.grad_buffer: Optional[torch.Tensor] = None

    def setup(self, requires_grad: bool = False) -> None:
        """Allocate weight matrices and buffers."""
        cfg = self.config

        self.weight_matrices = [
            torch.randn(
                cfg.gemm_size, cfg.gemm_size,
                dtype=self.dtype, device="cuda", requires_grad=requires_grad
            )
            for _ in range(cfg.gemm_layers)
        ]
        self._parameters = self.weight_matrices

        self.activation_buffer = torch.randn(
            cfg.gemm_size, cfg.gemm_size, dtype=self.dtype, device="cuda"
        )
        self.grad_buffer = torch.randn(
            cfg.gemm_size, cfg.gemm_size, dtype=self.dtype, device="cuda"
        )

    def forward(self, batch_gpu: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Run forward pass with GEMMs.

        Creates a data dependency on batch_gpu by using it as the initial
        activation, simulating how a real model would consume input data.
        """
        cfg = self.config

        # CRITICAL: Create real data dependency on H2D transfer
        # Forward MUST read batch_gpu to create H2D→Forward race opportunity
        batch_slice = batch_gpu[:cfg.gemm_size * cfg.gemm_size]
        x = batch_slice.view(cfg.gemm_size, cfg.gemm_size)

        # Run through GEMM layers
        for weight in self.weight_matrices:
            x = torch.mm(weight, x)
            x = torch.nn.functional.gelu(x)

        # Store result to prevent optimization
        self.activation_buffer = x
        return x

    def backward(
        self,
        forward_output: Optional[torch.Tensor],
        use_autograd: bool = False
    ) -> None:
        """
        Run backward pass with GEMMs.

        If use_autograd is True and forward_output is provided, uses PyTorch
        autograd. Otherwise, simulates backward with manual GEMMs.
        """
        if not self.config.include_backward_compute:
            return

        if use_autograd and forward_output is not None:
            # Real backward pass with autograd
            loss = forward_output.sum()
            loss.backward()
        else:
            # Simulate backward with manual GEMMs (no autograd)
            grad = self.grad_buffer
            for weight in reversed(self.weight_matrices):
                grad = torch.mm(weight.T, grad)
            self.grad_buffer = grad


class TransformerCompute(BaseCompute):
    """
    Transformer-based compute simulation (self-attention + FFN with GEMMs).

    Each layer consists of:
    - Multi-head self-attention (QKV projections + attention + output projection)
    - Feed-forward network (two linear layers with GELU)
    - Layer normalization

    Configuration:
        - model_dim: Hidden dimension / d_model (default: 2048)
        - num_layers: Number of transformer layers (default: 4)
        - include_backward_compute: Whether to run backward (default: True)

    The number of attention heads is derived from model_dim (head_dim=128).
    """

    def __init__(self, config: "ReproducerConfig", dtype: torch.dtype):
        super().__init__(config, dtype)
        self.layers: Optional[torch.nn.ModuleList] = None
        self.model: Optional[torch.nn.Module] = None
        self.activation_buffer: Optional[torch.Tensor] = None
        self.grad_buffer: Optional[torch.Tensor] = None

    def setup(self, requires_grad: bool = False) -> None:
        """Allocate transformer layers."""
        cfg = self.config
        hidden = cfg.model_dim
        head_dim = 128
        num_heads = max(1, hidden // head_dim)
        ffn_dim = hidden * 4

        self.layers = torch.nn.ModuleList()
        for _ in range(cfg.num_layers):
            layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                dtype=self.dtype,
            )
            self.layers.append(layer)

        self.model = torch.nn.Sequential(*self.layers).cuda()

        if requires_grad:
            for p in self.model.parameters():
                p.requires_grad_(True)
        else:
            for p in self.model.parameters():
                p.requires_grad_(False)

        self._parameters = list(self.model.parameters())

        self.grad_buffer = torch.randn(
            hidden, hidden, dtype=self.dtype, device="cuda"
        )

    def forward(self, batch_gpu: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Run forward pass through transformer layers.

        Creates a data dependency on batch_gpu by reshaping it into a
        (batch, seq_len, hidden) input tensor.
        """
        cfg = self.config
        hidden = cfg.model_dim

        batch_slice = batch_gpu[:hidden * hidden]
        x = batch_slice.view(1, hidden, hidden)

        for layer in self.layers:
            x = layer(x)

        self.activation_buffer = x
        return x

    def backward(
        self,
        forward_output: Optional[torch.Tensor],
        use_autograd: bool = False
    ) -> None:
        """Run backward pass through transformer layers."""
        if not self.config.include_backward_compute:
            return

        if use_autograd and forward_output is not None:
            loss = forward_output.sum()
            loss.backward()
        else:
            grad = self.grad_buffer
            for _ in reversed(range(len(self.layers))):
                grad = torch.mm(grad, grad)
                grad = torch.nn.functional.gelu(grad)
            self.grad_buffer = grad


# =============================================================================
# Registry - allows adding custom compute types
# =============================================================================

COMPUTE_REGISTRY: Dict[str, Type[BaseCompute]] = {
    "gemm": GEMMCompute,
    "transformer": TransformerCompute,
}


def register_compute(name: str, cls: Type[BaseCompute]) -> None:
    """
    Register a custom compute type.

    Example:
        from aorta.race.compute import BaseCompute, register_compute

        class MyCustomCompute(BaseCompute):
            def setup(self, requires_grad=False): ...
            def forward(self, batch): ...
            def backward(self, output, use_autograd=False): ...

        register_compute("my_custom", MyCustomCompute)

    Args:
        name: Name to register the compute type under.
        cls: Compute class (must inherit from BaseCompute).
    """
    COMPUTE_REGISTRY[name] = cls


def create_compute(
    compute_type: str,
    config: "ReproducerConfig",
    dtype: torch.dtype
) -> BaseCompute:
    """
    Factory function to create a compute simulator.

    Args:
        compute_type: Type of compute (e.g., "gemm").
        config: Reproducer configuration.
        dtype: Data type for tensors.

    Returns:
        Compute simulator instance.

    Raises:
        ValueError: If compute_type is not registered.
    """
    if compute_type not in COMPUTE_REGISTRY:
        available = list(COMPUTE_REGISTRY.keys())
        raise ValueError(
            f"Unknown compute_type: {compute_type}. Available: {available}"
        )
    return COMPUTE_REGISTRY[compute_type](config, dtype)


__all__ = [
    "BaseCompute",
    "GEMMCompute",
    "TransformerCompute",
    "COMPUTE_REGISTRY",
    "register_compute",
    "create_compute",
]
