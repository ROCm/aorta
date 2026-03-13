"""
Scheduling and memory policy evaluation framework.

Evaluates different GPU scheduling and memory management policies using
aorta's existing workload suite and metrics infrastructure.  Policies are
applied via ``rocm-smi`` hardware controls, eBPF-level observation, and
environment variable tuning, then compared across the same workload
configuration.

Reference: gpu_ext paper (arXiv:2512.12615) demonstrates up to 4.8x
throughput improvement and 2x tail latency reduction with programmable
policies on NVIDIA GPUs.  This evaluator lets aorta benchmark analogous
policy configurations on AMD ROCm.

Policies evaluated:

Scheduling:
  - baseline          -- default round-robin, no constraints
  - priority_lc       -- latency-critical: high clock, high power
  - priority_be       -- best-effort: reduced clocks, capped power
  - multi_tenant_fair -- per-process GPU resource limits via cgroups/env

Memory:
  - default_uvm       -- system defaults (XNACK on supported GPUs)
  - prefetch_hints     -- hipMemPrefetchAsync before kernels
  - double_buffer      -- H2D double-buffering (already in race/ module)
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aorta.hw_queue_eval.core.harness import HarnessConfig, HarnessResult


@dataclass
class PolicyConfig:
    """Configuration for a single policy to evaluate."""

    name: str
    description: str = ""
    policy_type: str = "scheduling"  # "scheduling", "memory", "combined"

    # Scheduling knobs (applied via GPUControlConfig)
    gpu_clock_level: Optional[int] = None
    power_limit_watts: Optional[int] = None

    # Environment overrides applied during the run
    env_overrides: Dict[str, str] = field(default_factory=dict)

    # eBPF tracing flags
    ebpf_tracing: bool = False
    ebpf_memory_tracing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Built-in policy presets
# ---------------------------------------------------------------------------

BUILTIN_POLICIES: Dict[str, PolicyConfig] = {
    "baseline": PolicyConfig(
        name="baseline",
        description="Default round-robin scheduling, no hardware constraints",
        policy_type="scheduling",
    ),
    "priority_lc": PolicyConfig(
        name="priority_lc",
        description="Latency-critical: max clocks, max power budget",
        policy_type="scheduling",
        gpu_clock_level=7,  # highest on AMD (0-7)
    ),
    "priority_be": PolicyConfig(
        name="priority_be",
        description="Best-effort: reduced clocks and power cap",
        policy_type="scheduling",
        gpu_clock_level=2,
        power_limit_watts=150,
    ),
    "multi_tenant_fair": PolicyConfig(
        name="multi_tenant_fair",
        description="Fair sharing via GPU_MAX_HW_QUEUES=2",
        policy_type="scheduling",
        env_overrides={"GPU_MAX_HW_QUEUES": "2"},
    ),
    "high_queue": PolicyConfig(
        name="high_queue",
        description="Maximum HW queues (GPU_MAX_HW_QUEUES=8)",
        policy_type="scheduling",
        env_overrides={"GPU_MAX_HW_QUEUES": "8"},
    ),
    "default_uvm": PolicyConfig(
        name="default_uvm",
        description="Default UVM / XNACK behaviour",
        policy_type="memory",
        env_overrides={"HSA_XNACK": "1"},
        ebpf_memory_tracing=True,
    ),
    "xnack_off": PolicyConfig(
        name="xnack_off",
        description="XNACK disabled (no retryable page faults)",
        policy_type="memory",
        env_overrides={"HSA_XNACK": "0"},
        ebpf_memory_tracing=True,
    ),
}


@dataclass
class PolicyResult:
    """Result of running a workload under a single policy."""

    policy: PolicyConfig
    harness_result: "HarnessResult"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "result": self.harness_result.to_dict(),
        }


@dataclass
class PolicyComparison:
    """Comparison of multiple policy results for the same workload."""

    workload_name: str
    stream_count: int
    results: List[PolicyResult] = field(default_factory=list)
    timestamp: str = ""

    def add(self, result: PolicyResult) -> None:
        self.results.append(result)

    def best_throughput(self) -> Optional[PolicyResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.harness_result.throughput)

    def best_latency(self) -> Optional[PolicyResult]:
        if not self.results:
            return None
        return min(
            self.results, key=lambda r: r.harness_result.latency_ms.get("p99", float("inf"))
        )

    def summary_table(self) -> str:
        """Generate a text summary table comparing policies."""
        lines = [
            f"{'Policy':<22} {'Throughput':>14} {'P50 (ms)':>10} "
            f"{'P95 (ms)':>10} {'P99 (ms)':>10}",
            "-" * 70,
        ]
        for pr in self.results:
            r = pr.harness_result
            lines.append(
                f"{pr.policy.name:<22} {r.throughput:>14.2f} "
                f"{r.latency_ms.get('p50', 0):>10.3f} "
                f"{r.latency_ms.get('p95', 0):>10.3f} "
                f"{r.latency_ms.get('p99', 0):>10.3f}"
            )

        best_tp = self.best_throughput()
        best_lat = self.best_latency()
        if best_tp:
            lines.append("")
            lines.append(f"Best throughput: {best_tp.policy.name}")
        if best_lat:
            lines.append(f"Best P99 latency: {best_lat.policy.name}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload": self.workload_name,
            "stream_count": self.stream_count,
            "timestamp": self.timestamp,
            "results": [pr.to_dict() for pr in self.results],
        }

    def save(self, filepath: str | Path) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class PolicyEvaluator:
    """
    Run the same workload under different policy configurations and compare.

    Usage::

        evaluator = PolicyEvaluator(base_config, workload)
        policies = ["baseline", "priority_lc", "priority_be"]
        comparison = evaluator.evaluate(policies)
        print(comparison.summary_table())
    """

    def __init__(
        self,
        base_config: "HarnessConfig",
        workload: Any,
    ):
        """
        Args:
            base_config: Base HarnessConfig (policy-specific knobs are merged in)
            workload: A BaseWorkload instance to evaluate
        """
        self._base_config = base_config
        self._workload = workload

    def evaluate(
        self,
        policy_names: Optional[List[str]] = None,
        custom_policies: Optional[List[PolicyConfig]] = None,
    ) -> PolicyComparison:
        """
        Run the workload under each policy and return a comparison.

        Args:
            policy_names: Names of built-in policies to evaluate
            custom_policies: Additional custom PolicyConfig objects

        Returns:
            PolicyComparison with results from all policies
        """
        from aorta.hw_queue_eval.core.harness import StreamHarness
        from aorta.utils.gpu_control import GPUControlConfig

        policies: List[PolicyConfig] = []
        if policy_names:
            for name in policy_names:
                if name not in BUILTIN_POLICIES:
                    raise ValueError(
                        f"Unknown policy '{name}'. "
                        f"Available: {list(BUILTIN_POLICIES.keys())}"
                    )
                policies.append(BUILTIN_POLICIES[name])
        if custom_policies:
            policies.extend(custom_policies)

        if not policies:
            policies = [BUILTIN_POLICIES["baseline"]]

        comparison = PolicyComparison(
            workload_name=getattr(self._workload, "name", "unknown"),
            stream_count=self._base_config.stream_count,
            timestamp=datetime.now().isoformat(),
        )

        import os

        original_env = os.environ.copy()

        for policy in policies:
            # Build per-policy config
            config = copy.copy(self._base_config)

            gpu_ctl_enabled = (
                policy.gpu_clock_level is not None
                or policy.power_limit_watts is not None
            )
            if gpu_ctl_enabled:
                config.gpu_control = GPUControlConfig(
                    enabled=True,
                    gpu_clock_level=policy.gpu_clock_level,
                    power_limit_watts=policy.power_limit_watts,
                )

            config.ebpf_tracing = policy.ebpf_tracing
            config.ebpf_memory_tracing = policy.ebpf_memory_tracing

            # Apply env overrides
            for key, val in policy.env_overrides.items():
                os.environ[key] = val

            try:
                harness = StreamHarness(config)
                result = harness.run_workload(self._workload)
                comparison.add(PolicyResult(policy=policy, harness_result=result))
            finally:
                # Restore original environment
                for key in policy.env_overrides:
                    if key in original_env:
                        os.environ[key] = original_env[key]
                    else:
                        os.environ.pop(key, None)

        return comparison
