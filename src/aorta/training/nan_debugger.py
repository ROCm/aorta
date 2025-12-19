"""
Runtime NaN Detection and Diagnosis Tool

This module provides hooks to detect and diagnose NaN/Inf issues during training.
Integrates with the training loop to catch issues when loss becomes NaN.

Strategy:
1. Primary detection: Check loss after forward pass
2. If loss is NaN → investigate gradients and parameters to find root cause
3. No preemptive gradient checking (avoids false positives)

Usage:
    from aorta.training.nan_debugger import NaNDebugger
    
    debugger = NaNDebugger(model, optimizer, config)
    
    # In training loop:
    loss = model(inputs)
    if debugger.check_loss(loss, step):  # Primary detection point
        # Loss is NaN, investigate what caused it
        debugger.check_gradients(step)
        debugger.check_parameters(step)
        # Stop training
        break
    
    loss.backward()
    optimizer.step()
"""

import torch
import torch.distributed as dist
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
import json
from datetime import datetime

log = logging.getLogger(__name__)


class NaNDebugger:
    """
    Real-time NaN/Inf detector and diagnostic tool.
    
    Detection Strategy:
    1. Primary detection: Check loss after forward pass
    2. Investigation: When loss is NaN, check gradients and parameters
    3. No preemptive checks (avoids false positives from race conditions)
    
    When NaN/Inf detected in loss, generates detailed diagnostic report
    identifying which gradients/parameters are affected.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[Dict] = None,
        output_dir: str = "nan_diagnostics",
        rank: int = 0,
        enabled: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.enabled = enabled
        
        # Track statistics
        self.step_history = []
        self.nan_detected = False
        self.first_nan_step = None
        
        # Configuration
        self.check_frequency = config.get("nan_check_interval", 1) if config else 1
        self.save_tensors = config.get("nan_save_tensors", False) if config else False
        
        log.info(f"[NaNDebugger] Initialized | rank={rank} enabled={enabled} check_freq={self.check_frequency}")
    
    def investigate_optimizer_failure(self, step: int, error_msg: str) -> None:
        """
        Investigate root cause when optimizer raises AssertionError.
        
        This checks both gradients and parameters to identify which ones contain NaN/Inf.
        Validates actual NaN/Inf counts to avoid false positives.
        """
        if not self.enabled:
            return
        
        log.error("[NaNDebugger] Starting investigation at step %d", step)
        
        # Check gradients for NaN/Inf (validate counts)
        nan_grads = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Count actual NaN/Inf
                num_nan = torch.isnan(param.grad).sum().item()
                num_inf = torch.isinf(param.grad).sum().item()
                
                # Only report if actually contains NaN/Inf
                if num_nan > 0 or num_inf > 0:
                    nan_grads.append({
                        "name": name,
                        "shape": list(param.grad.shape),
                        "num_nan": num_nan,
                        "num_inf": num_inf,
                        "grad_norm": param.grad.norm().item() if torch.isfinite(param.grad).any() else float('inf'),
                    })
        
        # Check parameters for NaN/Inf (validate counts)
        nan_params = []
        for name, param in self.model.named_parameters():
            # Count actual NaN/Inf
            num_nan = torch.isnan(param).sum().item()
            num_inf = torch.isinf(param).sum().item()
            
            # Only report if actually contains NaN/Inf
            if num_nan > 0 or num_inf > 0:
                nan_params.append({
                    "name": name,
                    "shape": list(param.shape),
                    "num_nan": num_nan,
                    "num_inf": num_inf,
                })
        
        # Only save report if this rank actually has NaN/Inf
        if not nan_grads and not nan_params:
            log.info(
                "[NaNDebugger] Optimizer failed on rank %d but no local NaN/Inf found (affected by distributed operation from another rank)",
                self.rank
            )
            return
        
        # Generate comprehensive report
        report = {
            "event": "optimizer_assertion",
            "step": step,
            "rank": self.rank,
            "timestamp": datetime.now().isoformat(),
            "optimizer_error": error_msg,
            "gradients_with_nan_inf": len(nan_grads),
            "parameters_with_nan_inf": len(nan_params),
            "affected_gradients": nan_grads[:10] if nan_grads else [],
            "affected_parameters": nan_params[:10] if nan_params else [],
            "all_affected_gradient_names": [g["name"] for g in nan_grads],
            "all_affected_parameter_names": [p["name"] for p in nan_params],
        }
        
        # Add diagnosis
        report["diagnosis"] = self._diagnose_optimizer_failure(nan_grads, nan_params, step, error_msg)
        
        # Save and print report
        self._save_report(report, f"optimizer_failure_step{step}_rank{self.rank}.json")
        self._print_optimizer_failure_report(report)
        
        if not self.nan_detected:
            self.nan_detected = True
            self.first_nan_step = step
    
    def check_loss(self, loss: torch.Tensor, step: int) -> bool:
        """Check if loss contains NaN/Inf."""
        if not self.enabled or step % self.check_frequency != 0:
            return False
        
        if not torch.isfinite(loss).all():
            self._handle_nan_loss(loss, step)
            return True
        return False
    
    def check_gradients(self, step: int) -> bool:
        """
        Check all gradients for NaN/Inf.
        
        NOTE: This is for investigation/diagnostics only, not for preemptive detection.
        Should be called AFTER detecting NaN in loss or optimizer failure.
        """
        if not self.enabled or step % self.check_frequency != 0:
            return False
        
        nan_grads = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    # Count actual NaN/Inf before reporting
                    num_nan = torch.isnan(param.grad).sum().item()
                    num_inf = torch.isinf(param.grad).sum().item()
                    
                    # Skip false positives: isfinite() can return False even when no NaN/Inf exist
                    # This is a known PyTorch/CUDA backend issue
                    if num_nan == 0 and num_inf == 0:
                        log.warning(
                            "[NaNDebugger] False positive: isfinite().all() returned False but num_nan=0 and num_inf=0 | "
                            "rank=%d step=%d param=%s shape=%s",
                            self.rank, step, name, list(param.grad.shape)
                        )
                        continue
                    
                    # Compute detailed gradient statistics
                    grad_stats = self._compute_gradient_stats(param.grad)
                    
                    nan_grads.append({
                        "name": name,
                        "shape": list(param.grad.shape),
                        "num_nan": num_nan,
                        "num_inf": num_inf,
                        "grad_norm": param.grad.norm().item() if torch.isfinite(param.grad).any() else float('inf'),
                        "grad_stats": grad_stats,
                    })
        
        if nan_grads:
            self._handle_nan_gradients(nan_grads, step)
            return True
        return False
    
    def _compute_gradient_stats(self, grad: torch.Tensor) -> Dict[str, Any]:
        """Compute detailed statistics for a gradient tensor."""
        try:
            # Get finite values only for statistics
            finite_mask = torch.isfinite(grad)
            finite_grads = grad[finite_mask]
            
            if finite_grads.numel() == 0:
                return {
                    "all_nan_or_inf": True,
                    "finite_count": 0,
                }
            
            stats = {
                "finite_count": finite_grads.numel(),
                "min": finite_grads.min().item(),
                "max": finite_grads.max().item(),
                "mean": finite_grads.mean().item(),
                "std": finite_grads.std().item() if finite_grads.numel() > 1 else 0.0,
                "abs_max": finite_grads.abs().max().item(),
            }
            
            # Compute percentiles for finite values
            if finite_grads.numel() > 100:
                q_levels = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], device=finite_grads.device)
                percentiles = torch.quantile(finite_grads.float(), q_levels)
                stats["percentiles"] = {
                    "p01": percentiles[0].item(),
                    "p25": percentiles[1].item(),
                    "p50": percentiles[2].item(),
                    "p75": percentiles[3].item(),
                    "p99": percentiles[4].item(),
                }
            
            return stats
        except Exception as e:
            log.warning("[NaNDebugger] Failed to compute gradient stats: %s", e)
            return {"error": str(e)}
    
    def check_parameters(self, step: int) -> bool:
        """
        Check all parameters for NaN/Inf.
        
        NOTE: This is for investigation/diagnostics only, not for preemptive detection.
        Should be called AFTER detecting NaN in loss or optimizer failure.
        """
        if not self.enabled or step % self.check_frequency != 0:
            return False
        
        nan_params = []
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                # Count actual NaN/Inf before reporting
                num_nan = torch.isnan(param).sum().item()
                num_inf = torch.isinf(param).sum().item()
                
                # Skip false positives: isfinite() can return False even when no NaN/Inf exist
                # This is a known PyTorch/CUDA backend issue
                if num_nan == 0 and num_inf == 0:
                    log.warning(
                        "[NaNDebugger] False positive: isfinite().all() returned False but num_nan=0 and num_inf=0 | "
                        "rank=%d step=%d param=%s shape=%s",
                        self.rank, step, name, list(param.shape)
                    )
                    continue
                
                nan_params.append({
                    "name": name,
                    "shape": list(param.shape),
                    "num_nan": num_nan,
                    "num_inf": num_inf,
                })
        
        if nan_params:
            self._handle_nan_parameters(nan_params, step)
            return True
        return False
    
    def _handle_nan_loss(self, loss: torch.Tensor, step: int):
        """Generate diagnostic report for NaN in loss."""
        if not self.nan_detected:
            self.nan_detected = True
            self.first_nan_step = step
        
        log.error(f"[NaNDebugger] NaN/Inf detected in LOSS at step {step}")
        
        report = {
            "event": "nan_in_loss",
            "step": step,
            "rank": self.rank,
            "timestamp": datetime.now().isoformat(),
            "loss_value": float(loss.item()) if loss.numel() == 1 else "multi-element",
            "is_nan": torch.isnan(loss).any().item(),
            "is_inf": torch.isinf(loss).any().item(),
        }
        
        # Diagnose likely causes
        report["diagnosis"] = self._diagnose_nan_loss(step)
        
        self._save_report(report, f"nan_loss_step{step}_rank{self.rank}.json")
        self._print_diagnosis(report)
    
    def _handle_nan_gradients(self, nan_grads: List[Dict], step: int):
        """Generate diagnostic report for NaN in gradients."""
        if not self.nan_detected:
            self.nan_detected = True
            self.first_nan_step = step
        
        log.error(f"[NaNDebugger] NaN/Inf detected in GRADIENTS at step {step}")
        log.error(f"[NaNDebugger] Affected parameters: {len(nan_grads)}")
        
        # Sort by gradient norm (largest first)
        nan_grads.sort(key=lambda x: x.get("grad_norm", 0), reverse=True)
        
        report = {
            "event": "nan_in_gradients",
            "step": step,
            "rank": self.rank,
            "timestamp": datetime.now().isoformat(),
            "num_affected_params": len(nan_grads),
            "affected_parameters": nan_grads[:10],  # Top 10
            "all_affected_names": [g["name"] for g in nan_grads],
        }
        
        # Diagnose likely causes
        report["diagnosis"] = self._diagnose_nan_gradients(nan_grads, step)
        
        self._save_report(report, f"nan_gradients_step{step}_rank{self.rank}.json")
        self._print_diagnosis(report)
    
    def _handle_nan_parameters(self, nan_params: List[Dict], step: int):
        """Generate diagnostic report for NaN in parameters."""
        if not self.nan_detected:
            self.nan_detected = True
            self.first_nan_step = step
        
        log.error(f"[NaNDebugger] NaN/Inf detected in PARAMETERS at step {step}")
        log.error(f"[NaNDebugger] Affected parameters: {len(nan_params)}")
        
        report = {
            "event": "nan_in_parameters",
            "step": step,
            "rank": self.rank,
            "timestamp": datetime.now().isoformat(),
            "num_affected_params": len(nan_params),
            "affected_parameters": nan_params[:10],  # Top 10
            "all_affected_names": [p["name"] for p in nan_params],
        }
        
        # Diagnose likely causes
        report["diagnosis"] = self._diagnose_nan_parameters(nan_params, step)
        
        self._save_report(report, f"nan_parameters_step{step}_rank{self.rank}.json")
        self._print_diagnosis(report)
    
    def _diagnose_optimizer_failure(self, nan_grads: List[Dict], nan_params: List[Dict], step: int, error_msg: str) -> Dict[str, Any]:
        """Diagnose optimizer failure due to NaN/Inf."""
        diagnosis = {
            "step_info": "first step" if step == 0 else f"step {step}",
            "detection_point": "optimizer step (AssertionError)",
            "error_message": error_msg,
        }
        
        # Identify which parameters are affected
        if nan_grads:
            diagnosis["first_affected_gradient"] = nan_grads[0]["name"]
            diagnosis["total_gradients_affected"] = len(nan_grads)
            
            # Check if embeddings affected
            embedding_affected = any("embed" in g["name"].lower() for g in nan_grads)
            if embedding_affected:
                diagnosis["embedding_gradients_affected"] = True
        
        if nan_params:
            diagnosis["first_affected_parameter"] = nan_params[0]["name"]
            diagnosis["total_parameters_affected"] = len(nan_params)
            
            # Check if embeddings affected
            embedding_affected = any("embed" in p["name"].lower() for p in nan_params)
            if embedding_affected:
                diagnosis["embedding_parameters_affected"] = True
        
        # Record optimizer configuration
        if self.config.get("optimizer", {}).get("name") == "shampoo":
            eps = self.config.get("optimizer", {}).get("eps", 1e-8)
            diagnosis["optimizer"] = "shampoo"
            diagnosis["optimizer_eps"] = eps
        
        # Record gradient statistics from affected parameters
        if nan_grads:
            max_grad_norm = max((g.get("grad_norm", 0) for g in nan_grads if g.get("grad_norm", 0) != float('inf')), default=0)
            if max_grad_norm > 0:
                diagnosis["max_affected_grad_norm"] = max_grad_norm
        
        return diagnosis
    
    def _diagnose_nan_loss(self, step: int) -> Dict[str, Any]:
        """Diagnose why loss became NaN."""
        diagnosis = {
            "step_info": "first step" if step == 0 else f"step {step}",
            "detection_point": "forward pass (loss computation)",
        }
        return diagnosis
    
    def _diagnose_nan_gradients(self, nan_grads: List[Dict], step: int) -> Dict[str, Any]:
        """Diagnose why gradients became NaN."""
        diagnosis = {
            "step_info": "first step" if step == 0 else f"step {step}",
            "detection_point": "backward pass (gradients)",
            "first_affected_param": nan_grads[0]["name"] if nan_grads else None,
        }
        
        # Factual information only
        embedding_affected = any("embed" in g["name"].lower() for g in nan_grads)
        if embedding_affected:
            diagnosis["embedding_affected"] = True
        
        # Record optimizer info if available
        if self.config.get("optimizer", {}).get("name") == "shampoo":
            eps = self.config.get("optimizer", {}).get("eps", 1e-8)
            diagnosis["optimizer"] = "shampoo"
            diagnosis["optimizer_eps"] = eps
        
        # Check gradient norm
        max_norm = max((g.get("grad_norm", 0) for g in nan_grads if g.get("grad_norm", 0) != float('inf')), default=0)
        if max_norm > 0:
            diagnosis["max_grad_norm"] = max_norm
        
        return diagnosis
    
    def _diagnose_nan_parameters(self, nan_params: List[Dict], step: int) -> Dict[str, Any]:
        """Diagnose why parameters became NaN."""
        diagnosis = {
            "step_info": "first step" if step == 0 else f"step {step}",
            "detection_point": "optimizer step (parameters)",
            "first_affected_param": nan_params[0]["name"] if nan_params else None,
        }
        
        # Record optimizer info if available
        optimizer_name = self.config.get("optimizer", {}).get("name", "unknown")
        diagnosis["optimizer"] = optimizer_name
        
        if optimizer_name == "shampoo":
            eps = self.config.get("optimizer", {}).get("eps", 1e-8)
            diagnosis["optimizer_eps"] = eps
        
        return diagnosis
    
    def _save_report(self, report: Dict, filename: str):
        """Save diagnostic report to file."""
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            log.info(f"[NaNDebugger] Report saved: {output_path}")
        except Exception as e:
            log.error(f"[NaNDebugger] Failed to save report: {e}")
    
    def _print_optimizer_failure_report(self, report: Dict):
        """Print optimizer failure investigation report."""
        log.error("=" * 70)
        log.error("OPTIMIZER NaN/Inf INVESTIGATION REPORT")
        log.error("=" * 70)
        log.error(f"Step: {report['step']}")
        log.error(f"Rank: {report['rank']}")
        log.error(f"Optimizer Error: {report['optimizer_error'][:200]}")
        
        log.error(f"\nGradients with NaN/Inf: {report['gradients_with_nan_inf']}")
        if report['affected_gradients']:
            log.error("Top affected gradients:")
            for grad in report['affected_gradients'][:5]:
                log.error(f"  - {grad['name']}: shape={grad['shape']}, "
                         f"NaN={grad['num_nan']}, Inf={grad['num_inf']}, "
                         f"norm={grad.get('grad_norm', 'N/A')}")
        
        log.error(f"\nParameters with NaN/Inf: {report['parameters_with_nan_inf']}")
        if report['affected_parameters']:
            log.error("Top affected parameters:")
            for param in report['affected_parameters'][:5]:
                log.error(f"  - {param['name']}: shape={param['shape']}, "
                         f"NaN={param['num_nan']}, Inf={param['num_inf']}")
        
        diagnosis = report.get("diagnosis", {})
        if diagnosis:
            log.error("\nDiagnostic Info:")
            for key, value in diagnosis.items():
                log.error(f"  {key}: {value}")
        
        log.error("=" * 70)
        log.error(f"Full report: {self.output_dir}/optimizer_failure_step{report['step']}_rank{report['rank']}.json")
        log.error("=" * 70)
    
    def _print_diagnosis(self, report: Dict):
        """Print human-readable diagnosis to log."""
        log.error("=" * 70)
        log.error("NaN DETECTION REPORT")
        log.error("=" * 70)
        log.error(f"Event: {report['event']}")
        log.error(f"Step: {report['step']}")
        log.error(f"Rank: {report['rank']}")
        
        if "affected_parameters" in report:
            log.error(f"\nAffected Parameters ({report['num_affected_params']} total):")
            for param in report["affected_parameters"][:5]:
                log.error(f"  - {param['name']}: shape={param['shape']}, "
                         f"NaN count={param.get('num_nan', 'N/A')}, "
                         f"Inf count={param.get('num_inf', 'N/A')}")
                
                # Print detailed gradient statistics if available
                if "grad_stats" in param:
                    stats = param["grad_stats"]
                    if "all_nan_or_inf" in stats:
                        log.error(f"    [All values are NaN/Inf]")
                    else:
                        log.error(f"    Finite gradients: count={stats.get('finite_count', 'N/A')}, "
                                 f"min={stats.get('min', 'N/A'):.6f}, max={stats.get('max', 'N/A'):.6f}, "
                                 f"mean={stats.get('mean', 'N/A'):.6f}, std={stats.get('std', 'N/A'):.6f}")
                        if "percentiles" in stats:
                            p = stats["percentiles"]
                            log.error(f"    Percentiles: p01={p['p01']:.6f}, p50={p['p50']:.6f}, p99={p['p99']:.6f}")
        
        diagnosis = report.get("diagnosis", {})
        if diagnosis:
            log.error("\nDiagnostic Info:")
            for key, value in diagnosis.items():
                if key not in ["likely_causes", "recommendations"]:  # Skip old fields if they exist
                    log.error(f"  {key}: {value}")
        
        log.error("=" * 70)
        log.error(f"Full report saved to: {self.output_dir}")
        log.error("=" * 70)
    
    def export_profiler_trace(
        self,
        torch_profiler,
        profiler_cfg,
        profiler_dir,
        trace_name: str,
    ) -> bool:
        """
        Export PyTorch profiler trace when NaN is detected.
        
        Args:
            torch_profiler: PyTorch profiler object (or None)
            profiler_cfg: Profiler configuration
            profiler_dir: Directory to save traces
            trace_name: Name of the trace file (e.g., "nan_loss_step5.json")
        
        Returns:
            True if trace was exported, False otherwise
        """
        if torch_profiler is None:
            log.warning("[Profiler] Skipping trace export (profiler disabled) | rank=%d", self.rank)
            return False
        
        if not profiler_cfg.chrome_trace:
            log.warning("[Profiler] Skipping trace export (chrome_trace disabled) | rank=%d", self.rank)
            return False
        
        # Check if profiler is actually active
        if getattr(torch_profiler, "_profiler", None) is None:
            log.warning("[Profiler] Skipping trace export (profiler inactive) | rank=%d", self.rank)
            return False
        
        try:
            from pathlib import Path
            trace_file = Path(profiler_dir) / f"rank{self.rank}" / trace_name
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            log.info("[Profiler] Exporting trace | rank=%d file=%s", self.rank, trace_file)
            torch_profiler.export_chrome_trace(str(trace_file))
            log.info("[Profiler] Trace export completed | rank=%d", self.rank)
            return True
        except Exception as export_err:
            log.warning("[Profiler] Failed to export trace: %s | rank=%d", export_err, self.rank)
            return False
    
    def signal_nan_to_ranks(self, store, step: int) -> bool:
        """
        Signal NaN detection to other ranks via TCPStore.
        
        Args:
            store: TCPStore object (or None)
            step: Current training step
        
        Returns:
            True if signal was sent, False otherwise
        """
        if store is None:
            return False
        
        try:
            import json
            signal_data = json.dumps({"step": step, "rank": self.rank})
            store.set("nan_detected", signal_data)
            log.info("[Coordination] Sent NaN signal to all ranks | rank=%d step=%d", self.rank, step)
            return True
        except Exception as signal_err:
            log.warning("[Coordination] Failed to send NaN signal: %s | rank=%d", signal_err, self.rank)
            return False
    
    def track_parameter_evolution(self, step: int, param_name: str = "_fsdp_wrapped_module.embedding.weight") -> None:
        """
        Track how a specific parameter evolves over time.
        Useful for debugging when/how weights become NaN.
        
        Args:
            step: Current training step
            param_name: Name of parameter to track (default: embedding.weight)
        """
        if not self.enabled:
            return
        
        try:
            for name, param in self.model.named_parameters():
                if name == param_name:
                    # Check parameter values
                    param_has_nan = torch.isnan(param).any().item()
                    param_has_inf = torch.isinf(param).any().item()
                    
                    # Check gradient values if available
                    grad_has_nan = False
                    grad_has_inf = False
                    grad_stats = None
                    if param.grad is not None:
                        grad_has_nan = torch.isnan(param.grad).any().item()
                        grad_has_inf = torch.isinf(param.grad).any().item()
                        if torch.isfinite(param.grad).any():
                            grad_stats = self._compute_gradient_stats(param.grad)
                    
                    # Compute parameter statistics
                    param_stats = {}
                    if torch.isfinite(param).any():
                        finite_params = param[torch.isfinite(param)]
                        param_stats = {
                            "min": finite_params.min().item(),
                            "max": finite_params.max().item(),
                            "mean": finite_params.mean().item(),
                            "std": finite_params.std().item() if finite_params.numel() > 1 else 0.0,
                            "abs_max": finite_params.abs().max().item(),
                        }
                    
                    evolution_data = {
                        "step": step,
                        "param_name": name,
                        "param_shape": list(param.shape),
                        "param_has_nan": param_has_nan,
                        "param_has_inf": param_has_inf,
                        "param_stats": param_stats,
                        "grad_has_nan": grad_has_nan,
                        "grad_has_inf": grad_has_inf,
                        "grad_stats": grad_stats,
                    }
                    
                    # Save to file
                    evolution_file = self.output_dir / f"param_evolution_{name.replace('.', '_')}_rank{self.rank}.jsonl"
                    with open(evolution_file, 'a') as f:
                        f.write(json.dumps(evolution_data) + "\n")
                    
                    # Log if NaN detected
                    if param_has_nan or grad_has_nan:
                        log.error(
                            "[NaNDebugger] Parameter evolution: step=%d param=%s param_nan=%s grad_nan=%s",
                            step, name, param_has_nan, grad_has_nan
                        )
                    
                    break  # Only track one parameter
        except Exception as e:
            log.warning("[NaNDebugger] Failed to track parameter evolution: %s", e)
    
    def broadcast_nan_stop_signal(self, has_nan: bool, device: torch.device) -> bool:
        """
        Broadcast NaN detection across all ranks using all_reduce.
        
        Args:
            has_nan: Whether this rank detected NaN
            device: Device to create tensor on
        
        Returns:
            True if any rank detected NaN
        """
        if not dist.is_initialized():
            return has_nan
        
        try:
            # Use all_reduce to broadcast NaN detection (1=NaN detected, 0=no NaN)
            nan_tensor = torch.tensor(1 if has_nan else 0, dtype=torch.int32, device=device)
            dist.all_reduce(nan_tensor, op=dist.ReduceOp.MAX)
            any_rank_has_nan = nan_tensor.item() > 0
            
            if any_rank_has_nan and not has_nan:
                log.error("[Coordination] Another rank detected NaN - stopping this rank too | rank=%d", self.rank)
            
            return any_rank_has_nan
        except Exception as e:
            log.warning("[Coordination] Failed to broadcast NaN signal: %s | rank=%d", e, self.rank)
            return has_nan
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of NaN detection session."""
        return {
            "nan_detected": self.nan_detected,
            "first_nan_step": self.first_nan_step,
            "total_steps_monitored": len(self.step_history),
            "output_dir": str(self.output_dir),
        }


class NaNDebuggerHook:
    """
    Simplified hook-based interface for NaN debugging.
    
    Usage:
        hook = NaNDebuggerHook(model, optimizer, config)
        
        # Training loop
        for step, batch in enumerate(dataloader):
            with hook.monitor_step(step):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                hook.check_loss(loss)
                
                loss.backward()
                hook.check_gradients()
                
                optimizer.step()
                hook.check_parameters()
    """
    
    def __init__(self, model, optimizer, config=None, **kwargs):
        self.debugger = NaNDebugger(model, optimizer, config, **kwargs)
        self.current_step = None
    
    def monitor_step(self, step: int):
        """Context manager for monitoring a training step."""
        class StepMonitor:
            def __init__(self, hook, step):
                self.hook = hook
                self.step = step
            
            def __enter__(self):
                self.hook.current_step = self.step
                return self.hook
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.hook.current_step = None
                return False
        
        return StepMonitor(self, step)
    
    def check_loss(self, loss):
        if self.current_step is not None:
            return self.debugger.check_loss(loss, self.current_step)
        return False
    
    def check_gradients(self):
        if self.current_step is not None:
            return self.debugger.check_gradients(self.current_step)
        return False
    
    def check_parameters(self):
        if self.current_step is not None:
            return self.debugger.check_parameters(self.current_step)
        return False

