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
                    
                    nan_grads.append({
                        "name": name,
                        "shape": list(param.grad.shape),
                        "num_nan": num_nan,
                        "num_inf": num_inf,
                        "grad_norm": param.grad.norm().item() if torch.isfinite(param.grad).any() else float('inf'),
                    })
        
        if nan_grads:
            self._handle_nan_gradients(nan_grads, step)
            return True
        return False
    
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
        
        diagnosis = report.get("diagnosis", {})
        if diagnosis:
            log.error("\nDiagnostic Info:")
            for key, value in diagnosis.items():
                if key not in ["likely_causes", "recommendations"]:  # Skip old fields if they exist
                    log.error(f"  {key}: {value}")
        
        log.error("=" * 70)
        log.error(f"Full report saved to: {self.output_dir}")
        log.error("=" * 70)
    
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

