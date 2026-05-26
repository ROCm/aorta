"""Workloads registered under the `aorta.workloads` entry-point group.

Public workloads live as submodules here (e.g., aorta.workloads.fsdp).
Private workloads live in separate downstream packages and register
against the same entry-point group from their own pyproject.toml.

The base contract for all workloads is in `_base.py`.
"""

from aorta.workloads._base import Workload, WorkloadResult

__all__ = ["Workload", "WorkloadResult"]
