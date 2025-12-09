"""
Workload registry for discovering and instantiating workloads.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from aorta.hw_queue_eval.workloads.base import BaseWorkload, WorkloadInfo


class WorkloadRegistry:
    """
    Registry for workload classes.

    Allows discovering workloads by name, category, or characteristics.
    """

    _workloads: Dict[str, Type[BaseWorkload]] = {}
    _instances: Dict[str, BaseWorkload] = {}

    @classmethod
    def register(cls, workload_class: Type[BaseWorkload]) -> Type[BaseWorkload]:
        """
        Register a workload class.

        Can be used as a decorator:
            @WorkloadRegistry.register
            class MyWorkload(BaseWorkload):
                ...

        Args:
            workload_class: Workload class to register

        Returns:
            The registered class (for decorator use)
        """
        name = workload_class.name
        if name in cls._workloads:
            # Allow re-registration (useful for reloading)
            pass
        cls._workloads[name] = workload_class
        return workload_class

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseWorkload:
        """
        Get a workload instance by name.

        Args:
            name: Workload name
            **kwargs: Arguments to pass to workload constructor

        Returns:
            Workload instance

        Raises:
            KeyError: If workload not found
        """
        if name not in cls._workloads:
            raise KeyError(
                f"Workload '{name}' not found. Available: {list(cls._workloads.keys())}"
            )

        return cls._workloads[name](**kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type[BaseWorkload]:
        """Get workload class by name."""
        if name not in cls._workloads:
            raise KeyError(
                f"Workload '{name}' not found. Available: {list(cls._workloads.keys())}"
            )
        return cls._workloads[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered workload names."""
        return list(cls._workloads.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """List workloads in a specific category."""
        return [
            name
            for name, workload_cls in cls._workloads.items()
            if workload_cls.category == category
        ]

    @classmethod
    def list_by_sensitivity(cls, sensitivity: str) -> List[str]:
        """List workloads by switch latency sensitivity."""
        return [
            name
            for name, workload_cls in cls._workloads.items()
            if workload_cls.switch_latency_sensitivity == sensitivity
        ]

    @classmethod
    def get_info(cls, name: str) -> WorkloadInfo:
        """Get workload info by name."""
        workload_cls = cls.get_class(name)
        return WorkloadInfo(
            name=workload_cls.name,
            description=workload_cls.description,
            category=workload_cls.category,
            min_streams=workload_cls.min_streams,
            max_streams=workload_cls.max_streams,
            recommended_streams=workload_cls.recommended_streams,
            switch_latency_sensitivity=workload_cls.switch_latency_sensitivity,
            memory_requirements_gb=workload_cls.memory_requirements_gb,
            multi_gpu_capable=workload_cls.multi_gpu_capable,
        )

    @classmethod
    def get_all_info(cls) -> Dict[str, WorkloadInfo]:
        """Get info for all registered workloads."""
        return {name: cls.get_info(name) for name in cls._workloads}

    @classmethod
    def clear(cls) -> None:
        """Clear all registered workloads."""
        cls._workloads.clear()
        cls._instances.clear()


def get_workload(name: str, **kwargs) -> BaseWorkload:
    """
    Convenience function to get a workload by name.

    Args:
        name: Workload name
        **kwargs: Arguments to pass to workload constructor

    Returns:
        Workload instance
    """
    return WorkloadRegistry.get(name, **kwargs)


def list_workloads(category: Optional[str] = None) -> List[str]:
    """
    List available workloads.

    Args:
        category: Optional category filter

    Returns:
        List of workload names
    """
    if category:
        return WorkloadRegistry.list_by_category(category)
    return WorkloadRegistry.list_all()


def register_all_workloads() -> None:
    """
    Register all built-in workloads.

    This imports all workload modules to trigger registration.
    """
    # Import workload modules to trigger @register decorators
    from aorta.hw_queue_eval.workloads import distributed, inference, latency_sensitive, pipeline


# Auto-register on module import
def _auto_register():
    """Automatically register workloads if available."""
    try:
        register_all_workloads()
    except ImportError:
        # Workloads may not be implemented yet
        pass


_auto_register()
