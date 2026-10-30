"""Monarch actor for one tiny-transformer rank.

Imported only by :func:`aorta.workloads.distributed_launch.launch_monarch`.
The class is defined here, at module scope, so a worker unpickles
``aorta.workloads.monarch_trainer.Trainer``. A class built inside a function
and then assigned onto another module is not an attribute of that module
when a fresh interpreter imports it, and actor spawn fails before training.
"""

from __future__ import annotations

from monarch.actor import Actor, endpoint

from aorta.workloads.distributed_launch import run_training_rank


class Trainer(Actor):
    """One rank of the resolved training config."""

    @endpoint
    def train(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise TypeError(f"train config must be a dict, got {type(config).__name__}")
        return run_training_rank(config)


__all__ = ["Trainer"]
