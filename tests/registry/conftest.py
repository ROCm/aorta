"""Shared fixtures for registry tests — fake entry-point discovery.

The `fake_eps` and `fake_env_eps` fixtures patch the `entry_points` lookup in
the mitigations and environments modules respectively, with a controlled list
of fake entry-points. Use them to simulate plugins being installed without
actually installing anything.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass
class _FakeDist:
    name: str


@dataclass
class _FakeEntryPoint:
    name: str
    payload: Any  # tests pass non-dict payloads (e.g. str) to exercise validation
    dist: _FakeDist

    def load(self):
        # An exception payload simulates an entry-point whose import blows up,
        # which the agents loader must log and skip rather than propagate.
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload


def _fake_eps_for(monkeypatch, module_path: str):
    """Build a factory that, when called, installs fake entry-points at module_path."""
    def _install(specs):
        eps = [
            _FakeEntryPoint(name=n, payload=p, dist=_FakeDist(name=d))
            for n, p, d in specs
        ]
        monkeypatch.setattr(module_path, lambda group: eps)

    return _install


@pytest.fixture
def fake_eps(monkeypatch):
    """Install fake mitigation entry-points: fake_eps([(name, payload, dist), ...])."""
    return _fake_eps_for(monkeypatch, "aorta.registry.mitigations.entry_points")


@pytest.fixture
def fake_env_eps(monkeypatch):
    """Install fake environment entry-points: fake_env_eps([(name, payload, dist), ...])."""
    return _fake_eps_for(monkeypatch, "aorta.registry.environments.entry_points")


@pytest.fixture
def fake_agent_eps(monkeypatch):
    """Install fake agent entry-points: fake_agent_eps([(name, payload, dist), ...])."""
    return _fake_eps_for(monkeypatch, "aorta.registry.agents.entry_points")


@pytest.fixture
def tmp_sidecar(tmp_path):
    """Write a tmp JSON sidecar: tmp_sidecar({"version": 1, ...}, name="x.json") -> Path."""
    def _write(payload: dict, name: str = "sidecar.json") -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    return _write
