from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class BundleContext:
    root: Path
    manifest: dict[str, Any]
    job_id: str

    def path(self, key: str) -> Path | None:
        rel = self.manifest.get("paths", {}).get(key)
        if not rel:
            return None
        return self.root / rel


@dataclass
class AdapterArtifact:
    adapter: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    next_probes: list[dict[str, Any]] = field(default_factory=list)
    tooling_gaps: list[dict[str, Any]] = field(default_factory=list)


class ToolAdapter(Protocol):
    adapter_id: str

    def collect(self, ctx: BundleContext) -> AdapterArtifact: ...


def load_manifest(bundle_root: Path) -> dict[str, Any]:
    import yaml

    manifest_path = bundle_root / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.yaml in {bundle_root}")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest.yaml must parse to a mapping")
    return data
