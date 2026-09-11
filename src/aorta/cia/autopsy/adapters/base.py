from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


def resolve_in_bundle(root: Path, rel: str) -> Path | None:
    """*rel* as a path inside *root*, or None when it points outside.

    The single definition of what counts as inside a bundle, because a second
    copy of a containment check is how one of them ends up without the symlink
    case.

    Three ways out, and a manifest is not a trusted document -- it is written
    into the bundle by whatever produced the run. ``../../../etc/passwd`` walks
    out. An absolute path skips the root entirely, because ``Path(root) / "/etc"``
    discards the root and needs no traversal to look at. And a symlink inside
    the bundle can point anywhere, which is why both sides are resolved rather
    than compared as strings.
    """
    if not rel:
        return None
    try:
        resolved_root = root.resolve()
        candidate = (resolved_root / str(rel)).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    if candidate != resolved_root and not candidate.is_relative_to(resolved_root):
        return None
    return candidate


@dataclass(frozen=True)
class BundleContext:
    root: Path
    manifest: dict[str, Any]
    job_id: str

    def path(self, key: str) -> Path | None:
        """The manifest's path for *key*, if it is inside this bundle.

        Every adapter reads its evidence through here and sends what it finds
        to the router, so a manifest naming /etc/passwd or an escaping symlink
        would have been read and quoted into a verdict. One check here covers
        all of them.
        """
        return resolve_in_bundle(self.root, self.manifest.get("paths", {}).get(key))


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
