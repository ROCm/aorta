"""AORTA tools: analysis utilities discovered via the ``aorta.tools`` group.

A **tool** wraps a utility (emulator, profiler, validator, sanitizer) that
consumes structured inputs (kernel shapes, an exported graph, a saved code
object) and returns structured analysis output. A tool is:

  * **not a workload** -- it does not reproduce a customer training loop on
    real hardware and has no ``setup -> run -> cleanup`` trial lifecycle; and
  * **not a mitigation** -- it ships no environment-variable bundle.

Tools register one entry point each under the ``aorta.tools`` group (mirroring
the ``aorta.workloads`` / ``aorta.mitigations`` extension points). Public
tools live in ``aorta.tools.*``; private tools register the same way from a
downstream package's ``pyproject.toml``. ``load_tools()`` returns the merged
registry keyed by name, each entry tagged with its ``source_package`` so
collisions can name both sides.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from aorta.registry.errors import RegistryCollisionError, RegistryError

_GROUP = "aorta.tools"


@runtime_checkable
class Tool(Protocol):
    """The shape every ``aorta.tools`` plugin implements.

    Implementations expose a ``name`` (matching their entry-point key) and a
    single ``invoke()`` that runs the tool once over ``inputs`` and returns a
    result dict. ``inputs`` is tool-specific; ``invoke`` should return at least
    a ``report`` (structured result, or ``None`` when nothing was produced) and
    an ``overall_verdict`` when the tool is a guardrail. Other keys are
    tool-specific.
    """

    name: str

    def invoke(
        self, *, inputs: dict[str, Any], output_dir: Path | None = None
    ) -> dict[str, Any]:
        """Run the tool once over ``inputs`` and return a result dict."""
        ...


def load_tools() -> dict[str, type[Tool]]:
    """Discover and merge all tools registered under ``aorta.tools``.

    Returns a dict of ``name -> Tool class``. No caching -- re-reads
    entry-points each call (cheap; mirrors ``load_mitigations``).

    Raises:
        RegistryError: a plugin entry point did not resolve to a ``Tool``.
        RegistryCollisionError: two packages registered the same tool name.
    """
    registry: dict[str, type[Tool]] = {}
    source: dict[str, str] = {}
    for ep in entry_points(group=_GROUP):
        obj = ep.load()
        plugin_name = ep.dist.name if ep.dist is not None else "unknown"
        if not (isinstance(obj, type) and _looks_like_tool(obj)):
            raise RegistryError(
                f"plugin '{plugin_name}' tool '{ep.name}' must resolve to a Tool "
                f"class (with a 'name' attribute and an 'invoke' method); got "
                f"{type(obj).__name__}"
            )
        if ep.name in registry:
            raise RegistryCollisionError(
                f"tool '{ep.name}' registered by both '{source[ep.name]}' and "
                f"'{plugin_name}' -- rename one or remove the duplicate"
            )
        registry[ep.name] = obj
        source[ep.name] = plugin_name
    return registry


def get_tool(name: str) -> type[Tool]:
    """Return the ``Tool`` class registered under ``name``.

    Raises ``RegistryError`` if the name is unknown.
    """
    registry = load_tools()
    if name not in registry:
        raise RegistryError(
            f"unknown tool '{name}'; available: {sorted(registry)}; if you "
            f"expected a plugin-contributed tool, ensure the plugin is installed"
        )
    return registry[name]


def _looks_like_tool(obj: type) -> bool:
    return hasattr(obj, "name") and callable(getattr(obj, "invoke", None))


__all__ = ["Tool", "load_tools", "get_tool"]
