"""Cluster Intelligence Agents: Launch a run, Watch its log, Autopsy the bundle.

Usable without the chat extras -- nothing here imports :mod:`aorta.chat`.
"""

# DSPy lazily re-executes numpy, and that re-execution fails if numpy is only
# part-way through its own import. Pulling it in eagerly keeps any import order
# working. Guarded so the package still imports for inspection -- entry points,
# help text -- in an environment without the agent extra.
try:
    import numpy  # noqa: F401
except ImportError:  # pragma: no cover - only when the extra is absent
    pass
