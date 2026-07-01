"""``layer_numerics`` collector: per-layer NaN / magnitude logger.

A workload-agnostic instrumentation sidecar that auto-hooks a torchrec
``DistributedModelParallel`` root and (optionally) every
``torch.optim.Optimizer``, recording per-(layer, step, channel)
NaN/Inf/huge counts plus finite-magnitude extrema. It locates the first
layer/step to go bad and captures the run-up trajectory leading to it,
without editing the training script (it patches the class ``__init__`` and
attaches hooks the moment the real model is built).

This submodule is the platform half of the planned ``--collect layer_numerics``
collector. Dispatcher validation accepts the collector name today; follow-up
wiring will launch the script (:data:`SCRIPT_PATH`) as a ``runpy`` front-end
around a workload's entry script with the :func:`build_env` env bundle applied.

The script is a verbatim upstream drop (see ``README.md`` for provenance);
all tunables are ``NANLOG_*`` environment variables, so this package adds
no logic on top of it -- only a discoverable script path and the default
env bundle the collector applies.
"""

from __future__ import annotations

from pathlib import Path

#: Absolute path to the logger script. Future collector wiring launches the
#: workload entry as ``python <SCRIPT_PATH> <entry.py>`` so the hooks arm before
#: the model is built. Exposed as data (not a function) so downstream callers
#: can compute a docker bind-mount from ``SCRIPT_PATH.parent`` without importing
#: torch.
SCRIPT_PATH: Path = (Path(__file__).parent / "instrument_nan_logger.py").resolve()

#: Default ``NANLOG_*`` bundle the collector applies. Chosen to match the
#: capture that produced the residual-NaN bundle: all seven channels on,
#: 10-step pre-context run-up, sample clean layers 1-in-50. ``NANLOG_DIR``
#: is filled in per-call by :func:`build_env` so outputs land inside the
#: per-trial results tree (and are picked up by ``aorta bundle``).
_DEFAULTS: dict[str, str] = {
    "NANLOG_PRE_CONTEXT": "10",
    "NANLOG_SAMPLE_EVERY": "50",
    "NANLOG_CHANNELS": "act,input,igrad,weight,bias,wgrad,bgrad",
}

#: Subdirectory (under the trial results dir) the logger writes into.
OUTPUT_SUBDIR: str = "layer_numerics"


def build_env(
    results_dir: Path | str,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the ``NANLOG_*`` env bundle for one collected trial.

    Args:
        results_dir: The trial's results directory. The logger writes its
            ``summary_rank*.json`` + ``layers_rank*.jsonl`` under
            ``<results_dir>/layer_numerics`` (:data:`OUTPUT_SUBDIR`) so the
            existing ``aorta bundle`` staging (which copies every file under
            the run dir) picks them up with no extra wiring.
        overrides: Recipe- or operator-supplied ``NANLOG_*`` values that win
            over the defaults (e.g. ``{"NANLOG_WATCH_NAMES": "encoder.blocks"}``).
            ``NANLOG_DIR`` in ``overrides`` wins over the computed default,
            for the rare case an operator wants the output elsewhere.

    Returns:
        A flat ``dict[str, str]`` the caller merges into the subprocess
        environment. Contains only ``NANLOG_*`` keys; never mutates
        ``os.environ``.
    """
    env: dict[str, str] = dict(_DEFAULTS)
    env["NANLOG_DIR"] = str(Path(results_dir) / OUTPUT_SUBDIR)
    if overrides:
        invalid = sorted(key for key in overrides if not key.startswith("NANLOG_"))
        if invalid:
            raise ValueError(f"build_env overrides must be NANLOG_* keys: {invalid}")
        env.update(overrides)
    return env


__all__ = ["SCRIPT_PATH", "OUTPUT_SUBDIR", "build_env"]
