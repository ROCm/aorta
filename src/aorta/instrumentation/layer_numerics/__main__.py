"""Module entry point: ``python -m aorta.instrumentation.layer_numerics <target.py> [args...]``.

Runs the per-layer NaN/magnitude logger as a standalone front-end around a
training/repro script -- the same invocation shape used for standalone
handoff, but resolvable from an installed ``aorta`` package (no need to know
where the script file physically lives). Delegates to the verbatim
:data:`~aorta.instrumentation.layer_numerics.SCRIPT_PATH` via ``runpy`` so
the ``__main__`` behavior is byte-identical to running the script directly:

    # equivalent invocations
    python -m aorta.instrumentation.layer_numerics train.py
    python src/aorta/instrumentation/layer_numerics/instrument_nan_logger.py train.py

All tunables remain ``NANLOG_*`` environment variables (see the module
docstring in ``instrument_nan_logger.py`` or this package's README).
"""

from __future__ import annotations

import runpy
import sys

from aorta.instrumentation.layer_numerics import SCRIPT_PATH


def main() -> None:
    """Exec the logger script under ``__main__`` with the current argv.

    ``argv[0]`` is rewritten to the script path so its own
    ``len(sys.argv) < 2`` usage check and ``sys.argv = [target] + rest``
    slicing behave exactly as in the standalone form.
    """
    sys.argv = [str(SCRIPT_PATH), *sys.argv[1:]]
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
