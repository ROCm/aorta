"""``rocjitsu_sanitizers`` -- waitcheck (static) + ConSan (dynamic) guardrail.

Runs the two rocjitsu sanitizers over the top kernels a workload launches and
folds the findings into one ``pass`` / ``warn`` / ``fail`` verdict. The tool
locates the rocjitsu artifacts (``rj_waitcheck`` + ``librocjitsu_dbi_hooks.so``)
at runtime via ``ROCJITSU_BUILD`` / ``RJ_WAITCHECK_BIN`` /
``ROCJITSU_SANITIZER_HOOK``; it does not build or vendor them. When the
artifacts are absent every check records ``skipped`` and the verdict is
``not_checked`` -- so the tool is safe to import/discover on any machine.

``invoke`` runs in-process (no subprocess wrapper); the engine lives in
``select.py`` / ``runner.py`` / ``backends.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import runner
from .select import select_kernels

# rocjitsu + ROCm env worth pinning next to a run for reproducibility.
_ENV_KEYS = ("ROCM_PATH", "HIP_VISIBLE_DEVICES", "ROCJITSU_BUILD",
             "ROCJITSU_SANITIZER_HOOK", "RJ_WAITCHECK_BIN", "HSA_TOOLS_LIB",
             "RJ_CONSAN_MODE", "RJ_CONSAN_POLICY")


def _as_path(value: Any) -> Path | None:
    return Path(value) if value is not None else None


def _as_command(value: Any) -> list[str] | None:
    if value is None:
        return None
    return value if isinstance(value, list) else str(value).split()


class RocjitsuSanitizersTool:
    """``aorta.tools`` plugin for the rocjitsu waitcheck + ConSan guardrail."""

    name = "rocjitsu_sanitizers"

    def invoke(self, *, inputs: dict[str, Any],
               output_dir: Path | None = None) -> dict[str, Any]:
        """Run the requested checks over a kernel worklist; return the report.

        Recognised ``inputs`` keys:
          * ``kernels`` -- path to a pre-built ``rocjitsu_sanitizers.kernels/1``
            worklist (any workload can emit one), OR
          * ``magpie_report`` / ``gemm_csv`` -- sources to build one, with
            ``top_n`` (default 20);
          * ``isa_dir`` -- directory of saved code objects for waitcheck;
          * ``target`` -- GPU target (default ``gfx950``);
          * ``checks`` -- list or comma string, subset of ``waitcheck,consan``;
          * ``consan_command`` -- app the ConSan hook wraps (str or list);
          * ``consan_mode`` / ``consan_policy`` / ``consan_log`` -- ConSan knobs;
          * ``simulator`` / ``simulator_config`` -- native-ISA simulator run;
          * ``dry_run`` / ``timeout``.
        """
        kernels_path = _as_path(inputs.get("kernels"))
        if kernels_path is not None:
            worklist = json.loads(kernels_path.read_text())
        else:
            worklist = select_kernels(
                magpie_report=_as_path(inputs.get("magpie_report")),
                gemm_csv=_as_path(inputs.get("gemm_csv")),
                top_n=int(inputs.get("top_n", 20)),
            )

        checks = inputs.get("checks", ["waitcheck", "consan"])
        if isinstance(checks, str):
            checks = [c.strip() for c in checks.split(",") if c.strip()]

        report = runner.run_sanitizers(
            worklist=worklist,
            target=str(inputs.get("target", "gfx950")),
            checks=checks,
            isa_dir=_as_path(inputs.get("isa_dir")),
            consan_command=_as_command(inputs.get("consan_command")),
            dry_run=bool(inputs.get("dry_run", False)),
            timeout=int(inputs.get("timeout", 1800)),
            base_env=dict(os.environ),
            consan_mode=inputs.get("consan_mode"),
            consan_policy=inputs.get("consan_policy"),
            consan_log=bool(inputs.get("consan_log", False)),
            simulator=inputs.get("simulator"),
            simulator_config=inputs.get("simulator_config"),
        )

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "sanitizer_report.json").write_text(
                json.dumps(report, indent=2) + "\n")

        return {
            "report": report,
            "overall_verdict": report["overall_verdict"],
            "env": {k: os.environ[k] for k in _ENV_KEYS if k in os.environ},
        }
