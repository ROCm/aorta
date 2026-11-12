"""`standin.parse`: the stand-in parse gate (design Section 6.2): the
candidate's source is the last ```cpp block of its final answer. No block is a gate failure with
reason `parse`. The source is written to the job's output directory as `source.hip`."""

from __future__ import annotations

import re
import time
from pathlib import Path

from ...atomic import atomic_write_bytes
from ...schemas import ControlSpec, LegJob, LegResult
from .base import make_result, unsettled

CPP_BLOCK = re.compile(r"```cpp[ \t]*\r?\n(.*?)```", re.DOTALL)


def last_cpp_block(text: str) -> str | None:
    blocks = CPP_BLOCK.findall(text)
    return blocks[-1] if blocks else None


class StandinParse:
    name = "standin.parse"
    version = "1"
    legs = ("gate.parse",)
    resource = "local_cpu"

    def __init__(self, clock=time.time):
        self._clock = clock

    def run(self, job: LegJob) -> LegResult:
        t0 = self._clock()
        try:
            response = job.inputs.get("response") or (str(Path(job.package) / "response.txt") if job.package else None)
            if job.leg != "gate.parse" or not response or not job.output_dir:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())
            text = Path(response).read_text(encoding="utf-8")
            source = last_cpp_block(text)
            if source is None:
                return make_result(job, "fail", "parse", t_start=t0, t_end=self._clock())
            out = Path(job.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(out / "source.hip", source.encode("utf-8"))
            return make_result(job, "clean", t_start=t0, t_end=self._clock())
        except Exception:
            return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())

    def controls(self) -> list[ControlSpec]:
        return []

    def health(self) -> dict:
        return {"name": self.name, "responsive": True}
