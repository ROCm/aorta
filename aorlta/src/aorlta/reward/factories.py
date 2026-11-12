"""Adapter factories for the shipped profiles: one adapter instance per worker, configured from the
profile's `adapters:` settings. `fake` adapters are built by the caller (they take a seed and knobs)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .adapters.base import SequentialBatch
from .adapters.standin_aorta import AortaAsan, AortaConsan
from .adapters.standin_hipcc import StandinHipcc
from .adapters.standin_parse import StandinParse
from .adapters.standin_rj_waitcheck import StandinRjWaitcheck
from .clock import Clock
from .profiles import Profile, tool_root


def standin_factories(profile: Profile, clock: Clock, *, timeouts: dict[str, float] | None = None,
                      check_s: dict[str, float] | None = None) -> dict[str, Callable[[str], Any]]:
    """Factories for every `standin.*` adapter the profile names. An adapter's own subprocess timeout
    sits a little under the watchdog's, so the adapter reports first when it can. `check_s` is
    `reward.gpu_check_s`: the GPU legs' per-check limit, by leg, inside that bound."""
    timeouts = timeouts or {}
    check_s = check_s or {}
    cfg = profile.adapters
    # the compile gate's compiler, as the profile resolved it when it was loaded: every adapter's default
    hipcc = cfg.get("standin.hipcc", {}).get("hipcc") or f"{tool_root('AORLTA_ROCM_ROOT')}/bin/hipcc"
    out: dict[str, Callable[[str], Any]] = {}
    names = {p.adapter for p in profile.pools.values()}
    pool_timeout = {p.adapter: timeouts.get(p.timeout, 120.0) for p in profile.pools.values()}

    def own(name: str) -> float:
        return max(1.0, 0.9 * pool_timeout.get(name, 120.0))

    if "standin.parse" in names:
        out["standin.parse"] = lambda wid: StandinParse(clock=clock.wall)
    if "standin.hipcc" in names:
        c = cfg.get("standin.hipcc", {})
        out["standin.hipcc"] = lambda wid, c=c: StandinHipcc(
            c.get("hipcc", hipcc), readelf=c.get("readelf", f"{tool_root('AORLTA_ROCM_ROOT')}/llvm/bin/llvm-readelf"),
            timeout_s=own("standin.hipcc"), clock=clock.wall)
    if "standin.rj_waitcheck" in names:
        c = cfg.get("standin.rj_waitcheck", {})
        out["standin.rj_waitcheck"] = lambda wid, c=c: StandinRjWaitcheck(
            c["binary"], aorta_src=c.get("aorta_src"), ld_library_path=c.get("ld_library_path"),
            timeout_s=own("standin.rj_waitcheck"), clock=clock.wall, hipcc=c.get("hipcc", hipcc))
    if "standin.aorta_consan" in names:
        c = cfg.get("standin.aorta_consan", {})
        out["standin.aorta_consan"] = lambda wid, c=c: AortaConsan(
            hook=c["hook"], aorta_src=c["aorta_src"], hipcc=c.get("hipcc", hipcc),
            target=c.get("target", "gfx950"), timeout_s=own("standin.aorta_consan"),
            check_timeout_s=check_s.get("san.consan"), control_source=c.get("control_source"), clock=clock.wall)
    if "standin.aorta_asan" in names:
        c = cfg.get("standin.aorta_asan", {})
        # aorta's ASan node-health canary builds with $HIPCC if it is set, else from aorta's own ROCm lookup, which
        # AORLTA_ROCM_ROOT does not reach. Name the candidates' compiler there now, before any worker is built, so the
        # canary and the candidates share one toolchain; a $HIPCC already set stands.
        os.environ.setdefault("HIPCC", c.get("hipcc", hipcc))
        out["standin.aorta_asan"] = lambda wid, c=c: SequentialBatch(AortaAsan(
            aorta_src=c["aorta_src"], hipcc=c.get("hipcc", hipcc), target=c.get("target", "gfx950"),
            timeout_s=max(1.0, own("standin.aorta_asan") / 18), check_timeout_s=check_s.get("san.asan"),
            control_source=c.get("control_source"), clock=clock.wall), max_packages=16, inband_controls=True)
    return out
