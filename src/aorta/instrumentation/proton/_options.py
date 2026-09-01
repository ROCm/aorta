"""Option schema for the ``proton`` collector.

Mirrors :mod:`aorta.instrumentation.rocprof._options`: every knob a recipe may
set under ``collect: {proton: {...}}`` is declared here and validated at
recipe-load time.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Attach modes. ``cli`` wraps argv with Proton's own command front-end;
#: ``env`` leaves the command alone and hands it ``AORTA_PROTON_*`` variables
#: for a workload that calls ``proton.start()`` / ``proton.finalize()`` itself
#: (scoped or intra-kernel measurement).
MODES: frozenset[str] = frozenset({"cli", "env"})

#: ``backend`` value meaning "let Proton pick". Not a Proton value: it is the
#: absence of Proton's ``-b`` flag / a ``backend=None`` argument.
AUTO_BACKEND: str = "auto"

#: Proton profiling backends. ``auto`` is aorta's spelling for "omit Proton's
#: ``-b`` and let it choose", which is the only value that is correct on every
#: Triton: ``rocprofiler`` is the preferred AMD path but was added after
#: Triton 3.7, whose CLI rejects the name at argparse before the payload runs;
#: ``roctracer`` is the deprecated AMD predecessor Proton still falls back to;
#: ``instrumentation`` is the intra-kernel path; ``cupti`` is the NVIDIA one,
#: accepted so a recipe stays portable even though aorta's examples are AMD.
#: Naming either *queue-intercepting* AMD backend (``rocprofiler`` /
#: ``roctracer``) also commits the recipe to ``mode: env``: the collector
#: refuses to pin those two under ``mode: cli``, where the capture would come
#: back empty (see :func:`aorta.instrumentation.proton.wrap_argv`).
#: ``instrumentation`` is an AMD backend too and is deliberately *not* covered
#: -- it installs no interceptor, so a CLI pin of it captures correctly.
BACKENDS: frozenset[str] = frozenset(
    {"auto", "cupti", "rocprofiler", "roctracer", "instrumentation"}
)

#: Backends that install an HSA queue interceptor and therefore cannot share a
#: process with ``rocprofv3``. Consumed by two guards outside this module --
#: the collector registry's ``rocprof`` conflict check and the collector's own
#: refusal to pin one of these under ``mode: cli`` -- which is why it lives in
#: the schema rather than in either. ``auto`` is included because on AMD it
#: resolves to ``rocprofiler`` or ``roctracer`` -- both intercepting -- and the
#: conflict guard has to reject a pairing it cannot prove is safe. The cli
#: guard excludes it explicitly instead, since omitting ``-b`` is the very
#: thing that keeps that path working.
QUEUE_INTERCEPTING_BACKENDS: frozenset[str] = frozenset({"auto", "rocprofiler", "roctracer"})

#: ``backend_mode`` values per backend: Proton's ``--mode`` is backend-specific,
#: and each backend accepts only its own names (the domains ``proton.start()``
#: documents). Keyed by backend so a mode can be validated against the backend
#: that will run it rather than against a flat union, where
#: ``roctracer`` + ``pcsampling`` would pass validation and then fail inside
#: Proton. ``instrumentation`` is deliberately absent: its modes are the
#: ``instrumentation_mode`` / ``granularity`` pair, which render the same
#: ``--mode`` argument with a different grammar.
BACKEND_MODES: dict[str, frozenset[str]] = {
    "cupti": frozenset({"pcsampling", "periodic_flushing"}),
    "rocprofiler": frozenset({"pcsampling", "periodic_flushing"}),
    "roctracer": frozenset({"periodic_flushing"}),
}

#: ``hook`` values (Proton's ``-k``). ``triton`` registers Proton's launch
#: hook, which records Triton kernel launch metadata alongside the timing.
#: Unlike the ``--mode`` knobs below, ``-k`` *is* forwarded by the shipped CLI,
#: so this option works in either attach mode.
HOOKS: frozenset[str] = frozenset({"triton"})

#: Options that render Proton's single ``--mode`` argument, and therefore
#: require ``mode: env``. Triton 3.7.1's front-end parses ``-m/--mode`` and then
#: calls ``start()`` without it, so a CLI wrap would render the flag and Proton
#: would never see the value -- a configured knob that silently does nothing,
#: which is worse than a rejected one. Upstream ``main`` forwards it, so this is a
#: version gate aorta cannot evaluate at recipe-load time and resolves the safe
#: way.
MODE_BEARING_KEYS: tuple[str, ...] = ("backend_mode", "granularity", "instrumentation_mode")

#: ``proton --context`` values.
CONTEXTS: frozenset[str] = frozenset({"shadow", "python"})

#: ``proton --data`` values. ``tree`` produces the ``.hatchet`` file the
#: summary parser reads; ``trace`` produces a chrome trace instead.
DATA_FORMATS: frozenset[str] = frozenset({"tree", "trace"})

#: ``instrumentation_mode`` values (Proton's ``--mode`` name component).
INSTRUMENTATION_MODES: frozenset[str] = frozenset({"default", "mma", "pcsampling"})

#: ``granularity`` values (Proton's ``--mode`` ``granularity=`` knob).
GRANULARITIES: frozenset[str] = frozenset(
    {
        "cta",
        "warp",
        "warp_2",
        "warp_4",
        "warp_8",
        "warp_group",
        "warp_group_2",
        "warp_group_4",
        "warp_group_8",
    }
)

#: Recipe-visible option keys, in the order they appear in the docs table.
OPTION_KEYS: tuple[str, ...] = (
    "mode",
    "backend",
    "backend_mode",
    "context",
    "data",
    "instrumentation_mode",
    "granularity",
    "hook",
)

_DEFAULTS: dict[str, str] = {
    "mode": "cli",
    "backend": "auto",
    "context": "shadow",
    "data": "tree",
}

_ENUMS: dict[str, frozenset[str]] = {
    "mode": MODES,
    "backend": BACKENDS,
    "context": CONTEXTS,
    "data": DATA_FORMATS,
    "instrumentation_mode": INSTRUMENTATION_MODES,
    "granularity": GRANULARITIES,
    "hook": HOOKS,
}


def validate_options(options: Mapping[str, str] | None) -> dict[str, str]:
    """Validate recipe-supplied ``proton`` options and apply defaults.

    Args:
        options: The raw ``collect: {proton: {...}}`` mapping, or ``None`` when
            the collector was enabled with no options.

    Returns:
        The effective options: the caller's values merged over the defaults
        (``mode=cli``, ``backend=auto``, ``context=shadow``, ``data=tree``),
        normalised to lowercase.

    Raises:
        ValueError: an unknown key, a value outside its declared domain, an
            intra-kernel knob (``instrumentation_mode`` / ``granularity``) set
            without ``backend: instrumentation`` -- Proton would ignore it, so
            accepting it would silently produce a profile the operator did not
            ask for -- or a ``backend_mode`` that its backend does not accept,
            that collides with an intra-kernel knob over Proton's single
            ``--mode``, or that was set without an explicit backend to
            validate it against. Any of :data:`MODE_BEARING_KEYS` outside
            ``mode: env`` is refused for the same reason the intra-kernel gate
            exists: the shipped CLI drops ``--mode``, so accepting it there
            would hand back a capture configured as if nothing had been asked.
    """
    raw = dict(options or {})
    unknown = sorted(set(raw) - set(OPTION_KEYS))
    if unknown:
        raise ValueError(f"proton: unknown option(s) {unknown}; valid: {list(OPTION_KEYS)}")
    for key, value in raw.items():
        if not isinstance(value, str):
            raise ValueError(f"proton option {key!r}: must be a string, got {type(value).__name__}")

    effective = dict(_DEFAULTS)
    effective.update({k: v.strip().lower() for k, v in raw.items()})

    for key, allowed in _ENUMS.items():
        value = effective.get(key)
        if value is not None and value not in allowed:
            raise ValueError(f"proton option {key!r}: {value!r} is not one of {sorted(allowed)}")

    intra_kernel = [k for k in ("instrumentation_mode", "granularity") if k in effective]
    # Before the per-backend gate below, because the two rejections have
    # different fixes and the collision is the one the operator can act on:
    # gating first would report the backend as the problem when the real
    # problem is asking for two mutually exclusive ``--mode`` values.
    if "backend_mode" in effective and intra_kernel:
        raise ValueError(
            f"proton option 'backend_mode' conflicts with {sorted(intra_kernel)}: "
            "both render into Proton's single '--mode' argument, so only one "
            "can be set. 'backend_mode' configures the whole-kernel backends "
            "(rocprofiler / roctracer / cupti); the intra-kernel pair "
            "configures backend: instrumentation."
        )
    if intra_kernel and effective["backend"] != "instrumentation":
        raise ValueError(
            f"proton option(s) {sorted(intra_kernel)} require "
            "backend: instrumentation (they configure Proton's intra-kernel "
            f"mode); got backend: {effective['backend']}"
        )
    if "backend_mode" in effective:
        allowed = BACKEND_MODES.get(effective["backend"])
        if allowed is None:
            raise ValueError(
                "proton option 'backend_mode' requires an explicit backend "
                f"from {sorted(BACKEND_MODES)} -- Proton's '--mode' domain is "
                "backend-specific, so there is nothing to validate it against "
                f"under backend: {effective['backend']}. Pin the backend "
                "(which needs mode: env for rocprofiler / roctracer), or drop "
                "the option."
            )
        if effective["backend_mode"] not in allowed:
            raise ValueError(
                f"proton option 'backend_mode': {effective['backend_mode']!r} is "
                f"not one of {sorted(allowed)} for backend: {effective['backend']}"
            )
    # Last, so a wrong value or a wrong backend is reported as itself rather
    # than as an attach-mode problem.
    mode_bearing = sorted(key for key in MODE_BEARING_KEYS if key in effective)
    if mode_bearing and effective["mode"] != "env":
        raise ValueError(
            f"proton option(s) {mode_bearing} require mode: env. They render "
            "Proton's '--mode', and no released Triton's command front-end "
            "forwards it: 3.7.1 parses '-m/--mode' and then calls start() "
            "without it, so under mode: cli the value is dropped and the "
            "capture comes back configured as if you had asked for nothing. "
            "Under mode: env the payload reads AORTA_PROTON_MODE and passes it "
            "to proton.start() itself, which is the only route that reaches "
            "Proton today (upstream main forwards it; this gate can relax when "
            "that ships)."
        )
    return effective


def mode_argument(options: Mapping[str, str]) -> str | None:
    """Render the Proton ``--mode`` value from whichever knob set it.

    ``backend_mode`` and the intra-kernel pair share Proton's single ``--mode``
    argument; :func:`validate_options` rejects them together, so reading
    ``backend_mode`` first is a precedence in spelling only.

    Returns ``None`` when no mode knob is set, so ``mode: env`` exports no
    ``AORTA_PROTON_MODE`` and Proton keeps its own default. Only the env bundle
    consumes this: every knob it reads requires ``mode: env``
    (:data:`MODE_BEARING_KEYS`), so the CLI wrap has no ``--mode`` to render.
    """
    backend_mode = options.get("backend_mode")
    if backend_mode is not None:
        return backend_mode
    name = options.get("instrumentation_mode")
    granularity = options.get("granularity")
    if name is None and granularity is None:
        return None
    rendered = name or "default"
    if granularity is not None:
        rendered = f"{rendered}:granularity={granularity}"
    return rendered


__all__ = [
    "AUTO_BACKEND",
    "BACKEND_MODES",
    "BACKENDS",
    "CONTEXTS",
    "DATA_FORMATS",
    "GRANULARITIES",
    "HOOKS",
    "INSTRUMENTATION_MODES",
    "MODE_BEARING_KEYS",
    "MODES",
    "OPTION_KEYS",
    "QUEUE_INTERCEPTING_BACKENDS",
    "mode_argument",
    "validate_options",
]
