"""Tier 5 user ``custom_patterns`` runner for ``aorta probe`` (issue #188).

Consumes the ``custom_patterns:`` block on a probe-mode recipe and
fires detectors per-trial. Each pattern has:

* ``id`` -- stable detector ID (the value persisted into
  ``failure_detectors_fired`` / ``warn_detectors_fired``). The
  ID is namespaced under ``custom:`` so a reader of
  ``result.json`` can route by prefix.
* ``match.regex`` -- compile-validated at recipe load
  (:func:`validate_custom_patterns`). Named groups (``(?P<name>...)``)
  contribute to the trial's ``capture`` dict when the pattern
  matches.
* ``match.condition`` (optional) -- a sandboxed expression
  evaluated AFTER the regex matches. Must validate via
  :func:`aorta.probe.sandbox.validate_and_compile` at load time.
  When present, the detector fires only if BOTH the regex matched
  AND the condition evaluated to True.
* ``on_match`` -- ``"fail"`` (contributes to
  ``failure_detectors_fired``), ``"warn"`` (contributes to
  ``warn_detectors_fired``), or ``"info"`` (only populates
  ``capture``).
* ``required_for_pass`` (optional bool, default False) -- when
  True and the pattern does NOT fire, the verdict resolver
  injects ``meta:missing_pass_signal`` into
  ``failure_detectors_fired`` (rubric §2.B FR 2.8.c).

The runner is purely text-driven so it's testable without a real
subprocess: pass a log string and a list of ``CompiledPattern``s
and assert the returned :class:`CustomScanResult`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import CodeType
from typing import Literal

from aorta.probe.sandbox import MAX_LOG_BYTES, SandboxError, evaluate, validate_and_compile
from aorta.triage.recipe import RecipeSchemaError

OnMatch = Literal["fail", "warn", "info"]
_VALID_ON_MATCH: frozenset[str] = frozenset({"fail", "warn", "info"})

# Per-pattern key bank. ``required_for_pass`` is allowed only when
# ``on_match == "fail"`` (a warn/info pattern can't be required for
# pass — it doesn't change the verdict). Validated at load time.
_VALID_PATTERN_KEYS = frozenset({"id", "match", "on_match", "required_for_pass"})
_VALID_MATCH_KEYS = frozenset({"regex", "condition"})


@dataclass(frozen=True)
class CompiledPattern:
    """A custom_patterns entry, pre-compiled and pre-validated.

    Produced by :func:`validate_custom_patterns` from the parsed
    recipe dict. The recipe loader hands a tuple of these onto
    :attr:`aorta.probe.recipe_builder.ProbeExtras.custom_patterns`
    so the SubprocessWorkload doesn't re-compile per-trial.

    Attributes:
        detector_id: The ``custom:<id>`` string surfaced in
            ``failure_detectors_fired`` / ``warn_detectors_fired``.
        regex: Compiled :class:`re.Pattern`.
        condition_code: Pre-compiled sandbox CodeType, or None if
            no ``condition`` was supplied.
        condition_source: The raw condition string (useful for
            error messages and ``aorta probe --list-patterns``).
        on_match: ``"fail"`` / ``"warn"`` / ``"info"``.
        required_for_pass: When True and the pattern doesn't fire,
            the verdict resolver injects ``meta:missing_pass_signal``.
    """

    detector_id: str
    regex: re.Pattern[str]
    condition_code: CodeType | None
    condition_source: str | None
    on_match: OnMatch
    required_for_pass: bool = False


@dataclass
class CustomScanResult:
    """Outcome of scanning one trial's log against ``custom_patterns``.

    The verdict resolver assembles the final
    ``failure_detectors_fired`` / ``warn_detectors_fired`` /
    ``capture`` from this struct plus Tier 1–4 detector lists,
    so the per-tier outputs stay decoupled.

    ``fired_required_ids`` is the subset of fired detectors that
    came from a pattern with ``required_for_pass=True``; the
    resolver checks ``required_for_pass`` patterns that did NOT
    fire and synthesises ``meta:missing_pass_signal`` accordingly.
    """

    fail_detectors: list[str] = field(default_factory=list)
    warn_detectors: list[str] = field(default_factory=list)
    capture: dict[str, str | float | int] = field(default_factory=dict)
    fired_required_ids: set[str] = field(default_factory=set)


def validate_custom_patterns(raw: object) -> tuple[CompiledPattern, ...]:
    """Parse, compile, and sandbox-validate the ``custom_patterns:`` block.

    Returns an empty tuple when ``raw`` is None / missing so the
    caller can ``validate_custom_patterns(data.get("custom_patterns"))``
    without branching. Raises :class:`RecipeSchemaError` (or its
    :class:`SandboxError` subclass) on the first invalid entry —
    fail-fast on recipe load.

    Validation rules:

    * ``raw`` must be a non-empty list of mappings.
    * Each entry has a non-empty string ``id`` and a ``match``
      mapping with a compile-valid ``regex``.
    * ``match.condition`` (if present) sandbox-validates.
    * ``on_match`` ∈ {fail, warn, info}; default is ``fail`` when
      absent (matches the rubric's "any pattern that fires is a
      failure unless explicitly opted out").
    * ``required_for_pass`` ∈ {True, False}; only meaningful with
      ``on_match == "fail"``. Rejected with a clear error when
      paired with warn/info because the user almost certainly
      didn't mean it.
    * Unknown keys at either level reject with the allowed set.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise RecipeSchemaError(f"recipe.custom_patterns: must be a non-empty list, got {raw!r}")
    compiled: list[CompiledPattern] = []
    seen_ids: set[str] = set()
    for idx, entry in enumerate(raw):
        compiled.append(_validate_one(idx, entry, seen_ids))
    return tuple(compiled)


def _validate_one(
    idx: int,
    entry: object,
    seen_ids: set[str],
) -> CompiledPattern:
    """Validate one ``custom_patterns[idx]`` entry.

    Per-entry errors carry the ``custom_patterns[idx]`` path hint
    so the operator can find the offending entry without counting
    mapping keys. Schema-error first, then regex-error, then
    sandbox-error (the order recipe authors are most likely to hit
    in the wild).
    """
    path = f"recipe.custom_patterns[{idx}]"
    if not isinstance(entry, dict):
        raise RecipeSchemaError(f"{path}: must be a mapping, got {type(entry).__name__}")
    unknown = set(entry) - _VALID_PATTERN_KEYS
    if unknown:
        raise RecipeSchemaError(
            f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(_VALID_PATTERN_KEYS)}"
        )

    raw_id = entry.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        raise RecipeSchemaError(f"{path}.id: must be a non-empty string")
    if raw_id in seen_ids:
        raise RecipeSchemaError(
            f"{path}.id: duplicate id {raw_id!r} (custom-pattern ids must be unique within a recipe)"
        )
    seen_ids.add(raw_id)
    detector_id = f"custom:{raw_id}"

    match_raw = entry.get("match")
    if not isinstance(match_raw, dict):
        raise RecipeSchemaError(
            f"{path}.match: must be a mapping with 'regex' (and optional 'condition'), "
            f"got {type(match_raw).__name__}"
        )
    match_unknown = set(match_raw) - _VALID_MATCH_KEYS
    if match_unknown:
        raise RecipeSchemaError(
            f"{path}.match: unknown keys {sorted(match_unknown)}; "
            f"allowed: {sorted(_VALID_MATCH_KEYS)}"
        )
    regex_raw = match_raw.get("regex")
    if not isinstance(regex_raw, str) or not regex_raw:
        raise RecipeSchemaError(f"{path}.match.regex: must be a non-empty string")
    try:
        compiled_regex = re.compile(regex_raw)
    except re.error as exc:
        raise RecipeSchemaError(
            f"{path}.match.regex: invalid regex ({exc}): {regex_raw!r}"
        ) from exc

    condition_raw = match_raw.get("condition")
    condition_code: CodeType | None = None
    condition_source: str | None = None
    if condition_raw is not None:
        try:
            condition_code = validate_and_compile(condition_raw)
        except SandboxError as exc:
            # Wrap with the recipe path so a multi-entry recipe makes
            # the bad entry findable without counting list positions.
            # ``SandboxError`` subclasses ``RecipeSchemaError`` per the
            # rubric, so callers that catch the parent type still
            # catch the wrapped instance; the wrapped message keeps
            # the original sandbox detail (forbidden node, magnitude
            # cap, ...) intact at the end so no diagnostic is lost.
            # Per Copilot's PR #197 review.
            raise SandboxError(
                f"{path}.match.condition: {exc}"
            ) from exc
        condition_source = condition_raw

    on_match_raw = entry.get("on_match", "fail")
    if on_match_raw not in _VALID_ON_MATCH:
        raise RecipeSchemaError(
            f"{path}.on_match: must be one of {sorted(_VALID_ON_MATCH)}, got {on_match_raw!r}"
        )

    required = entry.get("required_for_pass", False)
    if not isinstance(required, bool):
        raise RecipeSchemaError(
            f"{path}.required_for_pass: must be a boolean, got {type(required).__name__}"
        )
    if required and on_match_raw != "fail":
        raise RecipeSchemaError(
            f"{path}.required_for_pass: only meaningful with on_match='fail' "
            f"(got on_match={on_match_raw!r}); a warn/info pattern cannot be required for pass"
        )

    return CompiledPattern(
        detector_id=detector_id,
        regex=compiled_regex,
        condition_code=condition_code,
        condition_source=condition_source,
        on_match=on_match_raw,
        required_for_pass=required,
    )


def scan(
    log_text: str,
    patterns: tuple[CompiledPattern, ...],
    *,
    exit_code: int,
    walltime_sec: float,
    peak_vram_mib: int | None,
) -> CustomScanResult:
    """Apply every ``CompiledPattern`` to ``log_text`` and return the result.

    Each pattern runs at most once. Named regex capture groups
    populate :attr:`CustomScanResult.capture` for every fired
    pattern; later-fired patterns whose capture group names
    collide overwrite earlier values (recipe authors should choose
    unique names).

    The ``condition`` sandbox is invoked only when the regex
    matches, and only with the four documented variables. A regex
    miss skips the sandbox entirely — that's the rubric's FR 2.12
    "no eval reach for rejected input" guarantee at runtime, in
    addition to the parse-time enforcement.

    The scanner enforces :data:`MAX_LOG_BYTES` via window-chunked
    scanning so a 100MiB log cannot blow up the runner.
    """
    result = CustomScanResult()
    if not patterns:
        return result
    if not log_text:
        log_text = ""

    # First-window-match wins per pattern; once a pattern has
    # fired, skip it on subsequent windows. This keeps the result
    # ordered by encounter (which is what the verdict resolver
    # expects for ``failure_detectors_fired`` order) and bounds
    # total work to O(patterns * windows).
    fired_ids: set[str] = set()
    for window in _iter_windows(log_text, MAX_LOG_BYTES):
        for pattern in patterns:
            if pattern.detector_id in fired_ids:
                continue
            match = pattern.regex.search(window)
            if match is None:
                continue
            named_capture = {k: v for k, v in match.groupdict().items() if v is not None}
            if pattern.condition_code is not None:
                try:
                    fired = evaluate(
                        pattern.condition_code,
                        capture=named_capture,
                        exit_code=exit_code,
                        walltime_sec=walltime_sec,
                        peak_vram_mib=peak_vram_mib,
                    )
                except Exception:
                    # A condition that raises at eval is treated as
                    # "did not fire" rather than aborting the
                    # classifier — the alternative (re-raising)
                    # would let a single misbehaving condition
                    # break the verdict for an entire run. The
                    # parse-time sandbox already rejects everything
                    # we can statically detect; runtime errors are
                    # operator misconfiguration we degrade past.
                    fired = False
            else:
                fired = True
            if not fired:
                continue
            fired_ids.add(pattern.detector_id)
            # capture is populated regardless of on_match: even
            # ``info`` patterns are useful precisely for surfacing
            # numbers into the trial's capture dict.
            result.capture.update(named_capture)
            if pattern.on_match == "fail":
                result.fail_detectors.append(pattern.detector_id)
                if pattern.required_for_pass:
                    result.fired_required_ids.add(pattern.detector_id)
            elif pattern.on_match == "warn":
                result.warn_detectors.append(pattern.detector_id)
            # ``info`` matches are silent w.r.t. detector lists —
            # capture-only.
        if len(fired_ids) == len(patterns):
            break
    return result


# Same overlap rationale as Tier 4 (see
# :data:`aorta.probe.classifier.tier4_patterns._WINDOW_OVERLAP_BYTES`).
# Operator-supplied ``custom_patterns`` may legitimately match a
# multi-line region (a stack frame, a JSON payload); 4 KiB of
# overlap covers every plausible single-match width without
# meaningfully inflating scan cost.
_WINDOW_OVERLAP_BYTES = 4096


def _iter_windows(text: str, window: int):
    """Window-chunked scanner. Same shape as Tier 4 with an overlap.

    Kept separate from the Tier 4 helper so a future change to the
    Tier 4 window strategy doesn't have to move in lock-step with
    the custom-pattern runner. The overlap closes the same
    straddling-match gap that Tier 4 had: a custom pattern whose
    match crosses a 10 MiB seam was silently missed by the
    non-overlapping shape. Overlap is capped at half the window so
    each step still advances by at least ``window // 2`` --
    mirrors the Tier 4 helper's loop-safety guard. Per Sonbol's PR
    #197 review (sweep from the Tier 4 fix to the parallel Tier 5
    helper).
    """
    if len(text) <= window:
        yield text
        return
    overlap = min(_WINDOW_OVERLAP_BYTES, max(window // 2, 0))
    start = 0
    while start < len(text):
        end = min(start + window, len(text))
        yield text[start:end]
        if end == len(text):
            return
        start = end - overlap


__all__ = [
    "CompiledPattern",
    "CustomScanResult",
    "OnMatch",
    "scan",
    "validate_custom_patterns",
]
