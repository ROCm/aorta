"""Strict ConSan coverage parsing, adapted from RocJITsu's coverage gate."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PREFIX = "[rocjitsu-dbi-hooks] ConSan "
_SITE_KINDS = ("access", "barrier", "atomic", "fence")
_COUNT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_PAIR = re.compile(r"(0|[1-9][0-9]*)/(0|[1-9][0-9]*)\Z")
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

_COVERAGE_CATEGORIES = (
    "discovered",
    "supported",
    "selected",
    "patched",
    "unsupported",
    "resource_failed",
    "placement_or_lowering_failed",
    "expert_limit_omitted",
)
_VERDICT_COUNTS = (
    "applicable_code_objects",
    "incomplete_code_objects",
    "dynamic_incomplete",
    "replay_unsupported_access",
    "replay_unsupported_atomics",
    "replay_unsupported_fences",
    "replay_metadata_full",
)
_SITE_FIELDS = (
    "reader",
    "kind",
    "disposition",
    "reason",
    "outcome",
    "lowering_reason",
    "resource_reason",
    "container",
    "scope",
    "text",
    "mnemonic",
)
_NO_REASON = "none"


class CoverageParseError(ValueError):
    """Coverage output is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class CoverageRecord:
    reader: int
    load: int | None
    flavor: str
    engine: str
    analysis_complete: bool
    expert_limit: bool
    counts: tuple[tuple[str, int], ...]

    @property
    def count_map(self) -> dict[str, int]:
        return dict(self.counts)

    @property
    def applicable(self) -> bool:
        counts = self.count_map
        return any(counts[f"{kind}_discovered"] != 0 for kind in _SITE_KINDS)

    @property
    def identity(self) -> tuple[int, int | None]:
        return self.reader, self.load


@dataclass(frozen=True)
class AnalysisVerdict:
    applicable: bool
    analysis_complete: bool
    static_complete: bool
    dynamic_complete: bool
    counts: tuple[tuple[str, int], ...]
    patched_supported: tuple[tuple[str, int, int], ...]
    fields: tuple[tuple[str, str], ...]

    @property
    def count_map(self) -> dict[str, int]:
        return dict(self.counts)

    @property
    def pair_map(self) -> dict[str, tuple[int, int]]:
        return {kind: (patched, supported) for kind, patched, supported in self.patched_supported}


@dataclass(frozen=True)
class CoverageDecision:
    accepted: bool
    reasons: tuple[str, ...]
    coverage: tuple[CoverageRecord, ...]
    verdict: AnalysisVerdict


@dataclass(frozen=True)
class _SiteRecord:
    """One itemized ``coverage_site`` line."""

    reader: int
    load: int | None
    kind: str
    disposition: str
    outcome: str
    lowering_reason: str
    resource_reason: str

    @property
    def identity(self) -> tuple[int, int | None]:
        return self.reader, self.load

    @property
    def failure_reason(self) -> str | None:
        """Why this site failed, or None if it succeeded or gave no reason."""

        if self.outcome == "placement_or_lowering_failed":
            reason = self.lowering_reason
        elif self.outcome == "resource_failed":
            reason = self.resource_reason
        else:
            return None
        return None if reason == _NO_REASON else reason


def _fields(payload: str, context: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in payload.split():
        if "=" not in token:
            raise CoverageParseError(f"{context}: malformed field {token!r}")
        key, value = token.split("=", 1)
        if not _KEY.fullmatch(key) or not value or key in result:
            raise CoverageParseError(f"{context}: malformed field {token!r}")
        result[key] = value
    return result


def _require(fields: dict[str, str], names: tuple[str, ...], context: str) -> None:
    missing = [name for name in names if name not in fields]
    if missing:
        raise CoverageParseError(f"{context}: missing fields: {', '.join(missing)}")


def _count(fields: dict[str, str], name: str, context: str) -> int:
    value = fields[name]
    if not _COUNT.fullmatch(value):
        raise CoverageParseError(f"{context}: {name} is not an unsigned decimal count: {value!r}")
    return int(value)


def _boolean(fields: dict[str, str], name: str, context: str) -> bool:
    value = fields[name]
    if value not in {"true", "false"}:
        raise CoverageParseError(f"{context}: {name} must be true or false, got {value!r}")
    return value == "true"


def _pair(fields: dict[str, str], name: str, context: str) -> tuple[int, int]:
    match = _PAIR.fullmatch(fields[name])
    if match is None:
        raise CoverageParseError(f"{context}: {name} must be a patched/supported pair")
    return int(match.group(1)), int(match.group(2))


def _load(fields: dict[str, str], context: str) -> int | None:
    if "load" not in fields:
        return None
    value = _count(fields, "load", context)
    if value == 0:
        raise CoverageParseError(f"{context}: load must be nonzero")
    return value


def _parse_coverage(payload: str, line_number: int) -> CoverageRecord:
    context = f"coverage line {line_number}"
    fields = _fields(payload, context)
    count_names = tuple(
        f"{kind}_{category}" for kind in _SITE_KINDS for category in _COVERAGE_CATEGORIES
    )
    _require(
        fields,
        (
            "reader",
            "flavor",
            "engine",
            "analysis_complete",
            "expert_limit",
            *count_names,
        ),
        context,
    )
    if fields["flavor"] not in {"moi", "supercollider"}:
        raise CoverageParseError(f"{context}: unsupported flavor")
    if fields["engine"] not in {
        "record_replay",
        "inline_shadow",
        "sampled",
        "supercollider",
    }:
        raise CoverageParseError(f"{context}: unsupported engine")
    counts = {name: _count(fields, name, context) for name in count_names}
    complete = True
    for kind in _SITE_KINDS:
        discovered = counts[f"{kind}_discovered"]
        supported = counts[f"{kind}_supported"]
        selected = counts[f"{kind}_selected"]
        patched = counts[f"{kind}_patched"]
        unsupported = counts[f"{kind}_unsupported"]
        resource_failed = counts[f"{kind}_resource_failed"]
        placement_failed = counts[f"{kind}_placement_or_lowering_failed"]
        expert_omitted = counts[f"{kind}_expert_limit_omitted"]
        if discovered != supported + unsupported:
            raise CoverageParseError(f"{context}: {kind} discovered != supported + unsupported")
        if supported != selected + expert_omitted:
            raise CoverageParseError(f"{context}: {kind} supported != selected + expert omissions")
        if selected != patched + resource_failed + placement_failed:
            raise CoverageParseError(f"{context}: {kind} selected != patched + failures")
        complete &= unsupported == resource_failed == placement_failed == expert_omitted == 0
    analysis_complete = _boolean(fields, "analysis_complete", context)
    if analysis_complete != complete:
        raise CoverageParseError(f"{context}: analysis_complete contradicts counters")
    return CoverageRecord(
        reader=_count(fields, "reader", context),
        load=_load(fields, context),
        flavor=fields["flavor"],
        engine=fields["engine"],
        analysis_complete=analysis_complete,
        expert_limit=_boolean(fields, "expert_limit", context),
        counts=tuple(sorted(counts.items())),
    )


def _parse_verdict(payload: str, line_number: int) -> AnalysisVerdict:
    context = f"analysis verdict line {line_number}"
    fields = _fields(payload, context)
    _require(
        fields,
        (
            "applicable",
            "analysis_complete",
            "static_complete",
            "dynamic_complete",
            *_SITE_KINDS,
            *_VERDICT_COUNTS,
        ),
        context,
    )
    counts = {name: _count(fields, name, context) for name in _VERDICT_COUNTS}
    pairs = {kind: _pair(fields, kind, context) for kind in _SITE_KINDS}
    return AnalysisVerdict(
        applicable=_boolean(fields, "applicable", context),
        analysis_complete=_boolean(fields, "analysis_complete", context),
        static_complete=_boolean(fields, "static_complete", context),
        dynamic_complete=_boolean(fields, "dynamic_complete", context),
        counts=tuple(sorted(counts.items())),
        patched_supported=tuple((kind, *pairs[kind]) for kind in _SITE_KINDS),
        fields=tuple(sorted(fields.items())),
    )


def _aggregate(verdicts: list[AnalysisVerdict]) -> AnalysisVerdict:
    applicable = [verdict for verdict in verdicts if verdict.applicable]
    counts = {
        name: sum(verdict.count_map[name] for verdict in verdicts) for name in _VERDICT_COUNTS
    }
    pairs = {
        kind: (
            sum(verdict.pair_map[kind][0] for verdict in verdicts),
            sum(verdict.pair_map[kind][1] for verdict in verdicts),
        )
        for kind in _SITE_KINDS
    }
    return AnalysisVerdict(
        applicable=bool(applicable),
        analysis_complete=bool(applicable)
        and all(verdict.analysis_complete for verdict in applicable),
        static_complete=bool(applicable) and all(verdict.static_complete for verdict in applicable),
        dynamic_complete=all(verdict.dynamic_complete for verdict in verdicts),
        counts=tuple(sorted(counts.items())),
        patched_supported=tuple((kind, *pairs[kind]) for kind in _SITE_KINDS),
        fields=(),
    )


def _failure_attributions(sites: list[_SiteRecord]) -> list[str]:
    """Name why sites failed, so a report says more than "0 patched"."""

    tally: dict[tuple[str, str, str], int] = {}
    for site in sites:
        reason = site.failure_reason
        if reason is None:
            continue
        key = (site.kind, site.outcome, reason)
        tally[key] = tally.get(key, 0) + 1
    return [
        f"{kind} {outcome}: {count} {reason}"
        for (kind, outcome, reason), count in sorted(tally.items())
    ]


def parse_coverage_decision(log_text: str) -> CoverageDecision:
    """Parse and independently cross-check complete coverage evidence.

    Raises ``CoverageParseError`` when the evidence is malformed or internally
    inconsistent. A kind the hook counted but never itemized is neither: it is
    a coverage gap, so it is returned as a rejected decision naming the gap
    rather than raised. Both outcomes fail closed -- the difference is that a
    rejected decision still carries the per-object counts and any race finding,
    while a raised error discards them.
    """

    coverage: list[CoverageRecord] = []
    verdicts: list[AnalysisVerdict] = []
    site_records: list[_SiteRecord] = []
    for line_number, line in enumerate(log_text.splitlines(), 1):
        marker = line.find(_PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(_PREFIX) :]
        if payload.startswith("coverage "):
            coverage.append(_parse_coverage(payload[len("coverage ") :], line_number))
        elif payload.startswith("coverage_site "):
            context = f"coverage_site line {line_number}"
            fields = _fields(payload[len("coverage_site ") :], context)
            _require(fields, _SITE_FIELDS, context)
            if fields["kind"] not in _SITE_KINDS:
                raise CoverageParseError(f"{context}: unsupported kind")
            site_records.append(
                _SiteRecord(
                    reader=_count(fields, "reader", context),
                    load=_load(fields, context),
                    kind=fields["kind"],
                    disposition=fields["disposition"],
                    outcome=fields["outcome"],
                    lowering_reason=fields["lowering_reason"],
                    resource_reason=fields["resource_reason"],
                )
            )
        elif payload.startswith("analysis verdict "):
            verdicts.append(
                _parse_verdict(
                    payload[len("analysis verdict ") :],
                    line_number,
                )
            )

    if not coverage:
        raise CoverageParseError("missing ConSan coverage record")
    if not verdicts:
        raise CoverageParseError("missing ConSan analysis verdict")
    identities = [record.identity for record in coverage]
    if len(identities) != len(set(identities)):
        raise CoverageParseError("ambiguous duplicate coverage identities")
    identity_set = set(identities)
    if any(site.identity not in identity_set for site in site_records):
        raise CoverageParseError("coverage_site references an unknown load")
    unitemized: list[str] = []
    for record in coverage:
        counts = record.count_map
        for kind in _SITE_KINDS:
            retained = [
                site
                for site in site_records
                if site.identity == record.identity and site.kind == kind
            ]
            discovered = counts[f"{kind}_discovered"]
            if discovered and not retained:
                # The hook counts these sites but emits no per-site line for any
                # of them, so there is nothing to reconcile against. Coverage
                # this run cannot see is still refused below; naming the gap
                # keeps the rest of the evidence readable instead of aborting
                # the whole parse (ROCm/aorta#405).
                unitemized.append(
                    f"reader {record.reader} {kind} sites not itemized: 0 of {discovered}"
                )
                continue
            if len(retained) != discovered:
                raise CoverageParseError(f"reader {record.reader} {kind} site count mismatch")
            supported = sum(site.disposition == "supported" for site in retained)
            unsupported = sum(site.disposition == "unsupported" for site in retained)
            if supported != counts[f"{kind}_supported"]:
                raise CoverageParseError(f"reader {record.reader} {kind} supported count mismatch")
            if unsupported != counts[f"{kind}_unsupported"]:
                raise CoverageParseError(
                    f"reader {record.reader} {kind} unsupported count mismatch"
                )
            if not record.expert_limit:
                for outcome in ("patched", "resource_failed", "placement_or_lowering_failed"):
                    if (
                        sum(site.outcome == outcome for site in retained)
                        != counts[f"{kind}_{outcome}"]
                    ):
                        raise CoverageParseError(
                            f"reader {record.reader} {kind} {outcome} count mismatch"
                        )

    verdict = _aggregate(verdicts)
    applicable = [record for record in coverage if record.applicable]
    counts = verdict.count_map
    if len(applicable) != counts["applicable_code_objects"]:
        raise CoverageParseError("applicable code-object count disagrees with coverage")
    if (
        sum(not record.analysis_complete for record in applicable)
        != counts["incomplete_code_objects"]
    ):
        raise CoverageParseError("incomplete code-object count disagrees with coverage")
    for kind in _SITE_KINDS:
        aggregate = (
            sum(record.count_map[f"{kind}_patched"] for record in applicable),
            sum(record.count_map[f"{kind}_supported"] for record in applicable),
        )
        if aggregate != verdict.pair_map[kind]:
            raise CoverageParseError(f"{kind} aggregate disagrees with coverage")

    reasons: list[str] = []
    for name in (
        "applicable",
        "analysis_complete",
        "static_complete",
        "dynamic_complete",
    ):
        if not getattr(verdict, name):
            reasons.append(f"verdict {name}=false")
    if counts["applicable_code_objects"] == 0:
        reasons.append("no applicable code objects")
    for name in _VERDICT_COUNTS[1:]:
        if counts[name] != 0:
            reasons.append(f"{name}={counts[name]}")
    for kind, patched, supported in verdict.patched_supported:
        if patched != supported:
            reasons.append(f"{kind} patched/supported mismatch: {patched}/{supported}")
    reasons.extend(_failure_attributions(site_records))
    reasons.extend(unitemized)
    if any(record.expert_limit for record in coverage):
        reasons.append("expert patch limit enabled")
    return CoverageDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        coverage=tuple(coverage),
        verdict=verdict,
    )
