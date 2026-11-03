"""Archive-backed rocjitsu expert runtime."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

from aorta.rocjitsu import environment, rjdebug


@dataclass(frozen=True)
class ArchiveRunConfig:
    archive: Path
    scenario_id: str
    split: str = "test"
    reward: str = "v2"
    prompt_profile: str = "procedure"
    seed: int = rjdebug.EVAL_SEED


@dataclass(frozen=True)
class ArchiveOperation:
    step: int
    seed: int
    prompt: str
    reply: str
    kind: str
    tool: str
    kernel: str
    parse_error: str
    machine_result: Any
    rendered_result: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float


@dataclass(frozen=True)
class ArchiveRunResult:
    config: ArchiveRunConfig
    status: str
    answer: dict[str, Any] | None
    outcome: str
    score: float
    operations: tuple[ArchiveOperation, ...]
    system_prompt_sha256: str
    scenario_digest: str
    endpoint: str
    model: str
    server: dict[str, Any]
    seconds: float
    error: str | None = None


def _redact(value: Any, secret: str) -> Any:
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "<redacted>")
    if isinstance(value, dict):
        return {
            _redact(key, secret) if isinstance(key, str) else key: _redact(item, secret)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, secret) for item in value)
    return value


def _redacted_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.username is None:
        return endpoint
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def run_archive_episode(config: ArchiveRunConfig, client: rjdebug.Client) -> ArchiveRunResult:
    """Run one recorded V2 scenario through the trained multi-turn protocol."""
    archive = config.archive.resolve()
    try:
        if environment.corpus_layout(archive) != "v2":
            raise ValueError("the archive demo requires a corpus-v2 archive")
        scenarios = environment.load_corpus(
            archive,
            split=config.split,
            only=[config.scenario_id],
            reward=config.reward,
        )
        system = environment.system_prompt_v2(
            config.reward,
            profile=config.prompt_profile,
        )
    except SystemExit as exc:
        raise ValueError(str(exc)) from exc
    if len(scenarios) != 1:
        raise ValueError(f"expected one scenario, got {len(scenarios)}")
    scenario = scenarios[0]
    played = rjdebug.play_episode(
        scenario,
        0,
        client,
        system,
        seed=config.seed,
        environment=environment,
    )
    secret = client.api_key or ""
    operations = tuple(
        ArchiveOperation(
            step=step["step"],
            seed=step["seed"],
            prompt=_redact(step["prompt"], secret),
            reply=_redact(step["raw"], secret),
            kind=step["kind"],
            tool=step["tool"],
            kernel=step["kernel"],
            parse_error=_redact(step["error"], secret),
            machine_result=_redact(step["machine_result"], secret),
            rendered_result=_redact(step["rendered_result"], secret),
            prompt_tokens=step["prompt_tokens"],
            completion_tokens=step["completion_tokens"],
            latency_s=step["latency_s"],
        )
        for step in played.steps
    )
    return ArchiveRunResult(
        config=config,
        status="error" if played.error else "complete",
        answer=cast(dict[str, Any] | None, _redact(played.episode.answer, secret)),
        outcome=played.episode.outcome,
        score=played.episode.score().total,
        operations=operations,
        system_prompt_sha256=hashlib.sha256(system.encode()).hexdigest(),
        scenario_digest=scenario.digest,
        endpoint=cast(str, _redact(_redacted_endpoint(client.endpoint), secret)),
        model=client.model,
        server=cast(dict[str, Any], _redact(dict(client.server), secret)),
        seconds=played.seconds,
        error=cast(str | None, _redact(played.error, secret)),
    )


__all__ = [
    "ArchiveOperation",
    "ArchiveRunConfig",
    "ArchiveRunResult",
    "run_archive_episode",
]
