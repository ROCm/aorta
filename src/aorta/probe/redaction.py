"""Recipe-driven redaction scrubbers for ``aorta bundle`` (issue #188 Phase 3).

Implements the three scrubbers documented in ``docs/probe-188/redaction.md``:

* env-key glob removal (``fnmatch.fnmatchcase``),
* absolute path rewriting to ``<PATH:N>``,
* IPv4/IPv6 rewriting to ``<IPV4:N>`` / ``<IPV6:N>``.

The :class:`RedactingRedactor` satisfies the :class:`aorta.bundle.redactor.Redactor`
ABC so ``aorta bundle`` can inject it via ``bundle_run_dir(redactor=...)``.
When a probe recipe omits the ``redaction:`` block, the bundle CLI falls
back to :class:`aorta.bundle.redactor.IdentityRedactor`.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aorta.bundle.redactor import RedactionCounts, Redactor
from aorta.probe.sandbox import MAX_LOG_BYTES
from aorta.triage.recipe import RecipeSchemaError

# Path scrubber: absolute POSIX paths with at least one directory component.
_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+"
)

# IPv4 candidate -- validated with :func:`ipaddress.ip_address` before rewrite.
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# IPv6 bracketed literal (URL/log form): [::1], [2001:db8::1]
_IPV6_BRACKETED_RE = re.compile(r"\[([0-9a-fA-F:.]+)\]")

# IPv6 unbracketed -- no leading \b (allows ::1 / ::); validated via ipaddress.
_IPV6_UNBRACKETED_RE = re.compile(
    r"(?<![0-9a-fA-F:.])"
    r"(?:"
    r"(?:[0-9a-fA-F]{0,4}:)+[0-9a-fA-F]{0,4}|"
    r"::(?:[0-9a-fA-F]{0,4}:){0,6}[0-9a-fA-F]{0,4}|"
    r"[0-9a-fA-F]{0,4}::(?:[0-9a-fA-F]{0,4}:){0,6}[0-9a-fA-F]{0,4}"
    r")"
    r"(?![0-9a-fA-F:.])"
)

_VALID_REDACTION_KEYS = frozenset({"scrub_env_keys", "scrub_paths", "scrub_ip_addresses"})

_TEXT_SUFFIXES = frozenset({".log", ".md", ".yaml", ".yml", ".json", ".txt", ".env"})


@dataclass(frozen=True)
class RedactionCfg:
    """Parsed ``redaction:`` block from a probe-mode recipe."""

    scrub_env_keys: tuple[str, ...] = ()
    scrub_paths: bool = False
    scrub_ip_addresses: bool = False


def parse_redaction(raw: Any) -> RedactionCfg:
    """Validate and parse a recipe ``redaction:`` mapping."""
    if raw is None:
        raise RecipeSchemaError("recipe.redaction: must be a mapping when present")
    if not isinstance(raw, dict):
        raise RecipeSchemaError(
            f"recipe.redaction: must be a mapping, got {type(raw).__name__}"
        )
    unknown = set(raw) - _VALID_REDACTION_KEYS
    if unknown:
        raise RecipeSchemaError(
            f"recipe.redaction: unknown keys {sorted(unknown)}; "
            f"allowed: {sorted(_VALID_REDACTION_KEYS)}"
        )
    keys_raw = raw.get("scrub_env_keys", [])
    if not isinstance(keys_raw, list) or not all(isinstance(x, str) for x in keys_raw):
        raise RecipeSchemaError(
            "recipe.redaction.scrub_env_keys: must be a list[str], "
            f"got {type(keys_raw).__name__}"
        )
    for flag_name in ("scrub_paths", "scrub_ip_addresses"):
        flag_val = raw.get(flag_name, False)
        if not isinstance(flag_val, bool):
            raise RecipeSchemaError(
                f"recipe.redaction.{flag_name}: must be a bool, got {type(flag_val).__name__}"
            )
    return RedactionCfg(
        scrub_env_keys=tuple(keys_raw),
        scrub_paths=bool(raw.get("scrub_paths", False)),
        scrub_ip_addresses=bool(raw.get("scrub_ip_addresses", False)),
    )


def _key_matches_glob(key: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(key, pattern) for pattern in globs)


def scrub_env_keys(
    env: dict[str, str],
    globs: tuple[str, ...],
) -> tuple[dict[str, str], int]:
    """Remove env keys matching any glob pattern (case-sensitive)."""
    if not globs:
        return dict(env), 0
    kept: dict[str, str] = {}
    removed = 0
    for key, value in env.items():
        if _key_matches_glob(key, globs):
            removed += 1
        else:
            kept[key] = value
    return kept, removed


class _PathIndex:
    """Per-file deduplication index for ``<PATH:N>`` placeholders."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}
        self._next = 0
        self.rewrites = 0

    def replace(self, match: re.Match[str]) -> str:
        path = match.group(0)
        if path not in self._seen:
            self._seen[path] = self._next
            self._next += 1
        self.rewrites += 1
        return f"<PATH:{self._seen[path]}>"


class _IpIndex:
    """Per-file deduplication index for ``<IPV4:N>`` / ``<IPV6:N>`` placeholders."""

    def __init__(self) -> None:
        self._v4_seen: dict[str, int] = {}
        self._v6_seen: dict[str, int] = {}
        self._v4_next = 0
        self._v6_next = 0
        self.ipv4_rewrites = 0
        self.ipv6_rewrites = 0

    def _replace(self, ip_str: str, *, v4: bool) -> str:
        if v4:
            if ip_str not in self._v4_seen:
                self._v4_seen[ip_str] = self._v4_next
                self._v4_next += 1
            self.ipv4_rewrites += 1
            return f"<IPV4:{self._v4_seen[ip_str]}>"
        if ip_str not in self._v6_seen:
            self._v6_seen[ip_str] = self._v6_next
            self._v6_next += 1
        self.ipv6_rewrites += 1
        return f"<IPV6:{self._v6_seen[ip_str]}>"


def _scrub_paths_in_text(text: str, path_index: _PathIndex) -> str:
    if not text:
        return text
    return _PATH_RE.sub(lambda m: path_index.replace(m), text)


def _rewrite_ipv6(candidate: str, ip_index: _IpIndex) -> str | None:
    """Return ``<IPV6:N>`` when ``candidate`` is a valid IPv6 address."""
    try:
        addr = ipaddress.ip_address(candidate)
    except ValueError:
        return None
    if addr.version != 6:
        return None
    return ip_index._replace(candidate, v4=False)


def _scrub_ips_in_text(text: str, ip_index: _IpIndex) -> str:
    if not text:
        return text

    def _bracketed_v6_sub(match: re.Match[str]) -> str:
        repl = _rewrite_ipv6(match.group(1), ip_index)
        return repl if repl is not None else match.group(0)

    text = _IPV6_BRACKETED_RE.sub(_bracketed_v6_sub, text)

    def _unbracketed_v6_sub(match: re.Match[str]) -> str:
        repl = _rewrite_ipv6(match.group(0), ip_index)
        return repl if repl is not None else match.group(0)

    text = _IPV6_UNBRACKETED_RE.sub(_unbracketed_v6_sub, text)

    def _v4_sub(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            addr = ipaddress.ip_address(candidate)
        except ValueError:
            return candidate
        if addr.version != 4:
            return candidate
        return ip_index._replace(candidate, v4=True)

    return _IPV4_RE.sub(_v4_sub, text)


def scrub_text(
    text: str,
    *,
    scrub_paths: bool,
    scrub_ip_addresses: bool,
) -> tuple[str, int, int, int]:
    """Apply path + IP scrubbers to a text blob (per-file index scope).

    Returns ``(text, paths_rewritten, ipv4_rewritten, ipv6_rewritten)``.
    Large inputs are processed in ``MAX_LOG_BYTES`` windows so a hostile
    log cannot blow regex CPU past the documented bound.
    """
    path_index = _PathIndex()
    ip_index = _IpIndex()
    if not scrub_paths and not scrub_ip_addresses:
        return text, 0, 0, 0

    if len(text) <= MAX_LOG_BYTES:
        windows = [text]
    else:
        windows = [text[i : i + MAX_LOG_BYTES] for i in range(0, len(text), MAX_LOG_BYTES)]

    out_parts: list[str] = []
    for window in windows:
        chunk = window
        if scrub_paths:
            chunk = _scrub_paths_in_text(chunk, path_index)
        if scrub_ip_addresses:
            chunk = _scrub_ips_in_text(chunk, ip_index)
        out_parts.append(chunk)

    return (
        "".join(out_parts),
        path_index.rewrites,
        ip_index.ipv4_rewrites,
        ip_index.ipv6_rewrites,
    )


def _scrub_json_value(
    value: Any,
    *,
    cfg: RedactionCfg,
    env_removed: list[int],
) -> tuple[Any, int, int, int]:
    if isinstance(value, dict):
        total_p = total_v4 = total_v6 = 0
        new_dict: dict[str, Any] = {}
        for key, item in value.items():
            if key == "env" and isinstance(item, dict):
                scrubbed_env, removed = scrub_env_keys(
                    {str(k): str(v) for k, v in item.items()},
                    cfg.scrub_env_keys,
                )
                env_removed[0] += removed
                new_dict[key] = scrubbed_env
                continue
            scrubbed, p, v4, v6 = _scrub_json_value(item, cfg=cfg, env_removed=env_removed)
            new_dict[key] = scrubbed
            total_p += p
            total_v4 += v4
            total_v6 += v6
        return new_dict, total_p, total_v4, total_v6
    if isinstance(value, list):
        total_p = total_v4 = total_v6 = 0
        new_list: list[Any] = []
        for item in value:
            scrubbed, p, v4, v6 = _scrub_json_value(item, cfg=cfg, env_removed=env_removed)
            new_list.append(scrubbed)
            total_p += p
            total_v4 += v4
            total_v6 += v6
        return new_list, total_p, total_v4, total_v6
    if isinstance(value, str):
        text, paths, v4, v6 = scrub_text(
            value,
            scrub_paths=cfg.scrub_paths,
            scrub_ip_addresses=cfg.scrub_ip_addresses,
        )
        return text, paths, v4, v6
    return value, 0, 0, 0


def _parse_probe_env(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key] = value
    return env


def _format_probe_env(env: dict[str, str]) -> str:
    return "\n".join(f"{k}={env[k]}" for k in sorted(env)) + ("\n" if env else "")


def _is_text_artifact(path: Path) -> bool:
    if path.name in {"probe.env", "stdout.log", "stderr.log", "matrix.md", "recipe.resolved.yaml"}:
        return True
    return path.suffix.lower() in _TEXT_SUFFIXES


class RedactingRedactor(Redactor):
    """Applies a probe recipe's ``redaction:`` block during bundling."""

    kind = "probe.v1"

    def __init__(self, cfg: RedactionCfg) -> None:
        self._cfg = cfg

    def scrub_file(self, src: Path, dst: Path) -> RedactionCounts:
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = src.read_bytes()
        bytes_in = len(raw)

        if src.name == "probe.env":
            counts = self._scrub_probe_env(raw, dst)
        elif src.name == "result.json":
            counts = self._scrub_result_json(raw, dst)
        elif src.name == "host_env.json":
            counts = self._scrub_host_env_json(raw, dst)
        elif _is_text_artifact(src):
            counts = self._scrub_text_bytes(raw, dst)
        else:
            dst.write_bytes(raw)
            counts = RedactionCounts(bytes_in=bytes_in, bytes_out=bytes_in)

        return counts

    def _scrub_probe_env(self, raw: bytes, dst: Path) -> RedactionCounts:
        text = raw.decode("utf-8", errors="replace")
        env = _parse_probe_env(text)
        scrubbed, removed = scrub_env_keys(env, self._cfg.scrub_env_keys)
        out = _format_probe_env(scrubbed)
        dst.write_text(out, encoding="utf-8")
        dst.chmod(stat.S_IRUSR | stat.S_IWUSR)
        out_bytes = out.encode("utf-8")
        return RedactionCounts(
            env_keys_removed=removed,
            bytes_in=len(raw),
            bytes_out=len(out_bytes),
        )

    def _scrub_result_json(self, raw: bytes, dst: Path) -> RedactionCounts:
        text = raw.decode("utf-8", errors="replace")
        doc = json.loads(text)
        env_removed = [0]
        scrubbed, paths, v4, v6 = _scrub_json_value(doc, cfg=self._cfg, env_removed=env_removed)
        out = json.dumps(scrubbed, indent=2, sort_keys=False) + "\n"
        dst.write_text(out, encoding="utf-8")
        out_bytes = out.encode("utf-8")
        return RedactionCounts(
            env_keys_removed=env_removed[0],
            paths_rewritten=paths,
            ips_rewritten=v4 + v6,
            bytes_in=len(raw),
            bytes_out=len(out_bytes),
        )

    def _scrub_host_env_json(self, raw: bytes, dst: Path) -> RedactionCounts:
        text = raw.decode("utf-8", errors="replace")
        doc = json.loads(text)
        env_removed = [0]
        if isinstance(doc, dict) and "env" in doc and isinstance(doc["env"], dict):
            scrubbed_env, removed = scrub_env_keys(
                {str(k): str(v) for k, v in doc["env"].items()},
                self._cfg.scrub_env_keys,
            )
            doc = {**doc, "env": scrubbed_env}
            env_removed[0] += removed
        out_text, paths, v4, v6 = scrub_text(
            json.dumps(doc, indent=2, sort_keys=False),
            scrub_paths=self._cfg.scrub_paths,
            scrub_ip_addresses=self._cfg.scrub_ip_addresses,
        )
        dst.write_text(out_text + "\n", encoding="utf-8")
        out_bytes = (out_text + "\n").encode("utf-8")
        return RedactionCounts(
            env_keys_removed=env_removed[0],
            paths_rewritten=paths,
            ips_rewritten=v4 + v6,
            bytes_in=len(raw),
            bytes_out=len(out_bytes),
        )

    def _scrub_text_bytes(self, raw: bytes, dst: Path) -> RedactionCounts:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            dst.write_bytes(raw)
            return RedactionCounts(bytes_in=len(raw), bytes_out=len(raw))
        out, paths, v4, v6 = scrub_text(
            text,
            scrub_paths=self._cfg.scrub_paths,
            scrub_ip_addresses=self._cfg.scrub_ip_addresses,
        )
        out_bytes = out.encode("utf-8")
        dst.write_bytes(out_bytes)
        return RedactionCounts(
            paths_rewritten=paths,
            ips_rewritten=v4 + v6,
            bytes_in=len(raw),
            bytes_out=len(out_bytes),
        )


__all__ = [
    "RedactionCfg",
    "RedactingRedactor",
    "parse_redaction",
    "scrub_env_keys",
    "scrub_text",
]
