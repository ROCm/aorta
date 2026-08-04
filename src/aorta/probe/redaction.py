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
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aorta.bundle.errors import RedactionError
from aorta.bundle.redactor import RedactionCounts, Redactor
from aorta.probe.sandbox import MAX_LOG_BYTES
from aorta.triage.recipe import (
    RecipeSchemaError,
    dump_recipe_mapping,
    load_recipe_mapping,
)

# Path scrubber: absolute POSIX paths with at least one directory component.
# The negative lookbehind anchors the match at a path START so a sub-path of a
# larger filename token is not matched piecemeal -- but it deliberately EXCLUDES
# '/' so a leading '/' that itself follows another '/' still matches. Including
# '/' in the lookbehind would skip the path inside `file:///home/user/...` and
# `//host/home/user/...` (the leading slash precedes the real path), leaking
# exactly the absolute paths this scrubber documents it removes.
_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+"
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
_DISPATCHER_TRIAL_JSON_RE = re.compile(r"^trial_d\d+_m\d+_t\d+\.json$")
_PROBE_TRIAL_DIR_RE = re.compile(r"^trial_\d+$")


class _JsonKeyCollisionError(ValueError):
    pass


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
        # YAML permits non-string mapping keys (e.g. `1: x`); sorting a mixed
        # str/int set raises TypeError, which would escape as an unhandled
        # exception instead of a RecipeSchemaError. Sort by str repr so any
        # bad-key recipe fails closed with the schema error.
        raise RecipeSchemaError(
            f"recipe.redaction: unknown keys {sorted(map(str, unknown))}; "
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


# A UTF-8 code point is at most 4 bytes; if a string fits the byte cap even
# when every char is 4 bytes, no windowing is needed and we can skip the
# encode entirely (fast path for the many small JSON string values that
# scrub_text is called on).
_MAX_UTF8_BYTES_PER_CHAR = 4
# Keep a complete real POSIX path (PATH_MAX on Linux) or IP candidate on one
# side of an oversized-line split. Boundaries are moved back to the last
# character that cannot belong to either scrubber's token grammar.
_HARD_SPLIT_LOOKBACK_CHARS = 4096
_SCRUB_TOKEN_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-/:[]"
)


def _line_windows(text: str) -> list[str]:
    """Split ``text`` into ``<= MAX_LOG_BYTES`` (UTF-8 *byte*) windows, broken
    only at line boundaries.

    Two properties hold:

    * **Byte budget.** ``MAX_LOG_BYTES`` is a byte cap, so window size is
      measured in encoded UTF-8 bytes (``len(line.encode())``), not code
      points -- a multi-byte log must not slip past the regex-DoS bound just
      because it has fewer characters than bytes.
    * **No split tokens.** The naive ``text[i : i + N]`` slicing cuts a path
      or IP literal in half at the seam so neither regex pass matches it (a
      silent redaction miss). Paths/IPs never contain a line terminator, so
      breaking only *between* lines (``splitlines(keepends=True)``) keeps every
      token whole and ``"".join(...)`` reconstructs the input byte-for-byte.

    A single line whose own UTF-8 byte length exceeds the cap (e.g. a hostile
    newline-free log) would otherwise be emitted as one over-cap window and
    defeat the byte budget entirely, so it is hard-split into ``<= cap`` byte
    chunks. Each chunk holds at most ``MAX_LOG_BYTES // 4`` code points. Before
    splitting, the boundary moves back (up to Linux ``PATH_MAX``) to a
    delimiter outside both scrubbers' token grammars, so a path/IP at the seam
    is carried intact into the next chunk instead of leaking when the chunks
    are rejoined.
    """
    # max chars per window that is guaranteed <= MAX_LOG_BYTES UTF-8 bytes.
    max_chars = max(1, MAX_LOG_BYTES // _MAX_UTF8_BYTES_PER_CHAR)
    if len(text) <= max_chars:
        return [text]
    return list(_stream_line_windows(text.splitlines(keepends=True)))


def _stream_line_windows(lines: Iterable[str]) -> Iterator[str]:
    """Window an *iterable of lines* into ``<= MAX_LOG_BYTES`` byte chunks.

    Same budget/no-split-token semantics as :func:`_line_windows`, but driven
    off a line iterator so a caller streaming a large log off disk never
    materialises the whole file. ``_line_windows`` is the in-memory adapter
    (``str.splitlines(keepends=True)``); the streaming text path feeds a file
    handle's lines straight in.
    """
    max_chars = max(1, MAX_LOG_BYTES // _MAX_UTF8_BYTES_PER_CHAR)
    buf: list[str] = []
    size = 0
    for line in lines:
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > MAX_LOG_BYTES:
            if buf:
                yield "".join(buf)
                buf = []
                size = 0
            yield from _split_oversized_line(line, max_chars)
            continue
        if buf and size + line_bytes > MAX_LOG_BYTES:
            yield "".join(buf)
            buf = []
            size = 0
        buf.append(line)
        size += line_bytes
    if buf:
        yield "".join(buf)


def _split_oversized_line(line: str, max_chars: int) -> Iterator[str]:
    """Split one over-cap line without cutting a normal path/IP token."""
    start = 0
    while len(line) - start > max_chars:
        end = start + max_chars
        floor = max(start, end - _HARD_SPLIT_LOOKBACK_CHARS)
        split = end
        # Move the boundary to immediately after the nearest safe delimiter.
        # If none exists in PATH_MAX chars, keep the hard cap: a gigantic
        # delimiter-free token is itself outside the real path/IP grammars we
        # promise to recognize, and must not defeat the DoS bound.
        for index in range(end - 1, floor - 1, -1):
            if line[index] not in _SCRUB_TOKEN_CHARS:
                split = index + 1
                break
        if split <= start:  # defensive progress guard
            split = end
        yield line[start:split]
        start = split
    if start < len(line):
        yield line[start:]


def scrub_text(
    text: str,
    *,
    scrub_paths: bool,
    scrub_ip_addresses: bool,
) -> tuple[str, int, int, int]:
    """Apply path + IP scrubbers to a text blob (per-file index scope).

    Returns ``(text, paths_rewritten, ipv4_rewritten, ipv6_rewritten)``.
    Large inputs are processed in ``MAX_LOG_BYTES`` (UTF-8 byte) windows
    split at line boundaries by :func:`_line_windows`, so a hostile log
    cannot blow regex CPU past the documented bound while still scrubbing
    tokens that would otherwise fall on a fixed-slice seam.
    """
    path_index = _PathIndex()
    ip_index = _IpIndex()
    out = "".join(
        _scrub_windows_into(
            _line_windows(text),
            scrub_paths=scrub_paths,
            scrub_ip_addresses=scrub_ip_addresses,
            path_index=path_index,
            ip_index=ip_index,
        )
    )
    return (
        out,
        path_index.rewrites,
        ip_index.ipv4_rewrites,
        ip_index.ipv6_rewrites,
    )


def _scrub_windows_into(
    windows: Iterable[str],
    *,
    scrub_paths: bool,
    scrub_ip_addresses: bool,
    path_index: _PathIndex,
    ip_index: _IpIndex,
) -> Iterator[str]:
    """Scrub each window against *caller-owned* indices.

    Sharing one ``_PathIndex`` / ``_IpIndex`` across every window (and, for
    JSON, across every string leaf) keeps ``<PATH:N>`` / ``<IPV*:N>``
    placeholders consistent within a single file: the same path always maps to
    the same N and two distinct paths never collide on one N. Allocating a
    fresh index per window/leaf (the old per-leaf ``scrub_text`` call) broke
    that documented per-file scope.
    """
    if not scrub_paths and not scrub_ip_addresses:
        yield from windows
        return
    for window in windows:
        chunk = window
        if scrub_paths:
            chunk = _scrub_paths_in_text(chunk, path_index)
        if scrub_ip_addresses:
            chunk = _scrub_ips_in_text(chunk, ip_index)
        yield chunk


def _scrub_str_into(
    text: str,
    *,
    cfg: RedactionCfg,
    path_index: _PathIndex,
    ip_index: _IpIndex,
) -> str:
    return "".join(
        _scrub_windows_into(
            _line_windows(text),
            scrub_paths=cfg.scrub_paths,
            scrub_ip_addresses=cfg.scrub_ip_addresses,
            path_index=path_index,
            ip_index=ip_index,
        )
    )


def _collect_env_mapping_ids(doc: Any, document_kind: str) -> frozenset[int]:
    """Locate environment mappings from the artifact's actual schema.

    Key-name matching at arbitrary depth is unsafe: a collector may emit an
    unrelated ``env_vars`` metric. Restrict removal to known artifact fields,
    while path/IP rewriting still visits every string leaf.
    """

    found: set[int] = set()

    def add(value: Any) -> None:
        if isinstance(value, dict):
            found.add(id(value))

    def add_snapshot(value: Any) -> bool:
        if not isinstance(value, dict) or not isinstance(value.get("env_vars"), dict):
            return False
        add(value["env_vars"])
        for block in value.values():
            if isinstance(block, dict):
                add(block.get("env_overrides"))
        return True

    if not isinstance(doc, dict):
        return frozenset()

    if document_kind == "snapshot":
        if not add_snapshot(doc):
            # Legacy host_env.json used {"env": {NAME: value}}.
            add(doc.get("env"))
            descriptor = doc.get("descriptor")
            if isinstance(descriptor, dict):
                # Isolated-environment fallback env.json stores the unresolved
                # Environment descriptor rather than an EnvSnapshot.
                add(descriptor.get("env"))
    elif document_kind == "result":
        env = doc.get("env")
        if not add_snapshot(env):
            # Probe result.json uses a flat env mapping. Treat it as such even
            # when a malformed/legacy producer wrote a non-string value.
            add(env)
        execution_env = doc.get("execution_env")
        if isinstance(execution_env, dict):
            add(execution_env.get("env"))
        config = doc.get("config")
        if isinstance(config, dict):
            configured_environment = config.get("_aorta_environment")
            if isinstance(configured_environment, dict):
                add(configured_environment.get("env"))
            extras = config.get("_aorta_probe_extras")
            if isinstance(extras, dict):
                add(extras.get("cell_env_vars"))
    elif document_kind == "matrix":
        cells = doc.get("cells")
        if isinstance(cells, list):
            for cell in cells:
                if isinstance(cell, dict):
                    add(cell.get("resolved_env_vars"))
                    add(cell.get("extra_env"))
                    resolved_environment = cell.get("resolved_environment")
                    if isinstance(resolved_environment, dict):
                        add(resolved_environment.get("env"))
    elif document_kind == "sidecar":
        environments = doc.get("environments")
        if isinstance(environments, dict):
            for payload in environments.values():
                if isinstance(payload, dict):
                    add(payload.get("env"))
        mitigations = doc.get("mitigations")
        if isinstance(mitigations, dict):
            for payload in mitigations.values():
                add(payload)
    elif document_kind == "recipe":
        # recipe.resolved.yaml re-emits the same overlay that matrix.json
        # records as cells[*].extra_env, plus any inline-docker baseline env.
        add(doc.get("extra_env"))
        cells = doc.get("cells")
        if isinstance(cells, list):
            for cell in cells:
                if isinstance(cell, dict):
                    add(cell.get("extra_env"))
                    environment = cell.get("environment")
                    if isinstance(environment, dict):
                        add(environment.get("env"))
    return frozenset(found)


def _scrub_env_mapping(
    value: dict[Any, Any],
    *,
    cfg: RedactionCfg,
    env_removed: list[int],
    path_index: _PathIndex,
    ip_index: _IpIndex,
    env_mapping_ids: frozenset[int],
) -> dict[str, Any]:
    """Filter environment keys while preserving value types.

    ``env_vars`` contains ``str | null`` values. Converting the mapping to
    ``dict[str, str]`` would turn null into the literal ``"None"`` and would
    corrupt nested values, so use ``scrub_env_keys`` only to determine the kept
    key set, then recursively scrub the original values.
    """

    text_view = {str(key): item if isinstance(item, str) else "" for key, item in value.items()}
    kept, removed = scrub_env_keys(text_view, cfg.scrub_env_keys)
    env_removed[0] += removed
    scrubbed: dict[str, Any] = {}
    for key, item in value.items():
        source_key = str(key)
        if source_key not in kept:
            continue
        scrubbed_key = _scrub_str_into(
            source_key,
            cfg=cfg,
            path_index=path_index,
            ip_index=ip_index,
        )
        if scrubbed_key in scrubbed:
            raise _JsonKeyCollisionError("redaction produced duplicate JSON object keys")
        scrubbed[scrubbed_key] = _scrub_json_value(
            item,
            cfg=cfg,
            env_removed=env_removed,
            path_index=path_index,
            ip_index=ip_index,
            env_mapping_ids=env_mapping_ids,
        )
    return scrubbed


def _scrub_json_value(
    value: Any,
    *,
    cfg: RedactionCfg,
    env_removed: list[int],
    path_index: _PathIndex,
    ip_index: _IpIndex,
    env_mapping_ids: frozenset[int],
) -> Any:
    """Recursively scrub a JSON value, returning the scrubbed copy.

    Path/IP counts are NOT returned per node: ``path_index`` / ``ip_index``
    are shared across the whole document walk so placeholders stay file-
    consistent (Copilot review), and the caller reads the running totals off
    the indices once the walk completes.
    """
    if isinstance(value, dict):
        new_dict: dict[str, Any] = {}
        for key, item in value.items():
            scrubbed_key = _scrub_str_into(
                str(key),
                cfg=cfg,
                path_index=path_index,
                ip_index=ip_index,
            )
            if scrubbed_key in new_dict:
                raise _JsonKeyCollisionError("redaction produced duplicate JSON object keys")
            if isinstance(item, dict) and id(item) in env_mapping_ids:
                new_dict[scrubbed_key] = _scrub_env_mapping(
                    item,
                    cfg=cfg,
                    env_removed=env_removed,
                    path_index=path_index,
                    ip_index=ip_index,
                    env_mapping_ids=env_mapping_ids,
                )
                continue
            new_dict[scrubbed_key] = _scrub_json_value(
                item,
                cfg=cfg,
                env_removed=env_removed,
                path_index=path_index,
                ip_index=ip_index,
                env_mapping_ids=env_mapping_ids,
            )
        return new_dict
    if isinstance(value, list):
        return [
            _scrub_json_value(
                item,
                cfg=cfg,
                env_removed=env_removed,
                path_index=path_index,
                ip_index=ip_index,
                env_mapping_ids=env_mapping_ids,
            )
            for item in value
        ]
    if isinstance(value, str):
        return _scrub_str_into(value, cfg=cfg, path_index=path_index, ip_index=ip_index)
    return value


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


def _structured_document_kind(path: Path, *, run_root: Path | None = None) -> str | None:
    if run_root is None:
        # Direct/unit use has no bundle context. Recover the source root from
        # the tree and choose the outermost match so a collector cannot spoof a
        # nested root. Production bundling always supplies its exact run root;
        # inference is only a compatibility fallback.
        roots = [
            parent
            for parent in path.parents
            if (parent / "host_env.json").is_file()
            and (
                (parent / "recipe.resolved.yaml").is_file()
                or (parent / "matrix.json").is_file()
            )
        ]
        if not roots:
            return None
        if len(roots) != 1:
            raise RedactionError(
                path,
                ValueError(
                    "multiple probe run roots contain this artifact; "
                    "supply the exact run_root instead of inferring it"
                ),
            )
        run_root = roots[0]
    try:
        relative = path.absolute().relative_to(run_root.absolute())
    except ValueError:
        return None
    parts = relative.parts

    if (
        len(parts) == 3
        and parts[-1] == "result.json"
        and _PROBE_TRIAL_DIR_RE.fullmatch(parts[-2])
    ):
        return "result"
    dispatcher_layout = (
        len(parts) == 3
        or (len(parts) == 4 and parts[0] == "cells")
    )
    if dispatcher_layout and _DISPATCHER_TRIAL_JSON_RE.fullmatch(parts[-1]):
        return "result"
    if parts == ("host_env.json",):
        return "snapshot"
    if len(parts) == 3 and parts[0] == "environments" and parts[-1] == "env.json":
        return "snapshot"
    if parts == ("matrix.json",):
        return "matrix"
    if parts == ("recipe.resolved.yaml",):
        return "recipe"
    if parts == ("inline_environments.sidecar.json",) or (
        len(parts) == 2
        and parts[0] == "sidecars"
        and path.suffix.lower() == ".json"
    ):
        return "sidecar"
    return None


class RedactingRedactor(Redactor):
    """Applies a probe recipe's ``redaction:`` block during bundling."""

    kind = "probe.v1"

    def __init__(self, cfg: RedactionCfg, *, run_root: Path | None = None) -> None:
        self._cfg = cfg
        # ``bundle_run_dir`` canonicalises its source root before staging.
        # Canonicalise the matching context too: otherwise invoking bundle on
        # a symlink stores the alias here, receives real source paths later,
        # and classifies every platform JSON as generic text (so env-key
        # filtering silently never runs).
        self._run_root = run_root.resolve() if run_root is not None else None

    def scrub_file(self, src: Path, dst: Path) -> RedactionCounts:
        dst.parent.mkdir(parents=True, exist_ok=True)
        document_kind = _structured_document_kind(src, run_root=self._run_root)

        if src.name == "probe.env":
            counts = self._scrub_probe_env(src.read_bytes(), dst)
        elif document_kind == "recipe":
            counts = self._scrub_recipe_yaml(dst, src)
        elif document_kind is not None:
            counts = self._scrub_json_document(
                src.read_bytes(), dst, src, document_kind=document_kind
            )
        elif src.suffix.lower() == ".json":
            # Collector JSON is not schema-owned, so it gets no env-key
            # filtering. Parse it nonetheless: raw regex replacement misses
            # JSON escapes such as ``\u002fhome`` and can turn ``\/home`` into
            # the invalid escape ``\<PATH:0>``. A semantic JSON walk preserves
            # valid JSON while rewriting every string key/value.
            counts = self._scrub_json_document(src.read_bytes(), dst, src, document_kind=None)
        elif _is_text_artifact(src):
            counts = self._scrub_text_stream(src, dst)
        else:
            # Binary / non-scrubbable artifact (e.g. a multi-GB core dump):
            # stream the copy via shutil.copyfile rather than reading the whole
            # file into a bytes object and writing it back. The no-scrub branch
            # applies no transform, so byte counts come from stat() (in == out).
            shutil.copyfile(src, dst)
            size = dst.stat().st_size
            counts = RedactionCounts(bytes_in=size, bytes_out=size)

        # Carry the source's permission bits onto every staged copy. The
        # scrub branches all (re)create dst via write_text / write_bytes,
        # which land at the umask default (~0644) and would WIDEN a
        # restrictive source (e.g. probe.env at 0600) inside the shareable
        # bundle. Mirrors IdentityRedactor.scrub_file; never let the bundle
        # copy be less restrictive than the original (PR #199 review).
        shutil.copymode(src, dst)
        return counts

    def _scrub_probe_env(self, raw: bytes, dst: Path) -> RedactionCounts:
        text = raw.decode("utf-8", errors="replace")
        env = _parse_probe_env(text)
        kept, removed = scrub_env_keys(env, self._cfg.scrub_env_keys)
        # probe.env is a verbatim copy of the cell env mapping that
        # result.json and the dispatcher trial JSON also carry, so a retained
        # key's value must be path/IP scrubbed here too -- otherwise the same
        # LD_LIBRARY_PATH is a placeholder in one artifact and a customer path
        # in another. Names are validated env identifiers and cannot hold a
        # path or IP, so only values are rewritten.
        path_index = _PathIndex()
        ip_index = _IpIndex()
        scrubbed = {
            key: _scrub_str_into(
                value,
                cfg=self._cfg,
                path_index=path_index,
                ip_index=ip_index,
            )
            for key, value in kept.items()
        }
        out = _format_probe_env(scrubbed)
        dst.write_text(out, encoding="utf-8")
        # Mode is carried from the source by scrub_file's shutil.copymode;
        # the run-dir probe.env is written 0600 by the workload, so the
        # staged copy stays owner-only without a hardcoded chmod here.
        out_bytes = out.encode("utf-8")
        return RedactionCounts(
            env_keys_removed=removed,
            paths_rewritten=path_index.rewrites,
            ips_rewritten=ip_index.ipv4_rewrites + ip_index.ipv6_rewrites,
            bytes_in=len(raw),
            bytes_out=len(out_bytes),
        )

    def _scrub_recipe_yaml(self, dst: Path, src: Path) -> RedactionCounts:
        """Scrub ``recipe.resolved.yaml`` as a schema-owned document.

        The resolved recipe re-emits the recipe- and cell-scope ``extra_env``
        overlays (and any inline-docker baseline ``env``) verbatim. Treating it
        as plain text left those copies unscrubbed while the identical values
        were removed from ``matrix.json``.

        Parsing and re-emitting go through :mod:`aorta.triage.recipe` because
        this module is stdlib-only by rubric §3.F -- the YAML dependency stays
        behind that seam.
        """
        try:
            doc = load_recipe_mapping(src)
        except (RecipeSchemaError, UnicodeDecodeError) as exc:
            # Fail closed, matching the structured-JSON path: an unparseable
            # recipe must not be copied through unredacted, and a raw decode
            # error would otherwise escape staging as an unhandled traceback.
            raise RedactionError(src, exc) from exc
        if not isinstance(doc, dict):
            raise RedactionError(
                src,
                ValueError(
                    "recipe.resolved.yaml must contain a top-level mapping, "
                    f"got {type(doc).__name__}"
                ),
            )

        env_removed = [0]
        path_index = _PathIndex()
        ip_index = _IpIndex()
        try:
            scrubbed = _scrub_json_value(
                doc,
                cfg=self._cfg,
                env_removed=env_removed,
                path_index=path_index,
                ip_index=ip_index,
                env_mapping_ids=_collect_env_mapping_ids(doc, "recipe"),
            )
        except _JsonKeyCollisionError as exc:
            raise RedactionError(src, exc) from exc
        out = dump_recipe_mapping(scrubbed)
        dst.write_text(out, encoding="utf-8")
        out_bytes = out.encode("utf-8")
        return RedactionCounts(
            env_keys_removed=env_removed[0],
            paths_rewritten=path_index.rewrites,
            ips_rewritten=ip_index.ipv4_rewrites + ip_index.ipv6_rewrites,
            bytes_in=src.stat().st_size,
            bytes_out=len(out_bytes),
        )

    def _scrub_json_document(
        self,
        raw: bytes,
        dst: Path,
        src: Path,
        *,
        document_kind: str | None,
    ) -> RedactionCounts:
        text = raw.decode("utf-8", errors="replace")
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            # Fail closed: a corrupt/truncated structured artifact must not
            # slip through unredacted, and the raw decode error would otherwise
            # escape staging as an unhandled traceback (it is not an OSError,
            # so the writer's OSError->BundleIOError wrap misses it).
            raise RedactionError(src, exc) from exc
        if document_kind is not None and not isinstance(doc, dict):
            raise RedactionError(
                src,
                ValueError(
                    f"{document_kind} JSON must contain a top-level object, "
                    f"got {type(doc).__name__}"
                ),
            )
        env_removed = [0]
        path_index = _PathIndex()
        ip_index = _IpIndex()
        env_mapping_ids = (
            _collect_env_mapping_ids(doc, document_kind)
            if document_kind is not None
            else frozenset()
        )
        try:
            scrubbed = _scrub_json_value(
                doc,
                cfg=self._cfg,
                env_removed=env_removed,
                path_index=path_index,
                ip_index=ip_index,
                env_mapping_ids=env_mapping_ids,
            )
        except _JsonKeyCollisionError as exc:
            raise RedactionError(src, exc) from exc
        out = json.dumps(scrubbed, indent=2, sort_keys=False) + "\n"
        dst.write_text(out, encoding="utf-8")
        out_bytes = out.encode("utf-8")
        return RedactionCounts(
            env_keys_removed=env_removed[0],
            paths_rewritten=path_index.rewrites,
            ips_rewritten=ip_index.ipv4_rewrites + ip_index.ipv6_rewrites,
            bytes_in=len(raw),
            bytes_out=len(out_bytes),
        )

    def _scrub_text_stream(self, src: Path, dst: Path) -> RedactionCounts:
        # stdout.log / stderr.log can be very large in real runs, so stream the
        # scrub window-by-window off disk instead of reading the whole artifact
        # into memory (peak ~= one MAX_LOG_BYTES window, not O(file size)).
        #
        # * ``newline=""`` disables universal-newline translation so CR / CRLF
        #   terminators survive byte-for-byte; the scrubbed output is then
        #   identical to a whole-file pass (windows are processed in file order
        #   against one shared index, so placeholder assignment is unchanged).
        # * ``errors="replace"`` keeps scrubbing alive on stray non-UTF-8
        #   subprocess bytes rather than failing open -- same fail-safe as the
        #   former whole-file decode (a single bad byte must not disable path /
        #   IP scrubbing for the rest of the file).
        cfg = self._cfg
        path_index = _PathIndex()
        ip_index = _IpIndex()
        bytes_out = 0
        with open(src, encoding="utf-8", errors="replace", newline="") as fh, open(
            dst, "w", encoding="utf-8", newline=""
        ) as out_fh:
            for chunk in _scrub_windows_into(
                _stream_line_windows(fh),
                scrub_paths=cfg.scrub_paths,
                scrub_ip_addresses=cfg.scrub_ip_addresses,
                path_index=path_index,
                ip_index=ip_index,
            ):
                out_fh.write(chunk)
                bytes_out += len(chunk.encode("utf-8"))
        return RedactionCounts(
            paths_rewritten=path_index.rewrites,
            ips_rewritten=ip_index.ipv4_rewrites + ip_index.ipv6_rewrites,
            bytes_in=src.stat().st_size,
            bytes_out=bytes_out,
        )


__all__ = [
    "RedactionCfg",
    "RedactingRedactor",
    "parse_redaction",
    "scrub_env_keys",
    "scrub_text",
]
