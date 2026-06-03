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
from aorta.triage.recipe import RecipeSchemaError

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
    chunks. Each chunk holds ``MAX_LOG_BYTES // 4`` code points, which is
    ``<= MAX_LOG_BYTES`` bytes for any UTF-8 string (4 bytes/char max). A token
    straddling such a hard-split seam may be missed -- an accepted cost for a
    line that defeats line-based windowing by construction; correctness still
    holds for every newline-terminated log.
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
            for i in range(0, len(line), max_chars):
                yield line[i : i + max_chars]
            continue
        if buf and size + line_bytes > MAX_LOG_BYTES:
            yield "".join(buf)
            buf = []
            size = 0
        buf.append(line)
        size += line_bytes
    if buf:
        yield "".join(buf)


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


def _scrub_json_value(
    value: Any,
    *,
    cfg: RedactionCfg,
    env_removed: list[int],
    path_index: _PathIndex,
    ip_index: _IpIndex,
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
            if key == "env" and isinstance(item, dict):
                kept_env, removed = scrub_env_keys(
                    {str(k): str(v) for k, v in item.items()},
                    cfg.scrub_env_keys,
                )
                env_removed[0] += removed
                # Removing matching keys is not enough: a *retained* key's
                # value can still carry a path or IP (e.g.
                # LD_LIBRARY_PATH=/home/customer/...). Scrub the values too
                # so result.json env matches the host_env.json path, which
                # already scrubs values via its whole-document pass.
                new_dict[key] = {
                    env_key: _scrub_str_into(
                        env_val, cfg=cfg, path_index=path_index, ip_index=ip_index
                    )
                    for env_key, env_val in kept_env.items()
                }
                continue
            new_dict[key] = _scrub_json_value(
                item,
                cfg=cfg,
                env_removed=env_removed,
                path_index=path_index,
                ip_index=ip_index,
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


class RedactingRedactor(Redactor):
    """Applies a probe recipe's ``redaction:`` block during bundling."""

    kind = "probe.v1"

    def __init__(self, cfg: RedactionCfg) -> None:
        self._cfg = cfg

    def scrub_file(self, src: Path, dst: Path) -> RedactionCounts:
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.name == "probe.env":
            counts = self._scrub_probe_env(src.read_bytes(), dst)
        elif src.name == "result.json":
            counts = self._scrub_result_json(src.read_bytes(), dst, src)
        elif src.name == "host_env.json":
            counts = self._scrub_host_env_json(src.read_bytes(), dst, src)
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
        scrubbed, removed = scrub_env_keys(env, self._cfg.scrub_env_keys)
        out = _format_probe_env(scrubbed)
        dst.write_text(out, encoding="utf-8")
        # Mode is carried from the source by scrub_file's shutil.copymode;
        # the run-dir probe.env is written 0600 by the workload, so the
        # staged copy stays owner-only without a hardcoded chmod here.
        out_bytes = out.encode("utf-8")
        return RedactionCounts(
            env_keys_removed=removed,
            bytes_in=len(raw),
            bytes_out=len(out_bytes),
        )

    def _scrub_result_json(self, raw: bytes, dst: Path, src: Path) -> RedactionCounts:
        text = raw.decode("utf-8", errors="replace")
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            # Fail closed: a corrupt/truncated result.json must not slip
            # through unredacted, and the raw decode error would otherwise
            # escape staging as an unhandled traceback (it is not an
            # OSError, so the writer's OSError->BundleIOError wrap misses it).
            raise RedactionError(src, exc) from exc
        env_removed = [0]
        path_index = _PathIndex()
        ip_index = _IpIndex()
        scrubbed = _scrub_json_value(
            doc,
            cfg=self._cfg,
            env_removed=env_removed,
            path_index=path_index,
            ip_index=ip_index,
        )
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

    def _scrub_host_env_json(self, raw: bytes, dst: Path, src: Path) -> RedactionCounts:
        text = raw.decode("utf-8", errors="replace")
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            # Fail closed for the same reason as _scrub_result_json: a
            # parse failure must stop bundling rather than emit a
            # potentially unredacted host_env.json.
            raise RedactionError(src, exc) from exc
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
