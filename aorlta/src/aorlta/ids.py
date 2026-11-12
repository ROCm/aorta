"""Ids, canonical JSON, hashing and seeds (design Section 5, items 1-3).

Everything here is a pure function of its arguments, so any two processes, nodes or Python
versions agree on every id, hash and seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
STANDIN_PROMPT_PREFIX = "standin:"
INT32_MAX = 2**31 - 1
SEED_MASK = 0x7FFFFFFF


class IdError(ValueError):
    """An id, hash input or seed tuple that breaks a Section 5 rule."""


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# --- ids (item 1) ---------------------------------------------------------------------------------


def validate_run_id(run_id: Any) -> str:
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise IdError(f"run_id must match [A-Za-z0-9_.-]+, got {run_id!r}")
    if run_id in (".", ".."):
        # The run directory is <root>/<source>/<run_id>, so these would name a parent directory.
        raise IdError(f"run_id {run_id!r} names a directory, not a run")
    return run_id


def validate_iteration(iteration: Any) -> int:
    if not _is_int(iteration) or iteration < 1:
        raise IdError(f"iteration must be an int >= 1, got {iteration!r}")
    return iteration


def validate_k(k: Any) -> int:
    if not _is_int(k) or k < 0:
        raise IdError(f"k must be an int >= 0, got {k!r}")
    return k


def validate_prompt_id(prompt_id: Any) -> str:
    if not isinstance(prompt_id, str) or not prompt_id:
        raise IdError(f"prompt_id must be a non-empty str, got {prompt_id!r}")
    return prompt_id


def standin_prompt_id(kid: str) -> str:
    """The prompt id of a stand-in prompt built from rocjitsu kernel `kid` (Section 5 item 1)."""
    if not isinstance(kid, str) or not kid:
        raise IdError(f"kid must be a non-empty str, got {kid!r}")
    return STANDIN_PROMPT_PREFIX + kid


@dataclass(frozen=True)
class ChainKey:
    run_id: str
    iteration: int
    prompt_id: str
    k: int


def chain_id(run_id: str, iteration: int, prompt_id: str, k: int) -> str:
    """`f"{run_id}/{iteration}/{prompt_id}/{k}"`; unique because k is an int after the last `/`."""
    validate_run_id(run_id)
    validate_iteration(iteration)
    validate_prompt_id(prompt_id)
    validate_k(k)
    return f"{run_id}/{iteration}/{prompt_id}/{k}"


def candidate_id(run_id: str, iteration: int, prompt_id: str, k: int) -> str:
    """A candidate's id is its chain's id (Section 5 item 1)."""
    return chain_id(run_id, iteration, prompt_id, k)


_ITERATION_RE = re.compile(r"[1-9][0-9]*")
_K_RE = re.compile(r"0|[1-9][0-9]*")


def parse_chain_id(cid: str) -> ChainKey:
    """Invert `chain_id`. A prompt id may contain `/`; run_id cannot, and k is the last field."""
    if not isinstance(cid, str):
        raise IdError(f"chain_id must be a str, got {cid!r}")
    run_id, sep1, rest = cid.partition("/")
    iteration, sep2, rest = rest.partition("/")
    prompt_id, sep3, k = rest.rpartition("/")
    if not (sep1 and sep2 and sep3) or not _ITERATION_RE.fullmatch(iteration) or not _K_RE.fullmatch(k):
        raise IdError(f"not a chain_id: {cid!r}")
    key = ChainKey(validate_run_id(run_id), int(iteration), validate_prompt_id(prompt_id), int(k))
    if chain_id(key.run_id, key.iteration, key.prompt_id, key.k) != cid:
        raise IdError(f"not a canonical chain_id: {cid!r}")
    return key


def policy_version(run_id: str, n: int) -> str:
    """`f"{run_id}:v{n}"`: v0 is the merged base, vn the weights after iteration n."""
    validate_run_id(run_id)
    if not _is_int(n) or n < 0:
        raise IdError(f"policy version number must be an int >= 0, got {n!r}")
    return f"{run_id}:v{n}"


_POLICY_RE = re.compile(r"v(0|[1-9][0-9]*)")


def parse_policy_version(version: str) -> tuple[str, int]:
    if not isinstance(version, str):
        raise IdError(f"policy_version must be a str, got {version!r}")
    run_id, sep, v = version.partition(":")
    m = _POLICY_RE.fullmatch(v)
    if not sep or m is None:
        raise IdError(f"not a policy_version: {version!r}")
    return validate_run_id(run_id), int(m.group(1))


# --- canonical JSON and hashes (item 2) -----------------------------------------------------------


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, no spaces, UTF-8, floats as `repr`.

    Only JSON values are accepted: dict with str keys, list/tuple, str, int, float, bool, None.
    NaN, infinities, non-str keys and any other type raise IdError, so two different values can
    never canonicalise to the same bytes.
    """
    out: list[str] = []
    _emit(obj, out)
    try:
        return "".join(out).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise IdError(f"string is not valid Unicode: {exc}") from None


def _emit(o: Any, out: list[str]) -> None:
    if o is None:
        out.append("null")
    elif o is True:
        out.append("true")
    elif o is False:
        out.append("false")
    elif isinstance(o, int):
        out.append(int.__repr__(o))
    elif isinstance(o, float):
        if not math.isfinite(o):
            raise IdError(f"canonical JSON has no representation for {o!r}")
        out.append(float.__repr__(o))
    elif isinstance(o, str):
        out.append(json.dumps(o, ensure_ascii=False))
    elif isinstance(o, (list, tuple)):
        out.append("[")
        for i, item in enumerate(o):
            if i:
                out.append(",")
            _emit(item, out)
        out.append("]")
    elif isinstance(o, dict):
        for key in o:
            if not isinstance(key, str):
                raise IdError(f"canonical JSON keys must be str, got {key!r}")
        out.append("{")
        for i, key in enumerate(sorted(o)):
            if i:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _emit(o[key], out)
        out.append("}")
    else:
        raise IdError(f"canonical JSON cannot represent {type(o).__name__}")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    if not isinstance(text, str):
        raise IdError(f"sha256_text needs a str, got {type(text).__name__}")
    return sha256_hex(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def ids_bytes(ids: Sequence[int]) -> bytes:
    """Token ids as little-endian int32 bytes, the form every id-list hash is taken over."""
    if isinstance(ids, (str, bytes)):
        raise IdError("token ids must be a sequence of ints")
    ids = list(ids)
    for i, t in enumerate(ids):
        if not _is_int(t) or not 0 <= t <= INT32_MAX:
            raise IdError(f"token id {i} is not an int in [0, 2**31): {t!r}")
    return struct.pack(f"<{len(ids)}i", *ids)


def sha256_ids(ids: Sequence[int]) -> str:
    return sha256_hex(ids_bytes(ids))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# --- seeds (item 3) -------------------------------------------------------------------------------


def seed(run_seed: int, iteration: int, prompt_id: str, k: int, turn: int, salt: int = 0) -> int:
    """`int.from_bytes(sha256(canonical_json([run_seed, iteration, prompt_id, k, turn])).digest()[:4],
    "little") & 0x7FFFFFFF`.

    A nonzero `salt` is appended as a sixth field; it is only ever set by `IterationSeeds` to
    re-derive a seed that collided within one iteration.
    """
    if not _is_int(run_seed):
        raise IdError(f"run_seed must be an int, got {run_seed!r}")
    validate_iteration(iteration)
    validate_prompt_id(prompt_id)
    validate_k(k)
    if not _is_int(turn) or turn < 0:
        raise IdError(f"turn must be an int >= 0, got {turn!r}")
    if not _is_int(salt) or salt < 0:
        raise IdError(f"salt must be an int >= 0, got {salt!r}")
    fields: list[Any] = [run_seed, iteration, prompt_id, k, turn]
    if salt:
        fields.append(salt)
    digest = hashlib.sha256(canonical_json(fields)).digest()
    return int.from_bytes(digest[:4], "little") & SEED_MASK


@dataclass(frozen=True)
class SeedEntry:
    prompt_id: str
    k: int
    turn: int
    seed: int
    salt: int


class IterationSeeds:
    """Every request seed one iteration may use, unique within the iteration.

    Seeds are assigned up front for all (prompt, k, turn) the iteration can reach, in a fixed order:
    turn-major, then prompts in the order given, then k. A tuple whose seed is already taken is
    "the later one" (Section 5 item 3): it is re-derived with salt 1, 2, ... until unique, and the
    salt is recorded. Because the order is fixed, which tuple gets salted never depends on the order
    in which chains happen to reach a turn, so a re-run of the iteration gets the same seeds.
    """

    def __init__(self, run_seed: int, iteration: int, prompt_ids: Sequence[str], K: int, turns: int):
        prompt_ids = list(prompt_ids)
        if len(set(prompt_ids)) != len(prompt_ids):
            raise IdError("prompt_ids repeat within one iteration")
        if not _is_int(K) or K < 1:
            raise IdError(f"K must be an int >= 1, got {K!r}")
        if not _is_int(turns) or turns < 1:
            raise IdError(f"turns must be an int >= 1, got {turns!r}")
        self.run_seed, self.iteration = run_seed, iteration
        self._entries: dict[tuple[str, int, int], SeedEntry] = {}
        taken: set[int] = set()
        for turn in range(turns):
            for prompt_id in prompt_ids:
                for k in range(K):
                    salt = 0
                    s = seed(run_seed, iteration, prompt_id, k, turn)
                    while s in taken:
                        salt += 1
                        s = seed(run_seed, iteration, prompt_id, k, turn, salt=salt)
                    taken.add(s)
                    self._entries[(prompt_id, k, turn)] = SeedEntry(prompt_id, k, turn, s, salt)

    def entry(self, prompt_id: str, k: int, turn: int) -> SeedEntry:
        try:
            return self._entries[(prompt_id, k, turn)]
        except KeyError:
            raise IdError(f"no seed planned for {(prompt_id, k, turn)!r} in iteration {self.iteration}") from None

    def seed_for(self, prompt_id: str, k: int, turn: int) -> int:
        return self.entry(prompt_id, k, turn).seed

    def salted(self) -> list[SeedEntry]:
        """The entries that needed a salt, for the iteration record."""
        return [e for e in self._entries.values() if e.salt]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[SeedEntry]:
        return iter(self._entries.values())


# --- deterministic draws for fakes and tests ------------------------------------------------------


class HashRng:
    """Named draws that depend only on (key, name), never on call order.

    Each draw is a sha256 of the key and the draw's name, so reordering code that draws does not
    change any value. `stream(name)` gives a `random.Random` for long sequences; only its `random()`
    method is used, whose output for an int seed Python guarantees across versions.
    """

    def __init__(self, key: Any):
        self.key = sha256_json(key)

    def _u64(self, name: str, i: int = 0) -> int:
        return int.from_bytes(hashlib.sha256(canonical_json([self.key, name, i])).digest()[:8], "little")

    def uniform(self, name: str) -> float:
        """A float in [0, 1)."""
        return self._u64(name) / 2.0**64

    def randint(self, name: str, lo: int, hi: int) -> int:
        """An int in [lo, hi]."""
        if hi < lo:
            raise IdError(f"empty range [{lo}, {hi}]")
        return lo + self._u64(name) % (hi - lo + 1)

    def choice(self, name: str, options: Sequence[Any]) -> Any:
        if not options:
            raise IdError("choice from an empty sequence")
        return options[self._u64(name) % len(options)]

    def hex(self, name: str, n: int = 8) -> str:
        return hashlib.sha256(canonical_json([self.key, name, "hex"])).hexdigest()[:n]

    def stream(self, name: str) -> random.Random:
        return random.Random(self._u64(name, 1))
