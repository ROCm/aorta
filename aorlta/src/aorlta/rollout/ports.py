"""The host ports a rollout engine binds, and how they are chosen (`rollout.port_base`).

A TokenSpeed `Engine` at 3b684074 binds nothing at its `port` kwarg itself: on one node,
`PortArgs.init_new` (runtime/utils/server_args.py) derives what it binds from it, all on the host's
network (the run's containers use the host network):

- the rendezvous port d = port + 233, which the scheduler's torch.distributed TCP store listens on, and
  ZMQ sockets at d+1 (tokenizer), d+3 (rpc), d+4 (metrics) and d+5 (scheduler input). It also wants
  d+6 free. While any of the six is taken it moves d up by 10 and checks again, with no bound;
- an "nccl" port at a random offset in [100, 1000], moved up by 42 while taken, which it checks but a
  one-node engine does not bind (its rendezvous is d).

So an engine's ports stay inside [port, port + PORT_STRIDE) unless about 170 clusters there are taken.
Each check is a bind test that lets the port go again before the engine binds it, so another process
can take a port in between; the engine then fails to start with "Address already in use" (from ZMQ in
the engine's process) or "Initialization failed" (the TCP store, in the scheduler's).

With `rollout.port_base` null, each start of an engine picks a window here: one of the fixed slots
`AUTO_LOW + PORT_STRIDE * n`, in a random order, whose cluster passes the same bind test, whose window
lies outside the kernel's ephemeral port range (`ip_local_port_range`, where outgoing connections get
their ports at any time) and which another engine of this process does not hold. With the default
ephemeral range there are 12 slots, so eight engines of a node always find room on an idle host. A start that fails on a taken port tries another window, at
most START_ATTEMPTS times in all (EngineProc.start). With an int, engine i uses port_base + 2000 i,
as every run did before the key existed (20000), and is started once.
"""

from __future__ import annotations

import random
import re
import socket
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

PORT_BASE, PORT_STRIDE = 20000, 2000  # the fixed layout: engine i at PORT_BASE + PORT_STRIDE * i
CLUSTER_OFFSETS = (233, 234, 236, 237, 238, 239)  # d = port + 233 and the ports TokenSpeed derives from it
AUTO_LOW, AUTO_HIGH = 10000, 65536  # automatic windows are chosen in [AUTO_LOW, AUTO_HIGH), less the ephemeral range
EPHEMERAL_DEFAULT = (32768, 60999)  # Linux's default ip_local_port_range, if the file cannot be read
EPHEMERAL_FILE = Path("/proc/sys/net/ipv4/ip_local_port_range")
CHOOSE_TRIES = 200  # slots bind-tested for one window before giving up
START_ATTEMPTS = 3  # engine starts with automatic ports, the first included, before a taken port stops the run
_IN_USE = re.compile(r"address already in use|EADDRINUSE", re.IGNORECASE)

_LOCK = threading.Lock()
_HELD: dict[int, int] = {}  # holder id -> base of the window it holds, for the engines of this process


def engine_port(index: int, base: int = PORT_BASE) -> int:
    """Engine `index`'s port in the fixed layout."""
    return base + PORT_STRIDE * index


def cluster(port: int) -> list[int]:
    """The ports an engine started with `port` binds (and checks) when none of them is taken."""
    return [port + o for o in CLUSTER_OFFSETS]


def port_free(port: int) -> bool:
    """TokenSpeed's own test (`is_port_available`): the port can be bound on every address and listened on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except (OSError, OverflowError):
            return False


def taken(ports: Iterable[int], free: Callable[[int], bool] = port_free) -> list[int]:
    return [p for p in ports if not free(p)]


def ephemeral_range(path: Path = EPHEMERAL_FILE) -> tuple[int, int]:
    """The kernel's ephemeral port range, inclusive; Linux's default if it cannot be read."""
    try:
        lo, hi = (int(x) for x in path.read_text().split())
        return lo, hi
    except (OSError, ValueError):
        return EPHEMERAL_DEFAULT


def _overlaps(a: int, b: int) -> bool:
    return a < b + PORT_STRIDE and b < a + PORT_STRIDE


def candidate_bases(ephemeral: tuple[int, int], low: int = AUTO_LOW, high: int = AUTO_HIGH) -> list[tuple[int, int]]:
    """The intervals [lo, hi] of bases whose whole window lies in [low, high) and outside `ephemeral`."""
    e_lo, e_hi = ephemeral
    spans = [(low, min(high, e_lo) - PORT_STRIDE), (max(low, e_hi + 1), high - PORT_STRIDE)]
    return [(lo, hi) for lo, hi in spans if lo <= hi]


class PortsUnavailable(RuntimeError):
    """No free window of engine ports was found."""


def slot_bases(ephemeral: tuple[int, int], low: int = AUTO_LOW, high: int = AUTO_HIGH) -> list[int]:
    """The bases automatic windows are chosen from: `low + PORT_STRIDE * n`, each window wholly inside one span of
    `candidate_bases`. Fixed slots cannot overlap one another, so the windows held never fragment what is left (any
    base in a span could: seven windows at random bases can leave no room for an eighth on an idle host)."""
    out = []
    for lo, hi in candidate_bases(ephemeral, low, high):
        first = low + -(-(lo - low) // PORT_STRIDE) * PORT_STRIDE
        out.extend(range(first, hi + 1, PORT_STRIDE))
    return out


def hold_window(holder: int, *, rng: random.Random | None = None, free: Callable[[int], bool] = port_free,
                ephemeral: tuple[int, int] | None = None, tries: int = CHOOSE_TRIES,
                avoid: Iterable[int] = ()) -> int:
    """A base for `holder` (any id, one window each), from `slot_bases` in a random order: its cluster passes the bind
    test, its window is outside the ephemeral range and overlaps no window another holder in this process holds, nor
    one in `avoid` (bases a failed start of this holder used). At most `tries` slots are bind-tested. Held until
    `release_window(holder)`; a holder's earlier window is released first."""
    rng = rng or random.SystemRandom()
    slots = slot_bases(ephemeral if ephemeral is not None else ephemeral_range())
    if not slots:
        raise PortsUnavailable("no port window lies outside the kernel's ephemeral port range")
    rng.shuffle(slots)
    avoid = list(avoid)
    tested = 0
    with _LOCK:
        _HELD.pop(holder, None)
        others = list(_HELD.values())
        for base in slots:
            if any(_overlaps(base, b) for b in others + avoid):
                continue
            if tested == tries:
                break
            tested += 1
            if taken(cluster(base), free):
                continue
            _HELD[holder] = base
            return base
    raise PortsUnavailable(f"no free window of {PORT_STRIDE} engine ports among {len(slots)} slots ({tested} "
                           f"bind-tested; {len(others)} held by other engines of this process, {len(avoid)} avoided)")


def release_window(holder: int) -> None:
    with _LOCK:
        _HELD.pop(holder, None)


def held_windows() -> dict[int, int]:
    with _LOCK:
        return dict(_HELD)


def lost_to_another(error: str, base: int, free: Callable[[int], bool] = port_free) -> bool:
    """Whether a failed start with `base` looks like a port taken between the check and the engine's bind: the error
    says so, or a port of its cluster is held now that the engine's process is gone. Anything else is not retried."""
    return bool(_IN_USE.search(error)) or bool(taken(cluster(base), free))


def bound_ports(port_args) -> dict | None:
    """The ports a started engine resolved (TokenSpeed's `Engine.port_args`), by role; None if it has none."""
    if port_args is None:
        return None
    out: dict[str, int | None] = {}
    for role, attr in (("dist_init", "dist_init_addr"), ("tokenizer", "tokenizer_ipc_name"), ("rpc", "rpc_ipc_name"),
                       ("metrics", "metrics_ipc_name"), ("scheduler_input", "scheduler_input_ipc_name")):
        value = getattr(port_args, attr, None)
        m = re.search(r":(\d+)$", value) if isinstance(value, str) else None
        out[role] = int(m.group(1)) if m else None
    nccl = getattr(port_args, "nccl_port", None)
    out["nccl_checked"] = nccl if isinstance(nccl, int) else None
    return out
