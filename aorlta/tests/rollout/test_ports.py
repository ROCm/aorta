"""The engines' host ports (rollout/ports.py): TokenSpeed's layout, the bind test, and the automatic choice of a free
window, on real sockets of this host where the test needs a port to be taken (CPU only)."""

from __future__ import annotations

import socket
import types

import pytest

from aorlta.rollout import ports as P


class Seq:
    """A random source whose shuffle puts the given bases first, in order (as `hold_window` tries them)."""

    def __init__(self, bases):
        self.bases = list(bases)

    def shuffle(self, slots):
        rest = [b for b in slots if b not in self.bases]
        assert all(b in slots for b in self.bases), (self.bases, slots)
        slots[:] = self.bases + rest


def listen(port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    return s


def free_bases(n: int, start: int = 20000) -> list[int]:
    """`n` bases in the lower automatic span, windows apart, whose clusters are free on this host now."""
    out, base = [], start
    while len(out) < n:
        assert base < 30000, "no free port windows on this host"
        if not P.taken(P.cluster(base)):
            out.append(base)
        base += P.PORT_STRIDE
    return out


@pytest.fixture(autouse=True)
def no_held_windows():
    yield
    for holder in list(P.held_windows()):
        P.release_window(holder)


def test_the_fixed_layout_is_the_one_every_run_used_before_the_key():
    assert [P.engine_port(i) for i in range(4)] == [20000, 22000, 24000, 26000]
    assert [P.engine_port(i, 31000) for i in range(3)] == [31000, 33000, 35000]


def test_the_cluster_is_what_tokenspeed_derives_from_the_port():
    # PortArgs.init_new at 3b684074: d = port + 233; tokenizer d+1, rpc d+3, metrics d+4, scheduler input d+5; d+6
    # checked too
    assert P.cluster(20000) == [20233, 20234, 20236, 20237, 20238, 20239]
    assert all(o < P.PORT_STRIDE for o in P.CLUSTER_OFFSETS)


def test_the_bind_test_sees_a_listening_port():
    base = free_bases(1)[0]
    assert P.port_free(base + 233)
    with listen(base + 233):
        assert not P.port_free(base + 233)
        assert P.taken(P.cluster(base)) == [base + 233]


def test_automatic_windows_lie_outside_the_ephemeral_range():
    assert P.candidate_bases((32768, 60999)) == [(10000, 30768), (61000, 63536)]
    for lo, hi in P.candidate_bases((32768, 60999)):
        for base in (lo, hi):
            window = range(base, base + P.PORT_STRIDE)
            assert window[-1] < 65536 and (window[-1] < 32768 or window[0] > 60999)
    assert P.candidate_bases((1024, 65535)) == []
    with pytest.raises(P.PortsUnavailable, match="ephemeral"):
        P.hold_window(1, ephemeral=(1024, 65535))


def test_the_ephemeral_range_is_read_from_the_kernel(tmp_path):
    f = tmp_path / "range"
    f.write_text("40000\t50000\n")
    assert P.ephemeral_range(f) == (40000, 50000)
    assert P.ephemeral_range(tmp_path / "missing") == P.EPHEMERAL_DEFAULT


def test_a_window_with_a_taken_port_is_passed_over():
    busy, ok = free_bases(2)
    with listen(busy + 236):  # one port of the cluster, not its first
        base = P.hold_window(1, rng=Seq([busy, ok]), ephemeral=(32768, 60999))
    assert base == ok and P.held_windows() == {1: ok}


def test_engines_of_one_process_never_share_a_window():
    a = P.hold_window(1, rng=Seq([20000]), free=lambda p: True, ephemeral=(32768, 60999))
    b = P.hold_window(2, rng=Seq([20000, 22000]), free=lambda p: True, ephemeral=(32768, 60999))
    assert (a, b) == (20000, 22000)  # 20000 is engine 1's
    P.release_window(1)
    assert P.hold_window(3, rng=Seq([20000]), free=lambda p: True, ephemeral=(32768, 60999)) == 20000
    # a holder choosing again gives up its own window first, and skips the ones it is told to avoid
    assert P.hold_window(2, rng=Seq([22000]), free=lambda p: True, ephemeral=(32768, 60999)) == 22000
    again = P.hold_window(2, rng=Seq([22000, 12000]), free=lambda p: True, ephemeral=(32768, 60999), avoid=[22000])
    assert again == 12000 and P.held_windows() == {3: 20000, 2: 12000}


def test_automatic_windows_are_fixed_slots_inside_the_spans():
    slots = P.slot_bases((32768, 60999))
    assert slots == list(range(10000, 30001, P.PORT_STRIDE)) + [62000]
    assert all(any(lo <= b <= hi for lo, hi in P.candidate_bases((32768, 60999))) for b in slots)
    assert P.slot_bases((1024, 65535)) == []


@pytest.mark.parametrize("seed", [105, *range(40)])
def test_eight_engines_always_find_windows_on_an_idle_host(seed):
    # random bases fragmented the spans: with Random(105) the eighth of eight found no room in 200 tries
    import random

    rng = random.Random(seed)
    bases = [P.hold_window(h, rng=rng, free=lambda p: True, ephemeral=(32768, 60999)) for h in range(8)]
    assert len(set(bases)) == 8 and all(b in P.slot_bases((32768, 60999)) for b in bases)
    assert all(not P._overlaps(a, b) for i, a in enumerate(bases) for b in bases[i + 1:])
    # all twelve slots can be held at once, and a thirteenth is an error
    more = [P.hold_window(h, rng=rng, free=lambda p: True, ephemeral=(32768, 60999)) for h in range(8, 12)]
    assert sorted(bases + more) == P.slot_bases((32768, 60999))
    with pytest.raises(P.PortsUnavailable, match="12 slots"):
        P.hold_window(12, rng=rng, free=lambda p: True, ephemeral=(32768, 60999))


def test_no_free_window_is_an_error_not_a_guess():
    with pytest.raises(P.PortsUnavailable, match="no free window"):
        P.hold_window(1, free=lambda p: False, tries=20)
    assert P.held_windows() == {}


def test_a_random_choice_is_free_and_in_range():
    for holder in range(8):
        base = P.hold_window(holder)
        assert not P.taken(P.cluster(base))
        assert any(lo <= base <= hi for lo, hi in P.candidate_bases(P.ephemeral_range()))
    bases = sorted(P.held_windows().values())
    assert all(b - a >= P.PORT_STRIDE for a, b in zip(bases, bases[1:]))


def test_a_lost_port_is_told_from_another_failure():
    base = free_bases(1)[0]
    assert P.lost_to_another("zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:1')", base)
    assert P.lost_to_another("RuntimeError: ... code: -98, name: EADDRINUSE, message: address already in use", base)
    assert not P.lost_to_another("RuntimeError: Initialization failed. Please see the error messages above.", base)
    with listen(base + 233):  # the TCP store's port, which only the scheduler's own log names
        assert P.lost_to_another("RuntimeError: Initialization failed. Please see the error messages above.", base)


def test_the_bound_ports_are_read_from_tokenspeeds_port_args():
    args = types.SimpleNamespace(dist_init_addr="127.0.0.1:20243", tokenizer_ipc_name="tcp://127.0.0.1:20244",
                                 rpc_ipc_name="tcp://127.0.0.1:20246", metrics_ipc_name="tcp://127.0.0.1:20247",
                                 scheduler_input_ipc_name="tcp://127.0.0.1:20248", nccl_port=20517)
    assert P.bound_ports(args) == {"dist_init": 20243, "tokenizer": 20244, "rpc": 20246, "metrics": 20247,
                                   "scheduler_input": 20248, "nccl_checked": 20517}
    assert P.bound_ports(None) is None
    assert P.bound_ports(types.SimpleNamespace(dist_init_addr="ipc:///x"))["dist_init"] is None
