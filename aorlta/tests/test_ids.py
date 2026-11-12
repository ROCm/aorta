"""Ids, canonical JSON, hashes and seeds (Section 5 items 1-3); C1.T3 and the WP0 golden hashes.

The golden values were computed twice when this file was written, by `aorlta.ids` and by a direct
transcription of the Section 5 formulas (`json.dumps(sort_keys=True, separators=(",", ":"),
ensure_ascii=False)`, `hashlib`, `struct.pack("<Ni")`), and the two agreed.
"""

import hashlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from aorlta import ids
from aorlta.ids import IdError, IterationSeeds, canonical_json, seed

RUN_SEED = 20261102

# --- canonical JSON -------------------------------------------------------------------------------

GOLDEN_OBJ = {
    "z": [1, 2.5, None, True, False],
    "a": 'é\n"q"',
    "m": {"y": 0.1, "x": 1e-05, "w": 1.0, "v": 1e16, "u": -0.0, "t": 0.30000000000000004, "s": 123456789.123, "r": -7},
    "e": [],
}
GOLDEN_BYTES = (
    '{"a":"é\\n\\"q\\"","e":[],"m":{"r":-7,"s":123456789.123,"t":0.30000000000000004,"u":-0.0,'
    '"v":1e+16,"w":1.0,"x":1e-05,"y":0.1},"z":[1,2.5,null,true,false]}'
).encode("utf-8")
GOLDEN_SHA256 = "8d615bea0ed2217c36f67e62fdfe57e67378c16c510c6f61dc02cc1fa743ae6d"


def test_canonical_json_golden_bytes_and_hash():
    assert canonical_json(GOLDEN_OBJ) == GOLDEN_BYTES
    assert ids.sha256_json(GOLDEN_OBJ) == GOLDEN_SHA256


@pytest.mark.parametrize(
    "value, text",
    [(0.1, "0.1"), (1e-05, "1e-05"), (1.0, "1.0"), (1e16, "1e+16"), (-0.0, "-0.0"),
     (0.30000000000000004, "0.30000000000000004"), (123456789.123, "123456789.123"), (2.5, "2.5")],
)
def test_floats_are_written_as_repr(value, text):
    assert canonical_json(value) == text.encode()
    assert canonical_json([value]) == f"[{text}]".encode()


def test_key_order_and_tuples_do_not_change_the_bytes():
    assert canonical_json({"b": 1, "a": (2, 3)}) == canonical_json({"a": [2, 3], "b": 1}) == b'{"a":[2,3],"b":1}'


def test_utf8_not_ascii_escapes():
    assert canonical_json("é") == b'"\xc3\xa9"'


def test_int_and_float_stay_distinct():
    assert canonical_json(1) != canonical_json(1.0)
    assert canonical_json(True) == b"true" and canonical_json(1) == b"1"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), {1: "a"}, {"a": {(1,): 2}}, {1, 2},
                                 b"bytes", object(), "\ud800"])
def test_canonical_json_refuses_values_with_no_single_form(bad):
    with pytest.raises(IdError):
        canonical_json(bad)


# --- id lists -------------------------------------------------------------------------------------

GOLDEN_IDS = [248045, 8948, 198, 2610, 525, 458, 6203, 13, 248046]
GOLDEN_IDS_SHA256 = "a8b7373dbbf18053e5928465b4dd4e9fe231ca92e398f5bb98e241d07167e551"


def test_id_list_bytes_are_little_endian_int32():
    assert ids.ids_bytes([1, 256, 2**31 - 1]) == bytes.fromhex("01000000" "00010000" "ffffff7f")


def test_id_list_golden_hash():
    assert ids.sha256_ids(GOLDEN_IDS) == GOLDEN_IDS_SHA256
    assert ids.sha256_ids(tuple(GOLDEN_IDS)) == GOLDEN_IDS_SHA256
    assert ids.sha256_ids([]) == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("bad", [[-1], [2**31], [True], [1.0], ["1"], "abc"])
def test_id_list_refuses_non_int32(bad):
    with pytest.raises(IdError):
        ids.sha256_ids(bad)


def test_file_and_text_hashes(tmp_path: Path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc" * 100_000)
    assert ids.sha256_file(p, chunk=7) == hashlib.sha256(b"abc" * 100_000).hexdigest()
    assert ids.sha256_text("é") == hashlib.sha256("é".encode()).hexdigest()


# --- ids ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prompt_id", ["standin:k0001", "other/p/12", "a/1/2", "x y", "é"])
def test_chain_id_round_trips_including_slashes_in_prompt_ids(prompt_id):
    cid = ids.chain_id("run-1.a_b", 7, prompt_id, 3)
    assert cid == f"run-1.a_b/7/{prompt_id}/3"
    assert ids.parse_chain_id(cid) == ids.ChainKey("run-1.a_b", 7, prompt_id, 3)
    assert ids.candidate_id("run-1.a_b", 7, prompt_id, 3) == cid


@pytest.mark.parametrize("bad", ["run/0/p/1", "run/1/p/x", "run/1/p/01", "run/01/p/1", "run/1//1", "run/1/p", "a b/1/p/1",
                                 "../1/p/0"])
def test_parse_chain_id_refuses_malformed_ids(bad):
    with pytest.raises(IdError):
        ids.parse_chain_id(bad)


@pytest.mark.parametrize("bad", ["", "a/b", "a:b", ".", "..", "é", None, 3])
def test_run_id_rule(bad):
    with pytest.raises(IdError):
        ids.validate_run_id(bad)


def test_policy_version_and_standin_ids():
    assert ids.policy_version("r1", 0) == "r1:v0"
    assert ids.parse_policy_version("r1:v12") == ("r1", 12)
    with pytest.raises(IdError):
        ids.parse_policy_version("r1:v01")
    with pytest.raises(IdError):
        ids.policy_version("r1", -1)
    assert ids.standin_prompt_id("kid42") == "standin:kid42"


# --- C1.T3: seeds ---------------------------------------------------------------------------------

GOLDEN_SEEDS = [
    ((20261102, 1, "standin:k0001", 0, 0), 1905968311),
    ((20261102, 1, "standin:k0001", 0, 1), 2066405105),
    ((20261102, 1, "standin:k0001", 0, 2), 1911302606),
    ((20261102, 1, "standin:k0001", 7, 0), 1834556463),
    ((20261102, 2, "standin:k0001", 0, 0), 2064444172),
    ((0, 1, "p/with/slashes", 3, 1), 663211411),
    ((20261030, 12, "other:é", 15, 2), 964427139),
]


@pytest.mark.parametrize("args, expected", GOLDEN_SEEDS)
def test_seed_golden_values(args, expected):
    assert seed(*args) == expected


def test_seed_is_the_section5_formula():
    args = [RUN_SEED, 3, "standin:k0042", 5, 2]
    digest = hashlib.sha256(canonical_json(args)).digest()
    assert seed(*args) == int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def test_seed_is_deterministic_across_processes():
    code = "from aorlta.ids import seed; print(seed(20261102, 4, 'standin:k0003', 6, 1))"
    env = dict(os.environ, PYTHONPATH=str(Path(ids.__file__).resolve().parents[1]))
    outs = set()
    for hash_seed in ("0", "1", "12345"):
        env["PYTHONHASHSEED"] = hash_seed
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        outs.add(out.stdout.strip())
    assert outs == {str(seed(20261102, 4, "standin:k0003", 6, 1))}


def _tuples():
    # 5 iterations x 834 prompts x K=8 x 3 turns = 100,080 tuples
    return [(it, f"standin:k{p:04d}", k, turn) for it in range(1, 6) for p in range(834) for k in range(8) for turn in range(3)]


def test_seeds_over_1e5_tuples_are_31_bit_and_distinct_but_for_chance():
    tuples = _tuples()
    assert len(tuples) >= 100_000
    seeds = [seed(RUN_SEED, *t) for t in tuples]
    assert all(0 <= s < 2**31 for s in seeds)
    assert max(seeds) >= 2**30  # the 31st bit is in use
    duplicates = sum(n - 1 for n in Counter(seeds).values() if n > 1)
    # A 31-bit hash of 100,080 distinct tuples collides by chance about 2.3 times; this set does once.
    assert duplicates == 1
    within_iteration = Counter((t[0], s) for t, s in zip(tuples, seeds))
    assert all(n == 1 for n in within_iteration.values())


def test_every_field_of_the_tuple_changes_the_seed():
    for p in range(300):
        pid = f"standin:k{p:04d}"
        for k in range(4):
            base = seed(RUN_SEED, 2, pid, k, 1)
            assert base != seed(RUN_SEED, 2, pid, k, 0)
            assert base != seed(RUN_SEED, 2, pid, k, 2)
            assert base != seed(RUN_SEED, 2, pid, k + 4, 1)
            assert base != seed(RUN_SEED, 3, pid, k, 1)
            assert base != seed(RUN_SEED + 1, 2, pid, k, 1)
            assert base != seed(RUN_SEED, 2, pid + "x", k, 1)


def test_iteration_seeds_are_unique_within_each_iteration():
    prompt_ids = [f"standin:k{p:04d}" for p in range(834)]
    for it in range(1, 6):
        table = IterationSeeds(RUN_SEED, it, prompt_ids, 8, 3)
        assert len(table) == 834 * 8 * 3
        assert len({e.seed for e in table}) == len(table)
        assert table.salted() == []
        assert table.seed_for("standin:k0007", 5, 2) == seed(RUN_SEED, it, "standin:k0007", 5, 2)


# Two real collisions of the unsalted seed in iteration 1 (found by search when this test was written).
SAME_TURN = ("standin:c20009", "standin:c51229", 1173336757)  # both at k=0, turn=0
CROSS_TURN = ("standin:t2368", "standin:c195658", 2122559475)  # t2368 at turn 1 == c195658 at turn 0


def test_a_collision_gets_salted_and_recorded():
    first, second, shared = SAME_TURN
    assert seed(RUN_SEED, 1, first, 0, 0) == seed(RUN_SEED, 1, second, 0, 0) == shared
    table = IterationSeeds(RUN_SEED, 1, [first, second], 1, 1)
    assert table.entry(first, 0, 0) == ids.SeedEntry(first, 0, 0, shared, 0)
    later = table.entry(second, 0, 0)
    assert later.salt == 1 and later.seed == seed(RUN_SEED, 1, second, 0, 0, salt=1) != shared
    assert table.salted() == [later]


def test_the_later_tuple_is_fixed_by_order_not_by_arrival():
    turn1_prompt, turn0_prompt, shared = CROSS_TURN
    assert seed(RUN_SEED, 1, turn1_prompt, 0, 1) == seed(RUN_SEED, 1, turn0_prompt, 0, 0) == shared
    # The turn-1 prompt is listed first, but turn 0 is assigned before turn 1, so the turn-1 tuple is salted.
    table = IterationSeeds(RUN_SEED, 1, [turn1_prompt, turn0_prompt], 1, 2)
    assert table.seed_for(turn0_prompt, 0, 0) == shared
    assert table.entry(turn1_prompt, 0, 1).salt == 1
    assert [(e.prompt_id, e.turn) for e in table.salted()] == [(turn1_prompt, 1)]
    assert len({e.seed for e in table}) == 4


def test_iteration_seeds_refuse_repeats_and_unplanned_tuples():
    with pytest.raises(IdError):
        IterationSeeds(RUN_SEED, 1, ["a", "a"], 2, 1)
    table = IterationSeeds(RUN_SEED, 1, ["a"], 2, 1)
    with pytest.raises(IdError):
        table.seed_for("a", 2, 0)
    with pytest.raises(IdError):
        table.seed_for("a", 0, 1)


@pytest.mark.parametrize("args", [(1.0, 1, "p", 0, 0), (1, 0, "p", 0, 0), (1, 1, "", 0, 0), (1, 1, "p", -1, 0),
                                  (1, 1, "p", 0, -1), (1, 1, "p", True, 0), (1, 1, "p", 0, 0.0)])
def test_seed_refuses_mistyped_tuples(args):
    with pytest.raises(IdError):
        seed(*args)


# --- HashRng --------------------------------------------------------------------------------------


def test_hash_rng_draws_depend_on_key_and_name_only():
    a, b = ids.HashRng(["k", 1]), ids.HashRng(["k", 1])
    assert a.uniform("x") == b.uniform("x") and a.uniform("x") != a.uniform("y")
    assert ids.HashRng(["k", 2]).uniform("x") != a.uniform("x")
    sa, sb = a.stream("s"), b.stream("s")
    assert [sa.random() for _ in range(3)] == [sb.random() for _ in range(3)]
    assert 0 <= a.uniform("x") < 1 and a.randint("r", 3, 5) in (3, 4, 5) and a.choice("c", "xyz") in "xyz"
