"""The checkpoint verifier: a floor, a control, and a ceiling the optimiser cannot pass.

Every test builds real safetensors files on disk -- header length, JSON header,
raw buffer -- rather than mocking the reader, because the reader is half of the
tool and a mocked one would test nothing about it. Each is a few kilobytes.

The claim that matters is the one a floor-only check gets wrong: a pair in which
every trained tensor moved and the frozen control did not can still carry a
write the optimiser did not make, and saying so requires looking at *how far*
things moved.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))

import verify_checkpoint_delta as vcd  # noqa: E402

FROZEN = vcd.FROZEN
K_PROJ = "model.layers.0.self_attn.k_proj.weight"
UP_PROJ = "model.layers.1.mlp.up_proj.weight"


def _blob(tensors: dict[str, np.ndarray]) -> bytes:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset, buf = 0, bytearray()
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        buf += raw
        offset += len(raw)
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + bytes(buf)


def write_checkpoint(root: Path, tensors: dict[str, np.ndarray], *, shards: int = 1) -> Path:
    """A safetensors tree, sharded with an index or as a single file."""
    root.mkdir(parents=True, exist_ok=True)
    if shards == 0:
        (root / "model.safetensors").write_bytes(_blob(tensors))
        return root
    names = list(tensors)
    weight_map = {}
    for index in range(shards):
        part = names[index::shards]
        shard = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        (root / shard).write_bytes(_blob({n: tensors[n] for n in part}))
        weight_map.update(dict.fromkeys(part, shard))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return root


def base() -> dict[str, np.ndarray]:
    return {
        FROZEN: np.full((4, 4), 0.5, dtype=np.float32),
        K_PROJ: np.full((4, 4), 0.25, dtype=np.float32),
        UP_PROJ: np.full((4, 4), 0.1, dtype=np.float32),
    }


def pair(tmp_path: Path, nudge: float, *, damage: float | None = None,
         pre_shards: int = 1, post_shards: int = 1):
    before = base()
    after = {name: array.copy() for name, array in before.items()}
    for name in after:
        if name != FROZEN:
            after[name] += nudge
    if damage is not None:
        after[K_PROJ][2, 3] = damage
    return (write_checkpoint(tmp_path / "pre", before, shards=pre_shards),
            write_checkpoint(tmp_path / "post", after, shards=post_shards))


def run(pre: Path, post: Path, *extra: str) -> int:
    return vcd.main([str(pre), str(post), "--lr", "1e-6", "--steps", "17", *extra])


# ---------------------------------------------------------------------------
# the ceiling, and why it bounds Adam
# ---------------------------------------------------------------------------


def test_the_first_adam_step_moves_by_at_most_lr():
    assert vcd.adam_step_ceiling(1) == pytest.approx(1.0)


def test_the_worst_case_ceiling_matches_a_simulated_adversarial_run():
    """The closed form against the real update rule, with the gradient chosen
    to maximise the final step: g_{t-k} proportional to (b1/b2)^k, the
    Cauchy-Schwarz equality case."""
    b1, b2 = 0.9, 0.999
    for steps in (1, 5, 17, 60):
        worst = 0.0
        for t in range(1, steps + 1):
            grads = [(b1 / b2) ** (t - s) for s in range(1, t + 1)]
            m = v = 0.0
            for g in grads:
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
            step = (m / (1 - b1**t)) / ((v / (1 - b2**t)) ** 0.5)
            worst += step
        assert worst == pytest.approx(vcd.adam_step_ceiling(steps), rel=1e-9)


def test_four_lr_per_step_is_sound_for_realistic_runs_and_refused_beyond():
    assert vcd.optimiser_bound(1e-6, 17) == pytest.approx(4 * 1e-6 * 17)
    assert vcd.optimiser_bound(1e-6, 862) > 0
    with pytest.raises(ValueError, match="not a sound ceiling"):
        vcd.optimiser_bound(1e-6, 863)


def test_an_unsound_ceiling_is_a_refusal_not_a_verdict(tmp_path, capsys):
    pre, post = pair(tmp_path, nudge=1e-5)
    code = vcd.main([str(pre), str(post), "--lr", "1e-6", "--steps", "5000"])
    assert code == vcd.EXIT_INCOMPLETE
    assert "not a sound ceiling" in capsys.readouterr().err


def test_steps_and_lr_are_required():
    """A floor-only pass is the verdict this tool exists to stop issuing."""
    with pytest.raises(SystemExit) as exc:
        vcd.main(["pre", "post"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# end to end, on real files
# ---------------------------------------------------------------------------


def test_a_healthy_pair_passes(tmp_path, capsys):
    pre, post = pair(tmp_path, nudge=1e-5)
    assert run(pre, post) == 0
    out = capsys.readouterr().out
    assert "VERDICT: every trained tensor moved" in out
    assert "BEYOND THE CEILING" not in out


def test_a_write_the_optimiser_did_not_make_is_caught_and_located(tmp_path, capsys):
    """Every trained tensor moved and the control did not -- the floor passes
    -- and one element is far past what the optimiser can do."""
    pre, post = pair(tmp_path, nudge=1e-5, damage=10.81)
    assert run(pre, post) == vcd.EXIT_FAILED
    out = capsys.readouterr().out
    assert "VERDICT: DAMAGED" in out
    assert K_PROJ in out
    assert "[2,3] = +10.81" in out
    assert UP_PROJ not in out.split("BEYOND THE CEILING")[1]


def test_a_move_just_inside_the_ceiling_passes_and_just_outside_fails(tmp_path):
    ceiling = 4 * 1e-6 * 17
    inside, _ = pair(tmp_path / "a", nudge=0.9 * ceiling)
    assert run(inside, tmp_path / "a" / "post") == 0
    outside, _ = pair(tmp_path / "b", nudge=1.5 * ceiling)
    assert run(outside, tmp_path / "b" / "post") == vcd.EXIT_FAILED


def test_too_few_steps_fails_towards_a_false_alarm_not_a_miss(tmp_path):
    pre, post = pair(tmp_path, nudge=2.5e-5)
    assert run(pre, post) == 0
    assert vcd.main([str(pre), str(post), "--lr", "1e-6", "--steps", "4"]) == vcd.EXIT_FAILED


def test_a_non_finite_value_is_damage_not_an_unmoved_tensor(tmp_path, capsys):
    """``max`` over a NaN is NaN and every comparison with it is False, so a
    NaN would otherwise read as both "did not move" and "within the ceiling"."""
    pre, post = pair(tmp_path, nudge=1e-5, damage=float("nan"))
    assert run(pre, post) == vcd.EXIT_FAILED
    assert "non-finite" in capsys.readouterr().out


@pytest.mark.parametrize("shape, reason", [
    ("identical", "did not move at all"),
    ("control moved", "the frozen control moved"),
    ("control absent", "is not in the tree"),
    ("tensor missing", "do not hold the same tensors"),
    ("shape changed", "do not hold the same tensors"),
])
def test_a_pair_that_cannot_be_vouched_for_exits_incomplete(tmp_path, capsys, shape, reason):
    """Each arm says which reason applies: "the control did not move" and
    "there was no control" must not read the same."""
    before = base()
    after = {name: array + (0.0 if name == FROZEN else 1e-5) for name, array in before.items()}
    if shape == "identical":
        after = {name: array.copy() for name, array in before.items()}
    elif shape == "control moved":
        after[FROZEN] = after[FROZEN] + 1e-5
    elif shape == "control absent":
        del before[FROZEN], after[FROZEN]
    elif shape == "tensor missing":
        del after[UP_PROJ]
    else:
        after[UP_PROJ] = np.full((2, 8), 0.1 + 1e-5, dtype=np.float32)
    pre = write_checkpoint(tmp_path / "pre", before)
    post = write_checkpoint(tmp_path / "post", after)
    assert run(pre, post) == vcd.EXIT_INCOMPLETE
    verdict_line = capsys.readouterr().out.split("VERDICT: ")[1]
    assert verdict_line.startswith("INCOMPLETE") and reason in verdict_line


def test_an_unmoved_trained_tensor_is_incomplete(tmp_path, capsys):
    before = base()
    after = {name: array.copy() for name, array in before.items()}
    after[K_PROJ] += 1e-5
    pre = write_checkpoint(tmp_path / "pre", before)
    post = write_checkpoint(tmp_path / "post", after)
    assert run(pre, post) == vcd.EXIT_INCOMPLETE
    assert "did not move at all" in capsys.readouterr().out


def test_damage_outranks_incompleteness():
    """A damaged pair that is also incomplete is reported as damaged: a reader
    who sees INCOMPLETE reruns the job rather than looking at the weights."""
    result = {"over": [{"tensor": "x"}], "non_finite": [], "only_in_pre": ["y"],
              "only_in_post": [], "mismatched": [], "frozen_delta": None,
              "unmoved": ["z"], "moved": 0}
    assert vcd.verdict(result)[0] == vcd.EXIT_FAILED


def test_the_three_outcomes_have_three_exit_codes_and_none_is_argparses():
    assert len({0, vcd.EXIT_FAILED, vcd.EXIT_INCOMPLETE}) == 3
    assert 2 not in {vcd.EXIT_FAILED, vcd.EXIT_INCOMPLETE}


@pytest.mark.parametrize("pre_shards, post_shards", [(0, 0), (1, 2), (2, 0)])
def test_single_file_and_differently_sharded_trees_pair_by_name(
    tmp_path, pre_shards, post_shards
):
    pre, post = pair(tmp_path, nudge=1e-5, pre_shards=pre_shards, post_shards=post_shards)
    assert run(pre, post) == 0


def test_a_directory_with_no_checkpoint_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        vcd.tensor_map(tmp_path / "empty")


def test_bfloat16_on_disk_is_widened_not_reinterpreted(tmp_path):
    values = np.array([[1.5, -0.25, 0.0, 2.0]], dtype=np.float32)
    raw = (values.view(np.uint32) >> 16).astype(np.uint16)
    path = tmp_path / "s.safetensors"
    blob = json.dumps({"w": {"dtype": "BF16", "shape": [1, 4],
                             "data_offsets": [0, raw.nbytes]}}).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw.tobytes())
    meta, offset = vcd.read_header(path)
    assert vcd.read_tensor(path, meta["w"], offset).tolist() == values.tolist()


def test_an_f64_tree_is_compared_in_f64(tmp_path):
    """A move below float32's resolution is still a move in an F64 tree."""
    def f64_tree(root, value):
        root.mkdir()
        tensors = {FROZEN: np.full((2,), 0.5), K_PROJ: np.full((2,), value)}
        header, buf, offset = {}, bytearray(), 0
        for name, array in tensors.items():
            raw = np.ascontiguousarray(array, dtype=np.float64).tobytes()
            header[name] = {"dtype": "F64", "shape": [2], "data_offsets": [offset, offset + len(raw)]}
            buf += raw
            offset += len(raw)
        blob = json.dumps(header).encode()
        (root / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(buf))
        return root

    pre = f64_tree(tmp_path / "pre", 1.0)
    post = f64_tree(tmp_path / "post", 1.0 + 1e-12)
    assert np.float32(1.0 + 1e-12) == np.float32(1.0), "the move is below f32 resolution"
    assert run(pre, post) == 0


def test_the_half_ulp_table_is_two_to_minus_the_significand_bits():
    for dtype, bits in (("F32", 24), ("F64", 53), ("BF16", 8), ("F16", 11)):
        assert vcd.HALF_ULP[dtype] == 2.0**-bits, dtype


def test_a_bf16_tree_gets_the_bf16_rounding_allowance():
    """Each in-place update rounds to the storage dtype, so a bf16 tree can move
    an element by far more than lr per step through rounding alone."""
    assert vcd.HALF_ULP["BF16"] > 1e4 * vcd.HALF_ULP["F32"]
