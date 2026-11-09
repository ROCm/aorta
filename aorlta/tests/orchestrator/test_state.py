"""Prompt selection, run state and checkpoints (Section 6.6, "Checkpoint and resume")."""

import json
import random

import pytest

from aorlta.orchestrator.fakes import make_pool
from aorlta.orchestrator.state import (RUN_STATE_SCHEMA, CheckpointError, RunState, complete_checkpoints, cursor_for,
                                       epoch_order, pool_digest, prune_checkpoints, python_rng_state,
                                       restore_python_rng, select_prompts, verify_checkpoint, write_checkpoint)


def _ids(prompts):
    return [p.prompt_id for p in prompts]


def test_selection_is_deterministic_and_never_repeats_a_prompt_within_an_iteration():
    recs = make_pool(13).records()
    for n in range(1, 30):
        a, ca = select_prompts(recs, n, 7, 4)
        b, cb = select_prompts(recs, n, 7, 4)
        assert _ids(a) == _ids(b) and ca == cb
        assert len(set(_ids(a))) == 4


def test_each_epoch_visits_floor_m_over_p_blocks_and_the_remainder_waits():
    recs = make_pool(10).records()  # 2 iterations per epoch, 2 prompts left over each epoch
    epoch0 = _ids(select_prompts(recs, 1, 7, 4)[0]) + _ids(select_prompts(recs, 2, 7, 4)[0])
    assert len(set(epoch0)) == 8
    assert _ids(epoch_order(recs, 7, 0))[:8] == epoch0
    assert cursor_for(3, recs, 7, 4)["epoch"] == 1 and cursor_for(3, recs, 7, 4)["block"] == 0


def test_the_order_does_not_depend_on_the_pool_order_but_on_the_seed():
    recs = make_pool(12).records()
    shuffled = list(recs)
    random.Random(3).shuffle(shuffled)
    assert _ids(select_prompts(recs, 2, 7, 4)[0]) == _ids(select_prompts(shuffled, 2, 7, 4)[0])
    assert pool_digest(recs) == pool_digest(shuffled)
    assert _ids(epoch_order(recs, 7, 0)) != _ids(epoch_order(recs, 8, 0))


def test_the_cursor_names_the_pool_so_a_changed_pool_is_detected():
    a, b = make_pool(12).records(), make_pool(13).records()
    assert cursor_for(1, a, 7, 4)["pool_sha256"] != cursor_for(1, b, 7, 4)["pool_sha256"]


def test_a_pool_smaller_than_p_or_with_a_repeated_id_is_refused():
    with pytest.raises(ValueError):
        select_prompts(make_pool(3).records(), 1, 7, 4)
    recs = make_pool(5).records()
    with pytest.raises(ValueError):
        pool_digest(recs + recs[:1])


def _state(n: int) -> RunState:
    return RunState(schema=RUN_STATE_SCHEMA, run_id="r", source="standin", iteration=n, policy_version=f"r:v{n}",
                    config_sha256="0" * 64, next_cursor={"iteration": n + 1}, engines_slot=None,
                    history={"generation_s": [1.5], "scoring_s": [2.5]}, noop_streak=0, trainer_digest="1" * 64,
                    base_sha256=None, rng={"python_random": python_rng_state()})


def _save(blob: bytes):
    def save(d):
        (d / "latent.bin").write_bytes(blob)
    return save


def test_a_checkpoint_is_invisible_until_committed_and_then_verifies(tmp_path):
    root = tmp_path / "ckpt"
    pending = write_checkpoint(root, _state(3), _save(b"latent-3"))
    assert complete_checkpoints(root) == []  # written, fsynced, still .partial
    pending.commit()
    assert complete_checkpoints(root) == [3]
    state = verify_checkpoint(root / "iter-000003")
    assert state == _state(3).__class__.from_json(json.loads((root / "iter-000003" / "state.json").read_text()))
    assert state.iteration == 3 and state.history == {"generation_s": [1.5], "scoring_s": [2.5]}


@pytest.mark.parametrize("damage", ["change", "remove", "add", "manifest"])
def test_a_checkpoint_that_does_not_match_its_manifest_is_refused(tmp_path, damage):
    root = tmp_path / "ckpt"
    write_checkpoint(root, _state(2), _save(b"latent-2")).commit()
    d = root / "iter-000002"
    if damage == "change":
        (d / "trainer" / "latent.bin").write_bytes(b"latent-X")
    elif damage == "remove":
        (d / "trainer" / "latent.bin").unlink()
    elif damage == "add":
        (d / "trainer" / "extra.bin").write_bytes(b"?")
    else:
        (d / "MANIFEST.json").unlink()
    with pytest.raises(CheckpointError):
        verify_checkpoint(d)


def test_a_committed_checkpoint_is_never_overwritten_and_a_leftover_partial_is_refused(tmp_path):
    root = tmp_path / "ckpt"
    write_checkpoint(root, _state(1), _save(b"a")).commit()
    with pytest.raises(CheckpointError):
        write_checkpoint(root, _state(1), _save(b"b"))
    write_checkpoint(root, _state(2), _save(b"c"))  # left as .partial, as a crash would leave it
    with pytest.raises(FileExistsError):
        write_checkpoint(root, _state(2), _save(b"d"))


def test_pruning_keeps_the_newest_and_goes_through_a_partial_name(tmp_path):
    root = tmp_path / "ckpt"
    for n in (1, 2, 3, 4):
        write_checkpoint(root, _state(n), _save(bytes([n]))).commit()
    assert prune_checkpoints(root, 2) == ["iter-000001", "iter-000002"]
    assert complete_checkpoints(root) == [3, 4]
    assert not list(root.glob("*.partial"))


def test_the_python_rng_state_round_trips():
    saved = python_rng_state()
    a = [random.random() for _ in range(5)]
    restore_python_rng(json.loads(json.dumps(saved)))
    assert [random.random() for _ in range(5)] == a
