"""The file half of a resume (Section 6.6): what `prepare_resume` keeps, removes and rewrites."""

import hashlib
import json
import shutil

import pytest

from aorlta.episodes import tensorboard as tb
from aorlta.episodes.writer import AppendLedger, LedgerError, complete_iterations
from aorlta.orchestrator.fakes import FakeRunSpec, build_fake_run
from aorlta.orchestrator.resume import prepare_resume, remove_partials
from aorlta.orchestrator.state import CheckpointError, complete_checkpoints


@pytest.fixture
def done_run(tmp_path):
    """A finished 3-iteration fake run (keep_ckpts 3 so every checkpoint is still there)."""
    r = build_fake_run(FakeRunSpec(root=str(tmp_path / "runs"), iterations=3, eval_every=None, keep_ckpts=3))
    assert r.orchestrator.run()["status"] == "completed"
    return r.cfg.run_dir


def test_every_partial_file_and_directory_is_removed(tmp_path):
    (tmp_path / "a" / "b.partial" / "c").mkdir(parents=True)
    (tmp_path / "a" / "b.partial" / "c" / "d.json").write_text("{}")
    (tmp_path / "a" / "e.json.partial").write_text("{")
    (tmp_path / "a" / "keep.json").write_text("{}")
    assert sorted(remove_partials(tmp_path)) == ["a/b.partial", "a/e.json.partial"]
    assert [p.name for p in (tmp_path / "a").iterdir()] == ["keep.json"]


def test_a_finished_run_resumes_at_its_last_iteration_untouched(done_run):
    before = sorted(str(p) for p in done_run.rglob("*"))
    rp = prepare_resume(done_run, keep_ckpts=3)
    assert rp.iteration == 3 and rp.state.iteration == 3 and rp.removed == []
    assert sorted(str(p) for p in done_run.rglob("*")) == before


def test_a_checkpoint_whose_iteration_never_committed_is_discarded(done_run):
    shutil.rmtree(done_run / "iterations" / "iter-000003")  # renamed checkpoint, crash before the commit
    tb.remove_after(done_run / "tb", 2)
    rp = prepare_resume(done_run, keep_ckpts=3)
    assert rp.iteration == 2 and "ckpt/iter-000003" in rp.removed
    assert complete_checkpoints(done_run / "ckpt") == [1, 2] and complete_iterations(done_run) == [1, 2]


def test_partial_outputs_of_the_incomplete_iteration_are_removed(done_run):
    (done_run / "iterations" / "iter-000003").rename(done_run / "iterations" / "iter-000003.partial")
    (done_run / "ckpt" / "iter-000003").rename(done_run / "ckpt" / "iter-000003.partial")
    tb.remove_after(done_run / "tb", 2)
    rp = prepare_resume(done_run, keep_ckpts=3)
    assert rp.iteration == 2
    assert not list(done_run.rglob("*.partial"))


def test_a_committed_iteration_whose_tensorboard_file_was_not_yet_written_gets_it_back(done_run):
    want = tb.read_scalars(done_run / "tb")[3]
    (done_run / "tb" / tb.event_file_name(3)).unlink()
    rp = prepare_resume(done_run, keep_ckpts=3)
    assert rp.iteration == 3 and tb.read_scalars(done_run / "tb")[3] == want


def test_tensorboard_files_and_ledger_records_of_later_iterations_are_cut(done_run):
    shutil.rmtree(done_run / "iterations" / "iter-000003")
    led = AppendLedger(done_run / "ledgers" / "asan.jsonl")
    led.append([{"iteration": 2, "request": 1}, {"iteration": 3, "request": 2}])
    rp = prepare_resume(done_run, keep_ckpts=3)
    assert rp.iteration == 2
    assert [n for n, _ in tb.tb_files(done_run / "tb")] == [1, 2]
    assert [r["request"] for r in led.read()] == [1]


def test_a_committed_checkpoint_that_fails_its_manifest_stops_the_resume(done_run):
    state = done_run / "ckpt" / "iter-000003" / "trainer" / "trainer_state.json"
    state.write_text(json.dumps({"tampered": True}))
    with pytest.raises(CheckpointError):
        prepare_resume(done_run, keep_ckpts=3)


def test_committed_outputs_without_their_checkpoint_stop_the_resume(done_run):
    shutil.rmtree(done_run / "ckpt" / "iter-000003")
    with pytest.raises(RuntimeError):
        prepare_resume(done_run, keep_ckpts=3)


def test_checkpoints_beyond_keep_are_pruned(done_run):
    prepare_resume(done_run, keep_ckpts=2)
    assert complete_checkpoints(done_run / "ckpt") == [2, 3]


def test_a_bad_line_before_a_ledgers_end_stops_the_resume_before_anything_is_removed(done_run):
    """Only a torn final line is a crash's; a bad line inside a ledger is damage to committed history. Cutting it would
    rewrite that history, and, in the evidence manifest, let the prune delete the blob whose entry was damaged."""
    shutil.rmtree(done_run / "iterations" / "iter-000003")
    (done_run / "ckpt" / "iter-000003.partial").mkdir()
    manifest = done_run / "evidence" / "MANIFEST.jsonl"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_bytes(b'{"iteration": 1, "sha256": "a", "size": 1, "kind": "k"}\n{"iteration": 2, "sha2\n'
                         b'{"iteration": 3, "sha256": "b", "size": 1, "kind": "k"}\n')
    before = manifest.read_bytes()
    with pytest.raises(LedgerError, match="MANIFEST.jsonl:2"):
        prepare_resume(done_run, keep_ckpts=3)
    assert manifest.read_bytes() == before and (done_run / "ckpt" / "iter-000003.partial").is_dir()


def _files(run_dir):
    return {p.relative_to(run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in run_dir.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name, data", [
    ("ledgers/b.jsonl", b'{"iteration": 1}\n[]\n{"iteration": 2}\n'),
    ("ledgers/b.jsonl", b'{"iteration": 1}\n{}'),
    ("evidence/MANIFEST.jsonl", b'{"iteration": 1, "sha256": "a", "size": 1, "kind": "k"}\n'
                                b'{"iteration": -1, "sha256": "b", "size": 1, "kind": "k"}\n'),
], ids=["a-line", "the-final-line", "the-manifest"])
def test_a_record_that_names_no_iteration_stops_the_resume_before_anything_is_removed(done_run, name, data):
    shutil.rmtree(done_run / "iterations" / "iter-000003")
    (done_run / "ckpt" / "iter-000003.partial").mkdir()
    (done_run / "ckpt" / "iter-000003.partial" / "trainer_state.json").write_text("{")
    AppendLedger(done_run / "ledgers" / "a.jsonl").append([{"iteration": 2}, {"iteration": 3}])
    (done_run / name).parent.mkdir(exist_ok=True)
    (done_run / name).write_bytes(data)
    before = _files(done_run)
    with pytest.raises(LedgerError, match=name.rsplit("/", 1)[1] + ":2: a record with no iteration"):
        prepare_resume(done_run, keep_ckpts=3)
    assert _files(done_run) == before


def test_truncate_after_drops_a_torn_final_line_and_refuses_a_bad_line_anywhere_else(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_bytes(b'{"iteration": 1}\n{"iteration": 2}\n{"iterat')
    assert AppendLedger(path).truncate_after(1) == 2 and path.read_bytes() == b'{"iteration": 1}\n'
    for bad in (b'{"iteration": 1}\nnot json\n{"iteration": 2}\n', b'{"iteration": 1}\n[]\n{"iteration": 2}\n'):
        path.write_bytes(bad)
        with pytest.raises(LedgerError, match=":2:"):
            AppendLedger(path).truncate_after(1)
        assert path.read_bytes() == bad


def test_a_resume_leaves_the_external_evaluators_outbox_alone(done_run):
    """The evaluator may still be writing a result when the orchestrator restarts: its `.partial` is its own."""
    outbox = done_run / "eval" / "outbox"
    outbox.mkdir(parents=True)
    (outbox / "req-1.json.partial").write_text("{")
    (done_run / "stray.json.partial").write_text("{")
    rp = prepare_resume(done_run, keep_ckpts=3)
    assert (outbox / "req-1.json.partial").is_file() and not (done_run / "stray.json.partial").exists()
    assert rp.removed == ["stray.json.partial"]
