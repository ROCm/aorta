"""C8.T2: a partial iteration is invisible to readers; and the ledgers and the sealed evidence store."""

import errno
import os
import stat
from pathlib import Path

import pytest

from aorlta.episodes import tensorboard as tb
from aorlta.episodes.writer import (AppendLedger, EvidenceStore, IterationOutputs, LedgerError, complete_iterations,
                                    read_episodes, read_summary)
from aorlta.orchestrator.fakes import FakeRunSpec, build_fake_run
from aorlta.orchestrator.resume import last_committed
from aorlta.orchestrator.state import complete_checkpoints


def _readers_see(run_dir):
    return {"iterations": complete_iterations(run_dir), "tb": [n for n, _ in tb.tb_files(run_dir / "tb")],
            "committed": last_committed(run_dir)}


@pytest.mark.parametrize("moment", ["record.episodes", "record.summary", "ckpt.renamed"])
def test_c8_t2_until_the_commit_a_reader_sees_nothing_of_the_iteration(tmp_path, moment):
    seen = {}

    def point(name, iteration):
        if name == moment and iteration == 2:
            run_dir = r.cfg.run_dir
            seen["view"] = _readers_see(run_dir)
            with pytest.raises(FileNotFoundError):
                read_episodes(run_dir, 2)
            with pytest.raises(FileNotFoundError):
                read_summary(run_dir, 2)
            seen["partial"] = (run_dir / "iterations" / "iter-000002.partial").is_dir()
            seen["tb_names"] = sorted(p.name for p in (run_dir / "tb").iterdir())

    r = build_fake_run(FakeRunSpec(root=str(tmp_path / "runs"), iterations=2, eval_every=None), point=point)
    assert r.orchestrator.run()["status"] == "completed"
    assert seen["view"] == {"iterations": [1], "tb": [1], "committed": 1}
    assert seen["partial"] is True  # written, just not visible
    assert all("tfevents" not in n or n.endswith(".aorlta") for n in seen["tb_names"])
    assert _readers_see(r.cfg.run_dir) == {"iterations": [1, 2], "tb": [1, 2], "committed": 2}


def test_c8_t2_the_iteration_directory_is_renamed_after_its_checkpoint(tmp_path):
    order = []

    def point(name, iteration):
        if iteration == 1 and name in ("ckpt.renamed", "record.committed"):
            order.append((name, complete_checkpoints(r.cfg.run_dir / "ckpt"), complete_iterations(r.cfg.run_dir)))

    r = build_fake_run(FakeRunSpec(root=str(tmp_path / "runs"), iterations=1, eval_every=None), point=point)
    assert r.orchestrator.run()["status"] == "completed"
    assert order == [("ckpt.renamed", [1], []), ("record.committed", [1], [1])]


def _episode(n=1, k=0):
    return {"schema": "aorlta.episode/1", "source": "standin", "run_id": "r", "iteration": n,
            "chain_id": f"r/{n}/standin:p0/{k}", "prompt_id": "standin:p0", "k": k, "adapter_version": f"r:v{n - 1}",
            "engine_id": 0, "turns": [], "final_turn": -1, "end_reason": "engine_error", "candidate_sha256": None,
            "reward": None, "status": "unscored", "unscored_reason": "engine_error", "reward_version": None,
            "reward_code_sha256": None, "legs": {}, "plan": None, "audit": False, "advantage": None, "trained": False,
            "excluded_reason": "engine_error"}


def test_c8_t2_outputs_are_invisible_until_commit_and_a_leftover_partial_is_refused(tmp_path):
    out = IterationOutputs(tmp_path, 1)
    out.write_episodes([_episode()])
    assert complete_iterations(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        read_episodes(tmp_path, 1)
    with pytest.raises(FileExistsError):
        IterationOutputs(tmp_path, 1)  # a dead process's partial is never written into again
    out.commit()
    assert complete_iterations(tmp_path) == [1] and len(read_episodes(tmp_path, 1)) == 1
    with pytest.raises(FileExistsError):
        IterationOutputs(tmp_path, 1)


def test_the_writer_refuses_duplicate_chains_and_records_of_another_iteration(tmp_path):
    out = IterationOutputs(tmp_path, 1)
    with pytest.raises(ValueError):
        out.write_episodes([_episode(), _episode()])
    with pytest.raises(ValueError):
        out.write_episodes([_episode(n=2)])


def test_a_ledger_appends_skips_a_torn_last_line_and_truncates_by_iteration(tmp_path):
    led = AppendLedger(tmp_path / "x.jsonl")
    led.append([{"iteration": 1, "a": 1}, {"iteration": 2, "a": 2}])
    led.append([{"iteration": 3, "a": 3}])
    with open(led.path, "ab") as f:
        f.write(b'{"iteration": 4, "a"')  # a write the crash cut short
    assert [r["a"] for r in led.read()] == [1, 2, 3]
    assert led.truncate_after(2) == 2
    assert [r["a"] for r in led.read()] == [1, 2]
    with pytest.raises(LedgerError):
        led.append([{"a": 5}])  # every record names its iteration


def test_truncating_gives_a_kept_last_record_back_the_newline_a_crash_cut(tmp_path):
    led = AppendLedger(tmp_path / "x.jsonl")
    led.append([{"iteration": 1, "a": 1}, {"iteration": 2, "a": 2}])
    clean = os.stat(led.path).st_ino
    assert led.truncate_after(2) == 0 and os.stat(led.path).st_ino == clean  # nothing to change: not rewritten
    led.path.write_bytes(led.path.read_bytes()[:-1])  # the crash cut only the final newline
    assert led.truncate_after(2) == 0
    led.append([{"iteration": 3, "a": 3}])
    assert [r["a"] for r in led.read()] == [1, 2, 3]


def test_a_ledger_with_a_bad_line_before_the_end_is_refused(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"iteration": 1}\nnot json\n{"iteration": 2}\n')
    with pytest.raises(LedgerError):
        AppendLedger(p).read()


@pytest.mark.parametrize("tail", [b"\n", b""], ids=["a-line", "the-final-line"])
@pytest.mark.parametrize("bad", [b"[]", b"{}", b"5", b'{"iteration": "1"}', b'{"iteration": true}',
                                 b'{"iteration": -1}', b'{"iteration": 1.0}'])
def test_read_refuses_a_record_that_names_no_iteration_as_truncating_does(tmp_path, bad, tail):
    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"iteration": 1}\n' + bad + tail)
    with pytest.raises(LedgerError, match=":2: a record with no iteration"):
        AppendLedger(p).read()
    with pytest.raises(LedgerError, match=":2: a record with no iteration"):
        AppendLedger(p).truncate_after(1)
    assert p.read_bytes() == b'{"iteration": 1}\n' + bad + tail


def test_read_keeps_a_complete_final_record_that_only_lacks_its_newline(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"iteration": 1}\n{"iteration": 2}')
    assert AppendLedger(p).read() == [{"iteration": 1}, {"iteration": 2}]


def _fail_dir_fsync(monkeypatch, code):
    real = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(code, os.strerror(code))
        return real(fd)

    monkeypatch.setattr(os, "fsync", fsync)


@pytest.mark.parametrize("code", [errno.EIO, errno.ENOSPC])
def test_an_iteration_whose_directory_sync_fails_is_not_committed(tmp_path, monkeypatch, code):
    out = IterationOutputs(tmp_path, 1)
    out.write_episodes([_episode()])
    _fail_dir_fsync(monkeypatch, code)
    with pytest.raises(OSError) as exc:
        out.commit()
    assert exc.value.errno == code and complete_iterations(tmp_path) == [] and not out.committed


@pytest.mark.parametrize("code", [errno.EINVAL, errno.ENOTSUP])
def test_an_iteration_commits_on_a_filesystem_without_directory_fsync(tmp_path, monkeypatch, code):
    out = IterationOutputs(tmp_path, 1)
    out.write_episodes([_episode()])
    _fail_dir_fsync(monkeypatch, code)
    out.commit()
    assert complete_iterations(tmp_path) == [1] and len(read_episodes(tmp_path, 1)) == 1


def test_a_new_ledger_syncs_its_directory_so_a_power_loss_cannot_drop_the_file(tmp_path, monkeypatch):
    from aorlta.episodes import writer

    synced = []
    monkeypatch.setattr(writer, "_fsync_dir", lambda d: synced.append(Path(d)))
    ledger = AppendLedger(tmp_path / "ledgers" / "spans.jsonl")
    ledger.append([{"iteration": 1}])
    ledger.append([{"iteration": 2}])
    # the file's entry in ledgers/, and ledgers/'s in the run directory
    assert synced == [tmp_path / "ledgers", tmp_path] and [r["iteration"] for r in ledger.read()] == [1, 2]
    AppendLedger(ledger.path).append([{"iteration": 3}])  # a resumed run: the entries may never have been synced
    assert synced == [tmp_path / "ledgers", tmp_path] * 2


def test_the_evidence_store_syncs_its_root_and_the_run_directory_also_when_they_already_exist(tmp_path, monkeypatch):
    from aorlta.episodes import writer

    synced = []
    monkeypatch.setattr(writer, "_fsync_dir", lambda d: synced.append(Path(d)))
    EvidenceStore(tmp_path / "evidence")
    assert synced == [tmp_path / "evidence", tmp_path]  # blobs/ in evidence/, and evidence/ in the run directory
    synced.clear()
    EvidenceStore(tmp_path / "evidence")  # evidence/ exists: an interrupted attempt may not have synced it
    assert synced == [tmp_path / "evidence", tmp_path]


def test_the_run_directory_is_synced_before_an_iteration_commits_also_when_iterations_already_exists(
        tmp_path, monkeypatch):
    from aorlta.episodes import writer

    synced = []
    monkeypatch.setattr(writer, "_fsync_dir", lambda d: synced.append(Path(d)))
    IterationOutputs(tmp_path, 1).commit()
    assert synced[0] == tmp_path  # the new iterations/ entry, before anything in it is committed
    synced.clear()
    IterationOutputs(tmp_path, 2)  # iterations/ exists: an interrupted attempt may not have synced it
    assert synced == [tmp_path]


def test_a_new_evidence_shard_is_synced_with_its_parent_before_the_manifest_names_it(tmp_path, monkeypatch):
    from aorlta.episodes import writer

    events = []
    monkeypatch.setattr(writer, "_fsync_dir", lambda d: events.append(("sync", Path(d))))
    store = EvidenceStore(tmp_path / "evidence")
    real_append = store.manifest.append
    monkeypatch.setattr(store.manifest, "append", lambda recs: (events.append(("list", None)), real_append(recs)))
    blobs = tmp_path / "evidence" / "blobs"
    events.clear()  # the store's own syncs on opening are tested above

    def synced_before_listing():
        return {d for kind, d in events[:events.index(("list", None))]}

    sha = store.put(b"one", kind="stderr", iteration=1)
    shard = store.path_of(sha).parent
    assert synced_before_listing() == {shard, blobs}
    events.clear()
    store.put(b"one", kind="stderr", iteration=2)  # the same shard again in this process: no new sync
    assert synced_before_listing() == set()
    resumed = EvidenceStore(tmp_path / "evidence")  # a resumed run reusing the blob: the entry may never have been synced
    monkeypatch.setattr(resumed.manifest, "append", lambda recs: (events.append(("list", None)), real_append(recs)))
    events.clear()
    resumed.put(b"one", kind="stderr", iteration=3)
    assert synced_before_listing() == {shard, blobs}


def test_the_evidence_store_is_content_addressed_owner_only_and_verifiable(tmp_path):
    store = EvidenceStore(tmp_path / "evidence")
    sha = store.put(b"raw clang stderr", kind="stderr", iteration=1)
    assert store.put(b"raw clang stderr", kind="stderr", iteration=2) == sha  # stored once
    assert store.get(sha) == b"raw clang stderr" and store.verify() == []
    assert stat.S_IMODE(os.stat(tmp_path / "evidence").st_mode) == 0o700
    assert stat.S_IMODE(os.stat(store.path_of(sha)).st_mode) & 0o077 == 0
    store.path_of(sha).write_bytes(b"tampered")
    assert any("does not match" in p for p in store.verify())


def test_a_put_whose_listing_failed_lists_the_blob_when_it_is_retried(tmp_path, monkeypatch):
    store = EvidenceStore(tmp_path / "evidence")

    def disk_full(records):
        raise OSError("no space left on device")

    monkeypatch.setattr(store.manifest, "append", disk_full)
    with pytest.raises(OSError):
        store.put(b"raw", kind="stderr", iteration=1)
    monkeypatch.undo()
    sha = store.put(b"raw", kind="stderr", iteration=1)
    assert store.verify() == [] and store.prune_unlisted() == [] and store.get(sha) == b"raw"


def test_the_evidence_store_drops_blobs_of_uncommitted_iterations_at_resume(tmp_path):
    store = EvidenceStore(tmp_path / "evidence")
    keep = store.put(b"one", kind="stderr", iteration=1)
    drop = store.put(b"two", kind="stderr", iteration=2)
    store.manifest.truncate_after(1)
    assert store.prune_unlisted() == [drop]
    assert store.verify() == [] and store.get(keep) == b"one"
