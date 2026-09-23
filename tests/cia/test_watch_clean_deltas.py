"""The half of the corpus the system never recorded.

``write_bundle`` runs only after Watch alerts. So every full-length log delta
on disk belongs to a failure, and the only healthy deltas recorded anywhere are
the ``watchdog_ok`` excerpts the events file caps at 500 characters -- against
bundles of up to 4000. A ``watch_healthy`` corpus built from those two has a
positive class four thousand characters long and a negative class five hundred,
and a classifier can score well on it without reading a word, by measuring
length. ``aorta.laya.corpus.watch`` warns about exactly this on every build.

That is why the clean-gate cannot be measured today, and it is a data problem
rather than a model problem: no weights are needed to fix it. Shadow mode is
already handling every healthy delta, so the archive rides along with it.

Everything here is about the bounds. A new artifact that holds log text has to
be off by default, capped on disk, scrubbed the way Watch scrubs anything it
sends to a provider, and truncated at the same 4000 characters as the class it
will be compared against -- otherwise the fix reintroduces the bug it exists to
remove, in the other direction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import aorta.cia.watch.poll as poll_mod
from aorta.cia.watch.bundle_writer import (
    CLEAN_DELTA_CHARS,
    CLEAN_DELTA_FILE,
    archive_clean_delta,
)
from aorta.cia.watch.cursors import load_cursors
from aorta.cia.watch.poll import poll_jobs
from aorta.cia.watch.watcher import LayaObservation

_CONFIG = "src/aorta/cia/watch/watch_config.yaml"


def _archive(job_dir: Path, delta: str = "step 41 loss 0.31\n", **kwargs) -> bool:
    defaults = {
        "job_id": "cia-aaa",
        "healthy": True,
        "signal": "WATCH_CLEAN",
        "confidence": 0.88,
        "limit_bytes": 64_000,
    }
    return archive_clean_delta(job_dir, delta=delta, **{**defaults, **kwargs})


def _records(job_dir: Path) -> list[dict]:
    path = job_dir / CLEAN_DELTA_FILE
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestItIsOffUnlessAskedFor:
    def test_a_zero_budget_writes_nothing(self, tmp_path):
        assert _archive(tmp_path, limit_bytes=0) is False
        assert not (tmp_path / CLEAN_DELTA_FILE).exists()

    def test_the_shipped_config_asks_for_nothing(self, repo_root):
        config = yaml.safe_load((repo_root / _CONFIG).read_text(encoding="utf-8"))
        assert config["watch"]["laya"]["shadow_archive_bytes"] == 0

    def test_a_negative_budget_is_off_rather_than_unbounded(self, tmp_path):
        assert _archive(tmp_path, limit_bytes=-1) is False


class TestTheBoundsOnDisk:
    def test_it_stops_at_the_cap(self, tmp_path):
        """A cap checked before each append, so the file cannot grow with the run."""
        for index in range(200):
            _archive(tmp_path, delta=f"step {index} " + "x" * 900, limit_bytes=4_000)

        size = (tmp_path / CLEAN_DELTA_FILE).stat().st_size
        # The cap plus at most one record: the check is before the write, which
        # is what makes it cheap and what makes the overshoot bounded and
        # stated rather than discovered.
        assert 4_000 <= size <= 4_000 + CLEAN_DELTA_CHARS + 1_000

    def test_one_delta_is_truncated_at_the_length_write_bundle_uses(self, tmp_path):
        """The two files are the two classes of one corpus.

        ``write_bundle(job, job_dir, evidence or new_content[:4000], signal)``
        is the other one. A different cap here would put a length difference
        between the classes before any content difference, which is the whole
        problem this archive exists to remove.
        """
        _archive(tmp_path, delta="y" * 9_000)
        record = _records(tmp_path)[0]

        assert len(record["delta"]) == CLEAN_DELTA_CHARS
        assert record["truncated"] is True
        assert record["delta_chars"] == 9_000

    def test_the_prefix_is_the_one_write_bundle_would_have_kept(self, tmp_path):
        """Truncated before scrubbing, not after, so the two prefixes correspond."""
        delta = "".join(f"step {i} loss 0.3\n" for i in range(2_000))
        _archive(tmp_path, delta=delta)

        assert _records(tmp_path)[0]["delta"] == delta[:CLEAN_DELTA_CHARS]

    def test_a_short_delta_is_not_marked_truncated(self, tmp_path):
        _archive(tmp_path, delta="step 1 loss 0.4\n")
        assert _records(tmp_path)[0]["truncated"] is False


class TestItIsNotAPrivacyRegression:
    def test_paths_and_addresses_are_scrubbed(self, tmp_path):
        """The same gate Watch's outbound LLM traffic goes through.

        ``redact`` is what ``RedactingLM`` applies to everything Watch, Autopsy
        and Launch discovery send to a provider. A file written to be copied off
        the node and trained on has at least that claim on it.
        """
        _archive(
            tmp_path,
            delta="step 4 writing /home/alice/runs/job7/train.log from 10.1.2.3\n",
        )
        kept = _records(tmp_path)[0]["delta"]

        assert "/home/alice" not in kept
        assert "10.1.2.3" not in kept
        assert "<PATH:" in kept and "<IPV4:" in kept

    def test_every_record_says_it_was_redacted(self, tmp_path):
        """Because ``write_bundle`` does not redact, and the two get compared.

        A corpus pairing these records against raw bundles has one class
        carrying ``<PATH:0>`` markers and the other not, which is the same free
        signal as the length difference in the other direction. The flag is how
        a consumer knows it has to scrub the bundle side to match.
        """
        _archive(tmp_path)
        assert _records(tmp_path)[0]["redacted"] is True

    def test_a_scrubber_that_cannot_load_writes_nothing(self, tmp_path, monkeypatch, capsys):
        """Refusing is the same answer ``aorta.cia.llm.redact`` gives. Not a fallback."""
        import aorta.cia.llm as llm

        def _broken(_text: str) -> str:
            raise llm.RedactionUnavailable("no scrubber")

        monkeypatch.setattr(llm, "redact", _broken)

        assert _archive(tmp_path) is False
        assert not (tmp_path / CLEAN_DELTA_FILE).exists()
        assert "could not scrub" in capsys.readouterr().out


class TestWhatOneRecordSays:
    def test_the_verdict_travels_with_the_delta(self, tmp_path):
        """Otherwise the label has to be re-derived from nothing later."""
        _archive(tmp_path, healthy=True, signal="WATCH_CLEAN", confidence=0.91)
        record = _records(tmp_path)[0]

        assert record["healthy"] is True
        assert record["signal"] == "WATCH_CLEAN"
        assert record["confidence"] == pytest.approx(0.91)
        assert record["job_id"] == "cia-aaa"

    def test_an_unhealthy_delta_that_stayed_quiet_is_kept_too(self, tmp_path):
        """These are the most interesting examples in the corpus, not noise.

        Watch can decide a delta is unhealthy and stay quiet because it is not
        sure enough. Filtering to ``healthy`` here would drop exactly the
        examples nearest the boundary the gate has to sit on.
        """
        _archive(tmp_path, healthy=False, signal="WATCH_HANG", confidence=0.4)
        record = _records(tmp_path)[0]

        assert record["healthy"] is False
        assert record["signal"] == "WATCH_HANG"

    def test_a_laya_observation_rides_along_when_there_is_one(self, tmp_path):
        observation = LayaObservation(
            model_id="laya-typed-decisions",
            clean_probability=0.71,
            clean_threshold=0.9,
            gated=False,
            vetoed=False,
            signal="WATCH_HANG",
            signal_probability=0.5,
        )
        _archive(tmp_path, laya=observation.as_event_fields())

        assert _records(tmp_path)[0]["laya"]["model_id"] == "laya-typed-decisions"

    def test_it_appends_rather_than_replacing(self, tmp_path):
        _archive(tmp_path, delta="first\n")
        _archive(tmp_path, delta="second\n")

        assert [r["delta"] for r in _records(tmp_path)] == ["first\n", "second\n"]

    def test_it_lands_beside_the_events_and_not_inside_the_bundle(self, tmp_path):
        """A bundle is what Autopsy reads about a failure, and this is not one."""
        _archive(tmp_path)

        assert (tmp_path / CLEAN_DELTA_FILE).is_file()
        assert not (tmp_path / "bundle").exists()


# ── the poll loop's side of it ────────────────────────────────────────────


def _write_job(root: Path, job_id: str, content: str) -> Path:
    """A job on disk, laid out the way ``scan_active_jobs`` expects."""
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    log = job_dir / "watch.log"
    log.write_text(content)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "node": "node1",
                "recipe": "a recipe",
                "launched_at": "2026-01-01T00:00:00Z",
                "log_path": str(log),
                "aorta_output": str(job_dir / "aorta"),
                "status": "running",
                "schema_version": "0.1",
                "watch_files": [str(log)],
            }
        )
    )
    return job_dir


_OBSERVATION = LayaObservation(
    model_id="laya-typed-decisions@cpu",
    clean_probability=0.8123456,
    clean_threshold=0.9,
    gated=False,
    vetoed=False,
    signal="WATCH_HANG",
    signal_probability=0.44,
)


@pytest.fixture
def watching(monkeypatch):
    """A poll loop whose assessment is a fixture and whose Autopsy is recorded.

    Mirrors the fake watcher every other poll test here uses: the loop is the
    thing under test, so the tier that answers is a stub with attributes.
    """

    def _run(jobs_root: Path, *, healthy: bool, observation=None, config=None):
        class Pred:
            signal = "WATCH_CLEAN" if healthy else "WATCH_NUMERIC_NAN"
            confidence = 0.95
            evidence = "none" if healthy else "loss=nan"
            assessment = "looks fine" if healthy else "non-finite loss"

        Pred.healthy = healthy
        if observation is not None:
            Pred.laya = observation

        class FakeWatcher:
            def forward(self, **kwargs):
                return Pred()

        monkeypatch.setattr(poll_mod, "LogWatcher", lambda *a, **k: FakeWatcher())
        monkeypatch.setattr(poll_mod, "LogFinder", lambda *a, **k: object())
        monkeypatch.setattr(poll_mod.time, "sleep", lambda _: None)
        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy",
            lambda bundle, job, jobs_root, stop=None: None,
        )
        path = jobs_root / "watch_config.yaml"
        path.write_text(yaml.safe_dump(config or {}), encoding="utf-8")
        poll_jobs(jobs_root, max_rounds=1, config_path=path)

    return _run


class TestThePollLoopArchivesOnlyWhatItWasAskedTo:
    def test_nothing_is_written_with_the_knob_at_zero(self, tmp_path, watching):
        job_dir = _write_job(tmp_path, "cia-aaa", "step 41 loss 0.31\n")
        watching(tmp_path, healthy=True, config={"watch": {"laya": {"shadow_archive_bytes": 0}}})

        assert not (job_dir / CLEAN_DELTA_FILE).exists()

    def test_a_delta_that_did_not_alert_is_kept(self, tmp_path, watching):
        job_dir = _write_job(tmp_path, "cia-aaa", "step 41 loss 0.31\n")
        watching(
            tmp_path,
            healthy=True,
            config={"watch": {"laya": {"shadow_archive_bytes": 32_000}}},
        )
        records = _records(job_dir)

        assert len(records) == 1
        assert "step 41 loss 0.31" in records[0]["delta"]
        assert records[0]["healthy"] is True

    def test_an_alerting_delta_goes_to_the_bundle_and_not_here(self, tmp_path, watching):
        """The two writers are exclusive, which is what makes them two classes."""
        job_dir = _write_job(tmp_path, "cia-aaa", "step 41 loss nan\n")
        watching(
            tmp_path,
            healthy=False,
            config={"watch": {"laya": {"shadow_archive_bytes": 32_000}}},
        )

        assert (job_dir / "bundle" / "logs" / "watch.stderr.log").is_file()
        assert not (job_dir / CLEAN_DELTA_FILE).exists()

    def test_the_cursor_is_still_committed_when_the_archive_is_full(
        self, tmp_path, watching
    ):
        """The archive is a bystander: a full one must not make Watch re-read bytes."""
        job_dir = _write_job(tmp_path, "cia-aaa", "step 41 loss 0.31\n")
        already_full = job_dir / CLEAN_DELTA_FILE
        already_full.write_text("x" * 100, encoding="utf-8")
        watching(
            tmp_path,
            healthy=True,
            config={"watch": {"laya": {"shadow_archive_bytes": 50}}},
        )

        assert already_full.read_text(encoding="utf-8") == "x" * 100, "the cap did not hold"
        assert load_cursors(job_dir), "the delta was left to be read again"


class TestTheShadowEvent:
    def test_it_is_written_beside_the_verdict_under_its_own_type(self, tmp_path, watching):
        """A distinct ``event_type`` so nothing reading for verdicts finds a measurement."""
        job_dir = _write_job(tmp_path, "cia-aaa", "step 41 loss 0.31\n")
        watching(tmp_path, healthy=True, observation=_OBSERVATION)

        events = [
            json.loads(line)
            for line in (job_dir / "events.jsonl").read_text().splitlines()
        ]
        types = [event["event_type"] for event in events]

        assert types == ["watchdog_ok", "watchdog_shadow"]

    def test_nothing_is_written_when_the_tier_did_not_run(self, tmp_path, watching):
        job_dir = _write_job(tmp_path, "cia-aaa", "step 41 loss 0.31\n")
        watching(tmp_path, healthy=True)

        events = (job_dir / "events.jsonl").read_text().splitlines()
        assert len(events) == 1

    def test_it_names_the_checkpoint_that_answered(self, tmp_path, watching):
        """Decision 22, rule 2: inline, not beside the run."""
        job_dir = _write_job(tmp_path, "cia-aaa", "step 41 loss 0.31\n")
        watching(tmp_path, healthy=True, observation=_OBSERVATION)

        shadow = json.loads((job_dir / "events.jsonl").read_text().splitlines()[1])
        assert shadow["model_id"] == "laya-typed-decisions@cpu"
        assert shadow["clean_probability"] == 0.8123
        assert shadow["gated"] is False

    def test_it_repeats_the_verdict_that_was_actually_used(self, tmp_path, watching):
        """So one line can be scored without a positional join into an append-only file."""
        job_dir = _write_job(tmp_path, "cia-aaa", "step 41 loss 0.31\n")
        watching(tmp_path, healthy=True, observation=_OBSERVATION)

        shadow = json.loads((job_dir / "events.jsonl").read_text().splitlines()[1])
        assert shadow["watch_signal"] == "WATCH_CLEAN"
        assert shadow["watch_healthy"] is True
        assert shadow["signal"] == "WATCH_HANG", "Laya's own answer holds the usual field"

    def test_it_quotes_no_log_text(self, tmp_path, watching):
        """A tier that only scored the delta has nothing to quote out of it."""
        job_dir = _write_job(tmp_path, "cia-aaa", "step 41 loss 0.31\n")
        watching(tmp_path, healthy=True, observation=_OBSERVATION)

        shadow = json.loads((job_dir / "events.jsonl").read_text().splitlines()[1])
        assert shadow["excerpt"] == ""

    def test_an_alerting_round_still_records_the_shadow(self, tmp_path, watching):
        """The comparison needs the failures most of all; a gate is not scored on clean traffic."""
        job_dir = _write_job(tmp_path, "cia-aaa", "step 41 loss nan\n")
        watching(tmp_path, healthy=False, observation=_OBSERVATION)

        types = [
            json.loads(line)["event_type"]
            for line in (job_dir / "events.jsonl").read_text().splitlines()
        ]
        assert types == ["watchdog_alert", "watchdog_shadow"]
