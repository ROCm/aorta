"""Writer / staging / tarball tests for ``aorta bundle`` (issue #196)."""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import pytest

from aorta.bundle import (
    BundleIOError,
    EmptyRunDirError,
    IdentityRedactor,
    Manifest,
    NoTicketError,
    RedactionCounts,
    Redactor,
    RunDirNotFoundError,
    UnsafeSymlinkError,
    bundle_run_dir,
    resolve_ticket,
)
from aorta.bundle.manifest import MANIFEST_FILENAME
from aorta.bundle.writer import _bundle_timestamp, stage_run_dir, write_tarball

# --- resolve_ticket --------------------------------------------------------


def test_resolve_ticket_uses_flag_when_provided(tmp_path):
    run_dir = tmp_path / "TKT-XYZ"
    run_dir.mkdir()
    assert resolve_ticket(run_dir, "OVERRIDE-1") == "OVERRIDE-1"


def test_resolve_ticket_infers_from_basename(tmp_path):
    run_dir = tmp_path / "TKT-1"
    run_dir.mkdir()
    assert resolve_ticket(run_dir, None) == "TKT-1"


def test_resolve_ticket_strips_whitespace_then_falls_back(tmp_path):
    run_dir = tmp_path / "TKT-2"
    run_dir.mkdir()
    assert resolve_ticket(run_dir, "   ") == "TKT-2"


def test_resolve_ticket_refuses_no_ticket_slug(tmp_path):
    run_dir = tmp_path / "_no_ticket_"
    run_dir.mkdir()
    with pytest.raises(NoTicketError) as exc:
        resolve_ticket(run_dir, None)
    assert exc.value.run_dir == run_dir


def test_resolve_ticket_no_ticket_slug_with_flag_override_uses_flag(tmp_path, caplog):
    run_dir = tmp_path / "_no_ticket_"
    run_dir.mkdir()
    with caplog.at_level("WARNING"):
        out = resolve_ticket(run_dir, "RESCUED-1")
    assert out == "RESCUED-1"
    assert any("_no_ticket_" in r.message for r in caplog.records)


def test_resolve_ticket_mismatched_flag_warns_but_proceeds(tmp_path, caplog):
    run_dir = tmp_path / "TKT-1"
    run_dir.mkdir()
    with caplog.at_level("WARNING"):
        out = resolve_ticket(run_dir, "TKT-2")
    assert out == "TKT-2"
    assert any("does not match run-dir" in r.message for r in caplog.records)


# --- _validate_run_dir / bundle_run_dir error path -------------------------


def test_bundle_run_dir_missing_path_raises(tmp_path):
    missing = tmp_path / "no-such-dir"
    with pytest.raises(RunDirNotFoundError):
        bundle_run_dir(missing)


def test_bundle_run_dir_file_path_raises(tmp_path):
    f = tmp_path / "not-a-dir"
    f.write_text("nope")
    with pytest.raises(RunDirNotFoundError):
        bundle_run_dir(f)


def test_bundle_run_dir_empty_tree_raises(empty_run_dir):
    with pytest.raises(EmptyRunDirError) as exc:
        bundle_run_dir(empty_run_dir)
    assert exc.value.run_dir == empty_run_dir.resolve()


def test_bundle_run_dir_no_ticket_basename_raises(no_ticket_run_dir):
    with pytest.raises(NoTicketError):
        bundle_run_dir(no_ticket_run_dir)


def test_bundle_run_dir_no_ticket_basename_with_flag_succeeds(no_ticket_run_dir, tmp_path):
    out = tmp_path / "bundle.tar.gz"
    written = bundle_run_dir(no_ticket_run_dir, ticket="RESCUED-1", output=out)
    assert written == out.resolve()
    assert written.is_file()


# --- happy path: stage_run_dir / write_tarball ----------------------------


def test_stage_run_dir_copies_every_file_and_writes_manifest(synthetic_run_dir, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    manifest = stage_run_dir(
        synthetic_run_dir,
        staging,
        "TKT-1-bundle",
        redactor=IdentityRedactor(),
        ticket="TKT-1",
        aorta_version="0.2.0",
    )
    bundle_root = staging / "TKT-1-bundle"
    assert (bundle_root / MANIFEST_FILENAME).is_file()
    for f in manifest.files:
        assert (bundle_root / f.path).is_file()
    # Identity redactor: every count is 0 and bytes_in == bytes_out.
    for f in manifest.files:
        assert f.env_keys_removed == 0
        assert f.paths_rewritten == 0
        assert f.ips_rewritten == 0
        assert f.bytes_in == f.bytes_out


def test_stage_run_dir_manifest_excludes_itself_and_lockfile(synthetic_run_dir, tmp_path):
    """Defensive: pre-existing manifest.json + lockfile in the source
    are not re-bundled. Otherwise re-bundling an extracted bundle
    would double-count and a stale lockfile would survive."""
    (synthetic_run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (synthetic_run_dir / ".aorta-probe.lock").write_text("{}", encoding="utf-8")

    staging = tmp_path / "staging"
    staging.mkdir()
    manifest = stage_run_dir(
        synthetic_run_dir,
        staging,
        "TKT-1-bundle",
        redactor=IdentityRedactor(),
        ticket="TKT-1",
        aorta_version="0.2.0",
    )
    paths = {f.path for f in manifest.files}
    assert "manifest.json" not in paths
    assert ".aorta-probe.lock" not in paths


def test_write_tarball_round_trip(synthetic_run_dir, tmp_path):
    """Acceptance criterion 6: happy-path tarball round-trip."""
    staging = tmp_path / "staging"
    staging.mkdir()
    manifest = stage_run_dir(
        synthetic_run_dir,
        staging,
        "TKT-1-bundle",
        redactor=IdentityRedactor(),
        ticket="TKT-1",
        aorta_version="0.2.0",
    )
    out = tmp_path / "TKT-1.tar.gz"
    written = write_tarball(staging, "TKT-1-bundle", out)
    assert written == out.absolute()
    assert written.is_file()

    with tempfile.TemporaryDirectory() as extract_root:
        extract = Path(extract_root)
        with tarfile.open(written, "r:gz") as tar:
            tar.extractall(extract)  # noqa: S202 - test fixture, controlled input
        # Every file recorded in the manifest is present in the extracted
        # tree under the bundle-root directory.
        bundle_root = extract / "TKT-1-bundle"
        assert (bundle_root / MANIFEST_FILENAME).is_file()
        for f in manifest.files:
            assert (bundle_root / f.path).is_file()
        # Manifest is the tarball trailer.
        with tarfile.open(written, "r:gz") as tar:
            names = tar.getnames()
        assert names[-1] == f"TKT-1-bundle/{MANIFEST_FILENAME}"


# --- originals untouched (acceptance criterion 5) -------------------------


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


def test_bundle_run_dir_does_not_modify_source(synthetic_run_dir, tmp_path):
    """Acceptance criterion 5: originals untouched.

    Snapshot every file's bytes before and after bundling; the
    tree must be byte-identical.
    """
    before = _snapshot_tree(synthetic_run_dir)
    out = tmp_path / "out.tar.gz"
    bundle_run_dir(synthetic_run_dir, output=out)
    after = _snapshot_tree(synthetic_run_dir)
    assert before == after


def test_bundle_run_dir_default_output_in_cwd(synthetic_run_dir, tmp_path, monkeypatch):
    """Default ``--output`` lands a ``<ticket>-<ts>.tar.gz`` in CWD."""
    monkeypatch.chdir(tmp_path)
    out = bundle_run_dir(synthetic_run_dir)
    assert out.parent == tmp_path.resolve()
    assert out.name.startswith("TKT-1-")
    assert out.suffix == ".gz"
    assert out.is_file()


def test_bundle_run_dir_output_directory_drops_default_filename(synthetic_run_dir, tmp_path):
    target = tmp_path / "bundles"
    target.mkdir()
    out = bundle_run_dir(synthetic_run_dir, output=target)
    assert out.parent == target.resolve()
    assert out.name.startswith("TKT-1-")


# --- redactor injection point --------------------------------------------


class _RecordingRedactor(Redactor):
    """Sentinel redactor: records every (src, dst) pair the writer
    calls scrub_file with so we can assert the writer routes
    EVERY source file through the redactor.

    Also lets us prove the redactor's per-file count return is
    plumbed into the manifest verbatim (so when Phase 3 of #188's
    real RedactingRedactor lands, its counts will land in the
    manifest unchanged).
    """

    kind = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    def scrub_file(self, src: Path, dst: Path) -> RedactionCounts:
        self.calls.append((src, dst))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        size = dst.stat().st_size
        return RedactionCounts(
            env_keys_removed=1,
            paths_rewritten=2,
            ips_rewritten=3,
            bytes_in=size,
            bytes_out=size,
        )


def test_bundle_run_dir_routes_every_source_file_through_redactor(synthetic_run_dir, tmp_path):
    redactor = _RecordingRedactor()
    out = bundle_run_dir(synthetic_run_dir, output=tmp_path / "out.tar.gz", redactor=redactor)
    assert out.is_file()
    # Every file in the source tree (minus skipped basenames) was scrubbed.
    bundled_sources = {src for src, _ in redactor.calls}
    expected = {
        p
        for p in synthetic_run_dir.rglob("*")
        if p.is_file() and p.name not in {".aorta-probe.lock", "manifest.json"}
    }
    assert bundled_sources == expected


def test_bundle_run_dir_propagates_redactor_counts_into_manifest(synthetic_run_dir, tmp_path):
    """Phase 3 of #188 contract: per-file counts surfaced verbatim."""
    out = tmp_path / "out.tar.gz"
    bundle_run_dir(synthetic_run_dir, output=out, redactor=_RecordingRedactor())
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
        manifest_member = [n for n in names if n.endswith(MANIFEST_FILENAME)][0]
        extracted = tar.extractfile(manifest_member)
        assert extracted is not None
        manifest = Manifest.from_json(extracted.read().decode("utf-8"))
    assert manifest.redaction_applied is True
    assert manifest.redactor_kind == "recording"
    for f in manifest.files:
        assert f.env_keys_removed == 1
        assert f.paths_rewritten == 2
        assert f.ips_rewritten == 3


# --- review callback ------------------------------------------------------


def test_bundle_run_dir_review_yes_proceeds(synthetic_run_dir, tmp_path):
    seen: list[Manifest] = []

    def confirm(manifest):
        seen.append(manifest)
        return True

    out = tmp_path / "out.tar.gz"
    written = bundle_run_dir(synthetic_run_dir, output=out, review_callback=confirm)
    assert written.is_file()
    assert len(seen) == 1
    assert seen[0].ticket == "TKT-1"


def test_bundle_run_dir_review_no_aborts_with_typed_error(synthetic_run_dir, tmp_path):
    from aorta.bundle import BundleAbortedError

    out = tmp_path / "out.tar.gz"
    with pytest.raises(BundleAbortedError):
        bundle_run_dir(synthetic_run_dir, output=out, review_callback=lambda m: False)
    assert not out.exists()


# --- redaction-from via CLI uses RedactingRedactor when recipe has block ---


def test_bundle_cli_redaction_from_applies_scrubbers(synthetic_run_dir, tmp_path):
    """Phase 3: --redaction-from loads redaction: and scrubs bundled files."""
    recipe = synthetic_run_dir / "recipe.resolved.yaml"
    recipe.write_text(
        """\
schema_version: 1
mode: probe
trials: 1
mitigation_axis: [none]
diagnostic_axis: [none]
redaction:
  scrub_env_keys: ["AWS_*"]
  scrub_paths: true
  scrub_ip_addresses: true
""",
        encoding="utf-8",
    )
    trial = synthetic_run_dir / "none-none" / "trial_0"
    trial.joinpath("stdout.log").write_text(
        "host 10.0.0.1 path /var/secret/data\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.tar.gz"
    from click.testing import CliRunner

    from aorta.cli import main

    result = CliRunner().invoke(
        main,
        ["bundle", str(synthetic_run_dir), "--redaction-from", str(recipe), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    import tarfile

    from aorta.bundle import MANIFEST_FILENAME, Manifest

    with tarfile.open(out, "r:gz") as tar:
        member = next(n for n in tar.getnames() if n.endswith(MANIFEST_FILENAME))
        manifest = Manifest.from_json(tar.extractfile(member).read().decode("utf-8"))
    assert manifest.redaction_applied is True
    assert manifest.redactor_kind == "probe.v1"


# --- bundle name + timestamp -----------------------------------------------


def test_bundle_name_uses_safe_slug_of_ticket(synthetic_run_dir, tmp_path):
    """Tickets with slashes / spaces are slugged for the filename."""
    out = bundle_run_dir(synthetic_run_dir, ticket="TKT/with spaces", output=tmp_path)
    # safe_slug rewrites '/' and ' ' to '_'.
    assert out.name.startswith("TKT_with_spaces-")
    # The manifest still records the ORIGINAL ticket (un-slugged).
    with tarfile.open(out, "r:gz") as tar:
        member = next(n for n in tar.getnames() if n.endswith(MANIFEST_FILENAME))
        manifest = Manifest.from_json(tar.extractfile(member).read().decode("utf-8"))
    assert manifest.ticket == "TKT/with spaces"


def test_bundle_run_dir_result_json_round_trip(synthetic_run_dir, tmp_path):
    """The bundle's stdout.log / result.json files match the source bytes
    when running with the IdentityRedactor (no scrubbing).
    """
    out = tmp_path / "out.tar.gz"
    bundle_run_dir(synthetic_run_dir, output=out)
    with tempfile.TemporaryDirectory() as extract_root:
        extract = Path(extract_root)
        with tarfile.open(out, "r:gz") as tar:
            tar.extractall(extract)  # noqa: S202 - controlled fixture
        bundle_root = next(p for p in extract.iterdir() if p.is_dir())
        for rel in ("none-none/trial_0/stdout.log", "none-none/trial_0/result.json"):
            assert (bundle_root / rel).read_bytes() == (synthetic_run_dir / rel).read_bytes()
        # result.json is still parseable.
        doc = json.loads((bundle_root / "none-none/trial_0/result.json").read_text())
        assert doc["verdict"] == "pass"


# --- bundle name carries millisecond precision (review fix) ----------------


def test_bundle_timestamp_has_millisecond_resolution():
    """Two timestamps in the same second but different ms produce
    distinct strings -- otherwise back-to-back ``aorta bundle`` runs
    would collide on the staging mkdir(exist_ok=False) and fail.

    Pure-function test against ``_bundle_timestamp`` so we don't
    have to race the wall clock.
    """
    import datetime as dt

    base = dt.datetime(2026, 6, 1, 7, 0, 0, 123_000, tzinfo=dt.timezone.utc)
    twin = base.replace(microsecond=456_000)
    a = _bundle_timestamp(base)
    b = _bundle_timestamp(twin)
    assert a == "2026-06-01T07-00-00-123"
    assert b == "2026-06-01T07-00-00-456"
    assert a != b


def test_bundle_name_includes_milliseconds(synthetic_run_dir, tmp_path):
    """End-to-end: the bundle filename includes the ms suffix, so
    back-to-back invocations don't clobber each other or fail the
    staging mkdir.
    """
    import datetime as dt

    fixed = dt.datetime(2026, 6, 1, 7, 0, 0, 789_000, tzinfo=dt.timezone.utc)
    out = bundle_run_dir(synthetic_run_dir, output=tmp_path, now=fixed)
    assert out.name == "TKT-1-2026-06-01T07-00-00-789.tar.gz"


# --- OSError wrapping (review fix) -----------------------------------------


class _RaisingRedactor(Redactor):
    """Redactor whose scrub_file raises OSError on the first call.

    Models the operator-visible failure modes documented on
    :class:`Redactor` (permissions, ENOSPC, transient FS errors).
    """

    kind = "raising"

    def __init__(self, exc: OSError) -> None:
        self.exc = exc

    def scrub_file(self, src: Path, dst: Path) -> RedactionCounts:
        raise self.exc


def test_bundle_run_dir_wraps_oserror_from_redactor_into_bundle_io_error(
    synthetic_run_dir, tmp_path
):
    """The Redactor ABC docstring promises BundleError-shaped failures.

    Without the writer wrap, a permission denied (or ENOSPC) on a
    single file would escape as a raw OSError and bypass the CLI's
    BundleError -> ClickException mapping, leaving the operator with
    a Python traceback. The wrap below is what makes that promise
    real.
    """
    boom = PermissionError(13, "Permission denied", "stdout.log")
    with pytest.raises(BundleIOError) as exc:
        bundle_run_dir(
            synthetic_run_dir,
            output=tmp_path / "out.tar.gz",
            redactor=_RaisingRedactor(boom),
        )
    # The original OSError is preserved on .cause so ops tooling can
    # grade the failure (PermissionError vs ENOSPC vs ...) without
    # parsing the message.
    assert exc.value.cause is boom
    assert exc.value.run_dir == synthetic_run_dir.resolve()
    # And the message includes the original error so the CLI surface
    # is not "filesystem error: <empty>".
    assert "Permission denied" in str(exc.value)
    # No tarball was written despite the half-staged tree -- the
    # TemporaryDirectory cleans up.
    assert not (tmp_path / "out.tar.gz").exists()


# --- skip-filter is top-level only (review fix) ----------------------------


def test_iter_source_files_keeps_nested_manifest_and_lockfile_lookalikes(
    synthetic_run_dir, tmp_path
):
    """Per docstring: only ``./manifest.json`` and ``./.aorta-probe.lock``
    are dropped. A workload that writes those same basenames inside a
    cell / trial directory is emitting legitimate artifacts and must
    have them bundled.

    Before PR #199 round 2, ``_iter_source_files`` matched basenames
    anywhere in the tree, silently dropping nested artifacts.
    """
    nested_manifest = synthetic_run_dir / "none-none" / "trial_0" / "manifest.json"
    nested_lockfile = synthetic_run_dir / "tf32_off-none" / "trial_0" / ".aorta-probe.lock"
    nested_manifest.write_text('{"workload": "ok"}', encoding="utf-8")
    nested_lockfile.write_text("nested lock\n", encoding="utf-8")
    # Top-level versions of both should still be dropped.
    (synthetic_run_dir / "manifest.json").write_text("stale\n", encoding="utf-8")
    (synthetic_run_dir / ".aorta-probe.lock").write_text("stale\n", encoding="utf-8")

    out = tmp_path / "out.tar.gz"
    bundle_run_dir(synthetic_run_dir, output=out)
    with tarfile.open(out, "r:gz") as tar:
        ordered_names = [n for n in tar.getnames() if n]
    names = set(ordered_names)
    # Derive bundle_root from a file that lives ONLY at the top level
    # (host_env.json) so we don't accidentally pick a deeper match.
    bundle_root = next(n for n in names if n.endswith("/host_env.json"))[: -len("/host_env.json")]
    # Nested look-alikes survived the filter.
    assert f"{bundle_root}/none-none/trial_0/manifest.json" in names
    assert f"{bundle_root}/tf32_off-none/trial_0/.aorta-probe.lock" in names
    # Top-level stale ones did NOT.
    # (The OWN manifest.json the bundle writes at top-level is the
    # only ./manifest.json in the archive, written by stage_run_dir.)
    assert sum(1 for n in names if n == f"{bundle_root}/manifest.json") == 1
    assert f"{bundle_root}/.aorta-probe.lock" not in names
    # Ordering contract (Sonbol PR #199 review): the TOP-LEVEL
    # manifest.json is the tarball trailer even when a nested
    # ``*/manifest.json`` exists. The sort keys on the relative POSIX
    # path, so the nested manifest must NOT be the final member.
    file_entries = [n for n in ordered_names if n != bundle_root]
    assert file_entries[-1] == f"{bundle_root}/manifest.json"
    assert file_entries[-1] != f"{bundle_root}/none-none/trial_0/manifest.json"


# --- atomic tarball write (review fix) -------------------------------------


def test_write_tarball_failure_leaves_no_partial(synthetic_run_dir, tmp_path, monkeypatch):
    """If tarfile / gzip raises mid-write (ENOSPC, EIO, ...) and the
    final ``output`` path did not pre-exist, it must NOT exist after
    the failure -- BundleIOError's message promises 'no new tarball
    was written' and the writer has to honour that (here there is no
    prior file at the path, so nothing should appear).

    The partial sibling (``<output>.partial``) is also cleaned up so
    a retry does not race against stale temp data.
    """
    import aorta.bundle.writer as writer_mod

    real_open = tarfile.open

    def _raising_open(*args, **kwargs):
        tar = real_open(*args, **kwargs)
        original_add = tar.add
        call_count = {"n": 0}

        def boom_after_one_add(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise OSError(28, "No space left on device")  # ENOSPC
            return original_add(*a, **kw)

        tar.add = boom_after_one_add
        return tar

    monkeypatch.setattr(writer_mod.tarfile, "open", _raising_open)

    out = tmp_path / "out.tar.gz"
    with pytest.raises(BundleIOError) as exc:
        bundle_run_dir(synthetic_run_dir, output=out)
    assert "No space left on device" in str(exc.value)
    # The promise: neither the final file nor the partial survives.
    assert not out.exists(), "Atomic write violated -- partial tarball at final path"
    assert not out.with_name(out.name + ".partial").exists(), ".partial sibling leaked on failure"


def test_write_tarball_failure_preserves_prior_output(synthetic_run_dir, tmp_path, monkeypatch):
    """The atomicity guarantee is strongest when ``output`` already
    exists from a prior run: a mid-write failure must NOT corrupt
    or remove the prior copy.

    Pre-populates ``output`` with a sentinel, forces the new write
    to fail, then re-reads the file and asserts the sentinel is
    intact.
    """
    import aorta.bundle.writer as writer_mod

    out = tmp_path / "out.tar.gz"
    sentinel = b"PRIOR-BUNDLE-DO-NOT-DELETE"
    out.write_bytes(sentinel)

    def _exploding_open(*args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(writer_mod.tarfile, "open", _exploding_open)

    with pytest.raises(BundleIOError) as exc:
        bundle_run_dir(synthetic_run_dir, output=out)
    # Prior file untouched.
    assert out.read_bytes() == sentinel
    # No .partial left behind.
    assert not out.with_name(out.name + ".partial").exists()
    # The message must NOT claim the path is empty -- it says no NEW
    # tarball was written and the existing file was left untouched
    # (Copilot PR #199 review).
    assert "no new tarball was written" in str(exc.value).lower()
    assert "left untouched" in str(exc.value).lower()


def test_write_tarball_cleans_stale_partial_before_writing(synthetic_run_dir, tmp_path):
    """A leftover ``.partial`` from a previous crashed run must not
    survive into the next successful write -- ``write_tarball``
    proactively unlinks it before opening the new gzip stream.
    """
    out = tmp_path / "out.tar.gz"
    stale = out.with_name(out.name + ".partial")
    stale.write_bytes(b"stale partial bytes")
    # Should succeed and leave only the final tarball; no .partial.
    written = bundle_run_dir(synthetic_run_dir, output=out)
    assert written == out.resolve()
    assert out.is_file()
    assert not stale.exists()


# --- UTC normalisation (review fix) ----------------------------------------


def test_bundle_timestamp_normalises_naive_now_as_utc():
    """A naive datetime is treated as already-UTC (matching
    ``datetime.now(timezone.utc)``'s shape) instead of being
    silently mislabelled.
    """
    import datetime as dt

    naive = dt.datetime(2026, 6, 1, 7, 0, 0, 250_000)  # no tzinfo
    out = _bundle_timestamp(naive)
    # The naive wall-clock 07:00:00.250 is taken at face value.
    assert out == "2026-06-01T07-00-00-250"


def test_bundle_timestamp_normalises_non_utc_now_to_utc():
    """An aware datetime in +05:30 is converted to UTC before
    formatting, so the embedded clock matches the ``Z`` convention
    in the manifest.
    """
    import datetime as dt

    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    local = dt.datetime(2026, 6, 1, 12, 30, 0, 500_000, tzinfo=ist)
    # 12:30 IST == 07:00 UTC
    out = _bundle_timestamp(local)
    assert out == "2026-06-01T07-00-00-500"


# --- trust boundary: symlinks + manifest path leak (Sonbol review) ---------


def test_bundle_refuses_symlink_escaping_run_dir(synthetic_run_dir, tmp_path):
    """A symlink under the run dir whose target is OUTSIDE the tree
    must not be dereferenced into the bundle (Sonbol PR #199 review).

    ``aorta bundle`` ships a shareable artifact, so following
    ``run_dir/cell/trial_0/link -> ../../secret.txt`` would pull an
    unrelated local file's bytes across the trust boundary. The
    writer refuses with :class:`UnsafeSymlinkError`.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("CUSTOMER_PRIVATE_KEY=hunter2\n", encoding="utf-8")
    link = synthetic_run_dir / "none-none" / "trial_0" / "leak"
    link.symlink_to(secret)

    out = tmp_path / "out.tar.gz"
    with pytest.raises(UnsafeSymlinkError) as exc:
        bundle_run_dir(synthetic_run_dir, output=out)
    assert exc.value.target == secret.resolve()
    assert exc.value.path.name == "leak"
    # Nothing was written.
    assert not out.exists()


def test_bundle_follows_in_tree_symlink(synthetic_run_dir, tmp_path):
    """A symlink whose target stays INSIDE the run dir is still
    followed -- the guard only refuses escapes, not all symlinks.
    """
    target = synthetic_run_dir / "none-none" / "trial_0" / "stdout.log"
    link = synthetic_run_dir / "none-none" / "trial_0" / "stdout_alias.log"
    link.symlink_to(target)

    out = tmp_path / "out.tar.gz"
    written = bundle_run_dir(synthetic_run_dir, output=out)
    with tarfile.open(written, "r:gz") as tar:
        names = set(tar.getnames())
    assert any(n.endswith("/none-none/trial_0/stdout_alias.log") for n in names)


def test_identity_redactor_preserves_restrictive_mode(synthetic_run_dir, tmp_path):
    """A 0600 source file (e.g. ``probe.env``) must NOT be widened to
    the umask default when copied into the bundle staging tree
    (Copilot PR #199 review). ``IdentityRedactor`` carries the source
    mode via ``shutil.copymode`` so the bundle copy is never less
    restrictive than the original.
    """
    import stat

    secret = synthetic_run_dir / "none-none" / "trial_0" / "probe.env"
    secret.write_text("AWS_SECRET=shhh\n", encoding="utf-8")
    secret.chmod(0o600)

    staging = tmp_path / "staging"
    staging.mkdir()
    stage_run_dir(
        synthetic_run_dir,
        staging,
        "TKT-1-bundle",
        redactor=IdentityRedactor(),
        ticket="TKT-1",
        aorta_version="0.2.0",
    )
    staged = staging / "TKT-1-bundle" / "none-none" / "trial_0" / "probe.env"
    assert staged.is_file()
    assert stat.S_IMODE(staged.stat().st_mode) == 0o600


def test_tarball_headers_scrub_owner_identity(synthetic_run_dir, tmp_path):
    """Tar headers must not leak the operator's uid/gid/uname/gname and
    must pin mtime (Copilot PR #199 review): a shareable bundle should
    not carry workstation identity. File MODE is still preserved through
    the filter so a 0600 probe.env stays restrictive.
    """
    import stat

    secret = synthetic_run_dir / "none-none" / "trial_0" / "probe.env"
    secret.write_text("AWS_SECRET=shhh\n", encoding="utf-8")
    secret.chmod(0o600)

    out = tmp_path / "out.tar.gz"
    bundle_run_dir(synthetic_run_dir, output=out)
    with tarfile.open(out, "r:gz") as tar:
        members = tar.getmembers()
    assert members
    for m in members:
        assert m.uid == 0, f"{m.name}: uid leaked"
        assert m.gid == 0, f"{m.name}: gid leaked"
        assert m.uname == "", f"{m.name}: uname leaked"
        assert m.gname == "", f"{m.name}: gname leaked"
        assert m.mtime == 0, f"{m.name}: mtime not pinned"
    env_member = next(m for m in members if m.name.endswith("/none-none/trial_0/probe.env"))
    assert stat.S_IMODE(env_member.mode) == 0o600


def test_manifest_does_not_leak_absolute_source_path(synthetic_run_dir, tmp_path):
    """The extracted manifest must not carry the operator's absolute
    source path (Sonbol PR #199 review): that would leak workstation
    usernames, mount points, or customer directory names off the
    source machine. Only the run dir's leaf name is recorded.
    """
    out = tmp_path / "out.tar.gz"
    bundle_run_dir(synthetic_run_dir, output=out)
    with tarfile.open(out, "r:gz") as tar:
        manifest_member = next(n for n in tar.getnames() if n.endswith(f"/{MANIFEST_FILENAME}"))
        raw = tar.extractfile(manifest_member).read().decode("utf-8")
    manifest = Manifest.from_json(raw)
    # Only the leaf name -- never the absolute path or any parent.
    assert manifest.source_run_dir == synthetic_run_dir.name
    abs_source = str(synthetic_run_dir.resolve())
    assert abs_source not in raw
    assert str(tmp_path) not in raw
    assert "/" not in manifest.source_run_dir
