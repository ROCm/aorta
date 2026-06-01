"""Writer / staging / tarball tests for ``aorta bundle`` (issue #196)."""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import pytest

from aorta.bundle import (
    EmptyRunDirError,
    IdentityRedactor,
    Manifest,
    NoTicketError,
    RedactionCounts,
    Redactor,
    RunDirNotFoundError,
    bundle_run_dir,
    resolve_ticket,
)
from aorta.bundle.manifest import MANIFEST_FILENAME
from aorta.bundle.writer import stage_run_dir, write_tarball

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


# --- redaction-from flag is honoured but no-op until #188 Phase 3 ---------


def test_bundle_run_dir_redaction_from_logged_when_present(synthetic_run_dir, tmp_path, caplog):
    """Acceptance: --redaction-from is wired through (not yet consumed)."""
    recipe = synthetic_run_dir / "recipe.resolved.yaml"  # already created by fixture
    out = tmp_path / "out.tar.gz"
    with caplog.at_level("INFO", logger="aorta.bundle.writer"):
        bundle_run_dir(synthetic_run_dir, output=out, redaction_from=recipe)
    assert any(
        "aorta.probe.redaction is gated on issue #188 Phase 3" in r.message for r in caplog.records
    )


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
