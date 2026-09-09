"""Evidence is read from the bundle, and only from the bundle.

``read_evidence_file`` joined its *uri* to the bundle root and read whatever
came out. The uri arrives from the evidence list and from the model's own tool
call, so neither end of it is the bundle's own.

Two ways out. ``../../../etc/passwd`` walks out of the bundle, and an absolute
path skips it entirely, because ``Path(root) / "/etc/passwd"`` discards the
root -- the quieter of the two, and the one that needs no traversal at all.
Whatever came back went into the router's context, from where it can be quoted
into the rationale, which is written to the report and sent to a model.
"""

from __future__ import annotations

import pytest

from aorta.cia.autopsy.router import read_evidence_file, resolve_in_bundle


@pytest.fixture
def bundle(tmp_path):
    """A bundle with real evidence in it, and a secret next door."""
    root = tmp_path / "bundle"
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "watch.stderr.log").write_text("step=5 loss=nan\n")
    (root / "manifest.yaml").write_text("schema_version: '0.1'\n")
    (tmp_path / "outside.txt").write_text("NOT_EVIDENCE_SECRET\n")
    return root


class TestWhatItReads:
    def test_a_file_in_the_bundle(self, bundle):
        assert "loss=nan" in read_evidence_file("logs/watch.stderr.log", str(bundle))

    def test_a_file_at_the_bundle_root(self, bundle):
        assert "schema_version" in read_evidence_file("manifest.yaml", str(bundle))

    def test_a_path_that_stays_inside_while_looking_like_it_leaves(self, bundle):
        """logs/../manifest.yaml is inside; containment is not string matching."""
        assert "schema_version" in read_evidence_file("logs/../manifest.yaml", str(bundle))

    def test_a_missing_file_is_reported_as_missing_not_refused(self, bundle):
        """The two are different, and a reader should be told which."""
        assert "not found" in read_evidence_file("logs/absent.log", str(bundle))


class TestWhatItRefuses:
    @pytest.mark.parametrize(
        "uri",
        [
            "../outside.txt",
            "../../etc/passwd",
            "../../../../../../etc/passwd",
            "logs/../../outside.txt",
            "logs/../../../etc/passwd",
        ],
    )
    def test_a_path_that_walks_out(self, uri, bundle):
        out = read_evidence_file(uri, str(bundle))
        assert "refused" in out
        assert "NOT_EVIDENCE_SECRET" not in out
        assert "root:x:0:0" not in out

    @pytest.mark.parametrize("uri", ["/etc/passwd", "/etc/hostname", "/tmp"])
    def test_an_absolute_path_skips_the_bundle_entirely(self, uri, bundle):
        """Path(root) / "/etc/passwd" is "/etc/passwd"; no traversal needed."""
        assert "refused" in read_evidence_file(uri, str(bundle))

    def test_a_symlink_pointing_out_is_out(self, bundle, tmp_path):
        """Resolving both sides settles links, not just dot-dots."""
        (bundle / "logs" / "escape.log").symlink_to(tmp_path / "outside.txt")
        out = read_evidence_file("logs/escape.log", str(bundle))
        assert "NOT_EVIDENCE_SECRET" not in out

    def test_an_empty_uri_reads_nothing(self, bundle):
        assert "refused" in read_evidence_file("", str(bundle))


class TestTheResolver:
    def test_it_returns_a_path_inside(self, bundle):
        resolved = resolve_in_bundle("logs/watch.stderr.log", str(bundle))
        assert resolved is not None
        assert resolved.is_relative_to(bundle.resolve())

    def test_it_returns_none_for_anything_outside(self, bundle):
        assert resolve_in_bundle("../outside.txt", str(bundle)) is None
        assert resolve_in_bundle("/etc/passwd", str(bundle)) is None

    def test_the_bundle_root_itself_is_inside_it(self, bundle):
        assert resolve_in_bundle(".", str(bundle)) is not None

    def test_a_sibling_with_a_shared_prefix_is_not_inside(self, tmp_path):
        """bundle-evil must not pass because it starts with bundle."""
        (tmp_path / "bundle").mkdir()
        (tmp_path / "bundle-evil").mkdir()
        assert resolve_in_bundle("../bundle-evil", str(tmp_path / "bundle")) is None
