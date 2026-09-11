"""A manifest cannot point an adapter at a file outside its bundle.

``BundleContext.path()`` joined the manifest's ``paths`` entries to the bundle
root and handed back whatever came out. Every adapter reads its evidence through
that one method and sends what it finds to the router, from where it can be
quoted into a rationale -- so a manifest naming ``/etc/passwd``, ``../../..`` or
an escaping symlink was read and could reach a model.

A manifest is not a trusted document. It is written into the bundle by whatever
produced the run, which is the thing under investigation.
"""

from __future__ import annotations

import pytest

from aorta.cia.autopsy.adapters.base import BundleContext, resolve_in_bundle

#: Every key the shipped adapters ask for, so the check is not proven on one.
ADAPTER_KEYS = [
    "stderr",
    "sanitizer_report",
    "aorta_matrix",
    "aorta_matrix_md",
    "rocgdb_session",
]


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "watch.stderr.log").write_text("step=5 loss=nan\n")
    (root / "manifest.yaml").write_text("schema_version: '0.1'\n")
    (tmp_path / "outside.txt").write_text("NOT_EVIDENCE_SECRET\n")
    return root


def _ctx(root, **paths) -> BundleContext:
    return BundleContext(root=root, manifest={"paths": paths}, job_id="cia-test")


class TestWhatAnAdapterMayRead:
    def test_a_file_inside_the_bundle(self, bundle):
        got = _ctx(bundle, stderr="logs/watch.stderr.log").path("stderr")
        assert got is not None and got.read_text().startswith("step=5")

    def test_a_path_that_stays_inside_while_looking_like_it_leaves(self, bundle):
        assert _ctx(bundle, stderr="logs/../manifest.yaml").path("stderr") is not None

    def test_a_key_the_manifest_does_not_have(self, bundle):
        assert _ctx(bundle).path("stderr") is None

    def test_an_empty_value(self, bundle):
        assert _ctx(bundle, stderr="").path("stderr") is None


class TestWhatItRefuses:
    @pytest.mark.parametrize("key", ADAPTER_KEYS)
    def test_every_adapter_key_is_covered(self, key, bundle):
        """The check is on path(), so no adapter can opt out of it."""
        assert _ctx(bundle, **{key: "../outside.txt"}).path(key) is None

    @pytest.mark.parametrize(
        "value",
        ["../outside.txt", "../../etc/passwd", "logs/../../outside.txt", "../../../../etc/shadow"],
    )
    def test_a_path_that_walks_out(self, value, bundle):
        assert _ctx(bundle, stderr=value).path("stderr") is None

    @pytest.mark.parametrize("value", ["/etc/passwd", "/etc/hostname", "/tmp"])
    def test_an_absolute_path_needs_no_traversal(self, value, bundle):
        """Path(root) / "/etc/passwd" is "/etc/passwd"; the root is discarded."""
        assert _ctx(bundle, stderr=value).path("stderr") is None

    def test_a_symlink_pointing_out_is_out(self, bundle, tmp_path):
        (bundle / "logs" / "escape.log").symlink_to(tmp_path / "outside.txt")
        assert _ctx(bundle, stderr="logs/escape.log").path("stderr") is None

    def test_a_sibling_sharing_a_prefix_is_not_inside(self, tmp_path):
        (tmp_path / "bundle").mkdir()
        (tmp_path / "bundle-evil").mkdir()
        assert resolve_in_bundle(tmp_path / "bundle", "../bundle-evil") is None


class TestOneDefinitionOfInside:
    """Two copies of a containment check is how one loses the symlink case."""

    def test_the_router_uses_the_same_function(self):
        import inspect

        from aorta.cia.autopsy import router

        assert "_resolve_in_bundle" in inspect.getsource(router.resolve_in_bundle)

    def test_both_entry_points_agree(self, bundle):
        from aorta.cia.autopsy.router import resolve_in_bundle as router_resolve

        for value in ("logs/watch.stderr.log", "../outside.txt", "/etc/passwd"):
            adapter_says = resolve_in_bundle(bundle, value)
            router_says = router_resolve(value, str(bundle))
            assert (adapter_says is None) == (router_says is None), value
