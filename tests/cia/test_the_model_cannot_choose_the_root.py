"""The containment root is not something the model gets to name.

``read_evidence_file(uri, bundle_root)`` was a ReAct tool, so both arguments
came from the model. There was a containment check, and it worked -- it
refused anything outside ``bundle_root``. The flaw was what it checked
against: a call with ``bundle_root="/"`` made every host file genuinely
contained, and the check passed.

What came back went into the router's context and could be quoted into the
rationale, which is written to the report and sent to a model. So a read that
escaped did not fail loudly; it succeeded and was summarised.

The root is bound in a closure now, at the point the orchestrator opens the
bundle. The model can still name any URI it likes; every one is resolved
against a root it cannot reach.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("dspy", reason="the autopsy router needs the [cia] extra")

from aorta.cia.autopsy import router


@pytest.fixture()
def bundle(tmp_path):
    """A bundle with one real evidence file, and a secret outside it."""
    root = tmp_path / "bundle"
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "watch.log").write_text("real evidence\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("SECRET HOST FILE\n", encoding="utf-8")
    return root


class TestTheToolNoLongerTakesARoot:
    def test_the_model_sees_only_a_uri(self):
        """The schema is the fix: there is no root argument to supply."""
        bound = router.evidence_reader("/tmp")

        assert list(inspect.signature(bound).parameters) == ["uri"]

    def test_the_signature_does_not_carry_a_host_path(self):
        """bundle_root was an InputField too, putting the path in the prompt."""
        fields = getattr(router.TriageDecision, "model_fields", None) or {}

        assert "bundle_root" not in fields

    def test_the_router_is_built_with_a_root(self):
        assert "bundle_root" in inspect.signature(router.TriageRouter.__init__).parameters


class TestItStillReadsRealEvidence:
    def test_a_file_inside_the_bundle_is_returned(self, bundle):
        read = router.evidence_reader(bundle)

        assert "real evidence" in read("logs/watch.log")

    def test_a_missing_file_says_so(self, bundle):
        read = router.evidence_reader(bundle)

        assert "not found" in read("logs/absent.log")


class TestNothingOutsideIsReachable:
    @pytest.mark.parametrize(
        "attempt",
        [
            "../secret.txt",
            "/etc/passwd",
            "logs/../../secret.txt",
            "./../../secret.txt",
        ],
    )
    def test_an_escaping_uri_is_refused(self, bundle, attempt):
        read = router.evidence_reader(bundle)

        assert read(attempt).startswith("[refused")

    def test_an_absolute_path_to_the_real_secret_is_refused(self, bundle, tmp_path):
        read = router.evidence_reader(bundle)

        assert "SECRET" not in read(str(tmp_path / "secret.txt"))

    def test_a_symlink_out_of_the_bundle_is_refused(self, bundle, tmp_path):
        """Judged by where it points, not by where it sits."""
        (bundle / "logs" / "sneaky.log").symlink_to(tmp_path / "secret.txt")
        read = router.evidence_reader(bundle)

        assert "SECRET" not in read("logs/sneaky.log")

    def test_a_nested_symlinked_directory_is_refused(self, bundle, tmp_path):
        """The escape the review found in the report path: a link mid-way."""
        (bundle / "nested").mkdir()
        (bundle / "nested" / "out").symlink_to(tmp_path)
        read = router.evidence_reader(bundle)

        assert "SECRET" not in read("nested/out/secret.txt")


class TestOneJobsRootDoesNotServeAnother:
    def test_two_readers_keep_their_own_roots(self, tmp_path):
        """A module shared across jobs would carry the first job's root."""
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "note.txt").write_text(f"bundle {name}\n", encoding="utf-8")

        first = router.evidence_reader(tmp_path / "a")
        second = router.evidence_reader(tmp_path / "b")

        assert "bundle a" in first("note.txt")
        assert "bundle b" in second("note.txt")

    def test_one_cannot_read_the_other(self, tmp_path):
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "note.txt").write_text(f"bundle {name}\n", encoding="utf-8")

        first = router.evidence_reader(tmp_path / "a")

        assert "bundle b" not in first("../b/note.txt")


class TestTheCallerBindsIt:
    @staticmethod
    def _router_call() -> str:
        """Just the router invocation, not the whole module.

        ``build_report`` records which bundle a report describes and passes the
        root for that; it is not model-callable and is none of this test's
        business.
        """
        from pathlib import Path

        import aorta.cia.autopsy.orchestrator as orch

        source = Path(orch.__file__).read_text(encoding="utf-8")
        start = source.index("TriageRouter(")
        return source[start : source.index(")", source.index("router(", start)) + 1]

    def test_the_root_goes_to_the_constructor(self):
        """Not into the call, where it would travel beside model-written args."""
        assert "TriageRouter(bundle_root)" in self._router_call()

    def test_the_root_is_not_passed_to_the_invocation(self):
        assert "bundle_root" not in self._router_call().split("router(", 1)[1]
