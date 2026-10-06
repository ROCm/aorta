"""A recipe is not paired with a sidecar that has nothing to do with it.

``check_aorta_install`` looked for a sidecar beside the recipe, then anywhere
under the search roots, and if nothing was named for the recipe's ticket it
took ``candidates[0]`` -- on the wide search, whichever unrelated file ``find``
happened to list first.

Its own docstring says pairing a recipe with the wrong sidecar fails at load
with ``UnknownEnvironmentError``, so the fallback manufactured the failure it
warns about. And it did so invisibly: a returned path looks like an answer,
where an empty one is a question the planner can ask.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", reason="the planner needs the [cia] extra")

import aorta.cia.launch.planner as planner


@pytest.fixture()
def probe(monkeypatch):
    """Drive check_aorta_install by scripting what each remote probe returns."""

    def install(*, recipe_found: str, beside: list[str], anywhere: list[str]):
        def fake_run_probe(host, command):
            if "command -v aorta" in command:
                return "/usr/bin/aorta\n"
            if "--version" in command:
                return "aorta, version 1.0\n"
            if command.startswith("find") and "-name" in command and "sidecar" not in command:
                return recipe_found
            if command.startswith("ls ") and "sidecar" in command:
                return "\n".join(beside)
            if "sidecar" in command:
                return "\n".join(anywhere)
            return ""

        monkeypatch.setattr(planner, "run_probe", fake_run_probe)
        return planner.check_aorta_install("host", "TICKET-42.yaml")

    return install


class TestWhenThePairingIsKnown:
    def test_a_ticket_named_sidecar_is_used(self, probe):
        out = probe(
            recipe_found="/r/TICKET-42.yaml\n",
            beside=["/r/TICKET-42-sidecar.json", "/r/other-sidecar.json"],
            anywhere=[],
        )

        assert out["sidecar_path"] == "/r/TICKET-42-sidecar.json"
        assert out["sidecar_needs_confirmation"] is False

    def test_the_only_sidecar_beside_the_recipe_is_used(self, probe):
        """Not proof, but it is the pairing the layout asserts."""
        out = probe(
            recipe_found="/r/TICKET-42.yaml\n",
            beside=["/r/env-sidecar.json"],
            anywhere=[],
        )

        assert out["sidecar_path"] == "/r/env-sidecar.json"
        assert out["sidecar_owned"] is True


class TestWhenItIsNot:
    def test_an_unrelated_sidecar_elsewhere_is_not_taken(self, probe):
        """The finding: candidates[0] from a machine-wide find."""
        out = probe(
            recipe_found="/r/TICKET-42.yaml\n",
            beside=[],
            anywhere=["/somewhere/else/unrelated-sidecar.json"],
        )

        assert out["sidecar_path"] == ""

    def test_it_asks_instead(self, probe):
        out = probe(
            recipe_found="/r/TICKET-42.yaml\n",
            beside=[],
            anywhere=["/somewhere/else/unrelated-sidecar.json"],
        )

        assert out["sidecar_needs_confirmation"] is True
        assert "UnknownEnvironmentError" in out["sidecar_note"]

    def test_the_candidates_are_still_reported(self, probe):
        """So whoever is asked has something to choose between."""
        out = probe(
            recipe_found="/r/TICKET-42.yaml\n",
            beside=[],
            anywhere=["/a/one-sidecar.json", "/b/two-sidecar.json"],
        )

        assert out["sidecar_candidates"] == ["/a/one-sidecar.json", "/b/two-sidecar.json"]

    def test_several_beside_the_recipe_with_no_ticket_match_is_ambiguous(self, probe):
        """Co-located but ambiguous is still a guess."""
        out = probe(
            recipe_found="/r/TICKET-42.yaml\n",
            beside=["/r/alpha-sidecar.json", "/r/beta-sidecar.json"],
            anywhere=[],
        )

        assert out["sidecar_path"] == ""
        assert out["sidecar_needs_confirmation"] is True


class TestWhenThereIsNothingToPair:
    def test_no_candidates_needs_no_confirmation(self, probe):
        """Nothing to confirm; a recipe carrying its own environments is fine."""
        out = probe(recipe_found="/r/TICKET-42.yaml\n", beside=[], anywhere=[])

        assert out["sidecar_path"] == ""
        assert out["sidecar_needs_confirmation"] is False

    def test_the_note_is_empty_when_there_is_no_question(self, probe):
        out = probe(recipe_found="/r/TICKET-42.yaml\n", beside=[], anywhere=[])

        assert out["sidecar_note"] == ""


class TestTheGuessIsGone:
    def test_the_source_no_longer_takes_the_first_candidate(self):
        from pathlib import Path

        source = Path(planner.__file__).read_text(encoding="utf-8")
        guess = "if not sidecar_path and candidates:\n        sidecar_path = candidates[0]"

        assert guess not in source


class TestThePlannerIsToldWhatToDo:
    """A flag nothing reads is not a fix."""

    @staticmethod
    def _instructions() -> str:
        from pathlib import Path

        return Path(planner.__file__).read_text(encoding="utf-8")

    def test_the_confirmation_flag_is_named(self):
        assert "sidecar_needs_confirmation=true" in self._instructions()

    def test_it_is_told_not_to_choose_from_the_candidates(self):
        source = self._instructions()
        start = source.index("sidecar_needs_confirmation=true")

        assert "Do not pick one" in source[start : start + 320]

    def test_it_is_told_to_ask(self):
        source = self._instructions()
        start = source.index("sidecar_needs_confirmation=true")

        assert "needs_confirmation=true" in source[start : start + 400]
