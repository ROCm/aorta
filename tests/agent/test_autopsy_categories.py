"""The closed autopsy label set, and everything that has to agree with it.

A closed set is referenced in more places than it looks: the validator that
rejects anything outside it, the prompt that tells the model what exists, the
offline heuristic that labels without a model, the operator docs, and the
corpus ground truth. A member added in one place and missing in another is
worse than not adding it at all -- either the validator accepts a label the
model was never told about, or the model emits one the validator kills the
search over.

So these pin the *agreement between* those places rather than a hardcoded list
of names, and keep working as the set grows. The exceptions are the three
labels added because the measured end-to-end run had nowhere to put its
evidence; those are named explicitly, because losing one silently is the
regression this file exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.agent.llm import (
    AUTOPSY_CATEGORIES,
    AUTOPSY_CATEGORY_GUIDANCE,
    EVIDENCE_ONLY_CATEGORIES,
    PROBE_CATEGORIES,
    AgentStep,
    FakeLLMProposer,
    _build_prompt,
    _infer_category_from_detectors,
    _infer_category_from_symptom,
    format_category_guidance,
)
from aorta.agent.policy import AgentPolicy, PolicyViolation

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DOC = REPO_ROOT / "docs" / "agent" / "aorta-probe-agent.md"
SCENARIO_LABELS = REPO_ROOT / "examples" / "rl" / "corpus" / "scenario_labels.json"

#: Added because an end-to-end run against nine real sanitizer scenarios
#: answered ``unknown`` on eight of them and mislabelled the ninth: no correct
#: label existed. Named here rather than derived so that dropping one fails.
#:
#: These are ``main``'s spellings, not the ones this branch proposed. The CIA
#: port widened the same taxonomy from the other side within days of this PR,
#: reaching two of the same concepts under different names, and it merged
#: first -- so ``gpu_race`` became ``gpu_race`` and ``numeric_silent``
#: became ``numeric_silent``. ``tooling_gap`` is a third the CIA side found
#: that this one had not.
ADDED_CATEGORIES = ("gpu_race", "numeric_silent", "tooling_gap")


class TestTheSetItself:
    def test_the_set_is_derived_from_the_guidance_so_the_two_cannot_drift(self):
        assert AUTOPSY_CATEGORIES == frozenset(AUTOPSY_CATEGORY_GUIDANCE)

    def test_every_label_carries_a_gloss(self):
        """A label the model is shown without a gloss is a label it cannot route on."""
        assert [n for n, gloss in AUTOPSY_CATEGORY_GUIDANCE.items() if not gloss.strip()] == []

    @pytest.mark.parametrize("category", ADDED_CATEGORIES)
    def test_the_labels_the_evidence_demanded_are_present(self, category):
        assert category in AUTOPSY_CATEGORIES

    def test_gpu_race_did_not_replace_checkpoint_race(self):
        """They are different failures with different diagnostics.

        A checkpoint race is a concurrency problem around checkpoint I/O; an
        intra-wave LDS conflict is not. Collapsing them back together would
        re-create the mislabel that motivated the split.
        """
        assert {"gpu_race", "checkpoint_race"} <= AUTOPSY_CATEGORIES

    @pytest.mark.parametrize("category", sorted(EVIDENCE_ONLY_CATEGORIES))
    def test_every_evidence_only_label_carries_a_gloss_too(self, category):
        """The coupling the derived set creates, pinned from this side.

        ``AUTOPSY_CATEGORIES`` is derived from the guidance, so a name in
        ``EVIDENCE_ONLY_CATEGORIES`` without a gloss does not merely lose its
        gloss -- it drops out of the shared vocabulary altogether, and the CIA
        front door stops accepting a category it emits.
        ``tests/cia/test_probe_vocabulary.py`` catches that as a broken subset
        relation; this says why, at the place someone would add a name.
        """
        assert AUTOPSY_CATEGORY_GUIDANCE[category].strip()

    def test_the_guidance_is_read_only(self):
        with pytest.raises(TypeError):
            AUTOPSY_CATEGORY_GUIDANCE["invented"] = "nope"  # type: ignore[index]


class TestTheValidator:
    @pytest.mark.parametrize("category", sorted(PROBE_CATEGORIES))
    def test_every_probe_member_survives_validate_step(self, category):
        step = AgentStep(
            category=category,
            hypothesis="",
            next_mitigations=["tf32_off"],
            confidence=0.5,
            stop=False,
        )
        assert AgentPolicy().validate_step(step).category == category

    @pytest.mark.parametrize("category", sorted(EVIDENCE_ONLY_CATEGORIES))
    def test_an_evidence_only_member_is_refused(self, category):
        """Not a gap in the widening -- the point of it.

        These three are in the shared vocabulary so a report means one thing
        whichever front door wrote it, and out of `PROBE_CATEGORIES` because a
        mitigation sweep cannot establish them. A probe step asserting one
        would be a guess wearing an instrument's name.
        """
        step = AgentStep(
            category=category,
            hypothesis="",
            next_mitigations=["tf32_off"],
            confidence=0.5,
            stop=False,
        )
        with pytest.raises(PolicyViolation, match="invalid category"):
            AgentPolicy().validate_step(step)

    def test_unknown_stays_valid(self):
        """The honest answer when the evidence supports no label.

        Widening the set does not make declining wrong, and `validate_step`
        accepting it is load-bearing: four of the nine corpus scenarios are
        zero-finding controls whose correct label is `unknown`, and the probe
        downgrades an evidence-only reading to it.
        """
        step = AgentStep(
            category="unknown",
            hypothesis="",
            next_mitigations=[],
            confidence=0.0,
            stop=True,
        )
        assert AgentPolicy().validate_step(step).category == "unknown"

    @pytest.mark.parametrize("category", ["lds_race", "nan", "KERNEL_RACE", "", "kernel-race"])
    def test_a_label_outside_the_set_is_still_rejected(self, category):
        """Widening the set must not soften it -- it is still closed.

        `lds_race` in particular is the name `examples/rl/build_corpus.py` uses
        to synthesise its invalid-category negative example, so it has to stay
        outside.
        """
        step = AgentStep(
            category=category,
            hypothesis="",
            next_mitigations=[],
            confidence=0.5,
            stop=False,
        )
        with pytest.raises(PolicyViolation, match="invalid category"):
            AgentPolicy().validate_step(step)


class TestTheProposerIsTold:
    def test_the_prompt_carries_every_probe_label_and_its_gloss(self):
        """The model cannot use a label nobody mentioned to it."""
        system, _ = _build_prompt(None, [], ["tf32_off"], [])
        for name in PROBE_CATEGORIES:
            assert f"- {name}: " in system, f"{name} missing from the prompt"
            assert AUTOPSY_CATEGORY_GUIDANCE[name] in system, f"{name}'s gloss missing"

    @pytest.mark.parametrize("category", sorted(EVIDENCE_ONLY_CATEGORIES))
    def test_the_prompt_offers_no_label_the_probe_cannot_reach(self, category):
        """The other half, and the one that matters more.

        Showing the model a label it has no way to establish teaches it to
        guess one, and the guess then validates. The glosses still exist for
        these -- the CIA front door uses them -- they are just not offered here.
        """
        system, _ = _build_prompt(None, [], ["tf32_off"], [])
        assert f"- {category}: " not in system

    def test_the_rendering_is_one_line_per_label(self):
        lines = format_category_guidance().splitlines()
        assert len(lines) == len(AUTOPSY_CATEGORIES)
        assert lines == sorted(lines), "sorted so the prompt is stable between runs"

    def test_the_rendering_narrows_to_the_names_it_is_given(self):
        lines = format_category_guidance(PROBE_CATEGORIES).splitlines()
        assert len(lines) == len(PROBE_CATEGORIES)


def _label(*, detectors: list[str] | None = None, symptom: str | None = None) -> str:
    """What the two dispatchers identify, before the probe's policy narrows it.

    `FakeLLMProposer.propose` downgrades an evidence-only label to `unknown`,
    because `AgentPolicy` forbids a probe step from asserting one. Whether the
    evidence was *read* correctly is a separate question from whether the probe
    is *allowed to say so*, so the routing tests ask the dispatchers directly
    and `TestTheProbeSaysOnlyWhatItMayAssert` asks the proposer.
    """
    category = _infer_category_from_detectors(list(detectors or []))
    if symptom and category == "unknown":
        category = _infer_category_from_symptom(symptom.lower())
    return category


def _fake_step(*, detectors: list[str] | None = None, symptom: str | None = None) -> AgentStep:
    summaries = [{"cell_name": "none-none", "verdict": "fail",
                  "failure_detectors_fired": detectors or []}]
    return FakeLLMProposer().propose(
        symptom=symptom, cell_summaries=summaries, candidates=["tf32_off"], tried=[]
    )


class TestTheOfflineHeuristic:
    """`FakeLLMProposer` labels with no model at all, so it needs the new names too."""

    @pytest.mark.parametrize(
        ("detector", "expected"),
        [
            # Pre-existing mappings, pinned so the new branches cannot shadow them.
            ("tier2:hang", "rccl_hang"),
            ("tier4:hip_error", "illegal_mem"),
            ("tier1:exit_nonzero", "launch_error"),
            # `tier4:nan_signature` has shipped since before the set had a home
            # for it, and fell through to `unknown` every time it fired.
            ("tier4:nan_signature", "numeric_silent"),
            ("consan:data_race", "gpu_race"),
            ("waitcheck:missing_waitcnt", "gpu_race"),
        ],
    )
    def test_detectors_route_to_the_right_label(self, detector, expected):
        assert _label(detectors=[detector]) == expected

    def test_barrier_evidence_is_a_gpu_race_not_a_checkpoint_race(self):
        """A barrier here is a GPU barrier -- ConSan sites, barrier patching.

        Routing it to `checkpoint_race` is how intra-kernel evidence ended up
        under a checkpoint-I/O label in the first place.
        """
        assert _label(detectors=["consan:barrier_unpatched"]) == "gpu_race"

    def test_checkpoint_still_wins_for_checkpoint_evidence(self):
        assert _label(detectors=["checkpoint:race"]) == "checkpoint_race"

    @pytest.mark.parametrize(
        ("symptom", "expected"),
        [
            ("loss is NaN after 200 steps", "numeric_silent"),
            ("data race reported in the reduction kernel", "gpu_race"),
            ("race condition on the LDS tile", "gpu_race"),
        ],
    )
    def test_symptom_text_routes_to_the_right_label(self, symptom, expected):
        assert _label(symptom=symptom) == expected

    def test_a_traceback_is_not_a_race(self):
        """`race` is a substring of `traceback`, a plausible symptom word."""
        assert _label(symptom="python traceback at startup") != "gpu_race"

    @pytest.mark.parametrize(
        "symptom",
        [
            "canonical layout mismatch in the reduction kernel",
            "kernel launch overhead measured in nanoseconds",
            "maintenance window truncated the run",
        ],
    )
    def test_nan_inside_another_word_is_not_numeric_silent(self, symptom):
        """Same footgun as `race` inside `traceback`, on the symptom branch.

        `"nan" in "canonical"` holds, and this branch only runs once the
        detectors have fallen through to `unknown` -- exactly when a stray
        match is what decides the label.
        """
        assert _label(symptom=symptom) != "numeric_silent"

    @pytest.mark.parametrize(
        "symptom",
        ["loss went to nan at step 40", "NaNs in the gradients after the all-reduce"],
    )
    def test_nan_as_a_word_still_routes(self, symptom):
        """Including the plural, which is how it tends to be written."""
        assert _label(symptom=symptom) == "numeric_silent"

    @pytest.mark.parametrize(
        ("symptom", "expected"),
        [
            # A checkpoint race is a checkpoint race. With no checkpoint leg on
            # this branch at all, the phrase was landing on `gpu_race`.
            ("checkpoint save race condition", "checkpoint_race"),
            # Names a kernel, so the phrase is evidence of *where*.
            (
                "nondeterministic data race in the reduction kernel",
                "gpu_race",
            ),
            ("data race on the lds tile", "gpu_race"),
            ("consan reports a data race", "gpu_race"),
            # Forms the first version of this guard missed, because it required
            # the literal phrases "data race" / "race condition".
            ("LDS race", "gpu_race"),
            ("kernel race", "gpu_race"),
            ("waitcheck reports a missing s_waitcnt hazard", "gpu_race"),
            # No kernel and no tool named: understate rather than assert a
            # hazard nothing observed.
            ("intermittent data race in the host queue", "unknown"),
            ("race condition between the two writer threads", "unknown"),
            # `"lds" in "fields"` holds, so this satisfied the location test by
            # accident -- the same substring collision as `race` inside
            # `traceback`, reintroduced in the fix for it. Word boundaries now.
            ("data race between fields", "unknown"),
            # No race phrase and no kernel named: with `nondeterminism` held
            # out of the vocabulary this falls through, which is the honest
            # answer rather than a downgrade of one.
            ("nondeterministic eval scores between runs", "unknown"),
        ],
    )
    def test_a_race_phrase_needs_a_kernel_or_a_tool(self, symptom, expected):
        """`gpu_race` is a claim about *where*, so the phrase is not enough.

        Mirrors `_is_kernel_race_id` on the detector path, where an evidence
        token needs a sanitizer token beside it. Both paths now demand the same
        thing, which is why the kernel-race leg can sit ahead of the broad
        legs instead of being shadowed by them.
        """
        assert _label(symptom=symptom) == expected

    def test_the_traceback_detector_is_not_a_race(self):
        """The same substring collision, but reached through a shipping detector ID.

        `tier4:python_traceback` is a real detector
        (`probe/classifier/tier4_patterns.py`), so this path fires on ordinary
        Python failures rather than only on a symptom someone typed.
        """
        assert _label(detectors=["tier4:python_traceback"]) != "gpu_race"

    def test_race_evidence_still_wins_beside_a_traceback(self):
        """Suppressing the collision must not suppress the real label."""
        category = _label(detectors=["tier4:python_traceback", "consan:data_race"])
        assert category == "gpu_race"

    @pytest.mark.parametrize(
        "detector",
        [
            # A collective that never arrived, not a hazard inside a kernel.
            "custom:distributed_barrier_timeout",
            # The sanitizer itself falling over reports nothing about a race.
            "custom:consan_tool_failure",
            # A race, and not one inside a kernel. This is why `race` does not
            # stand alone either: it says there *was* a race, not where, and
            # `gpu_race` is a claim about where.
            "custom:host_data_race",
        ],
    )
    def test_an_ambiguous_custom_id_is_not_a_gpu_race(self, detector):
        """`custom:*` ids are free-form, so `barrier` and a bare sanitizer name
        are not evidence on their own.

        `tier5_custom.py` builds `custom:<raw_id>` from whatever a recipe named,
        so a substring test over them asserts a sanitizer-confirmed hazard that
        nothing observed -- the same class of miss as `race` inside `traceback`.
        """
        assert _label(detectors=[detector]) != "gpu_race"

    @pytest.mark.parametrize(
        "detector",
        [
            # The reachable spelling: tier 5 is the only route a sanitizer
            # finding has into `failure_detectors_fired`, and it prefixes
            # `custom:`. The `consan:`/`waitcheck:` ids used elsewhere in this
            # file are not valid detector prefixes at all.
            "custom:consan_data_race",
            "custom:consan_barrier_unpatched",
            "custom:waitcheck_lds_hazard",
            "custom:waitcheck_missing_waitcnt",
        ],
    )
    def test_a_real_sanitizer_custom_id_is_a_gpu_race(self, detector):
        """Narrowing the leg must not make it unreachable.

        Requiring a literal `consan:` / `waitcheck:` prefix would have done
        exactly that: `KNOWN_DETECTOR_PREFIXES` is tier1..tier4 plus custom, so
        no detector the classifier can emit carries those prefixes, and the leg
        would have been dead for every id that can actually fire.
        """
        assert _label(detectors=[detector]) == "gpu_race"

    def test_a_memory_race_reported_by_a_sanitizer_is_a_gpu_race(self):
        """The reported instance: a ConSan race report, labelled an illegal access.

        `custom:consan_global_memory_race` satisfies `_is_kernel_race_id`
        exactly -- a sanitizer token and an intra-kernel one, on one id -- and
        never reached that leg, because `"memory" in joined` decided it several
        legs earlier. The narrowest spelling available for this finding was the
        one the function got wrong.
        """
        category = _label(detectors=["custom:consan_global_memory_race"])
        assert category == "gpu_race"

    @pytest.mark.parametrize(
        "detector,expected",
        [
            # `memory`, against each of the three legs that used to sit behind
            # it. It is the most generic word any leg keys on, and an
            # intra-kernel race, a numerics fault and a checkpoint fault are all
            # describable as being about memory, so it shadowed all three.
            ("custom:consan_shared_memory_race", "gpu_race"),
            ("custom:memory_numerics_mismatch", "numeric_silent"),
            ("custom:checkpoint_memory_fault", "checkpoint_race"),
            # `hang` and `rccl`, which had the same relationship to the same
            # three legs one tier further up.
            ("custom:rccl_consan_data_race", "gpu_race"),
            ("custom:hang_nan_signature", "numeric_silent"),
            ("custom:rccl_checkpoint_stall", "checkpoint_race"),
            # And `oom`.
            ("custom:oom_consan_data_race", "gpu_race"),
            ("custom:oom_numerics_mismatch", "numeric_silent"),
        ],
    )
    def test_a_broad_leg_does_not_answer_for_a_narrower_one(self, detector, expected):
        """The defect class behind the instance above, not the instance.

        `custom:consan_global_memory_race` was reported, and it is one of
        twenty-eight combinations: every leg keyed on a generic single word sat
        ahead of all three legs that demand more evidence -- the two exact
        multi-word signatures and the per-id conjunction -- so each of those
        four broad legs answered for each of those three narrow ones.

        Reordering by how much evidence a leg demands fixes the class. This
        parametrisation is what stops a later leg being appended rather than
        placed, which is how the ordering came to be accidental in the first
        place.
        """
        assert _label(detectors=[detector]) == expected

    @pytest.mark.parametrize(
        "symptom,expected",
        [
            # The two forms reported on this path.
            ("NaN in device memory", "numeric_silent"),
            ("global memory race in the kernel", "gpu_race"),
            # `memory` against each leg that used to sit behind it, then `hang`
            # and `oom` likewise -- the same broad/narrow grid as the detector
            # path's, on the chain that mirrors it.
            ("illegal access and nondeterministic output", "illegal_mem"),
            ("checkpoint save stalled, memory pinned", "checkpoint_race"),
            ("hang with NaN losses", "numeric_silent"),
            ("rccl collective and a checkpoint race", "checkpoint_race"),
            ("nccl ring plus an LDS race in the kernel", "gpu_race"),
            ("oom after nondeterministic scores", "oom_fragment"),
            ("oom and a NaN gradient", "numeric_silent"),
            # And the broad legs still answer when nothing narrower applies, so
            # the reorder has not turned them off.
            ("process hang on the collective", "rccl_hang"),
            ("oom killer took the process", "oom_fragment"),
            ("illegal memory access", "illegal_mem"),
        ],
    )
    def test_a_broad_symptom_leg_does_not_answer_for_a_narrower_one(
        self, symptom, expected
    ):
        """The detector path's defect, in the chain that mirrors it.

        `_infer_category_from_detectors` was reordered by how much evidence each
        leg demands; this chain was not, and had the identical shape -- `memory`,
        `hang` and `oom` deciding any symptom containing them, ahead of the NaN,
        checkpoint and kernel-race legs. Twenty-four broad/narrow
        pairs here against twenty-eight there.

        The pair is the point. These are two parallel dispatchers over one
        taxonomy, so a class fixed in one is a class still open in the other
        until someone looks, and this one was found by review rather than by the
        sweep that fixed its twin. There are exactly two such chains.
        """
        assert _label(symptom=symptom) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "non-deterministic eval scores",
            "non deterministic eval scores",
            "nondeterministic eval scores",
        ],
    )
    def test_the_nondeterminism_spellings_fall_through_while_the_name_is_held(
        self, text
    ):
        """Provisional, and recorded here so restoring the leg is one edit.

        This chain had a `non[- ]?determin` leg reached through all three
        spellings -- `"nondetermin" in "non-deterministic"` is false, and the
        hyphenated form is the conventional English one, so the original
        `nondetermin` missed the spelling people are likeliest to write. The
        leg is out because `nondeterminism` is not in the vocabulary: the CIA
        port widened the same taxonomy independently and merged first, and
        whether this name joins `PROBE_CATEGORIES` is an open question on the
        PR rather than ours to answer alone.

        Falling through to `unknown` is the honest behaviour meanwhile -- a
        name the closed set does not contain cannot be asserted. If the name
        comes back, all three spellings must route again, which is what this
        pins.

        The sweep that found the hole stands and is unaffected: of every
        literal this file matches by substring, exactly four had a morpheme
        boundary a separator could fall on, and the other three
        (`nan_signature`, `numerics_mismatch`, `hip_error`) are on the detector
        path and still covered, one test down.
        """
        assert _label(symptom=text) == "unknown"

    @pytest.mark.parametrize(
        "detector,expected",
        [
            # `custom:<raw_id>` is free-form, so the hyphenated id is as legal
            # as the underscored one and means the same thing.
            ("custom:nan-signature", "numeric_silent"),
            ("custom:nan signature", "numeric_silent"),
            ("custom:numerics-mismatch", "numeric_silent"),
            ("custom:hip-error", "illegal_mem"),
            # The underscored spellings must keep working: this widens the
            # match, it does not move it.
            ("tier4:nan_signature", "numeric_silent"),
            ("custom:ts_kernel_numerics_mismatch", "numeric_silent"),
            ("tier4:hip_error", "illegal_mem"),
        ],
    )
    def test_a_separated_spelling_still_matches_on_the_detector_path(
        self, detector, expected
    ):
        """The other three of the four, found by sweeping rather than by report.

        These three are matched deliberately across a separator, which is why
        they were the only detector-path literals at risk: a token that spans a
        boundary can be spelled with a different separator, and `tier5_custom.py`
        emits whatever the recipe named.
        """
        assert _label(detectors=[detector]) == expected

    @pytest.mark.parametrize(
        "detector,expected",
        [
            # The reported spellings. `custom_patterns[*].id` is validated as
            # "non-empty string" and nothing more (`tier5_custom.py`), so a
            # colon or a slash is exactly as legal as an underscore.
            ("custom:nan:signature", "numeric_silent"),
            ("custom:numerics/mismatch", "numeric_silent"),
            ("custom:hip:error", "illegal_mem"),
            # Not reported, and the point of not enumerating separators: any
            # run of non-alphanumerics is one, so there is no list to be
            # missing from.
            ("custom:nan.signature", "numeric_silent"),
            ("custom:numerics|mismatch", "numeric_silent"),
            ("custom:hip--error", "illegal_mem"),
            ("custom:nan::signature", "numeric_silent"),
        ],
    )
    def test_any_separator_is_a_separator_on_a_free_form_id(self, detector, expected):
        """`[-_ ]` was an allowlist, and the id field it reads is free-form.

        The previous round widened these three legs from a literal `_` to
        `[-_ ]`, which fixed the hyphen that was reported and left every other
        punctuation mark out. Splitting on `[^a-z0-9]+` instead makes the
        question "is there a boundary here", which has no list to keep
        up to date.
        """
        assert _label(detectors=[detector]) == expected

    @pytest.mark.parametrize(
        "detector",
        [
            "custom:chip_error",
            "custom:chip-error",
            "custom:gpu_chip_error",
            "custom:whip_error",
        ],
    )
    def test_a_signature_word_glued_inside_another_word_is_not_that_signature(
        self, detector
    ):
        """Found by the separator sweep, and the reason it is a word test.

        `hip[-_ ]error` matched `custom:chip_error`, so a chip error was
        labelled an illegal memory access; generalising the separator class on
        its own keeps every one of these, because "chip" ends in "hip". Same
        collision as `race` inside `traceback`, `lds` inside `fields` and `nan`
        inside `canonical`, which this file has already fixed three times the
        same way.
        """
        assert _label(detectors=[detector]) != "illegal_mem"

    @pytest.mark.parametrize(
        "detectors",
        [
            ["custom:nan", "signature"],
            ["custom:chip", "error"],
            ["custom:numerics", "mismatch_report"],
        ],
    )
    def test_a_signature_is_not_assembled_from_two_ids(self, detectors):
        """The other half of judging per id, extended to the signature legs.

        `_is_kernel_race_id` has been per-id since the conjunction landed;
        these three legs were still regexes over `" ".join(detectors)`, where
        the joining space is itself a separator. Every shipped id carries a
        `tierN:` / `custom:` prefix and so cannot begin with the second half of
        a pair, which is why nothing has been mislabelled yet -- but that is an
        invariant of the classifier propping this function up, and deciding per
        id does not need it to hold.
        """
        assert _label(detectors=detectors) == "unknown"

    @pytest.mark.parametrize(
        "detector,expected",
        [
            # Every spelling the previous rounds pinned, re-asserted here
            # because this change moved the mechanism underneath them.
            ("tier4:nan_signature", "numeric_silent"),
            ("custom:nan-signature", "numeric_silent"),
            ("custom:nan signature", "numeric_silent"),
            ("custom:ts_kernel_numerics_mismatch", "numeric_silent"),
            ("tier4:hip_error", "illegal_mem"),
            ("custom:hip-error", "illegal_mem"),
            ("custom:rocm_hip_error", "illegal_mem"),
            # Surrounding words on either side are still fine: it is one word
            # next to another that is being asked for, not the whole id.
            ("custom:gpu_nan_signature_v2", "numeric_silent"),
        ],
    )
    def test_the_spellings_that_already_matched_still_match(self, detector, expected):
        """Narrowness control for the word test, in the direction that matters.

        The word split gives up exactly one kind of spelling -- a signature
        word glued to another word, `custom:isnan_signature` -- and these pin
        that nothing else went with it.
        """
        assert _label(detectors=[detector]) == expected

    def test_the_shipping_numerics_detector_routes(self):
        """The one detector in the tree that means this was falling through.

        `recipes/tokenspeed/tokenspeed-kernel-gemm-smoke.yaml` declares a
        tier-5 detector `ts_kernel_numerics_mismatch` on
        `TS_KERNEL_FAIL: numerics_mismatch`, which arrives as
        `custom:ts_kernel_numerics_mismatch`. An out-of-tolerance kernel result
        is exactly what `numeric_silent` names, and matching only the
        built-in `tier4:nan_signature` left it at `unknown` -- the gap this PR
        exists to close, in the category it adds.
        """
        category = _label(detectors=["custom:ts_kernel_numerics_mismatch"])
        assert category == "numeric_silent"

    def test_the_numerics_detector_is_declared_by_a_committed_recipe(self):
        """Pins the premise, not just the routing.

        The routing test above is only meaningful while that detector id is
        real; if the recipe renames it, this fails rather than the mapping
        quietly covering nothing.
        """
        recipe = REPO_ROOT / "recipes" / "tokenspeed" / "tokenspeed-kernel-gemm-smoke.yaml"
        assert "ts_kernel_numerics_mismatch" in recipe.read_text(encoding="utf-8")

    def test_two_half_matches_on_different_ids_do_not_combine(self):
        """Judged per id, not over the joined string.

        Scanning the concatenation lets one id supply `barrier` and another
        supply `consan`, producing a finding neither of them reports.
        """
        category = _label(
            detectors=["custom:distributed_barrier_timeout", "custom:consan_tool_failure"]
        )
        assert category != "gpu_race"


class TestTheProbeSaysOnlyWhatItMayAssert:
    """The seam between what the evidence says and what a probe step may claim.

    Both dispatchers can reach an evidence-only label -- a ConSan detector id
    really does say `gpu_race`, and `tier4:nan_signature` really does say
    `numeric_silent`. `AgentPolicy` validates a probe step against
    `PROBE_CATEGORIES`, so emitting one would raise `PolicyViolation` inside
    `run_agent_loop` rather than mislabel anything, and the loop would stop on
    a correct reading of the evidence.
    """

    @pytest.mark.parametrize(
        "detectors,identified",
        [
            (["custom:consan_data_race"], "gpu_race"),
            (["tier4:nan_signature"], "numeric_silent"),
        ],
    )
    def test_an_evidence_only_reading_is_downgraded_not_emitted(
        self, detectors, identified
    ):
        assert _label(detectors=detectors) == identified
        assert _fake_step(detectors=detectors).category == "unknown"

    @pytest.mark.parametrize(
        "detectors,expected",
        [
            (["custom:consan_data_race"], "gpu_race"),
            (["tier4:nan_signature"], "numeric_silent"),
            (["tier2:hang"], "rccl_hang"),
            (["tier4:hip_error"], "illegal_mem"),
        ],
    )
    def test_every_step_the_proposer_emits_survives_the_policy(
        self, detectors, expected
    ):
        """The property that matters, stated over the proposer rather than a list.

        Reading the evidence correctly is pinned above; this pins that doing so
        never produces a step the loop will refuse. Both are needed -- the
        downgrade could satisfy the policy by answering `unknown` to
        everything, and the first test is what stops that passing.
        """
        assert _label(detectors=detectors) == expected
        step = _fake_step(detectors=detectors)
        assert AgentPolicy().validate_step(step).category == step.category

    def test_the_downgrade_is_not_a_blanket_unknown(self):
        """Narrowness control: probe-reachable labels still come through."""
        assert _fake_step(detectors=["tier2:hang"]).category == "rccl_hang"
        assert _fake_step(symptom="oom killer took the process").category == "oom_fragment"


class TestTheDocsAgree:
    @pytest.mark.parametrize("category", sorted(AUTOPSY_CATEGORIES))
    def test_the_operator_doc_lists_every_label(self, category):
        """The taxonomy table is what a human reads; it drifts silently otherwise.

        Pinned to a table *row*, not to the name appearing anywhere. The gloss
        prose under the table names the labels too, so a bare
        `` `gpu_race` `` search passed with the row deleted -- the drift this
        test exists to catch was the one shape it could not see.
        """
        assert f"| `{category}` |" in AGENT_DOC.read_text(encoding="utf-8")


class TestTheCorpusGroundTruth:
    """`scenario_labels.json` is the training signal the widened set exists for."""

    @staticmethod
    def _labels() -> dict:
        return json.loads(SCENARIO_LABELS.read_text(encoding="utf-8"))["scenarios"]

    def test_every_scenario_is_labelled_from_the_closed_set(self):
        offenders = {
            name: entry["category"]
            for name, entry in self._labels().items()
            if entry["category"] not in AUTOPSY_CATEGORIES
        }
        assert offenders == {}

    def test_every_scenario_names_a_recipe_that_exists(self):
        missing = [
            entry["recipe"]
            for entry in self._labels().values()
            if not (REPO_ROOT / entry["recipe"]).is_file()
        ]
        assert missing == []

    def test_every_scenario_says_why(self):
        assert [n for n, e in self._labels().items() if not e.get("why", "").strip()] == []

    def test_the_lds_race_is_labelled_a_gpu_race(self):
        """The one scenario the model labelled, and labelled wrong.

        It drew `checkpoint_race` because that was the nearest available name.
        """
        assert self._labels()["consan-racy"]["category"] == "gpu_race"

    def test_the_coverage_note_counts_the_rows_the_map_actually_has(self):
        """The note explains away the `unknown` rows, so its count has to be theirs.

        It reads "three ... and six", and it reached review saying "five
        controls" plus "two more" harness errors, which totals seven against
        six `unknown` rows -- `consan-tiny` is in both groups. The prose is now
        written to say so, and this fails if either number drifts from the map
        again, which is the only part a reader cannot check at a glance.
        """
        from collections import Counter

        counts = Counter(e["category"] for e in self._labels().values())
        assert counts == {"gpu_race": 3, "tooling_gap": 2, "unknown": 4}

        note = " ".join(
            json.loads(SCENARIO_LABELS.read_text(encoding="utf-8"))["coverage_note"]
        )
        assert "three are `gpu_race`, two are `tooling_gap` and four are" in note
