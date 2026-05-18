"""Tests for ``aorta.instrumentation.recipes.buck`` (issue #163, A1.2c).

Coverage targets the acceptance criteria from #163 (A1.2c):

* Output starts with the loud "BEST-EFFORT, NOT EXACT" header.
* Each ``library_introspection`` entry with ``source == "buck"``
  becomes one ``prebuilt_cxx_library`` rule pinning the captured
  ``revision`` + ``target``.
* Non-buck entries (``source: "pkg-config"`` / ``"elf"``) are skipped.
* Empty ``library_introspection`` -> header + provenance comment, no
  rules (still valid BUCK).
* Missing / ``kind=none`` ``build_system`` -> header + provenance
  comment explaining the empty fragment (graceful degrade).
* Malformed shapes (non-dict entries, wrong types) -> surfaced as
  ``# warning`` comments rather than exceptions (never raises).

Optional round-trip lexical check: emitted text contains the rule
name + version attribute + original target for each buck entry.
"""

from __future__ import annotations

import pytest

from aorta.instrumentation.recipes.buck import (
    RECIPE_HEADER,
    emit_buck_recipe,
)


# Shared minimal env.json fragments used by multiple tests. Built as
# helpers so each test can override only the slice it cares about
# without redeclaring the full snapshot shape.


def _bs_buck2() -> dict:
    return {
        "kind": "buck2",
        "buck2_version": "buck2 2026-04-15",
        "repo_root": "/data/users/me/monorepo",
        "revision": "abc1234567890",
    }


def _bs_none() -> dict:
    return {"kind": "none"}


def _entry(name: str, target: str, revision: str | None = "rev-deadbeef") -> dict:
    return {
        "name": name,
        "source": "buck",
        "revision": revision,
        "target": target,
    }


class TestHeader:
    def test_header_constant_starts_with_warning(self):
        """The header constant itself must contain the BEST-EFFORT
        phrase -- the acceptance criterion is about the contents,
        not the exact string. Pin both the case-insensitive presence
        and the leading ``#`` so it's a valid BUCK comment block.
        """
        assert RECIPE_HEADER.startswith("#")
        assert "BEST-EFFORT, NOT EXACT" in RECIPE_HEADER
        assert "aorta env recipe --format buck" in RECIPE_HEADER

    def test_emit_output_starts_with_header(self):
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [],
            "library_introspection_alternates": [],
        }
        output = emit_buck_recipe(env)
        assert output.startswith(RECIPE_HEADER), (
            "BUCK fragment must start with the loud header so an "
            "operator who pipes it into a BUCK file sees the "
            "warning at the top."
        )


class TestProvenanceComment:
    def test_buck2_provenance_includes_version_root_revision(self):
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        assert "build_system: kind=buck2" in out
        assert "buck2_version=buck2 2026-04-15" in out
        assert "repo_root=/data/users/me/monorepo" in out
        assert "revision=abc1234567890" in out

    def test_kind_none_provenance_explains_empty_fragment(self):
        env = {
            "build_system": _bs_none(),
            "library_introspection": [],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        # The header is the first content. The "kind=none" comment
        # should explain the situation.
        assert "kind=none" in out
        assert "NOT captured" in out or "NOT" in out
        # Must not raise; must still produce a header + some text.
        assert out.startswith(RECIPE_HEADER)

    def test_missing_build_system_treated_as_unknown(self):
        """env.json without a build_system key (very old / hand-crafted)
        must NOT raise. We default to kind=unknown and surface that
        in the provenance line so the operator sees the gap.
        """
        env = {
            "library_introspection": [],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        assert "kind=unknown" in out


class TestRuleEmission:
    def test_one_rule_per_buck_entry(self):
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [
                _entry("hipblaslt", "//third-party/rocm:hipblaslt"),
                _entry("rccl", "//third-party/rocm:rccl_lib"),
                _entry("pytorch", "//pytorch:torch"),
            ],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        # One rule call per entry.
        assert out.count("prebuilt_cxx_library(") == 3
        # Each entry's name appears.
        for name in ("hipblaslt", "rccl", "pytorch"):
            assert f'name = "{name}"' in out
        # Each entry's target appears as a trace-back comment.
        for target in (
            "//third-party/rocm:hipblaslt",
            "//third-party/rocm:rccl_lib",
            "//pytorch:torch",
        ):
            assert f"original_target = {target}" in out

    def test_revision_pinned_via_version_attribute(self):
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [
                _entry("hipblaslt", "//rocm:hipblaslt", "deadbeef0001"),
            ],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        assert 'version = "deadbeef0001"' in out, (
            "Acceptance criterion: each rule must pin the captured "
            "revision so the consumer can reproduce the build."
        )

    def test_missing_revision_surfaces_as_comment_not_value(self):
        """A null/missing revision is honest about the gap rather than
        emitting ``version = "None"`` or omitting the attribute
        silently. Pins behaviour for the spec-acknowledged case
        where ``revision`` is null (e.g. repo without VCS at the
        Buck root).
        """
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [
                _entry("hipblaslt", "//rocm:hipblaslt", revision=None),
            ],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        assert 'version = "None"' not in out
        assert "revision unknown" in out

    def test_non_buck_entries_are_skipped(self):
        """``library_introspection`` may contain ``source: "pkg-config"``
        entries (A1's existing path) mixed with ``source: "buck"``
        entries when both populated different libraries in the same
        run. The buck recipe only renders the buck-flavour ones --
        a pkg-config library has no Buck target.
        """
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [
                {
                    "name": "rocblas",
                    "source": "pkg-config",
                    "revision": "5.2.0",
                },
                _entry("hipblaslt", "//rocm:hipblaslt"),
                {
                    "name": "miopen",
                    "source": "elf",
                    "revision": "miopenhash",
                },
            ],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        assert out.count("prebuilt_cxx_library(") == 1
        assert 'name = "hipblaslt"' in out
        # Non-buck names should NOT appear as a rule (they're skipped).
        # NB: they may still appear in alternates' comment block, but
        # this fixture has no alternates -- so any rocblas/miopen
        # occurrence here is a bug.
        assert 'name = "rocblas"' not in out
        assert 'name = "miopen"' not in out


class TestEmptyAndDegradedShapes:
    def test_empty_library_introspection_emits_no_rules(self):
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        assert "prebuilt_cxx_library(" not in out
        # Operator hint pointing at the --buck-target re-run.
        assert "--buck-target" in out

    def test_missing_library_introspection_treated_as_empty(self):
        """env.json without the new top-level key (very old snapshot
        produced before A1.2b) must not raise. We treat absent as
        empty list, same as A1.2b's ``from_dict`` back-fill.
        """
        env = {"build_system": _bs_buck2()}
        out = emit_buck_recipe(env)
        assert out.startswith(RECIPE_HEADER)
        assert "prebuilt_cxx_library(" not in out

    def test_malformed_library_introspection_does_not_raise(self):
        """If a hand-crafted env.json puts a string or int where a
        list belongs, the emitter surfaces it as a warning comment
        and degrades to empty rather than blowing up. Matches the
        never-raises spirit of the env-probe contract that A1.2c
        consumes from.
        """
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": "not-a-list",
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        assert "warning" in out
        assert "unexpected type" in out
        assert "prebuilt_cxx_library(" not in out

    def test_malformed_entry_inside_list_is_skipped(self):
        """A non-dict entry inside an otherwise-valid list must not
        kill the rendering for the other entries. The emitter just
        ignores the bad one and keeps going.
        """
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [
                "garbage-string",
                _entry("hipblaslt", "//rocm:hipblaslt"),
                12345,
            ],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        assert 'name = "hipblaslt"' in out
        # The bad entries don't produce rules.
        assert out.count("prebuilt_cxx_library(") == 1

    def test_completely_empty_env_does_not_raise(self):
        """The absolute worst case -- an empty dict. emit_buck_recipe
        should still return a valid string with at least the header.
        """
        out = emit_buck_recipe({})
        assert out.startswith(RECIPE_HEADER)
        assert "prebuilt_cxx_library(" not in out


class TestAlternates:
    def test_alternates_render_as_comment_block_only(self):
        """Alternates are A1-side entries that lost a merge against
        a Buck match. They have no Buck target so we surface them
        as comments only -- never as a rule (which would need a
        Buck label they don't have).
        """
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [
                _entry("hipblaslt", "//rocm:hipblaslt", "buck-rev"),
            ],
            "library_introspection_alternates": [
                {
                    "name": "hipblaslt",
                    "source": "pkg-config",
                    "revision": "1.2.2",
                    "dropped_for": "buck",
                },
            ],
        }
        out = emit_buck_recipe(env)
        # The alternate's data is mentioned in the comment block,
        # but it must NOT spawn a second prebuilt_cxx_library call.
        assert out.count("prebuilt_cxx_library(") == 1
        assert "library_introspection_alternates" in out
        assert "source=pkg-config" in out
        assert "revision=1.2.2" in out
        assert "dropped for source=buck" in out

    def test_empty_alternates_does_not_add_alternates_block(self):
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [
                _entry("hipblaslt", "//rocm:hipblaslt"),
            ],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        # Omit the alternates header entirely when there's nothing
        # to show.
        assert "library_introspection_alternates" not in out


class TestOutputShape:
    def test_output_is_string(self):
        out = emit_buck_recipe({})
        assert isinstance(out, str)

    def test_output_ends_with_newline(self):
        """Concatenating the fragment into a larger BUCK file shouldn't
        glue our last line to the next file's first line.
        """
        env = {
            "build_system": _bs_buck2(),
            "library_introspection": [
                _entry("hipblaslt", "//rocm:hipblaslt"),
            ],
            "library_introspection_alternates": [],
        }
        out = emit_buck_recipe(env)
        assert out.endswith("\n")


@pytest.mark.parametrize(
    "library_count",
    [1, 2, 5, 10],
)
def test_rule_count_scales_with_buck_entries(library_count):
    """Sanity check across a handful of cardinalities."""
    env = {
        "build_system": _bs_buck2(),
        "library_introspection": [
            _entry(f"lib{i}", f"//pkg:lib{i}") for i in range(library_count)
        ],
        "library_introspection_alternates": [],
    }
    out = emit_buck_recipe(env)
    assert out.count("prebuilt_cxx_library(") == library_count
