"""The architecture is read, or reported as unread.

``check_gpu_arch`` answered ``gfx90a`` for an MI25x and ``gfx942`` for
everything else -- including an empty probe, an ERROR, a Navi card, and the
MI355X these agents are developed against, which is ``gfx950``. MI300 happened
to be right, which is what kept it alive.

Being wrong was half of it. The other half is that the planner could not tell a
reading from a default: a guess arrived with the same authority as a fact, and
``uncertain_fields`` had nothing to carry because nothing ever said it did not
know.
"""

from __future__ import annotations

import pytest

from aorta.cia.launch import discovery
from aorta.cia.launch.discovery import parse_gpu_arch


class TestWhatTheClusterStates:
    def test_an_explicit_gfx_token_is_taken_at_its_word(self):
        assert parse_gpu_arch("gpu:mi300x:8 gfx942,xgmi") == "gfx942"

    def test_it_is_preferred_over_the_board_name_beside_it(self):
        """A cluster that publishes an arch is stating the answer."""
        assert parse_gpu_arch("MI250X gfx950") == "gfx950"

    def test_the_token_is_normalised(self):
        assert parse_gpu_arch("GFX90A") == "gfx90a"

    @pytest.mark.parametrize(
        "board,arch",
        [
            ("AMD Instinct MI210", "gfx90a"),
            ("AMD Instinct MI250X", "gfx90a"),
            ("AMD Instinct MI300X", "gfx942"),
            ("AMD Instinct MI325X", "gfx942"),
            ("AMD Instinct MI355X", "gfx950"),
        ],
    )
    def test_a_settled_board_name_maps_to_its_arch(self, board, arch):
        assert parse_gpu_arch(f"Card Series: {board}") == arch


class TestWhatItRefusesToGuess:
    @pytest.mark.parametrize(
        "probe,why",
        [
            ("", "an empty probe"),
            ("   \n  ", "whitespace only"),
            ("ERROR: timed out", "a failed probe"),
            ("(null) xgmi36,pod1", "sinfo naming no architecture"),
            ("Card Series: AMD Radeon RX 7900 XTX", "a board outside the table"),
            ("Card Series: AMD Instinct MI999", "a board that does not exist yet"),
        ],
    )
    def test_it_says_nothing_rather_than_gfx942(self, probe, why):
        assert parse_gpu_arch(probe) == "", why

    def test_the_mi355x_this_runs_on_is_not_called_gfx942(self):
        """The regression, on the hardware it was written against."""
        assert parse_gpu_arch("(null) xgmi36,pod1") != "gfx942"
        assert parse_gpu_arch("Card Series: AMD Instinct MI355X") == "gfx950"


class TestTheToolContract:
    @pytest.fixture
    def probed(self, monkeypatch):
        def _install(output: str):
            monkeypatch.setattr(discovery, "run_probe", lambda host, cmd, **k: output)

        return _install

    def test_a_silent_probe_yields_an_empty_arch(self, probed):
        probed("")
        assert discovery.check_gpu_arch("localhost", "node-01")["arch"] == ""

    def test_a_reading_still_comes_through(self, probed):
        probed("gpu:mi300x:8 gfx942")
        result = discovery.check_gpu_arch("localhost", "node-01")
        assert result["arch"] == "gfx942"
        assert result["count"] == 8

    def test_the_output_is_returned_either_way(self, probed):
        """Whatever it could not parse, a reader can still see."""
        probed("(null) xgmi36,pod1")
        result = discovery.check_gpu_arch("localhost", "node-01")
        assert result["arch"] == ""
        assert "xgmi36" in result["output"]


def test_the_signature_tells_the_model_what_an_empty_arch_means():
    """An empty field is only useful if the model knows to flag it."""
    desc = discovery.ClusterProfile.model_fields["gpu_arch"].json_schema_extra["desc"]
    assert "uncertain_fields" in desc
