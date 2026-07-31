"""Timing-contract tests for AINIC race recipe environment variables."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aorta.triage.recipe import load_recipe
from aorta.workloads.race import RaceWorkload

_ROOT = Path(__file__).resolve().parents[2]
_RECIPE_DIR = _ROOT / "recipes" / "race"
_AINIC_RECIPES = (
    "ainic-smoke.yaml",
    "ainic-gdr-flush-sdc-quick.yaml",
    "ainic-gdr-flush-sdc.yaml",
    "ainic-sdc-stress-reproducer.yaml",
)
_GDR_ACTIVATION_VARS = {
    "NCCL_NET_GDR_LEVEL",
    "NCCL_DMABUF_ENABLE",
    "NCCL_NET_GDR_READ",
}


@pytest.mark.parametrize("name", _AINIC_RECIPES)
def test_simple_protocol_cells_carry_gdr_startup_env(name: str) -> None:
    path = _RECIPE_DIR / name
    text = path.read_text(encoding="utf-8")
    document = yaml.safe_load(text)

    for cell in document["cells"]:
        env = cell.get("extra_env", {})
        if env.get("NCCL_PROTO") == "Simple":
            assert _GDR_ACTIVATION_VARS <= set(env), (
                f"{name} cell {cell['name']!r} does not fully enable GDR"
            )
    assert "GDR 1" in text


@pytest.mark.parametrize("name", _AINIC_RECIPES)
def test_ainic_race_recipe_still_loads(name: str) -> None:
    recipe = load_recipe(_RECIPE_DIR / name)
    assert recipe.workload == "race"
    assert recipe.trial_isolation == "auto"
    assert recipe.cells


def test_race_workload_requires_process_isolation() -> None:
    assert RaceWorkload.trial_isolation_default == "process"
    assert RaceWorkload.trial_isolation_required is True


def test_stress_recipe_omits_unimplemented_fsdp_controls() -> None:
    document = yaml.safe_load(
        (_RECIPE_DIR / "ainic-sdc-stress-reproducer.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert "reuse_buffers" not in document["workload_config"]
    assert "same_stream_mode" not in document["workload_config"]
    assert all(
        "same_stream_mode" not in cell.get("workload_config", {})
        for cell in document["cells"]
    )
    assert {cell["name"] for cell in document["cells"]} == {
        "baseline-stressed",
        "gdr-flush-stressed",
        "hw-queues-2-masked",
        "ll-protocol-control",
    }
    assert document["workload_config"]["expected_local_world_size"] == 1


def test_full_ainic_recipe_records_one_rank_per_host_assumption() -> None:
    document = yaml.safe_load(
        (_RECIPE_DIR / "ainic-gdr-flush-sdc.yaml").read_text(encoding="utf-8")
    )
    assert document["workload_config"]["expected_local_world_size"] == 1


def test_no_race_recipe_attempts_to_disable_required_isolation() -> None:
    for path in (_ROOT / "recipes").rglob("*.yaml"):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get("workload") == "race":
            assert document.get("trial_isolation", "auto") != "in_process", path


def test_llm_tf32_recipe_requests_process_isolation() -> None:
    recipe = load_recipe(_ROOT / "recipes" / "llm-determinism" / "example-llm-determinism.yaml")
    assert recipe.trial_isolation == "process"
