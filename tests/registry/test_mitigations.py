"""Tests for the mitigations registry."""

import pytest

from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownMitigationError,
)
from aorta.registry.mitigations import BUILTIN_MITIGATIONS, get_mitigation, load_mitigations

# Independently-spelled expected env-var bundles for every built-in
# registered by issue #195. Hand-spelled (NOT derived from BUILTIN_MITIGATIONS)
# so a typo in the registry source -- e.g., HSA_NO_SCRATCH_RECLAM vs
# HSA_NO_SCRATCH_RECLAIM, or "extendable_segments:True" instead of
# "expandable_segments:True" -- fails the parametrised assertion. The companion
# drift test (``test_probe_flag_expected_dict_subset_of_registry``) catches
# removals/renames but does not require set equality with all non-core
# built-ins, so future built-ins added for unrelated reasons aren't forced
# into this probe-flag-specific dict.
PROBE_FLAG_BUILTIN_EXPECTED: dict[str, dict[str, str]] = {
    "amd_log_level_4": {"AMD_LOG_LEVEL": "4"},
    "debug_clr_no_batch_cpu_sync": {"DEBUG_CLR_BATCH_CPU_SYNC_SIZE": "0"},
    "fa_prefer_aotriton": {"TORCH_ROCM_FA_PREFER_CK": "0"},
    "fa_prefer_ck": {"TORCH_ROCM_FA_PREFER_CK": "1"},
    "gpu_force_blit_copy_128": {"GPU_FORCE_BLIT_COPY_SIZE": "128"},
    "gpu_max_hw_queues_2": {"GPU_MAX_HW_QUEUES": "2"},
    "hip_launch_blocking": {"HIP_LAUNCH_BLOCKING": "1"},
    "hsa_no_scratch_reclaim": {"HSA_NO_SCRATCH_RECLAIM": "1"},
    "hsa_no_sdma": {"HSA_ENABLE_SDMA": "0"},
    "nccl_launch_order_implicit": {"NCCL_LAUNCH_ORDER_IMPLICIT": "1"},
    "pytorch_alloc_expandable_segments": {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    },
    "pytorch_no_cuda_memory_caching": {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"},
    "rccl_gfx942_cheap_fence_off": {"RCCL_GFX942_CHEAP_FENCE_OFF": "1"},
    "roc_aql_queue_size_1024": {"ROC_AQL_QUEUE_SIZE": "1024"},
    "roc_signal_pool_16k": {"ROC_SIGNAL_POOL_SIZE": "16384"},
}


def test_get_mitigation_returns_env_vars():
    assert get_mitigation("tf32_off") == {"DISABLE_TF32": "1"}


def test_get_mitigation_none_is_empty_dict():
    assert get_mitigation("none") == {}


def test_get_mitigation_unknown_raises_with_helpful_message():
    with pytest.raises(UnknownMitigationError) as exc:
        get_mitigation("not_a_real_mitigation")
    msg = str(exc.value)
    assert "available:" in msg
    assert "plugin" in msg
    # str() must NOT wrap message in quotes (KeyError's default repr behavior)
    assert not msg.startswith("'") and not msg.endswith("'")


def test_load_mitigations_includes_builtins(fake_eps):
    fake_eps([])
    result = load_mitigations()
    assert "tf32_off" in result
    assert result["tf32_off"].source_package == "aorta"


def test_load_mitigations_discovers_plugin(fake_eps):
    fake_eps([("foo", {"FOO": "1"}, "fake_plugin")])
    result = load_mitigations()
    assert result["foo"].env == {"FOO": "1"}
    assert result["foo"].source_package == "fake_plugin"


def test_collision_between_plugins_raises(fake_eps):
    fake_eps([
        ("foo", {"X": "1"}, "plugin_a"),
        ("foo", {"X": "2"}, "plugin_b"),
    ])
    with pytest.raises(RegistryCollisionError, match="plugin_a.*plugin_b"):
        load_mitigations()


def test_collision_plugin_vs_builtin_raises(fake_eps):
    fake_eps([("tf32_off", {"X": "1"}, "plugin_x")])
    with pytest.raises(RegistryCollisionError, match="aorta.*plugin_x"):
        load_mitigations()


def test_non_string_env_value_raises(fake_eps):
    fake_eps([("bad", {"K": 123}, "plugin_x")])
    with pytest.raises(RegistryError, match="plugin_x.*bad.*dict\\[str, str\\]"):
        load_mitigations()


def test_non_dict_payload_raises(fake_eps):
    fake_eps([("bad", "not-a-dict", "plugin_x")])
    with pytest.raises(RegistryError, match="plugin_x.*bad.*str"):
        load_mitigations()


@pytest.mark.parametrize(
    ("name", "expected_env"),
    sorted(PROBE_FLAG_BUILTIN_EXPECTED.items()),
    ids=sorted(PROBE_FLAG_BUILTIN_EXPECTED),
)
def test_probe_flag_builtin_mitigation_env_bundles(name: str, expected_env: dict[str, str]):
    assert get_mitigation(name) == expected_env


def test_probe_flag_expected_dict_subset_of_registry():
    """One-way drift detector for ``PROBE_FLAG_BUILTIN_EXPECTED``.

    Asserts every name listed here still exists in ``BUILTIN_MITIGATIONS``
    so a registry removal or rename fails this test instead of the more
    confusing parametrised ``UnknownMitigationError`` further down. The
    check is intentionally *one-way*: built-ins added to the registry
    for unrelated reasons should not be forced into this issue-#195
    expected dict, so we don't assert set equality against all non-core
    built-ins (a previous version coupled unrelated registry evolution
    to this dict; see PR #198 round-2 review).

    New probe-flag entries are caught indirectly: anyone adding a
    built-in that belongs in the issue-#195 sweep set is expected to
    add the matching key here as well -- there is no automated way to
    distinguish "probe-flag" built-ins from other future built-ins
    without a curated tag.
    """
    missing = set(PROBE_FLAG_BUILTIN_EXPECTED) - set(BUILTIN_MITIGATIONS)
    assert not missing, (
        "PROBE_FLAG_BUILTIN_EXPECTED references names no longer in "
        f"BUILTIN_MITIGATIONS (removed or renamed?): {sorted(missing)}"
    )
