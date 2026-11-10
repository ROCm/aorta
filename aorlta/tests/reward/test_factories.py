"""Each stand-in adapter is built from its own settings in the profile (a regression: late-bound
closures once gave every factory the last adapter's settings)."""

import os

from aorlta.reward.clock import Clock
from aorlta.reward.factories import standin_factories
from aorlta.reward.profiles import load_profile


def test_every_factory_reads_its_own_settings():
    profile = load_profile("standin-gpu")
    f = standin_factories(profile, Clock(), timeouts={"gate": 60, "compile": 120, "sim_run": 120, "asan_request": 900})
    assert set(f) == {"standin.parse", "standin.hipcc", "standin.rj_waitcheck", "standin.aorta_consan", "standin.aorta_asan"}
    hipcc = f["standin.hipcc"]("compile/1")
    wc = f["standin.rj_waitcheck"]("static/1")
    consan = f["standin.aorta_consan"]("consan/1")
    asan = f["standin.aorta_asan"]("asan/1")
    cfg = profile.adapters
    assert hipcc.hipcc == cfg["standin.hipcc"]["hipcc"]
    assert wc.binary == cfg["standin.rj_waitcheck"]["binary"]
    assert consan.hook == cfg["standin.aorta_consan"]["hook"]
    assert consan.control_source == cfg["standin.aorta_consan"]["control_source"]
    assert asan.adapter.control_source == cfg["standin.aorta_asan"]["control_source"]
    assert hipcc.timeout_s == 0.9 * 120 and consan.timeout_s == 0.9 * 120
    for a in (hipcc, wc, consan, asan):
        close = getattr(a, "close", None)
        if close:
            close()


def test_a_site_rocm_root_reaches_every_adapter_that_compiles(monkeypatch):
    monkeypatch.setenv("AORLTA_ROCM_ROOT", "/site/rocm")
    profile = load_profile("standin-gpu")
    f = standin_factories(profile, Clock())
    adapters = [f[name](f"{name}/1") for name in ("standin.hipcc", "standin.rj_waitcheck", "standin.aorta_consan")]
    adapters.append(f["standin.aorta_asan"]("asan/1").adapter)
    try:
        assert {a.hipcc for a in adapters} == {"/site/rocm/bin/hipcc"}
        assert adapters[0].readelf == "/site/rocm/llvm/bin/llvm-readelf"
    finally:
        for a in adapters:
            close = getattr(a, "close", None)
            if close:
                close()


def test_aortas_asan_canary_is_given_the_compiler_the_candidates_are_built_with(monkeypatch):
    """aorta's ASan node-health canary builds with $HIPCC when it is set, else from aorta's own ROCm lookup, which a
    site's AORLTA_ROCM_ROOT does not reach. Building the factories names the candidates' compiler there; a $HIPCC
    already set stands."""
    monkeypatch.setenv("AORLTA_ROCM_ROOT", "/site/rocm")
    monkeypatch.delenv("HIPCC", raising=False)
    standin_factories(load_profile("standin-gpu"), Clock())
    assert os.environ["HIPCC"] == "/site/rocm/bin/hipcc"
    monkeypatch.setenv("HIPCC", "/chosen/hipcc")
    standin_factories(load_profile("standin-gpu"), Clock())
    assert os.environ["HIPCC"] == "/chosen/hipcc"
