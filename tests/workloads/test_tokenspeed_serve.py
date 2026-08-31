"""Unit tests for the `tokenspeed_serve` online-serving benchmark workload.

Unit-level: no GPU, no Docker, no TokenSpeed container. `shutil.which`, the
`/dev/kfd` probe and `subprocess.run` are all monkeypatched, so what is under
test is the part this workload actually owns -- config validation, container env
and argv construction, parsing of what `tokenspeed bench serve` exports,
aggregation across steps, and the verdict.

The verdict tests carry the most weight. `tokenspeed bench serve` exits 0 no
matter how many requests failed, so "the harness ran" and "the server served" are
different questions, and a plausible-looking set of TTFT/throughput numbers can
be computed from a run that dropped most of its requests. Several tests below
exist specifically to pin the guard that separates the two.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from aorta.workloads import tokenspeed_serve as mod
from aorta.workloads.tokenspeed_serve import TokenSpeedServeWorkload

# The SIGTERM test runs a victim process outside pytest, so it needs the import
# root spelled out rather than inherited from the test session.
_SRC = Path(mod.__file__).resolve().parents[2]

_REQUIRED_METRICS = (
    "image",
    "model",
    "num_prompts",
    "input_len",
    "output_len",
    "bench_steps",
    "warmup_steps",
    "server_log",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "output_throughput",
    "total_token_throughput",
    "tokens_per_sec",
    "completed_total",
    "failed_total",
)


def _bench_doc(
    *,
    completed: int = 32,
    failed: int = 0,
    duration: float = 1.2,
    ttft: float = 46.7,
    throughput: float = 3585.9,
) -> dict:
    """A realistic `tokenspeed bench serve --output-file` export."""
    return {
        "date": "20260826-064255",
        "backend": "openai",
        "label": "step1",
        "model_id": "Qwen/Qwen3-0.6B",
        "tokenizer_id": "Qwen/Qwen3-0.6B",
        "num_prompts": 32,
        "request_rate": "inf",
        "burstiness": 1.0,
        "max_concurrency": 8,
        "duration": duration,
        "completed": completed,
        "failed": failed,
        "total_input_tokens": 16384,
        "total_output_tokens": 4096,
        "request_throughput": 28.0,
        "request_goodput": None,
        "output_throughput": throughput,
        "total_token_throughput": 17929.3,
        "max_output_tokens_per_s": 3571.0,
        "max_concurrent_requests": 32,
        "mean_ttft_ms": 47.0,
        "median_ttft_ms": ttft,
        "std_ttft_ms": 40.4,
        "p50_ttft_ms": ttft,
        "p90_ttft_ms": 51.6,
        "p99_ttft_ms": 194.9,
        "mean_tpot_ms": 1.81,
        "median_tpot_ms": 1.91,
        "std_tpot_ms": 0.34,
        "p99_tpot_ms": 2.0,
        "mean_itl_ms": 1.86,
        "median_itl_ms": 0.002,
        "std_itl_ms": 6.4,
        "p99_itl_ms": 34.8,
        "mean_e2el_ms": 277.4,
        "median_e2el_ms": 282.1,
        "std_e2el_ms": 18.2,
        "p99_e2el_ms": 300.8,
    }


@pytest.fixture(autouse=True)
def _host_ready(monkeypatch):
    """Pretend Docker and a ROCm GPU are present so setup() stays hermetic."""
    monkeypatch.setattr(mod.shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(mod.os, "access", lambda _p, _m: True)


def _make(tmp_path: Path, **cfg) -> TokenSpeedServeWorkload:
    # num_prompts matches `_bench_doc`'s `completed`, so a test that does not
    # care about the served-request audit does not trip it by accident.
    base = {"work_dir": str(tmp_path / "work"), "steps": 1, "num_prompts": 32}
    base.update(cfg)
    return TokenSpeedServeWorkload(base)


def _stub_docker(
    wl: TokenSpeedServeWorkload,
    monkeypatch,
    *,
    docs: list[dict] | None = None,
    exit_code: int = 0,
    stdout: str = "TS_BENCH_METRIC: server_startup_sec=307\nTS_BENCH_RESULT: pass\n",
    timeout: bool = False,
    capture: dict | None = None,
) -> None:
    """Stand in for the container: write the exports it would have written."""

    def fake_run(argv, **kwargs):
        if capture is not None:
            capture["argv"] = argv
            capture["kwargs"] = kwargs
        for index, doc in enumerate(docs or [], start=1):
            path = wl._out_dir / f"bench.{wl._run_token}.step{index}.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
        if timeout:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=1, output=stdout)
        return subprocess.CompletedProcess(argv, exit_code, stdout, "")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)


# --------------------------------------------------------------- packaging


def test_bench_script_is_packaged_and_executable():
    script = mod._SCRIPTS_DIR / mod._BENCH_SCRIPT
    assert script.is_file()
    assert script.stat().st_mode & 0o111, "script must be executable"


def test_bench_script_passes_bash_syntax_check():
    script = mod._SCRIPTS_DIR / mod._BENCH_SCRIPT
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"TS_PORT": "abc"}, "TS_PORT must be a positive integer"),
        ({"TS_PORT": "0"}, "TS_PORT must be between"),
        ({"TS_PORT": "65535"}, "TS_PORT must be between"),
        ({"TS_PORT": "9000", "TS_CONTROL_PORT": "9000"}, "must differ"),
        ({"TS_READY_TIMEOUT": "0"}, "TS_READY_TIMEOUT must be between"),
        ({"TS_TEARDOWN_GRACE": "0"}, "TS_TEARDOWN_GRACE must be between"),
        ({"TS_TEARDOWN_GRACE": "-5"}, "positive integer"),
        # The drain has to finish inside the grace, whichever way it is set.
        # `TS_DRAIN_TIMEOUT` arrives from a mitigation and used to replace the
        # derived value unchecked, so teardown SIGKILLed a gateway mid-drain --
        # the delayed VRAM release the derivation exists to avoid.
        ({"TS_DRAIN_TIMEOUT": "60"}, "TS_DRAIN_TIMEOUT must be between"),
        ({"TS_DRAIN_TIMEOUT": "45"}, "TS_DRAIN_TIMEOUT must be between"),
        ({"TS_DRAIN_TIMEOUT": "0"}, "TS_DRAIN_TIMEOUT must be between"),
        # Same bound, reached through `serve_args` instead. `--drain-timeout` is
        # a serve flag, so the bench-flag guard never saw it.
        (
            {"TS_SERVE_ARGS": '["--drain-timeout", "60"]'},
            "serve_args --drain-timeout must be between",
        ),
        (
            {"TS_SERVE_ARGS": '["--drain-timeout=60"]'},
            "serve_args --drain-timeout must be between",
        ),
    ],
)
def test_bench_script_rejects_unusable_ports_and_timeouts(tmp_path, env, expected):
    """These are documented settings used as arithmetic and `seq` operands.

    Unvalidated, each fails somewhere other than where the mistake is:
    `TS_PORT=abc` aborts inside `$(( PORT + 1 ))` with a bash arithmetic error
    rather than the documented usage exit; `TS_PORT=65535` derives a control port
    of 65536 that cannot be bound, which reads as a server that failed to start;
    a zero timeout leaves the readiness loop empty, so the bench reports a server
    that never became ready without having waited.

    Checked by running the script, not by reading it, because the point is which
    exit and message a caller actually gets.
    """
    script = mod._SCRIPTS_DIR / mod._BENCH_SCRIPT
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env={**os.environ, "TS_OUT_DIR": str(tmp_path), **env},
    )
    assert proc.returncode == 64, proc.stdout + proc.stderr
    assert expected in proc.stdout, proc.stdout


def test_bench_script_validates_before_it_creates_anything(tmp_path):
    """A usage error must not hide behind a side effect.

    The port arithmetic runs before the run area exists, so validating after
    `mkdir` would report an unwritable directory for what is really a bad port.
    """
    out = tmp_path / "never"
    proc = subprocess.run(
        ["bash", str(mod._SCRIPTS_DIR / mod._BENCH_SCRIPT)],
        capture_output=True,
        text=True,
        env={**os.environ, "TS_OUT_DIR": str(out), "TS_PORT": "abc"},
    )
    assert proc.returncode == 64
    assert not out.exists(), "validation must precede creating the run area"


def test_script_documents_every_exit_code_the_workload_maps():
    """The reason table and the script's contract must not drift apart.

    A code the script can return but the workload cannot name would surface as
    the generic "container_failed", losing the actual cause.
    """
    text = (mod._SCRIPTS_DIR / mod._BENCH_SCRIPT).read_text(encoding="utf-8")
    for code in mod._EXIT_REASONS:
        assert f"#   {code}" in text, f"exit code {code} undocumented in script"


def test_workload_is_registered_under_entry_point():
    from aorta.run.discovery import discover_workloads

    assert discover_workloads().get("tokenspeed_serve") is TokenSpeedServeWorkload


def test_declared_isolation_matches_registered_policy():
    """The side-effect-free policy entry point drives --dry-run; a mismatch
    would let a dry run promise an isolation mode the class rejects."""
    from aorta.run.discovery import get_workload_policy

    policy = get_workload_policy("tokenspeed_serve")
    assert policy is not None
    assert policy.default == TokenSpeedServeWorkload.trial_isolation_default
    assert policy.supported == TokenSpeedServeWorkload.trial_isolation_supported


# ------------------------------------------------------------ config errors


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("num_prompts", 0, "num_prompts"),
        ("input_len", 0, "input_len"),
        ("output_len", -1, "output_len"),
        ("num_warmups", -1, "num_warmups"),
        ("warmup_steps", -1, "warmup_steps"),
        ("max_concurrency", 0, "max_concurrency"),
        ("request_rate", 0, "request_rate"),
        ("request_rate", -3.5, "request_rate"),
        ("ready_timeout_sec", 0, "ready_timeout_sec"),
        ("bench_timeout_sec", 0, "bench_timeout_sec"),
        ("metric_percentiles", "0,50", "must be in"),
        ("metric_percentiles", "50,100", "must be in"),
        ("metric_percentiles", "50,abc", "is not a number"),
        ("metric_percentiles", "50,,90", "empty entry"),
        ("port", 0, "1024..65535"),
        ("port", 70000, "1024..65535"),
        # Rejected here because ts_bench_serve.sh rejects it in the container:
        # the engine runs unprivileged and cannot bind a reserved port. Accepting
        # it meant a recipe passed setup() and was then guaranteed to fail with
        # the script's exit 64, after occupying a node, reported as a workload
        # failure rather than as the configuration error it is.
        ("port", 80, "1024..65535"),
        ("control_port", 1023, "1024..65535"),
        ("port", "nope", 'must be an int or "auto"'),
        ("ignore_eos", "yes", "must be a bool"),
        ("serve_args", 5, "must be a string or list"),
        ("gates", [1], "must be a mapping"),
    ],
)
def test_invalid_config_rejected(tmp_path, key, value, match):
    with pytest.raises(ValueError, match=match):
        _make(tmp_path, **{key: value}).setup()


def test_zero_steps_rejected(tmp_path):
    with pytest.raises(ValueError, match="steps"):
        _make(tmp_path, steps=0).setup()


def test_unknown_gate_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown gate"):
        _make(tmp_path, gates={"max_ttft": 5}).setup()


@pytest.mark.parametrize("bound", [0, -1, float("inf")])
def test_gate_bound_must_be_finite_positive(tmp_path, bound):
    with pytest.raises(ValueError, match="finite positive"):
        _make(tmp_path, gates={"max_median_ttft_ms": bound}).setup()


def test_timeout_must_exceed_readiness_budget(tmp_path):
    """A timeout inside the readiness budget kills the container mid-startup and
    reports a timeout for what is really a misconfiguration."""
    with pytest.raises(ValueError, match="must exceed"):
        _make(tmp_path, ready_timeout_sec=600, timeout_sec=300).setup()


def test_unknown_config_key_warns_but_does_not_fail(tmp_path, caplog):
    wl = _make(tmp_path, not_a_real_key=1)
    with caplog.at_level("WARNING"):
        wl.setup()
    assert "not_a_real_key" in caplog.text


def test_request_rate_accepts_inf_spellings(tmp_path):
    for value in ("inf", "INF", float("inf")):
        wl = _make(tmp_path, request_rate=value)
        wl.setup()
        assert wl._request_rate == "inf"


# ------------------------------------------------------------ setup guards


def test_setup_requires_docker(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="'docker' not on PATH"):
        _make(tmp_path).setup()


def test_setup_requires_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.os, "access", lambda _p, _m: False)
    with pytest.raises(RuntimeError, match="no accessible ROCm GPU"):
        _make(tmp_path).setup()


def test_setup_reports_unusable_work_dir(tmp_path, monkeypatch):
    """The NFS-root-squash case: a work_dir that cannot be created must say so,
    since the container mount is what would otherwise fail cryptically."""

    def boom(*_a, **_k):
        raise OSError("Permission denied")

    monkeypatch.setattr(mod.Path, "mkdir", boom)
    with pytest.raises(RuntimeError, match="node-local, writable"):
        _make(tmp_path).setup()


def test_setup_stages_and_syntax_checks_the_script(tmp_path):
    wl = _make(tmp_path)
    wl.setup()
    staged = Path(wl._work_dir) / "scripts" / mod._BENCH_SCRIPT
    assert staged.is_file()
    assert staged.stat().st_mode & 0o111
    original = (mod._SCRIPTS_DIR / mod._BENCH_SCRIPT).read_text(encoding="utf-8")
    assert staged.read_text(encoding="utf-8") == original


def test_setup_rejects_a_corrupt_staged_script(tmp_path, monkeypatch):
    """Guards a truncated copy: it would otherwise fail deep inside the
    container, where the error reads as a container problem."""
    wl = _make(tmp_path)

    def bad_copy(_src, dst):
        Path(dst).write_text("if then fi done", encoding="utf-8")

    monkeypatch.setattr(mod.shutil, "copyfile", bad_copy)
    with pytest.raises(RuntimeError, match="bash -n"):
        wl.setup()


# ------------------------------------------------------- env / argv wiring


def test_mitigation_env_is_forwarded_into_the_container(tmp_path):
    """The whole matrix depends on this: without forwarding, a mitigation cell
    benchmarks the same configuration as baseline and reports a false null."""
    wl = _make(tmp_path, _aorta_trial_env={"HSA_NO_SCRATCH_RECLAIM": "1"})
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    env = wl._container_env()
    assert env["HSA_NO_SCRATCH_RECLAIM"] == "1"


def test_mitigation_env_wins_over_unowned_tokenspeed_vars(tmp_path):
    """A mitigation may still tune TokenSpeed itself -- the guard below is about
    the host/container protocol, not about TS_* as a namespace."""
    wl = _make(tmp_path, _aorta_trial_env={"TS_ATTENTION_BACKEND": "triton"})
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    assert wl._container_env()["TS_ATTENTION_BACKEND"] == "triton"


@pytest.mark.parametrize("key", ["TS_NUM_PROMPTS", "TS_BENCH_STEPS", "TS_RUN_TOKEN", "TS_OUT_DIR"])
def test_mitigation_cannot_redefine_the_host_container_protocol(tmp_path, key):
    """The host computes its audit from its own num_prompts/steps/token and finds
    the exports by globbing on that token. A mitigation that redefined any of
    them would leave the container running one configuration while the host
    audited another -- and the mismatch surfaces as a served-request shortfall on
    a run that was actually healthy, or worse, as exports the host never finds."""
    wl = _make(tmp_path, _aorta_trial_env={key: "999"})
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    with pytest.raises(ValueError, match=key):
        wl._container_env()


def test_mitigation_cannot_relabel_the_load_it_changes(tmp_path):
    """The quietest version of the same bug, and the reason the owned set is
    computed rather than listed.

    `TS_MAX_CONCURRENCY=1` changes the load the container actually applies while
    `_aggregate` keeps reporting the configured value. Nothing fails: the cell
    passes, carrying a number that describes a run that did not happen.
    """
    wl = _make(tmp_path, max_concurrency=8, _aorta_trial_env={"TS_MAX_CONCURRENCY": "1"})
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    with pytest.raises(ValueError, match="TS_MAX_CONCURRENCY"):
        wl._container_env()


def test_every_value_the_workload_sets_is_protected(tmp_path):
    """The guard must cover what the workload sets, not a list someone maintains.

    TS_MAX_CONCURRENCY, TS_NUM_WARMUPS and TS_IGNORE_EOS were all missing from
    the hand-written set while being both configured and reported, so this walks
    the real container env instead of naming keys.
    """
    wl = _make(
        tmp_path, max_concurrency=8, serve_args=["--tp", "2"], bench_args=["--burstiness", "1"]
    )
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    owned = wl._container_env()

    assert "TS_MAX_CONCURRENCY" in owned and "TS_NUM_WARMUPS" in owned
    for key in sorted(owned):
        probe = _make(
            tmp_path,
            max_concurrency=8,
            serve_args=["--tp", "2"],
            bench_args=["--burstiness", "1"],
            _aorta_trial_env={key: "overlay"},
        )
        probe.setup()
        probe._run_token = "tok"
        probe._port, probe._control_port = 8000, 8001
        with pytest.raises(ValueError, match=re.escape(key)):
            probe._container_env()


def test_the_documented_protocol_floor_is_actually_owned(tmp_path):
    """The floor is half of the operative set, so it must not drift into fiction.

    Every key it reserves has to be one the workload really sets under some
    configuration -- otherwise reserving it here forbids a mitigation from
    setting a knob the workload never owned.

    Checked against the union over configurations rather than any single one,
    because no single one sets every key: `max_concurrency` and the ShareGPT
    dataset path are absent by default, which is the whole reason the floor
    exists, and `TS_INPUT_LEN`/`TS_OUTPUT_LEN` are the mirror case -- they are
    sent only for the `random` dataset, since ShareGPT takes its lengths from
    the conversations and the bench CLI never sees them.
    """
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text("[]", encoding="utf-8")
    configs = [
        {"max_concurrency": 4},
        {"max_concurrency": 4, "dataset": "sharegpt", "dataset_path": str(dataset)},
    ]

    owned: set[str] = set()
    for overrides in configs:
        wl = _make(tmp_path, **overrides)
        wl.setup()
        wl._run_token = "tok"
        wl._port, wl._control_port = 8000, 8001
        owned |= set(wl._container_env())

    assert mod._PROTOCOL_ENV_KEYS <= owned, mod._PROTOCOL_ENV_KEYS - owned


def test_cache_dirs_are_redirected_into_the_mount(tmp_path):
    """Under `--user <uid>` the image's /root paths are unwritable, and an unset
    TORCHINDUCTOR_CACHE_DIR makes torch call getpass.getuser(), which raises
    KeyError for a uid with no passwd entry."""
    wl = _make(tmp_path)
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    env = wl._container_env()
    assert env["TORCHINDUCTOR_CACHE_DIR"].startswith("/ts-out")
    assert env["TRITON_CACHE_DIR"].startswith("/ts-out")
    assert env["HF_HOME"] == "/hf-cache"
    assert env["USER"]
    assert env["LOGNAME"] == env["USER"]


def test_username_env_omitted_when_not_running_as_current_user(tmp_path):
    wl = _make(tmp_path, run_as_current_user=False)
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    env = wl._container_env()
    assert "USER" not in env


def test_hf_token_forwarded_by_name_only(tmp_path, monkeypatch):
    """Read from the host env named by `hf_token_env`, so no recipe holds it --
    and kept out of `_container_env`, since everything there is rendered into
    the docker client's argv as `-e KEY=VALUE`."""
    monkeypatch.setenv("MY_HF_TOKEN", "secret-value")
    wl = _make(tmp_path, hf_token_env="MY_HF_TOKEN")
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    assert "HF_TOKEN" not in wl._container_env()
    assert wl._secret_env() == {"HF_TOKEN": "secret-value"}
    argv = wl._docker_argv(wl._container_env())
    assert argv[argv.index("HF_TOKEN") - 1] == "-e"


def test_offline_mode_sets_hub_offline(tmp_path):
    wl = _make(tmp_path, hf_offline=True)
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    env = wl._container_env()
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_max_concurrency_omitted_when_unbounded(tmp_path):
    """An empty --max-concurrency would be parsed as an int by the bench CLI and
    fail; unbounded must mean the var is absent."""
    wl = _make(tmp_path)
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    assert "TS_MAX_CONCURRENCY" not in wl._container_env()


def test_docker_argv_uses_host_network_and_current_user(tmp_path):
    wl = _make(tmp_path)
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    argv = wl._docker_argv({"A": "1"})
    assert argv[:2] == ["docker", "run"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "host"
    assert "--user" in argv
    assert argv[-1] == f"/ts-scripts/{mod._BENCH_SCRIPT}"
    assert "-e" in argv and "A=1" in argv


def test_docker_argv_omits_user_flag_when_disabled(tmp_path):
    wl = _make(tmp_path, run_as_current_user=False)
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    assert "--user" not in wl._docker_argv({})


def test_explicit_port_is_honoured_verbatim(tmp_path):
    """Serving on a different port than the operator pinned would make the run
    untraceable."""
    assert mod._resolve_port(9123) == 9123


def test_auto_port_prefers_the_conventional_neighbour(tmp_path):
    port = mod._resolve_port("auto", near=0, avoid={0})
    assert 1 <= port <= 65535


def test_auto_ports_are_distinct(tmp_path):
    gateway = mod._resolve_port("auto")
    control = mod._resolve_port("auto", avoid={gateway}, near=gateway + 1)
    assert control != gateway


# -------------------------------------------------------- happy-path result


def test_successful_run_reports_serving_metrics(tmp_path, monkeypatch):
    wl = _make(tmp_path, steps=2, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(), _bench_doc()])
    result = wl.run()

    assert result.passed is True
    assert result.failure_count == 0
    assert result.failure_details == []
    assert result.main_work_started is True
    assert result.executed_iterations == 2
    assert result.configured_iterations == 2
    assert result.total_iterations == 2
    assert len(result.step_times_ms) == 2
    for key in _REQUIRED_METRICS:
        assert key in result.metrics, f"missing metric {key}"


def test_step_times_come_from_the_bench_duration(tmp_path, monkeypatch):
    wl = _make(tmp_path, steps=2)
    wl.setup()
    _stub_docker(
        wl,
        monkeypatch,
        docs=[_bench_doc(duration=1.0), _bench_doc(duration=2.0)],
    )
    result = wl.run()
    assert result.step_times_ms == [1000.0, 2000.0]


def test_scalars_are_averaged_across_steps(tmp_path, monkeypatch):
    wl = _make(tmp_path, steps=2)
    wl.setup()
    _stub_docker(
        wl,
        monkeypatch,
        docs=[_bench_doc(throughput=1000.0), _bench_doc(throughput=2000.0)],
    )
    metrics = wl.run().metrics
    assert metrics["output_throughput"] == pytest.approx(1500.0)
    assert metrics["tokens_per_sec"] == pytest.approx(1500.0)


def test_audit_counters_are_summed_not_averaged(tmp_path, monkeypatch):
    """ "How many requests did this trial serve" must not be hidden by a mean."""
    wl = _make(tmp_path, steps=2, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(), _bench_doc()])
    metrics = wl.run().metrics
    assert metrics["completed_total"] == 64
    assert metrics["failed_total"] == 0


def test_per_step_detail_is_retained(tmp_path, monkeypatch):
    wl = _make(tmp_path, steps=2)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(), _bench_doc()])
    metrics = wl.run().metrics
    assert len(metrics["steps"]) == 2
    assert len(metrics["result_files"]) == 2


def test_startup_seconds_parsed_from_stdout(tmp_path, monkeypatch):
    wl = _make(tmp_path)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc()])
    assert wl.run().metrics["server_startup_sec"] == 307.0


def test_config_echo_keys_are_not_aggregated_from_the_export(tmp_path, monkeypatch):
    """Echoed arguments would widen perf.md with columns that cannot vary."""
    wl = _make(tmp_path, steps=1)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc()])
    metrics = wl.run().metrics
    assert "burstiness" not in metrics


def test_ports_are_strings_so_perf_md_does_not_average_them(tmp_path, monkeypatch):
    wl = _make(tmp_path)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc()])
    metrics = wl.run().metrics
    assert isinstance(metrics["gateway_port"], str)
    assert isinstance(metrics["control_port"], str)


def test_non_finite_export_values_are_dropped(tmp_path, monkeypatch):
    """NaN/inf do not round-trip through strict JSON, which matrix.json is.

    Dropping is only the right response for a metric the verdict does not rest
    on; a non-finite *core* metric fails the step instead (see the audit tests).
    """
    doc = _bench_doc()
    doc["std_ttft_ms"] = float("nan")
    doc["p99_itl_ms"] = float("inf")
    wl = _make(tmp_path)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[doc])
    metrics = wl.run().metrics
    assert "std_ttft_ms" not in metrics
    assert "p99_itl_ms" not in metrics


def test_booleans_are_not_treated_as_metrics(tmp_path, monkeypatch):
    doc = _bench_doc()
    doc["some_flag"] = True
    wl = _make(tmp_path)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[doc])
    assert "some_flag" not in wl.run().metrics


# ------------------------------------------------------ the silent-pass guard


def test_failed_requests_fail_the_trial_even_when_the_container_exits_zero(tmp_path, monkeypatch):
    """The core guard. `tokenspeed bench serve` returns 0 with failed>0, so
    trusting the exit code would publish throughput for a broken run."""
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=30, failed=2)], exit_code=0)
    result = wl.run()
    assert result.passed is False
    reasons = {d["reason"] for d in result.failure_details}
    assert "served_request_shortfall" in reasons


def test_completed_shortfall_fails_the_trial(tmp_path, monkeypatch):
    """Requests that never arrive are as invalidating as requests that error."""
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=20, failed=0)])
    result = wl.run()
    assert result.passed is False
    assert any(d["reason"] == "served_request_shortfall" for d in result.failure_details)


def test_a_negative_failure_count_is_a_shortfall_not_a_clean_run(tmp_path, monkeypatch):
    """`failed > 0` read a negative count as "better than none failed".

    So an export claiming `completed == num_prompts` and `failed == -1` passed
    the audit outright, and the trial went green on metrics computed by whatever
    produced the -1. A count below zero is not a stronger version of success, it
    is an export that cannot be believed.
    """
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=32, failed=-1)])
    result = wl.run()
    assert result.passed is False
    assert any(d["reason"] == "served_request_shortfall" for d in result.failure_details)


def test_shortfall_still_measured_so_it_is_not_reported_as_did_not_run(tmp_path, monkeypatch):
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=30, failed=2)])
    assert wl.run().main_work_started is True


def test_unusable_counts_fail_the_trial(tmp_path, monkeypatch):
    doc = _bench_doc()
    doc["completed"] = "many"
    wl = _make(tmp_path)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[doc])
    result = wl.run()
    assert result.passed is False
    assert any(d["reason"] == "result_json_unusable" for d in result.failure_details)


def test_an_export_with_only_request_counts_fails_the_trial(tmp_path, monkeypatch):
    """Auditing the counters alone left a hole: an export carrying nothing but
    completed/failed satisfied the shortfall check, and because the shipped
    recipes gate on nothing, the cell went green having measured no duration, no
    TTFT and no throughput."""
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[{"completed": 32, "failed": 0}])
    result = wl.run()
    assert result.passed is False
    detail = next(d for d in result.failure_details if d["reason"] == "result_json_unusable")
    for name in ("duration", "output_throughput", "median_ttft_ms"):
        assert name in detail["detail"]


@pytest.mark.parametrize(
    "name",
    [
        "duration",
        "output_throughput",
        "request_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
    ],
)
def test_a_non_finite_core_metric_fails_the_trial(tmp_path, monkeypatch, name):
    """A NaN here used to be dropped from the aggregate and otherwise ignored,
    so a step that produced no usable measurement still passed."""
    doc = _bench_doc()
    doc[name] = float("nan")
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[doc])
    result = wl.run()
    assert result.passed is False
    assert any(
        d["reason"] == "result_json_unusable" and name in d["detail"]
        for d in result.failure_details
    )


def test_a_zero_length_step_fails_the_trial(tmp_path, monkeypatch):
    """Whatever throughput a zero-duration step reported is an artefact."""
    doc = _bench_doc(duration=0.0)
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[doc])
    result = wl.run()
    assert result.passed is False
    assert any(
        d["reason"] == "result_json_unusable" and "duration" in d["detail"]
        for d in result.failure_details
    )


@pytest.mark.parametrize(
    "name",
    [
        "duration",
        "output_throughput",
        "request_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_non_positive_core_metric_fails_the_trial(tmp_path, monkeypatch, name, value):
    """Finite is not enough: these have to be positive.

    A step that served requests took time, produced tokens and had a latency, so
    zero or negative is a broken measurement rather than a fast one. The negative
    case is the dangerous one -- it is finite, so it used to pass the audit, and a
    negative latency makes a `max_*` gate read as an improvement.
    """
    doc = _bench_doc()
    doc[name] = value
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[doc])
    result = wl.run()
    assert result.passed is False
    assert any(
        d["reason"] == "result_json_unusable" and name in d["detail"]
        for d in result.failure_details
    ), result.failure_details


def test_an_optional_metric_is_not_aggregated_from_some_steps(tmp_path, monkeypatch):
    """A partial metric must not be published as if it summarised every step.

    Otherwise a `max_p99_tpot_ms` gate over three steps, with the metric in one,
    evaluates that single step while reading as a three-step aggregate -- a gate
    passing on a third of the evidence, which is worse than one reporting the
    metric missing, because the caller was promised the latter.
    """
    docs = [_bench_doc(), _bench_doc(), _bench_doc()]
    docs[0]["p99_tpot_ms"] = 12.0
    for doc in docs[1:]:
        doc.pop("p99_tpot_ms", None)

    wl = _make(tmp_path, num_prompts=32, steps=3)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=docs)
    result = wl.run()

    assert result.passed is True
    assert "p99_tpot_ms" not in result.metrics, "a one-step value was published as an aggregate"
    assert "p99_tpot_ms" in result.metrics["partial_metrics"]
    # Still recoverable per step, so nothing is actually lost.
    assert result.metrics["steps"][0]["p99_tpot_ms"] == 12.0


def test_a_metric_present_in_every_step_is_still_aggregated(tmp_path, monkeypatch):
    """The other half of the rule: complete coverage must aggregate as before."""
    docs = [_bench_doc(), _bench_doc()]
    for i, doc in enumerate(docs):
        doc["p99_tpot_ms"] = 10.0 + i
    wl = _make(tmp_path, num_prompts=32, steps=2)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=docs)
    result = wl.run()
    assert result.metrics["p99_tpot_ms"] == pytest.approx(10.5)
    assert "partial_metrics" not in result.metrics


def test_the_first_failure_iteration_is_zero_based(tmp_path, monkeypatch):
    """`step` is one-based because it comes from the export filename, but
    `first_failure_iteration` is an iteration index and the rest of the codebase
    keeps it in 0..total_iterations-1. Returning the filename's number put a
    single-step failure at 1 with one iteration recorded: out of range."""
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[{"completed": 1, "failed": 0}])
    result = wl.run()

    assert result.passed is False
    assert result.total_iterations == 1
    assert result.first_failure_iteration == 0
    assert 0 <= result.first_failure_iteration < result.total_iterations


def test_tpot_is_not_required_at_a_single_output_token(tmp_path, monkeypatch):
    """TPOT averages inter-token gaps, so at output_len 1 there are none to
    average and an absent value is correct rather than a fault. Requiring it
    would fail a legitimate prefill-only recipe."""
    doc = _bench_doc()
    for key in ("mean_tpot_ms", "median_tpot_ms"):
        doc.pop(key)
    wl = _make(tmp_path, num_prompts=32, output_len=1)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[doc])
    assert wl.run().passed is True


# ------------------------------------------------------------ failure paths


@pytest.mark.parametrize(
    ("exit_code", "reason"),
    sorted(mod._EXIT_REASONS.items()),
)
def test_script_exit_codes_map_to_named_reasons(tmp_path, monkeypatch, exit_code, reason):
    wl = _make(tmp_path)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[], exit_code=exit_code)
    result = wl.run()
    assert result.passed is False
    assert any(d["reason"] == reason for d in result.failure_details)


def test_bring_up_failure_is_reported_as_did_not_run(tmp_path, monkeypatch):
    """A server that never came up measured nothing; folding it into the matrix
    as a data point would be worse than reporting it as did-not-run."""
    wl = _make(tmp_path)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[], exit_code=51)
    result = wl.run()
    assert result.main_work_started is False
    assert result.executed_iterations == 0


def test_missing_export_is_reported(tmp_path, monkeypatch):
    wl = _make(tmp_path)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[], exit_code=0)
    result = wl.run()
    assert result.passed is False
    assert any(d["reason"] == "no_bench_export" for d in result.failure_details)


def test_bringup_failure_does_not_also_blame_the_gate_config(tmp_path, monkeypatch):
    """A server that never came up has nothing to gate. Reporting
    `gate_metric_missing` alongside the bring-up failure would send the reader
    to `percentile_metrics` -- a recipe problem -- when the server is the
    story."""
    wl = _make(tmp_path, gates={"max_median_ttft_ms": 500})
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[], exit_code=51)
    result = wl.run()
    assert result.passed is False
    reasons = {d["reason"] for d in result.failure_details}
    assert "readiness_timeout" in reasons
    assert "gate_metric_missing" not in reasons


def test_partial_steps_are_kept_and_flagged(tmp_path, monkeypatch):
    """The steps that did complete show where a degrading cell fell over."""
    wl = _make(tmp_path, steps=3)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc()], exit_code=53)
    result = wl.run()
    assert result.passed is False
    assert result.executed_iterations == 1
    assert result.configured_iterations == 3
    reasons = {d["reason"] for d in result.failure_details}
    assert "incomplete_steps" in reasons
    assert result.metrics["output_throughput"] > 0


def test_container_timeout_is_reported(tmp_path, monkeypatch):
    wl = _make(tmp_path)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[], timeout=True)
    result = wl.run()
    assert result.passed is False
    assert any(d["reason"] == "container_timeout" for d in result.failure_details)


def test_unparseable_export_is_skipped(tmp_path, monkeypatch):
    wl = _make(tmp_path)
    wl.setup()

    def fake_run(argv, **_kwargs):
        path = wl._out_dir / f"bench.{wl._run_token}.step1.json"
        path.write_text("{not json", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    result = wl.run()
    assert result.passed is False
    assert any(d["reason"] == "no_bench_export" for d in result.failure_details)


def test_exports_from_other_trials_are_not_picked_up(tmp_path, monkeypatch):
    """Token-qualified globbing: one out_dir is shared by every trial in a
    matrix, so a fixed name would let one trial report another's numbers."""
    wl = _make(tmp_path, steps=1)
    wl.setup()
    (wl._out_dir / "bench.someone-else.step1.json").write_text(
        json.dumps(_bench_doc(throughput=99999.0)), encoding="utf-8"
    )
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(throughput=100.0)])
    metrics = wl.run().metrics
    assert metrics["output_throughput"] == pytest.approx(100.0)


# ----------------------------------------------------------------- gates


def test_throughput_gate_breach_fails_the_trial(tmp_path, monkeypatch):
    wl = _make(tmp_path, gates={"min_output_throughput": 10_000})
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(throughput=3585.9)])
    result = wl.run()
    assert result.passed is False
    breach = next(d for d in result.failure_details if d["reason"] == "perf_gate_breached")
    assert breach["metric"] == "output_throughput"
    assert breach["bound"] == 10_000


def test_latency_gate_breach_fails_the_trial(tmp_path, monkeypatch):
    wl = _make(tmp_path, gates={"max_median_ttft_ms": 10})
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(ttft=46.7)])
    assert wl.run().passed is False


def test_satisfied_gates_pass(tmp_path, monkeypatch):
    wl = _make(
        tmp_path,
        gates={"max_median_ttft_ms": 500, "min_output_throughput": 100},
    )
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc()])
    assert wl.run().passed is True


def test_gate_on_an_unreported_metric_fails_loudly(tmp_path, monkeypatch):
    """Silently skipping a gate whose metric is absent would let a recipe
    believe it is gated when it is not."""
    doc = _bench_doc()
    del doc["p99_tpot_ms"]
    wl = _make(tmp_path, gates={"max_p99_tpot_ms": 5})
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[doc])
    result = wl.run()
    assert result.passed is False
    assert any(d["reason"] == "gate_metric_missing" for d in result.failure_details)


def test_every_gate_spec_names_a_real_comparison():
    for gate, (metric, comparison) in mod._GATE_SPECS.items():
        assert comparison in {"min", "max"}, gate
        assert metric


# --------------------------------------------------------------- cleanup


def test_cleanup_keeps_exports_by_default(tmp_path, monkeypatch):
    wl = _make(tmp_path)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc()])
    wl.run()
    wl.cleanup()
    assert list(wl._out_dir.glob("bench.*.json"))


def test_cleanup_removes_exports_when_asked(tmp_path, monkeypatch):
    wl = _make(tmp_path, keep_work_dir=False)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc()])
    wl.run()
    wl.cleanup()
    assert not list(wl._out_dir.glob("bench.*.json"))


def test_cleanup_before_run_is_safe(tmp_path):
    wl = _make(tmp_path, keep_work_dir=False)
    wl.setup()
    wl.cleanup()


# ---------------------------------------------------------------- recipes


def _recipe_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "recipes" / "tokenspeed"


def _discover_recipes() -> tuple[str, ...]:
    """Every committed recipe for this workload, found rather than listed.

    A hand-maintained tuple silently excludes new recipes from all four checks
    below, which is the opposite of what a list like this is for -- the recipe
    most likely to be malformed is the one just added. Selected by declared
    workload so the Phase 1 probe recipes in the same directory stay out.
    """
    names = []
    for path in sorted(_recipe_dir().glob("tokenspeed-serve-*.yaml")):
        if "workload: tokenspeed_serve" in path.read_text(encoding="utf-8"):
            names.append(path.name)
    assert names, "no tokenspeed_serve recipes discovered"
    return tuple(names)


_RECIPES = _discover_recipes()


@pytest.mark.parametrize("name", _RECIPES)
def test_recipe_loads_and_targets_this_workload(name):
    from aorta.triage.recipe import load_recipe

    recipe = load_recipe(_recipe_dir() / name)
    assert recipe.workload == "tokenspeed_serve"
    assert recipe.cells


@pytest.mark.parametrize("name", _RECIPES)
def test_recipe_mitigations_resolve_in_the_registry(name):
    """Resolved against the real registry, not a hardcoded list, so an upstream
    mitigation rename fails here instead of at run time."""
    from aorta.registry import load_mitigations
    from aorta.triage.recipe import load_recipe

    known = load_mitigations()
    recipe = load_recipe(_recipe_dir() / name)
    for cell in recipe.cells:
        for mitigation in cell.mitigations:
            assert mitigation in known, f"{name}: unknown mitigation {mitigation}"


@pytest.mark.parametrize("name", _RECIPES)
def test_recipe_workload_config_is_accepted_by_the_workload(name, tmp_path, caplog):
    """Every committed recipe must survive this workload's own validation --
    a typo'd key or an out-of-range value should not wait for a GPU to surface.

    An unknown key only *warns* at runtime, deliberately: a config carrying a
    key some other tool reads should not be fatal. But that made this gate
    weaker than it looked, because a committed recipe saying `num_prompt`
    passed here and then ran the whole matrix at the default request count --
    silently, with plausible numbers. Warnings are therefore fatal for the
    recipes in this repo, which are ours to keep correct.
    """
    from aorta.triage.recipe import load_recipe

    recipe = load_recipe(_recipe_dir() / name)
    for cell in recipe.cells:
        config = {**recipe.workload_config, **cell.workload_config}
        config["steps"] = cell.steps or recipe.steps
        config["work_dir"] = str(tmp_path / "work")
        caplog.clear()
        with caplog.at_level("WARNING", logger=mod.log.name):
            TokenSpeedServeWorkload(config).setup()
        unknown = [r.getMessage() for r in caplog.records if "unknown workload_config key" in r.getMessage()]
        assert not unknown, f"{name}: {unknown}"


@pytest.mark.parametrize("name", _RECIPES)
def test_recipe_image_is_digest_pinned(name):
    """These recipes carry perf gates, so the image has to be content-addressed.
    A registry can retarget a date tag, and a baseline blessed against one stack
    would then be compared against another with nothing in the report saying so.
    """
    from aorta.triage.recipe import load_recipe

    recipe = load_recipe(_recipe_dir() / name)
    image = recipe.workload_config["image"]
    assert "@sha256:" in image, f"{name}: image {image!r} is not digest-pinned"


def test_the_default_image_is_digest_pinned():
    """Same reasoning as the recipes, for anyone running the workload without
    one -- and it keeps the comment above _DEFAULT_IMAGE honest."""
    assert "@sha256:" in mod._DEFAULT_IMAGE


def test_equal_explicit_ports_are_rejected(tmp_path):
    """The gateway and the control endpoint are separate listeners.

    Two equal explicit values passed validation and were then handed over as
    both ports, so the configuration could not come up -- and it failed as a
    readiness timeout during bring-up, which reads as a slow or broken engine
    rather than as the recipe error it is.
    """
    wl = _make(tmp_path, port=9000, control_port=9000)
    with pytest.raises(ValueError, match="cannot share a port"):
        wl.setup()


def test_auto_resolution_still_yields_distinct_ports(tmp_path, monkeypatch):
    """The other half: rejecting the explicit collision must not disturb auto."""
    wl = _make(tmp_path, port="auto", control_port="auto")
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc()])
    wl.run()
    assert wl._port != wl._control_port


def test_one_explicit_port_beside_an_auto_one_is_allowed(tmp_path):
    """Pinning the gateway and letting the control port float is legitimate."""
    _make(tmp_path, port=9000, control_port="auto").setup()
    _make(tmp_path, port="auto", control_port=9000).setup()


def test_the_render_group_is_added_alongside_video(tmp_path):
    """Passing /dev/dri through is not sufficient under --user.

    Render nodes are commonly group-owned by `render` rather than `video`, so
    `video` alone leaves /dev/dri/renderD* unopenable and the failure surfaces as
    an unhelpful device or HIP init error. host_launch.sh and
    harvest_code_objects.py both add the pair; this now matches them.
    """
    wl = _make(tmp_path)
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    argv = wl._docker_argv(wl._container_env())

    groups = [argv[i + 1] for i, a in enumerate(argv) if a == "--group-add"]
    assert groups == ["video", "render"], argv


@pytest.mark.parametrize(
    "env_key,flag",
    [
        ("TS_SERVE_ARGS", "--port"),
        ("TS_SERVE_ARGS", "--control-port"),
        ("TS_SERVE_ARGS", "--host"),
        ("TS_BENCH_ARGS", "--num-prompts"),
        ("TS_BENCH_ARGS", "--output-file"),
        ("TS_BENCH_ARGS", "--base-url"),
        # The load controls, which are worse than the plumbing above rather than
        # merely equivalent: `--max-concurrency 1` against a configured cap of 8
        # changes the load actually applied while the host still publishes 8, and
        # both request audits pass -- every request completed, none failed -- so
        # the cell goes green describing a run that did not happen. Reserved even
        # at their defaults, where the script appends nothing at all, because the
        # default is what most cells run.
        ("TS_BENCH_ARGS", "--max-concurrency"),
        ("TS_BENCH_ARGS", "--request-rate"),
        ("TS_BENCH_ARGS", "--num-warmups"),
        ("TS_BENCH_ARGS", "--ignore-eos"),
        ("TS_BENCH_ARGS", "--seed"),
    ],
)
def test_bench_script_rejects_extra_args_that_shadow_owned_flags(tmp_path, env_key, flag):
    """Both CLIs take the *last* occurrence of a repeated flag, and the extras
    were appended after the owned ones -- so a caller's value silently won.

    Each failure then pointed somewhere else: `--port` starts the gateway where
    the readiness poll is not looking, so a healthy server reads as one that
    never came up; `--output-file` writes the export where neither the in-container
    audit nor the host's glob looks, so a completed benchmark reads as missing;
    `--num-prompts` runs a count the host does not audit against.
    """
    proc = subprocess.run(
        ["bash", str(mod._SCRIPTS_DIR / mod._BENCH_SCRIPT)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TS_OUT_DIR": str(tmp_path),
            # A JSON array, which is how the workload serializes these: joining
            # a list with spaces destroyed argument boundaries on the way in.
            env_key: json.dumps([flag, "somevalue"]),
        },
    )
    assert proc.returncode == 64, proc.stdout + proc.stderr
    assert f"{env_key} may not set {flag}" in proc.stdout, proc.stdout


@pytest.mark.parametrize(
    "env_key,extra",
    [("TS_SERVE_ARGS", "--tp 2"), ("TS_BENCH_ARGS", "--goodput ttft:200")],
)
def test_bench_script_still_accepts_unowned_extra_args(tmp_path, env_key, extra):
    """The guard is about the flags the workload owns, not about extras at all --
    tuning the engine or the bench through these is the point of having them."""
    proc = subprocess.run(
        ["bash", str(mod._SCRIPTS_DIR / mod._BENCH_SCRIPT)],
        capture_output=True,
        text=True,
        env={**os.environ, "TS_OUT_DIR": str(tmp_path), env_key: extra},
    )
    assert "may not set" not in proc.stdout, proc.stdout


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--name", "other"], "may not set --name"),
        (["--entrypoint", "sh"], "may not set --entrypoint"),
        (["-v", "/tmp:/ts-out"], "may not set -v"),
        (["--volume", "/tmp:/ts-out"], "may not set --volume"),
        (["--network", "none"], "may not set --network"),
        (["--user", "0:0"], "may not set --user"),
        (["--env-file", "/tmp/x"], "may not set --env-file"),
    ],
)
def test_docker_args_cannot_displace_a_generated_option(tmp_path, args, expected):
    """docker takes the last occurrence, and these are spliced in after ours.

    `--name other` is the dangerous one: the container then runs under a name
    `_force_remove_container` does not know, so a timed-out trial leaks a live
    server holding the GPU -- silently undoing the orphan cleanup this class
    advertises. `-v ...:/ts-out` sends the exports where the host does not glob,
    and `--entrypoint` means the bench script never runs at all.
    """
    wl = _make(tmp_path, docker_args=args)
    with pytest.raises(ValueError, match=re.escape(expected)):
        wl.setup()


@pytest.mark.parametrize(
    "spelling",
    ["-e TS_RUN_TOKEN=x", "--env TS_RUN_TOKEN=x", "-eTS_RUN_TOKEN=x", "--env=TS_RUN_TOKEN=x"],
)
def test_docker_args_cannot_smuggle_a_protocol_variable(tmp_path, spelling):
    """Otherwise this is a second route to the desynchronisation the mitigation
    guard rejects -- one that bypasses it entirely, in every spelling docker
    accepts for -e.

    Caught at `setup()`, because the check now consults the declared protocol
    floor and not only the keys this run happens to have populated -- so it no
    longer needs the resolved run token to fire, and a bad recipe fails before a
    node is occupied.
    """
    with pytest.raises(ValueError, match="TS_RUN_TOKEN"):
        _make(tmp_path, docker_args=spelling.split()).setup()


def test_docker_args_still_take_unowned_options(tmp_path):
    """The guard is about displacing generated options, not about docker_args --
    passing extra docker flags is why the field exists."""
    wl = _make(tmp_path, docker_args=["--cpus", "8", "-e", "MY_KNOB=1"])
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    argv = wl._docker_argv(wl._container_env())
    assert "--cpus" in argv and "MY_KNOB=1" in argv


def test_auto_ports_never_collide_even_when_the_kernel_reuses_one(tmp_path, monkeypatch):
    """The ephemeral fallback has to honour `avoid`.

    The gateway's probe socket is closed by the time the control port is
    resolved, so the kernel may hand back that very port -- and the pair would
    then be equal, which ts_bench_serve.sh rejects. A perfectly valid `auto`
    configuration would fail as a usage error, intermittently.
    """
    handed_out = iter([41000, 41000, 41000, 41001])

    class FakeSocket:
        def __init__(self, *a, **k):
            self._port = None

        def bind(self, addr):
            self._port = next(handed_out)

        def getsockname(self):
            return ("127.0.0.1", self._port)

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod.socket, "socket", FakeSocket)
    monkeypatch.setattr(mod, "_port_is_free", lambda port: False)

    gateway = mod._resolve_port("auto")
    control = mod._resolve_port("auto", avoid={gateway}, near=gateway + 1)
    assert gateway == 41000
    assert control != gateway, "the control port must not reuse the gateway's"


def test_sigterm_removes_the_container_before_exiting(tmp_path):
    """SIGTERM does not raise, so `except BaseException` never sees it.

    Under Python's default disposition the interpreter terminates without
    raising, which meant the case most likely to strand a container -- a
    cancelled sweep or an expired budget -- was the one not covered. Run in a
    subprocess because the assertion is about the process actually dying by
    signal after cleaning up.
    """
    marker = tmp_path / "removed.txt"
    script = tmp_path / "victim.py"
    script.write_text(
        "import os, signal, sys, time\n"
        f"sys.path.insert(0, {str(_SRC)!r})\n"
        "from aorta.workloads.tokenspeed_serve import TokenSpeedServeWorkload as W\n"
        f"wl = W({{'work_dir': {str(tmp_path / 'work')!r}, 'steps': 1}})\n"
        "wl._run_token = 'tok'\n"
        f"wl._force_remove_container = lambda: open({str(marker)!r}, 'w').write('removed')\n"
        "with wl._remove_container_on_termination():\n"
        "    print('ready', flush=True)\n"
        "    time.sleep(30)\n"
    )
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)

    assert marker.exists(), "the container was not removed on SIGTERM"
    assert (
        proc.returncode == -signal.SIGTERM
    ), f"expected death by SIGTERM so a supervisor still reads 143, got {proc.returncode}"


def test_an_unbounded_default_still_reserves_the_concurrency_cap(tmp_path):
    """The computed owned set cannot reach a key whose value is absence.

    `max_concurrency` defaults to unbounded and expresses that by setting no
    TS_MAX_CONCURRENCY, so `set(env)` guarded the configured case and left the
    *default* one open: a mitigation setting TS_MAX_CONCURRENCY=1 there had the
    container run capped while `_aggregate` published `max_concurrency: None`.
    Nothing fails, and the reported configuration is not the one that ran --
    the mislabelled pass, reachable only on the default.
    """
    wl = _make(tmp_path, _aorta_trial_env={"TS_MAX_CONCURRENCY": "1"})
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    assert wl._max_concurrency is None, "precondition: the unguarded default"
    with pytest.raises(ValueError, match="TS_MAX_CONCURRENCY"):
        wl._container_env()


def test_a_mitigation_may_still_set_a_knob_the_workload_does_not_own(tmp_path):
    """Unioning the floor must not turn the guard into "reject every TS_*".

    Varying a knob the workload does not set is the point of the matrix, and a
    reserved-key list that overreaches would silently forbid it.
    """
    wl = _make(tmp_path, _aorta_trial_env={"TS_SOME_UNOWNED_KNOB": "1"})
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    assert wl._container_env()["TS_SOME_UNOWNED_KNOB"] == "1"


def test_the_dataset_fields_are_not_reported_as_unknown(tmp_path, caplog):
    """Both are documented and consumed, so every valid ShareGPT cell was warned
    that the fields it depends on were being ignored -- which is both false and
    exactly the kind of message an operator acts on."""
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text("[]", encoding="utf-8")

    with caplog.at_level("WARNING"):
        wl = _make(tmp_path, dataset="sharegpt", dataset_path=str(dataset))
        wl.setup()

    assert "dataset" not in caplog.text or "unknown workload_config" not in caplog.text
    assert wl._dataset == "sharegpt"
    # The mechanism still works for a genuinely unknown key.
    with caplog.at_level("WARNING"):
        _make(tmp_path, definitely_not_a_key=1).setup()
    assert "definitely_not_a_key" in caplog.text


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"request_rate": True}, "request_rate"),
        ({"request_rate": False}, "request_rate"),
        ({"gates": {"max_median_ttft_ms": True}}, "max_median_ttft_ms"),
        ({"gates": {"min_output_throughput": True}}, "min_output_throughput"),
    ],
)
def test_booleans_are_rejected_where_a_number_is_required(tmp_path, config, expected):
    """`bool` subclasses `int`, so `float(True)` is `1.0`.

    A YAML `request_rate: true` therefore ran one request per second, and
    `max_median_ttft_ms: true` installed a real 1 ms gate that every run
    breaches -- in both cases a typo silently became a valid, wrong setting
    instead of the validation error the docstring promises.
    """
    with pytest.raises(ValueError, match=expected):
        _make(tmp_path, **config).setup()


@pytest.mark.parametrize("field", ["serve_args", "bench_args"])
def test_extra_args_reach_the_container_with_their_boundaries_intact(tmp_path, field):
    """The recipe documents these as lists, so they have to arrive as lists.

    They were joined with spaces and word-split again inside the container, so
    one item containing a space became two arguments and a `*` in a value was
    glob-expanded against the container's filesystem. Neither is what the recipe
    asked for, and both change what TokenSpeed was actually told.
    """
    items = ["--extra-body", '{"a": 1}', "--label-suffix", "step*"]
    wl = _make(tmp_path, **{field: items})
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001

    env_key = "TS_SERVE_ARGS" if field == "serve_args" else "TS_BENCH_ARGS"
    assert json.loads(wl._container_env()[env_key]) == items


@pytest.mark.parametrize(
    "env_key,abbreviation,owned",
    [
        ("TS_BENCH_ARGS", "--max-conc", "--max-concurrency"),
        ("TS_BENCH_ARGS", "--request-r", "--request-rate"),
        ("TS_BENCH_ARGS", "--se", "--seed"),
        ("TS_BENCH_ARGS", "--num-prom", "--num-prompts"),
        ("TS_BENCH_ARGS", "--output-f", "--output-file"),
        ("TS_SERVE_ARGS", "--po", "--port"),
        ("TS_BENCH_ARGS", "--max-conc=1", "--max-concurrency"),
    ],
)
def test_an_abbreviated_owned_flag_is_rejected_too(tmp_path, env_key, abbreviation, owned):
    """argparse resolves any unambiguous prefix unless `allow_abbrev=False`.

    An exact-match denylist therefore let `--max-conc 1` through to set
    `--max-concurrency`, which is the mislabeled pass this guard exists to
    stop: the applied load changes while the host publishes the configured cap,
    every request completes, and the cell goes green describing a run that did
    not happen. Count-changing flags fail closed via the shortfall audit; the
    load-shape ones do not.
    """
    proc = subprocess.run(
        ["bash", str(mod._SCRIPTS_DIR / mod._BENCH_SCRIPT)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TS_OUT_DIR": str(tmp_path),
            env_key: json.dumps([abbreviation, "somevalue"]),
        },
    )
    assert proc.returncode == 64, proc.stdout + proc.stderr
    assert f"{env_key} may not set {owned}" in proc.stdout, proc.stdout
    assert "is a prefix of it" in proc.stdout, proc.stdout


@pytest.mark.parametrize(
    "extra",
    [
        # Not a prefix of anything owned -- the owned flag is a prefix of *it*,
        # which argparse treats as a different option entirely.
        "--seed-offset",
        "--num-prompts-per-user",
        # Nothing to do with any owned flag.
        "--extra-body",
        "--served-model-name",
    ],
)
def test_a_flag_that_merely_resembles_an_owned_one_is_allowed(tmp_path, extra):
    """The guard must not become a substring match in the other direction.

    Rejecting anything that shares a prefix with an owned flag would forbid
    legitimate options, which is the failure mode of the joined-string version
    this replaced.
    """
    proc = subprocess.run(
        ["bash", str(mod._SCRIPTS_DIR / mod._BENCH_SCRIPT)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TS_OUT_DIR": str(tmp_path),
            "TS_BENCH_ARGS": json.dumps([extra, "somevalue"]),
        },
    )
    assert "may not set" not in proc.stdout, proc.stdout
    # Got past the guard: outside a container the next check is what stops it,
    # which is the evidence that the flag was accepted rather than rejected.
    assert "not on PATH inside the container" in proc.stdout, proc.stdout


@pytest.mark.parametrize("field", ["serve_args", "bench_args"])
def test_the_script_rebuilds_extra_args_verbatim(tmp_path, field):
    """The other half of the same contract, checked where it is consumed.

    A JSON array on the host is only useful if the script reconstructs the exact
    argv from it, so this drives the decoder out of ts_bench_serve.sh rather than
    trusting that it matches.
    """
    script = Path(mod.__file__).with_name("tokenspeed") / "ts_bench_serve.sh"
    body = subprocess.run(
        ["awk", "/^decode_json_args\\(\\) \\{/,/^\\}$/", str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "decode_json_args" in body, "failed to extract the decoder"

    items = ["--extra-body", '{"a": 1}', "step*", "trailing space "]
    harness = tmp_path / "drive.sh"
    harness.write_text(
        f"{body}\n"
        "declare -a out\n"
        'decode_json_args out TS_TEST "$1"\n'
        'printf "%s\\n" "${#out[@]}"\n'
        'for item in "${out[@]}"; do printf "[%s]\\n" "${item}"; done\n'
    )
    proc = subprocess.run(
        ["bash", str(harness), json.dumps(items)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    lines = proc.stdout.splitlines()
    assert lines[0] == str(len(items)), proc.stdout
    assert lines[1:] == [f"[{item}]" for item in items], proc.stdout


def test_sharegpt_does_not_advertise_a_shape_it_never_ran(tmp_path):
    """ISL/OSL are `random`-only knobs.

    The bench CLI maps them onto `--random-input-len`/`--random-output-len`, and
    ShareGPT takes its lengths from the conversations, so the script drops them.
    Sending them anyway left the container holding a shape it would not use, and
    publishing them labelled the result with a shape the run did not have --
    which a matrix mixing the two datasets would compare as if it meant
    something.
    """
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text("[]", encoding="utf-8")

    wl = _make(tmp_path, dataset="sharegpt", dataset_path=str(dataset))
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    env = wl._container_env()
    assert "TS_INPUT_LEN" not in env
    assert "TS_OUTPUT_LEN" not in env

    metrics = wl._aggregate([], stdout="")
    assert metrics["dataset"] == "sharegpt"
    assert metrics["input_len"] is None
    assert metrics["output_len"] is None

    # `random` is unaffected -- there the recipe's shape is exactly what ran.
    plain = _make(tmp_path)
    plain.setup()
    plain._run_token = "tok"
    plain._port, plain._control_port = 8000, 8001
    assert plain._container_env()["TS_OUTPUT_LEN"] == str(plain._output_len)
    assert plain._aggregate([], stdout="")["output_len"] == plain._output_len


def test_sharegpt_requires_tpot_from_the_export_not_from_output_len(tmp_path, monkeypatch):
    """TPOT is required whenever a second token was emitted.

    Under ShareGPT the recipe's `output_len` is ignored by the run, so deciding
    from it made a step's validity depend on a field that had no effect. The
    export knows: more output tokens than completed requests means at least one
    request had an inter-token interval.
    """
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text("[]", encoding="utf-8")

    # output_len 1 would have waived TPOT under the old rule, but this export
    # shows 64 requests producing 4096 tokens.
    wl = _make(tmp_path, dataset="sharegpt", dataset_path=str(dataset), output_len=1)
    wl.setup()

    doc = _bench_doc(completed=32, failed=0)
    doc["total_output_tokens"] = 4096
    doc.pop("mean_tpot_ms", None)
    doc.pop("median_tpot_ms", None)
    _stub_docker(wl, monkeypatch, docs=[doc])
    result = wl.run()

    unusable = [d for d in result.failure_details if d["reason"] == "result_json_unusable"]
    assert unusable, result.failure_details
    assert "tpot" in unusable[0]["detail"]


def test_tpot_is_not_required_when_eos_may_end_a_request_early(tmp_path, monkeypatch):
    """`ignore_eos: false` is supported, and it unpins the output length.

    Without `--ignore-eos` the model stops at its first EOS token, which for a
    short prompt can be immediately -- so every request may emit exactly one
    token however large `output_len` is, and TPOT is genuinely undefined. The
    audit keyed off `output_len > 1` and rejected that correct export.
    """
    wl = _make(tmp_path, num_prompts=32, output_len=128, ignore_eos=False)
    wl.setup()

    doc = _bench_doc(completed=32, failed=0)
    # One token per request: no inter-token interval exists to report.
    doc["total_output_tokens"] = 32
    doc.pop("mean_tpot_ms", None)
    doc.pop("median_tpot_ms", None)
    _stub_docker(wl, monkeypatch, docs=[doc])
    result = wl.run()

    assert result.passed is True, result.failure_details

    # Still required when the export shows a second token was emitted.
    wl2 = _make(tmp_path, num_prompts=32, output_len=128, ignore_eos=False)
    wl2.setup()
    doc2 = _bench_doc(completed=32, failed=0)
    doc2["total_output_tokens"] = 4096
    doc2.pop("mean_tpot_ms", None)
    doc2.pop("median_tpot_ms", None)
    _stub_docker(wl2, monkeypatch, docs=[doc2])
    result2 = wl2.run()

    assert result2.passed is False
    assert any(d["reason"] == "result_json_unusable" for d in result2.failure_details)


def test_a_corrupt_export_is_a_bench_failure_not_a_crash(tmp_path, monkeypatch):
    """`UnicodeDecodeError` is neither `OSError` nor `JSONDecodeError`.

    A truncated or non-UTF-8 export therefore escaped the handler and aborted
    the workload, losing the steps that did parse and reporting a crash where
    the script's own `result_json_unusable` verdict is the accurate answer.
    """
    wl = _make(tmp_path, steps=2, num_prompts=32)
    wl.setup()
    wl._run_token = "tok"

    good = wl._out_dir / "bench.tok.step1.json"
    good.write_text(json.dumps(_bench_doc(completed=32, failed=0)), encoding="utf-8")
    bad = wl._out_dir / "bench.tok.step2.json"
    bad.write_bytes(b'{"completed": 32, "failed": 0, "note": "\xff\xfe not utf-8"}')

    records = wl._collect_step_records()

    # Returned rather than raised, and the readable step survived.
    assert [record.step for record in records] == [1], records


def test_a_measured_step_that_started_counts_as_main_work(tmp_path, monkeypatch):
    """`main_work_started` is about the measured phase beginning.

    It was derived from parsed exports, so a step that ran and then wrote
    unparseable JSON reported `False` -- classifying a benchmark failure as a
    did-not-run/setup failure and hiding it from the triage matrix, which
    refuses step-time and confound analysis for such a cell.
    """
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    # No exports at all, but the script announced the measured step.
    _stub_docker(
        wl,
        monkeypatch,
        docs=[],
        exit_code=54,
        stdout=(
            "TS_BENCH_METRIC: server_startup_sec=307\n"
            "TS_BENCH_STEP_START: 1\n"
            "TS_BENCH_FAIL: step 1 exported no result JSON\n"
        ),
    )
    result = wl.run()

    assert result.passed is False
    assert result.main_work_started is True, "a step that ran was reported as never started"


def test_a_started_step_is_counted_in_total_iterations(tmp_path, monkeypatch):
    """`first_failure_iteration` has to be an index into `total_iterations`.

    Counting only parsed exports left a step that ran and wrote corrupt JSON
    with `main_work_started=True` and `first_failure_iteration=0` while
    `total_iterations` was 0 -- an index outside its own range.
    `executed_iterations` stays on parsed records, since that is the count a
    consumer averaging `step_times_ms` needs.
    """
    wl = _make(tmp_path, steps=2, num_prompts=32)
    wl.setup()
    _stub_docker(
        wl,
        monkeypatch,
        docs=[],
        exit_code=54,
        stdout=(
            "TS_BENCH_STEP_START: 1\n"
            "TS_BENCH_STEP_START: 2\n"
            "TS_BENCH_FAIL: step 2 exported no result JSON\n"
        ),
    )
    result = wl.run()

    assert result.passed is False
    assert result.main_work_started is True
    assert result.total_iterations == 2, result.total_iterations
    assert result.executed_iterations == 0, result.executed_iterations
    assert result.first_failure_iteration is not None
    assert 0 <= result.first_failure_iteration < result.total_iterations


@pytest.mark.parametrize("grace", [1, 2, 4])
def test_a_teardown_grace_too_small_to_hold_a_drain_is_rejected(tmp_path, grace):
    """The gateway drain is derived from the grace and must fit inside it.

    The script's small-grace branch was a flat 10 seconds, so any grace at or
    below 10 produced a drain equal to or longer than the window meant to
    contain it -- teardown escalated to SIGKILL mid-drain, which is the
    delayed-VRAM-release failure the drain exists to prevent, reached by the
    mechanism meant to prevent it. Rejected host-side so it fails in setup()
    rather than after occupying a node.
    """
    with pytest.raises(ValueError, match="teardown_grace_sec"):
        _make(tmp_path, teardown_grace_sec=grace).setup()


@pytest.mark.parametrize("grace", [5, 10, 15, 16, 45, 3600])
def test_the_derived_drain_always_fits_inside_the_grace(tmp_path, grace):
    """Checked across the whole accepted range, in the script that derives it."""
    script = (Path(mod.__file__).with_name("tokenspeed") / "ts_bench_serve.sh").read_text()
    expression = re.search(r"DRAIN_TIMEOUT=\"\$\{TS_DRAIN_TIMEOUT:-\$\(\((.+?)\)\)\}\"", script)
    assert expression, "could not find the drain derivation"

    drain = int(
        subprocess.run(
            ["bash", "-c", f'TEARDOWN_GRACE={grace}; echo $(( {expression.group(1)} ))'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    assert drain > 0, f"grace {grace} produced a non-positive drain {drain}"
    assert drain < grace, f"grace {grace} produced a drain of {drain} that does not fit"


def test_a_bringup_failure_still_reports_that_nothing_was_measured(tmp_path, monkeypatch):
    """The other side of the marker: exits 50-52 never reach a measured step, so
    the matrix should keep reporting did-not-run rather than folding a
    non-measurement in as a data point."""
    wl = _make(tmp_path, num_prompts=32)
    wl.setup()
    _stub_docker(
        wl,
        monkeypatch,
        docs=[],
        exit_code=51,
        stdout="TS_BENCH_FAIL: readiness timeout after 900s\n",
    )
    result = wl.run()

    assert result.passed is False
    assert result.main_work_started is False, "a bring-up failure measured nothing"


def test_the_script_announces_each_measured_step(tmp_path):
    """The marker above is a contract between the script and the host, so it is
    checked where it is produced -- and warmup steps must not emit it."""
    script = (Path(mod.__file__).with_name("tokenspeed") / "ts_bench_serve.sh").read_text()
    measured = script.index('run_bench_step "bench" "${step}"')
    warmup = script.index('run_bench_step "bench-warmup"')
    marker = "TS_BENCH_STEP_START"

    assert marker in script
    # Emitted in the measured loop, and the host's regex matches what is emitted.
    assert mod._MEASURED_STEP_START_RE.search("TS_BENCH_STEP_START: 1")
    # The only occurrence that drives a step is the measured one.
    emit = script.index(f'echo "{marker}')
    assert abs(emit - measured) < abs(emit - warmup), "the marker is on the warmup loop"


def test_a_failure_without_a_step_still_points_at_an_iteration(tmp_path, monkeypatch):
    """`first_failure_iteration=None` means "no failure observed".

    Not every failure names a step -- a perf-gate breach is computed from the
    aggregate across steps -- so returning None there contradicted both the
    WorkloadResult contract and the failure_details sitting beside it.
    """
    wl = _make(tmp_path, num_prompts=32, gates={"min_output_throughput": 10_000})
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=32, failed=0)])
    result = wl.run()

    assert result.passed is False
    assert result.main_work_started is True
    assert not any(isinstance(d.get("step"), int) for d in result.failure_details), (
        "this test is only meaningful for a failure that names no step"
    )
    assert result.first_failure_iteration == 0
    assert 0 <= result.first_failure_iteration < result.total_iterations


def _run_script_audit(tmp_path: Path, doc: dict, *, expected: int = 32) -> str:
    """Drive `audit_result_json` out of ts_bench_serve.sh directly.

    The in-container audit is the half of the guard that runs where the host
    cannot see, and it is deliberately independent of the host's -- so it needs
    its own coverage rather than inheriting confidence from the Python side. The
    function is extracted rather than the whole script sourced, because sourcing
    would run the top-level validation and the docker invocation.
    """
    script = Path(mod.__file__).with_name("tokenspeed") / "ts_bench_serve.sh"
    body = subprocess.run(
        ["awk", "/^audit_result_json\\(\\) \\{/,/^\\}$/", str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "audit_result_json" in body, "failed to extract the audit function"

    export = tmp_path / "export.json"
    export.write_text(json.dumps(doc))
    harness = tmp_path / "drive.sh"
    harness.write_text(f'NUM_PROMPTS={expected}\n{body}\naudit_result_json "$1"\n')
    proc = subprocess.run(["bash", str(harness), str(export)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout.strip()


def test_the_in_container_audit_passes_a_clean_export(tmp_path):
    """Establishes that the extraction harness actually exercises the audit, so
    the rejection tests below cannot pass by simply failing everything."""
    assert _run_script_audit(tmp_path, {"completed": 32, "failed": 0}).startswith("OK")


def test_the_in_container_audit_rejects_a_negative_failure_count(tmp_path):
    """`failed > 0` let a negative count through as a clean run.

    Both audits are meant to fail closed on the same contract (`failed == 0`);
    whichever one reads a negative count as success becomes the one a malformed
    export is believed through, which defeats the point of running two.
    """
    verdict = _run_script_audit(tmp_path, {"completed": 32, "failed": -1})
    assert verdict.startswith("SHORTFALL"), verdict


def test_a_failed_container_removal_is_reported(tmp_path, monkeypatch, caplog):
    """`docker rm -f` reports a daemon or permission failure by exit code.

    Nothing raises, so checking only for exceptions meant the one outcome worth
    knowing about -- the container is still running and still holding the GPU --
    produced no output at all, while the helper's docstring promised cleanup
    failures were logged. The next cell then fails for a reason recorded nowhere.
    """
    wl = _make(tmp_path)
    wl.setup()
    wl._run_token = "tok"

    def fake_run(argv, **kwargs):
        assert argv[:3] == ["docker", "rm", "-f"]
        return subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="permission denied while trying to connect"
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    with caplog.at_level("DEBUG"):
        wl._force_remove_container()

    assert any(r.levelname == "WARNING" for r in caplog.records)
    logged = caplog.text
    assert "permission denied" in logged, "the docker stderr is what makes it actionable"
    assert wl._container_name() in logged, "the name is needed to remove it by hand"


def test_an_already_gone_container_is_not_reported_as_a_failure(tmp_path, monkeypatch, caplog):
    """`--rm` usually gets there first, and a trial can fail before the container
    exists at all. Warning on that would make the real warning above noise."""
    wl = _make(tmp_path)
    wl.setup()
    wl._run_token = "tok"

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 1, stdout="", stderr=f"Error: No such container: {wl._container_name()}"
        ),
    )
    with caplog.at_level("DEBUG"):
        wl._force_remove_container()

    assert not any(r.levelname == "WARNING" for r in caplog.records), caplog.text


def test_request_counters_are_published_as_sums_not_also_as_means(tmp_path, monkeypatch):
    """`completed` and `failed` are per-step counters, not measurements.

    Generic aggregation published them as means beside the `completed_total` /
    `failed_total` sums, so a three-step 32-prompt trial reported `completed: 32`
    and `completed_total: 96` in the same performance table -- which reads as a
    discrepancy rather than as two units. The mean is also the misleading one: it
    hides a single bad step among good ones, which is why the sums exist.
    """
    wl = _make(tmp_path, num_prompts=32, steps=3)
    wl.setup()
    _stub_docker(
        wl,
        monkeypatch,
        docs=[_bench_doc(completed=32, failed=0) for _ in range(3)],
    )
    metrics = wl.run().metrics

    assert metrics["completed_total"] == 96
    assert metrics["failed_total"] == 0
    assert "completed" not in metrics, "the per-step mean is still published"
    assert "failed" not in metrics, "the per-step mean is still published"


@pytest.mark.parametrize("doc_value", [True, False])
def test_boolean_request_counters_are_rejected_not_counted(tmp_path, monkeypatch, doc_value):
    """`bool` is a subclass of `int`, and `json` decodes true/false into it.

    So `isinstance(completed, int)` accepted an export whose counters were
    booleans, and with the legitimate `num_prompts: 1` the values then compared
    equal to 1 and 0 -- an export that counted nothing satisfying the
    served-request audit outright.
    """
    doc = _bench_doc(completed=1, failed=0)
    doc["completed"] = doc_value
    doc["failed"] = False
    wl = _make(tmp_path, num_prompts=1)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[doc])
    result = wl.run()

    assert result.passed is False
    assert any(d["reason"] == "result_json_unusable" for d in result.failure_details)
    assert "completed_total" not in result.metrics or result.metrics["completed_total"] == 0


@pytest.mark.parametrize(
    "args", [["-u0:0"], ["-v/tmp/x:/ts-out"], ["--ipc=none"], ["--shm-size=1m"], ["--rm=false"]]
)
def test_attached_and_late_owned_docker_options_are_rejected(tmp_path, args):
    """docker accepts a value attached to a short option, and the guard split on
    `=` only -- so `-u0:0` and `-v/tmp:/ts-out` walked straight past a check that
    blocked their spaced spellings. `-u0:0` restores root-owned artifacts the
    next run cannot delete; `-v .../ts-out` displaces the mount the audit reads.

    `--ipc` and `--shm-size` are here because TokenSpeed's scheduler sizes its
    shared memory against them and a later smaller value fails at load time with
    an error about shared memory rather than about docker_args; `--rm=false`
    leaves every completed container behind, so a sweep fills the node up while
    no single trial looks wrong.
    """
    with pytest.raises(ValueError, match="docker_args may not set"):
        _make(tmp_path, docker_args=args).setup()


def test_docker_args_cannot_smuggle_a_key_whose_value_is_absence(tmp_path):
    """The same absence hole that was fixed for mitigations, one field sideways.

    `owned_env` can only name keys that are present, and an unbounded
    `max_concurrency` sets no TS_MAX_CONCURRENCY -- so on the default
    configuration this route ran the container capped while the host reported
    `max_concurrency: None`.
    """
    assert (
        _make(tmp_path).config.get("max_concurrency") is None
    ), "precondition: the unguarded default"
    with pytest.raises(ValueError, match="TS_MAX_CONCURRENCY"):
        _make(tmp_path, docker_args=["-e", "TS_MAX_CONCURRENCY=1"]).setup()


def test_a_late_step_failure_keeps_the_message_that_explains_it(tmp_path, monkeypatch):
    """Every TS_BENCH_FAIL diagnostic is printed, not raised.

    The nonzero-exit branch kept only stderr, which was survivable while a
    failure also produced no records -- the `no_bench_export` branch carries
    stdout. But a later step failing after earlier ones exported leaves that
    branch unreached, so the trial reported an exit code with the message
    explaining it discarded.
    """
    wl = _make(tmp_path, num_prompts=32, steps=3)
    wl.setup()
    _stub_docker(
        wl,
        monkeypatch,
        docs=[_bench_doc(completed=32, failed=0)],
        exit_code=55,
        stdout="TS_BENCH_INFO: step 1 ok\nTS_BENCH_FAIL: step 2 SHORTFALL completed=3\n",
    )
    result = wl.run()

    assert result.passed is False
    tails = " ".join(str(d.get("stdout_tail", "")) for d in result.failure_details)
    assert "TS_BENCH_FAIL: step 2" in tails, result.failure_details


@pytest.mark.parametrize("doc_value", [True, False])
def test_the_in_container_audit_rejects_boolean_counters(tmp_path, doc_value):
    """The other half of the boolean hole: neither audit may be the lenient one.

    With `num_prompts: 1` a `completed: true` compares equal to 1, so the
    script's standalone guard would have echoed OK for an export that counted
    nothing.
    """
    verdict = _run_script_audit(tmp_path, {"completed": doc_value, "failed": False}, expected=1)
    assert verdict.startswith("UNPARSEABLE"), verdict


@pytest.mark.parametrize("args", [["-d"], ["--detach"], ["-dit"], ["--detach=true"]])
def test_detaching_the_container_is_rejected(tmp_path, args):
    """Detach breaks every assumption that follows the `docker run` call.

    It returns 0 immediately, so the trial reports success with no exports while
    the container keeps benchmarking and holding the GPU -- and since nothing
    raised or timed out, no cleanup path runs and `_force_remove_container` is
    never reached. `-dit` is included because a combined short cluster carries
    `-d` without it ever appearing as a token.
    """
    with pytest.raises(ValueError, match="docker_args may not set"):
        _make(tmp_path, docker_args=args).setup()


def test_an_explicit_control_port_is_reserved_before_the_gateway_resolves(tmp_path, monkeypatch):
    """The mixed case: `port: auto` with an explicit `control_port`.

    Resolving the gateway blind let the kernel hand it the very port the control
    endpoint is configured to use, and the equality check then rejected a
    configuration that was entirely valid.

    Driven with a kernel that hands out the configured control port first, since
    otherwise the collision is a one-in-thousands accident and the test would
    pass whether or not the reservation exists.
    """
    handed_out = iter([9711, 9711, 9712, 9713])

    class FakeSocket:
        def __init__(self, *a, **k):
            self._port = None

        def bind(self, addr):
            self._port = next(handed_out)

        def getsockname(self):
            return ("127.0.0.1", self._port)

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod.socket, "socket", FakeSocket)
    monkeypatch.setattr(mod, "_port_is_free", lambda port: False)

    wl = _make(tmp_path, port="auto", control_port=9711)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=64, failed=0)])
    wl.run()

    assert wl._control_port == 9711
    assert wl._port != wl._control_port, "gateway took the reserved control port"


@pytest.mark.parametrize(
    "key,value",
    [
        ("ready_timeout_sec", 86401),
        ("teardown_grace_sec", 3601),
    ],
)
def test_timeouts_above_the_container_ceiling_fail_here(tmp_path, key, value):
    """ts_bench_serve.sh caps both with `require_uint`, so a larger value passed
    setup(), occupied a GPU node, and then exited 64 inside the container --
    reported as a workload failure with the violated range visible only in the
    container log."""
    with pytest.raises(ValueError, match="must be between"):
        _make(tmp_path, **{key: value}).setup()


@pytest.mark.parametrize("key", ["num_prompts", "steps", "input_len", "num_warmups", "seed"])
@pytest.mark.parametrize("value", [True, 1.9])
def test_integer_fields_reject_booleans_and_fractions(tmp_path, key, value):
    """A bare `int()` ran a *different* load rather than failing: `bool` is an
    int subclass, and a float is truncated, so `num_prompts: true` and
    `num_prompts: 1.9` both meant one prompt. A recipe that cannot be read the
    way it was written must not be executed the way it was not."""
    with pytest.raises(ValueError, match=key):
        _make(tmp_path, **{key: value}).setup()


def test_random_dataset_needs_no_file_and_keeps_the_length_knobs(tmp_path):
    """The default has to stay self-contained -- it is what every existing recipe
    uses, and its prompts are generated from input_len/output_len."""
    wl = _make(tmp_path)
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    env = wl._container_env()
    assert env["TS_DATASET"] == "random"
    assert "TS_DATASET_PATH" not in env
    assert int(env["TS_INPUT_LEN"]) > 0 and int(env["TS_OUTPUT_LEN"]) > 0


def test_sharegpt_requires_a_staged_file(tmp_path):
    """Given no path the bench CLI downloads ShareGPT from the Hub on first use.

    That is wrong here three ways over: the recipe stops being reproducible (the
    URL is not pinned and the file is not content-addressed), a bridged container
    often has no route to the Hub, and the download lands inside the measured
    window -- so the first trial reports as slower for a reason that is not the
    engine.
    """
    with pytest.raises(ValueError, match="requires dataset_path"):
        _make(tmp_path, dataset="sharegpt").setup()


def test_a_missing_dataset_file_fails_before_the_model_loads(tmp_path):
    """Otherwise this surfaces after the engine has come up, and reads as a bench
    failure rather than as the staging mistake it is."""
    with pytest.raises(ValueError, match="is not a file"):
        _make(tmp_path, dataset="sharegpt", dataset_path=str(tmp_path / "absent.json")).setup()


def test_sharegpt_is_mounted_read_only_and_drops_the_random_knobs(tmp_path):
    """The dataset defines what was measured, so a run able to rewrite it would
    make its own results unfalsifiable. The ISL/OSL knobs are `random`-only --
    sharegpt takes its lengths from the conversations, and forwarding them would
    advertise a shape the run did not have.
    """
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text("[]", encoding="utf-8")
    wl = _make(tmp_path, dataset="sharegpt", dataset_path=str(dataset))
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    env = wl._container_env()
    argv = wl._docker_argv(env)

    assert env["TS_DATASET"] == "sharegpt"
    assert env["TS_DATASET_PATH"] == mod._CONTAINER_DATASET_PATH
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert f"{dataset}:{mod._CONTAINER_DATASET_PATH}:ro" in mounts, mounts


def test_dataset_path_is_rejected_for_the_random_dataset(tmp_path):
    """The bench CLI raises "Cannot use 'random' dataset with --dataset-path";
    catching it here names the recipe field instead of the CLI flag."""
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="meaningless for"):
        _make(tmp_path, dataset_path=str(dataset)).setup()


def test_an_unknown_dataset_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be one of"):
        _make(tmp_path, dataset="wikitext").setup()


@pytest.mark.parametrize(
    "env_overrides,expect",
    [
        ({"TS_DATASET": "sharegpt"}, "requires TS_DATASET_PATH"),
        ({"TS_DATASET": "wikitext"}, "must be random or sharegpt"),
        (
            {"TS_DATASET": "sharegpt", "TS_DATASET_PATH": "/nonexistent/ds.json"},
            "is not readable",
        ),
    ],
)
def test_the_script_validates_the_dataset_independently(tmp_path, env_overrides, expect):
    """The script is meant to be runnable by hand, which is what keeps its audit
    trustworthy independently of the Python layer -- so it re-checks the dataset
    contract rather than trusting the host to have done it."""
    proc = subprocess.run(
        ["bash", str(mod._SCRIPTS_DIR / mod._BENCH_SCRIPT)],
        capture_output=True,
        text=True,
        env={**os.environ, "TS_OUT_DIR": str(tmp_path), **env_overrides},
    )
    assert proc.returncode == 64, proc.stdout + proc.stderr
    assert expect in proc.stdout, proc.stdout


def _cycle_samples(samples: list[dict[int, int]]):
    """VRAM readings: each entry once, then the last one forever.

    The poll loop reads until the memory comes back or the deadline passes, so a
    fixed-length iterator would raise StopIteration instead of exercising it.
    """
    remaining = list(samples)

    def read() -> dict[int, int]:
        return remaining.pop(0) if len(remaining) > 1 else dict(remaining[0])

    return read


def test_a_slow_tensor_parallel_teardown_is_waited_out(tmp_path, monkeypatch):
    """`docker run` returning is not the same event as the memory coming back.

    Measured after a *passing* TP=2 cell on gfx950: container gone, nothing
    holding /dev/kfd, and one GPU still reporting 256 GB of its 309 GB. It clears
    on its own in 30-45s -- only rank 0's device is released promptly. aorta
    starts the next cell immediately, which is how the `tp4` cell first died,
    with an out-of-memory error that mentioned neither tensor parallelism nor the
    cell that actually caused it.
    """
    samples = iter(
        [
            {0: 3 * 1024**3, 1: 1 * 1024**3},  # before
            {0: 3 * 1024**3, 1: 257 * 1024**3},  # still held right after exit
            {0: 3 * 1024**3, 1: 257 * 1024**3},
            {0: 3 * 1024**3, 1: 2 * 1024**3},  # released
        ]
    )
    monkeypatch.setattr(mod, "_vram_used_by_gpu", lambda: next(samples))
    monkeypatch.setattr(mod, "_VRAM_RELEASE_POLL_SEC", 0)

    wl = _make(tmp_path, num_prompts=64)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=64, failed=0)])
    result = wl.run()

    assert result.passed is True, result.failure_details
    assert not [d for d in result.failure_details if d["reason"] == "gpu_memory_not_reclaimed"]


def test_memory_never_released_fails_the_trial_that_held_it(tmp_path, monkeypatch):
    """Once waiting has stopped helping, a green cell would quietly break the
    next one -- nothing in this trial's own numbers looks wrong.

    `hip_visible_devices` is what makes the growth this trial's to claim; see
    the unattributed case below.
    """
    monkeypatch.setattr(
        mod,
        "_vram_used_by_gpu",
        _cycle_samples([{0: 1 * 1024**3}, {0: 257 * 1024**3}]),
    )
    monkeypatch.setattr(mod, "_VRAM_RELEASE_POLL_SEC", 0)
    monkeypatch.setattr(mod, "_VRAM_RELEASE_TIMEOUT_SEC", 0)

    wl = _make(tmp_path, num_prompts=64, hip_visible_devices="0")
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=64, failed=0)])
    result = wl.run()

    leaks = [d for d in result.failure_details if d["reason"] == "gpu_memory_not_reclaimed"]
    assert result.passed is False
    assert [d["gpu"] for d in leaks] == [0], result.failure_details
    assert "gpureset" in leaks[0]["detail"]


def test_unreleased_memory_is_not_blamed_on_a_trial_that_did_not_own_the_gpu(
    tmp_path, monkeypatch, caplog
):
    """A before/after delta shows growth; it does not show who allocated it.

    Without an exclusive assignment the workload cannot tell its own leftovers
    from a co-tenant starting a job mid-trial, and failing here would redden a
    healthy cell and point `rocm-smi --gpureset` at somebody else's device. The
    wait still happens -- it helps the next cell whoever owns the memory -- but
    the outcome is a warning, not this trial's failure.
    """
    monkeypatch.setattr(
        mod,
        "_vram_used_by_gpu",
        _cycle_samples([{0: 1 * 1024**3}, {0: 257 * 1024**3}]),
    )
    monkeypatch.setattr(mod, "_VRAM_RELEASE_POLL_SEC", 0)
    monkeypatch.setattr(mod, "_VRAM_RELEASE_TIMEOUT_SEC", 0)

    wl = _make(tmp_path, num_prompts=64)  # no hip_visible_devices
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=64, failed=0)])
    with caplog.at_level("WARNING"):
        result = wl.run()

    assert result.passed is True, result.failure_details
    assert not [d for d in result.failure_details if d["reason"] == "gpu_memory_not_reclaimed"]
    assert "cannot be attributed" in caplog.text


def test_only_the_trials_own_gpus_are_watched_for_unreleased_memory(tmp_path, monkeypatch):
    """Growth on a GPU this trial was never given is somebody else's.

    The check sampled every card on the node, so a co-tenant allocating on any
    of them failed this workload. Only the devices named by
    `hip_visible_devices` are this trial's to answer for.
    """
    monkeypatch.setattr(
        mod,
        "_vram_used_by_gpu",
        _cycle_samples(
            [
                {0: 1 * 1024**3, 1: 1 * 1024**3},
                # GPU 1 is the co-tenant's; GPU 0 -- ours -- released cleanly.
                {0: 2 * 1024**3, 1: 257 * 1024**3},
            ]
        ),
    )
    monkeypatch.setattr(mod, "_VRAM_RELEASE_POLL_SEC", 0)
    monkeypatch.setattr(mod, "_VRAM_RELEASE_TIMEOUT_SEC", 0)

    wl = _make(tmp_path, num_prompts=64, hip_visible_devices="0")
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=64, failed=0)])
    result = wl.run()

    assert result.passed is True, result.failure_details


def test_normal_driver_overhead_is_not_reported_as_a_leak(tmp_path, monkeypatch):
    """A few GiB survives any run, so the check is against growth past a margin.
    Reporting that as a leak would make every cell red and the signal useless."""
    samples = iter([{0: 1 * 1024**3}, {0: 3 * 1024**3}])
    monkeypatch.setattr(mod, "_vram_used_by_gpu", lambda: next(samples))

    wl = _make(tmp_path, num_prompts=64)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=64, failed=0)])
    result = wl.run()

    assert result.passed is True, result.failure_details
    assert not [d for d in result.failure_details if d["reason"] == "gpu_memory_not_reclaimed"]


def test_another_tenants_memory_is_not_attributed_to_this_trial(tmp_path, monkeypatch):
    """Compared against a pre-run sample, not an absolute ceiling: on a shared
    node a GPU can already be busy, and that is not this trial's doing."""
    samples = iter([{0: 200 * 1024**3}, {0: 200 * 1024**3}])
    monkeypatch.setattr(mod, "_vram_used_by_gpu", lambda: next(samples))

    wl = _make(tmp_path, num_prompts=64)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=64, failed=0)])
    result = wl.run()

    assert result.passed is True, result.failure_details


def test_the_leak_check_disables_itself_when_vram_cannot_be_read(tmp_path, monkeypatch):
    """A missing measurement is not evidence of a leak, and this has to keep
    working on any node whose sysfs does not expose the attribute."""
    monkeypatch.setattr(mod, "_vram_used_by_gpu", dict)

    wl = _make(tmp_path, num_prompts=64)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=64, failed=0)])
    result = wl.run()

    assert result.passed is True, result.failure_details


def test_elapsed_covers_the_drain_wait_the_cell_actually_spent(tmp_path, monkeypatch):
    """The GPU is unusable for the whole post-exit wait, so the clock cannot
    stop at container exit.

    A sweep's budget is summed from `elapsed_sec`, and stopping early made every
    cell under-report by its own drain time -- up to `_VRAM_RELEASE_TIMEOUT_SEC`
    each, which on a TP teardown is the 30-45s this workload explicitly waits
    for. The container-only figure is still available, as a metric.
    """
    clock = iter([1000.0, 1010.0, 1042.0])  # start, container exit, after drain
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(mod, "_vram_used_by_gpu", dict)  # drain wait returns at once

    wl = _make(tmp_path, num_prompts=64)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=64, failed=0)])
    result = wl.run()

    assert result.elapsed_sec == pytest.approx(42.0)
    assert result.metrics["container_elapsed_sec"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "serve_args",
    [
        ["--drain-timeout", "60"],
        ["--drain-timeout=60"],
        ["--drain-timeout", "45"],  # equal to the grace is already too long
        ["--drain-timeout", "0"],
    ],
)
def test_an_explicit_drain_timeout_must_fit_inside_the_teardown_grace(tmp_path, serve_args):
    """`--drain-timeout` is a *serve* flag, so the bench-flag guard never saw it.

    The script derives the drain from the grace precisely so teardown cannot
    SIGKILL a gateway mid-drain -- the delayed VRAM release this workload spends
    a poll loop waiting out. Passing the flag in `serve_args` replaced that
    derived value with an unchecked one, so `60` against the default 45s grace
    put every teardown back in the failure the derivation prevents.
    """
    with pytest.raises(ValueError, match="drain-timeout"):
        _make(tmp_path, serve_args=serve_args).setup()


def test_a_drain_timeout_inside_the_grace_is_left_alone(tmp_path):
    """The caller keeps the choice; it just has to fit. A recipe tuning the
    drain for a slow gateway is exactly what the flag is for."""
    wl = _make(tmp_path, serve_args=["--drain-timeout", "30"], teardown_grace_sec=45)
    wl.setup()
    assert "--drain-timeout" in wl._serve_args


def test_an_hf_token_is_never_written_into_the_docker_command_line(tmp_path, monkeypatch):
    """`/proc/<pid>/cmdline` is world-readable, and the client lives for the
    whole trial.

    `-e HF_TOKEN=<value>` put a credential in that argv for the hours a serving
    sweep runs, on nodes that are shared by construction. The name is forwarded
    on its own instead, and the value travels in the client's environment, which
    only the owning uid and root can read.
    """
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    capture: dict = {}

    wl = _make(tmp_path, num_prompts=64)
    wl.setup()
    _stub_docker(wl, monkeypatch, docs=[_bench_doc(completed=64, failed=0)], capture=capture)
    wl.run()

    argv = capture["argv"]
    assert not [arg for arg in argv if "hf_secret_value" in arg], argv
    assert "HF_TOKEN" in argv, argv
    assert capture["kwargs"]["env"]["HF_TOKEN"] == "hf_secret_value"


def test_an_env_name_is_not_mistaken_for_a_short_flag_cluster(tmp_path):
    """A short cluster holds boolean flags only as far as the first option that
    takes a value.

    `-emodel_id=x` is `-e` with an attached value, but scanning the whole word
    for owned letters found the `d` in `model` and rejected it as if the recipe
    had asked for `--detach`. The message named a flag that was never written,
    on a recipe that was correct.
    """
    wl = _make(tmp_path, docker_args=["-emodel_id=x"])
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    assert "-emodel_id=x" in wl._docker_argv(wl._container_env())


def test_two_users_on_one_node_get_separate_scratch(tmp_path, monkeypatch):
    """A fixed path in /tmp belongs to whoever created it.

    The second user failed while creating `scripts/` beneath a root made at the
    first user's umask -- a permission error unrelated to anything in their
    recipe -- and where the mode did allow writing, `keep_work_dir: false`
    removed the other user's exports mid-run. Every recipe here names the same
    `/tmp/ts-work-serve`, so the scoping applies to the configured value.
    """
    root = tmp_path / "shared"
    dirs = []
    for uid in (1000, 1001):
        monkeypatch.setattr(mod.os, "getuid", lambda uid=uid: uid)
        wl = _make(tmp_path, work_dir=str(root))
        wl.setup()
        dirs.append(wl._out_dir)
        # The cache is shared on purpose: content-addressed, read-only in
        # practice, and ~40 GB for gpt-oss.
        assert wl._hf_home == root / "hf"

    assert dirs[0] != dirs[1]
    assert all(str(d).startswith(str(root)) for d in dirs)
    # Enterable and writable by every user, sticky like /tmp so neither can
    # remove the other's scratch.
    assert root.stat().st_mode & 0o1777 == 0o1777
