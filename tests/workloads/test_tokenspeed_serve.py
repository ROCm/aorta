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
import subprocess
from pathlib import Path

import pytest

from aorta.workloads import tokenspeed_serve as mod
from aorta.workloads.tokenspeed_serve import TokenSpeedServeWorkload

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
        ("port", 0, "1..65535"),
        ("port", 70000, "1..65535"),
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
    monkeypatch.setenv("MY_HF_TOKEN", "secret-value")
    wl = _make(tmp_path, hf_token_env="MY_HF_TOKEN")
    wl.setup()
    wl._run_token = "tok"
    wl._port, wl._control_port = 8000, 8001
    assert wl._container_env()["HF_TOKEN"] == "secret-value"


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


_RECIPES = (
    "tokenspeed-serve-bench-smoke.yaml",
    "tokenspeed-serve-models.yaml",
    "tokenspeed-serve-load.yaml",
)


def _recipe_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "recipes" / "tokenspeed"


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
def test_recipe_workload_config_is_accepted_by_the_workload(name, tmp_path):
    """Every committed recipe must survive this workload's own validation --
    a typo'd key or an out-of-range value should not wait for a GPU to surface."""
    from aorta.triage.recipe import load_recipe

    recipe = load_recipe(_recipe_dir() / name)
    for cell in recipe.cells:
        config = {**recipe.workload_config, **cell.workload_config}
        config["steps"] = cell.steps or recipe.steps
        config["work_dir"] = str(tmp_path / "work")
        TokenSpeedServeWorkload(config).setup()


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
