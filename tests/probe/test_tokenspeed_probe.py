"""Tests for the TokenSpeed probe scripts and recipes.

None of these need a GPU, a container, or a TokenSpeed install. They cover the
parts of the integration that are easy to get wrong and expensive to discover on
hardware:

  * the shell scripts parse, and their guardrails fire (NFS bind-mount refusal,
    missing entry script, missing selector);
  * the numerics gate in ``ts_kernel_probe.sh`` fails a trial when the exported
    benchmark JSON reports a mismatch. This is the one verdict the upstream CLI
    does not signal through its exit code -- ``tokenspeed_kernel.benchmark``
    ends in an unconditional ``return 0`` even when ``--verify`` finds a wrong
    answer -- so without this gate a numerically broken kernel would produce a
    green aorta cell. It cannot be covered by running a real kernel, because the
    shipped kernels pass, hence the stubbed export;
  * the recipes and the mitigations sidecar stay loadable, and the sidecar's
    kernel/dtype pairings stay self-consistent.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SOURCE = _REPO / "src" / "aorta" / "workloads" / "tokenspeed"
_RECIPES = _REPO / "recipes" / "tokenspeed"

_SCRIPTS = (
    "host_launch.sh",
    "ts_serve_probe.sh",
    "ts_kernel_probe.sh",
    "ts_pytest_probe.sh",
    "stage_scripts.sh",
)

# Exit codes are a documented interface: the recipes' custom_patterns and anyone
# reading result.json depend on them, so pin them here.
_EXIT_USAGE = 64
_EXIT_NO_RECORDS = 31
_EXIT_NUMERICS = 32
_EXIT_SCHEMA = 34
_EXIT_PYTEST_FAILED = 40
_EXIT_PYTEST_NOTHING_RAN = 41


@pytest.fixture(scope="module")
def bash() -> str:
    found = shutil.which("bash")
    if found is None:  # pragma: no cover - bash is present everywhere we run
        pytest.skip("bash not available")
    return found


@pytest.mark.parametrize("name", _SCRIPTS)
def test_script_is_syntactically_valid(bash: str, name: str) -> None:
    script = _SOURCE / name
    assert script.exists(), f"{name} missing from {_SOURCE}"
    proc = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, f"{name} syntax error: {proc.stderr}"


@pytest.mark.parametrize("name", _SCRIPTS)
def test_script_is_executable(name: str) -> None:
    assert os.access(_SOURCE / name, os.X_OK), f"{name} is not executable"


def test_host_launch_requires_image(bash: str, tmp_path: Path) -> None:
    proc = subprocess.run(
        [bash, str(_SOURCE / "host_launch.sh")],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},
    )
    assert proc.returncode != 0
    assert "TS_IMAGE" in proc.stdout + proc.stderr


def test_host_launch_refuses_nfs_mount(bash: str, tmp_path: Path) -> None:
    """A /home path must be rejected before docker is ever invoked.

    The daemon runs as root against a root-squashed export, so the bind mount
    fails with an opaque "mkdir /home/<user>: permission denied" from docker
    itself. Refusing early keeps the error actionable.
    """
    env = dict(os.environ)
    env.update(
        {
            "TS_IMAGE": "example/image:tag",
            "TS_SCRIPTS_DIR": "/home/someone/scripts",
            "TS_HF_DIR": str(tmp_path / "hf"),
            "TS_OUT_DIR": str(tmp_path / "out"),
        }
    )
    proc = subprocess.run(
        [bash, str(_SOURCE / "host_launch.sh")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == _EXIT_USAGE
    assert "root-squashed" in proc.stderr


def test_host_launch_reports_missing_entry_script(bash: str, tmp_path: Path) -> None:
    """A stale staging dir must be named as such, not surface from inside docker."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "host_launch.sh").write_text("#!/bin/sh\n")
    env = dict(os.environ)
    env.update(
        {
            "TS_IMAGE": "example/image:tag",
            "TS_SCRIPTS_DIR": str(scripts),
            "TS_HF_DIR": str(tmp_path / "hf"),
            "TS_OUT_DIR": str(tmp_path / "out"),
            "TS_ENTRY": "ts_kernel_probe.sh",
        }
    )
    proc = subprocess.run(
        [bash, str(_SOURCE / "host_launch.sh")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == _EXIT_USAGE
    assert "entry script ts_kernel_probe.sh not found" in proc.stderr


def _run_host_launch_with_docker_stub(
    bash: str, tmp_path: Path, tag: str, extra_env: dict[str, str] | None = None
) -> str:
    """Run host_launch.sh against a docker that records its argv instead of running.

    Returns the recorded ``docker run`` command line.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / f"argv.{tag}"
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "image" ]; then exit 0; fi\n'
        f'printf "%s\\n" "$@" > "{record}"\n'
    )
    stub.chmod(0o755)

    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    for name in ("host_launch.sh", "ts_kernel_probe.sh", "ts_serve_probe.sh"):
        shutil.copy2(_SOURCE / name, scripts / name)

    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "TS_IMAGE": "example/image:tag",
            "TS_SCRIPTS_DIR": str(scripts),
            "TS_HF_DIR": str(tmp_path / "hf"),
            "TS_OUT_DIR": str(tmp_path / "out"),
            "TS_ENTRY": "ts_kernel_probe.sh",
        }
    )
    env.pop("TS_RUN_TOKEN", None)
    env.update(extra_env or {})
    proc = subprocess.run(
        [bash, str(scripts / "host_launch.sh")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return record.read_text()


def _token_from(argv: str) -> str:
    for line in argv.splitlines():
        if line.startswith("TS_RUN_TOKEN="):
            return line.split("=", 1)[1]
    raise AssertionError(f"TS_RUN_TOKEN not forwarded to docker; argv was:\n{argv}")


def test_the_cell_env_file_outranks_the_invoking_shell(bash: str, tmp_path: Path) -> None:
    """`docker run -e VAR=` overrides the same name from `--env-file`.

    So a selector left exported in the caller's shell would replace the per-cell
    value in *every* cell -- the matrix running one identical target while still
    labelling the cells as different. That is the failure the env file exists to
    prevent, arriving from the one direction nothing else guards.
    """
    env_file = tmp_path / "cell.env"
    env_file.write_text("TS_KERNEL_NAME=cell_kernel\n")

    argv = _run_host_launch_with_docker_stub(
        bash,
        tmp_path,
        "envfile-precedence",
        extra_env={
            "AORTA_ENV_FILE": str(env_file),
            # Leaked from the caller's shell, naming the same knob as the cell.
            "TS_KERNEL_NAME": "shell_kernel",
            # Not named by the cell, so this one should still be forwarded.
            "TS_KERNEL_DTYPE": "fp16",
        },
    )

    assert "TS_KERNEL_NAME=shell_kernel" not in argv
    assert "TS_KERNEL_DTYPE=fp16" in argv
    assert f"--env-file\n{env_file}" in argv


def test_a_cells_operator_survives_a_kernel_name_left_in_the_shell(
    bash: str, tmp_path: Path
) -> None:
    """The two kernel selectors are one decision, so the check is per group.

    ts_kernel_probe.sh resolves TS_KERNEL_NAME first and only falls back to
    TS_KERNEL_OP. A name-by-name check therefore lets a TS_KERNEL_NAME leaked
    from the caller's shell through untouched -- it is absent from the cell's env
    file, so nothing looks like a conflict -- and the container pins that one
    kernel for every cell of a sweep whose cells each still carry their own
    operator label. Same substitution the test above pins, entered through the
    other spelling of the pair.
    """
    env_file = tmp_path / "cell.env"
    env_file.write_text("TS_KERNEL_OP=gemm.mm\n")

    argv = _run_host_launch_with_docker_stub(
        bash,
        tmp_path,
        "selector-group",
        extra_env={
            "AORTA_ENV_FILE": str(env_file),
            "TS_KERNEL_NAME": "gluon_mm_a16w16_gfx950",
        },
    )

    assert (
        "TS_KERNEL_NAME=gluon_mm_a16w16_gfx950" not in argv
    ), "the shell's kernel name outranks the cell's --op inside the container"


def test_a_kernel_name_is_still_forwarded_when_the_cell_names_no_selector(
    bash: str, tmp_path: Path
) -> None:
    """Grouping must not cost the manual run its knob.

    Exporting TS_KERNEL_NAME is how a one-off run picks a kernel; the group only
    decides who wins when a cell has already made that choice.
    """
    env_file = tmp_path / "cell.env"
    env_file.write_text("TS_KERNEL_DTYPE=bf16\n")

    argv = _run_host_launch_with_docker_stub(
        bash,
        tmp_path,
        "selector-group-free",
        extra_env={
            "AORTA_ENV_FILE": str(env_file),
            "TS_KERNEL_NAME": "gluon_mm_a16w16_gfx950",
        },
    )

    assert "TS_KERNEL_NAME=gluon_mm_a16w16_gfx950" in argv


def test_host_launch_logs_mitigation_names_without_their_values(bash: str, tmp_path: Path) -> None:
    """Mitigations may carry credentials, and stdout.log is retained broadly.

    A mitigation need only resolve to dict[str, str] and can come from a plugin
    or sidecar, so aorta's registry declines to repr these; printing them here
    would undo that. The names are what makes a result attributable, and they are
    enough for it.
    """
    env_file = tmp_path / "cell.env"
    env_file.write_text("# a comment\n\nTS_KERNEL_NAME=some_kernel\nHF_TOKEN=hf_supersecret\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "docker"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n")
    stub.chmod(0o755)

    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    for name in ("host_launch.sh", "ts_kernel_probe.sh"):
        shutil.copy2(_SOURCE / name, scripts / name)

    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "TS_IMAGE": "example/image:tag",
            "TS_SCRIPTS_DIR": str(scripts),
            "TS_HF_DIR": str(tmp_path / "hf"),
            "TS_OUT_DIR": str(tmp_path / "out"),
            "TS_ENTRY": "ts_kernel_probe.sh",
            "AORTA_ENV_FILE": str(env_file),
        }
    )
    proc = subprocess.run(
        [bash, str(scripts / "host_launch.sh")],
        capture_output=True,
        text=True,
        env=env,
    )

    combined = proc.stdout + proc.stderr
    assert "hf_supersecret" not in combined, "a mitigation value reached the log"
    assert "some_kernel" not in combined
    for name in ("TS_KERNEL_NAME", "HF_TOKEN"):
        assert name in combined, f"{name} should still be recorded by name"


def test_host_launch_rejects_an_unreadable_env_file(bash: str, tmp_path: Path) -> None:
    """Set-but-unreadable must fail, not fall back to container defaults.

    The file carries the cell's mitigations. Dropping it silently would make
    every cell in the matrix run one identical configuration while the run still
    reported them as distinct -- a green matrix that measured nothing it claimed.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # The image-presence check runs first, so it needs to succeed for this test
    # to reach the env-file check it is about.
    stub = bin_dir / "docker"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n")
    stub.chmod(0o755)

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("host_launch.sh", "ts_kernel_probe.sh"):
        shutil.copy2(_SOURCE / name, scripts / name)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "TS_IMAGE": "example/image:tag",
            "TS_SCRIPTS_DIR": str(scripts),
            "TS_HF_DIR": str(tmp_path / "hf"),
            "TS_OUT_DIR": str(tmp_path / "out"),
            "TS_ENTRY": "ts_kernel_probe.sh",
            "AORTA_ENV_FILE": str(tmp_path / "does-not-exist.env"),
        }
    )
    proc = subprocess.run(
        [bash, str(scripts / "host_launch.sh")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == _EXIT_USAGE
    assert "is not readable" in proc.stderr
    assert "silently dropped" in proc.stderr


def test_host_launch_picks_the_network_per_route(bash: str, tmp_path: Path) -> None:
    """Bridge for the routes that need nothing off-node; host only where a
    documented external dependency requires it.

    The kernel and pytest probes run against the source tree already in the
    image, so host networking would only publish container ports on a possibly
    shared node. The serving routes resolve the model through huggingface_hub,
    which contacts the Hub even for a cached model, and a bridged container on a
    node with IPv4 forwarding disabled dies during startup with "Temporary
    failure in name resolution".
    """
    argv = _run_host_launch_with_docker_stub(bash, tmp_path, "net-kernel")
    assert "bridge" in argv.splitlines()
    assert "host" not in argv.splitlines()

    argv = _run_host_launch_with_docker_stub(
        bash, tmp_path, "net-serve", extra_env={"TS_ENTRY": "ts_serve_probe.sh"}
    )
    assert "host" in argv.splitlines()
    assert "bridge" not in argv.splitlines()

    # And the override wins over either default.
    argv = _run_host_launch_with_docker_stub(
        bash,
        tmp_path,
        "net-override",
        extra_env={"TS_ENTRY": "ts_serve_probe.sh", "TS_NETWORK": "bridge"},
    )
    assert "bridge" in argv.splitlines()


def test_host_launch_records_the_resolved_image_digest(bash: str, tmp_path: Path) -> None:
    """A date tag is mutable, so the trial log has to say what content ran."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "image" ]; then echo "example/image@sha256:abc123"; exit 0; fi\n'
        "exit 0\n"
    )
    stub.chmod(0o755)
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    for name in ("host_launch.sh", "ts_kernel_probe.sh"):
        shutil.copy2(_SOURCE / name, scripts / name)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "TS_IMAGE": "example/image:tag",
            "TS_SCRIPTS_DIR": str(scripts),
            "TS_HF_DIR": str(tmp_path / "hf"),
            "TS_OUT_DIR": str(tmp_path / "out"),
            "TS_ENTRY": "ts_kernel_probe.sh",
        }
    )
    proc = subprocess.run(
        [bash, str(scripts / "host_launch.sh")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "image_digest=example/image@sha256:abc123" in proc.stdout


def test_host_launch_forwards_a_unique_run_token(bash: str, tmp_path: Path) -> None:
    """Each trial must get its own output token, minted host-side.

    Output files are named after this token, and TS_OUT_DIR is a single host
    directory shared by every trial in the matrix. The tempting in-container
    ``$$`` cannot serve: each ``docker run`` gets a fresh PID namespace, so the
    entry script is always PID 1 and all trials would collide on one filename,
    leaving only the last trial's evidence. This test would have caught exactly
    that -- it was observed on hardware as 12 trials producing 1 export file.
    """
    first = _token_from(_run_host_launch_with_docker_stub(bash, tmp_path, "a"))
    second = _token_from(_run_host_launch_with_docker_stub(bash, tmp_path, "b"))

    assert first != second, "two trials received the same run token"
    # A bare PID-like token is the failure mode described above.
    assert first not in ("1", ""), f"token {first!r} is not per-trial"


def test_host_launch_respects_a_caller_supplied_token(bash: str, tmp_path: Path) -> None:
    """An explicit TS_RUN_TOKEN wins, so a caller can correlate artifacts."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / "argv.explicit"
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "image" ]; then exit 0; fi\n'
        f'printf "%s\\n" "$@" > "{record}"\n'
    )
    stub.chmod(0o755)
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    for name in ("host_launch.sh", "ts_kernel_probe.sh"):
        shutil.copy2(_SOURCE / name, scripts / name)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "TS_IMAGE": "example/image:tag",
            "TS_SCRIPTS_DIR": str(scripts),
            "TS_HF_DIR": str(tmp_path / "hf"),
            "TS_OUT_DIR": str(tmp_path / "out"),
            "TS_ENTRY": "ts_kernel_probe.sh",
            "TS_RUN_TOKEN": "cell7-trial2",
        }
    )
    proc = subprocess.run(
        [bash, str(scripts / "host_launch.sh")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _token_from(record.read_text()) == "cell7-trial2"


def test_kernel_probe_names_its_export_after_the_token(bash: str, tmp_path: Path) -> None:
    """The export path must carry the token, or per-trial evidence is lost."""
    fixture = tmp_path / "export.json"
    fixture.write_text(json.dumps([_record(passed=True, m=1)]))
    bin_dir = _stub_bin(tmp_path, fixture)
    out_dir = tmp_path / "out"
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "TS_OUT_DIR": str(out_dir),
            "TS_KERNEL_MODE": "bench",
            "TS_KERNEL_NAME": "gluon_mm_a16w16_gfx950",
            "TS_RUN_TOKEN": "tok123",
        }
    )
    proc = subprocess.run(
        [bash, str(_SOURCE / "ts_kernel_probe.sh")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out_dir / "kernel_bench.tok123.json").exists(), sorted(
        p.name for p in out_dir.iterdir()
    )


def test_kernel_probe_requires_a_selector(bash: str, tmp_path: Path) -> None:
    env = dict(os.environ)
    env["TS_OUT_DIR"] = str(tmp_path / "out")
    env.pop("TS_KERNEL_OP", None)
    env.pop("TS_KERNEL_NAME", None)
    proc = subprocess.run(
        [bash, str(_SOURCE / "ts_kernel_probe.sh")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == _EXIT_USAGE
    assert "set TS_KERNEL_OP" in proc.stdout


def test_kernel_probe_rejects_unknown_mode(bash: str, tmp_path: Path) -> None:
    env = dict(os.environ)
    env.update(
        {
            "TS_OUT_DIR": str(tmp_path / "out"),
            "TS_KERNEL_MODE": "sideways",
            "TS_KERNEL_NAME": "k",
        }
    )
    proc = subprocess.run(
        [bash, str(_SOURCE / "ts_kernel_probe.sh")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == _EXIT_USAGE
    assert "TS_KERNEL_MODE must be" in proc.stdout


def _stub_bin(tmp_path: Path, fixture: Path) -> Path:
    """Build a PATH dir whose python3 fakes the benchmark CLI.

    The stub mimics the real CLI's behaviour precisely where it matters: it
    writes the export file and exits 0 regardless of what the export says. Any
    other python3 call (the probe parses its own export with an inline script on
    stdin) falls through to the real interpreter.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    real_python = shutil.which("python3")
    assert real_python is not None
    # Placeholder tokens rather than str.format/f-string: the body is shell and
    # is dense with ${...}, which would all need brace-doubling.
    template = """#!/usr/bin/env bash
for arg in "$@"; do
  if [ "${arg}" = "tokenspeed_kernel.benchmark" ]; then
    prev=""
    for a in "$@"; do
      if [ "${prev}" = "--export" ]; then cp "@FIXTURE@" "${a}"; fi
      prev="${a}"
    done
    echo "(stub benchmark)"
    exit 0
  fi
done
exec @PYTHON@ "$@"
"""
    stub = bin_dir / "python3"
    stub.write_text(template.replace("@FIXTURE@", str(fixture)).replace("@PYTHON@", real_python))
    stub.chmod(0o755)
    tokenspeed = bin_dir / "tokenspeed"
    tokenspeed.write_text("#!/bin/sh\necho stub\n")
    tokenspeed.chmod(0o755)
    return bin_dir


def _record(*, passed: bool | None, m: int) -> dict:
    return {
        "kernel_name": "gluon_mm_a16w16_gfx950",
        "solution": "gluon",
        "shape_params": {"M": m, "N": 4096, "K": 4096},
        "platform_arch": "amd:9.5",
        "median_latency_us": 21.3,
        "p99_latency_us": 37.5,
        "tflops": 1.57,
        "bandwidth_gb_s": 1574.6,
        "numerics_passed": passed,
        "max_abs_diff": 0.0 if passed else 0.5,
        "max_rel_diff": 0.0 if passed else 0.25,
    }


def _run_bench_probe(bash: str, tmp_path: Path, export: object) -> subprocess.CompletedProcess:
    fixture = tmp_path / "export.json"
    fixture.write_text(json.dumps(export))
    bin_dir = _stub_bin(tmp_path, fixture)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "TS_OUT_DIR": str(tmp_path / "out"),
            "TS_KERNEL_MODE": "bench",
            "TS_KERNEL_NAME": "gluon_mm_a16w16_gfx950",
        }
    )
    return subprocess.run(
        [bash, str(_SOURCE / "ts_kernel_probe.sh")],
        capture_output=True,
        text=True,
        env=env,
    )


def test_clean_export_passes(bash: str, tmp_path: Path) -> None:
    proc = _run_bench_probe(
        bash, tmp_path, [_record(passed=True, m=1), _record(passed=True, m=4096)]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TS_KERNEL_RESULT: pass" in proc.stdout
    assert proc.stdout.count("TS_KERNEL_METRIC") == 2


def test_numerics_mismatch_fails_despite_cli_success(bash: str, tmp_path: Path) -> None:
    """The stub exits 0, as the real CLI does; the gate must still fail."""
    proc = _run_bench_probe(
        bash, tmp_path, [_record(passed=True, m=1), _record(passed=False, m=4096)]
    )
    assert proc.returncode == _EXIT_NUMERICS, proc.stdout + proc.stderr
    assert "TS_KERNEL_FAIL: numerics_mismatch count=1" in proc.stdout
    # The offending record must be named, with the diffs, so triage does not
    # require re-running the kernel.
    assert "TS_KERNEL_FAILREC" in proc.stdout
    assert "max_abs_diff=0.5" in proc.stdout
    assert "TS_KERNEL_RESULT: pass" not in proc.stdout


def test_unverified_records_do_not_fail(bash: str, tmp_path: Path) -> None:
    """``numerics_passed: null`` means "not checked", not "failed".

    The reference solutions (torch_mm) report null because they are what the
    others are compared against; treating that as a failure would fail every
    reference cell in the matrix.
    """
    proc = _run_bench_probe(bash, tmp_path, [_record(passed=None, m=1)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TS_KERNEL_RESULT: pass" in proc.stdout


def test_empty_export_fails(bash: str, tmp_path: Path) -> None:
    """An empty export means the selector matched nothing -- a recipe bug."""
    proc = _run_bench_probe(bash, tmp_path, [])
    assert proc.returncode == _EXIT_NO_RECORDS, proc.stdout + proc.stderr
    assert "benchmark_exported_no_records" in proc.stdout


def test_renamed_verdict_field_fails_instead_of_passing(bash: str, tmp_path: Path) -> None:
    """A missing `numerics_passed` must not read as `null`.

    `null` legitimately means "not verified", so reading the field with .get()
    makes an upstream rename indistinguishable from it -- the wrong-answer gate
    would quietly stop existing while every trial stayed green. This is the
    failure mode the probe exists to prevent, so it has to be loud.
    """
    record = _record(passed=False, m=1)
    record["numerics_ok"] = record.pop("numerics_passed")
    proc = _run_bench_probe(bash, tmp_path, [record])
    assert proc.returncode == _EXIT_SCHEMA, proc.stdout + proc.stderr
    assert "export_schema_changed" in proc.stdout
    assert "has no numerics_passed field" in proc.stdout


def test_wrongly_typed_verdict_field_fails(bash: str, tmp_path: Path) -> None:
    """Same guard for a type change: "false" is a non-empty string, so a truthy
    check would read a failing record as a pass."""
    record = _record(passed=True, m=1)
    record["numerics_passed"] = "false"
    proc = _run_bench_probe(bash, tmp_path, [record])
    assert proc.returncode == _EXIT_SCHEMA, proc.stdout + proc.stderr
    assert "expected bool or null" in proc.stdout


def test_non_object_record_fails(bash: str, tmp_path: Path) -> None:
    proc = _run_bench_probe(bash, tmp_path, ["not-an-object"])
    assert proc.returncode == _EXIT_SCHEMA, proc.stdout + proc.stderr
    assert "expected object" in proc.stdout


# --- recipes -------------------------------------------------------------


def _recipes() -> list[Path]:
    """The probe-mode recipes in the TokenSpeed recipe directory.

    Filtered by ``mode``, not by filename: the same directory also holds the
    triage-mode recipes that drive the ``tokenspeed_serve`` workload, which have
    no ``mitigation_axis`` or ``custom_patterns`` and would fail every
    assertion below. Selecting on ``mode`` means a new probe recipe is picked up
    automatically while a new triage recipe is correctly ignored --
    ``tests/workloads/test_tokenspeed_serve.py`` covers those.
    """
    probe_recipes = []
    for path in sorted(_RECIPES.glob("*.yaml")):
        with path.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        if isinstance(doc, dict) and doc.get("mode") == "probe":
            probe_recipes.append(path)
    return probe_recipes


def _sidecars() -> list[Path]:
    return sorted(_RECIPES.glob("*.json"))


def test_recipes_exist() -> None:
    assert _recipes(), f"no recipes found in {_RECIPES}"


def test_recipe_files_are_not_gitignored() -> None:
    """Every file in the recipe dir must be trackable.

    The repo's .gitignore has a blanket ``*.json`` rule for experiment output,
    which also swallows the mitigations sidecar. That fails in the worst way:
    the sidecar stays present locally so everything works for whoever wrote it,
    but a fresh clone has no sidecar and the kernel recipe dies with
    ``unknown mitigation 'ts_gemm_gluon_bf16'``. A negation in .gitignore keeps
    it tracked; this test is what notices if that negation is dropped or a new
    ignored extension shows up here.
    """
    if not (_REPO / ".git").exists():
        pytest.skip("not a git checkout")
    if shutil.which("git") is None:
        pytest.skip("git not available")

    files = sorted(p for p in _RECIPES.iterdir() if p.is_file())
    assert files, f"no files in {_RECIPES}"
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", *(str(p) for p in files)],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    # check-ignore exits 0 when it matched something, 1 when nothing is ignored.
    ignored = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "these recipe files are gitignored and would be missing from a clone: "
        f"{ignored}. Add a '!' negation in .gitignore."
    )


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.name)
def test_recipe_is_wellformed(recipe: Path) -> None:
    doc = yaml.safe_load(recipe.read_text())
    assert doc["schema_version"] == 1
    assert doc["mode"] == "probe"
    assert doc["ticket"]
    assert doc["mitigation_axis"]
    assert doc["diagnostic_axis"]

    # docker does not inherit the parent environment, so `inherit` mode would
    # silently drop every per-cell mitigation and make all cells identical.
    assert doc["env_passthrough_mode"] == "file", (
        f"{recipe.name} must use env_passthrough_mode: file -- the wrapped "
        "command is `docker run`, which does not inherit os.environ"
    )

    # The monitored PID is a bash wrapper blocking on `docker run`, so the hang
    # detector sees an idle process for the whole trial. The grace period has to
    # cover the trial or every cell latches a false tier2:hang.
    assert (
        doc["hang_grace_period_at_start"] >= doc["timeout_per_trial"]
    ), f"{recipe.name}: hang_grace_period_at_start must be >= timeout_per_trial"


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.name)
def test_recipe_patterns_are_valid_regexes(recipe: Path) -> None:
    import re

    doc = yaml.safe_load(recipe.read_text())
    patterns = doc.get("custom_patterns") or []
    assert patterns, f"{recipe.name} declares no custom_patterns"
    seen = set()
    for pattern in patterns:
        assert pattern["id"] not in seen, f"duplicate pattern id {pattern['id']}"
        seen.add(pattern["id"])
        re.compile(pattern["match"]["regex"])
        assert pattern["on_match"] in {"fail", "error", "warn"}


def _fake_workspace(tmp_path: Path, body: str) -> Path:
    """A minimal stand-in for /workspace holding one suite at pkg/test/ops.

    The `test` directory matters: ts_pytest_probe.sh derives the working
    directory from it, mirroring how both real TokenSpeed suites are laid out.
    """
    suite_dir = tmp_path / "pkg" / "test" / "ops"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test_stub.py").write_text(body)
    return tmp_path


def _run_pytest_probe(bash: str, workspace: Path, out: Path, **env: str):
    """Run ts_pytest_probe.sh against a stub workspace.

    The script invokes `python3 -m pytest`, and the bare `python3` on PATH is
    not necessarily the interpreter running these tests -- nor does it
    necessarily have pytest. A shim directory pointed at sys.executable makes
    the script exercise the real pytest without hardcoding an interpreter.
    """
    import sys

    shim_dir = out / "shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "python3"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    shim.chmod(0o755)

    return subprocess.run(
        [bash, str(_SOURCE / "ts_pytest_probe.sh")],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{shim_dir}:{os.environ['PATH']}",
            "TS_WORKSPACE": str(workspace),
            "TS_OUT_DIR": str(out),
            **env,
        },
    )


def test_pytest_probe_requires_a_suite(bash: str, tmp_path: Path) -> None:
    proc = subprocess.run(
        [bash, str(_SOURCE / "ts_pytest_probe.sh")],
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "TS_OUT_DIR": str(tmp_path)},
    )
    assert proc.returncode == _EXIT_USAGE
    assert "TS_PYTEST_SUITE" in proc.stdout + proc.stderr


def test_pytest_probe_rejects_a_missing_suite(bash: str, tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path, "def test_ok():\n    assert True\n")
    proc = _run_pytest_probe(
        bash,
        workspace,
        tmp_path / "out",
        TS_PYTEST_SUITE="pkg/test/ops/test_absent.py",
    )
    assert proc.returncode == _EXIT_USAGE
    assert "does not exist" in proc.stdout


def test_pytest_probe_passes_and_reports_counts(bash: str, tmp_path: Path) -> None:
    workspace = _fake_workspace(
        tmp_path,
        "import pytest\n\n"
        "def test_ok():\n    assert True\n\n"
        "@pytest.mark.skip(reason='not on this platform')\n"
        "def test_skipped():\n    assert False\n",
    )
    proc = _run_pytest_probe(
        bash,
        workspace,
        tmp_path / "out",
        TS_PYTEST_SUITE="pkg/test/ops/test_stub.py",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TS_PYTEST_METRIC: tests_passed=1" in proc.stdout
    assert "TS_PYTEST_METRIC: tests_skipped=1" in proc.stdout
    assert "TS_PYTEST_RESULT: pass" in proc.stdout


def test_pytest_probe_fails_the_trial_on_a_test_failure(bash: str, tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path, "def test_bad():\n    assert False\n")
    proc = _run_pytest_probe(
        bash,
        workspace,
        tmp_path / "out",
        TS_PYTEST_SUITE="pkg/test/ops/test_stub.py",
    )
    assert proc.returncode == _EXIT_PYTEST_FAILED
    assert "TS_PYTEST_FAIL: tests_failed" in proc.stdout


def test_pytest_probe_rejects_an_all_skipped_run(bash: str, tmp_path: Path) -> None:
    """The silent-pass guard.

    pytest exits 0 when every test was skipped, and these suites skip heavily --
    NVIDIA-only solutions are simply not registered on AMD, so a single file can
    report hundreds of skips. Without this guard a cell that proved nothing would
    be indistinguishable from a cell that verified the kernel.
    """
    workspace = _fake_workspace(
        tmp_path,
        "import pytest\n\n"
        "@pytest.mark.skip(reason='solution not registered')\n"
        "def test_skipped():\n    assert False\n",
    )
    proc = _run_pytest_probe(
        bash,
        workspace,
        tmp_path / "out",
        TS_PYTEST_SUITE="pkg/test/ops/test_stub.py",
    )
    assert proc.returncode == _EXIT_PYTEST_NOTHING_RAN, proc.stdout
    assert "TS_PYTEST_FAIL: nothing_executed" in proc.stdout


# An empty value is deliberately absent from this list: `${TS_MIN_PASSED:-1}`
# treats empty as unset, which is the right reading of `TS_MIN_PASSED=`.
@pytest.mark.parametrize("value", ["0", "abc", "-1", "1.5"])
def test_pytest_probe_rejects_an_unusable_min_passed(bash: str, tmp_path: Path, value: str) -> None:
    """TS_MIN_PASSED is what the all-skipped guard compares against, so an
    unusable value disables the guard rather than being merely wrong: 0 passes
    trivially, and a non-numeric makes `[` error out -- which, because pytest
    itself returned 0, also falls through to a green trial."""
    workspace = _fake_workspace(tmp_path, "def test_ok():\n    assert True\n")
    proc = _run_pytest_probe(
        bash,
        workspace,
        tmp_path / "out",
        TS_PYTEST_SUITE="pkg/test/ops/test_stub.py",
        TS_MIN_PASSED=value,
    )
    assert proc.returncode == _EXIT_USAGE, proc.stdout
    assert "TS_MIN_PASSED must be" in proc.stdout


def test_pytest_probe_accepts_a_valid_min_passed(bash: str, tmp_path: Path) -> None:
    workspace = _fake_workspace(tmp_path, "def test_ok():\n    assert True\n")
    proc = _run_pytest_probe(
        bash,
        workspace,
        tmp_path / "out",
        TS_PYTEST_SUITE="pkg/test/ops/test_stub.py",
        TS_MIN_PASSED="1",
    )
    assert proc.returncode == 0, proc.stdout


def test_pytest_probe_rejects_an_empty_selection(bash: str, tmp_path: Path) -> None:
    """Same guard, reached via -k rather than skips: deselection also exits 0."""
    workspace = _fake_workspace(tmp_path, "def test_ok():\n    assert True\n")
    proc = _run_pytest_probe(
        bash,
        workspace,
        tmp_path / "out",
        TS_PYTEST_SUITE="pkg/test/ops/test_stub.py",
        TS_PYTEST_K="no_such_selector",
    )
    assert proc.returncode == _EXIT_PYTEST_NOTHING_RAN, proc.stdout
    assert "TS_PYTEST_FAIL: nothing_executed" in proc.stdout


def test_pytest_probe_report_is_token_qualified(bash: str, tmp_path: Path) -> None:
    """Per-trial JUnit reports must not collide.

    Every `docker run` gets a fresh PID namespace, so an in-container $$ is
    always 1 and every trial in a matrix would write the same filename, leaving
    only the last trial's evidence. host_launch.sh mints the token host-side.
    """
    workspace = _fake_workspace(tmp_path, "def test_ok():\n    assert True\n")
    out = tmp_path / "out"
    proc = _run_pytest_probe(
        bash,
        workspace,
        out,
        TS_PYTEST_SUITE="pkg/test/ops/test_stub.py",
        TS_RUN_TOKEN="tok-abc123",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "pytest.tok-abc123.xml").exists(), sorted(p.name for p in out.iterdir())


def test_pytest_sidecar_entries_all_pin_a_suite() -> None:
    """Every pytest selector must set TS_PYTEST_SUITE.

    An entry that sets nothing would not fail loudly: the probe would report a
    usage error for a missing suite, which reads as a script bug rather than an
    incomplete sidecar.
    """
    sidecar = json.loads((_RECIPES / "tokenspeed-pytest-sidecar.json").read_text())
    assert sidecar["version"] == 1
    assert sidecar["mitigations"], "sidecar defines no selectors"
    for name, env in sidecar["mitigations"].items():
        assert "TS_PYTEST_SUITE" in env, f"{name} pins no suite"
        suite = env["TS_PYTEST_SUITE"]
        assert not suite.startswith("/"), f"{name}: suite must be workspace-relative"
        # The probe derives its working directory from a `test` path component,
        # because each suite is a package rooted at that directory's parent.
        assert "/test/" in suite, f"{name}: {suite} has no /test/ component"


def test_harvest_stages_code_objects_content_addressed() -> None:
    """Staged filenames must include a digest.

    The Triton cache keeps one directory per shape specialization and reuses
    file names across them -- a single attention run emits ten distinct
    ``_fwd_kernel.hsaco``. Staging by bare name overwrites them in turn, so the
    generated recipe pins digests that match nothing on disk but the last copy,
    and Waitcheck then rejects every earlier entry for a digest mismatch.
    """
    source = (_SOURCE / "harvest_code_objects.py").read_text()
    assert 'f"{obj.stem}.{digest[:12]}{obj.suffix}"' in source, (
        "harvest must stage code objects under a digest-qualified name; "
        "staging by obj.name silently collides across shape specializations"
    )


def _harvest_module():
    """Import harvest_code_objects.py by path.

    It lives under a workloads directory that is not an importable package, and
    it deliberately has no dependency beyond the standard library.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ts_harvest_consan", _SOURCE / "harvest_code_objects.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FAKE_LOADER = """\
import json
import sys

argv = sys.argv[1:]
opts = {}
for index, item in enumerate(argv):
    if item.startswith("--"):
        opts[item] = argv[index + 1] if index + 1 < len(argv) else None

obj = opts["--copy-object"]
open(obj, "wb").write(b"fake-code-object")
# The real loader lifts the sidecars too; a test only needs them to exist.
open(obj.replace(".hsaco", ".json"), "w").write("{}")

shim = opts["--output"]
open(shim, "w").write("#!/bin/sh\\nexit 0\\n")

# Record the argv so a test can assert on how the loader was driven. Named so
# it does not look like a shim to a consan_* glob.
import os

record = os.path.join(
    os.path.dirname(shim), "argv-" + os.path.basename(shim) + ".json"
)
with open(record, "w") as handle:
    json.dump(argv, handle)
"""


def _fake_kernels(count: int, cache: Path) -> list[dict]:
    """Harvest-shaped identities pointing at throwaway cache entries."""
    kernels = []
    for index in range(count):
        entry = cache / f"HASH{index}"
        entry.mkdir(parents=True, exist_ok=True)
        obj = entry / "_fwd_kernel.hsaco"
        payload = f"object-{index}".encode()
        obj.write_bytes(payload)
        kernels.append(
            {
                "name": "_fwd_kernel",
                "code_object": str(obj),
                # A real digest, so the 12-char prefixes differ. A counter
                # padded to 64 hex digits collides on the prefix, which is
                # exactly the collision the staging is designed to avoid.
                "sha256": hashlib.sha256(payload).hexdigest(),
                "code_object_index": 0,
                "cache_object": str(obj),
            }
        )
    return kernels


def test_harvest_consan_reports_a_missing_loader(tmp_path: Path) -> None:
    """``scripts/`` is not packaged, so an installed aorta has no loader.

    The message has to name the flag, or the failure reads as a broken install
    rather than a path that simply has to be supplied.
    """
    module = _harvest_module()
    with pytest.raises(SystemExit) as excinfo:
        module._write_consan_assets(
            tmp_path,
            _fake_kernels(1, tmp_path / "cache"),
            "gfx950",
            tmp_path / "absent" / "triton_consan_loader.py",
            "lenient",
            None,
        )
    message = str(excinfo.value)
    assert "--consan-loader" in message
    assert "wheel" in message


def test_harvest_consan_emits_one_recipe_per_identity(tmp_path: Path) -> None:
    """ConSan takes exactly one code object per run.

    A TokenSpeed kernel compiles to many shape-specialized objects -- an
    attention harvest yields twenty -- so a single kernel_list recipe cannot
    express it and the harvest has to fan out.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)

    module._write_consan_assets(
        tmp_path / "dest",
        _fake_kernels(3, tmp_path / "cache"),
        "gfx950",
        loader,
        "lenient",
        None,
    )

    consan = tmp_path / "dest" / "consan"
    recipes = sorted(consan.glob("consan-*.yaml"))
    assert len(recipes) == 3
    assert len(sorted((consan / "bin").glob("consan_*"))) == 3

    manifest = json.loads((consan / "manifest.json").read_text())
    assert manifest["consan_policy"] == "lenient"
    assert manifest["skipped"] == 0
    assert len(manifest["entries"]) == 3

    # Digest-qualified, for the same reason the Waitcheck staging is: every
    # object here is named _fwd_kernel, so bare names would collide and leave
    # one recipe per name instead of one per object.
    assert len({recipe.name for recipe in recipes}) == 3


def test_harvest_consan_recipe_pins_the_identity_and_the_shim(tmp_path: Path) -> None:
    """The recipe must name a single kernel, its digest, and the shim.

    Without ``code_object_sha256`` the run would accept whatever bytes sit at
    the path, which for a Triton cache is exactly the thing that changes.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)
    kernels = _fake_kernels(1, tmp_path / "cache")

    module._write_consan_assets(tmp_path / "dest", kernels, "gfx950", loader, "lenient", None)

    recipe_path = next((tmp_path / "dest" / "consan").glob("consan-*.yaml"))
    recipe = yaml.safe_load(recipe_path.read_text())
    plan = recipe["sanitizer_plan"]

    assert recipe["mode"] == "sanitizer"
    assert plan["sanitizers"] == ["consan"]
    assert plan["target"] == "gfx950"
    # kind: kernel, not kernel_list -- ConSan is one object per run.
    assert plan["source"]["kind"] == "kernel"
    assert plan["source"]["kernel"]["name"] == "_fwd_kernel"
    assert plan["source"]["kernel"]["code_object_sha256"] == kernels[0]["sha256"]
    assert plan["source"]["consan_command"].endswith(
        f"consan__fwd_kernel.{kernels[0]['sha256'][:12]}"
    )
    assert plan["policy"]["consan_policy"] == "lenient"


def test_harvest_consan_default_policy_explains_itself(tmp_path: Path) -> None:
    """A lenient default needs its reasoning in the file.

    strict sets RJ_CONSAN_MOI_REQUIRE_RECORDS, which wants visible dynamic
    records; the loader runs in load mode and never dispatches, so strict fails
    closed with exit 86 however healthy the run was. Someone will otherwise
    "fix" the policy and get an opaque failure.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)

    module._write_consan_assets(
        tmp_path / "dest",
        _fake_kernels(1, tmp_path / "cache"),
        "gfx950",
        loader,
        "lenient",
        None,
    )
    text = next((tmp_path / "dest" / "consan").glob("consan-*.yaml")).read_text()
    assert "86" in text
    assert "never dispatches" in text


def test_harvest_consan_limit_caps_the_fan_out(tmp_path: Path) -> None:
    """Each identity is a separate ConSan run, so 20 objects is 20 runs."""
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)

    module._write_consan_assets(
        tmp_path / "dest",
        _fake_kernels(5, tmp_path / "cache"),
        "gfx950",
        loader,
        "lenient",
        2,
    )
    consan = tmp_path / "dest" / "consan"
    assert len(sorted(consan.glob("consan-*.yaml"))) == 2
    assert json.loads((consan / "manifest.json").read_text())["skipped"] == 3


def test_harvest_consan_limit_of_zero_is_an_error(tmp_path: Path) -> None:
    """A limit of 0 is a caller mistake, not a request for everything.

    Guarded because the natural implementation tests the limit for truthiness,
    which silently turns 0 into "emit all 20".
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)
    module._write_consan_assets(
        tmp_path / "dest",
        _fake_kernels(2, tmp_path / "cache"),
        "gfx950",
        loader,
        "lenient",
        0,
    )
    assert sorted((tmp_path / "dest" / "consan").glob("consan-*.yaml")) == []

    # And the CLI rejects it up front rather than emitting nothing quietly.
    proc = subprocess.run(
        [
            "python3",
            str(_SOURCE / "harvest_code_objects.py"),
            "--image",
            "img",
            "--kernel",
            "k",
            "--dest",
            "/tmp/whatever",
            "--consan-limit",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "--consan-limit must be at least 1" in proc.stderr


def test_harvest_consan_disambiguates_by_kernel_name(tmp_path: Path) -> None:
    """One object can hold several kernels and the loader fails closed on that.

    An attention harvest produces ten objects all called ``_fwd_kernel``, so
    the selection has to be pinned rather than left to be inferred.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)

    module._write_consan_assets(
        tmp_path / "dest",
        _fake_kernels(1, tmp_path / "cache"),
        "gfx950",
        loader,
        "lenient",
        None,
    )
    recorded = next((tmp_path / "dest" / "consan" / "bin").glob("argv-*.json"))
    argv = json.loads(recorded.read_text())
    assert "emit-command" in argv
    assert "--kernel-name" in argv
    assert argv[argv.index("--kernel-name") + 1] == "_fwd_kernel"
    # --copy-object is what lifts the object out of the Triton cache, which the
    # next harvest deletes.
    assert "--copy-object" in argv


def test_harvest_records_the_cache_object_not_just_the_staged_copy() -> None:
    """ConSan needs the sidecars Triton wrote beside the object.

    Waitcheck staging copies only the ``.hsaco``, so the loader has to be
    pointed at the original cache entry to find the ``.json`` metadata and the
    ``.amdgcn`` listing it reads the kernarg segment size from.
    """
    source = (_SOURCE / "harvest_code_objects.py").read_text()
    assert '"cache_object": str(obj)' in source
    assert 'kernel["cache_object"]' in source


def test_harvest_suite_root_matches_the_probe_script() -> None:
    """The harvest tool and the probe must agree on a suite's package root.

    Both compute it as the parent of the `test` directory. If they diverge, the
    harvest compiles from the wrong working directory and collects nothing,
    which surfaces as a confusing "produced no .hsaco" rather than a path bug.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ts_harvest", _SOURCE / "harvest_code_objects.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._suite_root("tokenspeed-kernel/test/ops/test_attention.py") == (
        "tokenspeed-kernel"
    )
    assert module._suite_root("tokenspeed-kernel-amd/test/ops") == "tokenspeed-kernel-amd"
    # No `test` component: fall back rather than guessing a parent.
    assert module._suite_root("some/other/path.py") == "."


def test_harvest_does_not_double_prefix_an_absolute_suite() -> None:
    """An absolute --pytest-suite must survive unchanged.

    `_suite_root` already returns an absolute root for an absolute suite, so
    prefixing /workspace a second time yields `/workspace//workspace/...` and
    docker fails on the working directory before pytest ever starts -- which
    reads as a broken image rather than a bad argument.
    """
    import argparse

    module = _harvest_module()

    absolute = argparse.Namespace(
        pytest_suite="/workspace/tokenspeed-kernel/test/ops/test_attention.py",
        pytest_k=None,
    )
    cmd, workdir, _ = module._driver(absolute)
    assert workdir == "/workspace/tokenspeed-kernel"
    assert "/workspace/tokenspeed-kernel/test/ops/test_attention.py" in cmd
    assert not any("//" in part for part in [workdir, *cmd])

    relative = argparse.Namespace(
        pytest_suite="tokenspeed-kernel/test/ops/test_attention.py", pytest_k=None
    )
    cmd, workdir, _ = module._driver(relative)
    assert workdir == "/workspace/tokenspeed-kernel"
    assert "/workspace/tokenspeed-kernel/test/ops/test_attention.py" in cmd

    # No `test` component: the workdir is the mount root, not `/workspace/.`.
    bare = argparse.Namespace(pytest_suite="some/other/path.py", pytest_k=None)
    _, workdir, _ = module._driver(bare)
    assert workdir == "/workspace"


def test_harvest_carries_the_kernel_entry_offset_into_the_recipe(
    tmp_path: Path,
) -> None:
    """entry_offset is what makes a harvested identity exact.

    Without it KernelIdentity.exact is False, so Waitcheck collapses every
    kernel sharing a code object into a single whole-object scan and reports
    findings with no kernel name -- a helper kernel's finding could then be read
    as the harvested one's.
    """
    module = _harvest_module()

    assert module._entry_offset("0x1200") == 0x1200
    assert module._entry_offset(4608) == 4608
    assert module._entry_offset("4608") == 4608
    # Unusable values degrade to a whole-object scan rather than pinning a wrong
    # offset, which Waitcheck would reject outright.
    assert module._entry_offset(None) is None
    assert module._entry_offset("") is None
    assert module._entry_offset("not-a-number") is None
    assert module._entry_offset(True) is None

    recipe = module._write_recipe(
        tmp_path,
        "attention",
        "gfx950",
        [
            {
                "name": "_fwd_kernel",
                "code_object": str(tmp_path / "a.hsaco"),
                "sha256": "0" * 64,
                "code_object_index": 0,
                "entry_offset": 0x1200,
            },
            {
                "name": "_bwd_kernel",
                "code_object": str(tmp_path / "b.hsaco"),
                "sha256": "1" * 64,
                "code_object_index": 0,
                "entry_offset": None,
            },
        ],
    )
    doc = yaml.safe_load(recipe.read_text())
    kernels = doc["sanitizer_plan"]["source"]["kernels"]
    assert kernels[0]["entry_offset"] == 0x1200
    # Absent rather than null: the recipe loader validates this as an integer,
    # so a null would fail the whole recipe instead of degrading one entry.
    assert "entry_offset" not in kernels[1]


def test_harvest_aborts_when_a_supplied_waitcheck_rejects_the_object(
    tmp_path: Path,
) -> None:
    """A failing inventory must not fall back to a guessed name.

    The fallback exists for the case where no Waitcheck binary was supplied. If
    one was supplied and rejected the object, a guess would emit a recipe whose
    names and indices were never verified, and Waitcheck would then attribute a
    finding to whatever the guess happened to name.
    """
    module = _harvest_module()

    failing = tmp_path / "rj_waitcheck"
    failing.write_text("#!/usr/bin/env bash\necho 'bad code object' >&2\nexit 3\n")
    failing.chmod(0o755)
    obj = tmp_path / "kernel.hsaco"
    obj.write_bytes(b"fake")

    with pytest.raises(SystemExit) as excinfo:
        module._inventory(failing, obj)
    assert "--list-kernels failed" in str(excinfo.value)
    assert "bad code object" in str(excinfo.value)

    # Intentionally omitted stays a soft path: no binary, no inventory, no error.
    assert module._inventory(None, obj) == []
    assert module._inventory(tmp_path / "absent", obj) == []


def test_harvest_names_its_container_so_a_timeout_is_recoverable() -> None:
    """`subprocess.run(timeout=...)` kills the docker client, not the container.

    aorta escalates to SIGKILL 10s after SIGTERM while a compile can be given
    minutes, so without a handle a hung harvest keeps a GPU busy for the rest of
    the sweep.
    """
    source = (_SOURCE / "harvest_code_objects.py").read_text()
    assert 'f"aorta-ts-harvest-{os.getpid()}"' in source
    assert "--name" in source
    assert "_force_remove_container" in source
    assert "except subprocess.TimeoutExpired" in source


def _coverage_module():
    """Import map_kernel_test_coverage.py by path.

    Importable outside the container because every TokenSpeed import in it is
    deferred into a function; only the per-suite child needs the real registry.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ts_coverage_map", _SOURCE / "map_kernel_test_coverage.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coverage_map_aborts_when_a_suite_probe_produces_no_map(
    tmp_path: Path,
) -> None:
    """A crashed suite must abort, not be silently counted as uncovered.

    Continuing turns "we could not measure this suite" into "these kernels have
    no test", which understates coverage in the one number this tool exists to
    state precisely -- and does so without any signal that it happened.
    """
    workspace = tmp_path / "ws"
    suite = workspace / "pkg" / "test" / "ops"
    suite.mkdir(parents=True)
    (suite / "test_stub.py").write_text("def test_ok():\n    assert True\n")

    # Run outside the container, so the child's `_registry_inventory` import of
    # tokenspeed_kernel fails -- exactly the "probe crashed before writing its
    # map" shape this guards.
    proc = subprocess.run(
        [
            "python3",
            str(_SOURCE / "map_kernel_test_coverage.py"),
            "--workspace",
            str(workspace),
            "--suite",
            "pkg/test/ops",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == _EXIT_USAGE, proc.stdout + proc.stderr
    assert "coverage totals would be incomplete" in proc.stderr


@contextlib.contextmanager
def _fake_registry(monkeypatch, impls: dict[str, object], candidates=()):
    """Stand in for `tokenspeed_kernel.registry` so the probe can be driven.

    `_install_probe` patches the class, so a fake class is all it needs -- and
    driving it directly is the only way to assert what the probe records, which
    is the whole verdict this tool produces.
    """
    import types

    class KernelRegistry:
        @classmethod
        def get(cls):
            return cls()

        def get_for_operator(self, family, mode, **kwargs):
            return list(candidates)

        def get_impl(self, name, *args, **kwargs):
            return impls.get(name)

    pkg = types.ModuleType("tokenspeed_kernel")
    mod = types.ModuleType("tokenspeed_kernel.registry")
    mod.KernelRegistry = KernelRegistry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel", pkg)
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel.registry", mod)
    yield KernelRegistry


def test_coverage_map_counts_entry_not_lookup(monkeypatch) -> None:
    """`covered` has to mean the kernel *ran*.

    `get_impl` returning a callable proves only that a test asked for one.
    Upstream's `test/ops/moe/test_latent_input.py` calls it purely to assert each
    implementation's `__module__` and never launches anything, so crediting the
    lookup marks kernels covered that never executed -- the same overstatement as
    counting candidates instead of dispatches, one layer down.
    """
    from collections import defaultdict

    module = _coverage_module()
    entered: dict = defaultdict(set)
    dispatched: dict = defaultdict(set)
    candidates: dict = defaultdict(set)
    requested: dict = {}

    def ran_kernel(*args, **kwargs):
        return "result"

    inspected_kernel = types.SimpleNamespace(__module__="tokenspeed_kernel.moe.gluon")

    with _fake_registry(monkeypatch, {"k_ran": ran_kernel, "k_inspected": inspected_kernel}):
        module._install_probe(entered, dispatched, candidates, requested)
        from tokenspeed_kernel.registry import KernelRegistry  # type: ignore

        registry = KernelRegistry()

        # A test that actually runs the kernel.
        assert registry.get_impl("k_ran")(1, 2) == "result"
        # A test that only inspects where the implementation lives.
        assert registry.get_impl("k_inspected").__module__ == "tokenspeed_kernel.moe.gluon"

    assert set(entered) == {"k_ran"}, "only the entered kernel counts as covered"
    assert set(dispatched) == {"k_ran", "k_inspected"}, "both were looked up"
    assert entered["k_ran"] == {"call"}


def test_coverage_map_counts_a_triton_launch(monkeypatch) -> None:
    """Triton entry points are invoked as `kernel[grid](...)`.

    Subscripting is the launch decision -- what it returns is the launcher -- so
    the probe has to record there. This is why a `functools.wraps` wrapper will
    not do: it forwards neither `__getitem__` nor `__module__`.
    """
    from collections import defaultdict

    module = _coverage_module()
    entered: dict = defaultdict(set)
    dispatched: dict = defaultdict(set)

    launched = []

    class JitKernel:
        __module__ = "tokenspeed_kernel.gluon.attention"

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launched.append(grid)

            return launch

    with _fake_registry(monkeypatch, {"k_triton": JitKernel()}):
        module._install_probe(entered, dispatched, defaultdict(set), {})
        from tokenspeed_kernel.registry import KernelRegistry  # type: ignore

        impl = KernelRegistry().get_impl("k_triton")
        # Reading metadata is not a launch.
        assert impl.__module__ == "tokenspeed_kernel.gluon.attention"
        assert not entered
        impl[(4, 1, 1)](7)

    assert launched == [(4, 1, 1)]
    assert entered["k_triton"] == {"launch"}


def test_coverage_probe_does_not_disturb_the_suites_it_measures(monkeypatch) -> None:
    """The suites are the measurement, so the probe must be invisible to them.

    Anything the proxy changes -- an implementation's `__module__`, identity in a
    set, an attribute write landing on the wrapper instead of the kernel --
    corrupts the very run it is observing.
    """
    module = _coverage_module()
    recorded: list = []

    def impl(a, b):
        return a + b

    impl.solution = "gluon"  # type: ignore[attr-defined]
    original_module = impl.__module__

    proxy = module._EntryProbe(impl, "k", lambda n, how: recorded.append((n, how)))

    assert impl.__module__ == original_module, "wrapping must not mutate the kernel"
    assert proxy.__module__ == original_module
    assert proxy.__name__ == "impl"
    assert proxy.solution == "gluon"
    assert repr(impl) in repr(proxy) or "impl" in repr(proxy)
    assert not recorded, "inspecting the implementation is not an entry"

    # Identity-ish behaviour: a suite holding impls in a set, or comparing two
    # lookups of the same kernel, must not see two different objects.
    other = module._EntryProbe(impl, "k", lambda n, how: None)
    assert proxy == impl and proxy == other
    assert len({proxy, other, impl}) == 1

    # Attribute writes reach the kernel, not the wrapper.
    proxy.extra = 5
    assert impl.extra == 5  # type: ignore[attr-defined]

    assert proxy(2, 3) == 5
    assert recorded == [("k", "call")]


def test_coverage_map_reports_three_states() -> None:
    """Entered, looked-up-only and candidate-only are different claims, and each
    of the wider two has at some point been mistaken for coverage."""
    source = (_SOURCE / "map_kernel_test_coverage.py").read_text()

    assert '"covered": name in entered' in source
    assert '"lookup_only": name in dispatched and name not in entered' in source
    for key in (
        "kernels_covered",
        "kernels_lookup_only",
        "kernels_candidate_only",
        "kernels_uncovered",
    ):
        assert key in source, key


def test_harness_survey_aborts_on_an_import_failure() -> None:
    """A swallowed import makes an incomplete survey look like a finding.

    The numerics imports are what populate the generator and shape registries
    every status is derived from, so losing one reclassifies real operators as
    `no_input_generator` while the tool still exits 0 with normal-looking JSON.
    """
    source = (_SOURCE / "list_harness_coverage.py").read_text()
    assert "raise SystemExit(64)" in source
    assert "would be incomplete" in source

    # Outside the container the tokenspeed_kernel import fails at module import,
    # which must not be a zero exit either.
    proc = subprocess.run(
        ["python3", str(_SOURCE / "list_harness_coverage.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


def test_sidecar_pairings_are_self_consistent() -> None:
    """Each cell must carry a dtype its kernel actually accepts.

    A mismatched pairing does not fail loudly in an obvious place -- the harness
    just matches no signature -- so the constraint is asserted here instead.
    Derived from the registered format signatures on
    tokenspeed-amd:nightly-20260714.
    """
    sidecar = json.loads((_RECIPES / "tokenspeed-kernel-sidecar.json").read_text())
    assert sidecar["version"] == 1
    assert set(sidecar) <= {
        "version",
        "mitigations",
        "environments",
    }, "the sidecar schema rejects unknown top-level keys, comments included"

    fp8_only = {"triton_mm_fp8_blockscale", "torch_mm_fp8_blockscale"}
    non_fp8 = {"gluon_mm_a16w16_gfx950", "torch_mm"}

    for name, env in sidecar["mitigations"].items():
        assert "TS_KERNEL_DTYPE" in env, f"{name} pins no dtype"
        assert "TS_KERNEL_DTYPE_ROLE" in env, f"{name} pins no dtype role"
        # gemm.mm's dtype-selectable roles are its two operands.
        assert env["TS_KERNEL_DTYPE_ROLE"] in {"a", "b"}, name
        assert ("TS_KERNEL_NAME" in env) or ("TS_KERNEL_OP" in env), name

        kernel = env.get("TS_KERNEL_NAME")
        if kernel in fp8_only:
            assert env["TS_KERNEL_DTYPE"] == "fp8", f"{name}: {kernel} is fp8-only"
        elif kernel in non_fp8:
            assert env["TS_KERNEL_DTYPE"] != "fp8", f"{name}: {kernel} has no fp8 signature"


def test_recipes_reference_only_resolvable_mitigations() -> None:
    """Every axis entry must resolve through the real registry.

    Resolved the way the CLI does it -- ``load_mitigations`` over the built-ins
    plus every sidecar in the recipe dir -- rather than against a hand-copied
    list of names. That way renaming a built-in, or dropping an entry from a
    sidecar, fails here instead of part-way into a matrix run with
    ``unknown mitigation 'ts_gemm_gluon_bf16'``.
    """
    from aorta.registry.mitigations import load_mitigations

    sidecars = _sidecars()
    assert sidecars, f"no sidecars found in {_RECIPES}"
    resolvable = set(load_mitigations(extra_files=sidecars))

    for recipe in _recipes():
        doc = yaml.safe_load(recipe.read_text())
        for axis in ("mitigation_axis", "diagnostic_axis"):
            for entry in doc[axis]:
                assert entry in resolvable, (
                    f"{recipe.name}: {axis} entry '{entry}' does not resolve "
                    "from the built-ins or any sidecar in recipes/tokenspeed/"
                )


def test_suites_smoke_covers_every_operator_family_it_claims() -> None:
    """The suite recipe advertises attention, MoE, quantization, sampling and
    transform coverage, so running the file has to actually reach all of them.

    GDN and transform were previously reachable only by hand-picking a sidecar
    selector, which made the recipe's own claim unreproducible from the recipe.
    """
    doc = yaml.safe_load((_RECIPES / "tokenspeed-kernel-suites-smoke.yaml").read_text())
    axis = set(doc["mitigation_axis"])
    for required in (
        "ts_suite_attention",
        "ts_suite_attention_mla",
        "ts_suite_attention_dsa",
        "ts_suite_attention_gdn",
        "ts_suite_quantization",
        "ts_suite_transform",
        "ts_suite_moe_gluon_bf16",
        "ts_suite_sampling_gluon",
    ):
        assert required in axis, f"suites smoke does not run {required}"


def test_stage_scripts_mirrors_rather_than_accumulates(bash: str, tmp_path: Path) -> None:
    """Staging must remove what the source no longer has.

    A plain `cp` only ever adds, so a probe renamed or deleted upstream stays
    behind -- and because recipes name their entry script by filename, the stale
    copy is still executable. A run pointed at the old name would then succeed
    against code that no longer exists in the tree.
    """
    dest = tmp_path / "staged"
    dest.mkdir()
    stale_sh = dest / "ts_removed_probe.sh"
    stale_sh.write_text("#!/usr/bin/env bash\nexit 0\n")
    stale_py = dest / "removed_helper.py"
    stale_py.write_text("x = 1\n")
    # Caller-owned files in the same directory must survive: the staging dir is
    # also where an env file or an out dir can sit.
    keep = dest / "cell.env"
    keep.write_text("FOO=bar\n")

    proc = subprocess.run(
        [bash, str(_SOURCE / "stage_scripts.sh"), str(dest)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not stale_sh.exists(), "renamed probe survived staging"
    assert not stale_py.exists(), "removed helper survived staging"
    assert keep.read_text() == "FOO=bar\n", "staging clobbered a caller file"
    # And the real set did land.
    assert (dest / "host_launch.sh").exists()
    assert (dest / "ts_kernel_probe.sh").exists()


@pytest.mark.parametrize("spelling", ["plain", "trailing_slash", "relative", "symlink"])
def test_stage_scripts_refuses_to_stage_into_its_own_source(
    bash: str, tmp_path: Path, spelling: str
) -> None:
    """`dest == src` would delete the scripts it is supposed to copy.

    The mirror step clears the set this script owns before copying, so pointing
    it at the source tree wipes every probe `.sh` and `.py` and then fails with
    nothing left to copy -- taking the working tree with it. `dest` is a
    positional argument and "stage the scripts where the scripts are" is an easy
    thing to type, so this has to be rejected rather than documented.

    Parametrised over the spellings that resolve to the same directory: checking
    the string as given would leave three ways around the guard.

    Run against a *copy* of the script set, never the real tree. The script
    derives its source directory from ``BASH_SOURCE``, so copying it into
    ``tmp_path`` makes that the source, and a regression in the guard then
    deletes throwaway files instead of the checkout. That is not a hypothetical
    precaution: confirming this test fails without the fix deleted every probe
    script in the working tree, which is exactly the damage being guarded
    against.
    """
    src = tmp_path / "pkg"
    src.mkdir()
    for path in list(_SOURCE.glob("*.sh")) + list(_SOURCE.glob("*.py")):
        shutil.copy2(path, src / path.name)
    before = sorted(p.name for p in src.iterdir())
    assert "host_launch.sh" in before, "precondition: the source set is present"

    if spelling == "plain":
        dest = str(src)
    elif spelling == "trailing_slash":
        dest = f"{src}/"
    elif spelling == "relative":
        dest = f"{src}/../{src.name}"
    else:
        link = tmp_path / "link-to-src"
        link.symlink_to(src)
        dest = str(link)

    proc = subprocess.run(
        [bash, str(src / "stage_scripts.sh"), dest], capture_output=True, text=True
    )

    assert proc.returncode == 64, proc.stdout + proc.stderr
    assert "source directory" in proc.stderr
    assert (
        sorted(p.name for p in src.iterdir()) == before
    ), "stage_scripts deleted its own source files"


def test_the_documented_test_count_matches_this_file() -> None:
    """The count in docs/tokenspeed.md went stale twice during review.

    A number nobody can verify while reading is worse than no number, so it is
    checked here rather than maintained by hand. Counts function definitions
    rather than collected items, because collection from inside a running session
    would recurse -- and the function count is the half that drifts when someone
    adds a test.
    """
    doc = (Path(__file__).resolve().parents[2] / "docs" / "tokenspeed.md").read_text()
    source = Path(__file__).read_text()

    functions = len(re.findall(r"^def test_", source, re.MULTILINE))
    match = re.search(r"—\s*(\d+)\s*tests\s*\((\d+)\s*functions", doc)
    assert match, "docs/tokenspeed.md no longer states the test count in the expected form"
    documented_total, documented_functions = int(match.group(1)), int(match.group(2))

    assert (
        documented_functions == functions
    ), f"docs say {documented_functions} test functions, this file has {functions}"
    assert (
        documented_total >= functions
    ), f"documented total {documented_total} is below the {functions} functions in this file"
