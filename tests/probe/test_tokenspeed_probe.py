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

import argparse
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
from unittest import mock

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
_EXIT_PYTEST_REPORT_UNUSABLE = 42


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


def _mounts_file(tmp_path: Path, entries: list[tuple[str, str]]) -> Path:
    """A stand-in /proc/mounts: (mount point, fstype) pairs."""
    path = tmp_path / "mounts"
    lines = ["/dev/root / ext4 rw,relatime 0 0"]
    lines += [f"server:/export {point} {fstype} rw,relatime 0 0" for point, fstype in entries]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("fstype", ["nfs", "nfs4", "lustre", "cifs", "gpfs"])
def test_host_launch_refuses_a_network_mount_by_fstype(
    bash: str, tmp_path: Path, fstype: str
) -> None:
    """Rejected before docker is ever invoked, and decided by filesystem type.

    The daemon runs as root against a root-squashed export, so the bind mount
    fails with an opaque "mkdir /...: permission denied" from docker itself.
    Refusing early keeps the error actionable.

    By fstype rather than by path spelling: `/home/*|/nfs/*` passed `/mnt`,
    `/shared`, `/users` and any autofs path -- which is where a cluster usually
    puts its network storage -- while rejecting a perfectly local `/home`.
    """
    scripts = tmp_path / "net" / "scripts"
    scripts.mkdir(parents=True)
    env = dict(os.environ)
    env.update(
        {
            "TS_IMAGE": "example/image:tag",
            "TS_SCRIPTS_DIR": str(scripts),
            "TS_HF_DIR": str(tmp_path / "hf"),
            "TS_OUT_DIR": str(tmp_path / "out"),
            "TS_MOUNTS_FILE": str(_mounts_file(tmp_path, [(str(tmp_path / "net"), fstype)])),
        }
    )
    proc = subprocess.run(
        [bash, str(_SOURCE / "host_launch.sh")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == _EXIT_USAGE, proc.stdout + proc.stderr
    assert fstype in proc.stderr
    assert "root-squashed" in proc.stderr


@pytest.mark.parametrize("shape", ["missing_parents", "symlinked_leaf", "dotdot"])
def test_host_launch_resolves_a_path_before_matching_the_mount(
    bash: str, tmp_path: Path, shape: str
) -> None:
    """Two ways a network path was still classified local.

    Resolving only the immediate parent gave up when it did not exist, so
    `/mnt/nfs/new/deep` was called local and `mkdir -p` then created the whole
    thing on the filesystem this refuses -- the guard bypassed by the directory
    not being there yet, which is the normal case on a fresh node. And a final
    component that is a symlink was matched by its own path rather than by what
    it points at.
    """
    netroot = tmp_path / "net"
    netroot.mkdir()
    if shape == "missing_parents":
        scripts = netroot / "new" / "deep" / "scripts"
    elif shape == "symlinked_leaf":
        (tmp_path / "link").symlink_to(netroot)
        scripts = tmp_path / "link" / "scripts"
    else:
        # `..` traverses out of a local path and into the network mount. Left
        # unnormalised in the reattached suffix it matched the local prefix,
        # while mkdir -p and docker both resolved it onto the network one.
        local = tmp_path / "local"
        local.mkdir()
        scripts = local / "missing" / ".." / ".." / "net" / "scripts"

    env = dict(os.environ)
    env.update(
        {
            "TS_IMAGE": "example/image:tag",
            "TS_SCRIPTS_DIR": str(scripts),
            "TS_HF_DIR": str(tmp_path / "hf"),
            "TS_OUT_DIR": str(tmp_path / "out"),
            "TS_MOUNTS_FILE": str(_mounts_file(tmp_path, [(str(netroot), "nfs")])),
        }
    )
    proc = subprocess.run(
        [bash, str(_SOURCE / "host_launch.sh")], capture_output=True, text=True, env=env
    )

    assert proc.returncode == _EXIT_USAGE, proc.stdout + proc.stderr
    assert "nfs" in proc.stderr
    assert not scripts.exists(), "the guard let mkdir -p create it on the network mount"


def test_host_launch_accepts_a_local_path_whatever_it_is_called(
    bash: str, tmp_path: Path
) -> None:
    """The other half of the same fix.

    The prefix test refused any `/home` path outright, which made an ordinary
    workstation unable to run this from the obvious place. What matters is the
    filesystem, so an ext4 mount is accepted however the path is spelled.
    """
    argv = _run_host_launch_with_docker_stub(
        bash,
        tmp_path,
        "local-home",
        extra_env={"TS_MOUNTS_FILE": str(_mounts_file(tmp_path, []))},
    )
    # The helper asserts a zero exit, so reaching here means the guard let it
    # through; the recorded argv confirms docker was actually invoked.
    assert "--rm" in argv.splitlines()


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
    calls = tmp_path / f"calls.{tag}"
    stub = bin_dir / "docker"
    # Only `run` goes to the argv record. host_launch.sh now force-removes the
    # container from an EXIT trap, so an unfiltered stub would overwrite the
    # launch argv with the teardown's. Every subcommand is appended to `calls`
    # so the teardown itself can still be asserted on.
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{calls}"\n'
        'if [ "$1" = "image" ]; then exit 0; fi\n'
        f'if [ "$1" = "run" ]; then printf "%s\\n" "$@" > "{record}"; fi\n'
        "exit 0\n"
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


def test_host_launch_removes_the_container_even_without_a_signal(
    bash: str, tmp_path: Path
) -> None:
    """`--rm` only fires once the container exits, so it is not enough.

    The signal traps cover being killed mid-trial, but a docker *client* that
    dies on its own -- an API disconnect, a daemon restart, an OOM-killed
    client -- returns normally while the daemon keeps running the container,
    stranding a GPU for every later cell. An EXIT trap is what closes that gap;
    on the ordinary path it costs one no-op `docker rm`.
    """
    argv = _run_host_launch_with_docker_stub(bash, tmp_path, "exit-cleanup")
    calls = (tmp_path / "calls.exit-cleanup").read_text().splitlines()

    container = next(
        argv.splitlines()[index + 1]
        for index, line in enumerate(argv.splitlines())
        if line == "--name"
    )
    assert any(
        call.startswith("rm -f ") and container in call for call in calls
    ), f"no teardown for {container}; docker was called with:\n{calls}"


def test_host_launch_container_name_is_unique_per_launcher(bash: str, tmp_path: Path) -> None:
    """The container name must not be derivable from TS_RUN_TOKEN alone.

    A caller may reuse a token deliberately, to correlate artifacts across
    trials. But the name is what the EXIT trap falls back to when the daemon
    never wrote a cidfile -- and the daemon not writing one is exactly what
    happens when the name is already taken. A shared name would have this
    trial's cleanup force-remove the other trial's container.
    """
    token = "cell7-trial2"
    first = _run_host_launch_with_docker_stub(
        bash, tmp_path, "name-a", extra_env={"TS_RUN_TOKEN": token}
    )
    second = _run_host_launch_with_docker_stub(
        bash, tmp_path, "name-b", extra_env={"TS_RUN_TOKEN": token}
    )

    def name_of(argv: str) -> str:
        lines = argv.splitlines()
        return lines[lines.index("--name") + 1]

    assert name_of(first) != name_of(second), "two launchers shared a container name"
    # The token still leads the name, so `docker ps` ties back to the artifacts.
    assert name_of(first).startswith(f"aorta-ts-{token}")
    # The artifact token leads with the caller's value for the same reason, and
    # is likewise unique per launcher -- see
    # test_host_launch_respects_a_caller_supplied_token.
    assert _token_from(first).startswith(f"{token}-")
    assert _token_from(first) != _token_from(second)


def test_host_launch_does_not_share_the_host_ipc_namespace(bash: str, tmp_path: Path) -> None:
    """Processes in one container already share a private IPC namespace.

    `--shm-size=16g` is what tokenspeed's scheduler and detokenizer actually
    need; `--ipc=host` additionally exposed node-wide shared memory and
    semaphores to a third-party image, which buys nothing.
    """
    argv = _run_host_launch_with_docker_stub(bash, tmp_path, "ipc").splitlines()
    assert "--shm-size=16g" in argv
    assert not [line for line in argv if line.startswith("--ipc")], argv


def test_host_launch_respects_a_caller_supplied_token(bash: str, tmp_path: Path) -> None:
    """An explicit TS_RUN_TOKEN prefixes the token, so a caller can correlate
    artifacts -- but it cannot make two trials share a name.

    Taken verbatim, a token set once for a whole sweep (the natural way to label
    a run) gave every trial the same export and log filenames, so they
    overwrote each other and concurrent trials could read each other's verdict
    files. That is the collision the token exists to prevent, arriving through
    the knob meant to label it. Correlation only needs the prefix to be
    searchable, so it stays a prefix and the unique part is always appended.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / "argv.explicit"
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "image" ]; then exit 0; fi\n'
        f'if [ "$1" = "run" ]; then printf "%s\\n" "$@" > "{record}"; fi\n'
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
            "TS_RUN_TOKEN": "cell7-trial2",
        }
    )
    proc = subprocess.run(
        [bash, str(scripts / "host_launch.sh")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    first = _token_from(record.read_text())
    assert first.startswith("cell7-trial2-"), first
    assert first != "cell7-trial2", "the caller value must not be the whole token"

    # And the same caller token twice does not collide, which is the property
    # taking it verbatim gave away.
    proc = subprocess.run(
        [bash, str(scripts / "host_launch.sh")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    second = _token_from(record.read_text())
    assert second.startswith("cell7-trial2-"), second
    assert second != first, "two trials sharing a caller token got the same filenames"


def test_host_launch_sanitizes_a_caller_token_into_a_filename(
    bash: str, tmp_path: Path
) -> None:
    """The token is a filename component inside the container.

    A branch-shaped label like `feature/foo` is an ordinary thing to correlate a
    run by, and it put a path separator in the middle of every export and log
    path -- so the redirection failed on a directory that does not exist, which
    reads as a broken script rather than as a label with a slash in it. Reduced
    to the same filename-safe alphabet the container name already used.
    """
    argv = _run_host_launch_with_docker_stub(
        bash, tmp_path, "slashy", extra_env={"TS_RUN_TOKEN": "feature/foo:v2"}
    )
    token = _token_from(argv)

    assert "/" not in token and ":" not in token, token
    assert token.startswith("feature-foo-v2-"), token


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


def test_pytest_probe_extra_args_cannot_redirect_the_report(bash: str, tmp_path: Path) -> None:
    """`TS_PYTEST_ARGS` used to come after the probe's own `--junit-xml`.

    argparse takes the last occurrence, so a caller-supplied `--junitxml` sent
    the report elsewhere and the probe read whatever was already at its own
    path. With a reused `TS_RUN_TOKEN` that is a previous run's XML, and the
    trial returned that run's verdict -- green, for a suite that just failed.
    Driven exactly that way: a stale passing report is planted at the path this
    token resolves to, and the suite underneath it fails.
    """
    workspace = _fake_workspace(tmp_path, "def test_bad():\n    assert False\n")
    out = tmp_path / "out"
    out.mkdir()
    stale = out / "pytest.reused-token.xml"
    stale.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
        'errors="0" failures="0" skipped="0" tests="7" time="1.0"/></testsuites>'
    )

    proc = _run_pytest_probe(
        bash,
        workspace,
        out,
        TS_PYTEST_SUITE="pkg/test/ops/test_stub.py",
        TS_RUN_TOKEN="reused-token",
        TS_PYTEST_ARGS=f"--junitxml={tmp_path / 'elsewhere.xml'}",
    )

    assert proc.returncode == _EXIT_PYTEST_FAILED, proc.stdout + proc.stderr
    assert "TS_PYTEST_FAIL: tests_failed" in proc.stdout
    assert "tests_passed=7" not in proc.stdout, "the stale report was read"


def test_pytest_probe_counts_are_converted_inside_the_parse_boundary(tmp_path: Path) -> None:
    """A report can be well-formed XML and still unusable.

    `tests="bad"` parses and then raises on `int()`. That conversion sat outside
    the try, so the traceback left `counts` empty -- the script has no `set -e`
    by design -- the PARSE_ERROR case did not match, and the empty strings read
    as a suite that ran nothing: reported as `nothing_executed` (exit 41, a
    selection problem) rather than an unparseable report (exit 42).
    """
    body = (_SOURCE / "ts_pytest_probe.sh").read_text()
    snippet = body.split('counts=$(python3 - "${REPORT}" <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]

    report = tmp_path / "pytest.xml"
    report.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
        'errors="0" failures="0" skipped="0" tests="bad" time="1.0"/></testsuites>'
    )

    proc = subprocess.run(
        [sys.executable, "-c", snippet, str(report)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("PARSE_ERROR"), proc.stdout


def test_pytest_probe_treats_missing_counts_as_an_unparseable_report(
    bash: str, tmp_path: Path
) -> None:
    """And nothing at all is the same verdict, reached a different way.

    An interpreter that dies before printing -- or is killed -- leaves no counts,
    which the PARSE_ERROR branch cannot report. Letting the empty strings through
    to the arithmetic reads as a suite that ran nothing.
    """
    workspace = _fake_workspace(tmp_path, "def test_ok():\n    assert True\n")
    out = tmp_path / "out"
    shim_dir = out / "shim"
    shim_dir.mkdir(parents=True)
    shim = shim_dir / "python3"
    # Real interpreter for `python3 -m pytest`; dies silently for the counts
    # pass, which is the one invoked with a script on stdin.
    shim.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-" ]; then exit 1; fi\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    shim.chmod(0o755)

    proc = subprocess.run(
        [bash, str(_SOURCE / "ts_pytest_probe.sh")],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{shim_dir}:{os.environ['PATH']}",
            "TS_WORKSPACE": str(workspace),
            "TS_OUT_DIR": str(out),
            "TS_PYTEST_SUITE": "pkg/test/ops/test_stub.py",
        },
    )

    assert proc.returncode == _EXIT_PYTEST_REPORT_UNUSABLE, proc.stdout + proc.stderr
    assert "TS_PYTEST_FAIL: report_unparseable" in proc.stdout, proc.stdout


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
@pytest.mark.parametrize("value", ["0", "abc", "-1", "1.5", "08", "9" * 25])
def test_pytest_probe_rejects_an_unusable_min_passed(bash: str, tmp_path: Path, value: str) -> None:
    """TS_MIN_PASSED is what the all-skipped guard compares against, so an
    unusable value disables the guard rather than being merely wrong: 0 passes
    trivially, and a non-numeric makes `[` error out -- which, because pytest
    itself returned 0, also falls through to a green trial.

    `08` and the 25-digit value are all-digits, so they clear the character
    check and reach `[ -lt ]`, which evaluates arithmetically: the first is read
    as an invalid octal literal and aborts with exit 1, the second wraps past
    2^63. Both must be usage errors instead."""
    workspace = _fake_workspace(tmp_path, "def test_ok():\n    assert True\n")
    proc = _run_pytest_probe(
        bash,
        workspace,
        tmp_path / "out",
        TS_PYTEST_SUITE="pkg/test/ops/test_stub.py",
        TS_MIN_PASSED=value,
    )
    assert proc.returncode == _EXIT_USAGE, proc.stdout
    assert "usage TS_MIN_PASSED" in proc.stdout


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("08000", "must not have leading zeros"),
        ("0" * 3 + "8000", "must not have leading zeros"),
        ("9" * 23, "too large to evaluate"),
        ("abc", "must be a positive integer"),
        ("80", "must be between"),
    ],
)
def test_serve_probe_rejects_a_port_bash_cannot_evaluate(
    bash: str, tmp_path: Path, value: str, expected: str
) -> None:
    """All-digits is not the same as safe to hand to `[ -lt ]`.

    The comparison evaluates its operands arithmetically, so `08000` is read as
    an invalid octal literal -- the script died with exit 1 and `CONTROL_PORT`
    unbound, which reads as a broken probe rather than the documented usage exit
    64 -- and a 23-digit value wraps past 2^63 back into an accepted range.
    Both are operator mistakes and must say so.
    """
    proc = subprocess.run(
        [bash, str(_SOURCE / "ts_serve_probe.sh")],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TS_PORT": value,
            "TS_OUT_DIR": str(tmp_path / "out"),
        },
    )
    assert proc.returncode == _EXIT_USAGE, proc.stdout + proc.stderr
    assert expected in proc.stdout, proc.stdout


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


def test_the_network_fstype_list_agrees_across_all_three_copies() -> None:
    """One rule, three copies, and the list is the part that drifts.

    The two shells have to stand alone -- host_launch.sh runs before anything is
    staged and stage_scripts.sh is what stages it -- so the duplication is
    deliberate, but adding `9p` or `fuse.s3fs` to one copy and not the others
    leaves a guard that refuses a mount in the launcher and accepts it in the
    stager. Comments name the other copies; this makes them agree.
    """
    module = _harvest_module()
    copies = {"harvest_code_objects.py": set(module._NETWORK_FSTYPES)}
    for script in ("host_launch.sh", "stage_scripts.sh"):
        text = (_SOURCE / script).read_text()
        match = re.search(r'^_NETWORK_FSTYPES="(.*)"$', text, re.MULTILINE)
        assert match, f"{script} no longer declares _NETWORK_FSTYPES as one literal"
        copies[script] = set(match.group(1).split())
        # The shell match is `case " ${list} " in *" ${fstype} "*)`, so every
        # entry has to be space-delimited: a tab or a stray comma would make
        # that entry unmatchable while still reading as present.
        assert match.group(1) == " " + " ".join(sorted(copies[script])) + " ", script

    assert len(set(map(frozenset, copies.values()))) == 1, {
        name: sorted(values) for name, values in copies.items()
    }


def test_harvest_consan_emits_one_recipe_per_object_not_per_identity(tmp_path: Path) -> None:
    """`--kernel-name` selects nothing when the loader is given the object.

    It resolves the entry from the Triton metadata beside the object, so several
    identities sharing one object produced several recipes that all analyzed the
    same entry -- each attributing that one static result to a different kernel
    name and entry offset, including helper kernels that were never separately
    measured. `--consan-limit` also spent its budget re-analyzing one object
    while skipping others.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)

    kernels = _fake_kernels(2, tmp_path / "cache")
    # Two more identities inside the first object: a helper, and the entry the
    # metadata actually names.
    shared = kernels[0]
    (Path(shared["cache_object"]).with_suffix(".json")).write_text(
        json.dumps({"name": "_fwd_kernel_metadata_entry"})
    )
    kernels.insert(1, {**shared, "name": "_helper_kernel", "entry_offset": 4096})
    kernels.insert(2, {**shared, "name": "_fwd_kernel_metadata_entry", "entry_offset": 8192})

    module._write_consan_assets(tmp_path / "dest", kernels, "gfx950", loader, "lenient", None)

    consan = tmp_path / "dest" / "consan"
    manifest = json.loads((consan / "manifest.json").read_text())

    assert manifest["identities"] == 4
    assert manifest["objects"] == 2
    assert len(sorted(consan.glob("consan-*.yaml"))) == 2
    assert len(manifest["entries"]) == 2

    shared_entry = next(e for e in manifest["entries"] if e["sha256"] == shared["sha256"])
    # The metadata-backed name, not the harvest-order one: it is the only name
    # the loader will actually resolve, so the only one the result belongs to.
    assert shared_entry["kernel"] == "_fwd_kernel_metadata_entry"
    # The others are recorded rather than dropped -- what the object contains is
    # worth knowing; claiming a separate result for each is what was wrong.
    assert shared_entry["identities"] == [
        "_fwd_kernel",
        "_fwd_kernel_metadata_entry",
        "_helper_kernel",
    ]


def test_harvest_will_not_pin_a_consan_result_to_an_unharvested_identity(
    tmp_path: Path,
) -> None:
    """The metadata can name a kernel Waitcheck never listed for the object.

    Keeping the first harvested identity there wrote a recipe naming a kernel and
    an entry offset that are *not* what the loader resolves -- so the one static
    result was reported under a name that was never analyzed, with an exact-entry
    identity implying it had been. The metadata name is used instead, without an
    offset: a whole-object identity is weaker than an exact-entry one, and true.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)

    kernels = _fake_kernels(1, tmp_path / "cache")
    kernels[0]["entry_offset"] = 4096
    Path(kernels[0]["cache_object"]).with_suffix(".json").write_text(
        json.dumps({"name": "_kernel_waitcheck_never_listed"})
    )

    module._write_consan_assets(tmp_path / "dest", kernels, "gfx950", loader, "lenient", None)

    consan = tmp_path / "dest" / "consan"
    entry = json.loads((consan / "manifest.json").read_text())["entries"][0]
    assert entry["kernel"] == "_kernel_waitcheck_never_listed"
    assert entry["identity_source"] == "metadata_unharvested"
    # Both names are recorded: what the object contains is worth knowing even
    # when only one of them can carry the result.
    assert entry["identities"] == ["_fwd_kernel", "_kernel_waitcheck_never_listed"]

    plan = yaml.safe_load(Path(entry["recipe"]).read_text())["sanitizer_plan"]["source"]["kernel"]
    assert plan["name"] == "_kernel_waitcheck_never_listed"
    assert "entry_offset" not in plan, plan


def test_harvest_records_how_each_consan_identity_was_arrived_at(tmp_path: Path) -> None:
    """Three provenances, and they are not equally trustworthy.

    A reader comparing results across objects has to be able to tell a
    metadata-confirmed name from a name that was simply the first one harvested.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)

    kernels = _fake_kernels(2, tmp_path / "cache")
    # Object 0 has a sidecar naming one of its own harvested identities; object 1
    # has none at all.
    Path(kernels[0]["cache_object"]).with_suffix(".json").write_text(
        json.dumps({"name": "_fwd_kernel"})
    )

    module._write_consan_assets(tmp_path / "dest", kernels, "gfx950", loader, "lenient", None)

    manifest = json.loads((tmp_path / "dest" / "consan" / "manifest.json").read_text())
    sources = {entry["sha256"]: entry["identity_source"] for entry in manifest["entries"]}
    assert sources[kernels[0]["sha256"]] == "metadata"
    assert sources[kernels[1]["sha256"]] == "harvest_order"


def test_harvest_reads_the_metadata_sidecar_through_the_containment_guard(
    tmp_path: Path,
) -> None:
    """The sidecar is container-written, so it is untrusted input like the object.

    This read happens before the loop that stages the sidecars, so an unguarded
    `read_text()` here would be the first thing to touch a planted link -- and it
    is unbounded: `k.json -> /dev/zero` never returns, wedging the harvest with
    the GPU still held for the rest of the sweep.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)

    kernels = _fake_kernels(1, tmp_path / "cache")
    sidecar = Path(kernels[0]["cache_object"]).with_suffix(".json")
    outside = tmp_path / "elsewhere.json"
    outside.write_text(json.dumps({"name": "_name_from_outside_the_cache"}))
    sidecar.symlink_to(outside)

    assert module._metadata_kernel_name(kernels[0]) is None

    # The identity read declines quietly, since harvest order is a usable
    # fallback; the staging that follows refuses outright, because there is no
    # fallback for handing the loader a file the container chose. Either way the
    # planted link is never opened.
    with pytest.raises(SystemExit) as excinfo:
        module._write_consan_assets(tmp_path / "dest", kernels, "gfx950", loader, "lenient", None)
    assert "not a regular file inside the Triton cache" in str(excinfo.value)


def test_harvest_consan_limit_budgets_objects_rather_than_repeats(tmp_path: Path) -> None:
    """The limit exists because an attention harvest yields twenty objects.

    Applied to identities, a limit of 1 could be spent entirely on one object
    that four identities shared, analyzing it once and reporting nothing about
    the other nineteen.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)

    kernels = _fake_kernels(2, tmp_path / "cache")
    kernels.insert(1, {**kernels[0], "name": "_helper_kernel", "entry_offset": 4096})

    module._write_consan_assets(tmp_path / "dest", kernels, "gfx950", loader, "lenient", 2)

    manifest = json.loads((tmp_path / "dest" / "consan" / "manifest.json").read_text())
    assert manifest["skipped"] == 0, "the limit was spent on a repeat of one object"
    assert {entry["sha256"] for entry in manifest["entries"]} == {
        kernels[0]["sha256"],
        kernels[-1]["sha256"],
    }


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
    # Digests, never the kernel name: the name is read out of a third-party
    # binary, so putting it in a path let `../` in a name place the shim, the
    # staged object or the recipe outside the harvest directory. It survives as
    # recipe data, which is where it is actually needed. The stem also covers
    # the entry, since kernels sharing an object share the object's digest.
    assert plan["source"]["consan_command"].endswith(f"consan_{module._asset_stem(kernels[0])}")
    assert plan["source"]["consan_command"].startswith(str(tmp_path))
    assert plan["policy"]["consan_policy"] == "lenient"


def test_harvest_keeps_a_hostile_kernel_name_out_of_every_path(tmp_path: Path) -> None:
    """Kernel names come from a third-party image, so they are untrusted input.

    Waitcheck reads them out of the object it was handed. A name containing
    `../` used to be interpolated straight into the staged-object, shim and
    recipe paths, so a crafted image could make the harvester write outside
    `--dest`. Filenames are digest-and-index now; the name is recipe data only.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)

    dest = tmp_path / "dest"
    kernels = _fake_kernels(1, tmp_path / "cache")
    kernels[0]["name"] = "../../../../tmp/pwned"

    module._write_consan_assets(dest, kernels, "gfx950", loader, "lenient", None)

    consan = dest / "consan"
    written = [path for path in consan.rglob("*") if path.is_file()]
    assert written, "the assets were still emitted"
    for path in written:
        assert consan.resolve() in path.resolve().parents, path
    assert not (tmp_path / "tmp").exists(), "nothing escaped dest"

    # The name still reaches the recipe, because that is what ConSan selects on;
    # it is quoted so it cannot reshape the document either.
    recipe = yaml.safe_load(next(consan.glob("consan-*.yaml")).read_text())
    assert recipe["sanitizer_plan"]["source"]["kernel"]["name"] == "../../../../tmp/pwned"


@pytest.mark.parametrize(
    "label",
    [
        "../../../../tmp/pwned",
        "attn\nticket: HIJACKED",
        "attn: fused",
        "..",
    ],
)
def test_harvest_keeps_a_hostile_label_out_of_the_waitcheck_recipe(
    tmp_path: Path, label: str
) -> None:
    """`--kernel` / `--op` is pasted from an inventory of third-party names.

    The waitcheck recipe used it raw in two places with different escaping
    rules: as the filename under `--dest`, where `../` escapes the directory,
    and as an unquoted YAML ticket, where a newline or a colon reshapes the
    document into something other than what was harvested.
    """
    module = _harvest_module()
    dest = tmp_path / "dest"
    dest.mkdir()
    kernels = _fake_kernels(1, tmp_path / "cache")

    recipe_path = module._write_recipe(dest, label, "gfx950", kernels)

    assert dest.resolve() in recipe_path.resolve().parents, recipe_path
    assert not (tmp_path / "tmp").exists(), "the recipe escaped dest"

    recipe = yaml.safe_load(recipe_path.read_text())
    assert "HIJACKED" not in recipe, "the label injected a top-level key"
    assert recipe["ticket"].startswith("TOKENSPEED-WAITCHECK-")
    assert re.fullmatch(r"TOKENSPEED-WAITCHECK-[A-Z0-9-]+", recipe["ticket"]), recipe["ticket"]
    # Still a usable recipe, not just a safe one.
    assert recipe["mode"] == "sanitizer"
    assert recipe["sanitizer_plan"]["target"] == "gfx950"


def test_host_launch_puts_the_cidfile_in_a_private_directory(bash: str, tmp_path: Path) -> None:
    """`mktemp -u` only reserves a name in a shared namespace.

    Another local user could create that path as a symlink to a file holding a
    different container's id; docker would refuse to start against an existing
    cidfile, and the EXIT trap would then read the planted id and force-remove
    *that* container. `mktemp -d` is atomic and 0700, so the directory cannot be
    pre-created or its contents substituted.
    """
    argv = _run_host_launch_with_docker_stub(bash, tmp_path, "cid-dir")
    lines = argv.splitlines()
    cidfile = Path(lines[lines.index("--cidfile") + 1])

    assert cidfile.name == "cid", cidfile
    assert cidfile.parent.name.startswith("aorta-ts-cid."), cidfile

    script = (_SOURCE / "host_launch.sh").read_text()
    # Comments discuss `mktemp -u` by name, so only code lines are checked.
    code = [
        line for line in script.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    assert any("mktemp -d" in line for line in code)
    assert not [line for line in code if "mktemp -u" in line], "the racy reservation is still used"
    # The whole directory is removed, not just the file inside it.
    assert any('rm -rf "${CID_DIR}"' in line for line in code)


@pytest.mark.parametrize(
    "rc,pytest_args,expected",
    [
        (0, [], None),
        # Tests that failed still ran, and a kernel entered by a failing test
        # was entered.
        (1, [], None),
        (2, [], "interrupted"),
        (3, [], "internal error"),
        (4, [], "usage error"),
        (5, [], "collected no tests"),
        # rc=1 stops meaning "ran everything" once an early exit is requested.
        (1, ["--maxfail=1"], "early-exit"),
        (1, ["-x"], "early-exit"),
        (1, ["--exitfirst"], "early-exit"),
        (0, ["-x"], None),
        # Stepwise stops at the first failure too; it is spelled as a
        # session-scoped option rather than a failure limit, but the effect on
        # the map is identical.
        (1, ["--stepwise"], "early-exit"),
        (1, ["--sw"], "early-exit"),
        (1, ["--stepwise-skip"], "early-exit"),
        # Clustered short options: `-x` never appears as its own token.
        (1, ["-xq"], "early-exit"),
        (1, ["-qx"], "early-exit"),
        # But `-rx` is `-r x`, a report specifier -- the scan has to stop at the
        # first option that takes a value or this rejects healthy runs.
        (1, ["-rx"], None),
        # No execution at all, and these exit 0, so nothing else would hint that
        # the empty map should be doubted.
        (0, ["--collect-only"], "executes no tests"),
        (0, ["--co"], "executes no tests"),
        (0, ["--setup-only"], "executes no tests"),
    ],
)
@pytest.mark.parametrize("executed", [None, 7])
def test_coverage_mapper_option_checks(
    rc: int, pytest_args: list[str], expected: str | None, executed: int | None
) -> None:
    """The option-based checks, with execution either unknown or observed.

    Passing a positive count must not soften them: `-x` with rc=1 still means
    the suite stopped short, however many tests ran before it did.
    """
    module = _coverage_module()
    reason = module._incomplete_suite_reason(rc, pytest_args, executed)
    if expected is None:
        assert reason is None, reason
    else:
        assert reason is not None and expected in reason, reason


def test_coverage_mapper_rejects_a_suite_that_executed_nothing() -> None:
    """The check that does not depend on a list staying complete.

    A denylist of no-execution modes cannot be exhaustive -- `--collect-only`
    was missing, then `--fixtures`, `--help`, `--version` -- and each exits 0
    with an empty map, which merges as "no kernel is covered". Counting
    call-phase reports establishes execution directly.
    """
    module = _coverage_module()
    reason = module._incomplete_suite_reason(0, ["--fixtures"], 0)
    assert reason is not None and "executed no test bodies" in reason

    # And a suite that did run is unaffected, whatever exotic option was passed.
    assert module._incomplete_suite_reason(0, ["--fixtures"], 12) is None


@pytest.mark.parametrize(
    "rc,pytest_args,expected",
    [
        (0, [], None),
        (1, [], None),
        (2, [], "interrupted"),
        (1, ["-x"], "early-exit"),
    ],
)
def test_coverage_mapper_rejects_a_suite_that_did_not_finish(
    rc: int, pytest_args: list[str], expected: str | None
) -> None:
    """The child pytest exit code was recorded and never validated.

    A collection error, an internal error or an early exit still wrote a partial
    map, which was merged as though the suite had run -- so every test that
    never ran became an uncovered kernel and the tool exited 0. That is the same
    unknown-as-uncovered answer the missing-map guard prevents, reached from a
    suite that did start, and it feeds a number quoted in docs/tokenspeed.md.
    """
    module = _coverage_module()
    reason = module._incomplete_suite_reason(rc, pytest_args)

    if expected is None:
        assert reason is None, reason
    else:
        assert reason is not None and expected in reason, reason


def test_kernel_probe_extra_args_cannot_disable_verification() -> None:
    """`TS_KERNEL_ARGS` used to come after the probe's own flags.

    argparse takes the last occurrence, so `--no-verify` won. The export then
    carried `numerics_passed: null`, which the summary deliberately tolerates
    because some kernels have no numerics check, and `bench` mode exited 0
    having verified nothing.
    """
    script = (_SOURCE / "ts_kernel_probe.sh").read_text()

    for owned in ("--verify", "--export"):
        # The caller's args must appear before every option the verdict depends
        # on, in each invocation that uses them.
        for invocation in re.findall(r"python3 -m tokenspeed_kernel\.\w+ \\\n(?:.*\\\n)*.*", script):
            if owned not in invocation:
                continue
            assert invocation.index("TS_KERNEL_ARGS") < invocation.index(owned), invocation

    # And the selector/dtype, which the probe reports the run as having used.
    for invocation in re.findall(r"python3 -m tokenspeed_kernel\.\w+ \\\n(?:.*\\\n)*.*", script):
        assert "TS_KERNEL_ARGS" in invocation, invocation
        assert invocation.index("TS_KERNEL_ARGS") < invocation.index("--dtype"), invocation


def test_harvest_bounds_the_waitcheck_inventory(tmp_path: Path) -> None:
    """The object being parsed came out of a third-party image.

    Every other subprocess in the harvester is bounded. This one wedging would
    hang the whole harvest with the GPU still held for the rest of the sweep --
    the failure the docker timeout exists to prevent, one layer further in.
    """
    module = _harvest_module()
    waitcheck = tmp_path / "waitcheck"
    waitcheck.write_text("#!/bin/sh\nsleep 60\n")
    waitcheck.chmod(0o755)
    obj = tmp_path / "kernel.hsaco"
    obj.write_bytes(b"object")

    with mock.patch.object(module, "_SUBPROCESS_TIMEOUT_SEC", 1):
        with pytest.raises(SystemExit) as excinfo:
            module._inventory(waitcheck, obj)

    message = str(excinfo.value)
    assert "did not finish within" in message, message
    # Says what to do about it, since the object is not the operator's to fix.
    assert "--waitcheck" in message, message


@pytest.mark.parametrize("suffix", [".json", ".amdgcn"])
def test_harvest_refuses_a_symlinked_consan_sidecar(tmp_path: Path, suffix: str) -> None:
    """The object was filtered when the cache was walked; the loader reaches
    further.

    It resolves and copies the `.json` and `.amdgcn` beside the object with APIs
    that follow symlinks, and the container writes that directory -- so
    `k.json -> /etc/something` would have the host read and copy a file of the
    container's choosing, as the calling user.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)
    kernels = _fake_kernels(1, tmp_path / "cache")
    victim = tmp_path / "outside.txt"
    victim.write_text("not yours\n")
    Path(kernels[0]["cache_object"]).with_suffix(suffix).symlink_to(victim)

    with pytest.raises(SystemExit) as excinfo:
        module._write_consan_assets(tmp_path / "dest", kernels, "gfx950", loader, "lenient", None)

    message = str(excinfo.value)
    assert "sidecar" in message, message
    assert "written by the container" in message, message


def test_harvest_bounds_the_consan_loader(tmp_path: Path) -> None:
    """Same gap on the ConSan shim, which reads a Triton cache entry the image
    wrote."""
    module = _harvest_module()
    loader = tmp_path / "slow_loader.py"
    loader.write_text("import time\ntime.sleep(60)\n")
    kernels = _fake_kernels(1, tmp_path / "cache")
    dest = tmp_path / "dest"

    with mock.patch.object(module, "_SUBPROCESS_TIMEOUT_SEC", 1):
        with pytest.raises(SystemExit) as excinfo:
            module._write_consan_assets(dest, kernels, "gfx950", loader, "lenient", None)

    assert "did not finish within" in str(excinfo.value), excinfo.value


@pytest.mark.parametrize(
    "fstype,expected",
    [
        ("nfs4", "nfs4"),
        ("nfs", "nfs"),
        ("lustre", "lustre"),
        ("ext4", None),
        ("xfs", None),
        ("overlay", None),
    ],
)
def test_harvest_detects_a_network_dest_by_fstype_not_by_path(
    tmp_path: Path, fstype: str, expected: str | None
) -> None:
    """The old guard matched `/home/` and `/nfs/` prefixes.

    That passes `/mnt`, `/shared`, `/users` and any autofs path, which then
    fail later with docker's opaque bind-mount error -- the confusion the guard
    exists to prevent -- and it rejects a perfectly local `/home` on a node
    that does not export one.
    """
    module = _harvest_module()
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "/dev/sda1 / ext4 rw 0 0\n"
        "tmpfs /tmp tmpfs rw 0 0\n"
        f"server:/export /mnt/shared {fstype} rw 0 0\n"
    )

    assert module._network_filesystem(Path("/mnt/shared/run"), mounts) == expected
    # The longest matching mount wins, so an unrelated shorter one cannot mask it.
    assert module._network_filesystem(Path("/tmp/ts-work"), mounts) is None


def test_harvest_resolves_an_absolute_dest_symlink(tmp_path: Path) -> None:
    """An absolute path can still be a link.

    Absolute paths were used as given, so `--dest /tmp/run` pointing at
    `/mnt/shared/run` was matched against `/tmp` and called local -- the guard
    skipped by the one spelling most likely to be typed on the command line.
    """
    module = _harvest_module()
    mounts = tmp_path / "mounts"
    real = tmp_path / "shared"
    real.mkdir()
    link = tmp_path / "run"
    link.symlink_to(real)
    mounts.write_text(f"/dev/sda1 / ext4 rw 0 0\nserver:/export {real} nfs rw 0 0\n")

    assert module._network_filesystem(link, mounts) == "nfs"
    # And a path under it that does not exist yet keeps the same answer, since
    # resolve() is non-strict and preserves the missing suffix.
    assert module._network_filesystem(link / "new" / "deep", mounts) == "nfs"


def test_harvest_treats_an_unreadable_mount_table_as_local(tmp_path: Path) -> None:
    """Best-effort by construction: blocking a harvest on a guess is worse than
    letting docker report its own error."""
    module = _harvest_module()
    assert module._network_filesystem(Path("/mnt/shared"), tmp_path / "absent") is None


def test_harvest_clears_staged_objects_on_reharvest(tmp_path: Path) -> None:
    """`triton-cache/` and `consan/` are cleared; `code_objects/` was not.

    Milder than the ConSan case -- staged names are content-addressed, so a
    leftover is never referenced by this run's recipe -- but these are the
    largest files the harvest writes, and a re-harvest with a narrower
    selection kept every object the wider one staged.
    """
    module = _harvest_module()
    objects_dir = tmp_path / "dest" / "code_objects"
    objects_dir.mkdir(parents=True)
    stale = objects_dir / "_old_kernel.deadbeef1234.hsaco"
    stale.write_bytes(b"x" * 1024)

    module._reset_dir(objects_dir)

    assert not stale.exists(), "a staged object from the previous harvest survived"
    assert objects_dir.is_dir() and not list(objects_dir.iterdir())

    # Also creates the directory when it was never there, which is the
    # first-harvest path.
    fresh = tmp_path / "fresh" / "code_objects"
    module._reset_dir(fresh)
    assert fresh.is_dir()


def test_harvest_keeps_kernels_sharing_one_code_object_apart(tmp_path: Path) -> None:
    """Digest and index do not identify a kernel; entry does.

    Several kernels commonly sit in one code object at one index and differ only
    by entry offset and name -- the case `entry_offset` is carried for in the
    first place. They shared a filename stem, so each one's shim, staged object
    and recipe overwrote the previous one's silently. What fans out into recipes
    is now the object rather than the identity, so the collision is no longer
    reachable through `--consan`, but the stem is what keeps the staged object
    and shim of one identity from landing on another's path, and every other
    caller of it still hands it whole harvests.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)
    dest = tmp_path / "dest"
    dest.mkdir()

    shared = _fake_kernels(1, tmp_path / "cache")[0]
    kernels = [
        {**shared, "name": "_fwd_kernel", "entry_offset": 0},
        {**shared, "name": "_helper_kernel", "entry_offset": 4096},
        # Same object and index again, and the same name: only the entry differs.
        {**shared, "name": "_fwd_kernel", "entry_offset": 8192},
    ]
    stems = {module._asset_stem(kernel) for kernel in kernels}
    assert len(stems) == 3, stems
    # The names must stay out of the path even though they take part in the
    # identity: the stem carries a digest of them, not the strings.
    assert not [stem for stem in stems if "kernel" in stem], stems

    module._write_consan_assets(dest, kernels, "gfx950", loader, "lenient", None)

    # One object in, one recipe out, and it describes the identity the manifest
    # names -- not the last of three writes over one path.
    recipes = sorted((dest / "consan").glob("consan-*.yaml"))
    assert len(recipes) == 1, [p.name for p in recipes]
    plan = yaml.safe_load(recipes[0].read_text())["sanitizer_plan"]["source"]["kernel"]
    entry = json.loads((dest / "consan" / "manifest.json").read_text())["entries"][0]
    assert plan["name"] == entry["kernel"]
    assert Path(entry["recipe"]) == recipes[0]
    assert plan["entry_offset"] == 0


@pytest.mark.parametrize(
    "target",
    ["gfx950", "gfx950\nticket: HIJACKED", "gfx950: sub", "gfx950 #comment"],
)
def test_harvest_quotes_the_target_in_every_recipe(tmp_path: Path, target: str) -> None:
    """`target` comes from the Waitcheck JSON, so it is object-derived too.

    The kernel names were quoted and this was not, even though both arrive in
    the same parsed record -- a target carrying a newline or a colon reshapes
    the generated recipe into something other than what was harvested.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)
    dest = tmp_path / "dest"
    dest.mkdir()
    kernels = _fake_kernels(1, tmp_path / "cache")

    waitcheck = module._write_recipe(dest, "attn", target, kernels)
    recipe = yaml.safe_load(waitcheck.read_text())
    assert recipe["sanitizer_plan"]["target"] == target
    assert "HIJACKED" not in recipe

    module._write_consan_assets(dest, kernels, target, loader, "lenient", None)
    consan = yaml.safe_load(next((dest / "consan").glob("consan-*.yaml")).read_text())
    assert consan["sanitizer_plan"]["target"] == target
    assert "HIJACKED" not in consan


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "not-a-digest"),
        ("sha256", "deadbeef"),  # right alphabet, wrong length
        ("sha256", "z" * 64),  # right length, wrong alphabet
        ("sha256", "0\nticket: HIJACKED"),
        ("code_object_index", "0: sub"),
        ("code_object_index", "not-an-int"),
    ],
)
def test_harvest_rejects_malformed_inventory_numbers(tmp_path: Path, field: str, value: str) -> None:
    """The digest and index fields are emitted unquoted, because the recipe
    schema wants a hex scalar and a number.

    Quoting them would not make a malformed value safe -- it would produce a
    well-formed recipe selecting something that cannot match. So they are
    required to be what they claim, which also means they cannot carry YAML
    syntax.
    """
    module = _harvest_module()
    dest = tmp_path / "dest"
    dest.mkdir()
    kernels = _fake_kernels(1, tmp_path / "cache")
    kernels[0][field] = value

    with pytest.raises(SystemExit) as excinfo:
        module._write_recipe(dest, "attn", "gfx950", kernels)

    assert "malformed" in str(excinfo.value), excinfo.value


def test_harvest_replaces_stale_consan_assets(tmp_path: Path) -> None:
    """Re-harvesting the same --dest must not leave the previous run's recipes.

    The documented way to run these is `for r in consan/consan-*.yaml`, so a
    recipe left behind by a wider earlier harvest is executed against an object
    the current manifest does not list -- reporting on kernels this run never
    harvested.
    """
    module = _harvest_module()
    loader = tmp_path / "triton_consan_loader.py"
    loader.write_text(_FAKE_LOADER)
    dest = tmp_path / "dest"

    wide = _fake_kernels(3, tmp_path / "cache")
    module._write_consan_assets(dest, wide, "gfx950", loader, "lenient", None)
    assert len(sorted((dest / "consan").glob("consan-*.yaml"))) == 3

    # Narrower second pass: one kernel instead of three.
    module._write_consan_assets(dest, wide[:1], "gfx950", loader, "lenient", None)

    recipes = sorted((dest / "consan").glob("consan-*.yaml"))
    manifest = json.loads((dest / "consan" / "manifest.json").read_text())
    assert len(recipes) == 1, "the glob must match the manifest"
    assert len(manifest["entries"]) == 1
    assert len(sorted((dest / "consan" / "bin").glob("consan_*"))) == 1
    assert len(sorted((dest / "consan" / "isa").glob("*.hsaco"))) == 1


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

    # Only an *omitted* binary is a soft path: no binary, no inventory, no error.
    assert module._inventory(None, obj) == []

    # A supplied path that does not exist is a typo, not a request for guessed
    # identities. Returning [] here silently produced the same unverified recipe
    # the caller passed --waitcheck to avoid.
    with pytest.raises(SystemExit) as excinfo:
        module._inventory(tmp_path / "absent", obj)
    assert "does not exist" in str(excinfo.value)

    # Nor is a binary that runs cleanly but describes no kernels: rc=0 with an
    # empty inventory would fall through to the same guess.
    empty = tmp_path / "rj_waitcheck_empty"
    empty.write_text("#!/usr/bin/env bash\nexit 0\n")
    empty.chmod(0o755)
    with pytest.raises(SystemExit) as excinfo:
        module._inventory(empty, obj)
    assert "no kernel records" in str(excinfo.value)


def test_harvest_names_its_container_so_a_timeout_is_recoverable() -> None:
    """`subprocess.run(timeout=...)` kills the docker client, not the container.

    aorta escalates to SIGKILL 10s after SIGTERM while a compile can be given
    minutes, so without a handle a hung harvest keeps a GPU busy for the rest of
    the sweep.
    """
    source = (_SOURCE / "harvest_code_objects.py").read_text()
    assert 'f"aorta-ts-harvest-{os.getpid()}-{secrets.token_hex(8)}"' in source
    assert "--name" in source
    assert "_force_remove_container" in source
    assert "except subprocess.TimeoutExpired" in source


def test_harvest_removes_the_container_when_the_client_exits_nonzero(
    tmp_path: Path, monkeypatch
) -> None:
    """A nonzero client is not evidence the container is gone.

    `--rm` fires when the *container* exits. The client can return nonzero while
    the daemon still has it running -- an API disconnect, or the client being
    killed -- and the GPU is then held for everything that follows. Timeout was
    covered; this path reaches the same state and was not.
    """
    module = _harvest_module()
    removed: list[str] = []
    monkeypatch.setattr(
        module, "_force_remove_container", lambda name, env: removed.append(name)
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 1, "out", "err"),
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    args = argparse.Namespace(
        image="example/image:tag",
        gpus="0",
        network="bridge",
        docker_config=None,
        timeout=60,
        kernel="gluon_mm_a16w16_gfx950",
        op=None,
        dtype="bf16",
        dtype_role="a",
        pytest_suite=None,
        pytest_k=None,
    )
    with pytest.raises(SystemExit):
        module._run_kernel(args, cache_dir)

    assert removed, "a nonzero docker client left the container behind"


def test_harvest_only_removes_a_container_it_could_have_created(
    tmp_path: Path, monkeypatch
) -> None:
    """The name has to be unguessable before removal by name is safe.

    A pid is predictable and gets reused, so a stale harvest container -- or one
    another daemon user named first -- would collide on `docker run`, and the
    removal on the failure path would then delete somebody else's container.
    """
    module = _harvest_module()

    def one_run() -> tuple[str, list[str]]:
        removed: list[str] = []
        launched: list[str] = []
        monkeypatch.setattr(
            module, "_force_remove_container", lambda name, env: removed.append(name)
        )

        def fake_run(cmd, *a, **k):
            launched.append(cmd[cmd.index("--name") + 1])
            return subprocess.CompletedProcess(cmd, 1, "out", "err")

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        cache_dir = tmp_path / f"cache-{len(list(tmp_path.iterdir()))}"
        cache_dir.mkdir()
        args = argparse.Namespace(
            image="example/image:tag",
            gpus="0",
            network="bridge",
            docker_config=None,
            timeout=60,
            kernel="gluon_mm_a16w16_gfx950",
            op=None,
            dtype="bf16",
            dtype_role="a",
            pytest_suite=None,
            pytest_k=None,
        )
        with pytest.raises(SystemExit):
            module._run_kernel(args, cache_dir)
        assert len(launched) == 1
        # What it removes is what it named, or the removal is aimed elsewhere.
        assert removed == launched
        return launched[0], removed

    first, _ = one_run()
    second, _ = one_run()

    pid_prefix = f"aorta-ts-harvest-{os.getpid()}-"
    assert first.startswith(pid_prefix) and second.startswith(pid_prefix)
    assert first != second, "the pid is the whole name, so the name is guessable"
    # Long enough that a collision is not something to reason about.
    assert len(first) - len(pid_prefix) >= 16


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


def test_a_pytest_flag_reaches_the_child_instead_of_failing_its_argparse(
    tmp_path: Path, monkeypatch
) -> None:
    """`--pytest-arg` exists to forward pytest flags, and forwarded none of them.

    Almost every value starts with a dash, and rebuilt as two argv elements
    argparse reads `--pytest-arg -x` as this option followed by a *new* option
    and exits with "expected one argument". So `--pytest-arg=-x` -- the form the
    docs give -- died in the child before any suite ran.
    """
    module = _coverage_module()
    workspace = tmp_path / "ws"
    suite = workspace / "pkg" / "test" / "ops"
    suite.mkdir(parents=True)

    captured: list[list[str]] = []
    # Kept because the stub below replaces `subprocess.run` process-wide, and the
    # second half of this test needs to really spawn the child.
    real_run = subprocess.run

    def fake_run(cmd, **_kwargs):
        captured.append([str(part) for part in cmd])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "map_kernel_test_coverage.py",
            "--workspace",
            str(workspace),
            "--suite",
            "pkg/test/ops",
            "--pytest-arg=-x",
        ],
    )

    # Aborts afterwards because the stubbed child writes no map, which is not
    # what this test is about -- the argv it was given is.
    module.main()

    assert captured, "no child was spawned"
    child = captured[0]
    assert "--pytest-arg=-x" in child, child
    assert "--pytest-arg" not in child, "the split form is what the child rejects"

    # And the child really does reject the split form: proving the attached one
    # is necessary, not merely tidier. Neither invocation reaches a suite (the
    # registry import fails outside the container), so the distinction is the
    # argparse usage error.
    def run_child(*extra: str) -> subprocess.CompletedProcess:
        return real_run(
            [
                "python3",
                str(_SOURCE / "map_kernel_test_coverage.py"),
                "--_single",
                str(suite),
                "--out",
                str(tmp_path / "part.json"),
                "--workspace",
                str(workspace),
                *extra,
            ],
            capture_output=True,
            text=True,
        )

    split = run_child("--pytest-arg", "-x")
    assert "expected one argument" in split.stderr, split.stderr
    attached = run_child("--pytest-arg=-x")
    assert "expected one argument" not in attached.stderr, attached.stderr


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


def test_coverage_map_keeps_two_names_sharing_one_callable_apart(monkeypatch) -> None:
    """The registry maps each name to a callable independently, and two specs
    may register the same one.

    Caching proxies by callable identity alone handed the second name a proxy
    still carrying the first, so running the second kernel credited the first
    and left the second reading as lookup-only -- two wrong rows in the table
    this script exists to produce, from one lookup.
    """
    from collections import defaultdict

    module = _coverage_module()
    entered: dict = defaultdict(set)
    dispatched: dict = defaultdict(set)
    candidates: dict = defaultdict(set)
    requested: dict = {}

    def shared_kernel(*args, **kwargs):
        return "result"

    impls = {"k_first": shared_kernel, "k_second": shared_kernel}
    with _fake_registry(monkeypatch, impls):
        module._install_probe(entered, dispatched, candidates, requested)
        from tokenspeed_kernel.registry import KernelRegistry  # type: ignore

        registry = KernelRegistry()
        # Look the first up without running it, then run the second.
        registry.get_impl("k_first")
        assert registry.get_impl("k_second")(1) == "result"

    assert set(entered) == {"k_second"}, "the wrong kernel was credited with the run"
    assert set(dispatched) == {"k_first", "k_second"}


def test_coverage_map_counts_a_triton_launch(monkeypatch) -> None:
    """Triton entry points are invoked as `kernel[grid](...)`.

    The record belongs on the call, not the subscript: `kernel[grid]` only binds
    the grid and returns a launcher, so recording there would credit a launch
    that never happened. This is also why a `functools.wraps` wrapper will not
    do: it forwards neither `__getitem__` nor `__module__`.
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


def test_coverage_map_does_not_count_a_grid_binding_as_a_launch(monkeypatch) -> None:
    """`kernel[grid]` without a call is a lookup, not a dispatch.

    Triton's `__getitem__` binds the grid and hands back a launcher; nothing has
    run until that launcher is invoked. Recording at the subscript inflated
    `covered` with kernels a suite only prepared to run -- the same
    overstatement as counting candidates instead of dispatches, one layer
    further in -- so this kernel must stay `lookup_only`.
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
        launcher = impl[(4, 1, 1)]
        # The launcher exists and is inspectable, but was never invoked.
        assert callable(launcher)

    assert launched == [], "nothing was dispatched"
    assert not entered, "binding a grid must not count as coverage"
    assert set(dispatched) == {"k_triton"}, "the lookup itself still counts"


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

    Exercised across two staging runs, which is how the situation actually
    arises: what the second run may delete is what the first one recorded
    staging, not whatever happens to match `*.sh` in the destination.
    """
    src = tmp_path / "src"
    shutil.copytree(_SOURCE, src, ignore=shutil.ignore_patterns("__pycache__"))
    renamed_away = src / "ts_removed_probe.sh"
    renamed_away.write_text("#!/usr/bin/env bash\nexit 0\n")

    dest = tmp_path / "staged"
    dest.mkdir()
    # Caller-owned files in the same directory must survive: the staging dir is
    # also where an env file or an out dir can sit.
    keep = dest / "cell.env"
    keep.write_text("FOO=bar\n")

    def stage() -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            [bash, str(src / "stage_scripts.sh"), str(dest)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc

    stage()
    assert (dest / "ts_removed_probe.sh").exists(), "the first run staged the full set"

    # Upstream renames it away; the next staging run must not leave it behind.
    renamed_away.unlink()
    stage()

    assert not (dest / "ts_removed_probe.sh").exists(), "renamed probe survived staging"
    assert keep.read_text() == "FOO=bar\n", "staging clobbered a caller file"
    # And the real set did land.
    assert (dest / "host_launch.sh").exists()
    assert (dest / "ts_kernel_probe.sh").exists()


def test_stage_scripts_only_deletes_what_it_staged(bash: str, tmp_path: Path) -> None:
    """`dest` is a positional argument, so it may be a directory nobody owns.

    The mirror step used to `rm -f "$dest"/*.sh "$dest"/*.py`, which in a shared
    location deleted whatever matched -- `stage_scripts.sh /tmp` removed other
    people's staging scripts before copying. Deletion is now scoped to a
    manifest this script wrote, so a first run into a populated directory
    removes nothing.
    """
    dest = tmp_path / "shared"
    dest.mkdir()
    foreign_sh = dest / "someone_elses_run.sh"
    foreign_sh.write_text("#!/usr/bin/env bash\necho theirs\n")
    foreign_py = dest / "their_helper.py"
    foreign_py.write_text("theirs = True\n")

    for _ in range(2):
        proc = subprocess.run(
            [bash, str(_SOURCE / "stage_scripts.sh"), str(dest)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert foreign_sh.exists(), "staging deleted an unrelated script"
        assert foreign_py.exists(), "staging deleted an unrelated helper"

    # A syntax error in a file this script did not stage is likewise none of its
    # business -- globbing the destination made it fail the whole staging run.
    (dest / "broken_of_theirs.sh").write_text("if then fi\n")
    proc = subprocess.run(
        [bash, str(_SOURCE / "stage_scripts.sh"), str(dest)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_stage_scripts_writes_the_manifest_without_following_a_symlink(
    bash: str, tmp_path: Path
) -> None:
    """`>` follows a symlink, and `dest` may be a shared directory.

    Another user could pre-create `.aorta-staged` pointing at any file the
    caller can write, and the manifest write would then truncate it. The
    manifest goes to a `mktemp` file in the destination and is renamed over the
    path, and a manifest that is already a symlink is refused rather than read
    -- its contents drive a deletion loop.
    """
    dest = tmp_path / "shared"
    dest.mkdir()
    victim = tmp_path / "precious.txt"
    victim.write_text("do not truncate me\n")
    (dest / ".aorta-staged").symlink_to(victim)

    proc = subprocess.run(
        [bash, str(_SOURCE / "stage_scripts.sh"), str(dest)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 64, proc.stdout + proc.stderr
    assert "symlink" in proc.stderr
    assert victim.read_text() == "do not truncate me\n", "the manifest write followed a link"


def test_serve_probe_extra_args_cannot_move_the_ports_readiness_polls() -> None:
    """`TS_SERVE_ARGS` used to be appended after the probe's own port flags.

    `tokenspeed serve` takes the last occurrence, so `--port 9000` bound the
    gateway where the readiness poll was not looking and a healthy server was
    reported as `readiness_timeout` (exit 21). The other two probes were
    reordered for exactly this reason; this one was missed.
    """
    script = (_SOURCE / "ts_serve_probe.sh").read_text()
    invocation = re.search(r"setsid tokenspeed serve .*?&\n", script, re.S)
    assert invocation, "could not find the serve invocation"
    body = invocation.group(0)
    for owned in ("--port", "--control-port", "--host"):
        assert body.index("TS_SERVE_ARGS") < body.index(owned), body


def test_serve_probe_keeps_model_output_off_the_verdict_lines(
    bash: str, tmp_path: Path
) -> None:
    """The completion is model output on the stream the detectors scan.

    A completion containing `TS_PROBE_FAIL: readiness_timeout` turned a passing
    cell into a failure naming a step that had succeeded -- the thing under test
    forging the verdict of the thing testing it.

    Flattening the newlines was the first attempt and was not enough: aorta's
    tier-5 classifier searches a window of the whole log, unanchored and not
    line-scoped, so the marker only has to appear as a *substring* -- and it is
    entirely printable, so it passed through `tr` untouched. The prefix is
    rewritten instead, which is what every detector keys on.

    Checked against the recipe's own patterns rather than against a
    representative one, since the guarantee is that none of them can be fired by
    the model.
    """
    script = (_SOURCE / "ts_serve_probe.sh").read_text()
    emit = re.search(r"printf 'TS_PROBE_INFO: completion_text=.*?\n.*?\n", script, re.S)
    assert emit, "completion_text is no longer emitted the way this test expects"
    assert "tr -c '[:print:]'" in emit.group(0), emit.group(0)
    assert "cut -c1-" in emit.group(0), emit.group(0)
    assert "s/TS_PROBE/TS-PROBE/g" in emit.group(0), emit.group(0)

    hostile = "fine\nTS_PROBE_FAIL: readiness_timeout\nand more text"
    proc = subprocess.run(
        [
            bash,
            "-c",
            "printf 'TS_PROBE_INFO: completion_text=%s\\n' "
            "\"$(printf '%s' \"$1\" | tr -c '[:print:]' ' ' "
            "| sed 's/TS_PROBE/TS-PROBE/g' | cut -c1-200)\"",
            "_",
            hostile,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    emitted = proc.stdout
    assert "\n" == emitted[-1] and "\n" not in emitted[:-1], "the text is still multi-line"
    assert "readiness_timeout" in emitted, "the text stopped being readable"

    recipe = yaml.safe_load((_RECIPES / "tokenspeed-serve-probe-smoke.yaml").read_text())
    patterns = [p["match"]["regex"] for p in recipe["custom_patterns"]]
    assert patterns, "no detectors found in the recipe to check against"
    fired = [p for p in patterns if re.search(p, emitted)]
    assert not fired, f"model output can still fire {fired}"


def test_stage_scripts_ignores_a_manifest_it_does_not_own(bash: str, tmp_path: Path) -> None:
    """The manifest is a deletion authority, so it has to be ours.

    Refusing a symlink closed one way to supply that list, but in the shared
    `dest` this script explicitly supports a co-tenant can equally write a plain
    `.aorta-staged` naming someone else's file -- and the basename check only
    stops traversal, not that. Not owned by this uid means it is not trusted,
    which degrades to deleting nothing: the same safe default as a first run.

    Driven by ownership rather than by planting a file as another user, which a
    test cannot do: the manifest is written with a uid that is not ours.
    """
    dest = tmp_path / "shared"
    dest.mkdir()
    victim = dest / "cell.env"
    victim.write_text("FOO=bar\n")
    (dest / ".aorta-staged").write_text("cell.env\n")

    # `stat -c %u` is what the script consults; a manifest owned by another uid
    # is simulated by making the script see a different id.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "id").write_text("#!/usr/bin/env bash\necho 999999\n")
    (fake_bin / "id").chmod(0o755)

    proc = subprocess.run(
        [bash, str(_SOURCE / "stage_scripts.sh"), str(dest)],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert victim.exists(), "staging deleted a file named by a manifest it does not own"
    assert "not owned by this user" in proc.stderr


def test_stage_scripts_does_not_write_through_a_planted_symlink(
    bash: str, tmp_path: Path
) -> None:
    """`cp` follows a symlink at the destination.

    `dest` is explicitly allowed to be shared, so another user could plant
    `host_launch.sh -> <a file this caller can write>` and have staging
    overwrite that target instead of creating a file in `dest`. The manifest was
    protected; the copied scripts were not. Each file lands via a private
    temporary and a rename, which replaces the link rather than following it.
    """
    dest = tmp_path / "shared"
    dest.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite me\n")
    (dest / "host_launch.sh").symlink_to(victim)

    proc = subprocess.run(
        [bash, str(_SOURCE / "stage_scripts.sh"), str(dest)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert victim.read_text() == "do not overwrite me\n", "staging wrote through the symlink"
    staged = dest / "host_launch.sh"
    assert not staged.is_symlink(), "the planted link survived instead of being replaced"
    assert staged.read_text() == (_SOURCE / "host_launch.sh").read_text()
    assert staged.stat().st_mode & 0o111, "staged shell scripts must stay executable"


def test_stage_scripts_leaves_no_temporary_file_behind(bash: str, tmp_path: Path) -> None:
    """Both the per-file staging temporaries and the manifest one are renamed
    into place, so a shared directory does not accumulate `.aorta-stage.*`."""
    dest = tmp_path / "staged"
    dest.mkdir()
    proc = subprocess.run(
        [bash, str(_SOURCE / "stage_scripts.sh"), str(dest)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not list(dest.glob(".aorta-stage.*")), "per-file temporary left behind"


def test_stage_scripts_leaves_no_temporary_manifest_behind(bash: str, tmp_path: Path) -> None:
    """The rename is what makes the write atomic; the temporary must not linger,
    or a shared staging directory accumulates one per run and the next glob-free
    reader has several manifests to choose from."""
    dest = tmp_path / "staged"
    dest.mkdir()
    proc = subprocess.run(
        [bash, str(_SOURCE / "stage_scripts.sh"), str(dest)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (dest / ".aorta-staged").is_file()
    assert not list(dest.glob(".aorta-staged.*")), "temporary manifest left behind"


def test_stage_scripts_leaves_an_unrelated_pycache_alone(bash: str, tmp_path: Path) -> None:
    """The syntax check compiles Python, and py_compile writes bytecode next to
    the source -- so staging used to create `dest/__pycache__` and then delete it.

    That deletion sat outside the manifest's ownership boundary: `dest` is
    explicitly allowed to be shared or pre-populated, so a first staging run into
    a directory that already held an unrelated `__pycache__` removed somebody
    else's tree. The bytecode goes to a private prefix now, so there is nothing
    in `dest` to clean up and nothing there gets removed.
    """
    dest = tmp_path / "staged"
    dest.mkdir()
    squatter = dest / "__pycache__"
    squatter.mkdir()
    (squatter / "someone_elses.cpython-311.pyc").write_bytes(b"not ours")

    proc = subprocess.run(
        [bash, str(_SOURCE / "stage_scripts.sh"), str(dest)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "syntax OK" in proc.stdout, "the Python files were not compiled at all"
    assert (squatter / "someone_elses.cpython-311.pyc").read_bytes() == b"not ours"
    assert list(squatter.iterdir()) == [squatter / "someone_elses.cpython-311.pyc"], (
        "staging wrote its own bytecode into a directory it does not own"
    )


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
