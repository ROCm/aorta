#!/usr/bin/env python3
"""Harvest TokenSpeed kernel code objects and emit a mode: sanitizer recipe.

TokenSpeed's performance kernels are Gluon/Triton, so they do not exist as
committed binaries -- they are JIT-compiled on first use into the Triton cache.
aorta's sanitizer pipeline, meanwhile, consumes code objects by path plus a
SHA-256 identity (``source.kind: kernel_list``). This script bridges the two:
run the kernel once with a clean cache, collect the resulting ``.hsaco``,
inventory it, and write a ready-to-run recipe pointing at what was collected.

The output is a *run area*, not something to commit. A harvested object is
specific to the image, the GPU target, and the shapes that were compiled, so
pinning one in git would assert a provenance it does not have. Re-harvest
instead; it takes seconds.

Compilation can be driven two ways. ``--kernel`` / ``--op`` use TokenSpeed's
benchmark harness, which is precise but only reaches ``gemm.mm`` -- every other
family lacks a registered input generator. ``--pytest-suite`` runs one of
TokenSpeed's own op test suites instead; those build their inputs directly, so
they are the only way to compile the attention, MoE, quantization, sampling and
transform kernels, and therefore the only way to get them into a Waitcheck
corpus today.

Usage (on the compute node, with the image already pulled):

  # gemm, via the benchmark harness
  python3 harvest_code_objects.py \
      --image lightseekorg/tokenspeed-amd:nightly-20260714 \
      --kernel gluon_mm_a16w16_gfx950 --dtype bf16 --dtype-role a \
      --dest /tmp/ts-work/sanitizer-run \
      --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck"

  # attention, via TokenSpeed's own tests
  python3 harvest_code_objects.py \
      --image lightseekorg/tokenspeed-amd:nightly-20260714 \
      --pytest-suite tokenspeed-kernel/test/ops/test_attention.py \
      --pytest-k mha_prefill \
      --dest /tmp/ts-work/attn-run \
      --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck"

Then:

  aorta sweep run --recipe /tmp/ts-work/sanitizer-run/waitcheck-<label>.yaml \
      --output-dir /tmp/ts-work/sanitizer-out

where <label> is the --kernel / --op value reduced to [a-z0-9-]; the exact path
is printed at the end of the harvest.

``--consan`` additionally emits ConSan assets: one loader shim and one
single-kernel recipe per harvested object, built with the generic Triton loader
from ROCm/aorta#403. ConSan takes exactly one code object per run, so these are
per-object rather than one recipe over the whole list:

  python3 harvest_code_objects.py ... --consan --consan-limit 4
  for recipe in /tmp/ts-work/attn-run/consan/consan-*.yaml; do
      aorta sweep run --recipe "$recipe" \
          --output-dir "/tmp/ts-work/consan-out/$(basename "$recipe" .yaml)"
  done
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import posixpath
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

# Ceiling for the host-side helpers that read objects produced by the image:
# the Waitcheck inventory and the ConSan loader. Both parse third-party binaries
# and neither has any reason to take minutes, so this is generous rather than
# tuned -- its job is to turn a wedge into a diagnosable failure while the GPU
# is still needed by the rest of the sweep.
_SUBPROCESS_TIMEOUT_SEC = 120


# Filesystems the docker daemon cannot bind-mount from, or can only mount with
# root-squash surprises. Matched on fstype rather than on path, because the
# path spelling says nothing: /home is often local and /mnt, /shared, /users or
# an autofs mount point is often not.
#
# THREE COPIES OF THIS LIST EXIST, and adding an fstype means editing all of
# them: here, and as _NETWORK_FSTYPES in host_launch.sh and stage_scripts.sh.
# The two shells have to stand alone -- one runs before anything is staged and
# the other is what stages it -- so a shared sourced file was dropped rather
# than made conditional; see the note in host_launch.sh.
_NETWORK_FSTYPES = frozenset(
    {
        "afs",
        "beegfs",
        "ceph",
        "cifs",
        "fuse.cephfs",
        "fuse.glusterfs",
        "fuse.sshfs",
        "gfs2",
        "glusterfs",
        "gpfs",
        "lustre",
        "nfs",
        "nfs4",
        "smb3",
    }
)


def _network_filesystem(path: Path, mounts: Path = Path("/proc/mounts")) -> str | None:
    """The network fstype backing ``path``, or ``None`` if it looks local.

    Resolved by finding the longest mount point in /proc/mounts that is a
    prefix of the path, which is the mount the kernel would use. Best-effort by
    construction: a platform without /proc/mounts, or a path whose mount cannot
    be identified, is treated as local rather than blocking a harvest on a
    guess -- docker's own error is the backstop.
    """
    try:
        entries = mounts.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    # Resolved even when already absolute: an absolute `--dest` can still be a
    # symlink, and `/tmp/run -> /mnt/nfs/run` was then matched against `/tmp`
    # and called local. `resolve()` is non-strict, so a path that does not exist
    # yet keeps its missing suffix rather than raising.
    target = path.resolve()
    best: tuple[int, str] | None = None
    for entry in entries:
        fields = entry.split()
        if len(fields) < 3:
            continue
        # /proc/mounts octal-escapes spaces and tabs in the mount point.
        mount_point = fields[1].replace("\\040", " ").replace("\\011", "\t")
        fstype = fields[2]
        try:
            mount = Path(mount_point)
        except ValueError:
            continue
        if mount != target and mount not in target.parents:
            continue
        depth = len(mount.parts)
        if best is None or depth > best[0]:
            best = (depth, fstype)

    if best is None:
        return None
    fstype = best[1]
    return fstype if fstype in _NETWORK_FSTYPES else None


def _is_contained_file(path: Path, root: Path) -> bool:
    """A real file the container wrote inside ``root``, not a link out of it.

    The Triton cache is written by the third-party image, so everything under it
    is untrusted input to this host-side pass. ``rglob`` happily returns a
    symlink named ``*.hsaco``, and hashing or copying one follows it *as the
    host user* -- so a link to any file this user can read gets content-addressed
    into ``code_objects/`` and shipped into a sanitizer recipe. The ConSan path
    is worse, because it also picks up the ``.json`` and ``.amdgcn`` sidecars
    beside the object.

    Skipped with a warning rather than raising: one planted link should not
    discard an otherwise good harvest, but it must not be silent either.
    """
    try:
        if path.is_symlink() or not path.is_file():
            print(
                f"harvest: skipping {path}, which is not a regular file "
                "(the Triton cache is written by the container, so a symlink "
                "here would be followed as you)",
                file=sys.stderr,
            )
            return False
        resolved = path.resolve()
        root_resolved = root.resolve()
        if root_resolved != resolved and root_resolved not in resolved.parents:
            print(
                f"harvest: skipping {path}, which resolves outside {root_resolved}",
                file=sys.stderr,
            )
            return False
    except OSError:
        return False
    return True


def _reset_dir(path: Path) -> None:
    """Replace ``path`` with an empty directory, creating parents as needed."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _suite_root(suite: str) -> str:
    """The in-container directory a pytest suite must run from.

    Each suite is a package rooted at the parent of its ``test`` directory, with
    its own conftest and a top-level ``utils`` module; collection fails from
    anywhere else.
    """
    parts = Path(suite).parts
    if "test" in parts:
        return str(Path(*parts[: parts.index("test")]))
    return "."


def _in_container(path: str) -> str:
    """Resolve a caller-supplied suite path against the container's /workspace.

    Relative paths are taken as relative to the mount root; absolute ones are
    already container paths and are returned unchanged. Prefixing an absolute
    path would produce ``/workspace//workspace/...``.
    """
    return path if path.startswith("/") else f"/workspace/{path}"


def _driver(args: argparse.Namespace) -> tuple[list[str], str, str]:
    """The in-container command that compiles the kernels, plus workdir + label.

    Two ways in. The benchmark harness is precise but only reaches gemm.mm,
    because every other family lacks an input generator. TokenSpeed's own pytest
    suites build their inputs directly, so they compile the attention, MoE,
    quantization, sampling and transform kernels the harness cannot -- which is
    the only way to get those into a Waitcheck corpus today.
    """
    if args.pytest_suite:
        # Absolute, because the workdir below is the suite's package root rather
        # than /workspace -- a path relative to /workspace would not resolve
        # from there.
        suite = _in_container(args.pytest_suite)
        cmd = [
            "python3",
            "-m",
            "pytest",
            suite,
            "-q",
            "--no-header",
            # The source tree is read-only for a non-root container user and the
            # cache plugin turns that into per-run warning noise.
            "-p",
            "no:cacheprovider",
        ]
        if args.pytest_k:
            cmd += ["-k", args.pytest_k]
        # Derived from the already-absolute `suite`, so an absolute --pytest-suite
        # is not prefixed a second time: `/workspace//workspace/...` makes docker
        # fail with a path error before pytest ever starts, which reads as a
        # broken image rather than a bad argument.
        workdir = posixpath.normpath(_in_container(_suite_root(suite)))
        label = Path(args.pytest_suite).stem or "suite"
        return cmd, workdir, label

    selector = [args.kernel] if args.kernel else ["--op", args.op]
    cmd = [
        "python3",
        "-m",
        "tokenspeed_kernel.benchmark",
        *selector,
        "--dtype",
        args.dtype,
        "--dtype-role",
        args.dtype_role,
        # Minimal iterations: this run exists to trigger compilation, not to
        # measure anything. Verification stays off for the same reason.
        "--no-verify",
        "--warmup-iters",
        "1",
        "--bench-iters",
        "1",
    ]
    label = args.kernel or (args.op or "kernels").replace(".", "_")
    return cmd, "/workspace", label


def _assert_cache_writable(cache_dir: Path) -> None:
    """Fail early if the JIT cache is not writable by this user.

    Worth its own check because the downstream symptom is actively misleading.
    Triton initialises its AMD driver by compiling a small HIP utility into this
    directory; if that write fails, TokenSpeed catches the error with a bare
    ``except BaseException`` and re-raises "Triton is not supported on the
    current platform", which sends you looking at the GPU instead of at a
    permission bit. The usual cause is docker having auto-created the mount
    point as root because the path did not exist on this node.
    """
    if not cache_dir.is_dir():
        raise SystemExit(f"harvest: cache dir {cache_dir} was not created")
    probe = cache_dir / ".write-probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"harvest: cache dir {cache_dir} is not writable by uid {os.getuid()} "
            f"({exc}). If docker created it as root, remove it and re-run on the "
            "node that will execute the container -- /tmp is per-node."
        ) from exc


def _force_remove_container(name: str, env: dict[str, str]) -> None:
    """Stop and remove a container the docker client no longer supervises.

    Best-effort and deliberately quiet: the caller is already raising the real
    failure, and a cleanup error must not replace it with a less useful one.
    ``docker rm -f`` covers both the still-running and the already-exited case,
    so no ``docker stop`` is needed first.
    """
    try:
        subprocess.run(
            ["docker", "rm", "-f", name],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write(f"harvest: could not remove container {name}: {exc}\n")


def _run_kernel(args: argparse.Namespace, cache_dir: Path) -> None:
    """Compile the kernels in-container so the JIT populates cache_dir."""
    _assert_cache_writable(cache_dir)
    driver, workdir, _ = _driver(args)
    # Named so a timeout is recoverable. `subprocess.run(timeout=...)` kills the
    # docker *client*, which does not stop the daemon-managed container: a hung
    # compile would otherwise keep a GPU busy for the rest of the sweep with no
    # handle left to stop it. `--rm` covers the normal exit; the name covers the
    # abnormal one.
    #
    # The random half is what makes the name cleanup authority. A pid alone is
    # predictable and reused, so a stale container from an earlier harvest --
    # or one another daemon user named first -- would collide on `docker run`,
    # and the removal below would then delete a container this process never
    # started.
    container = f"aorta-ts-harvest-{os.getpid()}-{secrets.token_hex(8)}"
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container,
        # Bridge by default, matching host_launch.sh's kernel/pytest routes:
        # harvesting compiles from generated tensors and contacts nothing
        # off-node, so there is no reason to hand it the host's network
        # namespace. Overridable via --network for a node that needs it.
        "--network",
        args.network,
        # Some MoE and attention tests move data over shared memory; the 64MB
        # docker default makes them die with an opaque bus error. --shm-size is
        # the whole fix -- processes inside one container already share a
        # private IPC namespace -- so this deliberately does not pass
        # --ipc=host, which would expose node-wide shared memory and semaphores
        # to a third-party image for no gain.
        "--shm-size=16g",
        "-w",
        workdir,
        # Run as the calling user, not the image's root. As root the JIT writes
        # a root-owned cache, and the caller then cannot delete its own run area
        # -- including the rmtree in main(), so a second harvest fails with
        # EPERM. USER is set because Triton calls getpass.getuser(), which falls
        # through to pwd.getpwuid() and raises for a uid that is absent from the
        # container's /etc/passwd. HOME points into the mount so anything else
        # wanting a home directory gets a writable one.
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"USER={getpass.getuser()}",
        "-e",
        "HOME=/triton-cache",
        "-e",
        f"HIP_VISIBLE_DEVICES={args.gpus}",
        "-e",
        "TRITON_CACHE_DIR=/triton-cache",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add",
        "video",
        "--group-add",
        "render",
        "--security-opt",
        "seccomp=unconfined",
        "-v",
        f"{cache_dir}:/triton-cache",
        args.image,
        *driver,
    ]
    env = dict(os.environ)
    if args.docker_config:
        env["DOCKER_CONFIG"] = args.docker_config

    print(f"harvest: compiling via {' '.join(driver)}")
    try:
        proc = subprocess.run(
            docker_cmd, env=env, capture_output=True, text=True, timeout=args.timeout
        )
    except subprocess.TimeoutExpired:
        _force_remove_container(container, env)
        raise
    if proc.returncode != 0:
        # A nonzero client is not evidence the container is gone. `--rm` fires
        # when the *container* exits, and the client can return nonzero while
        # the daemon still has it running -- an API disconnect, or the client
        # being killed -- which leaves the GPU held for everything that follows.
        # Timeout was covered and this path was not, though it reaches the same
        # state. A no-op once `--rm` has done its job.
        _force_remove_container(container, env)
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(
            f"harvest: compile run failed (rc={proc.returncode}). "
            "'No input generator registered' means the operator is not drivable "
            "through the benchmark harness -- only gemm.mm is; use "
            "--pytest-suite to drive it through TokenSpeed's own tests instead. "
            "'not supported on the current platform' is usually an unwritable "
            "TRITON_CACHE_DIR rather than a real platform problem."
        )


def _inventory(waitcheck: Path | None, obj: Path) -> list[dict]:
    """Ask rj_waitcheck to describe the kernels inside one code object.

    Using the sanitizer's own parser rather than readelf keeps the recorded
    target and entry offsets consistent with what the sanitizer will later
    select on.
    """
    # Only an omitted --waitcheck may fall back to guessed identities. When one
    # is supplied, a typo in the path used to return the same empty inventory
    # as omitting it, so the run continued on names and indices that were never
    # verified against the object -- exactly the outcome the caller asked to
    # avoid by passing the binary. Fail instead, and say which case this is.
    if waitcheck is None:
        return []
    if not waitcheck.exists():
        raise SystemExit(
            f"harvest: --waitcheck {waitcheck} does not exist. Omit the option to "
            "harvest with guessed kernel names, or correct the path; a supplied "
            "binary is required to resolve exact identities."
        )
    if not os.access(waitcheck, os.X_OK):
        raise SystemExit(f"harvest: --waitcheck {waitcheck} is not executable.")
    # Bounded, because the object being parsed came out of a third-party image.
    # Every other subprocess here is already bounded; this one wedging would
    # hang the harvest with the GPU still held, which is the failure the docker
    # timeout exists to prevent, one layer further in.
    try:
        proc = subprocess.run(
            [str(waitcheck), "--list-kernels", str(obj)],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"harvest: {waitcheck} --list-kernels did not finish within "
            f"{_SUBPROCESS_TIMEOUT_SEC}s on {obj}. The object comes from the "
            "image and may be malformed; re-run with --waitcheck omitted to "
            "harvest with guessed names, or exclude this kernel."
        ) from exc
    if proc.returncode != 0:
        # A binary was supplied and it rejected the object, so this is a real
        # inventory failure. Falling back to the guessed stem here would emit a
        # recipe whose names and indices were never verified against the object
        # -- Waitcheck would then attribute a finding to whatever kernel the
        # guess happened to name. The fallback below is for the case where the
        # caller intentionally omitted the binary, not for this one.
        raise SystemExit(
            f"harvest: {waitcheck} --list-kernels failed on {obj} "
            f"(rc={proc.returncode}):\n{proc.stderr.strip()[-2000:]}"
        )
    entries = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "kernel":
            entries.append(record)
    if not entries:
        # rc was 0 but nothing describable came back. Falling through to the
        # guessed stem here would silently downgrade a run that asked for exact
        # identities, so treat an empty inventory from a supplied binary as the
        # failure it is.
        raise SystemExit(
            f"harvest: {waitcheck} --list-kernels reported no kernel records for "
            f"{obj}. Omit --waitcheck to harvest with guessed names instead of "
            "continuing without verified identities."
        )
    return entries


def _entry_offset(value: object) -> int | None:
    """Parse Waitcheck's ``kernel_entry`` into an ``entry_offset``.

    Accepts the hex form Waitcheck emits as well as a plain decimal, matching
    what the sanitizer's own parser accepts. Returns ``None`` when the field is
    absent or unusable so the entry degrades to a whole-object scan rather than
    carrying a wrong offset -- a wrong offset would make Waitcheck reject the
    identity outright.
    """
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text, 16 if text.lower().startswith("0x") else 10)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _write_recipe(dest: Path, label: str, target: str, kernels: list[dict]) -> Path:
    """Emit a mode: sanitizer recipe over the harvested identities.

    Written by hand rather than via a YAML library so the file carries the same
    explanatory comments a committed recipe would, and so this script has no
    dependency beyond the standard library.

    ``label`` reaches two places with different escaping rules -- a path
    component and an unquoted YAML scalar -- so it is reduced to the character
    set that is safe in both before either use. It comes from ``--kernel`` /
    ``--op``, which in practice is pasted from an inventory of third-party
    kernel names: one containing ``../`` would write the recipe outside
    ``--dest``, and one containing a newline or a colon would reshape the
    generated YAML. Same treatment the ConSan assets already get, including the
    containment check, since a sanitized name is an argument about the sanitizer
    rather than a proof about the path.
    """
    slug = _ticket_suffix(label).lower()
    recipe_path = dest / f"waitcheck-{slug}.yaml"
    _ensure_within(dest, recipe_path)
    lines = [
        "# GENERATED by harvest_code_objects.py -- do not commit.",
        "#",
        "# Waitcheck over TokenSpeed kernel code objects harvested from the Triton",
        "# JIT cache. Paths are absolute and the digests below pin the exact objects",
        "# that were compiled on this node; re-harvest after changing image, GPU",
        "# target, or shapes rather than editing this file.",
        "#",
        "# scope.kind: kernel keeps this to exact-entry analysis. ConSan is not",
        "# requested here because it takes exactly one code object per run, so it",
        "# cannot share a kernel_list recipe: harvest with --consan for one",
        "# ConSan recipe per object instead. Whole-application ConSan stays out of",
        "# reach either way -- aorta fails it closed because RocJITsu has no",
        "# entry-point allowlist and would instrument everything indiscriminately.",
        "schema_version: 1",
        "mode: sanitizer",
        f"ticket: TOKENSPEED-WAITCHECK-{_ticket_suffix(label)}",
        "",
        "sanitizer_plan:",
        f"  target: {_yaml_scalar(target)}",
        "  source:",
        "    kind: kernel_list",
        "    kernels:",
    ]
    for kernel in kernels:
        lines.extend(
            [
                f"      - name: {_yaml_scalar(kernel['name'])}",
                f"        code_object: {_yaml_scalar(kernel['code_object'])}",
                f"        code_object_sha256: {_yaml_digest(kernel['sha256'])}",
                f"        code_object_index: {_yaml_int('code_object_index', kernel['code_object_index'])}",
            ]
        )
        # Only when inventory resolved one: an absent entry_offset is a
        # whole-object scan, which is a weaker but valid identity. Emitting a
        # null would fail the recipe's integer validation instead.
        if kernel.get("entry_offset") is not None:
            lines.append(
                f"        entry_offset: "
                f"{_yaml_int('entry_offset', kernel['entry_offset'])}"
            )
    lines.extend(
        [
            "  scope:",
            "    kind: kernel",
            "  selection:",
            "    requirement: top_dispatch_count",
            # kernel_list gives every entry dispatch_count 1, so top_n must cover
            # the whole list or harvested kernels are silently dropped.
            f"    top_n: {len(kernels)}",
            "  sanitizers:",
            "    - waitcheck",
            "  policy:",
            "    consan_policy: strict",
            "    on_missing_backend: fail",
            "  output:",
            "    report: sanitizer_report.json",
            "",
        ]
    )
    recipe_path.write_text("\n".join(lines))
    return recipe_path


def _default_consan_loader() -> Path:
    """Where ``triton_consan_loader.py`` lives in a source checkout.

    Resolved from this file rather than the cwd so the script works from
    anywhere. ``scripts/`` is not packaged into the wheel, so an installed aorta
    has to be told the path explicitly; the caller-facing error says so.
    """
    return (
        Path(__file__).resolve().parents[4] / "scripts" / "sanitizers" / "triton_consan_loader.py"
    )


def _asset_stem(kernel: dict) -> str:
    """Filename stem for one harvested identity.

    ``kernel['name']`` is whatever Waitcheck read out of a third-party image's
    code object, so it is untrusted input. Interpolating it into a filename let
    a name containing ``../`` -- or a leading ``/`` -- place the staged object,
    the shim and the recipe outside the harvest directory. The name is therefore
    carried as recipe data only, never as a path component.

    Digest and index alone were not enough to name an identity, though. Several
    kernels commonly live in one code object at one index and differ only by
    entry -- which is the case this harvester exists to handle, since it is what
    ``entry_offset`` is carried for -- so they shared a stem, and each one's
    shim, staged object and recipe overwrote the previous one's while the
    manifest still listed them all. The run then reported N recipes of which
    only the last was the kernel it named. The identity is completed with a
    digest of the parts that vary within an object: the name, hashed rather than
    interpolated so it stays out of the path, and the entry offset.
    """
    entry = hashlib.sha256(
        "\0".join((str(kernel["name"]), str(kernel.get("entry_offset")))).encode()
    ).hexdigest()[:12]
    return f"{kernel['sha256'][:12]}.{int(kernel['code_object_index'])}.{entry}"


def _ensure_within(base: Path, *candidates: Path) -> None:
    """Refuse to write anything that resolves outside ``base``.

    A second line of defence behind :func:`_asset_stem`: filenames are derived
    from a digest now, but this is the check that keeps that property true if a
    caller-supplied path or a future field ever reintroduces a name into one.
    """
    root = base.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved != root and root not in resolved.parents:
            raise SystemExit(
                f"harvest: refusing to write {resolved}, which is outside {root}"
            )


def _ticket_suffix(name: str) -> str:
    """A ticket-safe rendering of an untrusted kernel name: ``[A-Z0-9-]`` only."""
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in str(name)).strip("-").upper()
    return cleaned or "KERNEL"


def _yaml_digest(value: object) -> str:
    """A sha256 from the inventory, checked to be one.

    Emitted unquoted because the recipe schema wants a scalar string of hex, and
    quoting would not make a malformed value safe -- it would make it a
    well-formed recipe selecting a digest that cannot match anything. So this
    validates instead: the field comes from the same third-party JSON as the
    kernel names, and everything else derived from it is either quoted or, like
    this, constrained to a shape that cannot carry YAML syntax.
    """
    text = str(value)
    if len(text) != 64 or not all(ch in "0123456789abcdef" for ch in text.lower()):
        raise SystemExit(
            f"harvest: inventory reported sha256 {text!r}, which is not a "
            "64-character hex digest. The code object's metadata is malformed; "
            "re-harvest, or exclude this kernel."
        )
    return text.lower()


def _yaml_int(field: str, value: object) -> str:
    """An integer field from the inventory, checked to be one.

    Same reasoning as ``_yaml_digest``: the recipe schema wants a number here,
    so the fix is to require one rather than to quote whatever arrived.
    """
    try:
        return str(int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"harvest: inventory reported {field}={value!r}, which is not an "
            "integer. The code object's metadata is malformed; re-harvest, or "
            "exclude this kernel."
        ) from exc


def _yaml_scalar(value: object) -> str:
    """Quote a value for YAML, so an odd kernel name cannot reshape the recipe.

    JSON's string form is a valid YAML double-quoted scalar, so this both
    escapes and quotes in one step. Names come from a third-party binary and
    may contain ``:``, ``#`` or a newline, any of which would otherwise produce
    a recipe that parses as something other than what was harvested.
    """
    return json.dumps(str(value))


def _metadata_kernel_name(kernel: dict) -> str | None:
    """The kernel name in the Triton metadata beside the object, if readable.

    That name is the one the loader resolves for itself, so it is the only one a
    ConSan result can honestly be attributed to. ``None`` when the sidecar is
    missing, unreadable, or not a regular file inside the cache entry -- all of
    which leave the caller to fall back on harvest order.
    """
    sidecar = Path(kernel["cache_object"]).with_suffix(".json")
    # Through the same guard as every other read of this cache, and before it
    # rather than after: this runs ahead of the loop that stages the sidecars, so
    # an unguarded read here would be the first thing to touch a planted link.
    # The name it could disclose is the weaker half; the read itself is
    # unbounded, and `k.json -> /dev/zero` would wedge the harvest with the GPU
    # still held for the rest of the sweep -- the failure _SUBPROCESS_TIMEOUT_SEC
    # bounds for subprocesses, arriving through a plain file read.
    if not _is_contained_file(sidecar, sidecar.parent):
        return None
    try:
        doc = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    name = doc.get("name") if isinstance(doc, dict) else None
    return name if isinstance(name, str) else None


def _consan_identities(kernels: list[dict]) -> list[dict]:
    """One identity per code object, named the way the loader will resolve it.

    The loader is driven with ``--hsaco``, and it takes the entry from the Triton
    metadata sitting beside that object -- ``--kernel-name`` selects nothing on
    that path. One recipe per *harvested identity* therefore emitted several
    recipes that all analyzed the same entry, each attributing the same static
    result to a different kernel name and entry offset. A helper kernel sharing
    an object with the harvested one got a report of its own, describing coverage
    that was never measured for it, and ``--consan-limit`` spent its budget
    re-analyzing one object while skipping others entirely.

    So the fan-out is per object, and the identity kept is the metadata-backed
    one where the sidecar says which that is. The other names are recorded in the
    manifest entry rather than dropped, since what an object contains is worth
    knowing -- it is the *attribution* that has to stop pretending to separate
    them.

    ``identity_source`` records which of three ways the name was arrived at,
    because they are not equally trustworthy:

    ``metadata``
        The sidecar named a kernel, and it is one of the identities harvested for
        this object. The name and the Waitcheck entry offset both describe what
        the loader will resolve.
    ``metadata_unharvested``
        The sidecar named a kernel that Waitcheck never listed for this object.
        Keeping the first harvested identity there would attribute the run to a
        name and an entry offset that are *not* what gets resolved -- so the
        metadata's name is used instead and the entry offset is dropped, leaving
        a whole-object identity, which is weaker than an exact-entry one but true.
    ``harvest_order``
        No readable sidecar, so there is nothing to check the harvested name
        against. It is the best available and the manifest says so.

    Order is harvest order, so ``--consan-limit`` still takes a stable prefix.
    """
    grouped: dict[str, list[dict]] = {}
    for kernel in kernels:
        grouped.setdefault(str(kernel["sha256"]), []).append(kernel)

    identities: list[dict] = []
    for group in grouped.values():
        # Any sidecar in the group: byte-identical objects compiled twice are one
        # digest at two cache paths, and either one names the same entry.
        metadata_name = next(
            (name for name in map(_metadata_kernel_name, group) if name is not None), None
        )
        if metadata_name is None:
            identities.append({**group[0], "identity_source": "harvest_order"})
            continue
        match = next((k for k in group if k["name"] == metadata_name), None)
        if match is not None:
            identities.append({**match, "identity_source": "metadata"})
            continue
        print(
            f"harvest: consan   -> {group[0]['code_object']} declares "
            f"'{metadata_name}' in its metadata, which Waitcheck did not list "
            f"for this object (it listed {sorted({k['name'] for k in group})}). "
            "Using the metadata name without an entry offset: it is what the "
            "loader resolves, and naming a harvested identity instead would "
            "report this object's result under a kernel and entry that are not "
            "the ones analyzed.",
            file=sys.stderr,
        )
        identities.append(
            {
                **group[0],
                "name": metadata_name,
                "entry_offset": None,
                "identity_source": "metadata_unharvested",
            }
        )
    return identities


def _emit_consan_shim(
    loader: Path,
    kernel: dict,
    isa_dir: Path,
    bin_dir: Path,
) -> tuple[Path, Path]:
    """Emit one ConSan shim for one harvested kernel identity.

    ``consan_command`` is executed as a bare argv with no arguments, so the
    loader cannot be named directly -- ``emit-command`` bakes the resolved
    arguments into a small shim instead. ``--copy-object`` lifts the object and
    its sidecars out of the Triton cache, which is scratch space: without it the
    recipe would point into a directory that the next harvest deletes.
    """
    stem = _asset_stem(kernel)
    staged_object = isa_dir / f"{stem}.hsaco"
    shim = bin_dir / f"consan_{stem}"
    _ensure_within(isa_dir, staged_object)
    _ensure_within(bin_dir, shim)

    # The object itself was filtered when the cache was walked, but the loader
    # reaches further: it resolves and copies the `.json` and `.amdgcn` sitting
    # beside it, with APIs that follow symlinks. The container writes that
    # directory, so `k.json -> /etc/something` would have the host read and copy
    # a file of its choosing. Each sidecar is checked the same way the object
    # was, before the loader is invoked at all.
    cache_object = Path(kernel["cache_object"])
    cache_root = cache_object.parent
    for sidecar in (cache_object.with_suffix(".json"), cache_object.with_suffix(".amdgcn")):
        if sidecar.exists() and not _is_contained_file(sidecar, cache_root):
            raise SystemExit(
                f"harvest: refusing to run the ConSan loader for {kernel['name']}: "
                f"its sidecar {sidecar} is not a regular file inside the Triton "
                "cache. The cache is written by the container, so this would "
                "have the loader read a file it chose, as you."
            )
    # Bounded for the same reason as the --list-kernels parse: the object this
    # reads is a Triton cache entry written by the third-party image, so a hang
    # here wedges the --consan path with the GPU still held.
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(loader),
                "emit-command",
                "--hsaco",
                kernel["cache_object"],
                # Passed for the loader's other resolution paths, and because it
                # is what the recipe names. It does *not* select an entry here:
                # with --hsaco the loader reads the entry out of the object's own
                # metadata, which is why the fan-out is one recipe per object and
                # the name emitted is the metadata-backed one (_consan_identities).
                "--kernel-name",
                kernel["name"],
                "--copy-object",
                str(staged_object),
                "--output",
                str(shim),
            ],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"harvest: the ConSan loader did not finish within "
            f"{_SUBPROCESS_TIMEOUT_SEC}s on {kernel['cache_object']}. Re-run "
            "without --consan, or narrow the selection with --consan-limit."
        ) from exc
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:])
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(
            f"harvest: emit-command failed for {kernel['name']} (rc={proc.returncode})"
        )
    return staged_object, shim


def _write_consan_recipe(
    consan_dir: Path,
    kernel: dict,
    staged_object: Path,
    shim: Path,
    target: str,
    policy: str,
) -> Path:
    """Emit a single-kernel mode: sanitizer recipe driving ConSan via the shim."""
    stem = _asset_stem(kernel)
    recipe_path = consan_dir / f"consan-{stem}.yaml"
    _ensure_within(consan_dir, recipe_path)
    lines = [
        "# GENERATED by harvest_code_objects.py --consan -- do not commit.",
        "#",
        "# ConSan over one TokenSpeed Gluon/Triton code object. One recipe per",
        "# object by necessity: ConSan takes exactly one per run, and a single",
        "# TokenSpeed kernel compiles to several shape-specialized objects.",
        "#",
        "# Per object, and not per kernel *name*. The loader is invoked with the",
        "# object and resolves the entry from the metadata beside it, so the name",
        "# below is that metadata's -- a second kernel sharing this object is not",
        "# separately analyzed, and manifest.json lists the others under",
        "# `identities` rather than implying a result for each.",
        "#",
        "# consan_command is the shim emit-command wrote. It pins the SHA-256 of",
        "# the object and its metadata, so it refuses to run after a recompile",
        "# rather than reporting new bytes under the old identity -- re-harvest",
        "# instead of editing this file.",
        "#",
        f"# policy.consan_policy: {policy}.",
    ]
    if policy == "lenient":
        lines.extend(
            [
                "# lenient is the default here, and the reason is worth stating.",
                "# strict sets RJ_CONSAN_MOI_REQUIRE_RECORDS, which demands visible",
                "# dynamic records. The loader runs in `load` mode: it loads and",
                "# instruments the object but never dispatches it, so there is no",
                "# dispatch packet and no records, and strict fails closed with",
                "# combined_hook_exit_86 no matter how healthy the run was.",
                "# Dispatching instead would need the argument signature, which",
                "# Triton does not write to the metadata for these kernels, plus",
                "# real shapes and buffer extents -- synthesized ones give a",
                "# zero-trip kernel that records nothing anyway.",
                "#",
                "# So this lane verifies static instrumentation coverage: that",
                "# ConSan can read, patch and analyze TokenSpeed's JIT kernels.",
                "# Dynamic race evidence needs ConSan on TokenSpeed's own",
                "# dispatches, which needs the RocJITsu entry-point allowlist.",
            ]
        )
    lines.extend(
        [
            "schema_version: 1",
            "mode: sanitizer",
            f"ticket: TOKENSPEED-CONSAN-{_ticket_suffix(kernel['name'])}",
            "",
            "sanitizer_plan:",
            f"  target: {_yaml_scalar(target)}",
            "  source:",
            "    kind: kernel",
            "    kernel:",
            f"      name: {_yaml_scalar(kernel['name'])}",
            f"      code_object: {_yaml_scalar(staged_object)}",
            f"      code_object_sha256: {_yaml_digest(kernel['sha256'])}",
            f"      code_object_index: {_yaml_int('code_object_index', kernel['code_object_index'])}",
            *(
                [f"      entry_offset: {_yaml_int('entry_offset', kernel['entry_offset'])}"]
                if kernel.get("entry_offset") is not None
                else []
            ),
            f"    consan_command: {_yaml_scalar(shim)}",
            "    consan_log: true",
            "  scope:",
            "    kind: kernel",
            "  selection:",
            "    requirement: top_dispatch_count",
            "    top_n: 1",
            "  sanitizers:",
            "    - consan",
            "  policy:",
            f"    consan_policy: {policy}",
            "    on_missing_backend: fail",
            "    timeout_seconds: 600",
            "  output:",
            "    report: sanitizer_report.json",
            "",
        ]
    )
    recipe_path.write_text("\n".join(lines))
    return recipe_path


def _write_consan_assets(
    dest: Path,
    kernels: list[dict],
    target: str,
    loader: Path,
    policy: str,
    limit: int | None,
) -> Path:
    """Emit one shim + one recipe per code object, plus a manifest."""
    if not loader.exists():
        raise SystemExit(
            f"harvest: ConSan loader not found at {loader}. It ships in the aorta "
            "source tree at scripts/sanitizers/triton_consan_loader.py but is not "
            "packaged into the wheel; pass --consan-loader explicitly."
        )
    consan_dir = dest / "consan"
    # Replace rather than merge. Re-harvesting the same --dest with fewer
    # kernels, or a smaller --consan-limit, used to leave the previous run's
    # consan-*.yaml behind, and the documented `for r in consan/consan-*.yaml`
    # loop then ran those stale recipes against objects the new manifest does
    # not list. This directory is generated in full on every run and nothing
    # else writes into it, so clearing it is safe and makes the glob match the
    # manifest by construction.
    if consan_dir.exists():
        shutil.rmtree(consan_dir)
    isa_dir = consan_dir / "isa"
    bin_dir = consan_dir / "bin"
    for directory in (isa_dir, bin_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # One entry per code object, not per harvested identity -- see
    # _consan_identities. Deduplicated before the limit is applied, so
    # --consan-limit budgets objects that will actually be analyzed rather than
    # spending itself on repeats of one.
    per_object = _consan_identities(kernels)
    # `is None` rather than falsiness: a limit of 0 is a caller error, not a
    # request for everything.
    selected = per_object if limit is None else per_object[:limit]
    records = []
    for kernel in selected:
        staged_object, shim = _emit_consan_shim(loader, kernel, isa_dir, bin_dir)
        recipe = _write_consan_recipe(consan_dir, kernel, staged_object, shim, target, policy)
        records.append(
            {
                "kernel": kernel["name"],
                "sha256": kernel["sha256"],
                "code_object": str(staged_object),
                "consan_command": str(shim),
                "recipe": str(recipe),
                # How the name above was arrived at; see _consan_identities. A
                # reader comparing results across objects needs to know which of
                # them are metadata-confirmed and which are not.
                "identity_source": kernel["identity_source"],
                # Every harvested identity that shares this object, so a reader
                # of the manifest can see what the analysis does and does not
                # separate. The result is the object's, not this name's alone.
                "identities": sorted(
                    {kernel["name"]}
                    | {k["name"] for k in kernels if k["sha256"] == kernel["sha256"]}
                ),
            }
        )

    manifest = consan_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "target": target,
                "consan_policy": policy,
                "loader": str(loader),
                "entries": records,
                "objects": len(per_object),
                "identities": len(kernels),
                "skipped": len(per_object) - len(selected),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"harvest: consan   -> {len(records)} recipe(s) under {consan_dir}")
    if len(per_object) != len(kernels):
        print(
            f"harvest: consan   -> {len(kernels)} identity(ies) share "
            f"{len(per_object)} code object(s); one recipe each, because the "
            "loader resolves the entry from the object's own metadata"
        )
    if len(selected) != len(per_object):
        print(
            f"harvest: consan   -> {len(per_object) - len(selected)} object(s) "
            "skipped by --consan-limit"
        )
    print(f"harvest: manifest -> {manifest}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="TokenSpeed container image")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kernel", help="single kernel name, e.g. gluon_mm_a16w16_gfx950")
    group.add_argument("--op", help="operator family.mode, e.g. gemm.mm")
    group.add_argument(
        "--pytest-suite",
        help=(
            "drive compilation with a TokenSpeed test suite instead of the "
            "benchmark harness, e.g. tokenspeed-kernel/test/ops/test_attention.py. "
            "This is the only route to attention and MoE code objects, which the "
            "benchmark harness cannot compile."
        ),
    )
    parser.add_argument("--pytest-k", default=None, help="-k expression for --pytest-suite")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "fp8"])
    parser.add_argument("--dtype-role", default="a", help="a/b for gemm.mm")
    parser.add_argument("--dest", required=True, help="run area (node-local)")
    parser.add_argument("--gpus", default="0", help="HIP_VISIBLE_DEVICES")
    parser.add_argument("--waitcheck", default=None, help="path to rj_waitcheck")
    parser.add_argument("--docker-config", default=None, help="DOCKER_CONFIG override")
    parser.add_argument(
        "--network",
        default="bridge",
        help=(
            "docker network mode (default: bridge). Harvesting compiles kernels "
            "from generated tensors and reaches nothing off-node, so it does not "
            "need the host's network namespace; use --network host only if your "
            "node needs it to pull the image or a tokenizer."
        ),
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--consan",
        action="store_true",
        help=(
            "also emit ConSan assets: one loader shim and one single-kernel "
            "recipe per harvested code object, via "
            "scripts/sanitizers/triton_consan_loader.py (ROCm/aorta#403)"
        ),
    )
    parser.add_argument(
        "--consan-loader",
        default=None,
        help="path to triton_consan_loader.py (default: resolve in the source tree)",
    )
    parser.add_argument(
        "--consan-policy",
        default="lenient",
        choices=["lenient", "strict"],
        help=(
            "lenient (default) verifies static instrumentation coverage; strict "
            "additionally requires visible dynamic records, which load mode "
            "cannot produce because it never dispatches"
        ),
    )
    parser.add_argument(
        "--consan-limit",
        type=int,
        default=None,
        help=(
            "emit ConSan assets for at most N identities. An attention harvest "
            "yields 20 objects and each is a separate ConSan run."
        ),
    )
    args = parser.parse_args()

    if args.consan_limit is not None and args.consan_limit < 1:
        parser.error("--consan-limit must be at least 1")

    dest = Path(args.dest).resolve()
    network_fs = _network_filesystem(dest)
    if network_fs:
        raise SystemExit(
            f"harvest: {dest} is on a {network_fs} filesystem; the docker daemon "
            "cannot bind-mount it. Use a node-local path such as "
            "/tmp/ts-work/sanitizer-run."
        )
    cache_dir = dest / "triton-cache"
    objects_dir = dest / "code_objects"
    # Start from an empty cache so the inventory contains only what this run
    # compiled, rather than whatever a previous harvest left behind.
    _reset_dir(cache_dir)
    # Cleared for the same reason as consan/, though the consequence is milder:
    # staged object names are content-addressed, so a leftover is never
    # referenced by this run's recipe. It is disk that nothing will reclaim --
    # a re-harvest with a narrower selection keeps every .hsaco the wider one
    # staged, and these are the largest files the harvest writes.
    _reset_dir(objects_dir)

    _run_kernel(args, cache_dir)

    waitcheck = Path(args.waitcheck).resolve() if args.waitcheck else None
    found = [obj for obj in sorted(cache_dir.rglob("*.hsaco")) if _is_contained_file(obj, cache_dir)]
    if not found:
        raise SystemExit(
            "harvest: the run produced no .hsaco. Either the selection compiled "
            "nothing (torch_* solutions call into rocBLAS and emit no code "
            "object; a --pytest-k that deselects everything does the same), or "
            "TRITON_CACHE_DIR did not reach the container."
        )

    kernels: list[dict] = []
    targets = set()
    seen: set[tuple[str, str, int, int | None]] = set()
    for obj in found:
        # Stage content-addressed. The Triton cache holds one directory per
        # shape specialization and they reuse file names -- a single attention
        # run emits ten distinct `_fwd_kernel.hsaco`. Staging by bare name
        # would overwrite them in turn, leaving a recipe whose digests match
        # nothing on disk except the last copy, and Waitcheck would then reject
        # every earlier entry for a digest mismatch.
        digest = _sha256(obj)
        staged = objects_dir / f"{obj.stem}.{digest[:12]}{obj.suffix}"
        if not staged.exists():
            shutil.copy2(obj, staged)
        entries = _inventory(waitcheck, staged)
        if not entries:
            # Only reachable with --waitcheck omitted -- a supplied binary now
            # fails rather than returning nothing. No inventory means no
            # verified symbol name; fall back to the original file stem, which
            # the Triton cache names after the kernel. (staged.stem now carries
            # the digest suffix, so it is not usable.)
            candidates = [{"kernel_name": obj.stem, "code_object_index": 0}]
        else:
            candidates = entries

        for entry in candidates:
            if entry.get("target"):
                targets.add(str(entry["target"]))
            name = str(entry.get("kernel_name", obj.stem))
            index = int(entry.get("code_object_index", 0))
            # Carried through as entry_offset, which is what makes the recipe an
            # exact-entry identity: without it KernelIdentity.exact is False,
            # Waitcheck collapses every kernel sharing an object into one
            # whole-object scan, and it reports findings with no kernel name --
            # so a helper kernel's finding could be read as the harvested one's.
            entry_offset = _entry_offset(entry.get("kernel_entry"))
            # Byte-identical objects compiled twice are one identity, not two.
            key = (name, digest, index, entry_offset)
            if key in seen:
                continue
            seen.add(key)
            kernels.append(
                {
                    "name": name,
                    "code_object": str(staged),
                    "sha256": digest,
                    "code_object_index": index,
                    "entry_offset": entry_offset,
                    # The cache entry, not the staged copy: the ConSan loader
                    # needs the sidecars Triton wrote beside the object (.json
                    # for the launch metadata, .amdgcn for the kernarg segment
                    # size), and staging copies only the .hsaco.
                    "cache_object": str(obj),
                }
            )

    target = next((t for t in sorted(targets) if t), "gfx950")
    _, _, label = _driver(args)
    inventory_path = dest / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "image": args.image,
                "target": target,
                "selector": {
                    "kernel": args.kernel,
                    "op": args.op,
                    "pytest_suite": args.pytest_suite,
                    "pytest_k": args.pytest_k,
                },
                "dtype": args.dtype,
                "dtype_role": args.dtype_role,
                "kernels": kernels,
            },
            indent=2,
            sort_keys=True,
        )
    )
    recipe_path = _write_recipe(dest, label, target, kernels)

    print(
        f"harvest: {len(kernels)} kernel identit{'y' if len(kernels) == 1 else 'ies'} "
        f"on {target}"
    )
    for kernel in kernels:
        print(f"  {kernel['name']}  {kernel['sha256'][:16]}...  {kernel['code_object']}")
    print(f"harvest: inventory -> {inventory_path}")
    print(f"harvest: recipe    -> {recipe_path}")

    if args.consan:
        loader = (
            Path(args.consan_loader).resolve() if args.consan_loader else _default_consan_loader()
        )
        _write_consan_assets(dest, kernels, target, loader, args.consan_policy, args.consan_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
