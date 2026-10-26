"""Pin ``docker/Dockerfile.asan-pytorch`` to its checksum file.

The image fetches the four ROCm 7.0.2 ASan runtime .deb packages over the network
(they are not in the base image and cannot be committed to the build context), so
the only thing standing between the build and an arbitrary payload is the
``sha256sum -c`` against ``docker/asan-pytorch/SHA256SUMS``. Two ways that gate
rots silently: a deb version is bumped in the Dockerfile without updating the
checksums (then every build fails closed -- annoying but safe), or the
verification is reordered after extraction (then it fails *open* -- a bad deb is
unpacked before anyone checks it). This test is CPU-only and catches both without
building the image.

It also guards the two properties the PR exists for: aorta is installed from the
build context (so the image matches the commit), not a pinned git URL; and the
device build stays xnack+ with ASan flags (a plain gfx950 build links fine and
checks nothing).
"""

from __future__ import annotations

import re
from pathlib import Path

_DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
_DOCKERFILE = _DOCKER_DIR / "Dockerfile.asan-pytorch"
_SHA256SUMS = _DOCKER_DIR / "asan-pytorch" / "SHA256SUMS"

_EXPECTED_DEBS = {
    "comgr-asan",
    "hip-runtime-amd-asan",
    "hsa-rocr-asan",
    "rocm-core-asan",
}


def _checksums() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in _SHA256SUMS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split()
        entries[name] = digest
    return entries


def test_sha256sums_lists_exactly_the_four_pinned_asan_debs() -> None:
    entries = _checksums()
    pkgs = {name.split("_", 1)[0] for name in entries}
    assert pkgs == _EXPECTED_DEBS
    for name, digest in entries.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), f"not a sha256: {name}"
        # Pinned to the ROCm 7.0.2 (-56~22.04) -asan build for jammy/amd64.
        assert "70002-56~22.04_amd64.deb" in name, f"not a pinned 7.0.2 deb: {name}"


def test_dockerfile_fetches_exactly_the_checksummed_filenames() -> None:
    text = _DOCKERFILE.read_text()
    deb_args = dict(re.findall(r"^ARG\s+(\w*_DEB)=(\S+)\s*$", text, re.MULTILINE))
    fetched = set(deb_args.values())
    assert fetched == set(_checksums()), (
        "the deb filenames the Dockerfile downloads must match SHA256SUMS exactly; "
        "bumping a version in one place without the other breaks the build"
    )


def test_checksum_is_verified_before_any_deb_is_extracted() -> None:
    # Fail-closed: the verification must precede extraction, or a tampered deb is
    # unpacked before it is checked.
    text = _DOCKERFILE.read_text()
    verify = text.index("sha256sum -c SHA256SUMS")
    extract = text.index("dpkg -x")
    assert verify < extract, "sha256sum -c must run before dpkg -x"


def test_aorta_is_installed_from_the_build_context_not_a_git_url() -> None:
    text = _DOCKERFILE.read_text()
    assert "git+https://github.com/ROCm/aorta" not in text
    assert "COPY src /tmp/aorta-src/src/" in text
    assert "/tmp/aorta-src" in text and "pip install" in text


def _env_value(text: str, name: str) -> list[str]:
    # The ENV/ARG assignment itself, not any mention: the compile and link flag
    # variables share options, so a whole-file search cannot tell them apart.
    match = re.search(rf'^\s*(?:ENV\s+)?{name}="([^"]*)"', text, re.MULTILINE)
    assert match, f"no {name}=\"...\" assignment in {_DOCKERFILE.name}"
    return match.group(1).split()


def test_device_build_is_xnack_plus_with_asan() -> None:
    text = _DOCKERFILE.read_text()
    assert _env_value(text, "PYTORCH_ROCM_ARCH") == ["gfx950:xnack+"]
    compile_flags = _env_value(text, "HIPCC_COMPILE_FLAGS_APPEND")
    assert "-fsanitize=address" in compile_flags, "device code would be built uninstrumented"
    assert "-fno-gpu-sanitize" not in compile_flags
    assert "-fsanitize=address" in _env_value(text, "HIPCC_LINK_FLAGS_APPEND")


_CAPABILITY_DOC = _DOCKER_DIR.parent / "docs" / "sanitizers" / "asan-gpu-capability.md"


def test_non_root_instructions_also_select_a_writable_working_directory() -> None:
    # `aorta run` writes ./results and the image's working directory is the root-owned /,
    # so `--user` without `-w` fails with PermissionError: '/results'.
    for path in (_DOCKERFILE, _CAPABILITY_DOC):
        lines = [ln for ln in path.read_text().splitlines() if "--user $(id -u)" in ln]
        assert lines, f"{path.name} no longer documents the non-root invocation"
        for line in lines:
            assert re.search(r"(^|\s)-w\s+\S", line), f"{path.name}: --user without -w: {line!r}"


def test_torch_wheel_is_bind_mounted_not_copied_into_a_layer() -> None:
    # A COPY layer keeps the ~240 MB wheel in the image even after the install deletes it.
    text = _DOCKERFILE.read_text()
    assert not re.search(r"^\s*COPY\s+--from=build\b", text, re.MULTILINE)
    mount = re.search(
        r"^RUN --mount=type=bind,from=build,source=/opt/pytorch/dist,target=(\S+)\s*\\$(.*?)(?=^\S)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert mount, "the torch wheel must be installed from a bind mount of the build stage"
    assert f"{mount.group(1)}/torch-*.whl" in mount.group(2)
    # Unquoted, so the shell expands it; pip rejects a literal 'torch-*.whl'.
    assert "'" not in mount.group(2) and '"' not in mount.group(2)


def test_aorta_version_advice_is_pep440() -> None:
    # AORTA_VERSION becomes SETUPTOOLS_SCM_PRETEND_VERSION, which must be PEP 440;
    # `git describe` output (v0.3.0-340-gabc1234) fails the install with InvalidVersion.
    text = _DOCKERFILE.read_text()
    assert "AORTA_VERSION=$(python -m setuptools_scm)" in text
    assert "(e.g. `git describe`)" not in text


def test_overflow_example_keeps_the_backing_tensor_alive() -> None:
    # The run mode disables the caching allocator, so data_ptr() of a temporary points at
    # storage freed before the view is used: the run aborts with no ASan report.
    text = _DOCKERFILE.read_text()
    assert not re.search(r"torch\.\w+\([^)]*\)\.data_ptr\(", text)
    assert "base.data_ptr()" in text and "torch.cuda.synchronize()" in text
