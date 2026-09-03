#!/usr/bin/env python3
"""Install the Docker CLI -- the client only -- into a CI image.

Run from docker/Dockerfile.ci-gpu. See the block that invokes it there for why
the image needs a docker client at all; this file is only concerned with getting
one in place reproducibly.

Kept in the image after use, like ``rocm_layout_guard.py`` next to it: re-running
it is the fastest way to repair or re-verify a client on a runner, and it records
in the image itself which artifact the binary came from.

Deliberately stdlib-only and downloader-free. The base image guarantees python
(the layout guard, the ROCm fixup blocks and the pip pins all run it) but
promises neither curl nor wget, so shelling out to one would add a dependency
this image does not have. Stdlib-only also means the Ubuntu 26.04 / py3.14 base
this now sits on moved it for free, which a pinned third-party dependency would
not have.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import urllib.request

# Pinned by version AND by the tarball's sha256, the same way the base image is
# pinned by manifest digest: the static channel keeps old versions around, but a
# bare URL still names a moving target the moment the version is bumped without
# review. Re-resolve both together when bumping:
#
#   curl -fsSL -O https://download.docker.com/linux/static/stable/x86_64/docker-<v>.tgz
#   sha256sum docker-<v>.tgz
VERSION = "29.7.2"
SHA256 = "803d433f226db4776e1768fd319fc6c6e4935a456acf84fcc0080818b854bc8f"

URL = f"https://download.docker.com/linux/static/stable/x86_64/docker-{VERSION}.tgz"

# The one member extracted. The same tarball also ships dockerd, containerd,
# containerd-shim, runc, docker-init and docker-proxy; this image wants a client
# that talks to the *host* daemon over a bind-mounted socket, not a daemon of its
# own, so naming the member explicitly is what keeps a daemon out of the image.
MEMBER = "docker/docker"
DEST = "/usr/local/bin/docker"


def main() -> int:
    with urllib.request.urlopen(URL, timeout=180) as response:
        blob = response.read()

    # Verify before extracting, not after installing: a truncated or substituted
    # download should fail the build, not leave a partial binary on PATH for the
    # next layer to find.
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        print(
            f"{URL}\n  sha256 {digest}\n  expected {SHA256}",
            file=sys.stderr,
        )
        return 1

    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        extracted = tar.extractfile(tar.getmember(MEMBER))
        if extracted is None:
            print(f"{MEMBER} is not a regular file in {URL}", file=sys.stderr)
            return 1
        with open(DEST, "wb") as target:
            target.write(extracted.read())

    os.chmod(DEST, 0o755)
    print(f"installed {MEMBER} from docker-{VERSION}.tgz to {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
