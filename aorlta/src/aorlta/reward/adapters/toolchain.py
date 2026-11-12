"""The identity of the compiler a `hipcc` runs, for a checker's version: a cache entry made with one compiler is not
reused with another.

`hipcc` is a driver. The clang it runs, and the flags it adds, come from its environment (HIP_CLANG_PATH, ROCM_PATH,
HIPCC_COMPILE_FLAGS_APPEND, ...) or from the ROCm tree it is installed in, so the driver's own bytes do not say which
compiler built an object. The identity covers the driver, the environment variables hipcc reads, and the clang and its
configuration files in every directory hipcc could take it from. It is read from files, never by running hipcc, so a
checker that is only constructed starts no compiler.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

HIPCC_ENV = ("HIP_CLANG_PATH", "HIP_PATH", "ROCM_PATH", "HIP_PLATFORM", "HIP_COMPILER", "HIP_RUNTIME", "HIP_ROCCLR_HOME",
             "HIP_LIB_PATH", "HIP_INCLUDE_PATH", "HIP_DEVICE_LIB_PATH", "DEVICE_LIB_PATH", "HIP_CLANG_HCC_COMPAT_MODE",
             "HIPCC_COMPILE_FLAGS_APPEND", "HIPCC_LINK_FLAGS_APPEND")


@functools.lru_cache(maxsize=64)
def _file_sha256(real: str, size: int, mtime_ns: int) -> str:  # keyed on size and mtime too: a rewrite is a new file
    h = hashlib.sha256()
    with open(real, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(path: str | Path) -> str:
    try:
        real = os.path.realpath(path)
        st = os.stat(real)
        return _file_sha256(real, st.st_size, st.st_mtime_ns)[:16]
    except OSError:
        return "missing"


def _llvm_dirs(hipcc: str, env: Mapping[str, str]) -> list[str]:
    dirs = [env.get("HIP_CLANG_PATH", "")]
    dirs += [os.path.join(env[k], "lib", "llvm", "bin") for k in ("ROCM_PATH", "HIP_PATH") if env.get(k)]
    dirs.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(hipcc))), "lib", "llvm", "bin"))
    seen: list[str] = []
    for d in dirs:
        if d and os.path.realpath(d) not in seen:
            seen.append(os.path.realpath(d))
    return seen


def toolchain_identity(hipcc: str, env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    parts: dict = {"hipcc": file_identity(hipcc), "env": {k: env[k] for k in HIPCC_ENV if k in env}, "llvm": {}}
    for d in _llvm_dirs(hipcc, env):
        try:
            cfgs = sorted(n for n in os.listdir(d) if n.endswith(".cfg"))
        except OSError:
            cfgs = []
        parts["llvm"][d] = {n: file_identity(os.path.join(d, n)) for n in ("clang", "clang++", *cfgs)}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
