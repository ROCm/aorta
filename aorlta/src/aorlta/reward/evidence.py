"""Keeping the GPU legs' evidence (run 2, step 4): the reward service's packages live in the run's node-local scratch,
which the orchestrator removes at shutdown, so what explains an unsettled ConSan or ASan leg would go with it (run 1's
evidence archive came out empty). `archive_legs` copies the sanitizer legs' logs, generated harness sources and JSON,
and each candidate's parsed source, into one tarball in the run's sealed evidence store (Section 6.10 item 3) before
the scratch goes. Package layout: `pkg/<sha[:2]>/<sha>/<leg>-<key[:16]>/...`.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

SUFFIXES = (".log", ".json", ".jsonl", ".hip")
MAX_FILE_BYTES = 2 << 20


def kept(rel: Path) -> bool:
    """Whether a file under the package root is evidence: a sanitizer leg's log, JSON or harness source, or the
    source the parse gate wrote."""
    parts = rel.parts
    if len(parts) < 4:
        return False
    leg = parts[2]
    return (leg.startswith("san.") and rel.suffix in SUFFIXES) or (leg.startswith("gate.parse-")
                                                                    and rel.name == "source.hip")


def archive_legs(pkg_root: str | Path, dest: str | Path, *, max_file_bytes: int = MAX_FILE_BYTES) -> int:
    """Write the evidence under `pkg_root` to the tarball `dest` (atomically: `dest.partial`, then renamed); returns
    the number of files kept. Nothing is written when there is no package root."""
    root, dest = Path(pkg_root), Path(dest)
    if not root.is_dir():
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    n = 0
    with tarfile.open(tmp, "w:gz") as tar:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if p.is_file() and not p.is_symlink() and kept(rel) and p.stat().st_size <= max_file_bytes:
                tar.add(p, arcname=str(rel))
                n += 1
    tmp.replace(dest)
    return n


def next_archive(directory: str | Path, stem: str = "gpu-legs") -> Path:
    """`<directory>/<stem>-<k>.tgz` for the first k not taken, so a resumed run never overwrites an earlier
    attempt's archive."""
    d = Path(directory)
    k = 1
    while (d / f"{stem}-{k}.tgz").exists():
        k += 1
    return d / f"{stem}-{k}.tgz"
