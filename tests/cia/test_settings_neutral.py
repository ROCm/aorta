"""No site's layout may be a default.

This code arrived from an internal repository where one filesystem layout, one
node naming scheme and one Slurm partition were safe to assume. None of them
are here. The failure mode is quiet: a wrong default does not raise, it just
searches a directory that does not exist, or submits to a partition the account
cannot use, and the user is left reading an empty result.

Sites say where their things are through the environment. The defaults say as
little as possible.
"""

from __future__ import annotations

import re

import pytest

_CIA = "src/aorta/cia"

#: Shapes that mean "somebody's machine", not "anybody's machine".
_SITE_SHAPED = (
    (re.compile(r"/apps/[a-z]"), "an absolute path under /apps"),
    (re.compile(r"/home/[a-z]"), "an absolute path under /home"),
    (re.compile(r"\b[a-z]{2,}\d{2,}-[a-z]{2,}\d?-[a-z]\d{2}"), "a cluster hostname"),
    (re.compile(r'"(interactive|meta\d+)"'), "a Slurm partition name"),
)

#: Paths that are the same everywhere, so naming them is not an assumption.
_UNIVERSAL = re.compile(r"^/(dev|proc|sys|tmp|etc|usr|bin|opt)/")


def _source_files(repo_root):
    return sorted(
        p for p in (repo_root / _CIA).rglob("*.py") if "__pycache__" not in p.parts
    )


def test_no_site_specific_value_is_hardcoded(repo_root):
    offenders: list[str] = []
    for path in _source_files(repo_root):
        rel = path.relative_to(repo_root)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _UNIVERSAL.search(line):
                continue
            for pattern, what in _SITE_SHAPED:
                if pattern.search(line):
                    offenders.append(f"{rel}:{lineno}: {what}: {line.strip()[:70]}")
    assert not offenders, (
        "site-specific values in shipped code:\n  " + "\n  ".join(offenders)
        + "\n\nMake it configurable with a neutral default instead."
    )


def test_the_jobs_root_defaults_under_the_home_directory(monkeypatch):
    """Somewhere every user has, rather than somewhere one site has."""
    from aorta.cia.launch.cluster import default_jobs_root

    monkeypatch.delenv("CIA_JOBS_ROOT", raising=False)
    assert default_jobs_root().startswith("/")
    assert "/apps/" not in default_jobs_root()


def test_search_roots_guesses_only_the_home_directory(monkeypatch):
    """It used to also guess ``/apps/$USER``, which exists at exactly one site."""
    from aorta.cia.launch.cluster import search_roots

    monkeypatch.delenv("CIA_SEARCH_ROOTS", raising=False)
    assert not any("/apps/" in root for root in search_roots())


def test_search_roots_is_overridable(monkeypatch, tmp_path):
    """The escape hatch that makes the narrow default acceptable."""
    from aorta.cia.launch.cluster import search_roots

    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("CIA_SEARCH_ROOTS", f"{first}:{second}")
    assert search_roots() == [str(first), str(second)]


@pytest.mark.parametrize("variable", ["CIA_PARTITION", "CIA_CONTAINER_IMAGE", "CIA_SBATCH_EXTRA"])
def test_scheduler_knobs_default_to_unset(monkeypatch, variable):
    """An unset knob adds no directive; a guessed one adds a wrong directive."""
    from aorta.cia.launch.cluster import build_sbatch_script

    monkeypatch.delenv(variable, raising=False)
    script = build_sbatch_script(
        command="true", job_name="j", log_path="/tmp/j.log"
    )
    assert "--partition=" not in script
    assert "docker run" not in script
