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

#: Everything that ships and can carry a site's assumptions. The chat tools
#: reach the same cluster as the agents and are written the same way, so
#: guarding only the agents leaves half the surface unguarded; the directory is
#: absent until the PR that adds those tools, and skipped until then.
_GUARDED = ("src/aorta/cia", "src/aorta/chat/tools")

#: Shapes that mean "somebody's machine", not "anybody's machine".
_SITE_SHAPED = (
    # No trailing character class: the value that prompted this test was
    # ``f"/apps/{os.environ.get('USER')}"``, where the next character is a
    # brace. Requiring a letter there would have missed it.
    (re.compile(r"/apps/"), "an absolute path under /apps"),
    (re.compile(r"/home/"), "an absolute path under /home"),
    # cv350-rck-g03 and the like.
    (re.compile(r"\b[a-z]{2,}\d{2,}-[a-z]{2,}\d?-[a-z]\d{2}"), "a cluster hostname"),
    # A bare address. The hostname shape above needs hyphens, so a literal IP
    # walked straight past it -- one sat in a shipped default and another in a
    # comment, in a public repository. The lookbehind keeps version strings
    # (rocm-7.0.2.2) from reading as addresses, and the lookahead lets through
    # the ones that mean the same thing on every machine.
    (
        re.compile(
            r"(?<![\w.-])(?!127\.0\.0\.1|0\.0\.0\.0|255\.255\.255\.255)"
            r"\d{1,3}(?:\.\d{1,3}){3}(?![\w.])"
        ),
        "an IP address",
    ),
    # A toolchain pinned to one installed version. /opt/rocm is everywhere;
    # /opt/rocm-7.0.2.2 is one machine's, and a default naming it fails on any
    # box with a different patch release -- by looking for a compiler that is
    # not there, which reads as "no ROCm" rather than "wrong path".
    (re.compile(r"/opt/[a-z]+-\d+(?:\.\d+)+"), "a version-pinned toolchain path"),
    # Which partition is hardcoded matters less than that one is: an installed
    # default no account can submit to fails the same way whatever it is named.
    (re.compile(r"--partition=[A-Za-z]"), "a hardcoded Slurm partition"),
    (re.compile(r'"(interactive|meta\d+)"'), "a Slurm partition name"),
)

#: Paths that are the same everywhere, so naming them is not an assumption.
_UNIVERSAL = re.compile(r"^/(dev|proc|sys|tmp|etc|usr|bin|opt)/")

#: A hostname with no hyphens to give it away, like chi2878.
_BARE_HOSTNAME = re.compile(r"\b[a-z]{2,}\d{3,}\b")

#: Hardware names shaped exactly like that and meaning the opposite: gfx950 is
#: an architecture, and every MI355X in the world is an MI355X. Removed from
#: the line before asking about hostnames, which is what lets the pattern above
#: want three digits rather than four -- chi287 is a node, gfx950 is not.
_HARDWARE_TOKEN = re.compile(r"\b(?:gfx\d+[a-z]*|mi\d+[a-z]*|sm_?\d+|navi\d+)\b", re.I)


def _offences(line: str) -> list[str]:
    """What is site-specific about *line*, if anything."""
    if _UNIVERSAL.search(line):
        return []
    found = [what for pattern, what in _SITE_SHAPED if pattern.search(line)]
    if _BARE_HOSTNAME.search(_HARDWARE_TOKEN.sub(" ", line)):
        found.append("a bare cluster hostname")
    return found


def _source_files(repo_root):
    return sorted(
        p
        for directory in _GUARDED
        for p in (repo_root / directory).rglob("*.py")
        if "__pycache__" not in p.parts
    )


def test_no_site_specific_value_is_hardcoded(repo_root):
    offenders: list[str] = []
    for path in _source_files(repo_root):
        rel = path.relative_to(repo_root)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for what in _offences(line):
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
