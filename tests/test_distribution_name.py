"""Guards the amd-aorta distribution rename (PR #243).

The PyPI name ``aorta`` belongs to an unrelated project, so AORTA ships as
``amd-aorta`` (the import package stays ``aorta``). The self-referential
extras in ``[project.optional-dependencies]`` must therefore point at
``amd-aorta[...]`` -- a stray bare ``aorta[...]`` would silently resolve to
the wrong PyPI project for ``pip install amd-aorta[all]``.

Kept stdlib-regex-only (like ``scripts/bump_version.py``): there is no
pytest CI job and ``tomli`` is not a declared dependency, so the test must
not import a TOML parser to run on a bare ``requires-python >= 3.10`` env.
"""

import re
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"
DISTRIBUTION = "amd-aorta"

# A self-referential extras requirement on the bare ``aorta`` distribution,
# e.g. "aorta[all]" but not "amd-aorta[all]".
BARE_SELF_EXTRA = re.compile(r'(?<![\w-])aorta\[')


def _read():
    return PYPROJECT.read_text(encoding="utf-8")


def _optional_dependencies_block(text):
    """The text of the [project.optional-dependencies] table only."""
    start = text.index("[project.optional-dependencies]")
    rest = text[start:]
    # The header's own '[' has no preceding newline in rest, so the next
    # "\n[" match is the following table -- i.e. the end of this block.
    end = re.search(r"\n\[", rest)
    return rest if end is None else rest[: end.start()]


def test_distribution_is_amd_aorta():
    name = re.search(r'(?m)^name\s*=\s*"([^"]+)"', _read())
    assert name is not None and name.group(1) == DISTRIBUTION


def test_self_referential_extras_use_amd_aorta():
    block = _optional_dependencies_block(_read())
    offenders = BARE_SELF_EXTRA.findall(block)
    assert not offenders, (
        "self-referential extras must use 'amd-aorta[...]', not bare "
        f"'aorta[...]' in [project.optional-dependencies]: found {len(offenders)}"
    )


def test_import_package_dir_still_aorta():
    assert (Path(__file__).parent.parent / "src" / "aorta" / "__init__.py").exists()
