"""Fresh-import checks for the lazy public exports in ``aorta.race.modes``.

Run in subprocesses so other race tests cannot populate the module state and
hide missing names from introspection (issue #448). No GPU work is performed.
"""

import subprocess
import sys
import textwrap

import pytest


def _probe(source: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_dir_advertises_exports_without_loading_modes() -> None:
    _probe("""
        import sys
        import aorta.race.modes as modes

        implementations = {
            'aorta.race.modes.default',
            'aorta.race.modes.ddp',
            'aorta.race.modes.fsdp',
        }
        assert implementations.isdisjoint(sys.modules)
        names = dir(modes)
        assert set(modes.__all__) <= set(names), set(modes.__all__) - set(names)
        assert set(vars(modes)) <= set(names)
        assert names == sorted(set(names))
        assert implementations.isdisjoint(sys.modules)
        """)


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("DefaultModeReproducer", "default"),
        ("DDPModeReproducer", "ddp"),
        ("FSDPModeReproducer", "fsdp"),
    ],
)
def test_lazy_exports_still_resolve(name: str, module: str) -> None:
    _probe(f"""
        import importlib
        import aorta.race.modes as modes

        exported = getattr(modes, {name!r})
        implementation = importlib.import_module('aorta.race.modes.' + {module!r})
        assert exported is getattr(implementation, {name!r})
        """)


def test_unknown_attribute_still_raises_attribute_error() -> None:
    _probe("""
        import aorta.race.modes as modes

        try:
            modes.no_such_reproducer
        except AttributeError as exc:
            assert 'no_such_reproducer' in str(exc)
        else:
            raise AssertionError('Unknown attribute did not raise AttributeError')
        """)
