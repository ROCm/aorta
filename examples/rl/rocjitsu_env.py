"""Compatibility alias for :mod:`aorta.rocjitsu.environment`."""

from __future__ import annotations

import sys

from aorta.rocjitsu import environment as _environment

sys.modules[__name__] = _environment
