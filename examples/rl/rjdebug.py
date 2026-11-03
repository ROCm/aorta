#!/usr/bin/env python3
"""Compatibility entry point for :mod:`aorta.rocjitsu.rjdebug`."""

from __future__ import annotations

import sys

from aorta.rocjitsu import rjdebug as _rjdebug

if __name__ == "__main__":
    raise SystemExit(_rjdebug.main())

sys.modules[__name__] = _rjdebug
