"""C9 measurement tools (design Section 6.9): timeline, per-256 metric, verify70,
compute curves, and the comparison report."""


class MeasureError(ValueError):
    """Input that the measurement cannot use as it stands. Never turned into a default."""
