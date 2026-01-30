"""
Shared utility functions for the aorta-report CLI.
"""

import numpy as np


def geometric_mean(values: np.ndarray) -> float:
    """
    Calculate geometric mean, handling zeros.

    Args:
        values: Array of values

    Returns:
        Geometric mean value
    """
    values = np.array(values)
    values = np.where(values == 0, 1e-10, values)
    return float(np.exp(np.mean(np.log(values))))
