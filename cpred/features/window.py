"""Window averaging for features.

Applies ±W residue window averaging (default W=3, 7-residue window).
For terminal residues, uses available neighbors only.
"""

from __future__ import annotations

import numpy as np

DEFAULT_WINDOW = 3  # ±3 residues


def window_average(values: np.ndarray, w: int = DEFAULT_WINDOW) -> np.ndarray:
    """Apply window averaging to a 1D feature array.

    For position i, the averaged value is the mean of
    values[max(0, i-w) : min(n, i+w+1)].

    Args:
        values: (N,) feature values.
        w: Half-window size (default 3).

    Returns:
        (N,) window-averaged values.
    """
    n = len(values)
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - w)
        end = min(n, i + w + 1)
        window = values[start:end]
        # Handle NaN values
        valid = window[~np.isnan(window)]
        result[i] = valid.mean() if len(valid) > 0 else np.nan
    return result


def window_average_dict(features: dict[str, np.ndarray],
                        w: int = DEFAULT_WINDOW) -> dict[str, np.ndarray]:
    """Apply window averaging to all features in a dictionary.

    Args:
        features: Dict mapping feature name to (N,) arrays.
        w: Half-window size.

    Returns:
        Dict with same keys, window-averaged values.
    """
    return {name: window_average(vals, w) for name, vals in features.items()}
