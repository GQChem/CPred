"""Window averaging and window expansion for features.

Applies ±W residue window averaging (default W=3, 7-residue window).
For terminal residues, uses available neighbors only.

Also supports window expansion: for each residue, output the feature value
at each window position (i-W, ..., i, ..., i+W) as separate features.
This is used for propensity features to produce 46 total features per the paper.
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


def window_expand(values: np.ndarray, w: int = DEFAULT_WINDOW) -> np.ndarray:
    """Expand a 1D feature array into window-position features.

    For each residue i, output the values at positions i-W..i+W as
    separate columns. Terminal positions are padded with the nearest
    available value (edge padding).

    Args:
        values: (N,) feature values.
        w: Half-window size (default 3).

    Returns:
        (N, 2*w+1) array where column j holds values at offset j-w.
    """
    n = len(values)
    width = 2 * w + 1
    result = np.zeros((n, width))
    for j in range(width):
        offset = j - w
        for i in range(n):
            src = i + offset
            src = max(0, min(n - 1, src))  # edge padding
            val = values[src]
            result[i, j] = 0.0 if np.isnan(val) else val
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


def window_expand_dict(features: dict[str, np.ndarray],
                       w: int = DEFAULT_WINDOW) -> dict[str, np.ndarray]:
    """Expand each feature into per-window-position features.

    For feature "foo", produces "foo_m3", "foo_m2", ..., "foo_0", ..., "foo_p3".

    Args:
        features: Dict mapping feature name to (N,) arrays.
        w: Half-window size.

    Returns:
        Dict with expanded feature names and (N,) arrays.
    """
    result = {}
    for name, vals in features.items():
        expanded = window_expand(vals, w)
        width = 2 * w + 1
        for j in range(width):
            offset = j - w
            if offset < 0:
                suffix = f"_m{abs(offset)}"
            elif offset == 0:
                suffix = "_0"
            else:
                suffix = f"_p{offset}"
            result[f"{name}{suffix}"] = expanded[:, j]
    return result
