"""Feature standardization: inversion and Z-score normalization.

Some features have inverted polarity (low value = more viable for CP).
These are inverted before Z-score normalization so that higher values
consistently indicate higher CP viability.

Features to invert (per Table S2 in Lo et al. 2012):
  - CN, WCN, closeness: packed environment → disfavored
  - H-bonds: more H-bonds → disfavored (buried)
  - Depth (DPX): deeper → disfavored
  - DIS_b, DIS_hpho: average distance to buried/hydrophobic (lower → closer to core)

Features NOT inverted (high = more viable):
  - RSA: high solvent accessibility → favored
  - CM: far from center → favored
  - B-factor, GNM-F: high flexibility → favored
  - Farness (Fb, Fhpho, F_union, F_inter): high farness from core → favored
"""

from __future__ import annotations

import numpy as np

# Features where low value correlates with CP viability (need inversion)
# These are base names — window-expanded variants (e.g. closeness_m3) also match.
_INVERT_BASE = {
    "cn", "wcn", "closeness", "hbond",
    "depth",
    "dis_b", "dis_hpho",
}

# Features that should NOT be inverted (high = more viable):
# rsa, cm, gnm_msf, farness_buried, farness_hydrophobic,
# farness_union, farness_inter


def _should_invert(name: str) -> bool:
    """Check if a feature name should be inverted."""
    return name in _INVERT_BASE


FEATURES_TO_INVERT = _INVERT_BASE


def invert_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Invert features where low value = more viable.

    Inversion: x_inv = -x (simple negation).

    Args:
        features: Dict of feature name -> (N,) arrays.

    Returns:
        Dict with inverted features replaced.
    """
    result = {}
    for name, vals in features.items():
        if _should_invert(name):
            result[name] = -vals
        else:
            result[name] = vals.copy()
    return result


def zscore_normalize(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Z-score normalize each feature per protein.

    z_i = (x_i - mean) / std

    Features with zero variance get all-zero values.

    Args:
        features: Dict of feature name -> (N,) arrays.

    Returns:
        Dict with Z-score normalized values.
    """
    result = {}
    for name, vals in features.items():
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            result[name] = np.zeros_like(vals)
            continue
        mu = valid.mean()
        sigma = valid.std()
        if sigma < 1e-10:
            result[name] = np.zeros_like(vals)
        else:
            normalized = (vals - mu) / sigma
            # Replace NaN with 0 after normalization
            normalized = np.where(np.isnan(normalized), 0.0, normalized)
            result[name] = normalized
    return result


def standardize_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Full standardization: invert then Z-score normalize."""
    inverted = invert_features(features)
    return zscore_normalize(inverted)
