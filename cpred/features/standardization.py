"""Feature standardization: inversion and Z-score normalization.

Per Table S2 in Lo et al. 2012, each feature has an "Applied form":
  M   = raw measure (positive correlation with viability)
  -M  = negated (negative correlation inverted to positive)
  1/M = reciprocal (packed/crowded measures inverted)

Applied forms per included feature:
  RSA+      : M    (high accessibility → favored)
  DPX+      : -M   (deep → disfavored, negate)
  CM+       : M    (far from center → favored)
  H-bonds+  : -M   (more H-bonds → buried → disfavored, negate)
  Closeness : 1/M  (high closeness → packed → disfavored, reciprocal)
  CN        : -M   (high CN → packed → disfavored, negate)
  WCN       : 1/M  (high WCN → packed → disfavored, reciprocal)
  GNM-F     : M    (high flexibility → favored)
  Disb+     : M    (far from buried → favored)
  Dishpho   : M    (far from hydrophobic → favored)
  Fb+       : M    (high farness → favored)
  Fhpho     : M    (high farness → favored)
"""

from __future__ import annotations

import numpy as np

# Features that need negation (-M form per Table S2)
_NEGATE_FEATURES = {"cn", "hbond", "depth"}

# Features that need reciprocal (1/M form per Table S2)
_RECIPROCAL_FEATURES = {"closeness", "wcn"}

# All other Cat C features use M (raw, not inverted)


def _should_negate(name: str) -> bool:
    return name in _NEGATE_FEATURES


def _should_reciprocal(name: str) -> bool:
    return name in _RECIPROCAL_FEATURES


FEATURES_TO_INVERT = _NEGATE_FEATURES | _RECIPROCAL_FEATURES


def invert_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply feature transformations per Table S2.

    -M features: simple negation.
    1/M features: reciprocal (with epsilon to avoid division by zero).

    Args:
        features: Dict of feature name -> (N,) arrays.

    Returns:
        Dict with transformed features.
    """
    result = {}
    for name, vals in features.items():
        if _should_negate(name):
            result[name] = -vals
        elif _should_reciprocal(name):
            # Reciprocal with small epsilon to avoid division by zero
            safe_vals = np.where(np.abs(vals) < 1e-10, 1e-10, vals)
            result[name] = 1.0 / safe_vals
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
