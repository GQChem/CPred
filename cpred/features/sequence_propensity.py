"""Category A features: sequence-based propensity scores.

Per Lo et al. 2012 (PLoS ONE), Cat A features are coupled residue propensities
extracted from the ±3-residue window around each position. For each position i
(treated as the CP site), the 7 positions i-3..i+3 are examined and propensity
scores are computed for specific single-residue and coupled-residue patterns.

Features (19 total, matching the 1dprop branch of Figure 4):
  Single AA at center and neighbors:
    R_aa    : propensity(AA at i)
    R_aac3  : mean propensity(AA at i-3, AA at i+3)
    R_aac5  : mean propensity(AA at i-2, AA at i+2)  [within-segment positions ±2]

  Di-residue (coupled pairs):
    2R_aa   : propensity(di(i, i+1))  — adjacent pair at center
    2R_aac3 : propensity(di(i, i+3))  — center + offset-3 pair
    2R_aac5 : propensity(di(i, i+2))  — center + offset-2 pair

  Skip di-residue (paired flanks):
    RxR_aa   : propensity(di(i-1, i+1))  — immediate flanks
    RxR_aac3 : propensity(di(i-3, i+3))  — outer flanks
    RxR_aac5 : propensity(di(i-2, i+2))  — middle flanks

  Longer coupled patterns:
    R2xR_aa   : propensity(di(i-2, i+1))
    R2xR_aac3 : propensity(di(i-3, i+2))
    R2xR_aac5 : propensity(di(i-2, i+3))

    R3xR_aac3 : propensity(di(i-3, i+1))
    R3xR_aac5 : propensity(di(i-1, i+3))

    3R_aac3 : propensity(oligo(i-3, i, i+3))  — triplet at outer positions
    3R_aac5 : propensity(oligo(i-2, i, i+2))  — triplet at middle positions

    4R_aac3 : propensity(oligo(i-3, i-1, i+1, i+3))  skipped to 4mer
    4R_aac5 : propensity(oligo(i-2, i-1, i+1, i+2))

    5R_aac3 : propensity(oligo(i-3, i-1, i, i+1, i+3))
"""

from __future__ import annotations

import numpy as np

from cpred.propensity.tables import PropensityTables


def _get_aa(sequence: str, idx: int) -> str | None:
    """Return the amino acid at idx, or None if out of bounds."""
    if 0 <= idx < len(sequence):
        return sequence[idx]
    return None


def _single_prop(tables: PropensityTables, aa: str | None) -> float:
    if aa is None:
        return 0.0
    return tables.get("single_aa", aa)


def _di_prop(tables: PropensityTables, aa1: str | None, aa2: str | None) -> float:
    if aa1 is None or aa2 is None:
        return 0.0
    return tables.get("di_residue", aa1 + aa2)


def _oligo_prop(tables: PropensityTables, *aas: str | None) -> float:
    if any(a is None for a in aas):
        return 0.0
    return tables.get("oligo_residue", "".join(aas))  # type: ignore[arg-type]


def extract_sequence_propensity_features(
        sequence: str, tables: PropensityTables) -> dict[str, np.ndarray]:
    """Extract all Category A coupled-residue propensity features.

    For each residue position i, computes propensity scores for patterns
    within the ±3 window, matching the '1dprop' branch of Figure 4.

    Returns:
        Dict mapping feature name to (N,) arrays (19 features).
    """
    n = len(sequence)
    features: dict[str, list[float]] = {
        "R_aa": [], "R_aac3": [], "R_aac5": [],
        "2R_aa": [], "2R_aac3": [], "2R_aac5": [],
        "RxR_aa": [], "RxR_aac3": [], "RxR_aac5": [],
        "R2xR_aa": [], "R2xR_aac3": [], "R2xR_aac5": [],
        "R3xR_aac3": [], "R3xR_aac5": [],
        "3R_aac3": [], "3R_aac5": [],
        "4R_aac3": [], "4R_aac5": [],
        "5R_aac3": [],
    }

    for i in range(n):
        aa = _get_aa(sequence, i)
        am3 = _get_aa(sequence, i - 3)
        am2 = _get_aa(sequence, i - 2)
        am1 = _get_aa(sequence, i - 1)
        ap1 = _get_aa(sequence, i + 1)
        ap2 = _get_aa(sequence, i + 2)
        ap3 = _get_aa(sequence, i + 3)

        # Single AA features
        features["R_aa"].append(_single_prop(tables, aa))
        features["R_aac3"].append(
            (_single_prop(tables, am3) + _single_prop(tables, ap3)) / 2
        )
        features["R_aac5"].append(
            (_single_prop(tables, am2) + _single_prop(tables, ap2)) / 2
        )

        # Di-residue features (center + neighbor)
        features["2R_aa"].append(_di_prop(tables, aa, ap1))
        features["2R_aac3"].append(_di_prop(tables, aa, ap3))
        features["2R_aac5"].append(_di_prop(tables, aa, ap2))

        # Skip di-residue (paired flanks)
        features["RxR_aa"].append(_di_prop(tables, am1, ap1))
        features["RxR_aac3"].append(_di_prop(tables, am3, ap3))
        features["RxR_aac5"].append(_di_prop(tables, am2, ap2))

        # Longer coupled patterns
        features["R2xR_aa"].append(_di_prop(tables, am2, ap1))
        features["R2xR_aac3"].append(_di_prop(tables, am3, ap2))
        features["R2xR_aac5"].append(_di_prop(tables, am2, ap3))

        features["R3xR_aac3"].append(_di_prop(tables, am3, ap1))
        features["R3xR_aac5"].append(_di_prop(tables, am1, ap3))

        # Triplet features
        features["3R_aac3"].append(_oligo_prop(tables, am3, aa, ap3))
        features["3R_aac5"].append(_oligo_prop(tables, am2, aa, ap2))

        # 4-mer features (skip center)
        features["4R_aac3"].append(_oligo_prop(tables, am3, am1, ap1, ap3))
        features["4R_aac5"].append(_oligo_prop(tables, am2, am1, ap1, ap2))

        # 5-mer
        features["5R_aac3"].append(_oligo_prop(tables, am3, am1, aa, ap1, ap3))

    return {k: np.array(v) for k, v in features.items()}
