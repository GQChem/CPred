"""Category A features: sequence-based propensity scores.

For each residue position, look up propensity scores in ±3 window for:
  - Single amino acid propensity
  - Di-residue propensity
  - Oligo-residue (triplet) propensity

Returns window-averaged propensity values.
"""

from __future__ import annotations

import numpy as np

from cpred.propensity.tables import PropensityTables
from cpred.features.window import DEFAULT_WINDOW


def compute_single_aa_propensity(sequence: str,
                                  tables: PropensityTables) -> np.ndarray:
    """Look up single amino acid propensity at each position."""
    return np.array([tables.get("single_aa", aa) for aa in sequence])


def compute_di_residue_propensity(sequence: str,
                                   tables: PropensityTables) -> np.ndarray:
    """Look up di-residue propensity at each position.

    For position i, uses the di-residue (i, i+1). Last residue gets 0.
    """
    n = len(sequence)
    scores = np.zeros(n)
    for i in range(n - 1):
        di = sequence[i:i + 2]
        scores[i] = tables.get("di_residue", di)
    return scores


def compute_oligo_residue_propensity(sequence: str,
                                      tables: PropensityTables) -> np.ndarray:
    """Look up oligo-residue (triplet) propensity at each position.

    For position i, uses the triplet (i, i+1, i+2). Last two residues get 0.
    """
    n = len(sequence)
    scores = np.zeros(n)
    for i in range(n - 2):
        tri = sequence[i:i + 3]
        scores[i] = tables.get("oligo_residue", tri)
    return scores


def extract_sequence_propensity_features(
        sequence: str, tables: PropensityTables) -> dict[str, np.ndarray]:
    """Extract all Category A sequence propensity features.

    These are the raw per-position propensity values (window averaging
    is applied later in the pipeline).

    Returns:
        Dict mapping feature name to (N,) arrays.
    """
    return {
        "prop_aa": compute_single_aa_propensity(sequence, tables),
        "prop_di": compute_di_residue_propensity(sequence, tables),
        "prop_oligo": compute_oligo_residue_propensity(sequence, tables),
    }
