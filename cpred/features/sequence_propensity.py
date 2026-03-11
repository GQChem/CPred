"""Category A features: sequence-based propensity scores.

Per Lo et al. 2012 (PLoS ONE), Cat A features are coupled residue propensities
using three amino acid alphabets:
  - aa:   standard 20 amino acids
  - aac3: 3-class reduced alphabet (hydrophobic/neutral/hydrophilic, ref [46])
  - aac5: 5-class reduced alphabet (Lehninger, ref [48])

Each feature name is <pattern>_<alphabet> where:
  - Pattern prefix defines which residue positions to combine
  - Alphabet suffix defines which propensity table to use

Coupling patterns (all use the same positional offsets regardless of alphabet):
  R:    single at i
  2R:   di (i, i+1)
  RxR:  di (i-1, i+1)
  R2xR: di (i-2, i+1)
  R3xR: di (i-3, i+1)
  3R:   tri (i-1, i, i+1)
  4R:   tetra (i-2, i-1, i+1, i+2) — skip center
  5R:   penta (i-2, i-1, i, i+1, i+2)

Features (19 total, matching the 1dprop branch of Figure 4):
  R_aa, R_aac3, R_aac5,
  2R_aa, 2R_aac3, 2R_aac5,
  RxR_aa, RxR_aac3, RxR_aac5,
  R2xR_aa, R2xR_aac3, R2xR_aac5,
  R3xR_aac3, R3xR_aac5,
  3R_aac3, 3R_aac5,
  4R_aac3, 4R_aac5,
  5R_aac3
"""

from __future__ import annotations

import numpy as np

from cpred.propensity.tables import PropensityTables
from cpred.features.aa_alphabets import convert_sequence_aac3, convert_sequence_aac5


def _get_code(seq: str, idx: int) -> str | None:
    """Return the character at idx, or None if out of bounds."""
    if 0 <= idx < len(seq):
        return seq[idx]
    return None


def _single_prop(tables: PropensityTables, table: str, code: str | None) -> float:
    if code is None:
        return 0.0
    return tables.get(table, code)


def _di_prop(tables: PropensityTables, table: str,
             c1: str | None, c2: str | None) -> float:
    if c1 is None or c2 is None:
        return 0.0
    return tables.get(table, c1 + c2)


def _oligo_prop(tables: PropensityTables, table: str, *codes: str | None) -> float:
    if any(c is None for c in codes):
        return 0.0
    return tables.get(table, "".join(codes))  # type: ignore[arg-type]


def _extract_coupled_aa_features(
        seq: str,
        tables: PropensityTables,
        single_table: str,
        di_table: str,
        oligo_table: str,
        suffix: str) -> dict[str, np.ndarray]:
    """Extract coupled propensity features for one AA alphabet.

    For 'aa' suffix: 6 features (R, 2R, RxR, R2xR) — no R3xR/3R/4R/5R
    For 'aac3' suffix: 9 features (all patterns)
    For 'aac5' suffix: 7 features (no 5R)

    Args:
        seq: Sequence in the target alphabet (raw AA, aac3 codes, or aac5 codes).
        tables: Propensity table container.
        single_table: Table name for single-code lookups.
        di_table: Table name for di-code lookups.
        oligo_table: Table name for oligo-code lookups.
        suffix: Feature name suffix ('aa', 'aac3', or 'aac5').
    """
    n = len(seq)
    features: dict[str, list[float]] = {}

    # Define which patterns this alphabet uses
    # aa:   R, 2R, RxR, R2xR (4 features)
    # aac3: R, 2R, RxR, R2xR, R3xR, 3R, 4R, 5R (9 features)
    # aac5: R, 2R, RxR, R2xR, R3xR, 3R, 4R (7 features)
    has_r3xr = suffix in ("aac3", "aac5")
    has_3r = suffix in ("aac3", "aac5")
    has_4r = suffix in ("aac3", "aac5")
    has_5r = suffix == "aac3"

    # Initialize all feature lists
    features[f"R_{suffix}"] = []
    features[f"2R_{suffix}"] = []
    features[f"RxR_{suffix}"] = []
    features[f"R2xR_{suffix}"] = []
    if has_r3xr:
        features[f"R3xR_{suffix}"] = []
    if has_3r:
        features[f"3R_{suffix}"] = []
    if has_4r:
        features[f"4R_{suffix}"] = []
    if has_5r:
        features[f"5R_{suffix}"] = []

    for i in range(n):
        c = _get_code(seq, i)
        cm3 = _get_code(seq, i - 3)
        cm2 = _get_code(seq, i - 2)
        cm1 = _get_code(seq, i - 1)
        cp1 = _get_code(seq, i + 1)
        cp2 = _get_code(seq, i + 2)

        # R: single at i
        features[f"R_{suffix}"].append(_single_prop(tables, single_table, c))

        # 2R: di (i, i+1)
        features[f"2R_{suffix}"].append(_di_prop(tables, di_table, c, cp1))

        # RxR: di (i-1, i+1)
        features[f"RxR_{suffix}"].append(_di_prop(tables, di_table, cm1, cp1))

        # R2xR: di (i-2, i+1)
        features[f"R2xR_{suffix}"].append(_di_prop(tables, di_table, cm2, cp1))

        if has_r3xr:
            # R3xR: di (i-3, i+1)
            features[f"R3xR_{suffix}"].append(_di_prop(tables, di_table, cm3, cp1))

        if has_3r:
            # 3R: tri (i-1, i, i+1)
            features[f"3R_{suffix}"].append(
                _oligo_prop(tables, oligo_table, cm1, c, cp1))

        if has_4r:
            # 4R: tetra (i-2, i-1, i+1, i+2) — skip center
            features[f"4R_{suffix}"].append(
                _oligo_prop(tables, oligo_table, cm2, cm1, cp1, cp2))

        if has_5r:
            # 5R: penta (i-2, i-1, i, i+1, i+2)
            features[f"5R_{suffix}"].append(
                _oligo_prop(tables, oligo_table, cm2, cm1, c, cp1, cp2))

    return {k: np.array(v) for k, v in features.items()}


def extract_sequence_propensity_features(
        sequence: str, tables: PropensityTables) -> dict[str, np.ndarray]:
    """Extract all Category A coupled-residue propensity features.

    Uses three amino acid alphabets (aa, aac3, aac5) with the same coupling
    patterns but different propensity lookup tables.

    Returns:
        Dict mapping feature name to (N,) arrays (19 features).
    """
    # Convert sequence to reduced alphabets
    seq_aac3 = convert_sequence_aac3(sequence)
    seq_aac5 = convert_sequence_aac5(sequence)

    features = {}

    # Standard 20-AA alphabet: R, 2R, RxR, R2xR (4 features)
    features.update(_extract_coupled_aa_features(
        sequence, tables, "single_aa", "di_residue", "oligo_residue", "aa"))

    # 3-class reduced alphabet: all 9 patterns (9 features)
    features.update(_extract_coupled_aa_features(
        seq_aac3, tables, "single_aac3", "di_aac3", "oligo_aac3", "aac3"))

    # 5-class reduced alphabet: 7 patterns — no 5R (7 features)
    features.update(_extract_coupled_aa_features(
        seq_aac5, tables, "single_aac5", "di_aac5", "oligo_aac5", "aac5"))

    # aa(4) + aac3(8) + aac5(7) = 19 features
    return features
