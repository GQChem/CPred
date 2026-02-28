"""Feature extraction and prediction pipeline.

Orchestrates: PDB parsing -> feature extraction -> standardization -> prediction.

The 46 features are organized into three categories (per Lo et al. 2012):
  Category A (sequence propensity): 3 features × 7 window positions = 21
  Category B (SS propensity):       3 features × 7 window positions = 21
  Category C (tertiary structure):  4 window-averaged structural features = 4

Total: 21 + 21 + 4 = 46 features

Category C structural features that are NOT window-expanded (window-averaged):
  rsa, closeness, gnm_msf, bfactor  (4 features)

Category C features that are kept as single per-residue values but are NOT
included in the 46-feature set (used internally only):
  cn, wcn, cm, depth, hbond, farness_buried, farness_hydrophobic,
  farness_sum, farness_product
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cpred.io.pdb_parser import ProteinStructure, parse_pdb
from cpred.propensity.tables import PropensityTables
from cpred.features.tertiary_structure import extract_tertiary_features
from cpred.features.contact_network import extract_contact_network_features
from cpred.features.gnm import compute_gnm_fluctuation
from cpred.features.sequence_propensity import extract_sequence_propensity_features
from cpred.features.secondary_structure import extract_secondary_structure_features
from cpred.features.window import (
    window_average_dict, window_average,
    window_expand_dict, DEFAULT_WINDOW,
)
from cpred.features.standardization import standardize_features

# Window positions suffixes for expanded features
_W = DEFAULT_WINDOW
_SUFFIXES = []
for j in range(2 * _W + 1):
    offset = j - _W
    if offset < 0:
        _SUFFIXES.append(f"_m{abs(offset)}")
    elif offset == 0:
        _SUFFIXES.append("_0")
    else:
        _SUFFIXES.append(f"_p{offset}")

# Category A base features (expanded to 21 = 3 × 7)
_CAT_A_BASE = ["prop_aa", "prop_di", "prop_oligo"]
CAT_A_FEATURES = [f"{base}{suf}" for base in _CAT_A_BASE for suf in _SUFFIXES]

# Category B base features (expanded to 21 = 3 × 7)
_CAT_B_BASE = ["prop_dssp", "prop_rama", "prop_kappa_alpha"]
CAT_B_FEATURES = [f"{base}{suf}" for base in _CAT_B_BASE for suf in _SUFFIXES]

# Category C: window-averaged structural features (4 features)
CAT_C_FEATURES = ["rsa", "closeness", "gnm_msf", "bfactor"]

# Full 46-feature canonical order
FEATURE_NAMES = CAT_A_FEATURES + CAT_B_FEATURES + CAT_C_FEATURES

NUM_FEATURES = len(FEATURE_NAMES)  # 46

# Feature groups for the HI model
FEATURE_GROUPS = {
    "seq_propensity": CAT_A_FEATURES,
    "ss_propensity": CAT_B_FEATURES,
    "tertiary_packing": ["rsa", "bfactor"],
    "contact_network": ["closeness"],
    "dynamics": ["gnm_msf"],
}


def extract_all_features(protein: ProteinStructure,
                         tables: PropensityTables) -> dict[str, np.ndarray]:
    """Extract all 46 features for a protein structure.

    Args:
        protein: Parsed protein structure.
        tables: Propensity lookup tables.

    Returns:
        Dictionary mapping feature name to (N,) arrays.
    """
    features = {}

    # Category A: sequence propensity — window EXPAND to 21 features
    seq_feats = extract_sequence_propensity_features(protein.sequence, tables)
    seq_expanded = window_expand_dict(seq_feats)
    features.update(seq_expanded)

    # Category B: SS propensity — window EXPAND to 21 features
    ss_feats = extract_secondary_structure_features(protein, tables)
    ss_expanded = window_expand_dict(ss_feats)
    features.update(ss_expanded)

    # Category C: tertiary structure — window AVERAGE to single values
    tert_feats = extract_tertiary_features(protein)
    contact_feats = extract_contact_network_features(protein)
    gnm_msf = compute_gnm_fluctuation(protein.ca_coords)

    cat_c = {}
    # Only keep the 4 Cat C features we use
    for key in ["rsa", "bfactor"]:
        if key in tert_feats:
            cat_c[key] = tert_feats[key]
    if "closeness" in contact_feats:
        cat_c["closeness"] = contact_feats["closeness"]
    cat_c["gnm_msf"] = gnm_msf

    # Window average Cat C features
    cat_c = window_average_dict(cat_c)
    features.update(cat_c)

    return features


def build_feature_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    """Assemble feature dictionary into a matrix.

    Args:
        features: Dictionary of feature name -> (N,) arrays.

    Returns:
        (N, 46) feature matrix.
    """
    ordered = []
    n = next(len(v) for v in features.values())
    for name in FEATURE_NAMES:
        if name in features:
            ordered.append(features[name])
        else:
            ordered.append(np.zeros(n))
    return np.column_stack(ordered)


def predict_from_pdb(pdb_path: str | Path, chain_id: str = "A",
                     tables: PropensityTables | None = None,
                     model=None,
                     threshold: float = 0.5) -> dict:
    """Full prediction pipeline from PDB file.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier.
        tables: Propensity tables (loaded if None).
        model: Trained ensemble model (uses default if None).
        threshold: Probability threshold for viable prediction.

    Returns:
        Dictionary with prediction results.
    """
    # Parse PDB
    protein = parse_pdb(pdb_path, chain_id=chain_id)

    # Load propensity tables
    if tables is None:
        tables = PropensityTables()
        tables.load()

    # Extract features
    features = extract_all_features(protein, tables)

    # Standardize
    features = standardize_features(features)

    # Build feature matrix
    X = build_feature_matrix(features)

    # Predict
    if model is not None:
        probabilities = model.predict(X)
    else:
        probabilities = 1 / (1 + np.exp(-X.mean(axis=1)))

    viable = probabilities >= threshold

    return {
        "pdb_id": protein.pdb_id,
        "chain_id": protein.chain_id,
        "sequence": protein.sequence,
        "residue_numbers": protein.residue_numbers,
        "probabilities": probabilities,
        "viable": viable,
        "features": X,
        "feature_names": FEATURE_NAMES,
    }
