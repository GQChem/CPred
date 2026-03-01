"""Feature extraction and prediction pipeline.

Orchestrates: PDB parsing -> feature extraction -> standardization -> prediction.

The features are organized into three categories (per Lo et al. 2012):
  Category A (sequence propensity): 3 features × 7 window positions = 21
  Category B (SS propensity):       3 features × 7 window positions = 21
  Category C (tertiary structure):  16 window-averaged structural features

Total: 21 + 21 + 16 = 58 features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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

# Category C: all 16 tertiary structural features (window-averaged)
# Per Lo et al. 2012, Table 1 and Figure 3:
#   RSA, DPX(depth), CM, H-bonds, closeness, CN, WCN, B-factor,
#   GNM-F, DIS_b, DIS_hpho, Fb(farness_buried), Fhpho(farness_hydrophobic),
#   Fb∪Fhpho(farness_union), Fb∩Fhpho(farness_inter), + bfactor already counted
# Note: paper lists RMSF+ separately but we use GNM-F as proxy (same as paper's approach)
CAT_C_FEATURES = [
    "rsa",                    # RSA (relative solvent accessibility)
    "depth",                  # DPX+ (distance to surface)
    "cm",                     # CM+ (distance to centroid)
    "hbond",                  # H-bonds+
    "closeness",              # Closeness centrality
    "cn",                     # CN (contact number)
    "wcn",                    # WCN (weighted contact number)
    "bfactor",                # B-factor
    "rmsf",                   # RMSF+ (from CABSflex coarse-grained MD)
    "gnm_msf",                # GNM-F (Gaussian Network Model fluctuation)
    "dis_b",                  # DIS_b+ (avg distance to buried residues)
    "dis_hpho",               # DIS_hpho (avg distance to hydrophobic residues)
    "farness_buried",         # Fb+ (farness from buried core)
    "farness_hydrophobic",    # Fhpho (farness from hydrophobic residues)
    "farness_union",          # Fb∪hpho+ (farness from buried ∪ hydrophobic)
    "farness_inter",          # Fb∩hpho+ (farness from buried ∩ hydrophobic)
]

# Full canonical feature order
FEATURE_NAMES = CAT_A_FEATURES + CAT_B_FEATURES + CAT_C_FEATURES

NUM_FEATURES = len(FEATURE_NAMES)  # 57

# Feature groups for the HI model (matching Figure 4 categories)
FEATURE_GROUPS = {
    "seq_propensity": CAT_A_FEATURES,
    "ss_propensity": CAT_B_FEATURES,
    "tertiary_packing": ["rsa", "depth", "cm", "bfactor"],
    "contact_network": ["closeness", "cn", "wcn"],
    "farness": ["farness_buried", "farness_hydrophobic",
                 "farness_union", "farness_inter",
                 "dis_b", "dis_hpho"],
    "hbonds": ["hbond"],
    "dynamics": ["rmsf", "gnm_msf"],
}


def load_rmsf(pdb_id: str, residue_numbers: list[int],
              rmsf_dir: Path | str | None = None) -> np.ndarray:
    """Load per-residue RMSF values from a CABSflex CSV file.

    Expects a CSV file at ``rmsf_dir/{pdb_id}.csv`` with columns
    ``residue_number`` and ``rmsf``.

    Args:
        pdb_id: PDB identifier (lowercase).
        residue_numbers: Residue numbers from the parsed protein structure.
        rmsf_dir: Directory containing RMSF CSV files.

    Returns:
        (N,) array of RMSF values.  NaN for residues not found in the CSV.
    """
    n = len(residue_numbers)
    if rmsf_dir is None:
        return np.full(n, np.nan)

    rmsf_dir = Path(rmsf_dir)
    csv_path = rmsf_dir / f"{pdb_id.lower()}.csv"
    if not csv_path.exists():
        return np.full(n, np.nan)

    df = pd.read_csv(csv_path)
    resnum_to_rmsf = dict(zip(df["residue_number"].astype(int), df["rmsf"].astype(float)))

    rmsf = np.full(n, np.nan)
    for i, rn in enumerate(residue_numbers):
        if rn in resnum_to_rmsf:
            rmsf[i] = resnum_to_rmsf[rn]
    return rmsf


def extract_all_features(protein: ProteinStructure,
                         tables: PropensityTables,
                         rmsf_dir: Path | str | None = None) -> dict[str, np.ndarray]:
    """Extract all features for a protein structure.

    Args:
        protein: Parsed protein structure.
        tables: Propensity lookup tables.
        rmsf_dir: Directory containing per-protein RMSF CSV files from
            CABSflex (one file per protein: ``{pdb_id}.csv`` with columns
            ``residue_number,rmsf``).  If None, RMSF is filled with NaN
            and will become 0 after Z-score normalization.

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
    # Include all tertiary features
    for key in ["rsa", "depth", "cm", "hbond", "cn", "wcn", "bfactor",
                "dis_b", "dis_hpho"]:
        if key in tert_feats:
            cat_c[key] = tert_feats[key]

    # Contact network features
    for key in ["closeness", "farness_buried", "farness_hydrophobic",
                "farness_union", "farness_inter"]:
        if key in contact_feats:
            cat_c[key] = contact_feats[key]

    # RMSF from CABSflex
    cat_c["rmsf"] = load_rmsf(protein.pdb_id, protein.residue_numbers, rmsf_dir)

    # GNM fluctuation
    cat_c["gnm_msf"] = gnm_msf

    # Window average all Cat C features
    cat_c = window_average_dict(cat_c)
    features.update(cat_c)

    return features


def build_feature_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    """Assemble feature dictionary into a matrix.

    Args:
        features: Dictionary of feature name -> (N,) arrays.

    Returns:
        (N, num_features) feature matrix.
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
