"""Feature extraction and prediction pipeline.

Orchestrates: PDB parsing -> feature extraction -> standardization -> prediction.

The features are organized into three categories (per Lo et al. 2012):
  Category A (sequence propensity): 19 coupled-residue propensity features
  Category B (SS propensity):       15 coupled-SS propensity features
  Category C (tertiary structure):  16 window-averaged structural features

Total: 19 + 15 + 16 = 50 features
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
from cpred.features.window import window_average_dict, window_average
from cpred.features.standardization import standardize_features

# Category A: 19 coupled-residue propensity features (per Figure 4, 1dprop branch)
CAT_A_FEATURES = [
    "R_aa", "R_aac3", "R_aac5",
    "2R_aa", "2R_aac3", "2R_aac5",
    "RxR_aa", "RxR_aac3", "RxR_aac5",
    "R2xR_aa", "R2xR_aac3", "R2xR_aac5",
    "R3xR_aac3", "R3xR_aac5",
    "3R_aac3", "3R_aac5",
    "4R_aac3", "4R_aac5",
    "5R_aac3",
]

# Category B: 15 coupled SS propensity features (per Figure 4, 2dprop branch)
# 5 features × 3 SS alphabets (DSSP, Ramachandran, kappa-alpha)
CAT_B_FEATURES = [
    "R_sse", "2R_sse", "RxR_sse", "R2xR_sse", "R3xR_sse",
    "R_rm",  "2R_rm",  "RxR_rm",  "R2xR_rm",  "R3xR_rm",
    "R_ka",  "2R_ka",  "RxR_ka",  "R2xR_ka",  "R3xR_ka",
]

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

NUM_FEATURES = len(FEATURE_NAMES)  # 50

# Feature groups for the HI model (matching Figure 4 categories)
FEATURE_GROUPS = {
    "seq_propensity": CAT_A_FEATURES,   # 1dprop, weight 0.14
    "ss_propensity": CAT_B_FEATURES,    # 2dprop, weight 0.57
    # 3dprop sub-groups (weight 0.29):
    "solacc": ["rsa"],                  # Solacc
    "eccent": ["cm", "depth"],          # Eccent
    "awaycore": ["farness_buried", "dis_b"],              # Awaybury
    "awayhpho": ["farness_hydrophobic", "dis_hpho"],      # Awayhpho
    "hbonds": ["hbond"],                # Nhbonds
    "uncrowd": ["wcn", "cn", "closeness"],  # Uncrowd
    "dynamics": ["gnm_msf", "rmsf", "bfactor"],           # Flex
    # Extra Cat C not in the 46 but we keep them
    "extra": ["farness_union", "farness_inter"],
}


def _parse_rmsf_csv(csv_path: Path, chain_id: str = "A") -> dict[int, float]:
    """Parse an RMSF CSV file, auto-detecting format.

    Supported formats:
      1. Header ``id,chain,resi,rmsf`` (CABSflex output with header)
      2. Header ``residue_number,rmsf``
      3. No header, two columns: ``<chain><resnum>,<rmsf>`` (e.g. ``A6,0.45``)

    Returns:
        Mapping from residue number (int) to RMSF value (float).
    """
    import re

    with open(csv_path) as f:
        first_line = f.readline().strip()

    resnum_to_rmsf: dict[int, float] = {}

    # Detect format from first line
    if "resi" in first_line.lower() and "rmsf" in first_line.lower():
        # Format 1: id,chain,resi,rmsf (CABSflex with header)
        df = pd.read_csv(csv_path)
        col_resi = [c for c in df.columns if c.lower() == "resi"][0]
        col_rmsf = [c for c in df.columns if c.lower() == "rmsf"][0]
        if "chain" in [c.lower() for c in df.columns]:
            col_chain = [c for c in df.columns if c.lower() == "chain"][0]
            df = df[df[col_chain].astype(str) == chain_id]
        resnum_to_rmsf = dict(zip(df[col_resi].astype(int),
                                   df[col_rmsf].astype(float)))
    elif "residue_number" in first_line.lower() and "rmsf" in first_line.lower():
        # Format 2: residue_number,rmsf
        df = pd.read_csv(csv_path)
        resnum_to_rmsf = dict(zip(df["residue_number"].astype(int),
                                   df["rmsf"].astype(float)))
    else:
        # Format 3: no header, <chain><resnum>,<rmsf> (e.g. A6,0.45)
        # or just <resnum>,<rmsf>
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                col0, col1 = parts[0].strip(), parts[1].strip()
                try:
                    rmsf_val = float(col1)
                except ValueError:
                    continue
                # Try parsing <chain><resnum> like "A6"
                m = re.match(r"([A-Za-z])(\d+)$", col0)
                if m:
                    if m.group(1).upper() == chain_id.upper():
                        resnum_to_rmsf[int(m.group(2))] = rmsf_val
                else:
                    # Plain integer residue number
                    try:
                        resnum_to_rmsf[int(col0)] = rmsf_val
                    except ValueError:
                        continue

    return resnum_to_rmsf


def load_rmsf_file(csv_path: Path | str, residue_numbers: list[int],
                   chain_id: str = "A") -> np.ndarray:
    """Load RMSF from a single CSV file.

    Args:
        csv_path: Path to RMSF CSV file.
        residue_numbers: Residue numbers from the parsed protein structure.
        chain_id: Chain ID to filter on (for multi-chain CSVs).

    Returns:
        (N,) array of RMSF values.  NaN for residues not found.
    """
    csv_path = Path(csv_path)
    n = len(residue_numbers)
    if not csv_path.exists():
        return np.full(n, np.nan)

    resnum_to_rmsf = _parse_rmsf_csv(csv_path, chain_id)
    rmsf = np.full(n, np.nan)
    for i, rn in enumerate(residue_numbers):
        if rn in resnum_to_rmsf:
            rmsf[i] = resnum_to_rmsf[rn]
    return rmsf


def load_rmsf(pdb_id: str, residue_numbers: list[int],
              rmsf_dir: Path | str | None = None,
              chain_id: str = "A") -> np.ndarray:
    """Load per-residue RMSF values from a CABSflex CSV file in a directory.

    Searches for ``{pdb_id}.csv`` or ``{pdb_id}_RMSF.csv`` in *rmsf_dir*.

    Args:
        pdb_id: PDB identifier (lowercase).
        residue_numbers: Residue numbers from the parsed protein structure.
        rmsf_dir: Directory containing RMSF CSV files.
        chain_id: Chain ID to filter on.

    Returns:
        (N,) array of RMSF values.  NaN for residues not found in the CSV.
    """
    n = len(residue_numbers)
    if rmsf_dir is None:
        return np.full(n, np.nan)

    rmsf_dir = Path(rmsf_dir)
    # Try multiple naming conventions
    for name in [f"{pdb_id.lower()}.csv", f"{pdb_id.lower()}_RMSF.csv",
                 f"{pdb_id.upper()}.csv", f"{pdb_id.upper()}_RMSF.csv"]:
        csv_path = rmsf_dir / name
        if csv_path.exists():
            return load_rmsf_file(csv_path, residue_numbers, chain_id)

    return np.full(n, np.nan)


def extract_all_features(protein: ProteinStructure,
                         tables: PropensityTables,
                         rmsf_dir: Path | str | None = None,
                         rmsf_file: Path | str | None = None) -> dict[str, np.ndarray]:
    """Extract all features for a protein structure.

    Args:
        protein: Parsed protein structure.
        tables: Propensity lookup tables.
        rmsf_dir: Directory containing per-protein RMSF CSV files from
            CABSflex.  Ignored if *rmsf_file* is given.
        rmsf_file: Path to a single RMSF CSV file for this protein.

    Returns:
        Dictionary mapping feature name to (N,) arrays.
    """
    features = {}

    # Category A: coupled-residue propensity features (19, already per-position)
    seq_feats = extract_sequence_propensity_features(protein.sequence, tables)
    features.update(seq_feats)

    # Category B: coupled SS propensity features (15, already per-position)
    ss_feats = extract_secondary_structure_features(protein, tables)
    features.update(ss_feats)

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
    if rmsf_file is not None:
        cat_c["rmsf"] = load_rmsf_file(rmsf_file, protein.residue_numbers,
                                        protein.chain_id)
    else:
        cat_c["rmsf"] = load_rmsf(protein.pdb_id, protein.residue_numbers,
                                   rmsf_dir, protein.chain_id)

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
                     threshold: float = 0.5,
                     rmsf_file: str | Path | None = None) -> dict:
    """Full prediction pipeline from PDB file.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier.
        tables: Propensity tables (loaded if None).
        model: Trained ensemble model (uses default if None).
        threshold: Probability threshold for viable prediction.
        rmsf_file: Path to RMSF CSV file from CABSflex.

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
    features = extract_all_features(protein, tables, rmsf_file=rmsf_file)

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
