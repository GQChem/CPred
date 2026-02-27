"""Training data loading and parsing.

Parses Dataset T (training set) and DHFR (independent test) from
supplementary files or pre-processed formats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cpred.io.pdb_parser import parse_pdb, ProteinStructure
from cpred.propensity.tables import PropensityTables
from cpred.pipeline import extract_all_features, build_feature_matrix, FEATURE_NAMES
from cpred.features.standardization import standardize_features


def load_training_sequences(data_dir: Path) -> tuple[list[str], list[list[int]], list[str]]:
    """Load training sequences and CP sites from supplementary data.

    Args:
        data_dir: Directory containing downloaded supplementary files.

    Returns:
        Tuple of (experimental_sequences, per_seq_cp_sites, comparison_sequences).
    """
    supp_dir = data_dir / "supplementary"

    exp_sequences = []
    exp_sites = []
    comp_sequences = []

    # Try loading Dataset S1 (CP site database)
    ds1_path = supp_dir / "Dataset_S1.xls"
    if ds1_path.exists():
        try:
            df = pd.read_excel(ds1_path)
            # Look for sequence and site columns
            for _, row in df.iterrows():
                seq = None
                sites = []
                for col in df.columns:
                    val = str(row[col]).strip()
                    col_lower = str(col).lower()
                    if "sequence" in col_lower and len(val) > 10:
                        seq = val
                    elif "site" in col_lower or "position" in col_lower:
                        try:
                            sites.append(int(val))
                        except (ValueError, TypeError):
                            pass
                if seq:
                    exp_sequences.append(seq)
                    exp_sites.append(sites)
        except Exception as e:
            print(f"Warning: Could not parse Dataset S1: {e}")

    # Try loading Dataset S2 (comparison sequences)
    ds2_path = supp_dir / "Dataset_S2.xls"
    if ds2_path.exists():
        try:
            df = pd.read_excel(ds2_path)
            for _, row in df.iterrows():
                for col in df.columns:
                    val = str(row[col]).strip()
                    if len(val) > 10 and val.isalpha():
                        comp_sequences.append(val)
                        break
        except Exception as e:
            print(f"Warning: Could not parse Dataset S2: {e}")

    return exp_sequences, exp_sites, comp_sequences


def load_structure_dataset(pdb_dir: Path, labels_file: Path | None = None,
                           tables: PropensityTables | None = None,
                           chain_id: str = "A"
                           ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a structure-based dataset for training/evaluation.

    For each PDB file in the directory, extracts features and constructs
    the feature matrix with labels.

    Args:
        pdb_dir: Directory containing PDB files.
        labels_file: CSV/TSV with columns: pdb_id, residue_number, label.
        tables: Propensity tables.
        chain_id: Default chain ID.

    Returns:
        Tuple of (X, y, protein_ids) where X is feature matrix, y is labels.
    """
    if tables is None:
        tables = PropensityTables()
        tables.load()

    all_X = []
    all_y = []
    all_ids = []

    # Load labels if provided
    labels_lookup: dict[str, dict[int, int]] = {}
    if labels_file is not None and labels_file.exists():
        df = pd.read_csv(labels_file, sep=None, engine="python")
        for _, row in df.iterrows():
            pdb_id = str(row.iloc[0]).strip().lower()
            resnum = int(row.iloc[1])
            label = int(row.iloc[2])
            if pdb_id not in labels_lookup:
                labels_lookup[pdb_id] = {}
            labels_lookup[pdb_id][resnum] = label

    # Process each PDB file
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.lower()
        try:
            protein = parse_pdb(pdb_path, chain_id=chain_id)
        except Exception as e:
            print(f"  Skipping {pdb_id}: {e}")
            continue

        features = extract_all_features(protein, tables)
        features = standardize_features(features)
        X = build_feature_matrix(features)

        # Assign labels
        if pdb_id in labels_lookup:
            y = np.zeros(protein.n_residues)
            for i, resnum in enumerate(protein.residue_numbers):
                if resnum in labels_lookup[pdb_id]:
                    y[i] = labels_lookup[pdb_id][resnum]
        else:
            # Default: all negative (no known CP sites)
            y = np.zeros(protein.n_residues)

        all_X.append(X)
        all_y.append(y)
        all_ids.extend([pdb_id] * len(y))

    if not all_X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0), []

    return np.vstack(all_X), np.concatenate(all_y), all_ids
