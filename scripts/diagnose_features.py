#!/usr/bin/env python3
"""Diagnostic: compute per-feature AUC on Dataset T and DHFR.

Compare against paper's Table S3 (10-fold CV AUC per feature on Dataset T)
and individual feature AUCs on DHFR to identify feature computation issues.

Usage:
    python scripts/diagnose_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from cpred.io.pdb_parser import parse_pdb
from cpred.propensity.tables import PropensityTables
from cpred.pipeline import get_feature_names, extract_all_features, build_feature_matrix
from cpred.features.standardization import standardize_features


def extract_features_for_protein(protein, tables, feature_names):
    features = extract_all_features(protein, tables)
    features = standardize_features(features)
    return build_feature_matrix(features, feature_names=feature_names)


def main():
    data_dir = Path("data")
    tables_dir = Path("cpred/data/propensity_tables")
    supp_dir = data_dir / "supplementary"
    pdb_dir = data_dir / "pdb"

    tables = PropensityTables(tables_dir)
    tables.load()
    feat_names = get_feature_names(include_rmsf=False)

    # --- Load Dataset T ---
    df_t = pd.read_csv(supp_dir / "dataset_t.csv")
    proteins = {}
    for _, row in df_t.iterrows():
        pdb_id = row['pdb_id']
        if pdb_id not in proteins:
            proteins[pdb_id] = {'chain': row['chain'],
                                'sites_viable': [], 'sites_inviable': []}
        if row['viable'] == 1:
            proteins[pdb_id]['sites_viable'].append(int(row['residue_number']))
        else:
            proteins[pdb_id]['sites_inviable'].append(int(row['residue_number']))

    all_X, all_y = [], []
    for pdb_id, info in proteins.items():
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        protein = parse_pdb(pdb_path, chain_id=info["chain"])
        X = extract_features_for_protein(protein, tables, feat_names)
        resnum_to_idx = {rn: i for i, rn in enumerate(protein.residue_numbers)}
        indices, labels = [], []
        for site in info["sites_viable"]:
            idx = resnum_to_idx.get(site)
            if idx is not None:
                indices.append(idx)
                labels.append(1.0)
        for site in info["sites_inviable"]:
            idx = resnum_to_idx.get(site)
            if idx is not None:
                indices.append(idx)
                labels.append(0.0)
        all_X.append(X[indices])
        all_y.append(np.array(labels))

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    # --- Load DHFR ---
    dhfr = parse_pdb(pdb_dir / "1rx4.pdb", chain_id="A")
    X_dhfr = extract_features_for_protein(dhfr, tables, feat_names)
    dhfr_df = pd.read_csv(supp_dir / "dataset_dhfr.csv")
    resnum_to_idx = {rn: i for i, rn in enumerate(dhfr.residue_numbers)}
    dhfr_indices, dhfr_labels = [], []
    for _, row in dhfr_df.iterrows():
        idx = resnum_to_idx.get(int(row['residue_number']))
        if idx is not None:
            dhfr_indices.append(idx)
            dhfr_labels.append(int(row['viable']))
    X_dhfr_labeled = X_dhfr[dhfr_indices]
    y_dhfr = np.array(dhfr_labels)

    # --- Per-feature AUC on Dataset T (10-fold CV) ---
    print(f"{'Feature':<25} {'Train AUC':>10} {'CV AUC':>10} {'DHFR AUC':>10}  {'Viable mean':>11} {'Inviable mean':>13}")
    print("-" * 95)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for i, fname in enumerate(feat_names):
        # Training AUC (resubstitution)
        try:
            train_auc = roc_auc_score(y_train, X_train[:, i])
        except ValueError:
            train_auc = 0.5

        # 10-fold CV AUC
        cv_aucs = []
        for train_idx, test_idx in skf.split(X_train, y_train):
            try:
                auc = roc_auc_score(y_train[test_idx], X_train[test_idx, i])
                cv_aucs.append(auc)
            except ValueError:
                cv_aucs.append(0.5)
        cv_auc = np.mean(cv_aucs)

        # DHFR AUC
        try:
            dhfr_auc = roc_auc_score(y_dhfr, X_dhfr_labeled[:, i])
        except ValueError:
            dhfr_auc = 0.5

        viable_mean = X_train[y_train == 1, i].mean()
        inviable_mean = X_train[y_train == 0, i].mean()

        print(f"{fname:<25} {train_auc:>10.4f} {cv_auc:>10.4f} {dhfr_auc:>10.4f}  "
              f"{viable_mean:>+11.4f} {inviable_mean:>+13.4f}")

    # --- Category-level summary ---
    from cpred.pipeline import CAT_A_FEATURES, CAT_B_FEATURES
    cat_a_end = len(CAT_A_FEATURES)
    cat_b_end = cat_a_end + len(CAT_B_FEATURES)

    print(f"\n{'Category':<25} {'Train AUC':>10} {'CV AUC':>10} {'DHFR AUC':>10}")
    print("-" * 60)
    for cat_name, start, end in [("Cat A (seq propensity)", 0, cat_a_end),
                                  ("Cat B (SS propensity)", cat_a_end, cat_b_end),
                                  ("Cat C (tertiary)", cat_b_end, len(feat_names))]:
        # Average feature as category score
        cat_train = X_train[:, start:end].mean(axis=1)
        cat_dhfr = X_dhfr_labeled[:, start:end].mean(axis=1)
        try:
            t_auc = roc_auc_score(y_train, cat_train)
        except ValueError:
            t_auc = 0.5
        cv_aucs = []
        for train_idx, test_idx in skf.split(X_train, y_train):
            try:
                auc = roc_auc_score(y_train[test_idx], cat_train[test_idx])
                cv_aucs.append(auc)
            except ValueError:
                cv_aucs.append(0.5)
        try:
            d_auc = roc_auc_score(y_dhfr, cat_dhfr)
        except ValueError:
            d_auc = 0.5
        print(f"{cat_name:<25} {t_auc:>10.4f} {np.mean(cv_aucs):>10.4f} {d_auc:>10.4f}")

    # Paper reference values (from Table S3, 10-fold CV MCC on Dataset T)
    print(f"""
Paper reference (from Table S3 and text):
  Category A average MCC:  0.27  (10-fold CV on Dataset T)
  Category B average MCC:  0.49
  Sequence propensity AUC: ~0.60 (page 5)
  SS propensity AUC:       ~0.73 (page 5)
  RSA AUC:                 0.69  (page 5)
  CM AUC:                  0.74  (page 5)
  CN AUC:                  0.78  (Figure 3E, 6.4A radius)
  WCN AUC:                 0.79  (Figure 3G)
  Closeness AUC:           0.73  (page 5)
  Depth AUC:               0.63  (page 5)

  Overall system:
    10-fold CV on T: AUC=0.91, Sens=0.86, Spec=0.79, MCC=0.63
    DHFR:            AUC=0.91, Sens=0.71, Spec=0.92, MCC=0.64
""")


if __name__ == "__main__":
    main()
