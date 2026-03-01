#!/usr/bin/env python3
"""10-fold cross-validation of the CPred ensemble on Dataset T.

Trains on Dataset T (176 labeled sites from 6 proteins), then runs
stratified 10-fold CV on the full ensemble and each individual model.
Also evaluates on DHFR as independent test set.

Usage:
    python scripts/cross_validate.py
    python scripts/cross_validate.py --no-gpu
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpred.io.pdb_parser import parse_pdb
from cpred.pipeline import FEATURE_NAMES
from cpred.models.ensemble import CPredEnsemble
from cpred.training.evaluate import compute_metrics

# Reuse helpers from the training script
from prepare_and_train import (
    parse_dataset_s3,
    parse_dataset_t,
    download_pdb,
    build_propensity_tables_from_data,
    extract_features_for_protein,
    _extract_features_dataset_t_worker,
)
from cpred.propensity.tables import PropensityTables


def run_cv(X: np.ndarray, y: np.ndarray, n_folds: int = 10) -> None:
    """Run 10-fold stratified CV on the full ensemble and individual models."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    ensemble_metrics = []
    individual_metrics = {"rf": [], "svm": [], "ann": [], "hi": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\n--- Fold {fold + 1}/{n_folds} "
              f"(train: {len(train_idx)}, test: {len(test_idx)}, "
              f"test_pos: {int(y_test.sum())}) ---")

        ensemble = CPredEnsemble(feature_names=FEATURE_NAMES)
        ensemble.fit(X_train, y_train, feature_names=FEATURE_NAMES)

        # Ensemble prediction (unsmoothed for CV since samples aren't sequential)
        probs = ensemble.predict_unsmoothed(X_test)
        metrics = compute_metrics(y_test, probs)
        ensemble_metrics.append(metrics)
        print(f"  Ensemble: AUC={metrics['auc']:.4f}  "
              f"Sens={metrics['sensitivity']:.4f}  "
              f"Spec={metrics['specificity']:.4f}  "
              f"MCC={metrics['mcc']:.4f}")

        # Individual model predictions
        ind_probs = ensemble.predict_individual(X_test)
        for name, p in ind_probs.items():
            m = compute_metrics(y_test, p)
            individual_metrics[name].append(m)

    # Print summary
    print("\n" + "=" * 60)
    print(f"{n_folds}-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)

    print("\nEnsemble (mean +/- std):")
    _print_summary(ensemble_metrics)

    for name in ["rf", "svm", "ann", "hi"]:
        print(f"\n{name.upper()} (mean +/- std):")
        _print_summary(individual_metrics[name])

    # Paper reference values (Table 2)
    print("\nPaper reference (Table 2, Combined/CPred):")
    print("  AUC:         0.905")
    print("  Sensitivity: 0.857")
    print("  Specificity: 0.790")
    print("  MCC:         0.632")


def _print_summary(metrics_list: list[dict]) -> None:
    for key in ["auc", "sensitivity", "specificity", "accuracy", "mcc", "threshold"]:
        values = [m.get(key, 0.5) for m in metrics_list]
        print(f"  {key:>13s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


def main():
    parser = argparse.ArgumentParser(description="CPred 10-fold cross-validation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tables-dir", type=Path,
                        default=Path("cpred/data/propensity_tables"))
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=99999,
                        help="Number of permutations for propensity p-values (Lo et al. 2012)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--rmsf-dir", type=Path, default=Path("data/rmsf"),
                        help="Directory with per-protein RMSF CSVs from CABSflex")
    args = parser.parse_args()

    supp_dir = args.data_dir / "supplementary"
    pdb_dir = args.data_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Parse Dataset T ===
    print("=" * 60)
    print("STEP 1: Parsing Dataset T")
    print("=" * 60)
    dataset_t_path = supp_dir / "dataset_t.csv"
    if not dataset_t_path.exists():
        print(f"  {dataset_t_path} not found. Run scripts/convert_dataset_l.py first.")
        sys.exit(1)
    dataset_t = parse_dataset_t(dataset_t_path)
    n_viable = sum(len(v["sites_viable"]) for v in dataset_t.values())
    n_inviable = sum(len(v["sites_inviable"]) for v in dataset_t.values())
    print(f"Dataset T: {len(dataset_t)} proteins, "
          f"{n_viable} viable + {n_inviable} inviable = {n_viable + n_inviable} sites")

    # Dataset S3 for propensity tables
    proteins_s3 = parse_dataset_s3(supp_dir)
    print(f"Dataset S3: {len(proteins_s3)} proteins (for propensity tables)")

    # === Step 2: Download PDBs ===
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 2: Downloading PDB structures")
        print("=" * 60)
        pdb_ids = list(proteins_s3.keys())
        downloaded, failed = 0, 0
        for i, pdb_id in enumerate(pdb_ids):
            path = download_pdb(pdb_id, pdb_dir)
            if path:
                downloaded += 1
            else:
                failed += 1
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(pdb_ids)} "
                      f"(downloaded: {downloaded}, failed: {failed})")
        print(f"  S3: Downloaded: {downloaded}, Failed: {failed}")

        for pdb_id in dataset_t:
            download_pdb(pdb_id, pdb_dir)

        download_pdb("1rx4", pdb_dir)

    # === Step 3: Build propensity tables ===
    print("\n" + "=" * 60)
    print("STEP 3: Loading/building propensity tables")
    print("=" * 60)

    use_gpu = not args.no_gpu
    required_tables = ["single_aa", "di_residue", "oligo_residue",
                       "dssp", "ramachandran", "kappa_alpha"]

    def _table_needs_rebuild(name):
        import json
        path = args.tables_dir / f"{name}.json"
        if not path.exists() or path.stat().st_size < 10:
            return True
        try:
            with open(path) as f:
                data = json.load(f)
            if all(v == 0.0 for v in data.values()):
                return True
        except Exception:
            return True
        return False

    missing = [t for t in required_tables if _table_needs_rebuild(t)]
    if not missing:
        print("  All propensity tables exist, loading...")
        tables = PropensityTables(args.tables_dir)
        tables.load()
    else:
        print(f"  Missing or incomplete tables: {missing}")
        tables = build_propensity_tables_from_data(
            proteins_s3, pdb_dir, args.tables_dir, use_gpu=use_gpu,
            n_permutations=args.n_permutations)

    # === Step 4: Extract features for Dataset T ===
    print("\n" + "=" * 60)
    print("STEP 4: Extracting features (Dataset T)")
    print("=" * 60)

    pdb_ids = list(dataset_t.keys())
    rmsf_dir = str(args.rmsf_dir) if args.rmsf_dir else None
    worker_args = [
        (pdb_id, dataset_t[pdb_id], str(pdb_dir / f"{pdb_id}.pdb"), tables, rmsf_dir)
        for pdb_id in pdb_ids
    ]

    all_X, all_y = [], []
    processed, skipped = 0, 0

    for wa in worker_args:
        result, reason = _extract_features_dataset_t_worker(wa)
        if result is None:
            skipped += 1
            print(f"  SKIP: {reason}")
        else:
            X, y = result
            all_X.append(X)
            all_y.append(y)
            processed += 1
            print(f"  {wa[0]}: {len(y)} sites ({int(y.sum())} viable)")

    if not all_X:
        print("ERROR: No training data extracted!")
        sys.exit(1)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    n_pos = int(y_all.sum())
    n_neg = len(y_all) - n_pos
    print(f"\n  Training matrix: {X_all.shape}")
    print(f"  Positive: {n_pos}, Negative: {n_neg}")
    print("  (No downsampling — Dataset T is naturally balanced)")

    # === Step 5: 10-fold CV ===
    print("\n" + "=" * 60)
    print(f"STEP 5: {args.n_folds}-fold Cross-Validation")
    print("=" * 60)

    run_cv(X_all, y_all, n_folds=args.n_folds)

    # === Step 6: Independent test on DHFR ===
    print("\n" + "=" * 60)
    print("STEP 6: Independent test on DHFR (1RX4)")
    print("=" * 60)

    dhfr_csv = supp_dir / "dataset_dhfr.csv"
    dhfr_path = pdb_dir / "1rx4.pdb"

    if dhfr_path.exists() and dhfr_csv.exists():
        try:
            # Train ensemble on full Dataset T
            print("  Training ensemble on full Dataset T...")
            ensemble = CPredEnsemble(feature_names=FEATURE_NAMES)
            ensemble.fit(X_all, y_all, feature_names=FEATURE_NAMES)

            dhfr = parse_pdb(dhfr_path, chain_id="A")
            X_dhfr = extract_features_for_protein(dhfr, tables, rmsf_dir=rmsf_dir)

            if X_dhfr is not None:
                dhfr_df = pd.read_csv(dhfr_csv)
                resnum_to_idx = {rn: i for i, rn in enumerate(dhfr.residue_numbers)}

                labeled_indices = []
                labels = []
                for _, row in dhfr_df.iterrows():
                    idx = resnum_to_idx.get(int(row['residue_number']))
                    if idx is not None:
                        labeled_indices.append(idx)
                        labels.append(int(row['viable']))

                if labeled_indices:
                    y_dhfr = np.array(labels)

                    # Predict on full protein (smoothed), then extract labeled
                    probs_full = ensemble.predict(X_dhfr)
                    probs_labeled = probs_full[labeled_indices]

                    dhfr_metrics = compute_metrics(y_dhfr, probs_labeled)
                    print(f"\n  DHFR results ({len(y_dhfr)} sites, "
                          f"{int(y_dhfr.sum())} viable):")
                    for key, val in dhfr_metrics.items():
                        print(f"    {key}: {val:.4f}")

                    print("\n  Paper reference (DHFR independent test):")
                    print("    AUC:         0.906")
                    print("    Sensitivity: 0.709")
                    print("    Specificity: 0.918")
                    print("    MCC:         0.633")
        except Exception as e:
            print(f"  DHFR test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  DHFR files not found, skipping")

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
