#!/usr/bin/env python3
"""10-fold cross-validation of the CPred ensemble, matching Table 1 of the paper.

Reuses the same data preparation as prepare_and_train.py (parse Dataset S3,
download PDBs, build propensity tables, extract features, downsample), then
runs stratified 10-fold CV on the full ensemble and each individual model.

Usage:
    python scripts/cross_validate.py
    python scripts/cross_validate.py --no-gpu
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpred.pipeline import FEATURE_NAMES
from cpred.models.ensemble import CPredEnsemble
from cpred.training.evaluate import compute_metrics

# Reuse helpers from the training script
from prepare_and_train import (
    parse_dataset_s3,
    download_pdb,
    build_propensity_tables_from_data,
    _extract_features_worker,
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

        # Ensemble prediction
        probs = ensemble.predict(X_test)
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

    # Paper reference values
    print("\nPaper reference (Table 1, CPred column):")
    print("  AUC:         0.940")
    print("  Sensitivity: 0.889")
    print("  Specificity: 0.898")
    print("  MCC:         0.787")


def _print_summary(metrics_list: list[dict]) -> None:
    for key in ["auc", "sensitivity", "specificity", "accuracy", "mcc"]:
        values = [m[key] for m in metrics_list]
        print(f"  {key:>13s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


def main():
    parser = argparse.ArgumentParser(description="CPred 10-fold cross-validation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tables-dir", type=Path,
                        default=Path("cpred/data/propensity_tables"))
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    supp_dir = args.data_dir / "supplementary"
    pdb_dir = args.data_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Parse dataset ===
    print("=" * 60)
    print("STEP 1: Parsing Dataset S3")
    print("=" * 60)
    proteins_data = parse_dataset_s3(supp_dir)
    total_proteins = len(proteins_data)
    total_sites = sum(len(v["sites"]) for v in proteins_data.values())
    print(f"Found {total_proteins} proteins with {total_sites} CP sites")

    # === Step 2: Download PDBs ===
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 2: Downloading PDB structures")
        print("=" * 60)
        pdb_ids = list(proteins_data.keys())
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
        print(f"  Downloaded: {downloaded}, Failed: {failed}")

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
            proteins_data, pdb_dir, args.tables_dir, use_gpu=use_gpu)

    # === Step 4: Extract features ===
    print("\n" + "=" * 60)
    print("STEP 4: Extracting features")
    print("=" * 60)

    pdb_ids = list(proteins_data.keys())
    worker_args = [
        (pdb_id, proteins_data[pdb_id], str(pdb_dir / f"{pdb_id}.pdb"), tables)
        for pdb_id in pdb_ids
    ]

    all_X, all_y = [], []
    processed, skipped = 0, 0
    n_workers = min(8, os.cpu_count() or 1)
    print(f"  Extracting features with {n_workers} workers...", flush=True)

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_extract_features_worker, a): a[0]
                   for a in worker_args}
        for future in as_completed(futures):
            result, reason = future.result()
            completed += 1
            if result is None:
                skipped += 1
            else:
                X, y = result
                all_X.append(X)
                all_y.append(y)
                processed += 1
            if completed % 50 == 0 or completed == len(pdb_ids):
                print(f"  Progress: {completed}/{len(pdb_ids)} "
                      f"(processed: {processed}, skipped: {skipped})", flush=True)

    if not all_X:
        print("ERROR: No training data extracted!")
        sys.exit(1)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    n_pos = int(y_all.sum())
    n_neg = len(y_all) - n_pos
    print(f"\n  Full matrix: {X_all.shape}")
    print(f"  Positive: {n_pos}, Negative: {n_neg}")

    # === Step 5: Downsample ===
    neg_ratio = 2.4
    target_neg = int(n_pos * neg_ratio)
    if target_neg < n_neg:
        print(f"\n  Downsampling negatives: {n_neg} -> {target_neg} "
              f"(ratio {neg_ratio}:1)")
        rng = np.random.RandomState(42)
        pos_idx = np.where(y_all == 1)[0]
        neg_idx = np.where(y_all == 0)[0]
        sampled_neg = rng.choice(neg_idx, size=target_neg, replace=False)
        keep_idx = np.sort(np.concatenate([pos_idx, sampled_neg]))
        X_all = X_all[keep_idx]
        y_all = y_all[keep_idx]
        n_pos = int(y_all.sum())
        n_neg = len(y_all) - n_pos
        print(f"  After sampling: {X_all.shape}")
        print(f"  Positive: {n_pos}, Negative: {n_neg}")

    # === Step 6: 10-fold CV ===
    print("\n" + "=" * 60)
    print(f"STEP 6: {args.n_folds}-fold Cross-Validation")
    print("=" * 60)

    run_cv(X_all, y_all, n_folds=args.n_folds)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
