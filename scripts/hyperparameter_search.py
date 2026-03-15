#!/usr/bin/env python3
"""Automated hyperparameter search for CPred models.

Selects hyperparameters via 10-fold stratified cross-validation on Dataset T.
DHFR is used ONLY for final blind evaluation — never for model selection.

Usage:
    python scripts/hyperparameter_search.py
    python scripts/hyperparameter_search.py --ann-only       # just ANN search
    python scripts/hyperparameter_search.py --quick           # reduced grid
    python scripts/hyperparameter_search.py --n-folds 5       # 5-fold CV
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import builtins
import functools

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Force flush on every print (avoids buffering when output is redirected)
print = functools.partial(builtins.print, flush=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpred.io.pdb_parser import parse_pdb
from cpred.propensity.tables import PropensityTables
from cpred.pipeline import FEATURE_NAMES, build_feature_matrix, extract_all_features, get_feature_names
from cpred.features.standardization import standardize_features
from cpred.models.ann import CPredANN
from cpred.models.svm import CPredSVM
from cpred.models.random_forest import CPredRandomForest
from cpred.models.hierarchical import CPredHierarchical
from cpred.models.ensemble import smooth_predictions
from cpred.training.evaluate import compute_metrics

from sklearn.model_selection import StratifiedKFold


def extract_features_for_protein(protein, tables, rmsf_dir=None,
                                  include_rmsf=False, feature_names=None):
    """Extract and standardize all features for one protein."""
    features = extract_all_features(protein, tables, rmsf_dir=rmsf_dir,
                                     include_rmsf=include_rmsf)
    features = standardize_features(features)
    return build_feature_matrix(features, feature_names=feature_names)


def load_training_data(data_dir: Path, tables_dir: Path, rmsf_dir: Path,
                       include_rmsf: bool = False):
    """Load Dataset T training features and DHFR test features."""
    supp_dir = data_dir / "supplementary"
    pdb_dir = data_dir / "pdb"

    tables = PropensityTables(tables_dir)
    tables.load()

    # --- Dataset T (training) ---
    dataset_t_csv = supp_dir / "dataset_t.csv"
    df_t = pd.read_csv(dataset_t_csv)
    proteins = {}
    for _, row in df_t.iterrows():
        pdb_id = row['pdb_id']
        if pdb_id not in proteins:
            proteins[pdb_id] = {
                'chain': row['chain'],
                'sites_viable': [],
                'sites_inviable': [],
            }
        if row['viable'] == 1:
            proteins[pdb_id]['sites_viable'].append(int(row['residue_number']))
        else:
            proteins[pdb_id]['sites_inviable'].append(int(row['residue_number']))

    feat_names = get_feature_names(include_rmsf=include_rmsf)

    all_X, all_y = [], []
    rmsf_dir_str = str(rmsf_dir) if rmsf_dir else None
    for pdb_id, info in proteins.items():
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        protein = parse_pdb(pdb_path, chain_id=info["chain"])
        X = extract_features_for_protein(protein, tables, rmsf_dir=rmsf_dir_str,
                                          include_rmsf=include_rmsf,
                                          feature_names=feat_names)
        if X is None:
            continue
        resnum_to_idx = {rn: i for i, rn in enumerate(protein.residue_numbers)}
        labeled_indices, labels = [], []
        for site in info["sites_viable"]:
            idx = resnum_to_idx.get(site)
            if idx is not None:
                labeled_indices.append(idx)
                labels.append(1.0)
        for site in info["sites_inviable"]:
            idx = resnum_to_idx.get(site)
            if idx is not None:
                labeled_indices.append(idx)
                labels.append(0.0)
        all_X.append(X[labeled_indices])
        all_y.append(np.array(labels))

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    # --- DHFR (test) ---
    dhfr_path = pdb_dir / "1rx4.pdb"
    dhfr_csv = supp_dir / "dataset_dhfr.csv"
    dhfr = parse_pdb(dhfr_path, chain_id="A")
    X_dhfr_full = extract_features_for_protein(dhfr, tables, rmsf_dir=rmsf_dir_str,
                                                include_rmsf=include_rmsf,
                                                feature_names=feat_names)
    dhfr_df = pd.read_csv(dhfr_csv)
    resnum_to_idx = {rn: i for i, rn in enumerate(dhfr.residue_numbers)}
    labeled_indices, labels = [], []
    for _, row in dhfr_df.iterrows():
        idx = resnum_to_idx.get(int(row['residue_number']))
        if idx is not None:
            labeled_indices.append(idx)
            labels.append(int(row['viable']))
    y_dhfr = np.array(labels)

    return X_train, y_train, X_dhfr_full, labeled_indices, y_dhfr, tables


def evaluate_on_dhfr(probs_full, labeled_indices, y_dhfr, smooth=True):
    """Evaluate predictions on DHFR labeled sites."""
    if smooth:
        probs_full = smooth_predictions(probs_full)
    probs_labeled = probs_full[labeled_indices]
    return compute_metrics(y_dhfr, probs_labeled)


# =====================================================================
# Cross-validation helper
# =====================================================================
def cv_evaluate(model_factory, X, y, n_folds=10, random_state=42):
    """Evaluate a model via stratified k-fold CV. Returns mean metrics.

    Args:
        model_factory: callable() -> model with fit(X,y) and predict(X).
        X: (N, F) feature matrix.
        y: (N,) binary labels.
        n_folds: Number of CV folds.

    Returns:
        Dict with mean AUC, MCC, sensitivity, specificity across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)
    fold_metrics = []
    for train_idx, test_idx in skf.split(X, y):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict(X[test_idx])
        metrics = compute_metrics(y[test_idx], probs)
        fold_metrics.append(metrics)

    mean_metrics = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        mean_metrics[key] = float(np.mean(values))
        mean_metrics[f"{key}_std"] = float(np.std(values))
    return mean_metrics


# =====================================================================
# ANN Search
# =====================================================================
def search_ann(X_train, y_train, n_folds=10, quick=False):
    """Search ANN hyperparameters via CV on training data."""
    n_features = X_train.shape[1]
    sqrt_nf = max(round(math.sqrt(n_features)), 2)

    if quick:
        grid = {
            "lr": [0.05, 0.1, 0.5],
            "momentum": [0.0, 0.1, 0.5],
            "n_iterations": [5000, 20000],
            "hidden_size": [sqrt_nf, 2 * sqrt_nf, n_features],
        }
        n_restarts = 30
    else:
        grid = {
            "lr": [0.05, 0.1, 0.2, 0.3],
            "momentum": [0.0, 0.1, 0.2, 0.3, 0.5],
            "n_iterations": [10000, 20000, 50000],
            "hidden_size": [10, 14, 20, 28],
        }
        n_restarts = 50

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)
    print(f"\n{'='*60}")
    print(f"ANN SEARCH: {total} configurations, {n_restarts} restarts each")
    print(f"  Selection: {n_folds}-fold CV on Dataset T")
    print(f"{'='*60}")

    results = []
    best_auc = 0.0
    best_config = None

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        t0 = time.time()

        def make_ann(_params=params, _n_features=n_features,
                     _n_restarts=n_restarts):
            return CPredANN(
                n_features=_n_features,
                lr=_params["lr"],
                momentum=_params["momentum"],
                n_iterations=_params["n_iterations"],
                n_restarts=_n_restarts,
                hidden_size=_params["hidden_size"],
            )

        metrics = cv_evaluate(make_ann, X_train, y_train, n_folds=n_folds)
        elapsed = time.time() - t0

        entry = {**params, "n_restarts": n_restarts, **metrics, "time": elapsed}
        results.append(entry)

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_config = entry
        print(f"  [{i+1:3d}/{total}] CV-AUC={metrics['auc']:.4f}±{metrics['auc_std']:.3f} "
              f"CV-MCC={metrics['mcc']:.3f} "
              f"| lr={params['lr']} mom={params['momentum']} "
              f"iter={params['n_iterations']} hidden={params['hidden_size']} "
              f"({'NEW BEST ' if metrics['auc'] == best_auc else ''}best={best_auc:.4f}) "
              f"({elapsed:.1f}s)")

    results.sort(key=lambda r: r["auc"], reverse=True)
    print(f"\n  ANN Best (CV): AUC={best_config['auc']:.4f}±{best_config['auc_std']:.3f} "
          f"Sens={best_config['sensitivity']:.3f} "
          f"Spec={best_config['specificity']:.3f} "
          f"MCC={best_config['mcc']:.3f}")
    print(f"  Config: lr={best_config['lr']} mom={best_config['momentum']} "
          f"iter={best_config['n_iterations']} hidden={best_config['hidden_size']}")

    return results


# =====================================================================
# SVM Search
# =====================================================================
def search_svm(X_train, y_train, n_folds=10, quick=False):
    """Search SVM hyperparameters via CV on training data."""
    if quick:
        grid = {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": [0.001, 0.01, 0.1, "scale"],
        }
    else:
        grid = {
            "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
            "gamma": [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, "scale", "auto"],
        }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)
    print(f"\n{'='*60}")
    print(f"SVM SEARCH: {total} configurations")
    print(f"  Selection: {n_folds}-fold CV on Dataset T")
    print(f"{'='*60}")

    results = []
    best_auc = 0.0
    best_config = None

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        t0 = time.time()

        def make_svm(_params=params):
            return CPredSVM(C=_params["C"], gamma=_params["gamma"])

        # CV with grid_search=False inside SVM (we're doing the search ourselves)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []
        for train_idx, test_idx in skf.split(X_train, y_train):
            svm = make_svm()
            svm.fit(X_train[train_idx], y_train[train_idx], grid_search=False)
            probs = svm.predict(X_train[test_idx])
            m = compute_metrics(y_train[test_idx], probs)
            fold_metrics.append(m)

        metrics = {}
        for key in fold_metrics[0]:
            values_list = [m[key] for m in fold_metrics]
            metrics[key] = float(np.mean(values_list))
            metrics[f"{key}_std"] = float(np.std(values_list))

        elapsed = time.time() - t0

        entry = {
            "C": params["C"],
            "gamma": str(params["gamma"]),
            **metrics,
            "time": elapsed,
        }
        results.append(entry)

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_config = entry
        print(f"  [{i+1:3d}/{total}] CV-AUC={metrics['auc']:.4f}±{metrics['auc_std']:.3f} "
              f"CV-MCC={metrics['mcc']:.3f} "
              f"| C={params['C']} gamma={params['gamma']} "
              f"({'NEW BEST ' if metrics['auc'] == best_auc else ''}best={best_auc:.4f}) "
              f"({elapsed:.1f}s)")

    results.sort(key=lambda r: r["auc"], reverse=True)
    print(f"\n  SVM Best (CV): AUC={best_config['auc']:.4f}±{best_config['auc_std']:.3f} "
          f"Sens={best_config['sensitivity']:.3f} "
          f"Spec={best_config['specificity']:.3f} "
          f"MCC={best_config['mcc']:.3f}")
    print(f"  Config: C={best_config['C']} gamma={best_config['gamma']}")

    return results


# =====================================================================
# RF Search
# =====================================================================
def search_rf(X_train, y_train, n_folds=10, quick=False):
    """Search Random Forest hyperparameters via CV on training data."""
    if quick:
        grid = {
            "n_estimators_grow": [1000],
            "n_estimators_keep": [500],
            "max_features_frac": [0.3, 0.5, 0.7],
        }
    else:
        grid = {
            "n_estimators_grow": [500, 1000, 2000],
            "n_estimators_keep": [250, 500, 1000],
            "max_features_frac": [0.3, 0.5, 0.7, 1.0],
        }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    # Filter: keep <= grow
    combos = [(g, k, f) for g, k, f in combos if k <= g]
    total = len(combos)
    print(f"\n{'='*60}")
    print(f"RF SEARCH: {total} configurations")
    print(f"  Selection: {n_folds}-fold CV on Dataset T")
    print(f"{'='*60}")

    results = []
    best_auc = 0.0
    best_config = None

    for i, (n_grow, n_keep, mf_frac) in enumerate(combos):
        t0 = time.time()

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []
        for train_idx, test_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_te, y_te = X_train[test_idx], y_train[test_idx]

            rf = CPredRandomForest(
                n_estimators_grow=n_grow,
                n_estimators_keep=n_keep,
            )
            # Patch max_features for this configuration
            n_samples, n_features = X_tr.shape
            max_features = max(1, round(mf_frac * n_features))
            rng = np.random.RandomState(rf.random_state)
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import matthews_corrcoef
            tree_mccs = []
            for j in range(rf.n_estimators_grow):
                seed = int(rng.randint(0, 2**31))
                idx_boot = rng.choice(n_samples, size=n_samples, replace=True)
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[np.unique(idx_boot)] = False
                tree = DecisionTreeClassifier(
                    criterion="entropy",
                    max_features=max_features,
                    random_state=seed,
                )
                tree.fit(X_tr[idx_boot], y_tr[idx_boot])
                if oob_mask.sum() >= 2 and len(np.unique(y_tr[oob_mask])) == 2:
                    oob_pred = tree.predict(X_tr[oob_mask])
                    mcc = matthews_corrcoef(y_tr[oob_mask], oob_pred)
                else:
                    mcc = 0.0
                tree_mccs.append((mcc, tree))
            tree_mccs.sort(key=lambda x: x[0], reverse=True)
            rf.trees_ = [t for _, t in tree_mccs[:rf.n_estimators_keep]]
            rf._fitted = True

            probs = rf.predict(X_te)
            m = compute_metrics(y_te, probs)
            fold_metrics.append(m)

        metrics = {}
        for key in fold_metrics[0]:
            values_list = [m[key] for m in fold_metrics]
            metrics[key] = float(np.mean(values_list))
            metrics[f"{key}_std"] = float(np.std(values_list))

        elapsed = time.time() - t0

        entry = {
            "n_grow": n_grow, "n_keep": n_keep,
            "max_features_frac": mf_frac,
            **metrics, "time": elapsed,
        }
        results.append(entry)

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_config = entry
        print(f"  [{i+1:3d}/{total}] CV-AUC={metrics['auc']:.4f}±{metrics['auc_std']:.3f} "
              f"CV-MCC={metrics['mcc']:.3f} "
              f"| grow={n_grow} keep={n_keep} mf={mf_frac} "
              f"({'NEW BEST ' if metrics['auc'] == best_auc else ''}best={best_auc:.4f}) "
              f"({elapsed:.1f}s)")

    results.sort(key=lambda r: r["auc"], reverse=True)
    print(f"\n  RF Best (CV): AUC={best_config['auc']:.4f}±{best_config['auc_std']:.3f} "
          f"Sens={best_config['sensitivity']:.3f} "
          f"Spec={best_config['specificity']:.3f} "
          f"MCC={best_config['mcc']:.3f}")
    print(f"  Config: grow={best_config['n_grow']} keep={best_config['n_keep']} "
          f"mf={best_config['max_features_frac']}")

    return results


# =====================================================================
# Final evaluation: retrain on full Dataset T, blind test on DHFR
# =====================================================================
def final_evaluation(X_train, y_train, X_dhfr_full, labeled_indices, y_dhfr,
                     best_ann_params, best_svm_params, best_rf_params,
                     output_dir: Path = Path("results/best_models"),
                     feature_names: list[str] | None = None):
    """Retrain best configs on full Dataset T, report blind DHFR results."""
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION: retrain on full Dataset T, blind test on DHFR")
    print(f"{'='*60}")

    n_features = X_train.shape[1]

    # Train best ANN on full training set
    print("  Training best ANN on full Dataset T...")
    ann = CPredANN(
        n_features=n_features,
        lr=best_ann_params["lr"],
        momentum=best_ann_params["momentum"],
        n_iterations=best_ann_params["n_iterations"],
        n_restarts=best_ann_params.get("n_restarts", 50),
        hidden_size=best_ann_params["hidden_size"],
    )
    ann.fit(X_train, y_train)
    p_ann = ann.predict(X_dhfr_full)

    # Train best SVM
    print("  Training best SVM on full Dataset T...")
    gamma = best_svm_params["gamma"]
    if gamma not in ("scale", "auto"):
        gamma = float(gamma)
    svm = CPredSVM(C=best_svm_params["C"], gamma=gamma)
    svm.fit(X_train, y_train, grid_search=False)
    p_svm = svm.predict(X_dhfr_full)

    # Train best RF
    print("  Training best RF on full Dataset T...")
    rf = CPredRandomForest(
        n_estimators_grow=best_rf_params["n_grow"],
        n_estimators_keep=best_rf_params["n_keep"],
    )
    n_samples_train = X_train.shape[0]
    max_features = max(1, round(best_rf_params["max_features_frac"] * n_features))
    rng = np.random.RandomState(rf.random_state)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import matthews_corrcoef
    tree_mccs = []
    for j in range(rf.n_estimators_grow):
        seed = int(rng.randint(0, 2**31))
        idx_boot = rng.choice(n_samples_train, size=n_samples_train, replace=True)
        oob_mask = np.ones(n_samples_train, dtype=bool)
        oob_mask[np.unique(idx_boot)] = False
        tree = DecisionTreeClassifier(
            criterion="entropy", max_features=max_features, random_state=seed)
        tree.fit(X_train[idx_boot], y_train[idx_boot])
        if oob_mask.sum() >= 2 and len(np.unique(y_train[oob_mask])) == 2:
            oob_pred = tree.predict(X_train[oob_mask])
            mcc = matthews_corrcoef(y_train[oob_mask], oob_pred)
        else:
            mcc = 0.0
        tree_mccs.append((mcc, tree))
    tree_mccs.sort(key=lambda x: x[0], reverse=True)
    rf.trees_ = [t for _, t in tree_mccs[:rf.n_estimators_keep]]
    rf._fitted = True
    p_rf = rf.predict(X_dhfr_full)

    # HI (fixed weights, no hyperparameters to tune)
    print("  Training HI on full Dataset T...")
    feat_names = feature_names if feature_names is not None else FEATURE_NAMES
    hi = CPredHierarchical(feature_names=feat_names)
    hi.fit(X_train, y_train, feature_names=feat_names)
    p_hi = hi.predict(X_dhfr_full)

    # Individual model DHFR results (blind)
    print("\n  DHFR blind test results (per model):")
    for name, probs in [("ANN", p_ann), ("SVM", p_svm), ("RF", p_rf), ("HI", p_hi)]:
        m = evaluate_on_dhfr(probs, labeled_indices, y_dhfr, smooth=False)
        print(f"    [{name:3s}] AUC={m['auc']:.4f}  Sens={m['sensitivity']:.3f}  "
              f"Spec={m['specificity']:.3f}  MCC={m['mcc']:.3f}")

    # Ensemble: simple average + smoothing (paper method)
    avg = (p_ann + p_svm + p_rf + p_hi) / 4.0
    m_ens = evaluate_on_dhfr(avg, labeled_indices, y_dhfr, smooth=True)
    print(f"\n  ENSEMBLE (avg+smooth): AUC={m_ens['auc']:.4f}  "
          f"Sens={m_ens['sensitivity']:.3f}  Spec={m_ens['specificity']:.3f}  "
          f"MCC={m_ens['mcc']:.3f}")

    print(f"\n  Paper reference (DHFR):")
    print(f"    AUC=0.906  Sens=0.709  Spec=0.918  MCC=0.633")

    # Save models
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ann.save(output_dir / "ann.pt")
    svm.save(output_dir / "svm.pkl")
    rf.save(output_dir / "rf.pkl")
    hi.save(output_dir / "hi.json")
    print(f"\n  Models saved to {output_dir}/")

    return {"ensemble": m_ens}


def main():
    parser = argparse.ArgumentParser(description="CPred hyperparameter search")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tables-dir", type=Path,
                        default=Path("cpred/data/propensity_tables"))
    parser.add_argument("--rmsf-dir", type=Path, default=Path("data/rmsf"))
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced search grid")
    parser.add_argument("--ann-only", action="store_true",
                        help="Only search ANN hyperparameters")
    parser.add_argument("--svm-only", action="store_true",
                        help="Only search SVM hyperparameters")
    parser.add_argument("--rf-only", action="store_true",
                        help="Only search RF hyperparameters")
    parser.add_argument("--n-folds", type=int, default=10,
                        help="Number of CV folds (default: 10)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path")
    parser.add_argument("--include-rmsf", action="store_true",
                        help="Include RMSF as a feature (47 features instead of 46)")
    args = parser.parse_args()

    # Auto-set output path
    if args.output is None:
        suffix = "_rmsf" if args.include_rmsf else ""
        args.output = Path(f"results/hyperparameter_search_cv{suffix}.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    feat_names = get_feature_names(include_rmsf=args.include_rmsf)

    print("Loading training data and extracting features...")
    t0 = time.time()
    X_train, y_train, X_dhfr_full, labeled_indices, y_dhfr, tables = \
        load_training_data(args.data_dir, args.tables_dir, args.rmsf_dir,
                           include_rmsf=args.include_rmsf)
    print(f"  Training: {X_train.shape}, Test (DHFR full): {X_dhfr_full.shape}, "
          f"DHFR labeled: {len(y_dhfr)} ({int(y_dhfr.sum())} viable)")
    print(f"  Feature extraction took {time.time()-t0:.1f}s")
    print(f"  Hyperparameter selection: {args.n_folds}-fold stratified CV on Dataset T")
    print(f"  DHFR used ONLY for final blind evaluation")

    search_all = not (args.ann_only or args.svm_only or args.rf_only)
    all_results = {}

    # --- ANN Search ---
    if search_all or args.ann_only:
        ann_results = search_ann(X_train, y_train, n_folds=args.n_folds,
                                  quick=args.quick)
        all_results["ann"] = ann_results[:20]  # top 20

    # --- SVM Search ---
    if search_all or args.svm_only:
        svm_results = search_svm(X_train, y_train, n_folds=args.n_folds,
                                  quick=args.quick)
        all_results["svm"] = svm_results[:20]

    # --- RF Search ---
    if search_all or args.rf_only:
        rf_results = search_rf(X_train, y_train, n_folds=args.n_folds,
                                quick=args.quick)
        all_results["rf"] = rf_results[:20]

    # --- Final: retrain best on full Dataset T, blind test on DHFR ---
    if search_all:
        best_ann = ann_results[0]
        best_svm = svm_results[0]
        best_rf = rf_results[0]

        models_dir = "best_models_rmsf" if args.include_rmsf else "best_models"
        ens_results = final_evaluation(
            X_train, y_train, X_dhfr_full, labeled_indices, y_dhfr,
            best_ann, best_svm, best_rf,
            output_dir=args.output.parent / models_dir,
            feature_names=feat_names)
        all_results["final_dhfr"] = ens_results

    # Save results
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {args.output}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Selection method: {args.n_folds}-fold CV on Dataset T")
    if "ann" in all_results and all_results["ann"]:
        a = all_results["ann"][0]
        print(f"  Best ANN (CV):  AUC={a['auc']:.4f}±{a['auc_std']:.3f}  "
              f"MCC={a['mcc']:.3f}  "
              f"lr={a['lr']} mom={a['momentum']} "
              f"iter={a['n_iterations']} hidden={a['hidden_size']}")
    if "svm" in all_results and all_results["svm"]:
        s = all_results["svm"][0]
        print(f"  Best SVM (CV):  AUC={s['auc']:.4f}±{s['auc_std']:.3f}  "
              f"MCC={s['mcc']:.3f}  "
              f"C={s['C']} gamma={s['gamma']}")
    if "rf" in all_results and all_results["rf"]:
        r = all_results["rf"][0]
        print(f"  Best RF  (CV):  AUC={r['auc']:.4f}±{r['auc_std']:.3f}  "
              f"MCC={r['mcc']:.3f}  "
              f"grow={r['n_grow']} keep={r['n_keep']} mf={r['max_features_frac']}")
    if "final_dhfr" in all_results:
        d = all_results["final_dhfr"]["ensemble"]
        print(f"\n  DHFR blind test (ensemble): AUC={d['auc']:.4f}  "
              f"Sens={d['sensitivity']:.3f}  Spec={d['specificity']:.3f}  "
              f"MCC={d['mcc']:.3f}")
    print(f"\n  Paper:          AUC=0.906  Sens=0.709  Spec=0.918  MCC=0.633")


if __name__ == "__main__":
    main()
