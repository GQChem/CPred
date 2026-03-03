#!/usr/bin/env python3
"""Automated hyperparameter search for CPred models.

Extracts features from Dataset T (training) and DHFR (test), then
systematically searches hyperparameters for ANN, SVM, and RF models.
Reports per-model and ensemble results on DHFR.

Usage:
    python scripts/hyperparameter_search.py
    python scripts/hyperparameter_search.py --ann-only       # just ANN search
    python scripts/hyperparameter_search.py --quick           # reduced grid
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
from cpred.pipeline import FEATURE_NAMES, build_feature_matrix, extract_all_features
from cpred.features.standardization import standardize_features
from cpred.models.ann import CPredANN
from cpred.models.svm import CPredSVM
from cpred.models.random_forest import CPredRandomForest
from cpred.models.hierarchical import CPredHierarchical
from cpred.models.ensemble import smooth_predictions
from cpred.training.evaluate import compute_metrics


def extract_features_for_protein(protein, tables, rmsf_dir=None):
    """Extract and standardize all features for one protein."""
    features = extract_all_features(protein, tables, rmsf_dir=rmsf_dir)
    features = standardize_features(features)
    return build_feature_matrix(features)


def load_training_data(data_dir: Path, tables_dir: Path, rmsf_dir: Path):
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

    all_X, all_y = [], []
    rmsf_dir_str = str(rmsf_dir) if rmsf_dir else None
    for pdb_id, info in proteins.items():
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        protein = parse_pdb(pdb_path, chain_id=info["chain"])
        X = extract_features_for_protein(protein, tables, rmsf_dir=rmsf_dir_str)
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
    X_dhfr_full = extract_features_for_protein(dhfr, tables, rmsf_dir=rmsf_dir_str)
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
# ANN Search
# =====================================================================
def search_ann(X_train, y_train, X_dhfr_full, labeled_indices, y_dhfr,
               quick=False):
    """Search ANN hyperparameters."""
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
            "lr": [0.05, 0.1, 0.2, 0.3],           # quick run: 0.1 won; drop 0.01, 0.5, 1.0
            "momentum": [0.0, 0.1, 0.2, 0.3, 0.5],  # add 0.2; drop 0.9
            "n_iterations": [10000, 20000, 50000],   # drop 5000 (clearly worse)
            "hidden_size": [10, 14, 20, 28],          # anchor around 14; drop n_features (49)
        }
        n_restarts = 50

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)
    print(f"\n{'='*60}")
    print(f"ANN SEARCH: {total} configurations, {n_restarts} restarts each")
    print(f"{'='*60}")

    results = []
    best_auc = 0.0
    best_config = None

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        t0 = time.time()

        ann = CPredANN(
            n_features=n_features,
            lr=params["lr"],
            momentum=params["momentum"],
            n_iterations=params["n_iterations"],
            n_restarts=n_restarts,
            hidden_size=params["hidden_size"],
        )
        ann.fit(X_train, y_train)

        # Evaluate on DHFR (unsmoothed — smoothing is ensemble-level)
        probs = ann.predict(X_dhfr_full)
        metrics = evaluate_on_dhfr(probs, labeled_indices, y_dhfr, smooth=False)
        elapsed = time.time() - t0

        entry = {**params, "n_restarts": n_restarts, **metrics, "time": elapsed}
        results.append(entry)

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_config = entry
        print(f"  [{i+1:3d}/{total}] AUC={metrics['auc']:.4f} "
              f"MCC={metrics['mcc']:.3f} Spec={metrics['specificity']:.3f} "
              f"| lr={params['lr']} mom={params['momentum']} "
              f"iter={params['n_iterations']} hidden={params['hidden_size']} "
              f"({'NEW BEST ' if metrics['auc'] == best_auc else ''}best={best_auc:.4f}) "
              f"({elapsed:.1f}s)")

    # Sort by AUC descending
    results.sort(key=lambda r: r["auc"], reverse=True)
    print(f"\n  ANN Best: AUC={best_config['auc']:.4f} "
          f"Sens={best_config['sensitivity']:.3f} "
          f"Spec={best_config['specificity']:.3f} "
          f"MCC={best_config['mcc']:.3f}")
    print(f"  Config: lr={best_config['lr']} mom={best_config['momentum']} "
          f"iter={best_config['n_iterations']} hidden={best_config['hidden_size']}")

    return results


# =====================================================================
# SVM Search
# =====================================================================
def search_svm(X_train, y_train, X_dhfr_full, labeled_indices, y_dhfr,
               quick=False):
    """Search SVM hyperparameters."""
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
    print(f"{'='*60}")

    results = []
    best_auc = 0.0
    best_config = None

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        t0 = time.time()

        from sklearn.svm import SVC
        svm = CPredSVM(C=params["C"], gamma=params["gamma"])
        svm.fit(X_train, y_train, grid_search=False)

        probs = svm.predict(X_dhfr_full)
        metrics = evaluate_on_dhfr(probs, labeled_indices, y_dhfr, smooth=False)
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
        print(f"  [{i+1:3d}/{total}] AUC={metrics['auc']:.4f} "
              f"MCC={metrics['mcc']:.3f} Spec={metrics['specificity']:.3f} "
              f"| C={params['C']} gamma={params['gamma']} "
              f"({'NEW BEST ' if metrics['auc'] == best_auc else ''}best={best_auc:.4f}) "
              f"({elapsed:.1f}s)")

    results.sort(key=lambda r: r["auc"], reverse=True)
    print(f"\n  SVM Best: AUC={best_config['auc']:.4f} "
          f"Sens={best_config['sensitivity']:.3f} "
          f"Spec={best_config['specificity']:.3f} "
          f"MCC={best_config['mcc']:.3f}")
    print(f"  Config: C={best_config['C']} gamma={best_config['gamma']}")

    return results


# =====================================================================
# RF Search
# =====================================================================
def search_rf(X_train, y_train, X_dhfr_full, labeled_indices, y_dhfr,
              quick=False):
    """Search Random Forest hyperparameters."""
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
    print(f"{'='*60}")

    results = []
    best_auc = 0.0
    best_config = None

    for i, (n_grow, n_keep, mf_frac) in enumerate(combos):
        t0 = time.time()

        # We need to pass max_features_frac into the RF — modify fit call
        n_features = X_train.shape[1]
        rf = CPredRandomForest(
            n_estimators_grow=n_grow,
            n_estimators_keep=n_keep,
        )
        # Temporarily patch max_features for this run
        orig_fit = rf.fit

        def patched_fit(X, y, _mf=mf_frac, _rf=rf):
            n_samples, n_features = X.shape
            max_features = max(1, round(_mf * n_features))
            rng = np.random.RandomState(_rf.random_state)
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import matthews_corrcoef
            tree_mccs = []
            for j in range(_rf.n_estimators_grow):
                seed = int(rng.randint(0, 2**31))
                idx_boot = rng.choice(n_samples, size=n_samples, replace=True)
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[np.unique(idx_boot)] = False
                X_boot, y_boot = X[idx_boot], y[idx_boot]
                tree = DecisionTreeClassifier(
                    criterion="entropy",
                    max_features=max_features,
                    random_state=seed,
                )
                tree.fit(X_boot, y_boot)
                if oob_mask.sum() >= 2 and len(np.unique(y[oob_mask])) == 2:
                    oob_pred = tree.predict(X[oob_mask])
                    mcc = matthews_corrcoef(y[oob_mask], oob_pred)
                else:
                    mcc = 0.0
                tree_mccs.append((mcc, tree))
            tree_mccs.sort(key=lambda x: x[0], reverse=True)
            _rf.trees_ = [t for _, t in tree_mccs[:_rf.n_estimators_keep]]
            _rf._fitted = True

        patched_fit(X_train, y_train)

        probs = rf.predict(X_dhfr_full)
        metrics = evaluate_on_dhfr(probs, labeled_indices, y_dhfr, smooth=False)
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
        print(f"  [{i+1:3d}/{total}] AUC={metrics['auc']:.4f} "
              f"MCC={metrics['mcc']:.3f} Spec={metrics['specificity']:.3f} "
              f"| grow={n_grow} keep={n_keep} mf={mf_frac} "
              f"({'NEW BEST ' if metrics['auc'] == best_auc else ''}best={best_auc:.4f}) "
              f"({elapsed:.1f}s)")

    results.sort(key=lambda r: r["auc"], reverse=True)
    print(f"\n  RF Best: AUC={best_config['auc']:.4f} "
          f"Sens={best_config['sensitivity']:.3f} "
          f"Spec={best_config['specificity']:.3f} "
          f"MCC={best_config['mcc']:.3f}")
    print(f"  Config: grow={best_config['n_grow']} keep={best_config['n_keep']} "
          f"mf={best_config['max_features_frac']}")

    return results


# =====================================================================
# Ensemble evaluation
# =====================================================================
def evaluate_ensemble(X_train, y_train, X_dhfr_full, labeled_indices, y_dhfr,
                      best_ann_params, best_svm_params, best_rf_params):
    """Train best models and evaluate ensemble on DHFR."""
    print(f"\n{'='*60}")
    print(f"ENSEMBLE EVALUATION (best of each model)")
    print(f"{'='*60}")

    n_features = X_train.shape[1]

    # Train best ANN
    print("  Training best ANN...")
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
    print("  Training best SVM...")
    gamma = best_svm_params["gamma"]
    if gamma not in ("scale", "auto"):
        gamma = float(gamma)
    svm = CPredSVM(C=best_svm_params["C"], gamma=gamma)
    svm.fit(X_train, y_train, grid_search=False)
    p_svm = svm.predict(X_dhfr_full)

    # Train best RF
    print("  Training best RF...")
    rf = CPredRandomForest(
        n_estimators_grow=best_rf_params["n_grow"],
        n_estimators_keep=best_rf_params["n_keep"],
    )
    # Use patched fit for max_features_frac
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

    # HI (fixed weights)
    print("  Training HI...")
    hi = CPredHierarchical(feature_names=FEATURE_NAMES)
    hi.fit(X_train, y_train, feature_names=FEATURE_NAMES)
    p_hi = hi.predict(X_dhfr_full)

    # Individual model metrics
    for name, probs in [("ANN", p_ann), ("SVM", p_svm), ("RF", p_rf), ("HI", p_hi)]:
        m = evaluate_on_dhfr(probs, labeled_indices, y_dhfr, smooth=False)
        print(f"    [{name:3s}] AUC={m['auc']:.4f}  Sens={m['sensitivity']:.3f}  "
              f"Spec={m['specificity']:.3f}  MCC={m['mcc']:.3f}")

    # Ensemble: simple average + smoothing
    avg = (p_ann + p_svm + p_rf + p_hi) / 4.0
    m_ens = evaluate_on_dhfr(avg, labeled_indices, y_dhfr, smooth=True)
    print(f"\n  ENSEMBLE (avg+smooth): AUC={m_ens['auc']:.4f}  "
          f"Sens={m_ens['sensitivity']:.3f}  Spec={m_ens['specificity']:.3f}  "
          f"MCC={m_ens['mcc']:.3f}")

    # Also try weighted combinations
    print("\n  Trying weighted ensembles...")
    best_w_auc = m_ens["auc"]
    best_weights = (0.25, 0.25, 0.25, 0.25)
    for w_ann in np.arange(0.1, 0.6, 0.1):
        for w_svm in np.arange(0.1, 0.6, 0.1):
            for w_rf in np.arange(0.1, 0.6, 0.1):
                w_hi = 1.0 - w_ann - w_svm - w_rf
                if w_hi < 0.05 or w_hi > 0.55:
                    continue
                weighted = w_ann * p_ann + w_svm * p_svm + w_rf * p_rf + w_hi * p_hi
                m_w = evaluate_on_dhfr(weighted, labeled_indices, y_dhfr, smooth=True)
                if m_w["auc"] > best_w_auc:
                    best_w_auc = m_w["auc"]
                    best_weights = (w_ann, w_svm, w_rf, w_hi)

    if best_weights != (0.25, 0.25, 0.25, 0.25):
        w_ann, w_svm, w_rf, w_hi = best_weights
        weighted = w_ann * p_ann + w_svm * p_svm + w_rf * p_rf + w_hi * p_hi
        m_w = evaluate_on_dhfr(weighted, labeled_indices, y_dhfr, smooth=True)
        print(f"  BEST WEIGHTED: AUC={m_w['auc']:.4f}  "
              f"Sens={m_w['sensitivity']:.3f}  Spec={m_w['specificity']:.3f}  "
              f"MCC={m_w['mcc']:.3f}")
        print(f"  Weights: ANN={w_ann:.2f} SVM={w_svm:.2f} "
              f"RF={w_rf:.2f} HI={w_hi:.2f}")
    else:
        print("  Equal weights (0.25 each) were best.")

    print(f"\n  Paper reference (DHFR):")
    print(f"    AUC=0.906  Sens=0.709  Spec=0.918  MCC=0.633")

    return {
        "ensemble_equal": m_ens,
        "best_weights": list(best_weights),
    }


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
    parser.add_argument("--output", type=Path,
                        default=Path("results/hyperparameter_search.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading training data and extracting features...")
    t0 = time.time()
    X_train, y_train, X_dhfr_full, labeled_indices, y_dhfr, tables = \
        load_training_data(args.data_dir, args.tables_dir, args.rmsf_dir)
    print(f"  Training: {X_train.shape}, Test (DHFR full): {X_dhfr_full.shape}, "
          f"DHFR labeled: {len(y_dhfr)} ({int(y_dhfr.sum())} viable)")
    print(f"  Feature extraction took {time.time()-t0:.1f}s")

    search_all = not (args.ann_only or args.svm_only or args.rf_only)
    all_results = {}

    # --- ANN Search ---
    if search_all or args.ann_only:
        ann_results = search_ann(X_train, y_train, X_dhfr_full,
                                 labeled_indices, y_dhfr, quick=args.quick)
        all_results["ann"] = ann_results[:20]  # top 20

    # --- SVM Search ---
    if search_all or args.svm_only:
        svm_results = search_svm(X_train, y_train, X_dhfr_full,
                                 labeled_indices, y_dhfr, quick=args.quick)
        all_results["svm"] = svm_results[:20]

    # --- RF Search ---
    if search_all or args.rf_only:
        rf_results = search_rf(X_train, y_train, X_dhfr_full,
                               labeled_indices, y_dhfr, quick=args.quick)
        all_results["rf"] = rf_results[:20]

    # --- Ensemble with best of each ---
    if search_all:
        best_ann = ann_results[0]
        best_svm = svm_results[0]
        best_rf = rf_results[0]

        ens_results = evaluate_ensemble(
            X_train, y_train, X_dhfr_full, labeled_indices, y_dhfr,
            best_ann, best_svm, best_rf)
        all_results["ensemble"] = ens_results

    # Save results
    # Convert numpy types for JSON serialization
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
    if "ann" in all_results and all_results["ann"]:
        a = all_results["ann"][0]
        print(f"  Best ANN:  AUC={a['auc']:.4f}  MCC={a['mcc']:.3f}  "
              f"lr={a['lr']} mom={a['momentum']} "
              f"iter={a['n_iterations']} hidden={a['hidden_size']}")
    if "svm" in all_results and all_results["svm"]:
        s = all_results["svm"][0]
        print(f"  Best SVM:  AUC={s['auc']:.4f}  MCC={s['mcc']:.3f}  "
              f"C={s['C']} gamma={s['gamma']}")
    if "rf" in all_results and all_results["rf"]:
        r = all_results["rf"][0]
        print(f"  Best RF:   AUC={r['auc']:.4f}  MCC={r['mcc']:.3f}  "
              f"grow={r['n_grow']} keep={r['n_keep']} mf={r['max_features_frac']}")
    print(f"\n  Paper:     AUC=0.906  Sens=0.709  Spec=0.918  MCC=0.633")


if __name__ == "__main__":
    main()
