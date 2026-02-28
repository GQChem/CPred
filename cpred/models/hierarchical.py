"""Hierarchical Inference (HI) model for CP site prediction.

Tree-structured weighted feature combination. Features are grouped into
categories, each category produces a sub-score, and sub-scores are
combined via optimized weights.

Tree structure (Lo et al. 2012, Figure 4):
  Level 1: Individual feature weights within groups
  Level 2: Category scores (Cat A=sequence, Cat B=SS, Cat C=tertiary)
  Level 3: Final IF score (weighted average of category scores)

Paper weights: 1dprop=0.14, 2dprop=0.57, 3dprop=0.29
               solacc=0.04, awaycore=0.61

Optimization uses coordinate descent over all weight levels for maximum AUC.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _build_default_tree(feature_names: list[str] | None = None) -> dict:
    """Build default HI tree, auto-detecting expanded feature names."""
    if feature_names is None:
        feature_names = []

    seq_features = [f for f in feature_names if f.startswith(("prop_aa", "prop_di", "prop_oligo"))]
    ss_features = [f for f in feature_names if f.startswith(("prop_dssp", "prop_rama", "prop_kappa"))]
    tert_features = [f for f in feature_names if f in ("rsa", "bfactor")]
    contact_features = [f for f in feature_names if f in ("closeness",)]
    dyn_features = [f for f in feature_names if f in ("gnm_msf",)]

    if not seq_features:
        seq_features = ["prop_aa", "prop_di", "prop_oligo"]
    if not ss_features:
        ss_features = ["prop_dssp", "prop_rama", "prop_kappa_alpha"]
    if not tert_features:
        tert_features = ["rsa", "bfactor"]
    if not contact_features:
        contact_features = ["closeness"]
    if not dyn_features:
        dyn_features = ["gnm_msf"]

    return {
        "groups": {
            "seq_propensity": {
                "features": seq_features,
                "weight": 0.14,
                "feature_weights": {f: 1.0 for f in seq_features},
            },
            "ss_propensity": {
                "features": ss_features,
                "weight": 0.57,
                "feature_weights": {f: 1.0 for f in ss_features},
            },
            "tertiary_packing": {
                "features": tert_features,
                "weight": 0.04,
                "feature_weights": {f: 1.0 for f in tert_features},
            },
            "contact_network": {
                "features": contact_features,
                "weight": 0.61,
                "feature_weights": {f: 1.0 for f in contact_features},
            },
            "dynamics": {
                "features": dyn_features,
                "weight": 0.29,
                "feature_weights": {f: 1.0 for f in dyn_features},
            },
        },
        "category_weights": {
            "cat_a": {"groups": ["seq_propensity"], "weight": 0.14},
            "cat_b": {"groups": ["ss_propensity"], "weight": 0.57},
            "cat_c": {"groups": ["tertiary_packing", "contact_network", "dynamics"],
                      "weight": 0.29},
        },
    }


DEFAULT_TREE = _build_default_tree()


class CPredHierarchical:
    """Hierarchical Inference model for CP site prediction."""

    def __init__(self, tree: dict | None = None,
                 feature_names: list[str] | None = None):
        self.feature_names = feature_names or []
        self.tree = tree or _build_default_tree(self.feature_names)
        self._fitted = False

    def _feature_index(self, name: str) -> int | None:
        try:
            return self.feature_names.index(name)
        except ValueError:
            return None

    def _compute_group_score(self, X: np.ndarray,
                             group: dict) -> np.ndarray:
        """Compute weighted average score for a feature group."""
        indices = []
        weights = []
        feat_weights = group.get("feature_weights", {})
        for feat_name in group["features"]:
            idx = self._feature_index(feat_name)
            if idx is not None:
                indices.append(idx)
                weights.append(feat_weights.get(feat_name, 1.0))
        if not indices:
            return np.zeros(X.shape[0])
        w = np.array(weights)
        total_w = w.sum()
        if total_w < 1e-10:
            return np.zeros(X.shape[0])
        return (X[:, indices] * w[np.newaxis, :]).sum(axis=1) / total_w

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Compute raw HI score (before sigmoid)."""
        groups = self.tree["groups"]
        group_scores = {}
        for gname, gconf in groups.items():
            group_scores[gname] = self._compute_group_score(X, gconf) * gconf["weight"]

        cat_weights = self.tree.get("category_weights", {})
        if cat_weights:
            cat_scores = []
            total_weight = 0
            for cname, cconf in cat_weights.items():
                cat_score = np.zeros(X.shape[0])
                for gname in cconf["groups"]:
                    if gname in group_scores:
                        cat_score += group_scores[gname]
                n_groups = len(cconf["groups"])
                if n_groups > 0:
                    cat_score /= n_groups
                cat_scores.append(cat_score * cconf["weight"])
                total_weight += cconf["weight"]
            final = sum(cat_scores) / max(total_weight, 1e-10)
        else:
            all_scores = list(group_scores.values())
            final = sum(all_scores) / max(len(all_scores), 1)

        return final

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CP viability probabilities."""
        raw = self.predict_raw(X)
        return 1 / (1 + np.exp(-raw))

    def _eval_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        """Helper to compute AUC with current weights."""
        probs = self.predict(X)
        try:
            return roc_auc_score(y, probs)
        except ValueError:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None,
            n_cd_rounds: int = 3) -> None:
        """Optimize tree weights via coordinate descent on training AUC.

        Three-level optimization:
        1. Category weights (coarse grid search)
        2. Group weights (coordinate descent)
        3. Per-feature weights within groups (coordinate descent)

        Repeats n_cd_rounds times for convergence.
        """
        if feature_names is not None:
            self.feature_names = feature_names
            self.tree = _build_default_tree(self.feature_names)

        cat_names = list(self.tree.get("category_weights", {}).keys())
        if not cat_names:
            self._fitted = True
            return

        # Weight options for different levels
        cat_weight_options = [round(x * 0.1, 1) for x in range(1, 21)]  # 0.1-2.0
        group_weight_options = [round(x * 0.1, 1) for x in range(1, 21)]
        feat_weight_options = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

        for cd_round in range(n_cd_rounds):
            # --- Level 1: Category weight grid search ---
            best_auc = 0
            best_cat_weights = {}
            for weights in product(cat_weight_options, repeat=len(cat_names)):
                for cname, w in zip(cat_names, weights):
                    self.tree["category_weights"][cname]["weight"] = w
                auc = self._eval_auc(X, y)
                if auc > best_auc:
                    best_auc = auc
                    best_cat_weights = {cn: w for cn, w in zip(cat_names, weights)}

            for cname, w in best_cat_weights.items():
                self.tree["category_weights"][cname]["weight"] = w

            # --- Level 2: Group weight coordinate descent ---
            for gname in self.tree["groups"]:
                best_gw = self.tree["groups"][gname]["weight"]
                best_g_auc = self._eval_auc(X, y)
                for gw in group_weight_options:
                    self.tree["groups"][gname]["weight"] = gw
                    auc = self._eval_auc(X, y)
                    if auc > best_g_auc:
                        best_g_auc = auc
                        best_gw = gw
                self.tree["groups"][gname]["weight"] = best_gw

            # --- Level 3: Per-feature weight coordinate descent ---
            for gname, gconf in self.tree["groups"].items():
                feat_weights = gconf.get("feature_weights", {})
                if len(feat_weights) <= 1:
                    continue  # nothing to optimize for single-feature groups
                for fname in gconf["features"]:
                    if fname not in feat_weights:
                        continue
                    best_fw = feat_weights[fname]
                    best_f_auc = self._eval_auc(X, y)
                    for fw in feat_weight_options:
                        feat_weights[fname] = fw
                        auc = self._eval_auc(X, y)
                        if auc > best_f_auc:
                            best_f_auc = auc
                            best_fw = fw
                    feat_weights[fname] = best_fw

        self._fitted = True

    def save(self, path: str | Path) -> None:
        data = {
            "tree": self.tree,
            "feature_names": self.feature_names,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self.tree = data["tree"]
        self.feature_names = data.get("feature_names", [])
        self._fitted = True
