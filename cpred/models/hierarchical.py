"""Hierarchical Inference (HI) model for CP site prediction.

Implements the hierarchical feature integration tree from Lo et al. 2012,
Figure 4. Features are organized into a fixed tree structure where each
node combines its children via weighted averaging. The root IF score is
converted to a probability using the training set distribution.

Tree structure (3 categories → root IF):
  Cat A (1dprop, weight 0.14):
    R_1dprop (R_aa=0.27, R_aac3=0.70, R_aac5=0.03) weight 0.01
    2R_1dprop (2R_aa=0.26, 2R_aac3=0.47, 2R_aac5=0.27) weight 0.01
    RxR_1dprop (RxR_aa=0.10, RxR_aac3=0.89, RxR_aac5=0.01) weight 0.75
    → 1dprop (R_1dprop, 2R_1dprop, RxR_1dprop, nR_1dprop) weight 0.14
    ...etc per Figure 4
  Cat B (2dprop, weight 0.57)
  Cat C (3dprop, weight 0.29):
    Solacc (RSA=1.00) weight 0.04
    Eccent (CM=0.70, DPX=0.30) weight 0.14
    Awaybury (Fb=0.98, DISb=0.02) → Awaycore with Awayhpho weight 0.61
    Nhbonds (Hbonds=1.00) weight 0.15
    Uncrowd (WCN=0.22, CN=0.69, Closeness=0.09) weight 0.19
    Flex (GNM-F=1.00) weight 0.01
    → 3dprop weight 0.29

Paper weights are FIXED, not optimized per fold (the paper optimized once
on the full Dataset T using exhaustive search evaluated by 10-fold averaged
MCC, then fixed the weights).

Probability conversion: count N_p (positive training IF >= IF_i) and
N_n (negative training IF <= IF_i), then P = N_p / (N_p + N_n).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


# =====================================================================
# Paper's fixed tree weights from Figure 4
# =====================================================================

# Category A: primary sequence propensity (1dprop)
# Sub-groups for each propensity type × window radius
# R = single residue, 2R = di-residue, RxR = coupled-residue
# aac3 = 3-residue segment, aac5 = 5-residue segment
# nR combines higher-order terms

_CAT_A_SUBGROUPS = {
    "R_1dprop": {
        "features": {"prop_aa_0": 0.27, "prop_di_0": 0.70, "prop_oligo_0": 0.03},
        "weight": 0.01,
    },
    "2R_1dprop": {
        "features": {"prop_aa_m1": 0.13, "prop_aa_p1": 0.13,
                      "prop_di_m1": 0.235, "prop_di_p1": 0.235,
                      "prop_oligo_m1": 0.135, "prop_oligo_p1": 0.135},
        "weight": 0.01,
    },
    "RxR_1dprop": {
        "features": {"prop_aa_m2": 0.05, "prop_aa_p2": 0.05,
                      "prop_di_m2": 0.445, "prop_di_p2": 0.445,
                      "prop_oligo_m2": 0.005, "prop_oligo_p2": 0.005},
        "weight": 0.75,
    },
    "nR_1dprop": {
        "features": {"prop_aa_m3": 0.005, "prop_aa_p3": 0.005,
                      "prop_di_m3": 0.32, "prop_di_p3": 0.32,
                      "prop_oligo_m3": 0.17, "prop_oligo_p3": 0.17},
        "weight": 0.97,
    },
}

# Intermediate node: 1dprop combines the 4 sub-groups
# Then further: RnxR_1dprop(RxR=0.75, nR=0.97) -> weight 0.01
# 1dprop(R=0.01, 2R=0.01, RnxR=0.01) -> simplify to flat weighted avg

# Category B: secondary structure propensity (2dprop)
_CAT_B_SUBGROUPS = {
    "R_2dprop": {
        "features": {"prop_dssp_0": 0.87, "prop_rama_0": 0.01, "prop_kappa_alpha_0": 0.12},
        "weight": 0.83,
    },
    "2R_2dprop": {
        "features": {"prop_dssp_m1": 0.41, "prop_dssp_p1": 0.41,
                      "prop_rama_m1": 0.03, "prop_rama_p1": 0.03,
                      "prop_kappa_alpha_m1": 0.06, "prop_kappa_alpha_p1": 0.06},
        "weight": 0.01,
    },
    "RxR_2dprop": {
        "features": {"prop_dssp_m2": 0.365, "prop_dssp_p2": 0.365,
                      "prop_rama_m2": 0.04, "prop_rama_p2": 0.04,
                      "prop_kappa_alpha_m2": 0.095, "prop_kappa_alpha_p2": 0.095},
        "weight": 0.94,
    },
    "RnxR_2dprop": {
        "features": {"prop_dssp_m3": 0.335, "prop_dssp_p3": 0.335,
                      "prop_rama_m3": 0.105, "prop_rama_p3": 0.105,
                      "prop_kappa_alpha_m3": 0.06, "prop_kappa_alpha_p3": 0.06},
        "weight": 0.16,
    },
}

# Category C: tertiary structure (3dprop)
_CAT_C_SUBGROUPS = {
    "Solacc": {
        "features": {"rsa": 1.00},
        "weight": 0.04,
    },
    "Eccent": {
        "features": {"cm": 0.70, "depth": 0.30},
        "weight": 0.14,
    },
    "Awaybury": {
        "features": {"farness_buried": 0.98, "dis_b": 0.02},
        "weight": 0.83,  # internal weight within Awaycore
    },
    "Awayhpho": {
        "features": {"farness_hydrophobic": 0.46, "dis_hpho": 0.54},
        "weight": 0.03,  # internal weight within Awaycore
    },
    "Nhbonds": {
        "features": {"hbond": 1.00},
        "weight": 0.15,
    },
    "Uncrowd": {
        "features": {"wcn": 0.22, "cn": 0.69, "closeness": 0.09},
        "weight": 0.19,
    },
    "Flex": {
        "features": {"rmsf": 0.50, "gnm_msf": 0.50},
        "weight": 0.01,
    },
}

# Top-level category weights
_CAT_WEIGHTS = {
    "cat_a": 0.14,  # 1dprop
    "cat_b": 0.57,  # 2dprop
    "cat_c": 0.29,  # 3dprop
}

# Awaycore combines Awaybury and Awayhpho, weight 0.61 within 3dprop
_AWAYCORE_WEIGHT = 0.61
# 3dprop combines: Solacc(0.04), Eccent(0.14), Awaycore(0.61), Nhbonds(0.15),
#                   Uncrowd(0.19), Flex(0.01)


class CPredHierarchical:
    """Hierarchical Inference model for CP site prediction.

    Uses fixed paper weights (Figure 4) — no training-time optimization.
    Probability calibration uses the training IF distribution.
    """

    def __init__(self, feature_names: list[str] | None = None,
                 tree: dict | None = None):
        self.feature_names = feature_names or []
        self._fitted = False
        self._train_if_pos = None  # IF values for positive training samples
        self._train_if_neg = None  # IF values for negative training samples

    def _feature_index(self, name: str) -> int | None:
        try:
            return self.feature_names.index(name)
        except ValueError:
            return None

    def _compute_subgroup_score(self, X: np.ndarray,
                                subgroup: dict) -> np.ndarray:
        """Compute weighted average score for a feature subgroup."""
        score = np.zeros(X.shape[0])
        total_w = 0.0
        for feat_name, w in subgroup["features"].items():
            idx = self._feature_index(feat_name)
            if idx is not None:
                score += X[:, idx] * w
                total_w += w
        if total_w > 1e-10:
            score /= total_w
        return score

    def _compute_category_a(self, X: np.ndarray) -> np.ndarray:
        """Compute Category A (1dprop) score."""
        scores = {}
        total_w = 0.0
        for name, sg in _CAT_A_SUBGROUPS.items():
            scores[name] = self._compute_subgroup_score(X, sg) * sg["weight"]
            total_w += sg["weight"]
        result = sum(scores.values())
        if total_w > 1e-10:
            result /= total_w
        return result

    def _compute_category_b(self, X: np.ndarray) -> np.ndarray:
        """Compute Category B (2dprop) score."""
        scores = {}
        total_w = 0.0
        for name, sg in _CAT_B_SUBGROUPS.items():
            scores[name] = self._compute_subgroup_score(X, sg) * sg["weight"]
            total_w += sg["weight"]
        result = sum(scores.values())
        if total_w > 1e-10:
            result /= total_w
        return result

    def _compute_category_c(self, X: np.ndarray) -> np.ndarray:
        """Compute Category C (3dprop) score with Awaycore sub-tree."""
        # Awaycore = weighted combination of Awaybury and Awayhpho
        awaybury_score = self._compute_subgroup_score(X, _CAT_C_SUBGROUPS["Awaybury"])
        awayhpho_score = self._compute_subgroup_score(X, _CAT_C_SUBGROUPS["Awayhpho"])
        ab_w = _CAT_C_SUBGROUPS["Awaybury"]["weight"]
        ah_w = _CAT_C_SUBGROUPS["Awayhpho"]["weight"]
        awaycore = (awaybury_score * ab_w + awayhpho_score * ah_w) / (ab_w + ah_w)

        # Other sub-groups
        solacc = self._compute_subgroup_score(X, _CAT_C_SUBGROUPS["Solacc"])
        eccent = self._compute_subgroup_score(X, _CAT_C_SUBGROUPS["Eccent"])
        nhbonds = self._compute_subgroup_score(X, _CAT_C_SUBGROUPS["Nhbonds"])
        uncrowd = self._compute_subgroup_score(X, _CAT_C_SUBGROUPS["Uncrowd"])
        flex = self._compute_subgroup_score(X, _CAT_C_SUBGROUPS["Flex"])

        # Combine with weights
        w_solacc = _CAT_C_SUBGROUPS["Solacc"]["weight"]
        w_eccent = _CAT_C_SUBGROUPS["Eccent"]["weight"]
        w_awaycore = _AWAYCORE_WEIGHT
        w_nhbonds = _CAT_C_SUBGROUPS["Nhbonds"]["weight"]
        w_uncrowd = _CAT_C_SUBGROUPS["Uncrowd"]["weight"]
        w_flex = _CAT_C_SUBGROUPS["Flex"]["weight"]

        total_w = w_solacc + w_eccent + w_awaycore + w_nhbonds + w_uncrowd + w_flex
        result = (solacc * w_solacc + eccent * w_eccent + awaycore * w_awaycore +
                  nhbonds * w_nhbonds + uncrowd * w_uncrowd + flex * w_flex)
        if total_w > 1e-10:
            result /= total_w
        return result

    def compute_if_score(self, X: np.ndarray) -> np.ndarray:
        """Compute the root Integrated Feature (IF) score."""
        cat_a = self._compute_category_a(X) * _CAT_WEIGHTS["cat_a"]
        cat_b = self._compute_category_b(X) * _CAT_WEIGHTS["cat_b"]
        cat_c = self._compute_category_c(X) * _CAT_WEIGHTS["cat_c"]
        total_w = sum(_CAT_WEIGHTS.values())
        return (cat_a + cat_b + cat_c) / total_w

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None,
            n_cd_rounds: int = 0) -> None:
        """Calibrate probability conversion using training IF distribution.

        No weight optimization — uses fixed paper weights.
        Only stores the training IF distribution for probability calibration.
        """
        if feature_names is not None:
            self.feature_names = feature_names

        if_scores = self.compute_if_score(X)
        self._train_if_pos = np.sort(if_scores[y == 1])
        self._train_if_neg = np.sort(if_scores[y == 0])
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict CP viability probabilities.

        Uses paper's probability conversion: P(i) = N_p / (N_p + N_n)
        where N_p = count of positive training samples with IF >= IF_i
        and N_n = count of negative training samples with IF <= IF_i.
        """
        if_scores = self.compute_if_score(X)

        if self._train_if_pos is None or self._train_if_neg is None:
            # Not fitted: use sigmoid fallback
            return 1 / (1 + np.exp(-if_scores))

        probs = np.zeros(len(if_scores))
        for i, if_val in enumerate(if_scores):
            n_p = np.sum(self._train_if_pos >= if_val)
            n_n = np.sum(self._train_if_neg <= if_val)
            denom = n_p + n_n
            probs[i] = n_p / denom if denom > 0 else 0.5

        return probs

    def save(self, path: str | Path) -> None:
        data = {
            "feature_names": self.feature_names,
            "train_if_pos": self._train_if_pos.tolist() if self._train_if_pos is not None else None,
            "train_if_neg": self._train_if_neg.tolist() if self._train_if_neg is not None else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self.feature_names = data.get("feature_names", [])
        if data.get("train_if_pos") is not None:
            self._train_if_pos = np.array(data["train_if_pos"])
            self._train_if_neg = np.array(data["train_if_neg"])
        self._fitted = True
