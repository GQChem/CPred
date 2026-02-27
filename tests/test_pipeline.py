"""Tests for the feature extraction pipeline."""

import numpy as np
import pytest

from cpred.propensity.scoring import (
    compute_frequencies,
    compute_propensity,
    build_propensity_table,
)
from cpred.propensity.tables import PropensityTables
from cpred.features.sequence_propensity import (
    compute_single_aa_propensity,
    compute_di_residue_propensity,
)
from cpred.pipeline import FEATURE_NAMES, build_feature_matrix
from cpred.training.evaluate import compute_metrics


# --- Propensity scoring tests ---

def test_compute_frequencies():
    """Frequencies should sum to 1."""
    elements = list("AAABBC")
    freq = compute_frequencies(elements)
    assert abs(sum(freq.values()) - 1.0) < 1e-10
    assert freq["A"] == pytest.approx(0.5)


def test_compute_propensity_zero_fc():
    """Propensity should be 0 when fc is 0."""
    assert compute_propensity(0.1, 0.0, 0.5) == 0.0


def test_compute_propensity_formula():
    """Check propensity formula: ((fe-fc)/fc) * (1-pval)."""
    fe, fc, pval = 0.3, 0.1, 0.2
    expected = ((0.3 - 0.1) / 0.1) * (1 - 0.2)
    assert compute_propensity(fe, fc, pval) == pytest.approx(expected)


def test_build_propensity_table():
    """Propensity table should contain entries for observed elements."""
    exp = list("AAABBB")
    comp = list("AABBCC")
    table = build_propensity_table(exp, comp, n_permutations=50)
    assert "A" in table
    assert "B" in table
    assert "C" in table


# --- Sequence propensity tests ---

def test_single_aa_propensity():
    """Single AA propensity should return correct length."""
    tables = PropensityTables()
    tables._tables = {"single_aa": {"A": 1.0, "G": -0.5}}
    result = compute_single_aa_propensity("AGG", tables)
    assert len(result) == 3
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(-0.5)


# --- Feature matrix tests ---

def test_build_feature_matrix_shape():
    """Feature matrix should have correct dimensions."""
    n = 50
    features = {name: np.random.default_rng(42).normal(size=n)
                for name in FEATURE_NAMES}
    X = build_feature_matrix(features)
    assert X.shape == (n, len(FEATURE_NAMES))


def test_build_feature_matrix_missing():
    """Should handle missing features with zeros."""
    n = 10
    features = {"rsa": np.ones(n)}
    X = build_feature_matrix(features)
    assert X.shape == (n, len(FEATURE_NAMES))


# --- Evaluation tests ---

def test_compute_metrics_perfect():
    """Perfect predictions should yield AUC=1, sensitivity=1, specificity=1."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)


def test_compute_metrics_random():
    """Random predictions should yield AUC ≈ 0.5."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=1000)
    y_prob = rng.uniform(size=1000)
    metrics = compute_metrics(y_true, y_prob)
    assert 0.4 < metrics["auc"] < 0.6
