"""Tests for ML models."""

import numpy as np
import pytest

from cpred.models.random_forest import CPredRandomForest
from cpred.models.svm import CPredSVM
from cpred.models.ann import CPredANN
from cpred.models.hierarchical import CPredHierarchical
from cpred.models.ensemble import CPredEnsemble
from cpred.pipeline import FEATURE_NAMES


@pytest.fixture
def synthetic_data():
    """Generate synthetic training data."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = len(FEATURE_NAMES)
    X = rng.normal(size=(n_samples, n_features))
    # Make labels correlate with first feature
    y = (X[:, 0] > 0).astype(float)
    return X, y


def test_random_forest(synthetic_data):
    """RF should fit and predict probabilities in [0, 1]."""
    X, y = synthetic_data
    rf = CPredRandomForest(n_estimators=10)
    rf.fit(X, y)
    probs = rf.predict(X)
    assert probs.shape == (len(X),)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_svm(synthetic_data):
    """SVM should fit and predict probabilities in [0, 1]."""
    X, y = synthetic_data
    svm = CPredSVM()
    svm.fit(X, y, grid_search=False)
    probs = svm.predict(X)
    assert probs.shape == (len(X),)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_ann(synthetic_data):
    """ANN should fit and predict probabilities in [0, 1]."""
    X, y = synthetic_data
    ann = CPredANN(n_features=X.shape[1], epochs=5)
    ann.fit(X, y)
    probs = ann.predict(X)
    assert probs.shape == (len(X),)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_hierarchical(synthetic_data):
    """HI should fit and predict probabilities in [0, 1]."""
    X, y = synthetic_data
    hi = CPredHierarchical(feature_names=FEATURE_NAMES)
    hi.fit(X, y, feature_names=FEATURE_NAMES)
    probs = hi.predict(X)
    assert probs.shape == (len(X),)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_ensemble(synthetic_data):
    """Ensemble should produce averaged predictions."""
    X, y = synthetic_data
    ens = CPredEnsemble(feature_names=FEATURE_NAMES)
    ens.rf = CPredRandomForest(n_estimators=10)
    ens.svm = CPredSVM()
    ens.ann = CPredANN(n_features=X.shape[1], epochs=5)
    ens.hi = CPredHierarchical(feature_names=FEATURE_NAMES)

    ens.rf.fit(X, y)
    ens.svm.fit(X, y, grid_search=False)
    ens.ann.fit(X, y)
    ens.hi.fit(X, y, feature_names=FEATURE_NAMES)
    ens._fitted = True

    probs = ens.predict(X)
    assert probs.shape == (len(X),)
    assert np.all(probs >= 0) and np.all(probs <= 1)

    # Should be average of individual models
    individual = ens.predict_individual(X)
    expected = sum(individual.values()) / 4
    np.testing.assert_allclose(probs, expected, atol=1e-6)
