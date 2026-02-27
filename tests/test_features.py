"""Tests for feature extraction modules."""

import numpy as np
import pytest

from cpred.features.gnm import build_kirchhoff, compute_gnm_fluctuation
from cpred.features.structural_codes import (
    assign_ramachandran_codes,
    assign_kappa_alpha_codes,
    compute_kappa_angle,
    compute_alpha_dihedral,
)
from cpred.features.window import window_average
from cpred.features.standardization import (
    invert_features,
    zscore_normalize,
    standardize_features,
)


# --- GNM tests ---

def test_kirchhoff_diagonal():
    """Kirchhoff diagonal should equal row contact count."""
    coords = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
    K = build_kirchhoff(coords, cutoff=7.0)
    # Residues 0-1 and 1-2 are in contact (dist=5), 0-2 not (dist=10)
    assert K[0, 0] == 1  # connected to residue 1 only
    assert K[1, 1] == 2  # connected to 0 and 2
    assert K[2, 2] == 1


def test_gnm_fluctuation_shape():
    """GNM fluctuation should return array of same length as input."""
    coords = np.random.default_rng(42).normal(size=(20, 3)) * 5
    msf = compute_gnm_fluctuation(coords)
    assert msf.shape == (20,)
    assert np.all(msf >= 0)


def test_gnm_small_protein():
    """GNM should handle very small proteins."""
    coords = np.array([[0, 0, 0], [3, 0, 0]], dtype=float)
    msf = compute_gnm_fluctuation(coords)
    assert msf.shape == (2,)


# --- Structural codes tests ---

def test_ramachandran_codes_length():
    """Ramachandran codes should match input length."""
    phi = np.array([-60, -120, -60, 60])
    psi = np.array([-40, 130, -40, 40])
    codes = assign_ramachandran_codes(phi, psi)
    assert len(codes) == 4
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVW" for c in codes)


def test_kappa_alpha_codes_terminals():
    """Terminal residues should get default code."""
    coords = np.array([[i * 3.8, 0, 0] for i in range(10)], dtype=float)
    codes = assign_kappa_alpha_codes(coords)
    assert len(codes) == 10
    # First 2 and last 2 should be default 'A'
    assert codes[0] == "A"
    assert codes[1] == "A"


def test_kappa_angle_geometry():
    """Kappa angle for linear chain should be ~180."""
    coords = np.array([[i * 3.8, 0, 0] for i in range(5)], dtype=float)
    kappa = compute_kappa_angle(coords, 2)
    assert abs(kappa - 180.0) < 1.0


# --- Window averaging tests ---

def test_window_average_uniform():
    """Window average of uniform values should return same values."""
    vals = np.ones(10)
    result = window_average(vals, w=3)
    np.testing.assert_allclose(result, 1.0)


def test_window_average_edges():
    """Window average at edges should use fewer neighbors."""
    vals = np.array([10, 0, 0, 0, 0])
    result = window_average(vals, w=3)
    # Position 0: average of [10, 0, 0, 0] = 2.5
    assert result[0] == pytest.approx(2.5)
    # Position 4: average of [0, 0, 0, 0] = 0
    assert result[4] == pytest.approx(0.0)


def test_window_average_nan_handling():
    """Window average should handle NaN values."""
    vals = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    result = window_average(vals, w=1)
    # Position 0: mean of [1.0, nan] -> mean of [1.0] = 1.0
    assert result[0] == pytest.approx(1.0)


# --- Standardization tests ---

def test_invert_features():
    """Inverted features should be negated."""
    features = {
        "cn": np.array([1.0, 2.0, 3.0]),
        "rsa": np.array([0.1, 0.2, 0.3]),
    }
    result = invert_features(features)
    np.testing.assert_allclose(result["cn"], [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(result["rsa"], [0.1, 0.2, 0.3])  # not inverted


def test_zscore_normalize():
    """Z-score should have mean=0, std=1."""
    features = {
        "test": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    result = zscore_normalize(features)
    np.testing.assert_allclose(result["test"].mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(result["test"].std(), 1.0, atol=1e-10)


def test_standardize_full():
    """Full standardization pipeline should work end-to-end."""
    features = {
        "cn": np.array([5.0, 10.0, 15.0]),
        "rsa": np.array([0.1, 0.5, 0.9]),
    }
    result = standardize_features(features)
    assert "cn" in result
    assert "rsa" in result
    assert result["cn"].shape == (3,)
