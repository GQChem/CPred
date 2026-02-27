"""Gaussian Network Model (GNM) fluctuation feature.

Computes mean-square fluctuation (MSF) from the GNM Kirchhoff matrix.
MSF_i = sum_k (u_ik² / lambda_k) for non-trivial modes (k >= 1).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

GNM_CUTOFF = 7.0  # Angstroms, CA atoms


def build_kirchhoff(ca_coords: np.ndarray,
                    cutoff: float = GNM_CUTOFF) -> np.ndarray:
    """Build Kirchhoff (connectivity) matrix for GNM.

    K_ij = -1 if dist(i,j) <= cutoff and i != j
    K_ii = -sum of off-diagonal elements in row i
    """
    n = len(ca_coords)
    dist_matrix = cdist(ca_coords, ca_coords)

    # Off-diagonal: -1 for contacts
    contact = (dist_matrix <= cutoff).astype(float)
    np.fill_diagonal(contact, 0.0)

    kirchhoff = -contact
    np.fill_diagonal(kirchhoff, contact.sum(axis=1))

    return kirchhoff


def compute_gnm_fluctuation(ca_coords: np.ndarray,
                            cutoff: float = GNM_CUTOFF) -> np.ndarray:
    """Compute GNM mean-square fluctuations.

    Args:
        ca_coords: (N, 3) CA atom coordinates.
        cutoff: Distance cutoff for contacts.

    Returns:
        (N,) array of mean-square fluctuations.
    """
    n = len(ca_coords)
    if n < 3:
        return np.zeros(n)

    kirchhoff = build_kirchhoff(ca_coords, cutoff)

    # Eigendecompose
    eigenvalues, eigenvectors = np.linalg.eigh(kirchhoff)

    # Skip the trivial zero mode (index 0)
    # Use modes with eigenvalue > small threshold
    msf = np.zeros(n)
    for k in range(1, n):
        if eigenvalues[k] < 1e-8:
            continue
        msf += eigenvectors[:, k] ** 2 / eigenvalues[k]

    return msf
