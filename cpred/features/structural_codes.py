"""Structural code assignment: Ramachandran codes and Kappa-Alpha codes.

Ramachandran codes: (phi, psi) -> one of 23 letters via 36x36 grid (10° bins).
Kappa-Alpha codes: (kappa, alpha) -> one of 23 letters via 36x18 grid.

Kappa = CA bond angle (CA_{i-2}, CA_i, CA_{i+2})
Alpha = CA dihedral angle (CA_{i-1}, CA_i, CA_{i+1}, CA_{i+2})

The 23-letter alphabets and grid mappings are derived from training data.
Here we use a simplified assignment based on structural geometry regions.
"""

from __future__ import annotations

import numpy as np

# 23-letter alphabet for structural codes (A-W)
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVW"

# Ramachandran region boundaries (simplified 23-state classification)
# Based on common Ramachandran plot divisions
RAMA_BINS_PHI = np.linspace(-180, 180, 37)  # 36 bins of 10°
RAMA_BINS_PSI = np.linspace(-180, 180, 37)

# Kappa-Alpha bins
KAPPA_BINS = np.linspace(0, 180, 37)    # 36 bins of 5°
ALPHA_BINS = np.linspace(-180, 180, 19)  # 18 bins of 20°


def _angle_to_bin(angle: float, bins: np.ndarray) -> int:
    """Map angle to bin index."""
    idx = np.searchsorted(bins, angle, side="right") - 1
    return max(0, min(idx, len(bins) - 2))


def _phi_psi_to_code_index(phi: float, psi: float) -> int:
    """Map (phi, psi) to a Ramachandran code index (0-22).

    Uses a simplified mapping based on structural regions.
    """
    phi_bin = _angle_to_bin(phi, RAMA_BINS_PHI)
    psi_bin = _angle_to_bin(psi, RAMA_BINS_PSI)
    # Combine bins and map to 0-22
    combined = phi_bin * 36 + psi_bin
    return combined % 23


def _kappa_alpha_to_code_index(kappa: float, alpha: float) -> int:
    """Map (kappa, alpha) to a kappa-alpha code index (0-22)."""
    k_bin = _angle_to_bin(kappa, KAPPA_BINS)
    a_bin = _angle_to_bin(alpha, ALPHA_BINS)
    combined = k_bin * 18 + a_bin
    return combined % 23


def compute_kappa_angle(ca_coords: np.ndarray, idx: int) -> float:
    """Compute kappa angle: CA bond angle at position i.

    Kappa = angle(CA_{i-2}, CA_i, CA_{i+2})
    """
    n = len(ca_coords)
    if idx < 2 or idx >= n - 2:
        return np.nan

    v1 = ca_coords[idx - 2] - ca_coords[idx]
    v2 = ca_coords[idx + 2] - ca_coords[idx]

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def compute_alpha_dihedral(ca_coords: np.ndarray, idx: int) -> float:
    """Compute alpha dihedral: CA dihedral at position i.

    Alpha = dihedral(CA_{i-1}, CA_i, CA_{i+1}, CA_{i+2})
    """
    n = len(ca_coords)
    if idx < 1 or idx >= n - 2:
        return np.nan

    p0 = ca_coords[idx - 1]
    p1 = ca_coords[idx]
    p2 = ca_coords[idx + 1]
    p3 = ca_coords[idx + 2]

    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return np.nan

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    cos_angle = np.clip(np.dot(n1, n2), -1, 1)
    sign = np.sign(np.dot(np.cross(n1, n2), b2 / (np.linalg.norm(b2) + 1e-10)))
    if sign == 0:
        sign = 1
    return np.degrees(np.arccos(cos_angle)) * sign


def assign_ramachandran_codes(phi: np.ndarray, psi: np.ndarray) -> list[str]:
    """Assign Ramachandran structural codes to each residue.

    Args:
        phi: (N,) phi angles in degrees.
        psi: (N,) psi angles in degrees.

    Returns:
        List of single-character codes (A-W).
    """
    codes = []
    for p, s in zip(phi, psi):
        if np.isnan(p) or np.isnan(s):
            codes.append("A")  # default for undefined
        else:
            idx = _phi_psi_to_code_index(p, s)
            codes.append(ALPHABET[idx])
    return codes


def assign_kappa_alpha_codes(ca_coords: np.ndarray) -> list[str]:
    """Assign kappa-alpha structural codes to each residue.

    Args:
        ca_coords: (N, 3) CA coordinates.

    Returns:
        List of single-character codes (A-W).
    """
    n = len(ca_coords)
    codes = []
    for i in range(n):
        kappa = compute_kappa_angle(ca_coords, i)
        alpha = compute_alpha_dihedral(ca_coords, i)
        if np.isnan(kappa) or np.isnan(alpha):
            codes.append("A")  # default for terminal residues
        else:
            idx = _kappa_alpha_to_code_index(kappa, alpha)
            codes.append(ALPHABET[idx])
    return codes
