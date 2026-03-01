"""Structural code assignment: Ramachandran codes and Kappa-Alpha codes.

Ramachandran codes: (phi, psi) -> one of 23 letters via frequency-ranked
grid regions derived from the SARST structural alphabet (Lo & Lyu, 2007).

Kappa-Alpha codes: (kappa, alpha) -> one of 23 letters via frequency-ranked
grid regions derived from the 3D-BLAST alphabet (Yang & Tung, 2006).

Kappa = CA bond angle (CA_{i-2}, CA_i, CA_{i+2})
Alpha = CA dihedral angle (CA_{i-1}, CA_i, CA_{i+1}, CA_{i+2})

The 23 codes are assigned by dividing the (phi,psi) or (kappa,alpha) space
into a grid, then ranking cells by population frequency in known protein
structures to produce a stable alphabet A-W.
"""

from __future__ import annotations

import numpy as np

# 23-letter alphabet for structural codes (A-W)
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVW"

# =====================================================================
# SARST-derived Ramachandran code regions
# =====================================================================
# The SARST Ramachandran map (Lo & Lyu, 2007; Lo et al., 2007 BMC Bioinf.)
# divides the Ramachandran plot into 23 regions based on backbone
# conformational preferences. The regions below approximate the SARST
# assignment using rectangular (phi, psi) bounds ordered by population
# frequency in known protein structures.
#
# Each entry: (phi_min, phi_max, psi_min, psi_max) -> code letter
# Regions are checked in order; first match wins.
# The ordering reflects the frequency ranking from SARST.

_RAMA_REGIONS = [
    # Code A: alpha-helix core (most populated)
    ("A", -80, -50, -50, -20),
    # Code B: alpha-helix extended
    ("B", -80, -50, -70, -50),
    ("B", -80, -50, -20, 0),
    # Code C: beta-strand core (extended, most populated beta region)
    ("C", -150, -100, 110, 170),
    # Code D: beta-strand secondary
    ("D", -150, -100, 80, 110),
    ("D", -100, -70, 110, 170),
    # Code E: polyproline II / left-handed helix region
    ("E", -80, -55, 120, 170),
    # Code F: alpha-L (left-handed alpha-helix)
    ("F", 40, 80, 20, 60),
    # Code G: 3_10 helix
    ("G", -80, -50, 0, 30),
    # Code H: pi-helix
    ("H", -80, -50, -100, -70),
    # Code I: turn region 1 (between alpha and beta)
    ("I", -120, -80, -50, 0),
    # Code J: wide beta
    ("J", -180, -150, 110, 180),
    ("J", -180, -150, -180, -150),
    # Code K: turn region 2
    ("K", -120, -80, 0, 50),
    # Code L: bridge region
    ("L", -120, -80, 80, 140),
    # Code M: alpha-helix C-cap
    ("M", -100, -80, -50, -20),
    # Code N: right-edge beta
    ("N", -100, -70, 80, 110),
    # Code O: coil region 1
    ("O", -180, -120, 50, 110),
    # Code P: coil region 2
    ("P", -70, -40, 120, 170),
    # Code Q: turn region 3
    ("Q", -120, -80, 50, 80),
    # Code R: gamma turn region
    ("R", -100, -60, 60, 100),
    # Code S: left-alpha extension
    ("S", 40, 100, -20, 20),
    # Code T: coil region 3
    ("T", -55, -30, -50, -20),
    # Code U: isolated region 1
    ("U", -180, -120, -60, 0),
    # Code V: isolated region 2
    ("V", -180, -120, 0, 50),
    # Code W: rare conformations
    ("W", 0, 180, -180, -60),
]

_RAMA_CODE_MAP = {}
for entry in _RAMA_REGIONS:
    code, phi_min, phi_max, psi_min, psi_max = entry
    _RAMA_CODE_MAP.setdefault(code, []).append((phi_min, phi_max, psi_min, psi_max))

# =====================================================================
# 3D-BLAST-derived Kappa-Alpha code regions
# =====================================================================
# The kappa-alpha codes describe 5-residue backbone conformations.
# Kappa = CA bond angle at position i (CA_{i-2}, CA_i, CA_{i+2}), range [0, 180]
# Alpha = CA dihedral angle (CA_{i-1}, CA_i, CA_{i+1}, CA_{i+2}), range [-180, 180]
#
# Based on the 3D-BLAST structural alphabet (Yang & Tung, 2006).
# Regions ordered by population frequency.

_KAPPA_ALPHA_REGIONS = [
    # Code A: extended/beta (kappa ~130, alpha ~0)
    ("A", 110, 150, -30, 30),
    # Code B: alpha-helix (kappa ~90, alpha ~50)
    ("B", 75, 105, 30, 70),
    # Code C: alpha-helix variant
    ("C", 75, 105, 70, 110),
    # Code D: extended variant
    ("D", 110, 150, -70, -30),
    # Code E: beta variant
    ("E", 110, 150, 30, 70),
    # Code F: turn region 1
    ("F", 60, 90, -30, 30),
    # Code G: turn region 2
    ("G", 90, 120, -30, 30),
    # Code H: coil/turn
    ("H", 90, 120, 30, 70),
    # Code I: helix extension
    ("I", 60, 90, 50, 90),
    # Code J: wide beta
    ("J", 130, 170, -100, -70),
    # Code K: extended coil
    ("K", 110, 150, 70, 120),
    # Code L: sharp turn
    ("L", 50, 80, -70, -30),
    # Code M: beta-hairpin
    ("M", 90, 120, -70, -30),
    # Code N: extended N-cap
    ("N", 90, 120, 70, 120),
    # Code O: sharp bend
    ("O", 40, 70, 90, 140),
    # Code P: loose coil
    ("P", 130, 170, 80, 140),
    # Code Q: compact turn
    ("Q", 60, 90, 90, 140),
    # Code R: wide turn
    ("R", 110, 150, 120, 180),
    # Code S: bend
    ("S", 40, 70, -100, -30),
    # Code T: left-turn
    ("T", 90, 120, 120, 180),
    # Code U: rare compact
    ("U", 40, 70, -180, -100),
    # Code V: rare extended
    ("V", 150, 180, -180, -100),
    # Code W: rare
    ("W", 0, 40, -180, 180),
]

_KAPPA_ALPHA_CODE_MAP = {}
for entry in _KAPPA_ALPHA_REGIONS:
    code, k_min, k_max, a_min, a_max = entry
    _KAPPA_ALPHA_CODE_MAP.setdefault(code, []).append((k_min, k_max, a_min, a_max))


def _assign_rama_code(phi: float, psi: float) -> str:
    """Map (phi, psi) to a Ramachandran code letter."""
    for code, regions in _RAMA_CODE_MAP.items():
        for phi_min, phi_max, psi_min, psi_max in regions:
            if phi_min <= phi < phi_max and psi_min <= psi < psi_max:
                return code
    # Fallback: find nearest region center
    return _nearest_rama_code(phi, psi)


def _nearest_rama_code(phi: float, psi: float) -> str:
    """Fallback: assign to nearest region center."""
    best_code = "A"
    best_dist = float("inf")
    for code, regions in _RAMA_CODE_MAP.items():
        for phi_min, phi_max, psi_min, psi_max in regions:
            cphi = (phi_min + phi_max) / 2
            cpsi = (psi_min + psi_max) / 2
            # Circular distance for angles
            dphi = min(abs(phi - cphi), 360 - abs(phi - cphi))
            dpsi = min(abs(psi - cpsi), 360 - abs(psi - cpsi))
            d = dphi ** 2 + dpsi ** 2
            if d < best_dist:
                best_dist = d
                best_code = code
    return best_code


def _assign_kappa_alpha_code(kappa: float, alpha: float) -> str:
    """Map (kappa, alpha) to a kappa-alpha code letter."""
    for code, regions in _KAPPA_ALPHA_CODE_MAP.items():
        for k_min, k_max, a_min, a_max in regions:
            if k_min <= kappa < k_max and a_min <= alpha < a_max:
                return code
    # Fallback: nearest
    best_code = "A"
    best_dist = float("inf")
    for code, regions in _KAPPA_ALPHA_CODE_MAP.items():
        for k_min, k_max, a_min, a_max in regions:
            ck = (k_min + k_max) / 2
            ca = (a_min + a_max) / 2
            dk = abs(kappa - ck)
            da = min(abs(alpha - ca), 360 - abs(alpha - ca))
            d = dk ** 2 + da ** 2
            if d < best_dist:
                best_dist = d
                best_code = code
    return best_code


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
            codes.append("A")  # default for undefined (terminal residues)
        else:
            codes.append(_assign_rama_code(p, s))
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
            codes.append(_assign_kappa_alpha_code(kappa, alpha))
    return codes
