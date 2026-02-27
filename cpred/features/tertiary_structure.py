"""Category C features: tertiary structure properties.

Per-residue features:
  - RSA (relative solvent accessibility, from DSSP)
  - CN (contact number: CB atoms within 6.4 Å)
  - WCN (weighted contact number: sum of 1/d² over all CA pairs)
  - CM (distance to centroid of all CA atoms)
  - Depth (distance to nearest surface atom, RSA > 20%)
  - B-factor (from PDB)
  - H-bond count (from DSSP)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from cpred.io.pdb_parser import ProteinStructure

CN_CUTOFF = 6.4  # Angstroms, CB atoms
SURFACE_RSA_THRESHOLD = 0.20


def compute_rsa(protein: ProteinStructure) -> np.ndarray:
    """Relative solvent accessibility from DSSP."""
    if protein.dssp is None:
        return np.full(protein.n_residues, np.nan)
    return protein.dssp.rsa.copy()


def compute_contact_number(protein: ProteinStructure,
                           cutoff: float = CN_CUTOFF) -> np.ndarray:
    """Contact number: count of CB atoms within cutoff distance."""
    dist_matrix = cdist(protein.cb_coords, protein.cb_coords)
    # Exclude self-contact
    np.fill_diagonal(dist_matrix, np.inf)
    cn = np.sum(dist_matrix <= cutoff, axis=1).astype(float)
    return cn


def compute_weighted_contact_number(protein: ProteinStructure) -> np.ndarray:
    """Weighted contact number: sum of 1/d² over all CA pairs."""
    dist_matrix = cdist(protein.ca_coords, protein.ca_coords)
    np.fill_diagonal(dist_matrix, np.inf)
    wcn = np.sum(1.0 / (dist_matrix ** 2), axis=1)
    return wcn


def compute_distance_to_centroid(protein: ProteinStructure) -> np.ndarray:
    """Distance from each CA to the centroid of all CAs."""
    centroid = protein.ca_coords.mean(axis=0)
    return np.linalg.norm(protein.ca_coords - centroid, axis=1)


def compute_depth(protein: ProteinStructure) -> np.ndarray:
    """Residue depth: distance to nearest surface residue (RSA > 20%).

    Approximation using CA distances to surface residues defined by RSA.
    """
    if protein.dssp is None:
        return np.full(protein.n_residues, np.nan)

    surface_mask = protein.dssp.rsa >= SURFACE_RSA_THRESHOLD
    if not np.any(surface_mask):
        return np.zeros(protein.n_residues)

    surface_coords = protein.ca_coords[surface_mask]
    dist_to_surface = cdist(protein.ca_coords, surface_coords)
    return dist_to_surface.min(axis=1)


def compute_bfactor(protein: ProteinStructure) -> np.ndarray:
    """B-factor (temperature factor) of CA atoms."""
    return protein.bfactors.copy()


def compute_hbond_count(protein: ProteinStructure) -> np.ndarray:
    """Count of backbone hydrogen bonds per residue from DSSP.

    Uses DSSP secondary structure as proxy: H/G/I residues participate in
    backbone H-bonds (helices = 2 H-bonds, sheets = 1-2).
    """
    if protein.dssp is None:
        return np.full(protein.n_residues, np.nan)

    hbond_count = np.zeros(protein.n_residues)
    for i, ss in enumerate(protein.dssp.ss):
        if ss in ("H", "G", "I"):
            hbond_count[i] = 2.0  # helical residues have ~2 backbone H-bonds
        elif ss in ("E", "B"):
            hbond_count[i] = 1.0  # sheet/bridge residues have ~1-2
    return hbond_count


def extract_tertiary_features(protein: ProteinStructure) -> dict[str, np.ndarray]:
    """Extract all Category C tertiary structure features.

    Returns:
        Dictionary mapping feature name to (N,) arrays.
    """
    return {
        "rsa": compute_rsa(protein),
        "cn": compute_contact_number(protein),
        "wcn": compute_weighted_contact_number(protein),
        "cm": compute_distance_to_centroid(protein),
        "depth": compute_depth(protein),
        "bfactor": compute_bfactor(protein),
        "hbond": compute_hbond_count(protein),
    }
