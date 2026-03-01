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
        raise RuntimeError(f"DSSP not available for {protein.pdb_id}")
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
        raise RuntimeError(f"DSSP not available for {protein.pdb_id}")

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
    """Count of backbone hydrogen bonds per residue from DSSP."""
    if protein.dssp is None:
        raise RuntimeError(f"DSSP not available for {protein.pdb_id}")

    return protein.dssp.nhbond.copy()


def compute_avg_distance_to_buried(protein: ProteinStructure) -> np.ndarray:
    """Average distance from each CA to buried residues (RSA < 10%).

    DIS_b in the paper (Figure 3K). Uses harmonic mean of distances.
    """
    if protein.dssp is None:
        raise RuntimeError(f"DSSP not available for {protein.pdb_id}")
    buried_mask = protein.dssp.rsa < 0.10
    if not np.any(buried_mask):
        return np.zeros(protein.n_residues)
    target_coords = protein.ca_coords[buried_mask]
    dist_matrix = cdist(protein.ca_coords, target_coords)
    dist_matrix = np.maximum(dist_matrix, 1e-6)
    # Harmonic mean: n / sum(1/d)
    n_targets = dist_matrix.shape[1]
    inv_sum = np.sum(1.0 / dist_matrix, axis=1)
    return n_targets / inv_sum


def compute_avg_distance_to_hydrophobic(protein: ProteinStructure) -> np.ndarray:
    """Average distance from each CA to hydrophobic residues.

    DIS_hpho in the paper (Figure 3L). Uses harmonic mean of distances.
    """
    hpho_mask = np.array([protein.is_hydrophobic(i)
                          for i in range(protein.n_residues)])
    if not np.any(hpho_mask):
        return np.zeros(protein.n_residues)
    target_coords = protein.ca_coords[hpho_mask]
    dist_matrix = cdist(protein.ca_coords, target_coords)
    dist_matrix = np.maximum(dist_matrix, 1e-6)
    n_targets = dist_matrix.shape[1]
    inv_sum = np.sum(1.0 / dist_matrix, axis=1)
    return n_targets / inv_sum


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
        "dis_b": compute_avg_distance_to_buried(protein),
        "dis_hpho": compute_avg_distance_to_hydrophobic(protein),
    }
