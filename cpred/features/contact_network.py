"""Contact network features: closeness centrality and farness measures.

Residue interaction graph: heavy atom distance < 4 Å between residues.
Features:
  - Closeness centrality (from NetworkX)
  - Farness to buried residues (Fb)
  - Farness to hydrophobic residues (Fhpho)
  - Farness sum (Fb + Fhpho)
  - Farness product (Fb * Fhpho)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

from cpred.io.pdb_parser import ProteinStructure

CONTACT_CUTOFF = 4.0  # Angstroms, heavy atoms


def build_residue_contact_graph(protein: ProteinStructure,
                                cutoff: float = CONTACT_CUTOFF) -> nx.Graph:
    """Build residue interaction graph based on heavy atom contacts.

    Two residues are connected if any pair of their heavy atoms is within
    the cutoff distance. Edges are weighted by minimum distance.
    """
    n = protein.n_residues
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        coords_i = protein.heavy_atom_coords[i]
        if len(coords_i) == 0:
            continue
        for j in range(i + 2, n):  # skip adjacent residues (i+1)
            coords_j = protein.heavy_atom_coords[j]
            if len(coords_j) == 0:
                continue
            dists = cdist(coords_i, coords_j)
            min_dist = dists.min()
            if min_dist < cutoff:
                G.add_edge(i, j, weight=min_dist)

    return G


def compute_closeness_centrality(G: nx.Graph, n: int) -> np.ndarray:
    """Closeness centrality for each residue in the contact graph."""
    centrality = nx.closeness_centrality(G)
    result = np.zeros(n)
    for node, val in centrality.items():
        result[node] = val
    return result


def _compute_farness(protein: ProteinStructure,
                     target_mask: np.ndarray) -> np.ndarray:
    """Compute farness to a set of target residues using 1/d² weighting.

    Farness(i) = sum over target residues j of 1/d(i,j)²
    """
    if not np.any(target_mask):
        return np.zeros(protein.n_residues)

    target_coords = protein.ca_coords[target_mask]
    dist_matrix = cdist(protein.ca_coords, target_coords)
    # Avoid division by zero for self
    dist_matrix = np.maximum(dist_matrix, 1e-6)
    farness = np.sum(1.0 / (dist_matrix ** 2), axis=1)
    return farness


def compute_farness_buried(protein: ProteinStructure) -> np.ndarray:
    """Farness to buried residues (RSA < 10%)."""
    if protein.dssp is None:
        return np.full(protein.n_residues, np.nan)
    buried_mask = protein.dssp.rsa < 0.10
    return _compute_farness(protein, buried_mask)


def compute_farness_hydrophobic(protein: ProteinStructure) -> np.ndarray:
    """Farness to hydrophobic residues (A, V, I, L, M, F, W, P)."""
    hpho_mask = np.array([protein.is_hydrophobic(i)
                          for i in range(protein.n_residues)])
    return _compute_farness(protein, hpho_mask)


def extract_contact_network_features(
        protein: ProteinStructure) -> dict[str, np.ndarray]:
    """Extract all contact network features.

    Returns:
        Dictionary mapping feature name to (N,) arrays.
    """
    G = build_residue_contact_graph(protein)
    closeness = compute_closeness_centrality(G, protein.n_residues)

    fb = compute_farness_buried(protein)
    fhpho = compute_farness_hydrophobic(protein)

    return {
        "closeness": closeness,
        "farness_buried": fb,
        "farness_hydrophobic": fhpho,
        "farness_sum": fb + fhpho,
        "farness_product": fb * fhpho,
    }
