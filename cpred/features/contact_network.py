"""Contact network features: closeness centrality, CN, WCN, and farness.

Residue interaction graph: heavy atom distance < 4 Å between residues.
Features:
  - Closeness centrality (from NetworkX)
  - Contact number (CN): CB atoms within 6.4 Å
  - Weighted contact number (WCN): sum of 1/d² over CA pairs
  - Farness to buried residues (Fb) — graph shortest-path based (eq. 3)
  - Farness to hydrophobic residues (Fhpho)
  - Farness union Fb∪Fhpho
  - Farness intersection Fb∩Fhpho
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
    the cutoff distance. Edges are unweighted (unit weight) for shortest-path
    computation per the paper's closeness and farness definitions.
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
                G.add_edge(i, j)

    return G


def compute_closeness_centrality(G: nx.Graph, n: int) -> np.ndarray:
    """Closeness centrality for each residue in the contact graph."""
    centrality = nx.closeness_centrality(G)
    result = np.zeros(n)
    for node, val in centrality.items():
        result[node] = val
    return result


def _compute_farness_graph(G: nx.Graph, n: int,
                           target_mask: np.ndarray) -> np.ndarray:
    """Compute farness from each residue to a target set using graph distances.

    Per Lo et al. 2012, equation 3:
      F_G(i) = 1 / sum_{j in G} (W(j) * d_ij^{-1})
    where d_ij is the shortest-path distance in the contact graph,
    and W(j) = d_ij^{-2} for Fb (farness from buried core).

    Simplifying: F_G(i) = 1 / sum_{j in G} d_ij^{-3}

    Higher values = farther from the target set (more viable for CP).
    """
    if not np.any(target_mask):
        return np.zeros(n)

    target_indices = np.where(target_mask)[0]
    farness = np.zeros(n)

    # Compute shortest paths from all nodes
    # Use dict_of_dicts format for efficiency
    all_lengths = dict(nx.all_pairs_shortest_path_length(G))

    for i in range(n):
        if i not in all_lengths:
            farness[i] = np.inf
            continue
        lengths_i = all_lengths[i]
        total = 0.0
        for j in target_indices:
            d = lengths_i.get(j, None)
            if d is not None and d > 0:
                # W(j) = d^{-2}, contribution = W(j) * d^{-1} = d^{-3}
                total += d ** (-3)
        if total > 1e-10:
            farness[i] = 1.0 / total
        else:
            farness[i] = np.inf

    # Replace inf with max finite value (disconnected nodes)
    finite_mask = np.isfinite(farness)
    if np.any(finite_mask):
        max_val = farness[finite_mask].max()
        farness[~finite_mask] = max_val * 2.0
    else:
        farness[:] = 0.0

    return farness


def compute_farness_buried(G: nx.Graph, protein: ProteinStructure) -> np.ndarray:
    """Farness to buried residues (RSA < 10%)."""
    if protein.dssp is None:
        return np.full(protein.n_residues, np.nan)
    buried_mask = protein.dssp.rsa < 0.10
    return _compute_farness_graph(G, protein.n_residues, buried_mask)


def compute_farness_hydrophobic(G: nx.Graph,
                                protein: ProteinStructure) -> np.ndarray:
    """Farness to hydrophobic residues (A, V, I, L, M, F, W, P)."""
    hpho_mask = np.array([protein.is_hydrophobic(i)
                          for i in range(protein.n_residues)])
    return _compute_farness_graph(G, protein.n_residues, hpho_mask)


def extract_contact_network_features(
        protein: ProteinStructure) -> dict[str, np.ndarray]:
    """Extract all contact network features.

    Returns:
        Dictionary mapping feature name to (N,) arrays.
    """
    G = build_residue_contact_graph(protein)
    n = protein.n_residues
    closeness = compute_closeness_centrality(G, n)

    fb = compute_farness_buried(G, protein)
    fhpho = compute_farness_hydrophobic(G, protein)

    # Buried and hydrophobic masks for union/intersection farness
    buried_mask = (protein.dssp.rsa < 0.10) if protein.dssp is not None else np.zeros(n, dtype=bool)
    hpho_mask = np.array([protein.is_hydrophobic(i) for i in range(n)])
    union_mask = buried_mask | hpho_mask
    inter_mask = buried_mask & hpho_mask

    f_union = _compute_farness_graph(G, n, union_mask)
    f_inter = _compute_farness_graph(G, n, inter_mask)

    return {
        "closeness": closeness,
        "farness_buried": fb,
        "farness_hydrophobic": fhpho,
        "farness_union": f_union,
        "farness_inter": f_inter,
    }
