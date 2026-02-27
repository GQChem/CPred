"""Category B features: secondary structure propensity scores.

For each residue position, look up propensity scores for:
  - DSSP secondary structure state
  - Ramachandran structural code
  - Kappa-alpha structural code

These are looked up from pre-computed propensity tables.
"""

from __future__ import annotations

import numpy as np

from cpred.io.pdb_parser import ProteinStructure
from cpred.propensity.tables import PropensityTables
from cpred.features.structural_codes import (
    assign_ramachandran_codes,
    assign_kappa_alpha_codes,
)


def compute_dssp_propensity(protein: ProteinStructure,
                             tables: PropensityTables) -> np.ndarray:
    """Look up DSSP secondary structure propensity at each position."""
    if protein.dssp is None:
        return np.zeros(protein.n_residues)
    return np.array([tables.get("dssp", ss) for ss in protein.dssp.ss])


def compute_ramachandran_propensity(protein: ProteinStructure,
                                     tables: PropensityTables) -> np.ndarray:
    """Look up Ramachandran code propensity at each position."""
    if protein.dssp is None:
        return np.zeros(protein.n_residues)
    rama_codes = assign_ramachandran_codes(protein.dssp.phi, protein.dssp.psi)
    return np.array([tables.get("ramachandran", code) for code in rama_codes])


def compute_kappa_alpha_propensity(protein: ProteinStructure,
                                    tables: PropensityTables) -> np.ndarray:
    """Look up kappa-alpha code propensity at each position."""
    ka_codes = assign_kappa_alpha_codes(protein.ca_coords)
    return np.array([tables.get("kappa_alpha", code) for code in ka_codes])


def extract_secondary_structure_features(
        protein: ProteinStructure,
        tables: PropensityTables) -> dict[str, np.ndarray]:
    """Extract all Category B secondary structure propensity features.

    Returns:
        Dict mapping feature name to (N,) arrays.
    """
    return {
        "prop_dssp": compute_dssp_propensity(protein, tables),
        "prop_rama": compute_ramachandran_propensity(protein, tables),
        "prop_kappa_alpha": compute_kappa_alpha_propensity(protein, tables),
    }
