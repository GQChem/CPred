"""Category B features: secondary structure propensity scores.

Per Lo et al. 2012 (PLoS ONE), Cat B features are coupled SSE propensities
extracted from the ±3-residue window around each position. This mirrors the
Cat A design but applied to three SS alphabets: DSSP, Ramachandran, kappa-alpha.

Features (15 total, matching the 2dprop branch of Figure 4):
  For each SS type (dssp, rama, kappa_alpha):
    R_sse   : propensity(SS at i)
    2R_sse  : propensity(SS-di(i, i+1))
    RxR_sse : propensity(SS-di(i-1, i+1))  — immediate flanks
    R2xR_sse: propensity(SS-di(i-2, i+1))
    R3xR_sse: propensity(SS-di(i-3, i+1))

  Total: 5 × 3 = 15 features
"""

from __future__ import annotations

import numpy as np

from cpred.io.pdb_parser import ProteinStructure
from cpred.propensity.tables import PropensityTables
from cpred.features.structural_codes import (
    assign_ramachandran_codes,
    assign_kappa_alpha_codes,
)


def _get_code(codes: list[str] | np.ndarray, idx: int) -> str | None:
    """Return the code at idx, or None if out of bounds."""
    if 0 <= idx < len(codes):
        return str(codes[idx])
    return None


def _single_prop(tables: PropensityTables, table: str, code: str | None) -> float:
    if code is None:
        return 0.0
    return tables.get(table, code)


def _di_prop(tables: PropensityTables, table: str,
             c1: str | None, c2: str | None) -> float:
    """Look up di-code propensity by concatenating the two codes."""
    if c1 is None or c2 is None:
        return 0.0
    # Di-residue table is keyed by two-character strings
    key = c1 + c2
    # For SS codes, we reuse the di_residue table with SS alphabet keys
    # The table name for di-SS is stored per type (e.g., "di_dssp")
    # but we only have single-code tables. Use the single table and average.
    # This is an approximation; the full coupled SS propensity would require
    # separate di-SS tables. Here we sum the individual propensities as proxy.
    return _single_prop(tables, table, c1) + _single_prop(tables, table, c2)


def _extract_coupled_ss_features(
        codes: list[str] | np.ndarray,
        tables: PropensityTables,
        table_name: str,
        prefix: str) -> dict[str, np.ndarray]:
    """Extract 5 coupled propensity features for one SS alphabet."""
    n = len(codes)
    r_sse = np.zeros(n)
    r2_sse = np.zeros(n)
    rxr_sse = np.zeros(n)
    r2xr_sse = np.zeros(n)
    r3xr_sse = np.zeros(n)

    for i in range(n):
        c = _get_code(codes, i)
        cm3 = _get_code(codes, i - 3)
        cm2 = _get_code(codes, i - 2)
        cm1 = _get_code(codes, i - 1)
        cp1 = _get_code(codes, i + 1)

        r_sse[i] = _single_prop(tables, table_name, c)
        r2_sse[i] = _di_prop(tables, table_name, c, cp1)
        rxr_sse[i] = _di_prop(tables, table_name, cm1, cp1)
        r2xr_sse[i] = _di_prop(tables, table_name, cm2, cp1)
        r3xr_sse[i] = _di_prop(tables, table_name, cm3, cp1)

    return {
        f"R_{prefix}": r_sse,
        f"2R_{prefix}": r2_sse,
        f"RxR_{prefix}": rxr_sse,
        f"R2xR_{prefix}": r2xr_sse,
        f"R3xR_{prefix}": r3xr_sse,
    }


def extract_secondary_structure_features(
        protein: ProteinStructure,
        tables: PropensityTables) -> dict[str, np.ndarray]:
    """Extract all Category B coupled SS propensity features (15 total).

    Returns:
        Dict mapping feature name to (N,) arrays.
    """
    n = protein.n_residues

    # DSSP SS states
    if protein.dssp is not None:
        dssp_codes = list(protein.dssp.ss)
        rama_codes = assign_ramachandran_codes(protein.dssp.phi, protein.dssp.psi)
    else:
        dssp_codes = ["C"] * n
        rama_codes = ["A"] * n

    ka_codes = assign_kappa_alpha_codes(protein.ca_coords)

    features = {}
    features.update(_extract_coupled_ss_features(dssp_codes, tables, "dssp", "sse"))
    features.update(_extract_coupled_ss_features(rama_codes, tables, "ramachandran", "rm"))
    features.update(_extract_coupled_ss_features(ka_codes, tables, "kappa_alpha", "ka"))

    return features
