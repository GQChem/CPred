#!/usr/bin/env python3
"""Compute propensity tables from training datasets.

Uses nrCPsite_cpdb-40 (experimental CP sites) vs whole protein sequences
(comparison) to compute propensity scores for:
  - Single amino acid
  - Di-residue (pairs)
  - Oligo-residue (triplets)
  - DSSP secondary structure states
  - Ramachandran structural codes
  - Kappa-alpha structural codes
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from cpred.propensity.scoring import build_propensity_table
from cpred.propensity.tables import PropensityTables


def extract_window_elements(sequence: str, sites: list[int],
                            w: int = 3) -> list[str]:
    """Extract single AA elements from ±w windows around CP sites."""
    elements = []
    n = len(sequence)
    for site in sites:
        for offset in range(-w, w + 1):
            pos = site + offset
            if 0 <= pos < n:
                elements.append(sequence[pos])
    return elements


def extract_di_residues(sequence: str, sites: list[int],
                        w: int = 3) -> list[str]:
    """Extract di-residue pairs from ±w windows."""
    elements = []
    n = len(sequence)
    for site in sites:
        for offset in range(-w, w):
            pos = site + offset
            if 0 <= pos < n - 1:
                elements.append(sequence[pos:pos + 2])
    return elements


def extract_oligo_residues(sequence: str, sites: list[int],
                           w: int = 3) -> list[str]:
    """Extract oligo-residue triplets from ±w windows."""
    elements = []
    n = len(sequence)
    for site in sites:
        for offset in range(-w, w - 1):
            pos = site + offset
            if 0 <= pos < n - 2:
                elements.append(sequence[pos:pos + 3])
    return elements


def build_all_tables(exp_sequences: list[str], exp_sites: list[list[int]],
                     comp_sequences: list[str],
                     exp_dssp: list[str] | None = None,
                     comp_dssp: list[str] | None = None,
                     exp_rama: list[str] | None = None,
                     comp_rama: list[str] | None = None,
                     exp_kappa_alpha: list[str] | None = None,
                     comp_kappa_alpha: list[str] | None = None,
                     n_permutations: int = 1000) -> dict[str, dict]:
    """Build all propensity tables from training data.

    Args:
        exp_sequences: Sequences with known CP sites.
        exp_sites: Per-sequence list of CP site positions.
        comp_sequences: Comparison (whole protein) sequences.
        exp_dssp: DSSP codes for experimental windows.
        comp_dssp: DSSP codes for comparison set.
        exp_rama: Ramachandran codes for experimental windows.
        comp_rama: Ramachandran codes for comparison set.
        exp_kappa_alpha: Kappa-alpha codes for experimental windows.
        comp_kappa_alpha: Kappa-alpha codes for comparison set.
        n_permutations: Number of permutations for p-value.

    Returns:
        Dictionary of table_name -> propensity table dict.
    """
    rng = np.random.default_rng(42)
    tables = {}

    # Single AA
    exp_aa = []
    comp_aa = list("".join(comp_sequences))
    for seq, sites in zip(exp_sequences, exp_sites):
        exp_aa.extend(extract_window_elements(seq, sites))
    tables["single_aa"] = build_propensity_table(exp_aa, comp_aa,
                                                  n_permutations, rng)

    # Di-residue
    exp_di = []
    comp_di = []
    for seq, sites in zip(exp_sequences, exp_sites):
        exp_di.extend(extract_di_residues(seq, sites))
    for seq in comp_sequences:
        comp_di.extend([seq[i:i + 2] for i in range(len(seq) - 1)])
    tables["di_residue"] = build_propensity_table(exp_di, comp_di,
                                                   n_permutations, rng)

    # Oligo-residue
    exp_oligo = []
    comp_oligo = []
    for seq, sites in zip(exp_sequences, exp_sites):
        exp_oligo.extend(extract_oligo_residues(seq, sites))
    for seq in comp_sequences:
        comp_oligo.extend([seq[i:i + 3] for i in range(len(seq) - 2)])
    tables["oligo_residue"] = build_propensity_table(exp_oligo, comp_oligo,
                                                      n_permutations, rng)

    # DSSP
    if exp_dssp is not None and comp_dssp is not None:
        tables["dssp"] = build_propensity_table(exp_dssp, comp_dssp,
                                                 n_permutations, rng)

    # Ramachandran codes
    if exp_rama is not None and comp_rama is not None:
        tables["ramachandran"] = build_propensity_table(exp_rama, comp_rama,
                                                         n_permutations, rng)

    # Kappa-alpha codes
    if exp_kappa_alpha is not None and comp_kappa_alpha is not None:
        tables["kappa_alpha"] = build_propensity_table(
            exp_kappa_alpha, comp_kappa_alpha, n_permutations, rng)

    return tables


def main():
    parser = argparse.ArgumentParser(
        description="Build propensity tables from training data")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory with downloaded datasets")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for tables (default: cpred/data/propensity_tables)")
    parser.add_argument("--permutations", type=int, default=1000,
                        help="Number of permutations for p-value")
    args = parser.parse_args()

    output_dir = args.output_dir or (
        Path(__file__).parent.parent / "cpred" / "data" / "propensity_tables")

    # Try to load training data
    try:
        from cpred.training.data_loader import load_training_sequences
        exp_seqs, exp_sites, comp_seqs = load_training_sequences(args.data_dir)
    except Exception as e:
        print(f"Could not load training data: {e}")
        print("Using placeholder tables. Run download_data.py first.")

        # Create minimal placeholder tables with standard AA propensities
        pt = PropensityTables(output_dir)
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
        for table_name in ["single_aa", "dssp", "ramachandran", "kappa_alpha"]:
            pt.save(table_name, {aa: 0.0 for aa in standard_aa})
        pt.save("di_residue", {})
        pt.save("oligo_residue", {})
        print(f"Placeholder tables saved to {output_dir}")
        return

    print(f"Building propensity tables with {args.permutations} permutations...")
    tables = build_all_tables(exp_seqs, exp_sites, comp_seqs,
                              n_permutations=args.permutations)

    pt = PropensityTables(output_dir)
    for name, table in tables.items():
        pt.save(name, table)
        print(f"  Saved {name}: {len(table)} entries")

    print("Done!")


if __name__ == "__main__":
    main()
