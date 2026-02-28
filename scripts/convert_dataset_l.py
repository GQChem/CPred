#!/usr/bin/env python3
"""Convert Dataset_L.xls into CSV files for Dataset T and DHFR.

Dataset T: 176 labeled CP sites from 6 proteins (training set)
DHFR: 159 labeled CP sites from E. coli DHFR (independent test set)

Usage:
    python scripts/convert_dataset_l.py
"""

import re
import sys
from pathlib import Path

import pandas as pd


def parse_pdb_entry(raw: str) -> tuple[str, str]:
    """Extract pdb_id and chain from entries like '1gflA (green fluorescent protein, GFP)'.

    Returns:
        (pdb_id, chain) e.g. ('1gfl', 'A')
    """
    raw = str(raw).strip()
    token = re.split(r'[\s(]', raw)[0]
    if len(token) >= 5:
        pdb_id = token[:4].lower()
        chain = token[4]
    else:
        pdb_id = token.lower()
        chain = "A"
    return pdb_id, chain


def parse_viable(val) -> int:
    """Parse viable column: '+' -> 1, everything else -> 0."""
    s = str(val).strip()
    if s == '+':
        return 1
    return 0


def extract_dataset(df, pdb_col, res_col, aa_col, viable_col, start_row=4):
    """Extract a dataset from the spreadsheet.

    PDB entry column is forward-filled (only first row of each protein has it).
    """
    rows = []
    current_pdb_id = None
    current_chain = None

    for i in range(start_row, len(df)):
        # Check if this row has a residue number
        res_val = df.iloc[i, res_col]
        try:
            resno = int(res_val)
        except (ValueError, TypeError):
            continue

        # Check for new PDB entry
        pdb_val = df.iloc[i, pdb_col]
        if pd.notna(pdb_val) and str(pdb_val).strip() not in ('', 'nan'):
            current_pdb_id, current_chain = parse_pdb_entry(str(pdb_val))

        if current_pdb_id is None:
            continue

        aa = str(df.iloc[i, aa_col]).strip()
        if not aa or len(aa) != 1:
            continue

        viable = parse_viable(df.iloc[i, viable_col])
        rows.append({
            'pdb_id': current_pdb_id,
            'chain': current_chain,
            'residue_number': resno,
            'amino_acid': aa,
            'viable': viable,
        })

    return pd.DataFrame(rows)


def main():
    supp_dir = Path("data/supplementary")
    xls_path = supp_dir / "Dataset_L.xls"

    if not xls_path.exists():
        print(f"ERROR: {xls_path} not found")
        sys.exit(1)

    df = pd.read_excel(xls_path, header=None)
    print(f"Read {xls_path}: {df.shape[0]} rows x {df.shape[1]} columns")

    # Dataset T: col 6=PDB entry, col 7=residue no, col 8=AA, col 12=viable
    df_t = extract_dataset(df, pdb_col=6, res_col=7, aa_col=8, viable_col=12)
    t_path = supp_dir / "dataset_t.csv"
    df_t.to_csv(t_path, index=False)
    n_viable_t = int(df_t['viable'].sum())
    print(f"\nDataset T: {len(df_t)} rows -> {t_path}")
    print(f"  Viable: {n_viable_t}, Inviable: {len(df_t) - n_viable_t}")
    print(f"  Proteins: {df_t['pdb_id'].nunique()} unique PDB IDs")
    print(f"  PDB IDs: {sorted(df_t['pdb_id'].unique())}")

    # DHFR: col 14=PDB entry, col 15=residue no, col 16=AA, col 21=viable
    df_d = extract_dataset(df, pdb_col=14, res_col=15, aa_col=16, viable_col=21)
    d_path = supp_dir / "dataset_dhfr.csv"
    df_d.to_csv(d_path, index=False)
    n_viable_d = int(df_d['viable'].sum())
    print(f"\nDHFR: {len(df_d)} rows -> {d_path}")
    print(f"  Viable: {n_viable_d}, Inviable: {len(df_d) - n_viable_d}")
    print(f"  PDB IDs: {sorted(df_d['pdb_id'].unique())}")


if __name__ == "__main__":
    main()
