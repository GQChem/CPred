#!/usr/bin/env python3
"""Download PLoS ONE supplementary datasets and PDB structures for CPred."""

import argparse
import os
from pathlib import Path

import requests

PLOS_ONE_DOI = "10.1371/journal.pone.0031791"
PLOS_ONE_BASE = "https://doi.org/10.1371/journal.pone.0031791"

# Supplementary file URLs from PLoS ONE (Lo et al., 2012)
SUPPLEMENTARY_FILES = {
    "Dataset_S1.xls": "https://doi.org/10.1371/journal.pone.0031791.s002",
    "Dataset_S2.xls": "https://doi.org/10.1371/journal.pone.0031791.s003",
    "Dataset_S3.xls": "https://doi.org/10.1371/journal.pone.0031791.s004",
    "Dataset_S4.xls": "https://doi.org/10.1371/journal.pone.0031791.s005",
    "Dataset_S5.xls": "https://doi.org/10.1371/journal.pone.0031791.s006",
    "Table_S1.xls": "https://doi.org/10.1371/journal.pone.0031791.s007",
    "Table_S2.xls": "https://doi.org/10.1371/journal.pone.0031791.s008",
    "Table_S3.xls": "https://doi.org/10.1371/journal.pone.0031791.s009",
}

# PDB IDs needed for training and verification
DHFR_PDB = "1RX4"


def download_file(url: str, dest: Path, timeout: int = 60) -> None:
    """Download a file from URL to destination path."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...")
    resp = requests.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  Saved {dest.name} ({len(resp.content)} bytes)")


def download_pdb(pdb_id: str, dest_dir: Path, timeout: int = 60) -> None:
    """Download a PDB file from RCSB."""
    pdb_id = pdb_id.upper()
    filename = f"{pdb_id.lower()}.pdb"
    dest = dest_dir / filename
    if dest.exists():
        print(f"  Already exists: {filename}")
        return
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {filename} from RCSB ...")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  Saved {filename} ({len(resp.content)} bytes)")


def extract_pdb_ids_from_dataset(dataset_path: Path) -> list[str]:
    """Extract PDB IDs from a downloaded dataset file.

    Attempts to read .xls files (which are often HTML tables from PLoS ONE).
    Falls back to pandas if openpyxl is available.
    """
    try:
        import pandas as pd
        df = pd.read_excel(dataset_path)
        # Look for columns containing PDB IDs
        for col in df.columns:
            col_lower = str(col).lower()
            if "pdb" in col_lower:
                return [str(v).strip().upper()[:4] for v in df[col].dropna().unique()
                        if len(str(v).strip()) >= 4]
    except Exception:
        pass
    return []


def main():
    parser = argparse.ArgumentParser(description="Download CPred training data")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory to store downloaded files (default: data)")
    parser.add_argument("--skip-pdb", action="store_true",
                        help="Skip downloading PDB structures")
    args = parser.parse_args()

    data_dir = args.data_dir
    supp_dir = data_dir / "supplementary"
    pdb_dir = data_dir / "pdb"

    supp_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    # Download supplementary files
    print("Downloading PLoS ONE supplementary files...")
    for filename, url in SUPPLEMENTARY_FILES.items():
        try:
            download_file(url, supp_dir / filename)
        except Exception as e:
            print(f"  WARNING: Failed to download {filename}: {e}")

    if not args.skip_pdb:
        # Download DHFR structure for verification
        print("\nDownloading DHFR (1RX4) for verification...")
        download_pdb(DHFR_PDB, pdb_dir)

        # Try to extract PDB IDs from Dataset S1 (training set) and download
        ds1 = supp_dir / "Dataset_S1.xls"
        if ds1.exists():
            print("\nExtracting PDB IDs from Dataset S1...")
            pdb_ids = extract_pdb_ids_from_dataset(ds1)
            if pdb_ids:
                print(f"Found {len(pdb_ids)} PDB IDs. Downloading structures...")
                for pdb_id in pdb_ids:
                    try:
                        download_pdb(pdb_id, pdb_dir)
                    except Exception as e:
                        print(f"  WARNING: Failed to download {pdb_id}: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
