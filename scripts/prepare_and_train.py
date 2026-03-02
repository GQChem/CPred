#!/usr/bin/env python3
"""Complete training pipeline: parse Dataset T -> download PDBs -> extract features -> train.

This script:
1. Parses Dataset T (176 labeled CP sites from 6 proteins) for training
2. Parses Dataset S3 (nrCPDB-40) for propensity table construction
3. Downloads PDB structures from RCSB
4. Builds propensity tables from Dataset S3
5. Extracts features only for the 6 Dataset T proteins
6. Trains the ensemble model on 176 labeled sites
7. Evaluates on DHFR (1RX4) as independent test set
"""

import argparse
import os
import sys
import time
import pickle
import warnings
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpred.io.pdb_parser import parse_pdb, ProteinStructure
from cpred.features.structural_codes import assign_ramachandran_codes, assign_kappa_alpha_codes
from cpred.features.standardization import standardize_features
from cpred.propensity.tables import PropensityTables
from cpred.propensity.scoring import build_propensity_table
from cpred.pipeline import FEATURE_NAMES, build_feature_matrix, extract_all_features
from cpred.models.ensemble import CPredEnsemble
from cpred.training.evaluate import compute_metrics, cross_validate


def download_pdb(pdb_id: str, pdb_dir: Path, timeout: int = 30) -> Path | None:
    """Download PDB from RCSB. Returns path or None on failure."""
    pdb_id = pdb_id.strip().lower()
    dest = pdb_dir / f"{pdb_id}.pdb"
    if dest.exists():
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception:
        return None


def parse_dataset_s4(supp_dir: Path) -> dict:
    """Parse Dataset S4 (nrCPsitecpdb-40) -> {pdb_id: {chain, sites: [int]}}.

    Dataset S4 contains 1087 non-redundant viable CP sites from nrCPDB-40.
    These form the EXPERIMENTAL group for propensity table construction.
    """
    ds4 = pd.read_csv(supp_dir / "Dataset_S4.csv")
    proteins = defaultdict(lambda: {"chain": "A", "sites": []})
    for _, row in ds4.iterrows():
        pdb_id = str(row["PDB ID"]).strip().lower()
        chain = str(row["Chain"]).strip() if pd.notna(row["Chain"]) else "A"
        site = int(row["CP site"])
        proteins[pdb_id]["chain"] = chain
        proteins[pdb_id]["sites"].append(site)
    return dict(proteins)


def parse_dataset_s2(supp_dir: Path) -> dict:
    """Parse Dataset S2 (nrCPDB-40) -> {pdb_id: chain}.

    Dataset S2 contains 1059 non-redundant proteins from nrCPDB-40.
    Their WHOLE SEQUENCES form the COMPARISON group for propensity tables.
    """
    ds2 = pd.read_csv(supp_dir / "Dataset_S2.csv")
    proteins = {}
    for _, row in ds2.iterrows():
        pdb_id = str(row["PDB ID"]).strip().lower()
        chain = str(row["Chain"]).strip() if pd.notna(row["Chain"]) else "A"
        proteins[pdb_id] = chain
    return proteins


def parse_dataset_t(csv_path: Path) -> dict:
    """Parse dataset_t.csv -> {pdb_id: {chain, sites_viable: [int], sites_inviable: [int]}}.

    Returns one entry per unique PDB ID with lists of viable and inviable residue numbers.
    """
    df = pd.read_csv(csv_path)
    proteins = {}
    for _, row in df.iterrows():
        pdb_id = row['pdb_id']
        if pdb_id not in proteins:
            proteins[pdb_id] = {
                'chain': row['chain'],
                'sites_viable': [],
                'sites_inviable': [],
            }
        if row['viable'] == 1:
            proteins[pdb_id]['sites_viable'].append(int(row['residue_number']))
        else:
            proteins[pdb_id]['sites_inviable'].append(int(row['residue_number']))
    return proteins


def _extract_features_worker(args):
    """Worker: parse PDB, extract features, return (X, y) or (None, reason)."""
    pdb_id, info, pdb_path_str, tables = args[:4]
    rmsf_dir = args[4] if len(args) > 4 else None
    pdb_path = Path(pdb_path_str)
    if not pdb_path.exists():
        return None, f"{pdb_id}: PDB file not found"
    try:
        protein = parse_pdb(pdb_path, chain_id=info["chain"])
    except Exception as e:
        return None, f"{pdb_id}: parse error: {e}"

    X = extract_features_for_protein(protein, tables, rmsf_dir=rmsf_dir)
    if X is None:
        return None, f"{pdb_id}: feature extraction failed"

    resnum_to_idx = {rn: i for i, rn in enumerate(protein.residue_numbers)}
    y = np.zeros(protein.n_residues)
    for site in info.get("sites", []):
        idx = resnum_to_idx.get(site)
        if idx is not None:
            y[idx] = 1.0

    return (X, y), None


def _extract_features_dataset_t_worker(args):
    """Worker: parse PDB, extract features for Dataset T labeled sites only.

    Returns (X_labeled, y_labeled, pdb_id) where only the 176 labeled positions
    are included.
    """
    pdb_id, info, pdb_path_str, tables = args[:4]
    rmsf_dir = args[4] if len(args) > 4 else None
    pdb_path = Path(pdb_path_str)
    if not pdb_path.exists():
        return None, f"{pdb_id}: PDB file not found"
    try:
        protein = parse_pdb(pdb_path, chain_id=info["chain"])
    except Exception as e:
        return None, f"{pdb_id}: parse error: {e}"

    X = extract_features_for_protein(protein, tables, rmsf_dir=rmsf_dir)
    if X is None:
        return None, f"{pdb_id}: feature extraction failed"

    resnum_to_idx = {rn: i for i, rn in enumerate(protein.residue_numbers)}

    # Only keep labeled positions
    labeled_indices = []
    labels = []
    for site in info.get("sites_viable", []):
        idx = resnum_to_idx.get(site)
        if idx is not None:
            labeled_indices.append(idx)
            labels.append(1.0)
    for site in info.get("sites_inviable", []):
        idx = resnum_to_idx.get(site)
        if idx is not None:
            labeled_indices.append(idx)
            labels.append(0.0)

    if not labeled_indices:
        return None, f"{pdb_id}: no labeled sites found in structure"

    X_labeled = X[labeled_indices]
    y_labeled = np.array(labels)
    return (X_labeled, y_labeled), None


def extract_features_for_protein(protein: ProteinStructure,
                                  tables: PropensityTables,
                                  rmsf_dir=None) -> np.ndarray | None:
    """Extract and standardize all features for one protein."""
    try:
        features = extract_all_features(protein, tables, rmsf_dir=rmsf_dir)
        features = standardize_features(features)
        return build_feature_matrix(features)
    except Exception as e:
        return None


def _parse_one_protein(args):
    """Worker: parse one PDB (experimental group) and return CP-site windows + whole sequence.

    Used for Dataset S4 proteins (nrCPsitecpdb-40): extracts ±3-residue windows
    around each CP site (experimental group) AND the whole sequence (comparison group).
    """
    pdb_id, info, pdb_path_str = args
    pdb_path = Path(pdb_path_str)
    if not pdb_path.exists():
        return None
    try:
        protein = parse_pdb(pdb_path, chain_id=info["chain"])
    except Exception:
        return None

    seq = protein.sequence
    dssp = list(protein.dssp.ss) if protein.dssp is not None else None

    rama_codes = None
    if protein.dssp is not None:
        rama_codes = assign_ramachandran_codes(protein.dssp.phi, protein.dssp.psi)
    ka_codes = assign_kappa_alpha_codes(protein.ca_coords)

    resnum_to_idx = {rn: i for i, rn in enumerate(protein.residue_numbers)}

    # exp_*_windows: list of per-site windows (each window = list of codes)
    # exp_*_flat: flat list for single-code tables
    exp_aa_windows = []
    exp_dssp_windows = []
    exp_rama_windows = []
    exp_ka_windows = []
    exp_aa_flat = []
    exp_dssp_flat = []
    exp_rama_flat = []
    exp_ka_flat = []

    for site in info["sites"]:
        idx = resnum_to_idx.get(site)
        if idx is None:
            continue
        positions = range(max(0, idx - 3), min(len(seq), idx + 4))
        win_aa, win_dssp, win_rama, win_ka = [], [], [], []
        for pos in positions:
            win_aa.append(seq[pos])
            if dssp is not None:
                win_dssp.append(dssp[pos])
            if rama_codes is not None:
                win_rama.append(rama_codes[pos])
            win_ka.append(ka_codes[pos])
        exp_aa_windows.append(win_aa)
        exp_aa_flat.extend(win_aa)
        if win_dssp:
            exp_dssp_windows.append(win_dssp)
            exp_dssp_flat.extend(win_dssp)
        if win_rama:
            exp_rama_windows.append(win_rama)
            exp_rama_flat.extend(win_rama)
        exp_ka_windows.append(win_ka)
        exp_ka_flat.extend(win_ka)

    return (seq, dssp, rama_codes, ka_codes,
            exp_aa_flat, exp_dssp_flat, exp_rama_flat, exp_ka_flat,
            exp_aa_windows, exp_dssp_windows, exp_rama_windows, exp_ka_windows)


def _parse_one_protein_comp(args):
    """Worker: parse one PDB (comparison group) and return whole-sequence data only.

    Used for Dataset S2 proteins (nrCPDB-40) that are NOT in Dataset S4:
    their whole sequences contribute only to the comparison group.
    """
    pdb_id, chain, pdb_path_str = args
    pdb_path = Path(pdb_path_str)
    if not pdb_path.exists():
        return None
    try:
        protein = parse_pdb(pdb_path, chain_id=chain)
    except Exception:
        return None

    seq = protein.sequence
    dssp = list(protein.dssp.ss) if protein.dssp is not None else None
    rama_codes = None
    if protein.dssp is not None:
        rama_codes = assign_ramachandran_codes(protein.dssp.phi, protein.dssp.psi)
    ka_codes = assign_kappa_alpha_codes(protein.ca_coords)

    return (seq, dssp, rama_codes, ka_codes)


def _make_ngrams_numpy(chars: list[str], n: int) -> list[str]:
    """Fast n-gram construction via numpy char array (whole sequence)."""
    arr = np.array(list(chars), dtype="U1")
    if n == 1:
        return list(arr)
    views = np.stack([arr[i: len(arr) - (n - 1 - i)] for i in range(n)], axis=1)
    return list(np.apply_along_axis(lambda r: "".join(r), 1, views))


def _make_windowed_ngrams(windows: list[list[str]], n: int) -> list[str]:
    """Extract n-grams from a list of windows without crossing window boundaries.

    Each window is a list of characters (up to 7 for a ±3 window).
    n-grams are only formed within a single window.
    """
    grams = []
    for window in windows:
        arr = np.array(window, dtype="U1")
        if len(arr) < n:
            continue
        views = np.stack([arr[i: len(arr) - (n - 1 - i)] for i in range(n)], axis=1)
        grams.extend(np.apply_along_axis(lambda r: "".join(r), 1, views))
    return grams


def build_propensity_tables_from_data(proteins_s4: dict,
                                       proteins_s2: dict,
                                       pdb_dir: Path,
                                       tables_dir: Path,
                                       use_gpu: bool = True,
                                       n_permutations: int = 99999) -> PropensityTables:
    """Build propensity tables from experimental CP sites vs whole sequences.

    Experimental group: CP site ±3-residue windows from Dataset S4 (nrCPsitecpdb-40,
        1087 sites from 760 PDB IDs).
    Comparison group: whole sequences of ALL Dataset S2 proteins (nrCPDB-40,
        1059 proteins). This is the correct background per Lo et al. 2012.

    Args:
        proteins_s4: {pdb_id: {chain, sites: [int]}} — experimental (CP sites)
        proteins_s2: {pdb_id: chain} — comparison (nrCPDB-40 whole sequences)
    """
    print("Building propensity tables from training data...")

    # --- Parse experimental proteins (Dataset S4: CP site windows + whole seq) ---
    s4_args = [
        (pdb_id, proteins_s4[pdb_id], str(pdb_dir / f"{pdb_id}.pdb"))
        for pdb_id in proteins_s4
    ]
    # S2 proteins NOT already in S4 contribute only to comparison group
    s4_pdb_ids = set(proteins_s4.keys())
    s2_only_args = [
        (pdb_id, proteins_s2[pdb_id], str(pdb_dir / f"{pdb_id}.pdb"))
        for pdb_id in proteins_s2
        if pdb_id not in s4_pdb_ids
    ]

    exp_aa_flat = []
    exp_dssp_flat = []
    exp_rama_flat = []
    exp_ka_flat = []
    exp_aa_windows = []
    exp_dssp_windows = []
    exp_rama_windows = []
    exp_ka_windows = []
    comp_aa = []
    comp_dssp = []
    comp_rama = []
    comp_ka = []

    n_workers = min(8, os.cpu_count() or 1)
    total_s4 = len(s4_args)
    total_s2_only = len(s2_only_args)

    # Parse S4 proteins (experimental + comparison)
    print(f"  Parsing {total_s4} S4 (experimental) PDB files with {n_workers} workers...",
          flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_parse_one_protein, a): a[0] for a in s4_args}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result is None:
                continue
            (seq, dssp, rama_codes, ka_codes,
             p_exp_aa, p_exp_dssp, p_exp_rama, p_exp_ka,
             p_win_aa, p_win_dssp, p_win_rama, p_win_ka) = result
            # S4 proteins contribute to BOTH experimental and comparison
            comp_aa.extend(seq)
            if dssp is not None:
                comp_dssp.extend(dssp)
            if rama_codes is not None:
                comp_rama.extend(rama_codes)
            comp_ka.extend(ka_codes)
            exp_aa_flat.extend(p_exp_aa)
            exp_dssp_flat.extend(p_exp_dssp)
            exp_rama_flat.extend(p_exp_rama)
            exp_ka_flat.extend(p_exp_ka)
            exp_aa_windows.extend(p_win_aa)
            exp_dssp_windows.extend(p_win_dssp)
            exp_rama_windows.extend(p_win_rama)
            exp_ka_windows.extend(p_win_ka)
            if completed % 50 == 0 or completed == total_s4:
                print(f"  Parsed {completed}/{total_s4} S4 proteins...", flush=True)

    # Parse S2-only proteins (comparison only)
    print(f"  Parsing {total_s2_only} S2-only (comparison) PDB files with {n_workers} workers...",
          flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_parse_one_protein_comp, a): a[0] for a in s2_only_args}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result is None:
                continue
            seq, dssp, rama_codes, ka_codes = result
            comp_aa.extend(seq)
            if dssp is not None:
                comp_dssp.extend(dssp)
            if rama_codes is not None:
                comp_rama.extend(rama_codes)
            comp_ka.extend(ka_codes)
            if completed % 50 == 0 or completed == total_s2_only:
                print(f"  Parsed {completed}/{total_s2_only} S2-only proteins...", flush=True)

    pt = PropensityTables(tables_dir)

    if not exp_aa_flat or not comp_aa:
        raise RuntimeError("No amino acid data collected — check PDB parsing")
    if not exp_dssp_flat or not comp_dssp:
        raise RuntimeError("No DSSP data collected — check DSSP is running correctly")
    if not exp_rama_flat or not comp_rama:
        raise RuntimeError("No Ramachandran data collected")
    if not exp_ka_flat or not comp_ka:
        raise RuntimeError("No Kappa-Alpha data collected")

    # --- single AA ---
    print(f"  Building single_aa ({len(exp_aa_flat)} exp, {len(comp_aa)} comp)...", flush=True)
    table = build_propensity_table(exp_aa_flat, comp_aa, n_permutations=n_permutations, use_gpu=use_gpu)
    pt.save("single_aa", table)
    print(f"  -> single_aa done: {len(table)} elements", flush=True)

    # --- di_residue: windowed experimental, whole-sequence background ---
    exp_di = _make_windowed_ngrams(exp_aa_windows, 2)
    comp_di = _make_ngrams_numpy(comp_aa, 2)
    print(f"  Building di_residue ({len(exp_di)} exp, {len(comp_di)} comp, "
          f"{len(set(exp_di)|set(comp_di))} unique)...", flush=True)
    table = build_propensity_table(exp_di, comp_di, n_permutations=n_permutations, use_gpu=use_gpu)
    pt.save("di_residue", table)
    print(f"  -> di_residue done: {len(table)} elements", flush=True)

    # --- oligo_residue ---
    exp_oligo = _make_windowed_ngrams(exp_aa_windows, 3)
    comp_oligo = _make_ngrams_numpy(comp_aa, 3)
    print(f"  Building oligo_residue ({len(exp_oligo)} exp, {len(comp_oligo)} comp, "
          f"{len(set(exp_oligo)|set(comp_oligo))} unique)...", flush=True)
    table = build_propensity_table(exp_oligo, comp_oligo, n_permutations=n_permutations, use_gpu=use_gpu)
    pt.save("oligo_residue", table)
    print(f"  -> oligo_residue done: {len(table)} elements", flush=True)

    # --- dssp ---
    print(f"  Building dssp ({len(exp_dssp_flat)} exp, {len(comp_dssp)} comp)...", flush=True)
    table = build_propensity_table(exp_dssp_flat, comp_dssp, n_permutations=n_permutations, use_gpu=use_gpu)
    pt.save("dssp", table)
    print(f"  -> dssp done: {len(table)} elements", flush=True)

    exp_di_dssp = _make_windowed_ngrams(exp_dssp_windows, 2)
    comp_di_dssp = _make_ngrams_numpy(comp_dssp, 2)
    print(f"  Building di_dssp ({len(exp_di_dssp)} exp, {len(comp_di_dssp)} comp, "
          f"{len(set(exp_di_dssp)|set(comp_di_dssp))} unique)...", flush=True)
    table = build_propensity_table(exp_di_dssp, comp_di_dssp, n_permutations=n_permutations, use_gpu=use_gpu)
    pt.save("di_dssp", table)
    print(f"  -> di_dssp done: {len(table)} elements", flush=True)

    # --- ramachandran ---
    print(f"  Building ramachandran ({len(exp_rama_flat)} exp, {len(comp_rama)} comp)...", flush=True)
    table = build_propensity_table(exp_rama_flat, comp_rama, n_permutations=n_permutations, use_gpu=use_gpu)
    pt.save("ramachandran", table)
    print(f"  -> ramachandran done: {len(table)} elements", flush=True)

    exp_di_rama = _make_windowed_ngrams(exp_rama_windows, 2)
    comp_di_rama = _make_ngrams_numpy(comp_rama, 2)
    print(f"  Building di_ramachandran ({len(exp_di_rama)} exp, {len(comp_di_rama)} comp, "
          f"{len(set(exp_di_rama)|set(comp_di_rama))} unique)...", flush=True)
    table = build_propensity_table(exp_di_rama, comp_di_rama, n_permutations=n_permutations, use_gpu=use_gpu)
    pt.save("di_ramachandran", table)
    print(f"  -> di_ramachandran done: {len(table)} elements", flush=True)

    # --- kappa_alpha ---
    print(f"  Building kappa_alpha ({len(exp_ka_flat)} exp, {len(comp_ka)} comp)...", flush=True)
    table = build_propensity_table(exp_ka_flat, comp_ka, n_permutations=n_permutations, use_gpu=use_gpu)
    pt.save("kappa_alpha", table)
    print(f"  -> kappa_alpha done: {len(table)} elements", flush=True)

    exp_di_ka = _make_windowed_ngrams(exp_ka_windows, 2)
    comp_di_ka = _make_ngrams_numpy(comp_ka, 2)
    print(f"  Building di_kappa_alpha ({len(exp_di_ka)} exp, {len(comp_di_ka)} comp, "
          f"{len(set(exp_di_ka)|set(comp_di_ka))} unique)...", flush=True)
    table = build_propensity_table(exp_di_ka, comp_di_ka, n_permutations=n_permutations, use_gpu=use_gpu)
    pt.save("di_kappa_alpha", table)
    print(f"  -> di_kappa_alpha done: {len(table)} elements", flush=True)

    pt.load()
    print("  Propensity tables built and saved.")
    return pt


def main():
    parser = argparse.ArgumentParser(description="Complete CPred training pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("cpred/data/trained_models"))
    parser.add_argument("--tables-dir", type=Path, default=Path("cpred/data/propensity_tables"))
    parser.add_argument("--max-proteins", type=int, default=None,
                        help="Limit number of proteins to process (for testing)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration for permutation tests")
    parser.add_argument("--n-permutations", type=int, default=99999,
                        help="Number of permutations for propensity p-values (Lo et al. 2012)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Process proteins in batches")
    parser.add_argument("--rmsf-dir", type=Path, default=Path("data/rmsf"),
                        help="Directory with per-protein RMSF CSVs from CABSflex")
    args = parser.parse_args()

    supp_dir = args.data_dir / "supplementary"
    pdb_dir = args.data_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Step 1: Parse datasets
    # =========================================================
    print("=" * 60)
    print("STEP 1: Parsing datasets")
    print("=" * 60)

    # Dataset T for training (176 labeled sites from 6 proteins)
    dataset_t_path = supp_dir / "dataset_t.csv"
    if not dataset_t_path.exists():
        print(f"  {dataset_t_path} not found. Run scripts/convert_dataset_l.py first.")
        sys.exit(1)
    dataset_t = parse_dataset_t(dataset_t_path)
    n_viable = sum(len(v["sites_viable"]) for v in dataset_t.values())
    n_inviable = sum(len(v["sites_inviable"]) for v in dataset_t.values())
    print(f"Dataset T: {len(dataset_t)} proteins, "
          f"{n_viable} viable + {n_inviable} inviable = {n_viable + n_inviable} sites")

    # Dataset S4 (nrCPsitecpdb-40): 1087 CP sites — experimental group for propensity
    proteins_s4 = parse_dataset_s4(supp_dir)
    total_sites_s4 = sum(len(v["sites"]) for v in proteins_s4.values())
    print(f"Dataset S4: {len(proteins_s4)} proteins with {total_sites_s4} CP sites (experimental group)")

    # Dataset S2 (nrCPDB-40): 1059 proteins — comparison group for propensity
    proteins_s2 = parse_dataset_s2(supp_dir)
    print(f"Dataset S2: {len(proteins_s2)} proteins (comparison group, whole sequences)")

    if args.max_proteins:
        pdb_ids = list(proteins_s4.keys())[:args.max_proteins]
        proteins_s4 = {k: proteins_s4[k] for k in pdb_ids}
        # Also limit S2 to the same PDB IDs for fast testing
        proteins_s2 = {k: v for k, v in proteins_s2.items() if k in proteins_s4}
        print(f"  Limited to {len(proteins_s4)} S4 proteins (--max-proteins)")

    # =========================================================
    # Step 2: Download PDB files
    # =========================================================
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 2: Downloading PDB structures")
        print("=" * 60)

        # Download all nrCPDB-40 PDBs (S4 experimental + S2 comparison)
        all_propensity_pdbs = set(proteins_s4.keys()) | set(proteins_s2.keys())
        pdb_ids = sorted(all_propensity_pdbs)
        downloaded = 0
        failed = 0
        for i, pdb_id in enumerate(pdb_ids):
            path = download_pdb(pdb_id, pdb_dir)
            if path:
                downloaded += 1
            else:
                failed += 1
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(pdb_ids)} "
                      f"(downloaded: {downloaded}, failed: {failed})")
        print(f"  S2+S4 PDBs - Downloaded: {downloaded}, Failed: {failed}")

        # Download Dataset T PDBs (for training)
        for pdb_id in dataset_t:
            download_pdb(pdb_id, pdb_dir)
        print(f"  Dataset T PDBs downloaded: {list(dataset_t.keys())}")

    # Download DHFR for independent test
    print("  Downloading DHFR (1RX4)...")
    download_pdb("1rx4", pdb_dir)

    # =========================================================
    # Step 3: Build propensity tables from Dataset S3
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 3: Building propensity tables (S4 experimental vs S2 comparison)")
    print("=" * 60)

    use_gpu = not args.no_gpu
    required_tables = ["single_aa", "di_residue", "oligo_residue",
                       "dssp", "di_dssp", "ramachandran", "di_ramachandran",
                       "kappa_alpha", "di_kappa_alpha"]

    def _table_needs_rebuild(name):
        path = args.tables_dir / f"{name}.json"
        if not path.exists() or path.stat().st_size < 10:
            return True
        import json
        try:
            with open(path) as f:
                data = json.load(f)
            if all(v == 0.0 for v in data.values()):
                return True
        except Exception:
            return True
        return False

    missing = [t for t in required_tables if _table_needs_rebuild(t)]

    if not missing:
        print("  All propensity tables already exist, loading from disk...")
        tables = PropensityTables(args.tables_dir)
        tables.load()
    else:
        print(f"  Missing or incomplete tables: {missing}")
        tables = build_propensity_tables_from_data(proteins_s4, proteins_s2,
                                                    pdb_dir, args.tables_dir,
                                                    use_gpu=use_gpu,
                                                    n_permutations=args.n_permutations)

    # =========================================================
    # Step 4: Extract features for Dataset T proteins only
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 4: Extracting features (Dataset T proteins only)")
    print("=" * 60)

    all_X = []
    all_y = []
    processed = 0
    skipped = 0

    pdb_ids = list(dataset_t.keys())
    rmsf_dir = str(args.rmsf_dir) if args.rmsf_dir else None
    worker_args = [
        (pdb_id, dataset_t[pdb_id], str(pdb_dir / f"{pdb_id}.pdb"), tables, rmsf_dir)
        for pdb_id in pdb_ids
    ]

    # Extract features for each Dataset T protein, keeping only labeled sites
    for wa in worker_args:
        result, reason = _extract_features_dataset_t_worker(wa)
        if result is None:
            skipped += 1
            print(f"  SKIP: {reason}", flush=True)
        else:
            X, y = result
            all_X.append(X)
            all_y.append(y)
            processed += 1
            # Per-protein debug: feature stats for a few key features
            cat_a_mean = X[:, :len([f for f in FEATURE_NAMES if f in
                                     ["R_aa","R_aac3","RxR_aac3","2R_aac3"]])].mean()
            rmsf_idx = FEATURE_NAMES.index("rmsf") if "rmsf" in FEATURE_NAMES else None
            rmsf_info = (f", RMSF nonzero={int(np.count_nonzero(~np.isnan(X[:, rmsf_idx])))}/{len(y)}"
                         if rmsf_idx is not None else "")
            print(f"  {wa[0]}: {len(y)} labeled sites "
                  f"({int(y.sum())} viable, {len(y) - int(y.sum())} inviable)"
                  f"{rmsf_info}",
                  flush=True)

    print(f"  Processed: {processed}, Skipped: {skipped}", flush=True)

    if not all_X:
        print("ERROR: No training data extracted!")
        sys.exit(1)

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"\n  Training matrix: {X_train.shape}")
    print(f"  Feature names ({len(FEATURE_NAMES)}): {FEATURE_NAMES[:5]}...{FEATURE_NAMES[-3:]}")
    print(f"  Positive (viable CP sites): {n_pos}")
    print(f"  Negative (inviable CP sites): {n_neg}")
    print(f"  Ratio: 1:{n_neg / max(n_pos, 1):.1f}")
    print("  (No downsampling needed — Dataset T is naturally balanced)")

    # Feature sanity check: viable vs inviable mean per category
    pos_mask = y_train == 1
    neg_mask = y_train == 0
    cat_a_end = len([f for f in FEATURE_NAMES if f.startswith(("R_", "2R_", "RxR", "R2x", "R3x", "3R_", "4R_", "5R_"))])
    cat_b_end = cat_a_end + len([f for f in FEATURE_NAMES if f.endswith(("_sse", "_rm", "_ka"))])
    print(f"\n  Feature means (viable vs inviable):")
    print(f"    Cat A (seq propensity):  viable={X_train[pos_mask, :cat_a_end].mean():.4f}  "
          f"inviable={X_train[neg_mask, :cat_a_end].mean():.4f}")
    print(f"    Cat B (SS propensity):   viable={X_train[pos_mask, cat_a_end:cat_b_end].mean():.4f}  "
          f"inviable={X_train[neg_mask, cat_a_end:cat_b_end].mean():.4f}")
    print(f"    Cat C (tertiary):        viable={X_train[pos_mask, cat_b_end:].mean():.4f}  "
          f"inviable={X_train[neg_mask, cat_b_end:].mean():.4f}")
    # Check key individual features
    for fname in ["R_aa", "RxR_aac3", "rsa", "farness_buried", "rmsf", "gnm_msf"]:
        if fname in FEATURE_NAMES:
            idx = FEATURE_NAMES.index(fname)
            v_mean = X_train[pos_mask, idx].mean()
            n_mean = X_train[neg_mask, idx].mean()
            print(f"    {fname:20s}: viable={v_mean:+.3f}  inviable={n_mean:+.3f}  "
                  f"diff={v_mean - n_mean:+.3f}")

    # =========================================================
    # Step 5: Train final ensemble
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 5: Training final ensemble model")
    print("=" * 60)

    ensemble = CPredEnsemble(feature_names=FEATURE_NAMES)
    ensemble.fit(X_train, y_train, feature_names=FEATURE_NAMES)

    # Evaluate on training set
    train_probs = ensemble.predict_unsmoothed(X_train)
    train_metrics = compute_metrics(y_train, train_probs)
    print("\nEnsemble training set metrics:")
    for key, val in train_metrics.items():
        print(f"  {key}: {val:.4f}")

    # Individual model metrics
    individual = ensemble.predict_individual(X_train)
    for model_name, probs in individual.items():
        metrics = compute_metrics(y_train, probs)
        print(f"\n  {model_name.upper()} metrics:")
        for key, val in metrics.items():
            print(f"    {key}: {val:.4f}")

    # Save models
    ensemble.save(args.output_dir)
    print(f"\nModels saved to {args.output_dir}")

    # =========================================================
    # Step 6: Independent test on DHFR (1RX4)
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 6: Independent test on DHFR (1RX4)")
    print("=" * 60)

    dhfr_csv = supp_dir / "dataset_dhfr.csv"
    dhfr_path = pdb_dir / "1rx4.pdb"

    if dhfr_path.exists() and dhfr_csv.exists():
        try:
            dhfr = parse_pdb(dhfr_path, chain_id="A")
            X_dhfr = extract_features_for_protein(dhfr, tables, rmsf_dir=rmsf_dir)

            if X_dhfr is not None:
                # Get DHFR labels
                dhfr_df = pd.read_csv(dhfr_csv)
                resnum_to_idx = {rn: i for i, rn in enumerate(dhfr.residue_numbers)}

                labeled_indices = []
                labels = []
                for _, row in dhfr_df.iterrows():
                    idx = resnum_to_idx.get(int(row['residue_number']))
                    if idx is not None:
                        labeled_indices.append(idx)
                        labels.append(int(row['viable']))

                if labeled_indices:
                    X_dhfr_labeled = X_dhfr[labeled_indices]
                    y_dhfr = np.array(labels)

                    # Predict (smoothed, on full protein then extract labeled)
                    probs_full = ensemble.predict(X_dhfr)
                    probs_labeled = probs_full[labeled_indices]

                    dhfr_metrics = compute_metrics(y_dhfr, probs_labeled)
                    print(f"\n  DHFR results ({len(y_dhfr)} sites, "
                          f"{int(y_dhfr.sum())} viable, "
                          f"{len(y_dhfr) - int(y_dhfr.sum())} inviable):")
                    for key, val in dhfr_metrics.items():
                        print(f"    {key}: {val:.4f}")

                    # Per-model DHFR metrics
                    individual_dhfr = ensemble.predict_individual(X_dhfr)
                    for mname, mprobs in individual_dhfr.items():
                        m = compute_metrics(y_dhfr, mprobs[labeled_indices])
                        print(f"    [{mname.upper():3s}] AUC={m['auc']:.4f}  "
                              f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
                              f"MCC={m['mcc']:.3f}")

                    print("\n  Paper reference (DHFR independent test):")
                    print("    AUC:         0.906")
                    print("    Sensitivity: 0.709")
                    print("    Specificity: 0.918")
                    print("    MCC:         0.633")

                    # Show top 10 predicted sites
                    top_indices = np.argsort(probs_full)[-10:][::-1]
                    print("\n  Top 10 predicted CP sites:")
                    print(f"  {'Res#':>5} {'AA':>3} {'Prob':>6}")
                    for idx in top_indices:
                        print(f"  {dhfr.residue_numbers[idx]:>5} "
                              f"{dhfr.sequence[idx]:>3} "
                              f"{probs_full[idx]:>6.4f}")
        except Exception as e:
            print(f"  DHFR test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        missing = []
        if not dhfr_path.exists():
            missing.append("PDB")
        if not dhfr_csv.exists():
            missing.append("CSV (run convert_dataset_l.py)")
        print(f"  DHFR files not found ({', '.join(missing)}), skipping")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
