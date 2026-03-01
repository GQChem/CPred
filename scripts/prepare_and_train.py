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


def parse_dataset_s3(supp_dir: Path) -> dict:
    """Parse Dataset S3 -> {pdb_id: {chain, sites: [int]}}."""
    ds3 = pd.read_excel(supp_dir / "Dataset_S3.xls")
    proteins = defaultdict(lambda: {"chain": "A", "sites": []})
    for _, row in ds3.iterrows():
        pdb_id = str(row["PDB ID"]).strip().lower()
        chain = str(row["Chain"]).strip() if pd.notna(row["Chain"]) else "A"
        site = int(row["CP site"])
        proteins[pdb_id]["chain"] = chain
        proteins[pdb_id]["sites"].append(site)
    return dict(proteins)


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
    """Worker: parse one PDB and return sequence/DSSP/structural codes/CP-site data."""
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

    exp_aa = []
    exp_dssp = []
    exp_rama = []
    exp_ka = []
    for site in info["sites"]:
        idx = resnum_to_idx.get(site)
        if idx is None:
            continue
        for pos in range(max(0, idx - 3), min(len(seq), idx + 4)):
            exp_aa.append(seq[pos])
            if dssp is not None:
                exp_dssp.append(dssp[pos])
            if rama_codes is not None:
                exp_rama.append(rama_codes[pos])
            exp_ka.append(ka_codes[pos])

    return seq, dssp, exp_aa, exp_dssp, rama_codes, ka_codes, exp_rama, exp_ka


def _make_ngrams_numpy(chars: list[str], n: int,
                       boundary_stride: int | None = None) -> list[str]:
    """Fast n-gram construction via numpy char array."""
    arr = np.array(list(chars), dtype="U1")
    if n == 1:
        return list(arr)
    views = np.stack([arr[i: len(arr) - (n - 1 - i)] for i in range(n)], axis=1)
    grams = np.apply_along_axis(lambda r: "".join(r), 1, views)
    if boundary_stride is not None:
        indices = np.arange(len(grams))
        keep = (indices % boundary_stride) <= (boundary_stride - n)
        grams = grams[keep]
    return list(grams)


def build_propensity_tables_from_data(proteins_data: dict,
                                       pdb_dir: Path,
                                       tables_dir: Path,
                                       use_gpu: bool = True,
                                       n_permutations: int = 99999) -> PropensityTables:
    """Build propensity tables from experimental CP sites vs whole sequences.

    Uses Dataset S3 (nrCPDB-40) for propensity table construction per paper.
    """
    print("Building propensity tables from training data...")

    pdb_ids_list = list(proteins_data.keys())
    total_pdb = len(pdb_ids_list)
    worker_args = [
        (pdb_id, proteins_data[pdb_id], str(pdb_dir / f"{pdb_id}.pdb"))
        for pdb_id in pdb_ids_list
    ]

    exp_aa = []
    comp_aa = []
    exp_dssp = []
    comp_dssp = []
    exp_rama = []
    comp_rama = []
    exp_ka = []
    comp_ka = []

    n_workers = min(8, os.cpu_count() or 1)
    print(f"  Parsing {total_pdb} PDB files with {n_workers} workers...", flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_parse_one_protein, a): a[0] for a in worker_args}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result is None:
                continue
            (seq, dssp, p_exp_aa, p_exp_dssp,
             rama_codes, ka_codes, p_exp_rama, p_exp_ka) = result
            comp_aa.extend(seq)
            if dssp is not None:
                comp_dssp.extend(dssp)
            if rama_codes is not None:
                comp_rama.extend(rama_codes)
            comp_ka.extend(ka_codes)
            exp_aa.extend(p_exp_aa)
            exp_dssp.extend(p_exp_dssp)
            exp_rama.extend(p_exp_rama)
            exp_ka.extend(p_exp_ka)
            if completed % 50 == 0 or completed == total_pdb:
                print(f"  Parsed {completed}/{total_pdb} proteins...", flush=True)

    pt = PropensityTables(tables_dir)

    if exp_aa and comp_aa:
        print(f"  Building single_aa table ({len(exp_aa)} exp, {len(comp_aa)} comp)...",
              flush=True)
        table = build_propensity_table(exp_aa, comp_aa, n_permutations=n_permutations, use_gpu=use_gpu)
        pt.save("single_aa", table)
        print(f"  -> single_aa done: {len(table)} elements", flush=True)

        exp_di = _make_ngrams_numpy(exp_aa, 2, boundary_stride=7)
        comp_di = _make_ngrams_numpy(comp_aa, 2)
        n_di = len(set(exp_di) | set(comp_di))
        print(f"  Building di_residue table ({len(exp_di)} exp, {len(comp_di)} comp, "
              f"{n_di} unique elements)...", flush=True)
        table = build_propensity_table(exp_di, comp_di, n_permutations=n_permutations, use_gpu=use_gpu)
        pt.save("di_residue", table)
        print(f"  -> di_residue done: {len(table)} elements", flush=True)

        exp_oligo = _make_ngrams_numpy(exp_aa, 3, boundary_stride=7)
        comp_oligo = _make_ngrams_numpy(comp_aa, 3)
        n_oligo = len(set(exp_oligo) | set(comp_oligo))
        print(f"  Building oligo_residue table ({len(exp_oligo)} exp, {len(comp_oligo)} comp, "
              f"{n_oligo} unique elements)...", flush=True)
        table = build_propensity_table(exp_oligo, comp_oligo, n_permutations=n_permutations, use_gpu=use_gpu)
        pt.save("oligo_residue", table)
        print(f"  -> oligo_residue done: {len(table)} elements", flush=True)
    else:
        for name in ["single_aa", "di_residue", "oligo_residue"]:
            pt.save(name, {aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"})

    if exp_dssp and comp_dssp:
        n_dssp = len(set(exp_dssp) | set(comp_dssp))
        print(f"  Building DSSP table ({len(exp_dssp)} exp, {len(comp_dssp)} comp, "
              f"{n_dssp} unique elements)...", flush=True)
        table = build_propensity_table(exp_dssp, comp_dssp, n_permutations=n_permutations, use_gpu=use_gpu)
        pt.save("dssp", table)
    else:
        pt.save("dssp", {ss: 0.0 for ss in "HBEGITSC"})

    if exp_rama and comp_rama:
        n_rama = len(set(exp_rama) | set(comp_rama))
        print(f"  Building ramachandran table ({len(exp_rama)} exp, {len(comp_rama)} comp, "
              f"{n_rama} unique elements)...", flush=True)
        table = build_propensity_table(exp_rama, comp_rama, n_permutations=n_permutations, use_gpu=use_gpu)
        pt.save("ramachandran", table)
        print(f"  -> ramachandran done: {len(table)} elements", flush=True)
    else:
        print("  WARNING: No Ramachandran codes available, using zeros")
        pt.save("ramachandran", {chr(65+i): 0.0 for i in range(23)})

    if exp_ka and comp_ka:
        n_ka = len(set(exp_ka) | set(comp_ka))
        print(f"  Building kappa_alpha table ({len(exp_ka)} exp, {len(comp_ka)} comp, "
              f"{n_ka} unique elements)...", flush=True)
        table = build_propensity_table(exp_ka, comp_ka, n_permutations=n_permutations, use_gpu=use_gpu)
        pt.save("kappa_alpha", table)
        print(f"  -> kappa_alpha done: {len(table)} elements", flush=True)
    else:
        print("  WARNING: No Kappa-Alpha codes available, using zeros")
        pt.save("kappa_alpha", {chr(65+i): 0.0 for i in range(23)})

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
    # Step 1: Parse Dataset T (training) and Dataset S3 (propensity)
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

    # Dataset S3 for propensity table construction
    proteins_s3 = parse_dataset_s3(supp_dir)
    total_s3 = len(proteins_s3)
    total_sites_s3 = sum(len(v["sites"]) for v in proteins_s3.values())
    print(f"Dataset S3: {total_s3} proteins with {total_sites_s3} CP sites (for propensity tables)")

    if args.max_proteins:
        pdb_ids = list(proteins_s3.keys())[:args.max_proteins]
        proteins_s3 = {k: proteins_s3[k] for k in pdb_ids}
        print(f"  Limited S3 to {len(proteins_s3)} proteins")

    # =========================================================
    # Step 2: Download PDB files
    # =========================================================
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 2: Downloading PDB structures")
        print("=" * 60)

        # Download Dataset S3 PDBs (for propensity tables)
        pdb_ids = list(proteins_s3.keys())
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
        print(f"  S3 PDBs - Downloaded: {downloaded}, Failed: {failed}")

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
    print("STEP 3: Building propensity tables (from Dataset S3)")
    print("=" * 60)

    use_gpu = not args.no_gpu
    required_tables = ["single_aa", "di_residue", "oligo_residue", "dssp",
                       "ramachandran", "kappa_alpha"]

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
        tables = build_propensity_tables_from_data(proteins_s3, pdb_dir, args.tables_dir,
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
            print(f"  {wa[0]}: {len(y)} labeled sites "
                  f"({int(y.sum())} viable, {len(y) - int(y.sum())} inviable)",
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
    print(f"  Positive (viable CP sites): {n_pos}")
    print(f"  Negative (inviable CP sites): {n_neg}")
    print(f"  Ratio: 1:{n_neg / max(n_pos, 1):.1f}")
    print("  (No downsampling needed — Dataset T is naturally balanced)")

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
