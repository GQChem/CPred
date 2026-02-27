#!/usr/bin/env python3
"""Complete training pipeline: parse datasets -> download PDBs -> extract features -> train.

This script:
1. Parses Dataset S3 (experimental CP sites) to get PDB IDs + CP site positions
2. Downloads PDB structures from RCSB
3. Extracts features for each protein
4. Builds propensity tables from the data
5. Trains the ensemble model
6. Evaluates via cross-validation
7. Runs verification on DHFR (1RX4)
"""

import argparse
import os
import sys
import time
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpred.io.pdb_parser import parse_pdb, ProteinStructure
from cpred.features.tertiary_structure import extract_tertiary_features
from cpred.features.contact_network import extract_contact_network_features
from cpred.features.gnm import compute_gnm_fluctuation
from cpred.features.sequence_propensity import extract_sequence_propensity_features
from cpred.features.secondary_structure import extract_secondary_structure_features
from cpred.features.window import window_average_dict
from cpred.features.standardization import standardize_features
from cpred.propensity.tables import PropensityTables
from cpred.propensity.scoring import build_propensity_table
from cpred.pipeline import FEATURE_NAMES, build_feature_matrix
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


def extract_features_for_protein(protein: ProteinStructure,
                                  tables: PropensityTables) -> np.ndarray | None:
    """Extract and standardize all features for one protein."""
    try:
        features = {}

        # Category A
        seq_feats = extract_sequence_propensity_features(protein.sequence, tables)
        seq_feats = window_average_dict(seq_feats)
        features.update(seq_feats)

        # Category B
        ss_feats = extract_secondary_structure_features(protein, tables)
        ss_feats = window_average_dict(ss_feats)
        features.update(ss_feats)

        # Category C
        tert_feats = extract_tertiary_features(protein)
        contact_feats = extract_contact_network_features(protein)
        gnm_msf = compute_gnm_fluctuation(protein.ca_coords)

        cat_c = {}
        cat_c.update(tert_feats)
        cat_c.update(contact_feats)
        cat_c["gnm_msf"] = gnm_msf
        cat_c = window_average_dict(cat_c)
        features.update(cat_c)

        # Standardize
        features = standardize_features(features)

        return build_feature_matrix(features)
    except Exception as e:
        return None


def build_propensity_tables_from_data(proteins_data: dict,
                                       pdb_dir: Path,
                                       tables_dir: Path,
                                       use_gpu: bool = True) -> PropensityTables:
    """Build propensity tables from experimental CP sites vs whole sequences."""
    print("Building propensity tables from training data...")

    exp_aa = []  # AA at CP site windows
    comp_aa = []  # AA from whole sequences
    exp_dssp = []
    comp_dssp = []

    pdb_ids_list = list(proteins_data.keys())
    total_pdb = len(pdb_ids_list)
    for pi, pdb_id in enumerate(pdb_ids_list, 1):
        info = proteins_data[pdb_id]
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            continue
        try:
            protein = parse_pdb(pdb_path, chain_id=info["chain"])
        except Exception:
            continue

        seq = protein.sequence
        comp_aa.extend(list(seq))

        # DSSP
        if protein.dssp is not None:
            comp_dssp.extend(protein.dssp.ss)

        # Extract elements from CP site windows
        for site in info["sites"]:
            # Find index matching the site residue number
            try:
                idx = protein.residue_numbers.index(site)
            except ValueError:
                continue
            for offset in range(-3, 4):
                pos = idx + offset
                if 0 <= pos < len(seq):
                    exp_aa.append(seq[pos])
                    if protein.dssp is not None:
                        exp_dssp.append(protein.dssp.ss[pos])

        if pi % 50 == 0 or pi == total_pdb:
            print(f"  Parsed {pi}/{total_pdb} proteins for propensity data...",
                  flush=True)

    pt = PropensityTables(tables_dir)

    if exp_aa and comp_aa:
        print(f"  Building single_aa table ({len(exp_aa)} exp, {len(comp_aa)} comp)...",
              flush=True)
        table = build_propensity_table(exp_aa, comp_aa, n_permutations=1000, use_gpu=use_gpu)
        pt.save("single_aa", table)
        print(f"  -> single_aa done: {len(table)} elements", flush=True)

        # Di-residue from concatenated sequences
        exp_di = [exp_aa[i] + exp_aa[i+1] for i in range(len(exp_aa)-1)
                  if i % 7 != 6]  # don't cross window boundaries
        comp_di = [comp_aa[i] + comp_aa[i+1] for i in range(len(comp_aa)-1)]
        n_di = len(set(exp_di) | set(comp_di))
        print(f"  Building di_residue table ({len(exp_di)} exp, {len(comp_di)} comp, "
              f"{n_di} unique elements)...", flush=True)
        table = build_propensity_table(exp_di, comp_di, n_permutations=1000, use_gpu=use_gpu)
        pt.save("di_residue", table)
        print(f"  -> di_residue done: {len(table)} elements", flush=True)

        # Oligo-residue
        exp_oligo = [exp_aa[i] + exp_aa[i+1] + exp_aa[i+2]
                     for i in range(len(exp_aa)-2)
                     if i % 7 < 5]
        comp_oligo = [comp_aa[i] + comp_aa[i+1] + comp_aa[i+2]
                      for i in range(len(comp_aa)-2)]
        n_oligo = len(set(exp_oligo) | set(comp_oligo))
        print(f"  Building oligo_residue table ({len(exp_oligo)} exp, {len(comp_oligo)} comp, "
              f"{n_oligo} unique elements)...", flush=True)
        table = build_propensity_table(exp_oligo, comp_oligo, n_permutations=1000, use_gpu=use_gpu)
        pt.save("oligo_residue", table)
        print(f"  -> oligo_residue done: {len(table)} elements", flush=True)
    else:
        for name in ["single_aa", "di_residue", "oligo_residue"]:
            pt.save(name, {aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"})

    if exp_dssp and comp_dssp:
        n_dssp = len(set(exp_dssp) | set(comp_dssp))
        print(f"  Building DSSP table ({len(exp_dssp)} exp, {len(comp_dssp)} comp, "
              f"{n_dssp} unique elements)...", flush=True)
        table = build_propensity_table(exp_dssp, comp_dssp, n_permutations=1000, use_gpu=use_gpu)
        pt.save("dssp", table)
    else:
        pt.save("dssp", {ss: 0.0 for ss in "HBEGITSC"})

    # Placeholder for rama and kappa_alpha (need structural codes from all proteins)
    for name in ["ramachandran", "kappa_alpha"]:
        pt.save(name, {chr(65+i): 0.0 for i in range(23)})

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
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration for permutation tests")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Process proteins in batches")
    args = parser.parse_args()

    supp_dir = args.data_dir / "supplementary"
    pdb_dir = args.data_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Step 1: Parse Dataset S3
    # =========================================================
    print("=" * 60)
    print("STEP 1: Parsing Dataset S3")
    print("=" * 60)
    proteins_data = parse_dataset_s3(supp_dir)
    total_proteins = len(proteins_data)
    total_sites = sum(len(v["sites"]) for v in proteins_data.values())
    print(f"Found {total_proteins} proteins with {total_sites} CP sites")

    if args.max_proteins:
        pdb_ids = list(proteins_data.keys())[:args.max_proteins]
        proteins_data = {k: proteins_data[k] for k in pdb_ids}
        print(f"Limited to {len(proteins_data)} proteins")

    # =========================================================
    # Step 2: Download PDB files
    # =========================================================
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 2: Downloading PDB structures")
        print("=" * 60)
        pdb_ids = list(proteins_data.keys())
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
        print(f"  Downloaded: {downloaded}, Failed: {failed}")

    # Also download DHFR for verification
    print("  Downloading DHFR (1RX4)...")
    download_pdb("1rx4", pdb_dir)

    # =========================================================
    # Step 3: Build propensity tables
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 3: Building propensity tables")
    print("=" * 60)

    # First pass: quick parse to build propensity tables
    use_gpu = not args.no_gpu
    tables = build_propensity_tables_from_data(proteins_data, pdb_dir, args.tables_dir,
                                                use_gpu=use_gpu)

    # =========================================================
    # Step 4: Extract features and build training matrix
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 4: Extracting features")
    print("=" * 60)

    all_X = []
    all_y = []
    processed = 0
    skipped = 0

    pdb_ids = list(proteins_data.keys())
    for i, pdb_id in enumerate(pdb_ids):
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            skipped += 1
            continue

        info = proteins_data[pdb_id]
        try:
            protein = parse_pdb(pdb_path, chain_id=info["chain"])
        except Exception:
            skipped += 1
            continue

        X = extract_features_for_protein(protein, tables)
        if X is None:
            skipped += 1
            continue

        # Create labels: 1 for CP sites, 0 for all others
        y = np.zeros(protein.n_residues)
        for site in info["sites"]:
            try:
                idx = protein.residue_numbers.index(site)
                y[idx] = 1.0
            except ValueError:
                pass

        all_X.append(X)
        all_y.append(y)
        processed += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(pdb_ids):
            print(f"  Feature extraction: {i+1}/{len(pdb_ids)} "
                  f"(processed: {processed}, skipped: {skipped})", flush=True)

    print(f"  Processed: {processed}, Skipped: {skipped}", flush=True)

    if not all_X:
        print("ERROR: No training data extracted!")
        sys.exit(1)

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"\n  Training matrix: {X_train.shape}")
    print(f"  Positive (CP sites): {n_pos}")
    print(f"  Negative (non-CP): {n_neg}")
    print(f"  Ratio: 1:{n_neg // max(n_pos, 1)}")

    # Cache features for future use
    cache_path = args.data_dir / "features_cache.npz"
    np.savez(cache_path, X=X_train, y=y_train)
    print(f"  Cached features to {cache_path}")

    # =========================================================
    # Step 5: Cross-validation (optional)
    # =========================================================
    if not args.skip_cv:
        print("\n" + "=" * 60)
        print("STEP 5: 10-fold Cross-validation")
        print("=" * 60)
        from cpred.training.evaluate import cross_validate
        from cpred.models.random_forest import CPredRandomForest

        # Quick CV with just RF (fastest model)
        print("Running 10-fold CV with Random Forest...")
        cv_metrics = cross_validate(
            CPredRandomForest, X_train, y_train, n_folds=10, n_estimators=100)
        print(f"\nRF CV Results:")
        for key, val in cv_metrics.items():
            print(f"  {key}: {val:.4f}")

    # =========================================================
    # Step 6: Train final ensemble
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 6: Training final ensemble model")
    print("=" * 60)

    ensemble = CPredEnsemble(feature_names=FEATURE_NAMES)
    ensemble.fit(X_train, y_train, feature_names=FEATURE_NAMES)

    # Evaluate on training set
    train_probs = ensemble.predict(X_train)
    train_metrics = compute_metrics(y_train, train_probs)
    print("\nTraining set metrics:")
    for key, val in train_metrics.items():
        print(f"  {key}: {val:.4f}")

    # Save models
    ensemble.save(args.output_dir)
    print(f"\nModels saved to {args.output_dir}")

    # =========================================================
    # Step 7: Verification on DHFR (1RX4)
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 7: Verification on DHFR (1RX4)")
    print("=" * 60)

    dhfr_path = pdb_dir / "1rx4.pdb"
    if dhfr_path.exists():
        try:
            dhfr = parse_pdb(dhfr_path, chain_id="A")
            X_dhfr = extract_features_for_protein(dhfr, tables)
            if X_dhfr is not None:
                probs = ensemble.predict(X_dhfr)
                viable = probs >= 0.5
                high_conf = probs >= 0.85

                print(f"  Sequence length: {dhfr.n_residues}")
                print(f"  Viable (>=0.5): {viable.sum()} residues")
                print(f"  High confidence (>=0.85): {high_conf.sum()} residues")
                print(f"  Mean probability: {probs.mean():.4f}")
                print(f"  Max probability: {probs.max():.4f}")

                # Show top 10 predicted sites
                top_indices = np.argsort(probs)[-10:][::-1]
                print("\n  Top 10 predicted CP sites:")
                print(f"  {'Res#':>5} {'AA':>3} {'Prob':>6}")
                for idx in top_indices:
                    print(f"  {dhfr.residue_numbers[idx]:>5} "
                          f"{dhfr.sequence[idx]:>3} "
                          f"{probs[idx]:>6.4f}")
        except Exception as e:
            print(f"  DHFR verification failed: {e}")
    else:
        print("  DHFR PDB not found, skipping verification")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
