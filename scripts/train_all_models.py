#!/usr/bin/env python3
"""End-to-end training pipeline.

Steps:
  1. Download supplementary data (if needed)
  2. Compute features for all training structures
  3. Build propensity tables
  4. Train ensemble model
  5. Evaluate via 10-fold CV
  6. Save trained models
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from cpred.pipeline import FEATURE_NAMES
from cpred.propensity.tables import PropensityTables
from cpred.training.data_loader import load_structure_dataset
from cpred.training.train import train_ensemble
from cpred.training.evaluate import cross_validate
from cpred.models.ensemble import CPredEnsemble


def main():
    parser = argparse.ArgumentParser(description="Full CPred training pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Data directory (with pdb/ and supplementary/)")
    parser.add_argument("--output-dir", type=Path, default=Path("models"),
                        help="Output directory for trained models")
    parser.add_argument("--tables-dir", type=Path, default=None,
                        help="Propensity tables directory")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download step")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip cross-validation")
    args = parser.parse_args()

    tables_dir = args.tables_dir or (
        Path(__file__).parent.parent / "cpred" / "data" / "propensity_tables")

    # Step 1: Download data
    if not args.skip_download:
        print("=" * 60)
        print("Step 1: Downloading data")
        print("=" * 60)
        try:
            from scripts.download_data import main as download_main
            sys.argv = ["download_data.py", "--data-dir", str(args.data_dir)]
            download_main()
        except Exception as e:
            print(f"Warning: Data download failed: {e}")
            print("Continuing with existing data...")

    # Step 2: Build propensity tables
    print("\n" + "=" * 60)
    print("Step 2: Building propensity tables")
    print("=" * 60)
    try:
        from scripts.build_propensity_tables import main as build_tables_main
        sys.argv = ["build_propensity_tables.py",
                     "--data-dir", str(args.data_dir),
                     "--output-dir", str(tables_dir)]
        build_tables_main()
    except Exception as e:
        print(f"Warning: Table building failed: {e}")
        print("Creating placeholder tables...")
        pt = PropensityTables(tables_dir)
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
        for table_name in ["single_aa", "dssp", "ramachandran", "kappa_alpha"]:
            pt.save(table_name, {aa: 0.0 for aa in standard_aa})
        pt.save("di_residue", {})
        pt.save("oligo_residue", {})

    # Step 3: Load training data
    print("\n" + "=" * 60)
    print("Step 3: Loading training data")
    print("=" * 60)
    tables = PropensityTables(tables_dir)
    tables.load()

    pdb_dir = args.data_dir / "pdb"
    labels_file = args.data_dir / "labels.csv"
    if not labels_file.exists():
        labels_file = None

    X, y, ids = load_structure_dataset(pdb_dir, labels_file, tables)
    if len(X) == 0:
        print("Error: No training data found. Run download_data.py first.")
        sys.exit(1)

    print(f"Loaded {len(X)} samples, {int(y.sum())} positive, "
          f"{len(set(ids))} proteins")

    # Step 4: Cross-validation
    if not args.skip_cv:
        print("\n" + "=" * 60)
        print("Step 4: 10-fold Cross-validation")
        print("=" * 60)
        cv_metrics = cross_validate(
            CPredEnsemble, X, y, n_folds=10,
            feature_names=FEATURE_NAMES,
        )
        print("\nCV Results:")
        for key, val in cv_metrics.items():
            print(f"  {key}: {val:.4f}")

    # Step 5: Train final model
    print("\n" + "=" * 60)
    print("Step 5: Training final model")
    print("=" * 60)
    ensemble = train_ensemble(X, y, feature_names=FEATURE_NAMES,
                              output_dir=args.output_dir)

    print("\nTraining complete!")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
