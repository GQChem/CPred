"""Command-line interface for CPred."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cpred.io.output import write_results
from cpred.models.ensemble import CPredEnsemble
from cpred.pipeline import predict_from_pdb, FEATURE_NAMES
from cpred.propensity.tables import PropensityTables


def cmd_predict(args: argparse.Namespace) -> None:
    """Run CP site prediction on a PDB file."""
    pdb_path = Path(args.pdb)
    if not pdb_path.exists():
        print(f"Error: PDB file not found: {pdb_path}", file=sys.stderr)
        sys.exit(1)

    rmsf_path = Path(args.rmsf)
    if not rmsf_path.exists():
        print(f"Error: RMSF file not found: {rmsf_path}", file=sys.stderr)
        sys.exit(1)

    # Load propensity tables
    tables = PropensityTables(args.tables_dir)
    tables.load()
    if not tables.is_loaded:
        print("Warning: No propensity tables found. Using defaults.",
              file=sys.stderr)

    # Load trained model if available
    model = None
    if args.model_dir and Path(args.model_dir).exists():
        model = CPredEnsemble(feature_names=FEATURE_NAMES)
        model.load(args.model_dir)
        print(f"Loaded model from {args.model_dir}", file=sys.stderr)

    # Run prediction
    results = predict_from_pdb(
        pdb_path,
        chain_id=args.chain,
        tables=tables,
        model=model,
        threshold=args.threshold,
        rmsf_file=rmsf_path,
    )

    # Output format
    fmt = "tsv"
    if args.output:
        suffix = Path(args.output).suffix.lower()
        if suffix == ".csv":
            fmt = "csv"
        elif suffix == ".json":
            fmt = "json"
    if args.format:
        fmt = args.format

    write_results(results, args.output, fmt=fmt, threshold=args.threshold)

    # Summary
    n_viable = int(results["viable"].sum())
    n_total = len(results["sequence"])
    print(f"\nPredicted {n_viable}/{n_total} viable CP sites "
          f"(threshold={args.threshold})", file=sys.stderr)


def cmd_train(args: argparse.Namespace) -> None:
    """Train CPred models from data."""
    from cpred.training.data_loader import load_structure_dataset
    from cpred.training.train import train_ensemble

    data_dir = Path(args.data_dir)
    pdb_dir = data_dir / "pdb"
    labels_file = data_dir / "labels.csv" if (data_dir / "labels.csv").exists() else None

    tables = PropensityTables(args.tables_dir)
    tables.load()

    print(f"Loading training data from {pdb_dir}...")
    X, y, ids = load_structure_dataset(pdb_dir, labels_file, tables)

    if len(X) == 0:
        print("Error: No training data found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(X)} samples from {len(set(ids))} proteins")
    output_dir = Path(args.output_dir)
    train_ensemble(X, y, feature_names=FEATURE_NAMES, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cpred",
        description="CPred: Circular Permutation site predictor",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Predict subcommand
    pred_parser = subparsers.add_parser("predict",
                                         help="Predict CP sites from PDB")
    pred_parser.add_argument("pdb", help="Path to PDB file")
    pred_parser.add_argument("rmsf", help="Path to RMSF CSV file from CABSflex")
    pred_parser.add_argument("--chain", "-c", default="A",
                             help="Chain ID (default: A)")
    pred_parser.add_argument("--threshold", "-t", type=float, default=0.5,
                             help="Probability threshold (default: 0.5)")
    pred_parser.add_argument("--output", "-o", default=None,
                             help="Output file (default: stdout)")
    pred_parser.add_argument("--format", "-f", choices=["tsv", "csv", "json"],
                             default=None, help="Output format")
    pred_parser.add_argument("--model-dir", default=None,
                             help="Directory with trained models")
    pred_parser.add_argument("--tables-dir", type=Path, default=None,
                             help="Propensity tables directory")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train CPred models")
    train_parser.add_argument("--data-dir", required=True,
                              help="Directory with training data")
    train_parser.add_argument("--output-dir", default="models",
                              help="Output directory for trained models")
    train_parser.add_argument("--tables-dir", type=Path, default=None,
                              help="Propensity tables directory")

    args = parser.parse_args()

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
