"""Output formatters for CPred prediction results."""

from __future__ import annotations

import csv
import json
import sys
from io import StringIO
from pathlib import Path


def format_results_tsv(results: dict, threshold: float = 0.5) -> str:
    """Format prediction results as TSV.

    Columns: residue_number, amino_acid, probability_score, predicted_viable
    """
    output = StringIO()
    writer = csv.writer(output, delimiter="\t")
    writer.writerow(["residue_number", "amino_acid", "probability_score",
                     "predicted_viable"])

    for i in range(len(results["sequence"])):
        writer.writerow([
            results["residue_numbers"][i],
            results["sequence"][i],
            f"{results['probabilities'][i]:.4f}",
            "YES" if results["viable"][i] else "NO",
        ])

    return output.getvalue()


def format_results_csv(results: dict, threshold: float = 0.5) -> str:
    """Format prediction results as CSV."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["residue_number", "amino_acid", "probability_score",
                     "predicted_viable"])

    for i in range(len(results["sequence"])):
        writer.writerow([
            results["residue_numbers"][i],
            results["sequence"][i],
            f"{results['probabilities'][i]:.4f}",
            "YES" if results["viable"][i] else "NO",
        ])

    return output.getvalue()


def format_results_json(results: dict) -> str:
    """Format prediction results as JSON."""
    output = {
        "pdb_id": results["pdb_id"],
        "chain_id": results["chain_id"],
        "threshold": 0.5,
        "residues": [],
    }

    for i in range(len(results["sequence"])):
        output["residues"].append({
            "residue_number": int(results["residue_numbers"][i]),
            "amino_acid": results["sequence"][i],
            "probability": round(float(results["probabilities"][i]), 4),
            "viable": bool(results["viable"][i]),
        })

    return json.dumps(output, indent=2)


def write_results(results: dict, output_path: str | Path | None = None,
                  fmt: str = "tsv", threshold: float = 0.5) -> None:
    """Write prediction results to file or stdout.

    Args:
        results: Prediction results dictionary.
        output_path: Output file path (None for stdout).
        fmt: Output format ("tsv", "csv", "json").
        threshold: Viability threshold.
    """
    if fmt == "json":
        text = format_results_json(results)
    elif fmt == "csv":
        text = format_results_csv(results, threshold)
    else:
        text = format_results_tsv(results, threshold)

    if output_path is None:
        sys.stdout.write(text)
    else:
        Path(output_path).write_text(text)
