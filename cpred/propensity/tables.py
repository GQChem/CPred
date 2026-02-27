"""Pre-computed propensity lookup tables.

Tables are stored as JSON files in cpred/data/propensity_tables/.
This module handles loading and querying them.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "propensity_tables"

# Table names correspond to different element types
TABLE_NAMES = [
    "single_aa",       # Single amino acid propensity
    "di_residue",      # Di-residue (pair) propensity
    "oligo_residue",   # Oligo-residue (triplet) propensity
    "dssp",            # DSSP secondary structure propensity
    "ramachandran",    # Ramachandran code propensity
    "kappa_alpha",     # Kappa-alpha code propensity
]


class PropensityTables:
    """Container for all propensity lookup tables."""

    def __init__(self, tables_dir: Path | None = None):
        self.tables_dir = tables_dir or DATA_DIR
        self._tables: dict[str, dict[str, float]] = {}

    def load(self) -> None:
        """Load all propensity tables from JSON files."""
        for name in TABLE_NAMES:
            path = self.tables_dir / f"{name}.json"
            if path.exists():
                with open(path) as f:
                    self._tables[name] = json.load(f)
            else:
                self._tables[name] = {}

    def get(self, table_name: str, key: str, default: float = 0.0) -> float:
        """Look up a propensity value.

        Args:
            table_name: Which table to query (e.g., "single_aa").
            key: Element key (e.g., "A" for alanine).
            default: Default value if key not found.

        Returns:
            Propensity score.
        """
        return self._tables.get(table_name, {}).get(key, default)

    def save(self, table_name: str, table: dict[str, float]) -> None:
        """Save a propensity table to JSON."""
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        path = self.tables_dir / f"{table_name}.json"
        with open(path, "w") as f:
            json.dump(table, f, indent=2, sort_keys=True)
        self._tables[table_name] = table

    @property
    def is_loaded(self) -> bool:
        return len(self._tables) > 0
