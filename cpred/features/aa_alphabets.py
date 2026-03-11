"""Reduced amino acid alphabet mappings for Cat A propensity features.

aac3 (3 classes) — IMGT hydropathy classification (Pommie et al. 2004, ref [46]):
  H (Hydrophobic): A, C, F, I, L, M, V, W
  N (Neutral):     G, H, P, S, T, Y
  P (Hydrophilic): D, E, K, N, Q, R

aac5 (5 classes) — Lehninger / Nelson & Cox (ref [48]):
  A (Nonpolar aliphatic):    A, G, I, L, M, P, V
  R (Aromatic):              F, W, Y
  U (Polar uncharged):       C, N, Q, S, T
  K (Positively charged):    H, K, R
  D (Negatively charged):    D, E
"""

from __future__ import annotations

# IMGT 3-class hydropathy (Pommie et al. 2004, Table 1)
# Cross-checked against 3-class.pdf
AAC3_MAP: dict[str, str] = {
    "A": "H", "C": "H", "F": "H", "I": "H", "L": "H",
    "M": "H", "V": "H", "W": "H",
    "G": "N", "H": "N", "P": "N", "S": "N", "T": "N", "Y": "N",
    "D": "P", "E": "P", "K": "P", "N": "P", "Q": "P", "R": "P",
}

# Lehninger 5-class (ref [48])
AAC5_MAP: dict[str, str] = {
    "A": "A", "G": "A", "I": "A", "L": "A", "M": "A", "P": "A", "V": "A",
    "F": "R", "W": "R", "Y": "R",
    "C": "U", "N": "U", "Q": "U", "S": "U", "T": "U",
    "H": "K", "K": "K", "R": "K",
    "D": "D", "E": "D",
}


def convert_sequence_aac3(seq: str) -> str:
    """Convert amino acid sequence to 3-class reduced alphabet."""
    return "".join(AAC3_MAP.get(aa, "N") for aa in seq)


def convert_sequence_aac5(seq: str) -> str:
    """Convert amino acid sequence to 5-class reduced alphabet."""
    return "".join(AAC5_MAP.get(aa, "A") for aa in seq)


def convert_list_aac3(aas: list[str]) -> list[str]:
    """Convert list of amino acids to 3-class reduced alphabet codes."""
    return [AAC3_MAP.get(aa, "N") for aa in aas]


def convert_list_aac5(aas: list[str]) -> list[str]:
    """Convert list of amino acids to 5-class reduced alphabet codes."""
    return [AAC5_MAP.get(aa, "A") for aa in aas]
