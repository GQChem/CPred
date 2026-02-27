"""PDB parsing and ProteinStructure data class."""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

# Standard amino acid 3-letter to 1-letter mapping
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

HYDROPHOBIC = set("AVILMFWP")


@dataclass
class DSSPResult:
    """Per-residue DSSP results."""
    ss: list[str]           # secondary structure assignment (H, E, C, etc.)
    rsa: np.ndarray         # relative solvent accessibility
    phi: np.ndarray         # phi angles
    psi: np.ndarray         # psi angles
    nhbond_energy: list[tuple[int, float]]  # H-bond info


@dataclass
class ProteinStructure:
    """Parsed protein structure with extracted coordinates and properties."""
    pdb_id: str
    chain_id: str
    sequence: str
    residue_numbers: list[int]
    ca_coords: np.ndarray       # (N, 3) CA coordinates
    cb_coords: np.ndarray       # (N, 3) CB coordinates (CA for Gly)
    bfactors: np.ndarray        # (N,) CA B-factors
    residues: list[Residue]     # BioPython residue objects
    dssp: DSSPResult | None = None
    heavy_atom_coords: list[np.ndarray] = field(default_factory=list)  # per-residue heavy atoms

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    def is_buried(self, idx: int, threshold: float = 0.10) -> bool:
        """Check if residue is buried (RSA < threshold)."""
        if self.dssp is None:
            return False
        return self.dssp.rsa[idx] < threshold

    def is_hydrophobic(self, idx: int) -> bool:
        """Check if residue is hydrophobic."""
        return self.sequence[idx] in HYDROPHOBIC


def _get_standard_residues(chain: Chain) -> list[Residue]:
    """Extract standard amino acid residues from chain."""
    residues = []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue  # skip HETATMs
        resname = res.get_resname().strip()
        if resname in AA3TO1:
            residues.append(res)
    return residues


def _get_cb_coord(residue: Residue) -> np.ndarray:
    """Get CB coordinate, or CA for glycine."""
    if residue.get_resname().strip() == "GLY":
        return residue["CA"].get_vector().get_array()
    try:
        return residue["CB"].get_vector().get_array()
    except KeyError:
        return residue["CA"].get_vector().get_array()


def _get_heavy_atoms(residue: Residue) -> np.ndarray:
    """Get coordinates of all heavy (non-hydrogen) atoms in a residue."""
    coords = []
    for atom in residue.get_atoms():
        if atom.element != "H":
            coords.append(atom.get_vector().get_array())
    return np.array(coords) if coords else np.empty((0, 3))


def run_dssp(pdb_path: Path, model, chain_id: str,
             residues: list[Residue]) -> DSSPResult | None:
    """Run DSSP on structure and extract per-residue results."""
    try:
        dssp = DSSP(model, str(pdb_path), dssp="mkdssp")
    except Exception:
        try:
            dssp = DSSP(model, str(pdb_path))
        except Exception as e:
            warnings.warn(f"DSSP failed: {e}. Secondary structure features unavailable.")
            return None

    n = len(residues)
    ss = ["C"] * n
    rsa = np.zeros(n)
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    nhbond = [(0, 0.0)] * n

    # Build lookup from (chain, resid) -> index
    res_lookup = {}
    for i, res in enumerate(residues):
        key = (chain_id, res.id)
        res_lookup[key] = i

    for dssp_key in dssp.keys():
        chain, res_id = dssp_key[0], dssp_key[1]
        lookup_key = (chain, (" ", res_id[1], res_id[2]) if len(res_id) == 3
                       else res_id)
        # Try direct match
        idx = res_lookup.get((chain, res_id))
        if idx is None:
            # Try matching by residue number
            for i, res in enumerate(residues):
                if res.id[1] == res_id[1]:
                    idx = i
                    break
        if idx is None:
            continue

        dssp_data = dssp[dssp_key]
        ss[idx] = dssp_data[2] if dssp_data[2] != "-" else "C"
        rsa[idx] = dssp_data[3]
        phi[idx] = dssp_data[4]
        psi[idx] = dssp_data[5]

    return DSSPResult(ss=ss, rsa=rsa, phi=phi, psi=psi, nhbond_energy=nhbond)


def parse_pdb(pdb_path: str | Path, chain_id: str = "A",
              run_dssp_flag: bool = True) -> ProteinStructure:
    """Parse a PDB file and extract structure information.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier (default "A").
        run_dssp_flag: Whether to run DSSP for secondary structure.

    Returns:
        ProteinStructure with coordinates, sequence, and optionally DSSP.
    """
    pdb_path = Path(pdb_path)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = structure[0]

    if chain_id not in [c.id for c in model.get_chains()]:
        available = [c.id for c in model.get_chains()]
        raise ValueError(f"Chain '{chain_id}' not found. Available: {available}")

    chain = model[chain_id]
    residues = _get_standard_residues(chain)

    if not residues:
        raise ValueError(f"No standard amino acid residues in chain {chain_id}")

    sequence = "".join(AA3TO1[r.get_resname().strip()] for r in residues)
    residue_numbers = [r.id[1] for r in residues]
    ca_coords = np.array([r["CA"].get_vector().get_array() for r in residues])
    cb_coords = np.array([_get_cb_coord(r) for r in residues])
    bfactors = np.array([r["CA"].get_bfactor() for r in residues])
    heavy_atoms = [_get_heavy_atoms(r) for r in residues]

    dssp_result = None
    if run_dssp_flag:
        dssp_result = run_dssp(pdb_path, model, chain_id, residues)

    return ProteinStructure(
        pdb_id=pdb_path.stem,
        chain_id=chain_id,
        sequence=sequence,
        residue_numbers=residue_numbers,
        ca_coords=ca_coords,
        cb_coords=cb_coords,
        bfactors=bfactors,
        residues=residues,
        dssp=dssp_result,
        heavy_atom_coords=heavy_atoms,
    )
