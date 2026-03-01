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
    nhbond: np.ndarray      # number of backbone H-bonds per residue (int)


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
        if resname in AA3TO1 and "CA" in res:
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


def _ensure_dssp_env():
    """Ensure LD_LIBRARY_PATH includes conda lib dir for mkdssp's libboost."""
    import os, shutil
    mkdssp = shutil.which("mkdssp")
    if mkdssp is None:
        return
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        return
    conda_lib = os.path.join(conda_prefix, "lib")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_path}" if ld_path else conda_lib


def run_dssp(pdb_path: Path, model, chain_id: str,
             residues: list[Residue]) -> DSSPResult | None:
    """Run DSSP on structure and extract per-residue results."""
    _ensure_dssp_env()
    try:
        dssp = DSSP(model, str(pdb_path), dssp="mkdssp")
    except Exception:
        except Exception as e:
            raise RuntimeError(f"DSSP failed for {pdb_path}: {e}") from e

    n = len(residues)
    ss = ["C"] * n
    rsa = np.zeros(n)
    phi = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    # Track H-bond partners: each residue can donate NH->O and accept O<-HN
    # DSSP provides 4 H-bond columns: NH->O(1), O->NH(1), NH->O(2), O->NH(2)
    # Each has (partner_offset, energy). A bond exists if energy < -0.5 kcal/mol
    hbond_count = np.zeros(n, dtype=np.float64)

    # Build lookup from (chain, resid) -> index
    res_lookup = {}
    for i, res in enumerate(residues):
        key = (chain_id, res.id)
        res_lookup[key] = i

    for dssp_key in dssp.keys():
        chain, res_id = dssp_key[0], dssp_key[1]
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

        # Count H-bonds from DSSP data
        # BioPython DSSP tuple: (dssp_idx, aa, ss, rsa, phi, psi,
        #   NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
        #   NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
        # Energy indices: 7, 9, 11, 13 — bond exists if energy < -0.5 kcal/mol
        n_bonds = 0
        for hb_idx in (7, 9, 11, 13):
            if hb_idx < len(dssp_data):
                energy = dssp_data[hb_idx]
                if isinstance(energy, (int, float)) and energy < -0.5:
                    n_bonds += 1
        hbond_count[idx] = float(n_bonds)

    return DSSPResult(ss=ss, rsa=rsa, phi=phi, psi=psi, nhbond=hbond_count)


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
