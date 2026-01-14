"""
Helper functions for force field calculations.
Ported from terray_pose_engine/tpose/rank_poses.py
"""

import os
import subprocess
import logging
from typing import List, Optional
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO
from rdkit import Chem

logger = logging.getLogger(__name__)


def parse_energy_from_xtb_output(output: str) -> Optional[float]:
    """
    Extract energy value from xTB output.

    Args:
        output: xTB stdout

    Returns:
        Energy value in Hartree or None if parsing fails
    """
    try:
        energy = float([i for i in output.split("\n") if "TOTAL ENERGY" in i][0][36:-7])
        return energy
    except Exception as e:
        logger.error(f"Failed to parse energy from xTB output: {e}")
        return None


def run_xtb_spe(input_pdb: str, method: str = "gfnff") -> Optional[float]:
    """
    Run xTB single point energy calculation.

    Args:
        input_pdb: Path to input PDB file
        method: xTB method (gfnff, gfn 2, etc.)

    Returns:
        Energy in Hartree or None if failed
    """
    try:
        # Get directory of input file to run xTB there
        input_dir = os.path.dirname(os.path.abspath(input_pdb))
        input_name = os.path.basename(input_pdb)

        xtb_command = f"xtb --{method} {input_name} --alpb water"
        result = subprocess.run(
            xtb_command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=input_dir
        )

        # Check for errors
        if result.returncode != 0:
            logger.error(f"xTB SPE failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr[:500]}")
            logger.error(f"stdout (last 500 chars): {result.stdout[-500:]}")
            return None

        energy = parse_energy_from_xtb_output(result.stdout)
        return energy
    except Exception as e:
        logger.error(f"Failed to run xTB SPE: {e}")
        return None


def run_xtb_opt(input_pdb: str, H_only: bool = False,
                prot_indices: Optional[List[int]] = None,
                method: str = "gfnff") -> Optional[float]:
    """
    Run xTB geometry optimization with optional constraints.

    Args:
        input_pdb: Path to input PDB file
        H_only: Constrain all heavy atoms (C, N, O, F, P, S, Cl, I)
        prot_indices: List of atom indices to constrain (1-indexed)
        method: xTB method

    Returns:
        Energy in Hartree or None if failed
    """
    try:
        if H_only and prot_indices:
            raise ValueError(
                "Only one of H_only or prot_indices can be specified, not both."
            )

        # Get directory of input file to run xTB there
        input_dir = os.path.dirname(os.path.abspath(input_pdb))
        input_name = os.path.basename(input_pdb)

        if H_only:
            inp_file = """$constrain\n   elements: 6, 7, 8, 9, 15, 16, 17, 53\n   force constant 2\n$end"""
            with open(os.path.join(input_dir, "xtb.inp"), "w") as f:
                f.write(inp_file)
            xtb_command = (
                f"xtb --{method} {input_name} --opt --input xtb.inp --alpb water"
            )

        elif prot_indices:
            prot_indices_list_str = ",".join([str(i) for i in prot_indices])
            inp_file = f"""$constrain\n   atoms: {prot_indices_list_str}\n   force constant 2\n$end"""
            with open(os.path.join(input_dir, "xtb.inp"), "w") as f:
                f.write(inp_file)
            xtb_command = (
                f"xtb --{method} {input_name} --opt --input xtb.inp --alpb water"
            )

        else:
            xtb_command = f"xtb --{method} {input_name} --opt --alpb water"

        # Run xTB in the input directory so output files are created there
        result = subprocess.run(
            xtb_command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=input_dir
        )

        # Check for errors
        if result.returncode != 0:
            logger.error(f"xTB optimization failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr[:500]}")
            logger.error(f"stdout (last 500 chars): {result.stdout[-500:]}")
            return None

        energy = parse_energy_from_xtb_output(result.stdout)
        return energy
    except Exception as e:
        logger.error(f"Failed to run xTB optimization: {e}")
        return None


def clean_up_tmp_xtb_files():
    """Clean up temporary xTB files from current directory."""
    for tmp_file in [
        "wbo",
        "charges",
        "xtbopt.log",
        "xtbopt.sdf",
        "xtbout.log",
        "xtb.out",
        "xtbrestart",
        "xtbtopo.mol",
        "xtbtopo.sdf",
        "xtbout.json",
        ".xtboptok",
        "xtblast.pdb",
        "xtbopt.pdb",
        "gfnff_adjacency",
        "gfnff_charges",
        "gfnff_topo",
        "gfnff_lists.json",
        "xtb.inp",
    ]:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def run_so3lr_energy(structure_file: str, lr_cutoff: float = 12.0,
                     charge: float = 0.0) -> Optional[float]:
    """
    Calculate energy using SO3LR.

    Args:
        structure_file: Path to PDB or XYZ file
        lr_cutoff: Long-range interaction cutoff (Å)
                   12.0 for chopped systems
                   20.0 for full protein systems
        charge: Total molecular charge

    Returns:
        energy: Total potential energy in eV or None if failed

    Note:
        - Only relative energies are meaningful
        - Convert to kcal/mol: energy × 23.0609
    """
    try:
        from so3lr import So3lrCalculator
        from ase.io import read
        import numpy as np

        atoms = read(structure_file)
        atoms.info['charge'] = charge

        calc = So3lrCalculator(
            calculate_stress=False,
            lr_cutoff=lr_cutoff,
            dtype=np.float64
        )
        atoms.calc = calc

        energy = atoms.get_potential_energy()  # eV
        return energy
    except ImportError as e:
        logger.error("SO3LR not installed. Install with: pip install so3lr")
        return None
    except Exception as e:
        logger.error(f"Failed to calculate SO3LR energy: {e}")
        return None


def run_so3lr_optimize(structure_file: str, constrained_indices: Optional[List[int]] = None,
                       lr_cutoff: float = 12.0, charge: float = 0.0,
                       fmax: float = 0.05) -> Optional[float]:
    """
    Optimize geometry using SO3LR with optional constraints.

    Args:
        structure_file: Path to PDB or XYZ file
        constrained_indices: List of atom indices to fix (0-indexed)
                            Typically protein heavy atoms
        lr_cutoff: Long-range interaction cutoff (Å)
        charge: Total molecular charge
        fmax: Force convergence criterion (eV/Å)
              0.05 is recommended (paper default)

    Returns:
        energy: Optimized energy in eV or None if failed

    Side effects:
        Updates structure_file with optimized geometry
    """
    try:
        from so3lr import So3lrCalculator
        from ase.io import read, write
        from ase.optimize import FIRE
        from ase.constraints import FixAtoms
        import numpy as np

        atoms = read(structure_file)
        atoms.info['charge'] = charge

        # Apply constraints if specified
        if constrained_indices is not None and len(constrained_indices) > 0:
            constraint = FixAtoms(indices=constrained_indices)
            atoms.set_constraint(constraint)

        calc = So3lrCalculator(
            calculate_stress=False,
            lr_cutoff=lr_cutoff,
            dtype=np.float64
        )
        atoms.calc = calc

        # Optimize using FIRE algorithm
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=fmax)

        # Save optimized structure
        write(structure_file, atoms)

        energy = atoms.get_potential_energy()
        return energy
    except ImportError as e:
        logger.error("SO3LR or ASE not installed. Install with: pip install so3lr ase")
        return None
    except Exception as e:
        logger.error(f"Failed to optimize with SO3LR: {e}")
        return None


def chop_pdb(input_pdb: str, ligand_resname: str = "UNL",
             distance_cutoff: float = 6.0,
             minimize_chain_breaks: bool = False) -> None:
    """
    Find residues that maximize proximity to a ligand and save chopped PDB.

    Args:
        input_pdb: Path to the input PDB file
        ligand_resname: Residue name of the ligand (default is 'UNL')
        distance_cutoff: Maximum distance (in Å) to consider a residue
        minimize_chain_breaks: Whether to fill gaps in chains

    Side effects:
        Creates {stem}_chopped.pdb in same directory as input_pdb
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("complex", input_pdb)

    ligand_residues = []
    protein_residues = []

    # Separate ligand and protein residues
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == ligand_resname:
                    ligand_residues.append((chain, residue))
                elif residue.get_resname() not in ["HOH", "Na", "Cl", "Zn"]:
                    protein_residues.append((chain, residue))

    # Determine contact residues
    contact_residues = set()
    for ligand_chain, ligand_residue in ligand_residues:
        for ligand_atom in ligand_residue:
            for protein_chain, protein_residue in protein_residues:
                for protein_atom in protein_residue:
                    distance = ligand_atom - protein_atom
                    if distance < distance_cutoff:
                        contact_residues.add((protein_chain, protein_residue))

    # Expand contact residues to maintain chain integrity
    if minimize_chain_breaks:
        full_contact_residues = set()
        for chain, residue in protein_residues:
            if (chain, residue) in contact_residues:
                prev_residue_id = None
                for res in chain.get_residues():
                    current_residue_id = res.id[1]
                    if (chain, res) in contact_residues or (
                        chain,
                        res,
                    ) in full_contact_residues:
                        full_contact_residues.add((chain, res))
                        prev_residue_id = current_residue_id
                    elif (
                        prev_residue_id is not None
                        and current_residue_id == prev_residue_id + 1
                    ):
                        full_contact_residues.add((chain, res))
                        prev_residue_id = current_residue_id
            full_contact_residues.update(contact_residues)
    else:
        full_contact_residues = contact_residues

    # Define a class for filtering residues
    class ResidueSelector(PDB.Select):
        def __init__(self, keep_residues):
            self.keep_residues = set(keep_residues)

        def accept_residue(self, residue):
            return (residue.parent, residue) in self.keep_residues

    io = PDB.PDBIO()
    stem = input_pdb.rsplit(".", 1)[0]
    # Save the entire complex including contact and ligand residues
    io.set_structure(structure)
    io.save(
        f"{stem}_chopped.pdb",
        ResidueSelector(full_contact_residues | set(ligand_residues)),
    )


def get_protein_atom_indices(input_pdb: str, ligand_resname: str = "UNL",
                             include_H: bool = False) -> List[int]:
    """
    Get list of protein atom indices (1-indexed for xTB, 0-indexed for SO3LR).

    Args:
        input_pdb: Path to PDB file
        ligand_resname: Ligand residue name to exclude
        include_H: Whether to include hydrogen atoms

    Returns:
        List of atom serial numbers (1-indexed as in PDB file)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("complex", input_pdb)

    protein_atom_indices = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() not in [
                    ligand_resname,
                    "HOH",
                    "Na",
                    "Cl",
                    "Zn",
                ]:
                    for atom in residue:
                        if include_H:
                            protein_atom_indices.append(atom.get_serial_number())
                        elif not atom.get_id().startswith("H"):
                            protein_atom_indices.append(atom.get_serial_number())

    return protein_atom_indices


def split_pdb(input_pdb: str, ligand_resname: str = "UNL",
              run_dir: str = ".") -> List[int]:
    """
    Separate PDB file into distinct files for protein and ligand.

    Args:
        input_pdb: Path to the input PDB file
        ligand_resname: Residue name of the ligand (default is 'UNL')
        run_dir: Directory to save the split PDB files

    Returns:
        List of protein atom serial numbers (1-indexed)

    Side effects:
        Creates {run_dir}/prot_split.pdb and {run_dir}/lig_split.pdb
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("complex", input_pdb)

    ligand_residues = []
    protein_residues = []
    protein_atom_indices = []

    # Separate ligand and protein residues
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == ligand_resname:
                    ligand_residues.append(residue)
                elif residue.get_resname() not in ["HOH", "Na", "Cl", "Zn"]:
                    protein_residues.append(residue)
                    for atom in residue:
                        if not atom.get_id().startswith("H"):
                            protein_atom_indices.append(atom.get_serial_number())

    # Define a class for filtering residues
    class ResidueSelector(PDB.Select):
        def __init__(self, keep_residues):
            self.keep_residues = set(keep_residues)

        def accept_residue(self, residue):
            return residue in self.keep_residues

    io = PDB.PDBIO()

    # Save the split structures to a file
    io.set_structure(structure)
    io.save(f"{run_dir}/prot_split.pdb", ResidueSelector(protein_residues))
    io.save(f"{run_dir}/lig_split.pdb", ResidueSelector(ligand_residues))

    return protein_atom_indices


def form_complex(protein_pdb: str, ligand_sdf: str, output_pdb: str,
                 ligand_resname: str = "UNL") -> None:
    """
    Combine protein PDB and ligand SDF into a single complex PDB file.

    Args:
        protein_pdb: Path to protein PDB file
        ligand_sdf: Path to ligand SDF file
        output_pdb: Path for output complex PDB file
        ligand_resname: Residue name for ligand in output PDB

    Side effects:
        Creates output_pdb file
    """
    parser = PDBParser()
    io = PDBIO()

    # Load protein structure
    structure_protein = parser.get_structure("protein", protein_pdb)

    # Convert SDF to PDB
    mol = Chem.SDMolSupplier(ligand_sdf)[0]
    if mol is None:
        # Try reading as mol block
        with open(ligand_sdf, 'r') as f:
            sdf_str = f.read()
        mol = Chem.MolFromMolBlock(sdf_str)

    mol = Chem.AddHs(mol, addCoords=True)

    # Write ligand to temporary PDB
    ligand_pdb_tmp = output_pdb.replace(".pdb", "_lig_tmp.pdb")
    Chem.MolToPDBFile(mol, ligand_pdb_tmp)

    # Load ligand structure and add to protein
    structure_ligand = parser.get_structure("ligand", ligand_pdb_tmp)
    for model in structure_ligand:
        for chain in model:
            # Update ligand residue name
            for residue in chain:
                residue.resname = ligand_resname
            structure_protein[0].add(chain)

    # Save complex
    io.set_structure(structure_protein)
    io.save(output_pdb)

    # Clean up temporary file
    if os.path.exists(ligand_pdb_tmp):
        os.remove(ligand_pdb_tmp)
