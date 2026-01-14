"""
xTB GFN2 force field implementation for pose ranking.
"""

import os
import subprocess
import time
import logging
from typing import Tuple
from pathlib import Path

from .base import BaseForceField
from .helpers import (
    form_complex,
    chop_pdb,
    get_protein_atom_indices,
    run_xtb_opt,
    run_xtb_spe,
    split_pdb,
    clean_up_tmp_xtb_files
)
from shared.models.ranking_result import RankingResult

logger = logging.getLogger(__name__)


class XTBForceField(BaseForceField):
    """
    xTB GFN2 force field for pose ranking (CPU only).

    Workflow:
    1. Form complex from protein PDB and ligand SDF
    2. Chop to 5Å region around ligand
    3. Optimize complex with GFNFF (protein constrained)
    4. Calculate energies with GFN2:
       - E(complex)
       - E(protein)
       - E(ligand_bound)
    5. Optimize free ligand with GFN2
    6. Calculate E(ligand_free)
    7. Compute:
       - interaction_energy = (E_complex - E_protein - E_ligand_bound) × 627.5095
       - strain_energy = (E_ligand_bound - E_ligand_free) × 627.5095

    Note: Conversion factor 627.5095 converts Hartree to kcal/mol
    """

    # Conversion factor: Hartree to kcal/mol
    HARTREE_TO_KCAL = 627.5095

    def __init__(self, device: str = "cpu"):
        """
        Initialize xTB force field.

        Args:
            device: Device to use (always "cpu" for xTB)
        """
        super().__init__(device="cpu")  # xTB is CPU-only
        self.ligand_resname = "UNL"
        self.distance_cutoff = 5.0  # Angstroms

    def check_availability(self) -> Tuple[bool, str]:
        """
        Check if xTB is available.

        Returns:
            Tuple of (is_available, message)
        """
        try:
            result = subprocess.run(
                ["xtb", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0] if result.stdout else "unknown version"
                return True, f"xTB available: {version_line}"
            else:
                return False, "xTB binary found but returned error"
        except FileNotFoundError:
            return False, "xTB binary not found in PATH"
        except Exception as e:
            return False, f"Error checking xTB availability: {e}"

    def rank_pose(self, protein_pdb: str, ligand_sdf: str,
                  work_dir: str, params: dict) -> RankingResult:
        """
        Rank a single pose using xTB GFN2.

        Args:
            protein_pdb: Path to protein PDB file
            ligand_sdf: Path to ligand SDF file
            work_dir: Working directory for temporary files
            params: Additional parameters (distance_cutoff, etc.)

        Returns:
            RankingResult object with energies

        Raises:
            Exception: If ranking fails
        """
        start_time = time.time()
        pose_id = params.get('pose_id', 'unknown')

        # Check availability
        available, message = self.check_availability()
        if not available:
            return RankingResult.from_error(
                pose_id=pose_id,
                error_message=f"xTB not available: {message}"
            )

        # Override distance cutoff if provided
        distance_cutoff = params.get('distance_cutoff', self.distance_cutoff)

        try:
            # Ensure we're in the working directory
            original_dir = os.getcwd()
            os.chdir(work_dir)

            logger.info(f"[{pose_id}] Starting xTB ranking in {work_dir}")

            # Step 1: Form complex
            complex_pdb = os.path.join(work_dir, "complex.pdb")
            form_complex(protein_pdb, ligand_sdf, complex_pdb, self.ligand_resname)
            logger.info(f"[{pose_id}] Complex formed: {complex_pdb}")

            # Step 2: Chop complex to 5Å region
            chop_pdb(
                complex_pdb,
                ligand_resname=self.ligand_resname,
                distance_cutoff=distance_cutoff,
                minimize_chain_breaks=False
            )
            complex_chopped = complex_pdb.replace(".pdb", "_chopped.pdb")
            logger.info(f"[{pose_id}] Complex chopped: {complex_chopped}")

            # Step 3: Get protein atom indices for constraints
            prot_indices = get_protein_atom_indices(
                complex_chopped,
                ligand_resname=self.ligand_resname,
                include_H=False
            )
            logger.info(f"[{pose_id}] Protein atom indices: {len(prot_indices)} atoms")

            # Step 4: Optimize complex with GFNFF (protein constrained)
            logger.info(f"[{pose_id}] Optimizing complex with GFNFF...")
            _ = run_xtb_opt(
                complex_chopped,
                prot_indices=prot_indices,
                method="gfnff"
            )

            # Move optimized structure
            complex_opt = os.path.join(work_dir, "complex_opt.pdb")
            if os.path.exists("xtbopt.pdb"):
                subprocess.run(f"mv xtbopt.pdb {complex_opt}", shell=True, check=True)
            else:
                raise FileNotFoundError("xtbopt.pdb not found after optimization")

            clean_up_tmp_xtb_files()
            logger.info(f"[{pose_id}] Complex optimized: {complex_opt}")

            # Step 5: Calculate E(complex) with GFN2
            logger.info(f"[{pose_id}] Calculating complex energy with GFN2...")
            complex_energy = run_xtb_spe(complex_opt, method="gfn 2")
            if complex_energy is None:
                raise ValueError("Failed to calculate complex energy")
            clean_up_tmp_xtb_files()

            # Step 6: Split complex into protein and ligand
            split_pdb(complex_opt, ligand_resname=self.ligand_resname, run_dir=work_dir)
            prot_split = os.path.join(work_dir, "prot_split.pdb")
            lig_split = os.path.join(work_dir, "lig_split.pdb")
            logger.info(f"[{pose_id}] Complex split into protein and ligand")

            # Step 7: Calculate E(protein) with GFN2
            logger.info(f"[{pose_id}] Calculating protein energy with GFN2...")
            prot_energy = run_xtb_spe(prot_split, method="gfn 2")
            if prot_energy is None:
                raise ValueError("Failed to calculate protein energy")
            clean_up_tmp_xtb_files()

            # Step 8: Calculate E(ligand_bound) with GFN2
            logger.info(f"[{pose_id}] Calculating ligand bound energy with GFN2...")
            lig_bound_energy = run_xtb_spe(lig_split, method="gfn 2")
            if lig_bound_energy is None:
                raise ValueError("Failed to calculate ligand bound energy")
            clean_up_tmp_xtb_files()

            # Step 9: Optimize free ligand with GFN2
            logger.info(f"[{pose_id}] Optimizing free ligand with GFN2...")
            lig_free_energy = run_xtb_opt(lig_split, H_only=False, method="gfn 2")
            if lig_free_energy is None:
                raise ValueError("Failed to optimize free ligand")

            # Save optimized ligand
            lig_opt = os.path.join(work_dir, "lig_opt.pdb")
            if os.path.exists("xtbopt.pdb"):
                subprocess.run(f"mv xtbopt.pdb {lig_opt}", shell=True, check=True)

            clean_up_tmp_xtb_files()

            # Step 10: Calculate energies in kcal/mol
            strain_energy = (lig_bound_energy - lig_free_energy) * self.HARTREE_TO_KCAL
            interaction_energy = (complex_energy - prot_energy - lig_bound_energy) * self.HARTREE_TO_KCAL

            logger.info(
                f"[{pose_id}] Raw energies (Hartree): "
                f"complex={complex_energy:.6f}, protein={prot_energy:.6f}, "
                f"ligand_bound={lig_bound_energy:.6f}, ligand_free={lig_free_energy:.6f}"
            )
            logger.info(
                f"[{pose_id}] Final energies (kcal/mol): "
                f"strain={strain_energy:.2f}, interaction={interaction_energy:.2f}"
            )

            # Return to original directory
            os.chdir(original_dir)

            # Calculate computation time
            computation_time = time.time() - start_time

            # Create result
            result = RankingResult.from_success(
                pose_id=pose_id,
                interaction_energy=interaction_energy,
                strain_energy=strain_energy,
                energy_method="gfn2",
                force_field_device="cpu",
                complex_energy=complex_energy * self.HARTREE_TO_KCAL,
                protein_energy=prot_energy * self.HARTREE_TO_KCAL,
                ligand_bound_energy=lig_bound_energy * self.HARTREE_TO_KCAL,
                ligand_free_energy=lig_free_energy * self.HARTREE_TO_KCAL,
                metadata=params.get('metadata', {})
            )

            # Set file paths (relative to work_dir)
            result.optimized_complex_pdb = complex_opt
            result.split_protein_pdb = prot_split
            result.split_ligand_pdb = lig_split
            result.optimized_ligand_pdb = lig_opt
            result.computation_time_seconds = computation_time

            logger.info(f"[{pose_id}] xTB ranking completed in {computation_time:.2f}s")

            return result

        except Exception as e:
            # Return to original directory
            os.chdir(original_dir)

            logger.error(f"[{pose_id}] xTB ranking failed: {e}", exc_info=True)
            return RankingResult.from_error(
                pose_id=pose_id,
                error_message=f"xTB ranking failed: {str(e)}",
                metadata=params.get('metadata', {})
            )
        finally:
            # Ensure we're back in original directory and cleanup
            try:
                os.chdir(original_dir)
                os.chdir(work_dir)
                clean_up_tmp_xtb_files()
                os.chdir(original_dir)
            except Exception:
                pass
