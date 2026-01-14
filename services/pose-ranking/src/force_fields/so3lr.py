"""
SO3LR ML force field implementation for pose ranking.
"""

import os
import subprocess
import time
import logging
from typing import Tuple, Optional, List
from pathlib import Path

from .base import BaseForceField
from .helpers import (
    form_complex,
    chop_pdb,
    get_protein_atom_indices,
    run_so3lr_energy,
    run_so3lr_optimize,
    split_pdb
)
from shared.models.ranking_result import RankingResult
from shared.utils.gpu_utils import detect_gpu

logger = logging.getLogger(__name__)


class SO3LRForceField(BaseForceField):
    """
    SO3LR ML force field for pose ranking (GPU with CPU fallback).

    Workflow:
    1. Form complex from protein PDB and ligand SDF
    2. Optionally chop to 5Å region around ligand
    3. Optimize complex (protein constrained) or calculate energy
    4. Calculate energies:
       - E(complex)
       - E(protein)
       - E(ligand_bound)
    5. Optionally optimize free ligand
    6. Calculate E(ligand_free)
    7. Compute:
       - interaction_energy = (E_complex - E_protein - E_ligand_bound) × 23.0609
       - strain_energy = (E_ligand_bound - E_ligand_free) × 23.0609

    Note: Conversion factor 23.0609 converts eV to kcal/mol
    """

    # Conversion factor: eV to kcal/mol
    EV_TO_KCAL = 23.0609

    def __init__(self, device: str = "auto"):
        """
        Initialize SO3LR force field.

        Args:
            device: Device to use ("auto", "cpu", "cuda:0", etc.)
                   "auto" will detect GPU and fall back to CPU
        """
        if device == "auto":
            gpu_available, detected_device = detect_gpu()
            device = detected_device
            logger.info(f"SO3LR auto-detected device: {device}")

        super().__init__(device=device)
        self.ligand_resname = "UNL"
        self.distance_cutoff = 5.0
        self.enable_fallback = True  # Enable CPU fallback on GPU errors

    def check_availability(self) -> Tuple[bool, str]:
        """
        Check if SO3LR is available.

        Returns:
            Tuple of (is_available, message)
        """
        try:
            from so3lr import So3lrCalculator
            from ase.io import read
            return True, "SO3LR available"
        except ImportError as e:
            return False, f"SO3LR not installed: {e}"
        except Exception as e:
            return False, f"Error checking SO3LR availability: {e}"

    def _convert_indices_to_ase(self, pdb_indices: List[int]) -> List[int]:
        """
        Convert PDB atom indices (1-indexed) to ASE indices (0-indexed).

        Args:
            pdb_indices: List of 1-indexed atom serial numbers from PDB

        Returns:
            List of 0-indexed atom indices for ASE
        """
        return [idx - 1 for idx in pdb_indices]

    def rank_pose(self, protein_pdb: str, ligand_sdf: str,
                  work_dir: str, params: dict) -> RankingResult:
        """
        Rank a single pose using SO3LR.

        Args:
            protein_pdb: Path to protein PDB file
            ligand_sdf: Path to ligand SDF file
            work_dir: Working directory for temporary files
            params: Additional parameters:
                - use_chopping: Use 5Å chopped system (default: True)
                - optimize_complex: Optimize complex (default: True)
                - optimize_ligand: Optimize ligand for strain (default: True)
                - lr_cutoff: Long-range cutoff in Å (default: 12.0 for chopped, 20.0 for full)
                - distance_cutoff: Chopping distance (default: 5.0)

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
                error_message=f"SO3LR not available: {message}"
            )

        # Extract parameters
        use_chopping = params.get('use_chopping', True)
        optimize_complex = params.get('optimize_complex', True)
        optimize_ligand = params.get('optimize_ligand', True)
        lr_cutoff = params.get('lr_cutoff', 12.0 if use_chopping else 20.0)
        distance_cutoff = params.get('distance_cutoff', self.distance_cutoff)

        try:
            # Ensure we're in the working directory
            original_dir = os.getcwd()
            os.chdir(work_dir)

            logger.info(
                f"[{pose_id}] Starting SO3LR ranking in {work_dir} "
                f"(device={self.device}, chopping={use_chopping}, "
                f"lr_cutoff={lr_cutoff})"
            )

            # Step 1: Form complex
            complex_pdb = os.path.join(work_dir, "complex.pdb")
            form_complex(protein_pdb, ligand_sdf, complex_pdb, self.ligand_resname)
            logger.info(f"[{pose_id}] Complex formed: {complex_pdb}")

            # Step 2: Optionally chop complex
            if use_chopping:
                chop_pdb(
                    complex_pdb,
                    ligand_resname=self.ligand_resname,
                    distance_cutoff=distance_cutoff,
                    minimize_chain_breaks=False
                )
                complex_file = complex_pdb.replace(".pdb", "_chopped.pdb")
                logger.info(f"[{pose_id}] Complex chopped: {complex_file}")
            else:
                complex_file = complex_pdb
                logger.info(f"[{pose_id}] Using full complex (no chopping)")

            # Step 3: Get protein atom indices for constraints
            pdb_indices = get_protein_atom_indices(
                complex_file,
                ligand_resname=self.ligand_resname,
                include_H=False
            )
            # Convert to 0-indexed for ASE
            prot_indices = self._convert_indices_to_ase(pdb_indices)
            logger.info(f"[{pose_id}] Protein constraint indices: {len(prot_indices)} atoms")

            # Step 4: Calculate/optimize complex energy
            if optimize_complex:
                logger.info(f"[{pose_id}] Optimizing complex with SO3LR...")
                complex_energy = run_so3lr_optimize(
                    complex_file,
                    constrained_indices=prot_indices,
                    lr_cutoff=lr_cutoff,
                    charge=0.0,
                    fmax=0.05
                )
            else:
                logger.info(f"[{pose_id}] Calculating complex energy with SO3LR...")
                complex_energy = run_so3lr_energy(
                    complex_file,
                    lr_cutoff=lr_cutoff,
                    charge=0.0
                )

            if complex_energy is None:
                raise ValueError("Failed to calculate complex energy")

            # Step 5: Split complex into protein and ligand
            split_pdb(complex_file, ligand_resname=self.ligand_resname, run_dir=work_dir)
            prot_split = os.path.join(work_dir, "prot_split.pdb")
            lig_split = os.path.join(work_dir, "lig_split.pdb")
            logger.info(f"[{pose_id}] Complex split into protein and ligand")

            # Step 6: Calculate E(protein)
            logger.info(f"[{pose_id}] Calculating protein energy with SO3LR...")
            prot_energy = run_so3lr_energy(prot_split, lr_cutoff=lr_cutoff, charge=0.0)
            if prot_energy is None:
                raise ValueError("Failed to calculate protein energy")

            # Step 7: Calculate E(ligand_bound)
            logger.info(f"[{pose_id}] Calculating ligand bound energy with SO3LR...")
            lig_bound_energy = run_so3lr_energy(lig_split, lr_cutoff=lr_cutoff, charge=0.0)
            if lig_bound_energy is None:
                raise ValueError("Failed to calculate ligand bound energy")

            # Step 8: Optimize free ligand and calculate E(ligand_free)
            if optimize_ligand:
                logger.info(f"[{pose_id}] Optimizing free ligand with SO3LR...")
                lig_opt = os.path.join(work_dir, "lig_opt.pdb")
                subprocess.run(f"cp {lig_split} {lig_opt}", shell=True, check=True)

                lig_free_energy = run_so3lr_optimize(
                    lig_opt,
                    constrained_indices=None,  # No constraints
                    lr_cutoff=lr_cutoff,
                    charge=0.0,
                    fmax=0.05
                )
                if lig_free_energy is None:
                    raise ValueError("Failed to optimize free ligand")
            else:
                lig_free_energy = lig_bound_energy
                lig_opt = None
                logger.info(f"[{pose_id}] Skipping ligand optimization")

            # Step 9: Calculate energies in kcal/mol
            strain_energy = (lig_bound_energy - lig_free_energy) * self.EV_TO_KCAL
            interaction_energy = (complex_energy - prot_energy - lig_bound_energy) * self.EV_TO_KCAL

            logger.info(
                f"[{pose_id}] Raw energies (eV): "
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
                energy_method="so3lr",
                force_field_device=self.device,
                complex_energy=complex_energy * self.EV_TO_KCAL,
                protein_energy=prot_energy * self.EV_TO_KCAL,
                ligand_bound_energy=lig_bound_energy * self.EV_TO_KCAL,
                ligand_free_energy=lig_free_energy * self.EV_TO_KCAL,
                metadata=params.get('metadata', {})
            )

            # Set file paths
            result.optimized_complex_pdb = complex_file if optimize_complex else None
            result.split_protein_pdb = prot_split
            result.split_ligand_pdb = lig_split
            result.optimized_ligand_pdb = lig_opt if optimize_ligand else None
            result.computation_time_seconds = computation_time

            logger.info(f"[{pose_id}] SO3LR ranking completed in {computation_time:.2f}s")

            return result

        except Exception as e:
            # Check if this is a CUDA error and we can fall back to CPU
            if self.enable_fallback and self.device != "cpu" and "CUDA" in str(e).upper():
                logger.warning(
                    f"[{pose_id}] GPU failed with error: {e}. "
                    "Retrying on CPU..."
                )
                self.set_device("cpu")

                # Return to original directory before retrying
                try:
                    os.chdir(original_dir)
                except Exception:
                    pass

                # Retry on CPU
                return self.rank_pose(protein_pdb, ligand_sdf, work_dir, params)

            # Return to original directory
            try:
                os.chdir(original_dir)
            except Exception:
                pass

            logger.error(f"[{pose_id}] SO3LR ranking failed: {e}", exc_info=True)
            return RankingResult.from_error(
                pose_id=pose_id,
                error_message=f"SO3LR ranking failed: {str(e)}",
                metadata=params.get('metadata', {})
            )
