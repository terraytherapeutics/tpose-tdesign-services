"""
Ranking result data model for TPose ranking.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class RankingResult:
    """
    Represents the result of ranking a single pose.

    Attributes:
        pose_id: Unique identifier for the pose
        ranking_success: Whether ranking completed successfully
        error_message: Error message if ranking failed
        interaction_energy: Protein-ligand interaction energy (kcal/mol)
        strain_energy: Ligand strain energy (kcal/mol)
        total_score: Total score (interaction + strain)
        complex_energy: Complex energy (kcal/mol or eV)
        protein_energy: Protein energy (kcal/mol or eV)
        ligand_bound_energy: Ligand energy in bound state (kcal/mol or eV)
        ligand_free_energy: Ligand energy in free state (kcal/mol or eV)
        energy_method: Force field method used ("gfn2" or "so3lr")
        force_field_device: Device used ("cpu" or "cuda:0")
        optimized_complex_pdb: S3 path to optimized complex
        split_protein_pdb: S3 path to split protein
        split_ligand_pdb: S3 path to split ligand (bound conformation)
        optimized_ligand_pdb: S3 path to optimized ligand (free conformation)
        computation_time_seconds: Time taken for ranking
        metadata: Original user metadata
    """

    pose_id: str
    ranking_success: bool = False
    error_message: str = ""
    interaction_energy: Optional[float] = None
    strain_energy: Optional[float] = None
    total_score: Optional[float] = None
    complex_energy: Optional[float] = None
    protein_energy: Optional[float] = None
    ligand_bound_energy: Optional[float] = None
    ligand_free_energy: Optional[float] = None
    energy_method: str = "gfn2"
    force_field_device: str = "cpu"
    optimized_complex_pdb: Optional[str] = None
    split_protein_pdb: Optional[str] = None
    split_ligand_pdb: Optional[str] = None
    optimized_ligand_pdb: Optional[str] = None
    computation_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_total_score(self) -> Optional[float]:
        """
        Calculate total score as sum of interaction and strain energies.

        Returns:
            Total score or None if energies not available
        """
        if (self.interaction_energy is not None and
            self.strain_energy is not None):
            self.total_score = self.interaction_energy + self.strain_energy
            return self.total_score
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert RankingResult to dictionary for TDesign output.

        Returns:
            Dictionary representation
        """
        # Calculate total score if possible
        self.calculate_total_score()

        return {
            'pose_id': self.pose_id,
            'ranking_success': self.ranking_success,
            'error_message': self.error_message,
            'interaction_energy': self.interaction_energy,
            'strain_energy': self.strain_energy,
            'total_score': self.total_score,
            'complex_energy': self.complex_energy,
            'protein_energy': self.protein_energy,
            'ligand_bound_energy': self.ligand_bound_energy,
            'ligand_free_energy': self.ligand_free_energy,
            'energy_method': self.energy_method,
            'force_field_device': self.force_field_device,
            'optimized_complex_pdb': self.optimized_complex_pdb,
            'split_protein_pdb': self.split_protein_pdb,
            'split_ligand_pdb': self.split_ligand_pdb,
            'optimized_ligand_pdb': self.optimized_ligand_pdb,
            'computation_time_seconds': self.computation_time_seconds,
            'metadata': self.metadata
        }

    @classmethod
    def from_error(cls, pose_id: str, error_message: str,
                   metadata: Optional[Dict[str, Any]] = None) -> 'RankingResult':
        """
        Create RankingResult for a failed ranking.

        Args:
            pose_id: Pose identifier
            error_message: Description of the error
            metadata: Optional metadata to preserve

        Returns:
            RankingResult with failure state
        """
        return cls(
            pose_id=pose_id,
            ranking_success=False,
            error_message=error_message,
            metadata=metadata or {}
        )

    @classmethod
    def from_success(cls, pose_id: str, interaction_energy: float,
                    strain_energy: float, energy_method: str,
                    force_field_device: str,
                    complex_energy: Optional[float] = None,
                    protein_energy: Optional[float] = None,
                    ligand_bound_energy: Optional[float] = None,
                    ligand_free_energy: Optional[float] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> 'RankingResult':
        """
        Create RankingResult for a successful ranking.

        Args:
            pose_id: Pose identifier
            interaction_energy: Calculated interaction energy
            strain_energy: Calculated strain energy
            energy_method: Method used
            force_field_device: Device used
            complex_energy: Raw complex energy
            protein_energy: Raw protein energy
            ligand_bound_energy: Raw ligand bound energy
            ligand_free_energy: Raw ligand free energy
            metadata: Optional metadata to preserve

        Returns:
            RankingResult with success state
        """
        result = cls(
            pose_id=pose_id,
            ranking_success=True,
            interaction_energy=interaction_energy,
            strain_energy=strain_energy,
            complex_energy=complex_energy,
            protein_energy=protein_energy,
            ligand_bound_energy=ligand_bound_energy,
            ligand_free_energy=ligand_free_energy,
            energy_method=energy_method,
            force_field_device=force_field_device,
            metadata=metadata or {}
        )
        result.calculate_total_score()
        return result
