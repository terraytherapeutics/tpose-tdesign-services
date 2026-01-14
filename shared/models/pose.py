"""
Pose data model for TPose ranking.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    """
    Represents a single protein-ligand pose to be ranked.

    Attributes:
        pose_id: Unique identifier for the pose
        structure_cif: CIF file path or content (Option 1)
        protein_pdb: Protein PDB file path or content (Option 2)
        ligand_sdf: Ligand SDF file path or content (Option 2)
        structure_path: S3 path where optimized complex structure will be written
        energy_method: Force field method ("gfn2" or "so3lr")
        metadata: Optional user metadata
    """

    pose_id: str
    structure_cif: Optional[str] = None
    protein_pdb: Optional[str] = None
    ligand_sdf: Optional[str] = None
    structure_path: Optional[str] = None  # S3 path where optimized complex will be written
    energy_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """
        Validate pose data.

        Returns:
            True if valid, False otherwise
        """
        if not self.pose_id:
            logger.error("Pose ID is required")
            return False

        # Must have either CIF or PDB+SDF
        has_cif = self.structure_cif is not None
        has_pdb_sdf = (self.protein_pdb is not None and
                       self.ligand_sdf is not None)

        if not (has_cif or has_pdb_sdf):
            logger.error(
                f"Pose {self.pose_id}: Must provide either structure_cif "
                "or both protein_pdb and ligand_sdf"
            )
            return False

        if has_cif and has_pdb_sdf:
            logger.warning(
                f"Pose {self.pose_id}: Both CIF and PDB+SDF provided, "
                "will use PDB+SDF"
            )

        if self.energy_method and self.energy_method not in ["gfn2", "so3lr"]:
            logger.error(
                f"Pose {self.pose_id}: Invalid energy_method '{self.energy_method}'. "
                "Must be 'gfn2' or 'so3lr'"
            )
            return False

        return True

    def needs_cif_conversion(self) -> bool:
        """
        Check if CIF conversion is needed.

        Returns:
            True if CIF needs to be converted to PDB+SDF
        """
        return (self.structure_cif is not None and
                (self.protein_pdb is None or self.ligand_sdf is None))

    def has_direct_structures(self) -> bool:
        """
        Check if direct PDB+SDF structures are provided.

        Returns:
            True if both protein_pdb and ligand_sdf are provided
        """
        return (self.protein_pdb is not None and
                self.ligand_sdf is not None)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pose':
        """
        Create Pose from dictionary.

        Args:
            data: Dictionary containing pose data

        Returns:
            Pose instance
        """
        return cls(
            pose_id=data.get('pose_id', ''),
            structure_cif=data.get('structure_cif'),
            protein_pdb=data.get('protein_pdb'),
            ligand_sdf=data.get('ligand_sdf'),
            structure_path=data.get('structure_path'),
            energy_method=data.get('energy_method'),
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Pose to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'pose_id': self.pose_id,
            'structure_cif': self.structure_cif,
            'protein_pdb': self.protein_pdb,
            'ligand_sdf': self.ligand_sdf,
            'structure_path': self.structure_path,
            'energy_method': self.energy_method,
            'metadata': self.metadata
        }
