"""
Base force field abstract class for TPose ranking.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import logging

from shared.models.ranking_result import RankingResult

logger = logging.getLogger(__name__)


class BaseForceField(ABC):
    """
    Abstract base class for force field implementations.

    All force fields must implement:
    - rank_pose: Calculate energies for a protein-ligand pose
    - check_availability: Verify force field can be used
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize force field.

        Args:
            device: Device to use ("cpu", "cuda:0", etc.)
        """
        self.device = device
        logger.info(f"{self.__class__.__name__} initialized with device: {device}")

    @abstractmethod
    def rank_pose(self, protein_pdb: str, ligand_sdf: str,
                  work_dir: str, params: dict) -> RankingResult:
        """
        Rank a single pose by calculating energies.

        Args:
            protein_pdb: Path to protein PDB file
            ligand_sdf: Path to ligand SDF file
            work_dir: Working directory for temporary files
            params: Additional parameters (e.g., chopping, optimization settings)

        Returns:
            RankingResult object with energies and file paths

        Raises:
            Exception: If ranking fails
        """
        pass

    @abstractmethod
    def check_availability(self) -> Tuple[bool, str]:
        """
        Check if force field is available and can be used.

        Returns:
            Tuple of (is_available, message)
            - is_available: True if force field can be used
            - message: Description of availability status or error
        """
        pass

    def get_name(self) -> str:
        """
        Get force field name.

        Returns:
            Force field name
        """
        return self.__class__.__name__

    def get_device(self) -> str:
        """
        Get current device.

        Returns:
            Device string
        """
        return self.device

    def set_device(self, device: str) -> None:
        """
        Set device for force field.

        Args:
            device: Device string
        """
        logger.info(f"{self.get_name()}: Changing device from {self.device} to {device}")
        self.device = device
