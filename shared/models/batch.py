"""
Pose batch container for TPose ranking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

from .pose import Pose

logger = logging.getLogger(__name__)


@dataclass
class PoseBatch:
    """
    Container for a batch of poses to be ranked.

    Attributes:
        poses: List of Pose objects
        batch_id: Optional batch identifier
        global_energy_method: Global energy method override
    """

    poses: List[Pose] = field(default_factory=list)
    batch_id: Optional[str] = None
    global_energy_method: Optional[str] = None

    def validate(self) -> bool:
        """
        Validate all poses in the batch.

        Returns:
            True if all poses are valid, False otherwise
        """
        if not self.poses:
            logger.error("PoseBatch is empty")
            return False

        if self.global_energy_method and self.global_energy_method not in ["gfn2", "so3lr"]:
            logger.error(
                f"Invalid global_energy_method '{self.global_energy_method}'. "
                "Must be 'gfn2' or 'so3lr'"
            )
            return False

        all_valid = True
        for i, pose in enumerate(self.poses):
            if not pose.validate():
                logger.error(f"Pose {i} ({pose.pose_id}) is invalid")
                all_valid = False

        return all_valid

    def size(self) -> int:
        """
        Get number of poses in batch.

        Returns:
            Number of poses
        """
        return len(self.poses)

    def get_pose_ids(self) -> List[str]:
        """
        Get list of all pose IDs in batch.

        Returns:
            List of pose IDs
        """
        return [pose.pose_id for pose in self.poses]

    def get_energy_method(self, pose: Pose) -> str:
        """
        Get the energy method to use for a pose.

        Priority:
        1. Pose-level energy_method
        2. Batch-level global_energy_method
        3. Default "gfn2"

        Args:
            pose: Pose object

        Returns:
            Energy method string
        """
        if pose.energy_method:
            return pose.energy_method
        if self.global_energy_method:
            return self.global_energy_method
        return "gfn2"

    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]],
                       batch_id: Optional[str] = None,
                       global_energy_method: Optional[str] = None) -> 'PoseBatch':
        """
        Create PoseBatch from list of dictionaries.

        Args:
            data: List of pose dictionaries
            batch_id: Optional batch identifier
            global_energy_method: Global energy method override

        Returns:
            PoseBatch instance
        """
        poses = [Pose.from_dict(d) for d in data]
        return cls(
            poses=poses,
            batch_id=batch_id,
            global_energy_method=global_energy_method
        )

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert PoseBatch to list of dictionaries.

        Returns:
            List of pose dictionaries
        """
        return [pose.to_dict() for pose in self.poses]

    def add_pose(self, pose: Pose) -> None:
        """
        Add a pose to the batch.

        Args:
            pose: Pose to add
        """
        self.poses.append(pose)

    def filter_valid_poses(self) -> 'PoseBatch':
        """
        Create a new batch containing only valid poses.

        Returns:
            New PoseBatch with only valid poses
        """
        valid_poses = [pose for pose in self.poses if pose.validate()]
        logger.info(
            f"Filtered batch: {len(valid_poses)}/{len(self.poses)} poses valid"
        )
        return PoseBatch(
            poses=valid_poses,
            batch_id=self.batch_id,
            global_energy_method=self.global_energy_method
        )
