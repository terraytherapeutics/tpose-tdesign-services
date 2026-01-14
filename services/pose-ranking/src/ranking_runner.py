"""
Ranking runner for orchestrating pose ranking workflows.
"""

import os
import tempfile
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys
sys.path.insert(0, '/app')

from shared.config.settings import Settings
from shared.models.pose import Pose
from shared.models.ranking_result import RankingResult
from shared.models.batch import PoseBatch
from shared.utils.s3_client import S3Client
from shared.utils.cif_converter import convert_cif_to_pdb_sdf
from shared.utils.logging_config import get_logger

from .force_fields import XTBForceField, SO3LRForceField

logger = get_logger(__name__)


class RankingRunner:
    """Orchestrates pose ranking workflows using various force fields."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize RankingRunner.

        Args:
            settings: Configuration settings (if None, uses defaults)
        """
        self.settings = settings or Settings()

        # Initialize S3 client
        self.s3_client = S3Client.from_env()

        # Force field instances (lazy-loaded)
        self._xtb_ff = None
        self._so3lr_ff = None

        logger.info("RankingRunner initialized")

    def get_force_field(self, energy_method: str, device: str = "auto"):
        """
        Get force field instance (lazy-loaded).

        Args:
            energy_method: "gfn2" or "so3lr"
            device: Device for force field

        Returns:
            Force field instance

        Raises:
            ValueError: If unknown energy method
        """
        if energy_method == "gfn2":
            if self._xtb_ff is None:
                self._xtb_ff = XTBForceField(device="cpu")
                available, message = self._xtb_ff.check_availability()
                if available:
                    logger.info(f"xTB force field loaded: {message}")
                else:
                    logger.warning(f"xTB force field not available: {message}")
            return self._xtb_ff

        elif energy_method == "so3lr":
            if self._so3lr_ff is None:
                self._so3lr_ff = SO3LRForceField(device=device)
                available, message = self._so3lr_ff.check_availability()
                if available:
                    logger.info(f"SO3LR force field loaded: {message}")
                else:
                    logger.warning(f"SO3LR force field not available: {message}")
            return self._so3lr_ff

        else:
            raise ValueError(
                f"Unknown energy_method: {energy_method}. "
                "Must be 'gfn2' or 'so3lr'"
            )

    def _download_from_s3_if_needed(self, path: str, bucket: str,
                                    work_dir: str) -> str:
        """
        Download file from S3 if path starts with s3://.

        Args:
            path: File path or S3 URI
            bucket: S3 bucket name
            work_dir: Working directory

        Returns:
            Local file path
        """
        if path.startswith('s3://'):
            # Parse S3 URI
            s3_path = path.replace('s3://', '')
            if '/' in s3_path:
                bucket_part, key = s3_path.split('/', 1)
                if bucket_part != bucket:
                    bucket = bucket_part
            else:
                key = s3_path

            # Download to working directory
            local_path = os.path.join(work_dir, os.path.basename(key))
            logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")

            if self.s3_client.download_file(bucket, key, local_path):
                return local_path
            else:
                raise FileNotFoundError(f"Failed to download s3://{bucket}/{key}")

        return path

    def _upload_results_to_s3(self, result: RankingResult, bucket: str,
                             output_folder: str, structure_path: Optional[str] = None) -> RankingResult:
        """
        Upload result files to S3 and update paths.

        Args:
            result: RankingResult with local file paths
            bucket: S3 bucket name
            output_folder: S3 output folder
            structure_path: Optional custom S3 path for optimized complex structure

        Returns:
            Updated RankingResult with S3 paths
        """
        pose_id = result.pose_id

        # Upload optimized complex PDB
        if result.optimized_complex_pdb and os.path.exists(result.optimized_complex_pdb):
            # Use custom structure_path if provided, otherwise use default
            if structure_path:
                # Parse bucket and key from s3:// path if needed
                if structure_path.startswith('s3://'):
                    # Extract bucket and key from s3://bucket/key format
                    path_parts = structure_path[5:].split('/', 1)
                    target_bucket = path_parts[0] if len(path_parts) > 0 else bucket
                    s3_key = path_parts[1] if len(path_parts) > 1 else f"{output_folder}/{pose_id}_complex_opt.pdb"
                else:
                    # Assume it's just a key
                    target_bucket = bucket
                    s3_key = structure_path
            else:
                target_bucket = bucket
                s3_key = f"{output_folder}/{pose_id}_complex_opt.pdb"

            if self.s3_client.upload_pdb_file(result.optimized_complex_pdb, target_bucket, s3_key):
                result.optimized_complex_pdb = f"s3://{target_bucket}/{s3_key}"
                logger.info(f"[{pose_id}] Uploaded complex PDB to {result.optimized_complex_pdb}")

        # Upload split protein PDB
        if result.split_protein_pdb and os.path.exists(result.split_protein_pdb):
            s3_key = f"{output_folder}/{pose_id}_protein.pdb"
            if self.s3_client.upload_pdb_file(result.split_protein_pdb, bucket, s3_key):
                result.split_protein_pdb = f"s3://{bucket}/{s3_key}"
                logger.info(f"[{pose_id}] Uploaded protein PDB to S3")

        # Upload split ligand PDB (bound)
        if result.split_ligand_pdb and os.path.exists(result.split_ligand_pdb):
            s3_key = f"{output_folder}/{pose_id}_ligand_bound.pdb"
            if self.s3_client.upload_pdb_file(result.split_ligand_pdb, bucket, s3_key):
                result.split_ligand_pdb = f"s3://{bucket}/{s3_key}"
                logger.info(f"[{pose_id}] Uploaded bound ligand PDB to S3")

        # Upload optimized ligand PDB (free)
        if result.optimized_ligand_pdb and os.path.exists(result.optimized_ligand_pdb):
            s3_key = f"{output_folder}/{pose_id}_ligand_opt.pdb"
            if self.s3_client.upload_pdb_file(result.optimized_ligand_pdb, bucket, s3_key):
                result.optimized_ligand_pdb = f"s3://{bucket}/{s3_key}"
                logger.info(f"[{pose_id}] Uploaded optimized ligand PDB to S3")

        return result

    def rank_single_pose(self, pose: Pose, bucket: str, output_folder: str,
                        params: Dict[str, Any]) -> RankingResult:
        """
        Rank a single pose.

        Args:
            pose: Pose object
            bucket: S3 bucket name
            output_folder: S3 output folder
            params: Additional parameters

        Returns:
            RankingResult object
        """
        start_time = time.time()
        pose_id = pose.pose_id

        logger.info(f"[{pose_id}] Starting pose ranking")

        # Create working directory
        work_dir = tempfile.mkdtemp(prefix=f"tpose_{pose_id}_")
        logger.info(f"[{pose_id}] Working directory: {work_dir}")

        try:
            # Determine energy method
            energy_method = pose.energy_method or params.get('energy_method', 'gfn2')

            # Handle CIF conversion if needed
            if pose.needs_cif_conversion():
                logger.info(f"[{pose_id}] Converting CIF to PDB+SDF")
                try:
                    # Download CIF if from S3
                    cif_path = self._download_from_s3_if_needed(
                        pose.structure_cif, bucket, work_dir
                    )

                    # Convert CIF to PDB+SDF
                    protein_pdb, ligand_sdf = convert_cif_to_pdb_sdf(cif_path, work_dir)

                    logger.info(
                        f"[{pose_id}] CIF converted: "
                        f"protein={protein_pdb}, ligand={ligand_sdf}"
                    )
                except NotImplementedError:
                    return RankingResult.from_error(
                        pose_id=pose_id,
                        error_message="CIF conversion not yet implemented",
                        metadata=pose.metadata
                    )
            else:
                # Download PDB and SDF from S3 if needed
                protein_pdb = self._download_from_s3_if_needed(
                    pose.protein_pdb, bucket, work_dir
                )
                ligand_sdf = self._download_from_s3_if_needed(
                    pose.ligand_sdf, bucket, work_dir
                )

                logger.info(
                    f"[{pose_id}] Using direct structures: "
                    f"protein={protein_pdb}, ligand={ligand_sdf}"
                )

            # Get force field
            device = params.get('device', 'auto')
            force_field = self.get_force_field(energy_method, device)

            # Prepare force field parameters
            ff_params = {
                'pose_id': pose_id,
                'metadata': pose.metadata,
                'use_chopping': params.get('use_chopping', True),
                'optimize_complex': params.get('optimize_complex', True),
                'optimize_ligand': params.get('optimize_ligand', True),
                'lr_cutoff': params.get('lr_cutoff', 12.0),
                'distance_cutoff': params.get('distance_cutoff', 5.0),
            }

            # Rank pose using force field
            logger.info(
                f"[{pose_id}] Ranking with {energy_method} "
                f"on {force_field.get_device()}"
            )
            result = force_field.rank_pose(protein_pdb, ligand_sdf, work_dir, ff_params)

            # Upload results to S3 if successful
            if result.ranking_success:
                result = self._upload_results_to_s3(result, bucket, output_folder, pose.structure_path)

            # Log summary
            elapsed = time.time() - start_time
            if result.ranking_success:
                logger.info(
                    f"[{pose_id}] Ranking successful in {elapsed:.2f}s: "
                    f"interaction={result.interaction_energy:.2f} kcal/mol, "
                    f"strain={result.strain_energy:.2f} kcal/mol"
                )
            else:
                logger.error(
                    f"[{pose_id}] Ranking failed in {elapsed:.2f}s: "
                    f"{result.error_message}"
                )

            return result

        except Exception as e:
            logger.error(f"[{pose_id}] Unexpected error: {e}", exc_info=True)
            return RankingResult.from_error(
                pose_id=pose_id,
                error_message=f"Unexpected error: {str(e)}",
                metadata=pose.metadata
            )

        finally:
            # Clean up working directory
            try:
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)
                logger.info(f"[{pose_id}] Cleaned up working directory")
            except Exception as e:
                logger.warning(f"[{pose_id}] Failed to clean up work dir: {e}")

    def rank_batch(self, batch: PoseBatch, bucket: str, output_folder: str,
                   params: Dict[str, Any]) -> List[RankingResult]:
        """
        Rank a batch of poses sequentially.

        Args:
            batch: PoseBatch containing poses
            bucket: S3 bucket name
            output_folder: S3 output folder
            params: Additional parameters

        Returns:
            List of RankingResult objects
        """
        logger.info(f"Starting batch ranking: {batch.size()} poses")

        # Validate batch
        if not batch.validate():
            logger.error("Batch validation failed")
            # Return errors for all poses
            return [
                RankingResult.from_error(
                    pose_id=pose.pose_id,
                    error_message="Batch validation failed",
                    metadata=pose.metadata
                )
                for pose in batch.poses
            ]

        # Rank poses sequentially
        results = []
        for i, pose in enumerate(batch.poses):
            logger.info(f"Processing pose {i+1}/{batch.size()}: {pose.pose_id}")

            # Determine energy method (pose-level overrides batch-level)
            effective_energy_method = batch.get_energy_method(pose)
            params['energy_method'] = effective_energy_method

            # Rank pose
            result = self.rank_single_pose(pose, bucket, output_folder, params)
            results.append(result)

        # Log summary
        successful = sum(1 for r in results if r.ranking_success)
        logger.info(
            f"Batch ranking complete: {successful}/{batch.size()} successful"
        )

        return results
