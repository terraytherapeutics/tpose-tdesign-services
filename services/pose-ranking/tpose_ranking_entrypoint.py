#!/usr/bin/env python3
"""
Tengine-compatible entrypoint for TPose pose ranking service.

TDesign will launch multiple instances of this container for parallelization.
Each instance processes its assigned subset of poses independently.
"""
import sys
import logging
from typing import List, Dict, Any

sys.path.insert(0, '/app')

from shared.config.settings import Settings
from shared.utils.logging_config import setup_logging, get_logger
from shared.models.batch import PoseBatch
from src.ranking_runner import RankingRunner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _extract_param_values(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameter values from TDesign parameter format.

    TDesign may pass parameters in format: {"param_name": {"value": x, "description": y, ...}}
    This function extracts just the values: {"param_name": x}

    Args:
        params: Raw parameters dict from TDesign

    Returns:
        Clean parameters dict with just values
    """
    clean_params = {}
    for key, val in params.items():
        if isinstance(val, dict) and 'value' in val:
            # TDesign format: extract value field
            clean_params[key] = val['value']
        else:
            # Already a simple value
            clean_params[key] = val
    return clean_params


def tpose_rank_poses(poses: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tengine-compatible pose ranking function.

    This function ranks protein-ligand poses using force field calculations.
    TDesign handles parallelization by launching multiple container instances,
    each with a subset of poses.

    Args:
        poses: List of pose dictionaries. Each must contain:
            - pose_id: Unique identifier (required)
            - structure_cif: CIF file path or S3 URI (Option 1)
              OR
            - protein_pdb: Protein PDB file path or S3 URI (Option 2)
            - ligand_sdf: Ligand SDF file path or S3 URI (Option 2)
            Optional per-pose fields (override global params):
            - energy_method: "gfn2" or "so3lr"
            - metadata: Dict with user metadata

        params: Parameters dictionary containing global defaults:
            - energy_method: Force field method (default: "gfn2")
                "gfn2" - xTB GFN2 (CPU only)
                "so3lr" - SO3LR ML force field (GPU/CPU)
            - s3_bucket: S3 bucket for I/O (required)
            - s3_output_folder: S3 output folder path (required)

            SO3LR-specific parameters:
            - so3lr_use_chopping: Use 5Å chopped system (default: True)
            - so3lr_optimize: Optimize geometries (default: True)
            - so3lr_lr_cutoff: Long-range cutoff in Å (default: 12.0 for chopped, 20.0 for full)

            Advanced parameters:
            - distance_cutoff: Chopping distance in Å (default: 5.0)
            - device: Device for computation (default: "auto")
                "auto" - Auto-detect GPU, fallback to CPU
                "cpu" - Force CPU usage
                "cuda:0" - Use specific GPU

    Returns:
        List of result dictionaries with same length and order as input poses.
        Each result contains original pose data augmented with:
            - ranking_success: bool
            - error_message: str (if failed)
            - interaction_energy: float (kcal/mol)
            - strain_energy: float (kcal/mol)
            - total_score: float (kcal/mol)
            - complex_energy: float (kcal/mol or eV converted to kcal/mol)
            - protein_energy: float (kcal/mol or eV converted to kcal/mol)
            - ligand_bound_energy: float (kcal/mol or eV converted to kcal/mol)
            - ligand_free_energy: float (kcal/mol or eV converted to kcal/mol)
            - energy_method: str ("gfn2" or "so3lr")
            - force_field_device: str ("cpu" or "cuda:0")
            - optimized_complex_pdb: str (S3 path)
            - split_protein_pdb: str (S3 path)
            - split_ligand_pdb: str (S3 path)
            - optimized_ligand_pdb: str (S3 path)
            - computation_time_seconds: float
    """
    # Extract clean parameter values from TDesign format
    params = _extract_param_values(params)

    logger.info("=" * 80)
    logger.info("TPOSE POSE RANKING - Starting")
    logger.info(f"Received {len(poses)} poses to process")
    logger.info(f"Parameters: {list(params.keys())}")
    logger.info("=" * 80)

    try:
        # Load settings (uses defaults + environment variables)
        settings = Settings()

        # Setup logging with appropriate level
        log_level = params.get('log_level', settings.log_level)
        setup_logging(
            level=log_level,
            log_file=None  # Log to stdout for TDesign
        )

        logger.info("Settings loaded successfully")

        # Get S3 parameters from params
        bucket = params.get('s3_bucket')
        output_folder = params.get('s3_output_folder') or 'tpose-output'

        if not bucket:
            error_msg = "Missing required parameter: s3_bucket"
            logger.error(error_msg)
            # TDesign expects input poses to be augmented with results
            for pose in poses:
                pose['ranking_success'] = False
                pose['error_message'] = error_msg
            return poses

        logger.info(f"S3 Bucket: {bucket}")
        logger.info(f"Output Folder: {output_folder}")

        # Extract energy method
        global_energy_method = params.get('energy_method', 'gfn2')
        logger.info(f"Global Energy Method: {global_energy_method}")

        # Validate energy method
        if global_energy_method not in ['gfn2', 'so3lr']:
            error_msg = f"Invalid energy_method: {global_energy_method}. Must be 'gfn2' or 'so3lr'"
            logger.error(error_msg)
            for pose in poses:
                pose['ranking_success'] = False
                pose['error_message'] = error_msg
            return poses

        # Initialize ranking runner
        logger.info("Initializing RankingRunner...")
        runner = RankingRunner(settings)
        logger.info("RankingRunner initialized successfully")

        # Create pose batch from input data
        logger.info("Creating pose batch...")
        pose_batch = PoseBatch.from_dict_list(
            poses,
            batch_id=f"tpose_batch_{len(poses)}",
            global_energy_method=global_energy_method
        )

        logger.info(f"Created batch with {pose_batch.size()} poses")

        # Validate batch
        if not pose_batch.validate():
            error_msg = "Invalid pose data in batch"
            logger.error(error_msg)
            # TDesign expects input poses to be augmented with results
            for pose in poses:
                pose['ranking_success'] = False
                pose['error_message'] = error_msg
            return poses

        logger.info("Pose batch validation passed")

        # Extract additional parameters for ranking
        ranking_params = {
            'energy_method': global_energy_method,
            'device': params.get('device', 'auto'),
            'use_chopping': params.get('so3lr_use_chopping', True),
            'optimize_complex': params.get('so3lr_optimize', True),
            'optimize_ligand': params.get('so3lr_optimize', True),
            'lr_cutoff': params.get('so3lr_lr_cutoff', 12.0),
            'distance_cutoff': params.get('distance_cutoff', 5.0),
        }

        logger.info(f"Ranking parameters: {ranking_params}")

        # Run ranking
        logger.info("Starting batch ranking...")
        results = runner.rank_batch(
            pose_batch,
            bucket,
            output_folder,
            ranking_params
        )

        # TDesign expects input poses to be augmented with results, not replaced
        # Merge ranking results back into the original input dictionaries
        logger.info("Merging results into input pose dictionaries...")
        for i, pose_dict in enumerate(poses):
            if i < len(results):
                result = results[i]
                # Convert RankingResult to dictionary and merge
                result_dict = result.to_dict()

                # Add all result fields to the input pose dictionary
                for key, value in result_dict.items():
                    pose_dict[key] = value
            else:
                # Handle case where we have fewer results than input poses
                pose_dict['ranking_success'] = False
                pose_dict['error_message'] = 'No result returned for this pose'

        # Log summary
        successful = sum(1 for pose in poses if pose.get('ranking_success', False))
        logger.info("=" * 80)
        logger.info(f"TPOSE POSE RANKING - Completed")
        logger.info(f"Successful: {successful}/{len(poses)}")
        logger.info(f"Failed: {len(poses) - successful}/{len(poses)}")

        # Log performance summary for successful poses
        if successful > 0:
            times = [
                pose.get('computation_time_seconds', 0)
                for pose in poses
                if pose.get('ranking_success', False)
            ]
            avg_time = sum(times) / len(times) if times else 0
            logger.info(f"Average computation time: {avg_time:.2f}s per pose")

            # Log energy summary
            interaction_energies = [
                pose.get('interaction_energy', 0)
                for pose in poses
                if pose.get('ranking_success', False)
            ]
            if interaction_energies:
                avg_interaction = sum(interaction_energies) / len(interaction_energies)
                logger.info(f"Average interaction energy: {avg_interaction:.2f} kcal/mol")

        logger.info("=" * 80)

        return poses

    except Exception as e:
        logger.error(f"CRITICAL ERROR in tpose_rank_poses: {e}", exc_info=True)

        # TDesign expects input poses to be augmented with results
        # Add error information to all input poses
        for pose in poses:
            pose['ranking_success'] = False
            pose['error_message'] = f"Critical error during ranking: {str(e)}"

        return poses


if __name__ == "__main__":
    try:
        # Use tengine2.entrypoint for TDesign integration
        # This handles pickled I/O from S3 automatically
        from tengine2.entrypoint import entrypoint
        logger.info("Launching tengine2 entrypoint wrapper...")
        entrypoint(tpose_rank_poses)
    except ImportError as e:
        logger.error(f"Failed to import tengine2.entrypoint: {e}")
        logger.error("Make sure tengine2 is installed via CodeArtifact")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to launch entrypoint: {e}", exc_info=True)
        sys.exit(1)
