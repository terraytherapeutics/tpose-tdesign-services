#!/usr/bin/env python3
"""
Register TPose Pose Ranking task for TDesign with direct SLURM execution.

Architecture: Direct execution on SLURM GPU/CPU nodes
- TDesign launches multiple container instances for parallelization
- Each instance processes poses directly using xTB (CPU) or SO3LR (GPU)
- No AWS Batch orchestration layer

Features:
- Two force field methods: xTB GFN2 (CPU) and SO3LR (GPU with CPU fallback)
- CIF file conversion or direct PDB+SDF input
- Configurable optimization and chopping parameters
- Automatic GPU detection with CPU fallback
- S3 integration for input/output files
"""

from tengine2.clients.engine_client import EngineClient


def register_tpose_ranking_task():
    """
    Register TPose Pose Ranking task with direct SLURM execution.

    This registration maps TDesign parameters to the tpose_rank_poses() function
    in services/pose-ranking/tpose_ranking_entrypoint.py. Executes directly on
    SLURM nodes (GPU for SO3LR, CPU for xTB).

    TDesign handles parallelization by launching multiple container instances,
    each processing a subset of poses independently.
    """

    # Initialize client for staging environment
    client = EngineClient(env="staging")

    # Define TPose Pose Ranking task
    TPOSE_TASK = {
        "name": "TPose - Pose Ranking",
        "unique_metadata": None,
        "task_category": "predictive",
        "visibility": {"tdesign": True},
        "author": "Gavin Bascom",
        "version": "1.0.0",
        "summary": """TPose: Protein-ligand pose ranking using force field energy calculations.

Ranks protein-ligand poses by calculating:
- Interaction energy: Protein-ligand binding energy
- Strain energy: Ligand conformational strain
- Total score: Sum of interaction and strain energies

Two force field methods available:
1. xTB GFN2: Semi-empirical quantum chemistry (CPU only, ~2-5 min/pose)
2. SO3LR: Machine learning force field (GPU with CPU fallback, ~30s-2 min/pose on GPU)

Input formats:
- Option 1: CIF file (e.g., from Boltz predictions) - automatically converted to PDB+SDF
- Option 2: Direct PDB (protein) + SDF (ligand) files

All energies are reported in kcal/mol. Recommended maximum of 50-100 poses per run.

Outputs:
"ranking_success":              # Whether ranking completed successfully
"interaction_energy":           # Protein-ligand interaction energy (kcal/mol)
"strain_energy":                # Ligand conformational strain energy (kcal/mol)
"total_score":                  # Total score = interaction + strain (kcal/mol)
"complex_energy":               # Raw complex energy (kcal/mol)
"protein_energy":               # Raw protein energy (kcal/mol)
"ligand_bound_energy":          # Ligand energy in bound state (kcal/mol)
"ligand_free_energy":           # Ligand energy in free state (kcal/mol)
"energy_method":                # Force field used ("gfn2" or "so3lr")
"force_field_device":           # Device used ("cpu" or "cuda:0")
"optimized_complex_pdb":        # S3 path to optimized complex structure
"split_protein_pdb":            # S3 path to protein structure
"split_ligand_pdb":             # S3 path to ligand bound conformation
"optimized_ligand_pdb":         # S3 path to ligand free conformation
"computation_time_seconds":     # Time taken for ranking
"error_message":                # Error description if ranking failed""",
        "entrypoint_version": "v2",
        "parameters": {
            # ========== Force Field Selection ==========
            "energy_method": {
                "value": "gfn2",
                "input_type": "text",
                "description": "Force field method: 'gfn2' (xTB GFN2, CPU, slower but robust) or 'so3lr' (ML force field, GPU, faster)"
            },

            # ========== SO3LR Configuration (only used if energy_method='so3lr') ==========
            "so3lr_use_chopping": {
                "value": True,
                "input_type": "boolean",
                "description": "Use 5Å chopped system (recommended). False = full protein system."
            },
            "so3lr_optimize": {
                "value": True,
                "input_type": "boolean",
                "description": "Optimize geometries before energy calculation (recommended for accuracy)"
            },
            "so3lr_lr_cutoff": {
                "value": 12.0,
                "input_type": "float",
                "description": "Long-range interaction cutoff (Å). Use 12.0 for chopped systems, 20.0 for full protein."
            },

            # ========== Advanced Parameters ==========
            "distance_cutoff": {
                "value": 5.0,
                "input_type": "float",
                "description": "Distance cutoff (Å) for protein chopping around ligand"
            },
            "device": {
                "value": "auto",
                "input_type": "text",
                "description": "Device for computation: 'auto' (detect GPU, fallback to CPU), 'cpu', or 'cuda:0'"
            },

            # ========== Infrastructure ==========
            "s3_bucket": {
                "value": "terray-tpose",
                "input_type": "text",
                "description": "S3 bucket for I/O"
            },
            "s3_output_folder": {
                "value": "tpose-output",
                "input_type": "text",
                "description": "S3 output folder path for results"
            },
        },
        "executor_params": [
            {
                "type": "slurm",
                "container_repo": "291672471869.dkr.ecr.us-west-2.amazonaws.com/tdesign-tpose-ranking",
                "container_tag": "dev",
                "command": "python3 /app/tpose_ranking_entrypoint.py",
                "gpus": 1,  # GPU for SO3LR (xTB will use CPU regardless)
                "cpus": 4,
                "memory": "16G",
                "environment": {
                    "AWS_S3_BUCKET": "terray-tpose",
                    "PYTHONPATH": "/app",
                    # Tengine2 will automatically provide:
                    # - INPUT_PREFIX: S3 path with pickled poses
                    # - OUTPUT_PREFIX: S3 path for pickled results
                    # - PARAMS: JSON string with task parameters
                }
            }
        ],
        "output_properties": [
            # Core ranking outputs
            'pose_id',
            'ranking_success',
            'error_message',

            # Energy values (kcal/mol)
            'interaction_energy',
            'strain_energy',
            'total_score',
            'complex_energy',
            'protein_energy',
            'ligand_bound_energy',
            'ligand_free_energy',

            # Method information
            'energy_method',
            'force_field_device',

            # Output structures (S3 paths)
            'optimized_complex_pdb',
            'split_protein_pdb',
            'split_ligand_pdb',
            'optimized_ligand_pdb',

            # Execution metadata
            'computation_time_seconds',
            'metadata',
        ]
    }

    print("Registering TPose Pose Ranking Task (Direct SLURM Execution)...")
    print(f"Task Name: {TPOSE_TASK['name']}")
    print(f"Version: {TPOSE_TASK['version']}")
    print(f"Container: {TPOSE_TASK['executor_params'][0]['container_repo']}:{TPOSE_TASK['executor_params'][0]['container_tag']}")
    print(f"GPUs: {TPOSE_TASK['executor_params'][0]['gpus']}")
    print(f"Architecture: Direct execution on SLURM nodes (GPU/CPU)")
    print()

    # Register the task (undeploy_old=True to replace any existing version)
    response = client.register_task(TPOSE_TASK, undeploy_old=True)

    print("Task Registration Response:")
    print(f"  Name: {response.get('name')}")
    print(f"  Environment: {response.get('environment')}")
    print(f"  Active: {response.get('active')}")
    print(f"  Deployment ID: {response.get('deployment_id')}")
    print(f"  Visibility: {response.get('visibility')}")

    return response


def test_task_registration():
    """
    Test the registered TPose Pose Ranking task with sample use cases.

    Use cases covered:
    1. xTB GFN2 ranking (CPU, robust)
    2. SO3LR ranking (GPU, fast)
    3. CIF file input (from Boltz predictions)
    4. Direct PDB+SDF input
    """

    print("\n" + "="*70)
    print("TPose Pose Ranking Task - Example Use Cases")
    print("="*70)

    # Use Case 1: xTB GFN2 ranking (CPU, robust)
    print("\nUse Case 1: xTB GFN2 Ranking (CPU)")
    print("-" * 70)
    use_case_1_poses = [
        {
            "pose_id": "pose_001",
            "protein_pdb": "s3://terray-tpose/input/protein_001.pdb",
            "ligand_sdf": "s3://terray-tpose/input/ligand_001.sdf",
        },
        {
            "pose_id": "pose_002",
            "protein_pdb": "s3://terray-tpose/input/protein_002.pdb",
            "ligand_sdf": "s3://terray-tpose/input/ligand_002.sdf",
        },
    ]
    use_case_1_params = {
        "energy_method": "gfn2",  # xTB GFN2 (CPU)
        "s3_bucket": "terray-tpose",
        "s3_output_folder": "output/xtb_results",
    }
    print(f"  Poses: {len(use_case_1_poses)}")
    print(f"  Method: xTB GFN2 (semi-empirical QM)")
    print(f"  Device: CPU only")
    print(f"  Expected time: ~2-5 min/pose")
    print(f"  Features: Robust, well-validated")

    # Use Case 2: SO3LR ranking (GPU, fast)
    print("\nUse Case 2: SO3LR Ranking (GPU)")
    print("-" * 70)
    use_case_2_poses = [
        {
            "pose_id": "pose_001",
            "protein_pdb": "s3://terray-tpose/input/protein_001.pdb",
            "ligand_sdf": "s3://terray-tpose/input/ligand_001.sdf",
        },
        {
            "pose_id": "pose_002",
            "protein_pdb": "s3://terray-tpose/input/protein_002.pdb",
            "ligand_sdf": "s3://terray-tpose/input/ligand_002.sdf",
        },
    ]
    use_case_2_params = {
        "energy_method": "so3lr",  # SO3LR ML force field
        "so3lr_use_chopping": True,
        "so3lr_optimize": True,
        "so3lr_lr_cutoff": 12.0,
        "device": "auto",  # Auto-detect GPU, fallback to CPU
        "s3_bucket": "terray-tpose",
        "s3_output_folder": "output/so3lr_results",
    }
    print(f"  Poses: {len(use_case_2_poses)}")
    print(f"  Method: SO3LR (ML force field)")
    print(f"  Device: GPU with CPU fallback")
    print(f"  Expected time: ~30s-2 min/pose on GPU")
    print(f"  Features: Fast, quantum-accurate")

    # Use Case 3: CIF file input (from Boltz predictions)
    print("\nUse Case 3: CIF File Input (Boltz → TPose Pipeline)")
    print("-" * 70)
    use_case_3_poses = [
        {
            "pose_id": "boltz_pose_001",
            "structure_cif": "s3://terray-boltz/output/structure_001.cif",
        },
        {
            "pose_id": "boltz_pose_002",
            "structure_cif": "s3://terray-boltz/output/structure_002.cif",
        },
    ]
    use_case_3_params = {
        "energy_method": "so3lr",
        "s3_bucket": "terray-tpose",
        "s3_output_folder": "output/boltz_to_tpose",
    }
    print(f"  Poses: {len(use_case_3_poses)}")
    print(f"  Input: CIF files from Boltz predictions")
    print(f"  Features: Automatic CIF → PDB+SDF conversion")
    print(f"  Use case: Rank Boltz-generated poses")

    # Use Case 4: Mixed input (some CIF, some PDB+SDF)
    print("\nUse Case 4: Mixed Input Formats")
    print("-" * 70)
    use_case_4_poses = [
        {
            "pose_id": "mixed_001",
            "structure_cif": "s3://terray-tpose/input/structure_001.cif",
        },
        {
            "pose_id": "mixed_002",
            "protein_pdb": "s3://terray-tpose/input/protein_002.pdb",
            "ligand_sdf": "s3://terray-tpose/input/ligand_002.sdf",
        },
    ]
    use_case_4_params = {
        "energy_method": "gfn2",
        "s3_bucket": "terray-tpose",
        "s3_output_folder": "output/mixed_input",
    }
    print(f"  Poses: {len(use_case_4_poses)}")
    print(f"  Input: Mixed CIF and PDB+SDF")
    print(f"  Features: Flexible input handling")

    print("\n" + "="*70)
    print("To run any of these use cases:")
    print("="*70)
    print("""
from tengine2.clients.engine_client import EngineClient

client = EngineClient(env='staging')

# Run Use Case 1 (xTB GFN2)
results = client.run_task(
    molecules=use_case_1_poses,  # 'molecules' is TDesign's generic term
    task_name='TPose - Pose Ranking',
    parameters=use_case_1_params
)

# Check results
for result in results:
    print(f"Pose {result['pose_id']}: Success={result['ranking_success']}")
    if result['ranking_success']:
        print(f"  Interaction energy: {result['interaction_energy']:.2f} kcal/mol")
        print(f"  Strain energy: {result['strain_energy']:.2f} kcal/mol")
        print(f"  Total score: {result['total_score']:.2f} kcal/mol")
        print(f"  Method: {result['energy_method']} on {result['force_field_device']}")
        print(f"  Time: {result['computation_time_seconds']:.1f}s")
    else:
        print(f"  Error: {result['error_message']}")
    """)


if __name__ == "__main__":
    print("🚀 Registering TPose Pose Ranking Task - Direct SLURM Execution")
    print("="*70)
    print("Architecture: Direct execution on SLURM GPU/CPU nodes")
    print("  - TDesign launches multiple container instances for parallelization")
    print("  - Each instance ranks poses directly using xTB or SO3LR")
    print("  - GPU for SO3LR (with CPU fallback), CPU for xTB")
    print("="*70)
    print()

    # Register the task
    response = register_tpose_ranking_task()

    # Show example use cases
    test_task_registration()

    print("\n🎉 Task registration complete!")
    print("\nNext steps:")
    print("1. Build and push tdesign-tpose-ranking:dev container to ECR:")
    print("   cd /path/to/tpose-tdesign-services")
    print("   docker build -f services/pose-ranking/Dockerfile -t 291672471869.dkr.ecr.us-west-2.amazonaws.com/tdesign-tpose-ranking:dev .")
    print("   docker push 291672471869.dkr.ecr.us-west-2.amazonaws.com/tdesign-tpose-ranking:dev")
    print("2. Ensure tengine2 is installed in container (from CodeArtifact)")
    print("3. Test one of the use cases above in TDesign staging")
    print("4. Verify outputs match expected energy fields")
    print("5. Deploy to production when validated")
