# TPose TDesign Services

Pose ranking service for TDesign integration using xTB and SO3LR force fields.

## Overview

This service ranks protein-ligand poses by calculating interaction energies and strain energies using either:
- **xTB GFN2**: Semi-empirical quantum chemistry (CPU)
- **SO3LR**: Machine learning force field (GPU with CPU fallback)

## Features

- CIF file conversion to PDB/SDF format
- Direct PDB + SDF input support
- Multiple pose batch processing
- S3 integration for input/output
- TDesign task integration via tengine2
- Comprehensive error handling and logging

## Repository Structure

```
tpose-tdesign-services/
├── shared/                    # Reusable utilities
│   ├── models/               # Data models (Pose, RankingResult, PoseBatch)
│   ├── utils/                # S3, logging, CIF converter, GPU utils
│   └── config/               # Settings management
├── services/pose-ranking/    # Main service
│   ├── src/                  # Implementation
│   │   ├── ranking_runner.py # Orchestration
│   │   └── force_fields/     # Force field implementations
│   ├── tests/                # Test suite
│   ├── Dockerfile            # Container definition
│   └── requirements.txt      # Python dependencies
├── register_tpose_task.py    # TDesign task registration
└── claude.md                 # Progress tracking
```

## Installation

### Prerequisites
- Python 3.10+
- xTB v6.6.1 (for GFN2 method)
- CUDA 12.2+ (for SO3LR GPU acceleration)

### Dependencies
```bash
pip install -r services/pose-ranking/requirements.txt
```

## Usage

### As TDesign Task
The service is designed to run as a TDesign task. See `register_tpose_task.py` for task definition.

### Standalone (Development)
```python
from services.pose_ranking.tpose_ranking_entrypoint import tpose_rank_poses

poses = [
    {
        "pose_id": "pose_001",
        "protein_pdb": "s3://bucket/protein.pdb",
        "ligand_sdf": "s3://bucket/ligand.sdf"
    }
]

params = {
    "energy_method": {"value": "gfn2"},
    "s3_bucket": {"value": "my-bucket"},
    "s3_output_folder": {"value": "output/"}
}

results = tpose_rank_poses(poses, params)
```

## Development

See `claude.md` for implementation progress and development notes.

### Running Tests
```bash
pytest services/pose-ranking/tests/
```

## Docker

Build the container:
```bash
docker build -t tpose-ranking services/pose-ranking/
```

## License

Terray Therapeutics Internal Use
