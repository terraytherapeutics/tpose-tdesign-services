"""
Configuration settings for TPose services.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Configuration settings for TPose ranking service."""

    # AWS Configuration
    aws_region: str = "us-west-2"
    s3_bucket: Optional[str] = None
    s3_output_folder: Optional[str] = None

    # Force Field Configuration
    energy_method: str = "gfn2"  # "gfn2" or "so3lr"
    so3lr_use_chopping: bool = True
    so3lr_optimize: bool = True
    so3lr_lr_cutoff: float = 12.0  # 12.0 for chopped, 20.0 for full

    # Computation Configuration
    default_device: str = "auto"  # "auto", "cpu", "cuda:0"
    enable_gpu_fallback: bool = True

    # File Processing
    distance_cutoff: float = 5.0  # Angstroms for chopping

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'Settings':
        """Create Settings from environment variables."""
        return cls(
            aws_region=os.getenv('AWS_DEFAULT_REGION', 'us-west-2'),
            s3_bucket=os.getenv('S3_BUCKET'),
            s3_output_folder=os.getenv('S3_OUTPUT_FOLDER'),
            energy_method=os.getenv('ENERGY_METHOD', 'gfn2'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
        )

    def validate(self) -> bool:
        """
        Validate settings.

        Returns:
            True if valid, False otherwise
        """
        if self.energy_method not in ["gfn2", "so3lr"]:
            return False

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            return False

        return True
