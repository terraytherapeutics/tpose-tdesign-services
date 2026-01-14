"""
GPU detection and utilities for TPose services.
"""

import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def detect_gpu() -> Tuple[bool, Optional[str]]:
    """
    Detect if GPU is available and return device string.

    Returns:
        Tuple of (is_available, device_string)
        - is_available: True if CUDA GPU is available
        - device_string: "cuda:0" if available, "cpu" otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {device_name} ({device_count} devices)")
            return True, "cuda:0"
        else:
            logger.info("No GPU detected, using CPU")
            return False, "cpu"
    except ImportError:
        logger.warning("PyTorch not installed, GPU detection unavailable")
        return False, "cpu"
    except Exception as e:
        logger.error(f"Error detecting GPU: {e}")
        return False, "cpu"


def get_device_info(device: str) -> str:
    """
    Get detailed information about the specified device.

    Args:
        device: Device string (e.g., "cuda:0", "cpu")

    Returns:
        Human-readable device information
    """
    if device == "cpu":
        return "CPU"

    try:
        import torch
        if device.startswith("cuda"):
            device_id = int(device.split(":")[1]) if ":" in device else 0
            name = torch.cuda.get_device_name(device_id)
            memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            return f"{name} ({memory:.1f} GB)"
    except Exception as e:
        logger.error(f"Error getting device info: {e}")

    return device


def check_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
