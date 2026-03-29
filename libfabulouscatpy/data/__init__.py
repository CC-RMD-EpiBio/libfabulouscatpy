"""Bundled model data for public psychometric instruments."""

import importlib.resources
from pathlib import Path


def get_grm_params_path(dataset: str) -> Path:
    """Return the path to the bundled GRM parameters .npz file.

    Args:
        dataset: One of 'grit', 'npi', 'tma', 'eqsq', 'rwa', 'wpi', 'gcbs', 'scs'.

    Returns:
        Path to the .npz file.
    """
    ref = importlib.resources.files(__package__) / "models" / f"{dataset}_grm_params.npz"
    return Path(str(ref))
