"""Global seeding and device selection."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU+CUDA) for reproducibility.

    Args:
        seed: Integer seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick the compute device automatically (never hardcoded).

    Returns:
        ``cuda`` if available else ``cpu``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
