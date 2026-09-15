"""
Deterministic global seeding utility for Python random, NumPy, PyTorch, and TensorFlow.

Guards third-party library imports so that scripts do not fail if those
libraries are not yet installed in the current environment phase.
"""

from __future__ import annotations

import os
import random
from typing import Any


def set_global_seed(seed: int = 42) -> dict[str, bool]:
    """
    Seed all standard random number generators deterministically.

    Args:
        seed: The integer seed to apply globally.

    Returns:
        A dictionary indicating which frameworks were successfully seeded.
    """
    status: dict[str, bool] = {
        "python_random": False,
        "python_hash_seed": False,
        "numpy": False,
        "torch": False,
        "tensorflow": False,
    }

    # 1. Standard Python random
    random.seed(seed)
    status["python_random"] = True

    # 2. Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    status["python_hash_seed"] = True

    # 3. NumPy (guarded)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        status["numpy"] = True
    except ImportError:
        pass

    # 4. PyTorch (guarded)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        status["torch"] = True
    except ImportError:
        pass

    # 5. TensorFlow (guarded)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
        status["tensorflow"] = True
    except ImportError:
        pass

    return status


if __name__ == "__main__":
    res = set_global_seed(42)
    print(f"Global seed 42 set. Framework status: {res}")
