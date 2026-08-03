"""PyTorch interface for the mHC CUDA extension."""

import torch as _torch  # noqa: F401  # Load PyTorch before the extension.

from . import _internal
from .module import MHCProjectionN4, MHCProjectionN8, MHCSinkhornN4, MHCSinkhornN8

torch = _internal.torch
__version__ = "0.1.0"

__all__ = [
    "MHCProjectionN4",
    "MHCProjectionN8",
    "MHCSinkhornN4",
    "MHCSinkhornN8",
    "torch",
]
