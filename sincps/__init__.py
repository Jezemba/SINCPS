"""
SINCPS: Semantic-aware Implicit Neural Compression for Physics Simulations

A compression framework achieving 150x to 25,000x ratios across diverse physics
domains while preserving conservation laws.
"""

from .model import SINCPS, SirenLayer, FourierEncoder
from .decompress import (
    load_checkpoint,
    load_model,
    reconstruct,
    denormalize,
    SINCPSDecompressor,
)

__version__ = "1.0.0"
__all__ = [
    "SINCPS",
    "SirenLayer",
    "FourierEncoder",
    "load_checkpoint",
    "load_model",
    "reconstruct",
    "denormalize",
    "SINCPSDecompressor",
]
