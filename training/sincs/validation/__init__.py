"""
SINCS Validation Framework

Tier 1: Physics-Preserving Validation
- Wave speed consistency
- Acoustic impedance at boundaries
- Energy conservation
- Frequency spectrum preservation
- Boundary condition adherence
- Pressure-velocity phase relationship

Tier 2: FNO Surrogate Model Comparison
- Train FNO on raw vs compressed data
- Compare downstream ML performance
- Evaluate compression acceptability
"""

from .physics_validation import PhysicsValidator, load_raw_data
from .fno_surrogate import FNO2d, PhysicsDataset, run_comparison

__all__ = [
    'PhysicsValidator',
    'load_raw_data',
    'FNO2d',
    'PhysicsDataset',
    'run_comparison',
]
