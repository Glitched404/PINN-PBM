"""
Physics kernels for breakage problems.

Includes:
- Selection functions: S(v) for different cases
- Breakage distributions: β(v,v')
- PDE residual computations
"""

# Physics kernels (Step 7 - Complete)
from .kernels import (
    selection_linear,
    selection_quadratic,
    breakage_symmetric,
    get_selection_function,
    get_breakage_distribution,
    validate_selection_function,
    validate_breakage_distribution
)

__all__ = [
    'selection_linear',
    'selection_quadratic',
    'breakage_symmetric',
    'get_selection_function',
    'get_breakage_distribution',
    'validate_selection_function',
    'validate_breakage_distribution'
]
