"""
Analytical solutions for breakage test cases.

Provides exact solutions for validation:
- Case 1: Linear selection
- Case 2: Quadratic selection
- Case 3: Linear selection with delta peak
- Case 4: Quadratic selection with delta peak
"""

# Analytical solutions (Step 6 - Complete)
from .analytical import (
    analytic_f_case1,
    analytic_f_case2,
    analytic_f_case3,
    analytic_f_case4,
    get_analytical_solution,
    validate_analytical_solution
)

__all__ = [
    'analytic_f_case1',
    'analytic_f_case2',
    'analytic_f_case3',
    'analytic_f_case4',
    'get_analytical_solution',
    'validate_analytical_solution'
]
