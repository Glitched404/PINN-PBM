"""
Analytical Solutions for Breakage Test Cases.

Provides exact analytical solutions for validation of PINN predictions.
All cases use selection functions S(v) and breakage distribution β(v,v') = 2/v'.

Cases:
1. Linear selection: S(v) = v
2. Quadratic selection: S(v) = v²
3. Linear selection with delta peak: S(v) = v, discontinuity at Rx
4. Quadratic selection with delta peak: S(v) = v², discontinuity at Rx
"""

from typing import Union
import numpy as np
from pinn_pbm.core.utils import delta_peak


def analytic_f_case1(v: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
    """Analytical solution for Case 1: Linear selection function.
    
    Solves the breakage PBE with:
    - Selection function: S(v) = v (linear)
    - Breakage distribution: β(v,v') = 2/v'
    - Initial condition: f(v,0) = exp(-v)
    
    Solution:
        f(v,t) = exp(-v(1+t)) * (1+t)²
    
    Args:
        v: Volume values (1D array)
        t: Time value or array
        
    Returns:
        Number density f(v,t) at given volumes and time(s)
        
    Reference:
        Standard breakage equation with linear selection kernel
        
    Example:
        >>> v = np.linspace(0.001, 10, 100)
        >>> t = 5.0
        >>> f = analytic_f_case1(v, t)
    """
    return np.exp(-v * (1.0 + t)) * (1.0 + t) ** 2 + 1e-30


def analytic_f_case2(v: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
    """Analytical solution for Case 2: Quadratic selection function.
    
    Solves the breakage PBE with:
    - Selection function: S(v) = v² (quadratic)
    - Breakage distribution: β(v,v') = 2/v'
    - Initial condition: f(v,0) = exp(-v)
    
    Solution:
        f(v,t) = exp(-t*v² - v) * (1 + 2t(1+v))
    
    Args:
        v: Volume values (1D array)
        t: Time value or array
        
    Returns:
        Number density f(v,t) at given volumes and time(s)
        
    Reference:
        Breakage equation with quadratic selection kernel
        
    Example:
        >>> v = np.linspace(0.001, 100, 100)
        >>> t = 2.0
        >>> f = analytic_f_case2(v, t)
    """
    return np.exp(-t * v ** 2 - v) * (1.0 + 2.0 * t * (1.0 + v)) + 1e-30


def analytic_f_case3(v: np.ndarray, t: Union[float, np.ndarray], Rx: float = 1.0) -> np.ndarray:
    """Analytical solution for Case 3: Linear selection with delta peak.
    
    Solves the breakage PBE with:
    - Selection function: S(v) = v (linear)
    - Breakage distribution: β(v,v') = 2/v'
    - Initial condition: f(v,0) = δ(v - Rx) where Rx is a reference volume
    
    Solution has two parts:
    1. Delta peak at v = Rx with area exp(-t*Rx)
    2. Continuous distribution for v < Rx
    
    For v < Rx:
        f_continuous(v,t) = exp(-t*v) * (2t + t²(Rx - v))
    
    Delta peak:
        f_delta(v,t) = exp(-t*Rx) * δ(v - Rx)
    
    Args:
        v: Volume values (1D array)
        t: Time value or array
        Rx: Reference volume for initial delta peak (default: 1.0)
        
    Returns:
        Number density f(v,t) at given volumes and time(s)
        
    Note:
        The delta function is approximated as a narrow Gaussian peak
        using the delta_peak() function.
        
    Reference:
        Breakage with monodisperse initial condition
        
    Example:
        >>> v = np.linspace(0.0001, 2, 100)
        >>> t = 100.0
        >>> f = analytic_f_case3(v, t, Rx=1.0)
    """
    # Heaviside function: 1 for v < Rx, 0 otherwise
    heaviside = (v < Rx).astype(float)
    
    # Delta peak at Rx with decaying area
    area_delta = np.exp(-t * Rx)
    delta_part = delta_peak(v, Rx, area_delta)
    
    # Continuous part (only for v < Rx)
    continuous_part = np.exp(-t * v) * (2.0 * t + t ** 2 * (Rx - v)) * heaviside
    
    return delta_part + continuous_part + 1e-30


def analytic_f_case4(v: np.ndarray, t: Union[float, np.ndarray], Rx: float = 1.0) -> np.ndarray:
    """Analytical solution for Case 4: Quadratic selection with delta peak.
    
    Solves the breakage PBE with:
    - Selection function: S(v) = v² (quadratic)
    - Breakage distribution: β(v,v') = 2/v'
    - Initial condition: f(v,0) = δ(v - Rx) where Rx is a reference volume
    
    Solution has two parts:
    1. Delta peak at v = Rx with area exp(-t*Rx²)
    2. Continuous distribution for v < Rx
    
    For v < Rx:
        f_continuous(v,t) = exp(-t*v²) * 2t*Rx
    
    Delta peak:
        f_delta(v,t) = exp(-t*Rx²) * δ(v - Rx)
    
    Args:
        v: Volume values (1D array)
        t: Time value or array
        Rx: Reference volume for initial delta peak (default: 1.0)
        
    Returns:
        Number density f(v,t) at given volumes and time(s)
        
    Note:
        The delta function is approximated as a narrow Gaussian peak
        using the delta_peak() function.
        
    Reference:
        Breakage with monodisperse initial condition and quadratic kernel
        
    Example:
        >>> v = np.linspace(0.0001, 1.5, 100)
        >>> t = 500.0
        >>> f = analytic_f_case4(v, t, Rx=1.0)
    """
    # Heaviside function: 1 for v < Rx, 0 otherwise
    heaviside = (v < Rx).astype(float)
    
    # Delta peak at Rx with decaying area
    area_delta = np.exp(-t * Rx ** 2)
    delta_part = delta_peak(v, Rx, area_delta)
    
    # Continuous part (only for v < Rx)
    continuous_part = np.exp(-t * v ** 2) * (2.0 * t * Rx) * heaviside
    
    return delta_part + continuous_part + 1e-30


def get_analytical_solution(
    v: np.ndarray,
    t: Union[float, np.ndarray],
    case_type: str,
    Rx: float = 1.0
) -> np.ndarray:
    """Get analytical solution for specified breakage case.
    
    Convenience function that dispatches to the appropriate case-specific
    analytical solution function.
    
    Args:
        v: Volume values (1D array)
        t: Time value or array
        case_type: Case identifier ('case1', 'case2', 'case3', or 'case4')
        Rx: Reference volume for cases 3 and 4 (default: 1.0)
        
    Returns:
        Number density f(v,t) at given volumes and time(s)
        
    Raises:
        ValueError: If case_type is not recognized
        
    Example:
        >>> v = np.linspace(0.001, 10, 100)
        >>> t = 5.0
        >>> f = get_analytical_solution(v, t, 'case1')
        >>> # Equivalent to: f = analytic_f_case1(v, t)
    """
    case_functions = {
        'case1': analytic_f_case1,
        'case2': analytic_f_case2,
        'case3': lambda v, t: analytic_f_case3(v, t, Rx),
        'case4': lambda v, t: analytic_f_case4(v, t, Rx)
    }
    
    if case_type not in case_functions:
        raise ValueError(
            f"Unsupported case type: {case_type}. "
            f"Must be one of: {list(case_functions.keys())}"
        )
    
    return case_functions[case_type](v, t)


def validate_analytical_solution(
    v: np.ndarray,
    t: float,
    f_analytical: np.ndarray,
    case_type: str
) -> dict:
    """Validate analytical solution satisfies basic physical constraints.
    
    Checks:
    - Non-negativity: f(v,t) >= 0 for all v
    - Smoothness: No NaN or Inf values
    - Conservation: Total number is finite and positive
    
    Args:
        v: Volume grid
        t: Time value
        f_analytical: Analytical solution values
        case_type: Case identifier
        
    Returns:
        Dictionary with validation results and diagnostics
        
    Example:
        >>> v = np.linspace(0.001, 10, 100)
        >>> t = 5.0
        >>> f = get_analytical_solution(v, t, 'case1')
        >>> results = validate_analytical_solution(v, t, f, 'case1')
        >>> if results['valid']:
        ...     print("Solution is physically valid")
    """
    results = {
        'valid': True,
        'case_type': case_type,
        'time': t,
        'errors': []
    }
    
    # Check non-negativity
    if np.any(f_analytical < 0):
        results['valid'] = False
        results['errors'].append('Negative values detected')
    
    # Check for NaN or Inf
    if not np.all(np.isfinite(f_analytical)):
        results['valid'] = False
        results['errors'].append('Non-finite values (NaN or Inf) detected')
    
    # Check conservation (finite total)
    total = np.trapz(f_analytical, v)
    if not np.isfinite(total) or total <= 0:
        results['valid'] = False
        results['errors'].append(f'Invalid total number: {total}')
    else:
        results['total_number'] = total
    
    # Case-specific checks
    if case_type in ['case3', 'case4']:
        # Check for peak near Rx = 1.0
        peak_idx = np.argmax(f_analytical)
        peak_location = v[peak_idx]
        if not (0.9 < peak_location < 1.1):
            results['errors'].append(
                f'Peak location {peak_location} not near Rx=1.0'
            )
        results['peak_location'] = peak_location
        results['peak_value'] = f_analytical[peak_idx]
    
    return results


__all__ = [
    'analytic_f_case1',
    'analytic_f_case2',
    'analytic_f_case3',
    'analytic_f_case4',
    'get_analytical_solution',
    'validate_analytical_solution'
]
