"""
Physics Kernels for Breakage Population Balance Equations.

Provides selection functions S(v) and breakage distribution β(v,v') used
in the breakage PBE:

∂f/∂t = Birth(v,t) - Death(v,t)

where:
Death(v,t) = S(v) * f(v,t)
Birth(v,t) = ∫_v^∞ β(v,v') * S(v') * f(v',t) dv'
"""

from typing import Union, Callable
import tensorflow as tf


def selection_linear(v: tf.Tensor) -> tf.Tensor:
    """Linear selection function: S(v) = v.
    
    This represents a breakage rate proportional to particle volume.
    Larger particles break more frequently.
    
    Args:
        v: Volume values (tensor)
        
    Returns:
        Selection function values S(v) = v
        
    Used in:
        - Case 1: Linear selection without delta peak
        - Case 3: Linear selection with delta peak
        
    Example:
        >>> v = tf.constant([1.0, 2.0, 3.0])
        >>> S = selection_linear(v)
        >>> # Returns: [1.0, 2.0, 3.0]
    """
    return v


def selection_quadratic(v: tf.Tensor) -> tf.Tensor:
    """Quadratic selection function: S(v) = v².
    
    This represents a breakage rate proportional to particle surface area
    (assuming spherical particles). Larger particles break much more frequently.
    
    Args:
        v: Volume values (tensor)
        
    Returns:
        Selection function values S(v) = v²
        
    Used in:
        - Case 2: Quadratic selection without delta peak
        - Case 4: Quadratic selection with delta peak
        
    Example:
        >>> v = tf.constant([1.0, 2.0, 3.0])
        >>> S = selection_quadratic(v)
        >>> # Returns: [1.0, 4.0, 9.0]
    """
    return tf.square(v)


def breakage_symmetric(v: tf.Tensor, vp: tf.Tensor) -> tf.Tensor:
    """Symmetric binary breakage distribution: β(v,v') = 2/v'.
    
    This represents equal probability of forming any daughter particle
    size from breakage of a parent particle. The factor of 2 accounts
    for the two daughter particles formed from each breakage event.
    
    Physical interpretation:
    - When a particle of volume v' breaks, it can form any combination
      of daughter particles whose volumes sum to v'
    - This kernel gives equal probability to all breakage combinations
    - Conserves mass: ∫_0^v' v*β(v,v')dv = v'
    
    Args:
        v: Daughter particle volumes (tensor)
        vp: Parent particle volumes (tensor), must have vp >= v
        
    Returns:
        Breakage distribution β(v,v') = 2/v'
        
    Note:
        This is the only breakage distribution used in all four test cases.
        
    Example:
        >>> v = tf.constant([0.5, 1.0])  # Daughter volumes
        >>> vp = tf.constant([2.0, 3.0])  # Parent volumes
        >>> beta = breakage_symmetric(v, vp)
        >>> # Returns: [1.0, 0.667]
    """
    return 2.0 / vp


def get_selection_function(case_type: str) -> Callable:
    """Get selection function for specified case type.
    
    Args:
        case_type: Case identifier ('case1', 'case2', 'case3', or 'case4')
        
    Returns:
        Selection function callable: S(v) -> tf.Tensor
        
    Raises:
        ValueError: If case_type is not recognized
        
    Mapping:
        - case1, case3: Linear selection S(v) = v
        - case2, case4: Quadratic selection S(v) = v²
        
    Example:
        >>> selection_fn = get_selection_function('case1')
        >>> v = tf.constant([1.0, 2.0, 3.0])
        >>> S = selection_fn(v)
    """
    selection_map = {
        'case1': selection_linear,
        'case2': selection_quadratic,
        'case3': selection_linear,
        'case4': selection_quadratic
    }
    
    if case_type not in selection_map:
        raise ValueError(
            f"Unsupported case type: {case_type}. "
            f"Must be one of: {list(selection_map.keys())}"
        )
    
    return selection_map[case_type]


def get_breakage_distribution(case_type: str = None) -> Callable:
    """Get breakage distribution function.
    
    Args:
        case_type: Case identifier (optional, currently unused since
                  all cases use the same distribution)
        
    Returns:
        Breakage distribution callable: β(v, v') -> tf.Tensor
        
    Note:
        All current test cases use the symmetric binary distribution
        β(v,v') = 2/v'. This function is provided for API consistency
        and future extensibility to other breakage distributions.
        
    Example:
        >>> beta_fn = get_breakage_distribution('case1')
        >>> v = tf.constant([1.0])
        >>> vp = tf.constant([2.0])
        >>> beta = beta_fn(v, vp)
    """
    # Currently all cases use symmetric breakage
    # This can be extended in the future for other distributions
    return breakage_symmetric


def validate_selection_function(
    selection_fn: Callable,
    v_test: tf.Tensor
) -> dict:
    """Validate selection function properties.
    
    Checks:
    - Non-negativity: S(v) >= 0 for all v > 0
    - Monotonicity: S(v) increases with v (for physical cases)
    - Finite values: No NaN or Inf
    
    Args:
        selection_fn: Selection function to validate
        v_test: Test volume values (should be > 0)
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> v_test = tf.constant([0.1, 1.0, 10.0])
        >>> result = validate_selection_function(selection_linear, v_test)
        >>> if result['valid']:
        ...     print("Selection function is valid")
    """
    results = {
        'valid': True,
        'errors': []
    }
    
    # Compute selection values
    S = selection_fn(v_test)
    
    # Check non-negativity
    if tf.reduce_any(S < 0):
        results['valid'] = False
        results['errors'].append('Negative values detected')
    
    # Check for finite values
    if not tf.reduce_all(tf.math.is_finite(S)):
        results['valid'] = False
        results['errors'].append('Non-finite values (NaN or Inf) detected')
    
    # Check monotonicity (should increase with v)
    if len(v_test) > 1:
        dS = S[1:] - S[:-1]
        if tf.reduce_any(dS < 0):
            results['errors'].append('Non-monotonic (not always increasing)')
    
    return results


def validate_breakage_distribution(
    breakage_fn: Callable,
    v_test: tf.Tensor,
    vp_test: tf.Tensor
) -> dict:
    """Validate breakage distribution properties.
    
    Checks:
    - Non-negativity: β(v,v') >= 0
    - Domain: β(v,v') = 0 for v > v' (can't form daughter larger than parent)
    - Finite values: No NaN or Inf
    
    Args:
        breakage_fn: Breakage distribution function to validate
        v_test: Daughter volume test values
        vp_test: Parent volume test values
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> v = tf.constant([0.5, 1.0])
        >>> vp = tf.constant([2.0, 2.0])
        >>> result = validate_breakage_distribution(breakage_symmetric, v, vp)
    """
    results = {
        'valid': True,
        'errors': []
    }
    
    # Compute breakage distribution
    beta = breakage_fn(v_test, vp_test)
    
    # Check non-negativity
    if tf.reduce_any(beta < 0):
        results['valid'] = False
        results['errors'].append('Negative values detected')
    
    # Check for finite values
    if not tf.reduce_all(tf.math.is_finite(beta)):
        results['valid'] = False
        results['errors'].append('Non-finite values detected')
    
    # Check domain constraint: should be very small or zero when v > vp
    invalid_domain = tf.logical_and(v_test > vp_test, beta > 1e-10)
    if tf.reduce_any(invalid_domain):
        results['errors'].append('Non-zero values for v > vp (unphysical)')
    
    return results


__all__ = [
    'selection_linear',
    'selection_quadratic',
    'breakage_symmetric',
    'get_selection_function',
    'get_breakage_distribution',
    'validate_selection_function',
    'validate_breakage_distribution'
]
