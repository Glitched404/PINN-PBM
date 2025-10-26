"""
Helper Functions for PINN-PBM.

General-purpose utility functions used across all problem types.
Includes TensorFlow-based numerical methods and loss functions.
"""

from typing import Union
import numpy as np
import tensorflow as tf

try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    tfp = None
    TFP_AVAILABLE = False


@tf.function
def trapz_tf(y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    """TensorFlow implementation of trapezoidal integration.
    
    Computes the definite integral using the trapezoidal rule:
    ∫ y(x) dx ≈ Σ 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    
    Args:
        y: Function values at grid points (1D tensor)
        x: Grid points (1D tensor, same length as y)
        
    Returns:
        Scalar tensor containing the integral value
        
    Note:
        If x has fewer than 2 points, returns 0.0
        
    Example:
        >>> x = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
        >>> y = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
        >>> result = trapz_tf(y, x)
        >>> # Returns: 2.0 (area under y=x from 0 to 2)
    """
    n = tf.size(x)
    
    def no_integration():
        return tf.constant(0.0, dtype=tf.float32)
    
    def do_integration():
        # Trapezoidal rule: 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
        y_avg = 0.5 * (y[:-1] + y[1:])
        dx = x[1:] - x[:-1]
        return tf.reduce_sum(y_avg * dx)
    
    return tf.cond(n < 2, no_integration, do_integration)


@tf.function
def tail_trapz(g: tf.Tensor, v_grid: tf.Tensor) -> tf.Tensor:
    """Compute tail integral from each point to infinity using trapezoidal rule.
    
    For a grid of points, computes the integral from each point v_i to infinity:
    TI[i] = ∫_{v_i}^{∞} g(v) dv
    
    This is useful for birth terms in population balance equations where
    particles at volume v can only be formed from particles at volumes > v.
    
    Args:
        g: Function values on grid (can be batched: [batch, n_points])
        v_grid: Volume grid points (1D tensor)
        
    Returns:
        Tail integrals at each grid point (same shape as g)
        
    Note:
        Uses reverse cumulative sum for efficient computation
        
    Example:
        >>> v_grid = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
        >>> g = tf.constant([4.0, 3.0, 2.0, 1.0], dtype=tf.float32)
        >>> TI = tail_trapz(g, v_grid)
        >>> # TI[0] = integral from v=1 to v=4
        >>> # TI[1] = integral from v=2 to v=4
        >>> # TI[2] = integral from v=3 to v=4
        >>> # TI[3] = 0 (no points beyond)
    """
    # Compute segment widths
    dv = v_grid[1:] - v_grid[:-1]
    
    # Average function values for each segment
    g_avg = 0.5 * (g[..., :-1] + g[..., 1:])
    
    # Segment contributions
    seg = g_avg * dv
    
    # Reverse cumulative sum to get tail integrals
    rev = tf.reverse(seg, axis=[-1])
    rev_cumsum = tf.cumsum(rev, axis=-1)
    rev_cum = tf.reverse(rev_cumsum, axis=[-1])
    
    # Append zero for the last point (no tail)
    TI = tf.concat([rev_cum, tf.zeros_like(rev_cum[..., :1])], axis=-1)
    
    return TI


@tf.function
def huber_loss(residuals: tf.Tensor, delta: float = 0.1) -> tf.Tensor:
    """Huber loss function for robust residual handling.
    
    The Huber loss is quadratic for small residuals (< delta) and linear
    for large residuals (> delta), providing robustness against outliers.
    
    L(r) = {
        0.5 * r^2                    if |r| <= delta
        delta * (|r| - 0.5 * delta)  if |r| > delta
    }
    
    Args:
        residuals: Residual values (any shape)
        delta: Threshold for quadratic vs linear behavior (default: 0.1)
        
    Returns:
        Huber loss values (same shape as residuals)
        
    Note:
        - Quadratic for small residuals: more sensitive to small errors
        - Linear for large residuals: less sensitive to outliers
        
    Example:
        >>> residuals = tf.constant([-2.0, -0.05, 0.0, 0.05, 2.0])
        >>> loss = huber_loss(residuals, delta=0.1)
        >>> # Small residuals get quadratic loss
        >>> # Large residuals get linear loss
    """
    abs_res = tf.abs(residuals)
    
    # Quadratic part: min(|r|, delta)
    quadratic = tf.minimum(abs_res, delta)
    
    # Linear part: |r| - delta (for |r| > delta)
    linear = abs_res - quadratic
    
    # Combined Huber loss
    return 0.5 * quadratic**2 + delta * linear


@tf.function
def percentile_clip_loss(
    residuals: tf.Tensor,
    weights: tf.Tensor,
    percentile: float = 95.0
) -> tf.Tensor:
    """Compute loss with outlier rejection via percentile clipping.
    
    Ignores the worst residuals by clipping weighted squared residuals
    at a specified percentile, making the loss robust to outliers.
    
    Args:
        residuals: Residual values (1D tensor)
        weights: Weight for each residual (same shape as residuals)
        percentile: Percentile threshold for clipping (default: 95.0)
                   Values above this percentile are clipped
        
    Returns:
        Scalar loss value (mean of clipped weighted squared residuals)
        
    Raises:
        RuntimeError: If TensorFlow Probability is not available
        
    Note:
        Requires TensorFlow Probability for percentile computation
        
    Example:
        >>> residuals = tf.constant([0.1, 0.2, 0.3, 10.0])  # One outlier
        >>> weights = tf.ones(4)
        >>> loss = percentile_clip_loss(residuals, weights, percentile=75.0)
        >>> # Large residual (10.0) is clipped, reducing its influence
    """
    if not TFP_AVAILABLE:
        raise RuntimeError(
            "percentile_clip_loss requires TensorFlow Probability. "
            "Install with: pip install tensorflow-probability"
        )
    
    # Compute weighted squared residuals
    weighted_sq = weights * tf.square(residuals)
    
    # Find percentile threshold
    threshold = tfp.stats.percentile(weighted_sq, percentile)
    
    # Clip to threshold (ignore worst outliers)
    clipped = tf.minimum(weighted_sq, threshold)
    
    return tf.reduce_mean(clipped)


def delta_peak(
    x_grid: Union[np.ndarray, list],
    center: float,
    area: float,
    width_fraction: float = 0.01
) -> np.ndarray:
    """Approximate a delta function as a narrow Gaussian peak.
    
    Creates a smooth approximation of δ(x - center) with specified area,
    useful for representing discrete events in continuous distributions
    (e.g., particles at a specific size in population balance models).
    
    Args:
        x_grid: Grid points where to evaluate the peak
        center: Center location of the delta peak
        area: Total area under the peak (integral)
        width_fraction: Peak width as fraction of center (default: 0.01 = 1%)
        
    Returns:
        Array of peak values at each grid point
        
    Note:
        - Uses Gaussian: A * exp(-0.5 * ((x - center) / width)^2)
        - Height A is chosen to give specified area
        - Minimum width is ensured for numerical stability
        
    Example:
        >>> x = np.linspace(0, 2, 100)
        >>> peak = delta_peak(x, center=1.0, area=0.5, width_fraction=0.01)
        >>> # Creates narrow peak at x=1.0 with area ≈ 0.5
        >>> np.trapz(peak, x)  # Should be close to 0.5
    """
    x_grid_np = np.array(x_grid)
    
    # Compute peak width
    width = center * width_fraction
    
    # Ensure minimum width for numerical stability
    if width < 1e-9:
        positive_x = x_grid_np[x_grid_np > 0]
        if len(positive_x) > 0:
            width = np.min(positive_x) * width_fraction
        else:
            width = 1e-9
    
    # Compute height to achieve desired area
    # For Gaussian: area = height * width * sqrt(2*pi)
    height = area / (width * np.sqrt(2 * np.pi) + 1e-30)
    
    # Create Gaussian peak
    peak = height * np.exp(-0.5 * ((x_grid_np - center) / width) ** 2)
    
    return peak


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - TensorFlow
    
    Args:
        seed: Random seed value (default: 42)
        
    Example:
        >>> set_random_seed(42)
        >>> # Now all random operations are reproducible
    """
    import os
    import random
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu_memory_growth() -> None:
    """Configure TensorFlow to allocate GPU memory as needed.
    
    Prevents TensorFlow from allocating all GPU memory at startup,
    allowing multiple processes to share GPU resources.
    
    Note:
        Should be called at the start of your program, before any
        TensorFlow operations are executed.
        
    Example:
        >>> configure_gpu_memory_growth()
        >>> # Now TensorFlow will allocate GPU memory incrementally
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory growth configuration failed: {e}")


def check_tensorflow_probability() -> bool:
    """Check if TensorFlow Probability is available.
    
    Returns:
        True if TFP is available, False otherwise
        
    Example:
        >>> if check_tensorflow_probability():
        ...     # Use TFP features
        ...     pass
        ... else:
        ...     print("Install TFP: pip install tensorflow-probability")
    """
    return TFP_AVAILABLE


__all__ = [
    'trapz_tf',
    'tail_trapz',
    'huber_loss',
    'percentile_clip_loss',
    'delta_peak',
    'set_random_seed',
    'configure_gpu_memory_growth',
    'check_tensorflow_probability',
    'TFP_AVAILABLE'
]
