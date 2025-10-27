"""
Optimizer Wrappers for PINN-PBM.

Provides wrappers for L-BFGS optimization using both scipy and TensorFlow Probability,
enabling second-order fine-tuning after first-order (ADAM) training.
"""

from typing import List, Tuple, Any
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from tqdm import tqdm

try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    tfp = None
    TFP_AVAILABLE = False


def lbfgs_optimizer_scipy(
    model: tf.keras.Model,
    loss_fn: callable,
    initial_weights: List[tf.Variable],
    max_iter: int = 1000,
    verbose: bool = True,
    callback: callable = None
) -> Any:
    """L-BFGS optimizer using scipy.optimize.minimize.
    
    Wraps scipy's L-BFGS-B implementation for use with TensorFlow models.
    This is useful for fine-tuning after ADAM training, as L-BFGS can
    achieve better convergence using second-order information.
    
    Args:
        model: TensorFlow/Keras model to optimize
        loss_fn: Function that computes loss (should return scalar tensor)
        initial_weights: List of model variables to optimize
        max_iter: Maximum number of L-BFGS iterations (default: 1000)
        verbose: Whether to show progress bar (default: True)
        callback: Optional callback function called after each iteration
                 Signature: callback(current_weights) -> None
        
    Returns:
        scipy.optimize.OptimizeResult object with optimization details
        
    Note:
        - Model weights are modified in-place
        - Loss function is called many times for line search
        - Uses float64 for scipy compatibility
        
    Example:
        >>> def loss_fn():
        ...     predictions = model(x_train)
        ...     return tf.reduce_mean(tf.square(predictions - y_train))
        >>> 
        >>> result = lbfgs_optimizer_scipy(
        ...     model=pinn.model,
        ...     loss_fn=loss_fn,
        ...     initial_weights=pinn.model.trainable_variables,
        ...     max_iter=1000
        ... )
    """
    # Get shapes and flatten initial weights
    shapes = [v.shape.as_list() for v in initial_weights]
    w0 = np.concatenate([v.numpy().flatten() for v in initial_weights]).astype(np.float64)
    
    def set_weights(w_flat: np.ndarray) -> None:
        """Assign flattened weights back to model variables."""
        idx = 0
        for var, shape in zip(initial_weights, shapes):
            size = int(np.prod(shape))
            var.assign(tf.reshape(
                tf.convert_to_tensor(w_flat[idx:idx+size], dtype=tf.float32),
                shape
            ))
            idx += size
    
    def loss_and_grad(w_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute loss and gradients for scipy."""
        set_weights(w_flat)
        
        with tf.GradientTape() as tape:
            loss = loss_fn()
        
        grads = tape.gradient(loss, initial_weights)
        
        # Handle None gradients (shouldn't happen but be safe)
        safe_grads = [
            (g if g is not None else tf.zeros_like(v))
            for g, v in zip(grads, initial_weights)
        ]
        
        grad_flat = np.concatenate([
            g.numpy().flatten() for g in safe_grads
        ]).astype(np.float64)
        
        return float(loss.numpy().astype(np.float64)), grad_flat
    
    # Progress bar
    pbar = tqdm(total=max_iter, desc="L-BFGS (scipy)", disable=not verbose)
    
    def lbfgs_callback(xk: np.ndarray) -> None:
        """Callback for progress updates."""
        pbar.update(1)
        if callback is not None:
            callback(xk)
    
    # Run optimization
    result = minimize(
        fun=loss_and_grad,
        x0=w0,
        jac=True,
        method='L-BFGS-B',
        callback=lbfgs_callback,
        options={
            'maxiter': max_iter,
            'disp': False,
            'gtol': 0.0,
            'ftol': 0.0,
            'maxfun': int(1e9)
        }
    )
    
    pbar.close()
    
    # Set final weights
    set_weights(result.x)
    
    return result


def lbfgs_optimizer_tfp(
    model: tf.keras.Model,
    loss_fn: callable,
    initial_weights: List[tf.Variable],
    max_iter: int = 3000,
    tolerance: float = 1e-12,
    verbose: bool = True,
    fallback_to_scipy: bool = True,
    line_search_iterations: int = 50,
) -> Any:
    """L-BFGS optimizer using TensorFlow Probability.
    
    Wraps TFP's L-BFGS implementation for use with TensorFlow models.
    This version stays entirely in TensorFlow's computation graph,
    which can be faster than scipy version.
    
    Args:
        model: TensorFlow/Keras model to optimize
        loss_fn: Function that computes loss (should return scalar tensor)
        initial_weights: List of model variables to optimize
        max_iter: Maximum number of L-BFGS iterations (default: 1500)
        tolerance: Convergence tolerance (default: 1e-7)
        verbose: Whether to print progress (default: True)
        
    Returns:
        TFP OptimizeResults object with optimization details
        
    Raises:
        RuntimeError: If TensorFlow Probability is not installed
        
    Note:
        - Requires tensorflow-probability package
        - Model weights are modified in-place
        - Stays in TensorFlow graph (can be faster than scipy)
        - Uses float32 throughout
        
    Example:
        >>> def loss_fn():
        ...     predictions = model(x_train)
        ...     return tf.reduce_mean(tf.square(predictions - y_train))
        >>> 
        >>> result = lbfgs_optimizer_tfp(
        ...     model=pinn.model,
        ...     loss_fn=loss_fn,
        ...     initial_weights=pinn.model.trainable_variables,
        ...     max_iter=1500
        ... )
    """
    if not TFP_AVAILABLE:
        raise RuntimeError(
            "lbfgs_optimizer_tfp requires TensorFlow Probability. "
            "Install with: pip install tensorflow-probability"
        )
    
    # Compute shapes and sizes for flattening/unflattening
    shapes = [v.shape for v in initial_weights]
    sizes = [int(np.prod(s)) for s in shapes]
    idxs = np.cumsum([0] + sizes)
    
    def pack(vars_: List[tf.Variable]) -> tf.Tensor:
        """Flatten variable list into 1D tensor."""
        return tf.concat([tf.reshape(v, [-1]) for v in vars_], axis=0)
    
    def unpack(w: tf.Tensor) -> List[tf.Tensor]:
        """Unflatten 1D tensor back into variable shapes."""
        return [
            tf.reshape(w[idxs[i]:idxs[i+1]], shapes[i])
            for i in range(len(initial_weights))
        ]
    
    @tf.function
    def value_and_gradients(w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute loss and gradients for TFP."""
        # Unpack and assign weights
        ws = unpack(w)
        for var, new_val in zip(initial_weights, ws):
            var.assign(tf.cast(new_val, var.dtype))
        
        # Compute loss and gradients
        with tf.GradientTape() as tape:
            loss = loss_fn()
            
            # Clamp non-finite values
            loss = tf.where(
                tf.math.is_finite(loss),
                loss,
                tf.constant(1e10, dtype=tf.float32)
            )
        
        grads = tape.gradient(loss, initial_weights)
        
        # Ensure all gradients are safe
        safe_grads = []
        for g, v in zip(grads, initial_weights):
            if g is None:
                g = tf.zeros_like(v)
            else:
                g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            safe_grads.append(g)
        
        # Flatten gradients
        grad_vector = pack(safe_grads)
        grad_vector = tf.where(
            tf.math.is_finite(grad_vector),
            grad_vector,
            tf.zeros_like(grad_vector)
        )
        
        # Return with consistent dtype
        return tf.cast(loss, w.dtype), tf.cast(grad_vector, w.dtype)
    
    # Flatten initial weights
    w0 = pack(initial_weights)
    
    if verbose:
        print(f"\nStarting TFP L-BFGS with {len(w0)} parameters...")
    
    # Run L-BFGS optimization
    try:
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=value_and_gradients,
            initial_position=w0,
            max_iterations=max_iter,
            tolerance=tolerance,
            f_relative_tolerance=1e-12,
            x_tolerance=1e-15,
            initial_inverse_hessian_estimate=None,
            max_line_search_iterations=line_search_iterations,
            parallel_iterations=1,
        )
    except Exception as exc:  # pragma: no cover - fallback path
        if not fallback_to_scipy:
            raise
        if verbose:
            print("TFP L-BFGS failed; falling back to scipy due to:", exc)
        return lbfgs_optimizer_scipy(
            model=model,
            loss_fn=loss_fn,
            initial_weights=initial_weights,
            max_iter=max_iter,
            verbose=verbose,
        )
    
    # Unpack and assign final weights
    final_vars = unpack(results.position)
    for var, new_val in zip(initial_weights, final_vars):
        var.assign(tf.cast(new_val, var.dtype))
    
    if verbose:
        print(
            f"L-BFGS Complete: "
            f"Converged={bool(results.converged.numpy())}, "
            f"Iterations={int(results.num_iterations.numpy())}"
        )
    
    if not bool(results.converged.numpy()) and fallback_to_scipy and SCIPY_AVAILABLE:
        if verbose:
            print("TFP L-BFGS did not converge; retrying with scipy backend.")
        return lbfgs_optimizer_scipy(
            model=model,
            loss_fn=loss_fn,
            initial_weights=initial_weights,
            max_iter=max_iter,
            verbose=verbose,
        )

    return results


def check_tfp_availability() -> bool:
    """Check if TensorFlow Probability is available.
    
    Returns:
        True if TFP is available, False otherwise
        
    Example:
        >>> if check_tfp_availability():
        ...     optimizer_fn = lbfgs_optimizer_tfp
        ... else:
        ...     optimizer_fn = lbfgs_optimizer_scipy
    """
    return TFP_AVAILABLE


__all__ = [
    'lbfgs_optimizer_scipy',
    'lbfgs_optimizer_tfp',
    'check_tfp_availability',
    'TFP_AVAILABLE'
]
