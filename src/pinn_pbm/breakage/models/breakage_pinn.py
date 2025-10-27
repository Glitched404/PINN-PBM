"""
BreakagePINN Model - Refactored Implementation.

Physics-Informed Neural Network for solving breakage population balance equations.
Inherits from BasePINN and uses modular components for physics, solutions, and utilities.
"""

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf

from pinn_pbm.core.models import BasePINN
from pinn_pbm.core.utils import tail_trapz, huber_loss, check_tensorflow_probability
from pinn_pbm.breakage.physics import get_selection_function, breakage_symmetric

# Check if TFP is available for adaptive loss scaling
TFP_AVAILABLE = check_tensorflow_probability()
if TFP_AVAILABLE:
    import tensorflow_probability as tfp
else:
    tfp = None  # type: ignore


class BreakagePINN(BasePINN):
    """Physics-Informed Neural Network for Breakage Problems.
    
    Solves the breakage population balance equation:
        ∂f/∂t = Birth(v,t) - Death(v,t)
    
    where:
        Death(v,t) = S(v) * f(v,t)
        Birth(v,t) = ∫_v^∞ β(v,v') * S(v') * f(v',t) dv'
    
    Supports four test cases with different selection functions and initial conditions.
    
    Attributes:
        case_type: Case identifier ('case1', 'case2', 'case3', 'case4')
        v_min, v_max: Volume domain bounds
        t_min, t_max: Time domain bounds
        loss_scaling: Loss scaling strategy
        model: Main neural network (log f)
        amt: Auxiliary model for delta peak amplitude (cases 3 & 4)
    """
    
    def __init__(
        self,
        v_min: float,
        v_max: float,
        t_min: float,
        t_max: float,
        case_type: str = 'case1',
        n_v: int = 500,
        n_t: int = 100,
        n_hidden_layers: int = 8,
        n_neurons: int = 128,
        loss_scaling: str = 'adaptive_huber',
        **kwargs
    ):
        """Initialize BreakagePINN.
        
        Args:
            v_min: Minimum volume
            v_max: Maximum volume
            t_min: Minimum time
            t_max: Maximum time
            case_type: Case identifier (default: 'case1')
            n_v: Number of volume grid points for integration (default: 500)
            n_t: Number of time grid points (default: 100)
            n_hidden_layers: Number of hidden layers (default: 8)
            n_neurons: Neurons per layer (default: 128)
            loss_scaling: Loss scaling strategy (default: 'adaptive_huber')
            **kwargs: Additional arguments for BasePINN
        """
        super().__init__(
            n_hidden_layers=n_hidden_layers,
            n_neurons=n_neurons,
            **kwargs
        )
        
        self.v_min = v_min
        self.v_max = v_max
        self.t_min = t_min
        self.t_max = t_max
        self.case_type = case_type
        self.loss_scaling = loss_scaling
        
        # Domain normalization constants
        self.log_vmin = tf.constant(np.log(v_min), dtype=tf.float32)
        self.log_vmax = tf.constant(np.log(v_max), dtype=tf.float32)
        self.t_min_tensor = tf.constant(t_min, dtype=tf.float32)
        self.log_t_min_tensor = tf.constant(np.log(t_min + 1e-8), dtype=tf.float32)
        self.log_t_max_tensor = tf.constant(np.log(t_max), dtype=tf.float32)
        
        # Integration grid for birth term
        # Build logarithmically spaced integration grid while avoiding overflow
        ratio = 2.0 ** (1.0 / 3.0)
        max_steps = int(np.ceil(np.log(max(v_max / v_min, 1.0 + 1e-6)) / np.log(ratio))) + 1
        n_v_eff = max_steps if n_v is None else max(2, min(int(n_v), max_steps))
        v_grid = np.logspace(
            np.log10(v_min),
            np.log10(max(v_max, v_min * 1.01)),
            num=n_v_eff,
            dtype=np.float64,
        )
        self.v_grid_tf = tf.constant(v_grid.astype(np.float32))
        
        # Build main model (predicts log(f))
        self.model = self.build_model(input_dim=2, output_dim=1)
        
        # Build auxiliary model for delta peak amplitude (cases 3 & 4)
        self.amt: Optional[tf.keras.Model] = None
        if case_type in ['case3', 'case4']:
            self.amt = self._build_amt_head()
        
        # Get selection function for this case
        self.selection_fn = get_selection_function(case_type)
    
    def _build_amt_head(
        self,
        n_hidden: int = 2,
        width: int = 32
    ) -> tf.keras.Model:
        """Build auxiliary model for delta peak amplitude.
        
        Used in cases 3 and 4 to predict the time-varying amplitude
        of the delta peak at Rx.
        
        Args:
            n_hidden: Number of hidden layers (default: 2)
            width: Layer width (default: 32)
            
        Returns:
            Keras model mapping time -> amplitude
        """
        t_in = tf.keras.Input(shape=(1,), dtype=tf.float32)
        x = t_in
        for _ in range(n_hidden):
            x = tf.keras.layers.Dense(
                width,
                activation='tanh',
                kernel_initializer='glorot_normal',
                kernel_regularizer=tf.keras.regularizers.l2(self.weight_regularization)
            )(x)
        out = tf.keras.layers.Dense(
            1,
            activation='softplus',  # Ensure positive amplitude
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_regularization)
        )(x)
        return tf.keras.Model(t_in, out)
    
    def _normalize_inputs(
        self,
        v: tf.Tensor,
        t: tf.Tensor
    ) -> tf.Tensor:
        """Normalize volume and time to [-1, 1] range.
        
        Uses logarithmic scaling for better numerical stability.
        
        Args:
            v: Volume values
            t: Time values
            
        Returns:
            Normalized inputs stacked as [batch, 2]
        """
        log_v = tf.math.log(v)
        log_t = tf.math.log(t + 1e-8)
        
        # Map to [-1, 1]
        xi = 2.0 * (log_v - self.log_vmin) / (self.log_vmax - self.log_vmin) - 1.0
        tau = 2.0 * (log_t - self.log_t_min_tensor) / (self.log_t_max_tensor - self.log_t_min_tensor) - 1.0
        
        return tf.stack([xi, tau], axis=1)
    
    def net_logf(self, v: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Predict log(f(v,t)) from neural network.
        
        Args:
            v: Volume values
            t: Time values
            
        Returns:
            log(f(v,t)) predictions
        """
        vt_normalized = self._normalize_inputs(v, t)
        return tf.squeeze(self.model(vt_normalized), axis=1)
    
    def net_f(self, v: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        """Predict f(v,t) with delta peak for cases 3 & 4.
        
        Args:
            v: Volume values
            t: Time values
            
        Returns:
            f(v,t) predictions
        """
        # Continuous part from main network
        logf = self.net_logf(v, t)
        logf = tf.clip_by_value(logf, -30.0, 30.0)  # Prevent overflow
        f_cont = tf.exp(logf)
        
        # Add delta peak for cases 3 & 4
        if self.case_type in ['case3', 'case4']:
            Rx = tf.constant(1.0, tf.float32)
            
            # Zero out continuous part for v >= Rx
            f_cont = tf.where(v < Rx, f_cont, tf.zeros_like(f_cont))
            
            # Add narrow Gaussian peak at Rx
            sigma = 0.01 * Rx
            if self.amt is not None:
                A = tf.squeeze(self.amt(tf.expand_dims(t, 1)), axis=1)
            else:
                A = tf.zeros_like(t)
            
            spike = A * tf.exp(-0.5 * tf.square((v - Rx) / sigma)) / (sigma * tf.sqrt(2.0 * np.pi))
            return f_cont + spike
        
        return f_cont
    
    @tf.function
    def compute_pointwise_residuals(
        self,
        v_colloc: tf.Tensor,
        t_colloc: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute PDE residuals at collocation points."""

        v_grid = self.v_grid_tf
        total_points = tf.shape(v_colloc)[0]

        # Handle edge case with no collocation points
        if tf.equal(total_points, 0):
            empty = tf.zeros((0,), dtype=tf.float32)
            return empty, empty

        batch_size = tf.minimum(total_points, 1000)
        batch_size = tf.maximum(batch_size, 1)

        n_batches = tf.cast(
            tf.math.ceil(
                tf.cast(total_points, tf.float32)
                / tf.cast(batch_size, tf.float32)
            ),
            tf.int32,
        )

        residuals_ta = tf.TensorArray(tf.float32, size=n_batches)
        f_ta = tf.TensorArray(tf.float32, size=n_batches)

        def body(i, res_arr, f_arr):
            start_idx = i * batch_size
            end_idx = tf.minimum(start_idx + batch_size, total_points)

            v_batch = v_colloc[start_idx:end_idx]
            t_batch = t_colloc[start_idx:end_idx]

            with tf.GradientTape() as tape:
                tape.watch(t_batch)
                f_batch = self.net_f(v_batch, t_batch)
            df_dt = tape.gradient(f_batch, t_batch)
            df_dt = tf.where(tf.math.is_finite(df_dt), df_dt, tf.zeros_like(df_dt))

            death = self.selection_fn(v_batch) * f_batch

            B = tf.shape(v_batch)[0]
            N = tf.shape(v_grid)[0]

            t_grid = tf.tile(tf.expand_dims(t_batch, 1), [1, N])
            v_grid_batch = tf.tile(tf.expand_dims(v_grid, 0), [B, 1])

            f_grid = self.net_f(
                tf.reshape(v_grid_batch, [-1]),
                tf.reshape(t_grid, [-1])
            )
            f_grid = tf.reshape(f_grid, [B, N])

            s_grid = self.selection_fn(v_grid)
            g_grid = f_grid * (2.0 / v_grid) * s_grid
            tail_integrals = tail_trapz(g_grid, v_grid)

            idx = tf.searchsorted(v_grid, v_batch, side='right')
            idx = tf.minimum(idx, N - 1)
            batch_ids = tf.range(B, dtype=idx.dtype)
            birth = tf.gather_nd(tail_integrals, tf.stack([batch_ids, idx], axis=1))

            residual = df_dt - (birth - death)
            residual = tf.where(tf.math.is_finite(residual), residual, tf.zeros_like(residual))

            res_arr = res_arr.write(i, residual)
            f_arr = f_arr.write(i, f_batch)
            return i + 1, res_arr, f_arr

        def cond(i, *_):
            return i < n_batches

        _, residuals_ta, f_ta = tf.while_loop(
            cond,
            body,
            loop_vars=(tf.constant(0, tf.int32), residuals_ta, f_ta),
            parallel_iterations=1,
        )

        residuals = residuals_ta.concat()
        f_pred = f_ta.concat()

        return residuals, f_pred

    def compute_physics_loss(
        self,
        v_physics: tf.Tensor,
        t_physics: tf.Tensor,
    ) -> tf.Tensor:
        """Compute physics loss with optimized adaptive scaling."""

        residuals, f_pred = self.compute_pointwise_residuals(v_physics, t_physics)
        residuals = tf.clip_by_value(residuals, -1e3, 1e3)

        if self.loss_scaling == "adaptive_huber":
            f_sq = tf.square(f_pred)
            f_sq_mean, _ = tf.nn.moments(f_sq, axes=[0])
            adaptive_eps = 0.01 * f_sq_mean + 1e-10

            weights = 1.0 / (f_sq + adaptive_eps)
            weights = tf.minimum(weights, 50.0)
            loss = tf.reduce_mean(weights * huber_loss(residuals, delta=0.1))
        elif self.loss_scaling == "adaptive_epsilon":
            f_sq = tf.square(f_pred)
            f_sq_mean, _ = tf.nn.moments(f_sq, axes=[0])
            adaptive_eps = 0.01 * f_sq_mean + 1e-10

            weights = 1.0 / (f_sq + adaptive_eps)
            weights = tf.minimum(weights, 100.0)
            loss = tf.reduce_mean(weights * tf.square(residuals))
        else:
            loss = tf.reduce_mean(tf.square(residuals))

        return tf.clip_by_value(loss, 0.0, 1e6)

    def compute_data_loss(
        self,
        v_data: tf.Tensor,
        t_data: tf.Tensor,
        f_data: tf.Tensor
    ) -> tf.Tensor:
        """Compute data loss from observations.
        
        Uses log-space with adaptive weighting for better numerical stability.
        
        Args:
            v_data: Observed volumes
            t_data: Observed times
            f_data: Observed values [batch, 1]
            
        Returns:
            Scalar data loss
        """
        # Work in log space for stability
        log_f_obs = tf.math.log(tf.squeeze(f_data, axis=1) + 1e-12)
        log_f_pred = self.net_logf(v_data, t_data)
        
        # Clip to prevent overflow
        log_f_pred = tf.clip_by_value(log_f_pred, -20.0, 20.0)
        log_f_obs = tf.clip_by_value(log_f_obs, -20.0, 20.0)
        
        # Optimized adaptive weighting without TFP percentile
        log_f_sq = tf.square(log_f_obs)
        mean_log_f_sq, _ = tf.nn.moments(log_f_sq, axes=[0])
        adaptive_eps = 0.01 * mean_log_f_sq + 1e-8

        weights = 1.0 / (log_f_sq + adaptive_eps)
        weights = tf.minimum(weights, 50.0)

        return tf.reduce_mean(weights * tf.square(log_f_pred - log_f_obs))
    
    def predict(
        self,
        v_points: np.ndarray,
        t_points: np.ndarray
    ) -> np.ndarray:
        """Generate predictions on a grid.
        
        Args:
            v_points: Volume grid (1D array)
            t_points: Time grid (1D array)
            
        Returns:
            Predictions f(v,t) with shape [len(t_points), len(v_points)]
        """
        V, T = np.meshgrid(v_points, t_points)
        f_pred = self.net_f(
            tf.constant(V.flatten(), tf.float32),
            tf.constant(T.flatten(), tf.float32)
        )
        return f_pred.numpy().reshape(V.shape)
    
    def get_trainable_variables(self):
        """Get all trainable variables including auxiliary model."""
        vars_list = self.model.trainable_variables
        if self.amt is not None:
            vars_list = vars_list + self.amt.trainable_variables
        return vars_list
    
    def save_weights(self, filepath: str):
        """Save model weights."""
        self.model.save_weights(filepath)
        if self.amt is not None:
            amt_path = filepath.replace('.weights.h5', '_amt.weights.h5')
            self.amt.save_weights(amt_path)
    
    def load_weights(self, filepath: str):
        """Load model weights."""
        self.model.load_weights(filepath)
        if self.amt is not None:
            amt_path = filepath.replace('.weights.h5', '_amt.weights.h5')
            self.amt.load_weights(amt_path)


__all__ = ['BreakagePINN']
