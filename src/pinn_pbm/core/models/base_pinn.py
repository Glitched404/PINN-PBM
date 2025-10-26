"""
Base PINN Model for PINN-PBM.

Provides abstract base class with common functionality for all PINN implementations.
Subclasses implement problem-specific physics and boundary conditions.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2


class BasePINN(ABC):
    """Abstract base class for Physics-Informed Neural Networks.
    
    This class provides common infrastructure for all PINN models:
    - Network architecture building with residual connections
    - Training history tracking
    - Abstract methods for physics and data losses
    
    Subclasses must implement:
    - compute_physics_loss(): Problem-specific PDE residuals
    - compute_data_loss(): Problem-specific data fitting
    - predict(): Problem-specific predictions
    
    Attributes:
        model: Main neural network model
        train_loss_history: Total loss over training
        data_loss_history: Data loss over training
        physics_loss_history: Physics loss over training
    """
    
    def __init__(
        self,
        n_hidden_layers: int = 8,
        n_neurons: int = 128,
        activation: str = 'tanh',
        weight_regularization: float = 1e-6,
        use_residual: bool = True,
        residual_freq: int = 2
    ):
        """Initialize BasePINN.
        
        Args:
            n_hidden_layers: Number of hidden layers (default: 8)
            n_neurons: Number of neurons per layer (default: 128)
            activation: Activation function (default: 'tanh')
            weight_regularization: L2 regularization strength (default: 1e-6)
            use_residual: Whether to use residual connections (default: True)
            residual_freq: Frequency of residual connections in layers (default: 2)
        """
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.weight_regularization = weight_regularization
        self.use_residual = use_residual
        self.residual_freq = residual_freq
        
        # Training history
        self.train_loss_history: List[float] = []
        self.data_loss_history: List[float] = []
        self.physics_loss_history: List[float] = []
        
        # Model will be built by subclass
        self.model: Optional[tf.keras.Model] = None
    
    def build_model(
        self,
        input_dim: int,
        output_dim: int = 1
    ) -> tf.keras.Model:
        """Build neural network with optional residual connections.
        
        Creates a feed-forward network with:
        - Input normalization layer
        - Multiple hidden layers with residual connections
        - Output layer
        
        Args:
            input_dim: Input dimension (e.g., 2 for (v,t))
            output_dim: Output dimension (default: 1)
            
        Returns:
            Compiled Keras model
            
        Example:
            >>> model = self.build_model(input_dim=2, output_dim=1)
            >>> # Model ready for training
        """
        inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
        
        # First hidden layer
        x = tf.keras.layers.Dense(
            self.n_neurons,
            activation=self.activation,
            kernel_initializer='glorot_normal',
            kernel_regularizer=l2(self.weight_regularization)
        )(inputs)
        
        # Additional hidden layers with optional residual connections
        for i in range(self.n_hidden_layers - 1):
            # Save residual connection every residual_freq layers
            if self.use_residual and (i % self.residual_freq) == 0:
                residual = x
            
            x_inner = tf.keras.layers.Dense(
                self.n_neurons,
                activation=self.activation,
                kernel_initializer='glorot_normal',
                kernel_regularizer=l2(self.weight_regularization)
            )(x)
            
            # Apply residual connection
            if self.use_residual and (i % self.residual_freq) != 0:
                x = tf.keras.layers.Add()([x_inner, residual])
            else:
                x = x_inner
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            output_dim,
            activation=None,
            kernel_regularizer=l2(self.weight_regularization)
        )(x)
        
        return tf.keras.Model(inputs, outputs)
    
    @abstractmethod
    def compute_physics_loss(self, *args, **kwargs) -> tf.Tensor:
        """Compute physics-based loss (PDE residuals).
        
        Must be implemented by subclasses to compute problem-specific
        residuals from governing equations.
        
        Returns:
            Scalar tensor representing physics loss
        """
        pass
    
    @abstractmethod
    def compute_data_loss(self, *args, **kwargs) -> tf.Tensor:
        """Compute data-based loss (boundary/initial conditions).
        
        Must be implemented by subclasses to compute loss from
        observed/known data points.
        
        Returns:
            Scalar tensor representing data loss
        """
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        """Generate predictions from trained model.
        
        Must be implemented by subclasses for problem-specific prediction.
        
        Returns:
            NumPy array with predictions
        """
        pass
    
    @tf.function
    def train_step(
        self,
        v_data: tf.Tensor,
        t_data: tf.Tensor,
        f_data: tf.Tensor,
        v_physics: tf.Tensor,
        t_physics: tf.Tensor,
        w_data: tf.Tensor,
        w_physics: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Perform single training step.
        
        Args:
            v_data: Data point volumes
            t_data: Data point times
            f_data: Data point values
            v_physics: Physics collocation volumes
            t_physics: Physics collocation times
            w_data: Data loss weight
            w_physics: Physics loss weight
            optimizer: TensorFlow optimizer
            
        Returns:
            Tuple of (total_loss, data_loss, physics_loss)
        """
        trainable_vars = self.get_trainable_variables()
        
        with tf.GradientTape() as tape:
            data_loss = self.compute_data_loss(v_data, t_data, f_data)
            physics_loss = self.compute_physics_loss(v_physics, t_physics)
            total_loss = w_data * data_loss + w_physics * physics_loss
        
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Filter out None gradients
        grads_and_vars = [
            (g, v) for g, v in zip(gradients, trainable_vars)
            if g is not None
        ]
        
        optimizer.apply_gradients(grads_and_vars)
        
        return total_loss, data_loss, physics_loss
    
    def get_trainable_variables(self) -> List[tf.Variable]:
        """Get all trainable variables from model(s).
        
        Returns:
            List of trainable TensorFlow variables
        """
        if self.model is None:
            return []
        return self.model.trainable_variables
    
    def save_weights(self, filepath: str) -> None:
        """Save model weights.
        
        Args:
            filepath: Path to save weights (should end with .weights.h5)
        """
        if self.model is not None:
            self.model.save_weights(filepath)
    
    def load_weights(self, filepath: str) -> None:
        """Load model weights.
        
        Args:
            filepath: Path to load weights from
        """
        if self.model is not None:
            self.model.load_weights(filepath)
    
    def reset_history(self) -> None:
        """Reset training history."""
        self.train_loss_history = []
        self.data_loss_history = []
        self.physics_loss_history = []
    
    def get_loss_history(self) -> Dict[str, List[float]]:
        """Get training loss history.
        
        Returns:
            Dictionary with 'train', 'data', and 'physics' loss histories
        """
        return {
            'train': self.train_loss_history,
            'data': self.data_loss_history,
            'physics': self.physics_loss_history
        }
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"layers={self.n_hidden_layers}, "
            f"neurons={self.n_neurons}, "
            f"activation='{self.activation}')"
        )


__all__ = ['BasePINN']
