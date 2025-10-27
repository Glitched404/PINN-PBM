"""
Learning Rate and Loss Weight Schedulers for PINN-PBM.

Provides various scheduling strategies for learning rates and physics loss weights
during training, enabling better convergence and stability.
"""

from typing import Union
import numpy as np
import tensorflow as tf


def get_dynamic_loss_weight(
    epoch: int,
    total_epochs: int,
    final_weight: float,
    burn_in_fraction: float = 0.3,
    ramp_up_fraction: float = 0.6
) -> float:
    """Compute dynamic physics loss weight with burn-in and ramp-up phases.
    
    This scheduler delays physics loss to allow data fitting first, then
    gradually increases physics loss weight to encourage PDE satisfaction.
    
    Schedule:
    1. Burn-in phase (0 to burn_in_fraction): weight = 0
    2. Ramp-up phase (burn_in to burn_in+ramp_up): weight increases with sqrt
    3. Full phase (remaining epochs): weight = final_weight
    
    Args:
        epoch: Current epoch number (0-indexed)
        total_epochs: Total number of training epochs
        final_weight: Target weight value after ramp-up
        burn_in_fraction: Fraction of epochs for burn-in (default: 0.3)
        ramp_up_fraction: Fraction of epochs for ramp-up (default: 0.6)
        
    Returns:
        Physics loss weight for current epoch
        
    Example:
        >>> # In training loop
        >>> for epoch in range(3000):
        ...     w_phys = get_dynamic_loss_weight(epoch, 3000, final_weight=0.01)
        ...     loss = w_data * data_loss + w_phys * physics_loss
        
        Epoch 0-899: w_phys = 0.0
        Epoch 900-2699: w_phys increases from 0.0 to 0.01
        Epoch 2700-2999: w_phys = 0.01
    """
    burn_in_epochs = int(total_epochs * burn_in_fraction)
    ramp_up_epochs = int(total_epochs * ramp_up_fraction)
    
    if epoch < burn_in_epochs:
        # Phase 1: Burn-in (no physics loss)
        return 0.0
    elif epoch < burn_in_epochs + ramp_up_epochs:
        # Phase 2: Ramp-up (gradual increase with sqrt for smooth transition)
        progress = (epoch - burn_in_epochs) / ramp_up_epochs
        return final_weight * np.sqrt(progress)
    else:
        # Phase 3: Full weight
        return final_weight


def progressive_loss_weights(
    epoch: int,
    total_epochs: int,
    phase1_fraction: float = 0.3,
    phase2_fraction: float = 0.4,
    final_physics_weight: float = 100.0,
) -> dict:
    """Compute progressive data/physics loss weights as described in the PRD.

    Phase layout (fractions must sum to <= 1.0):
    - Phase 1 (data-only): physics weight is 0
    - Phase 2 (progressive ramp): physics weight increases to final_physics_weight
    - Phase 3 (full physics): physics weight fixed at final_physics_weight

    Returns a dictionary with ``data`` and ``physics`` weights.
    """

    phase1_end = int(total_epochs * phase1_fraction)
    phase2_end = int(total_epochs * (phase1_fraction + phase2_fraction))

    if epoch < phase1_end:
        return {"data": 1.0, "physics": 0.0}

    if epoch < phase2_end:
        progress = (epoch - phase1_end) / max(phase2_end - phase1_end, 1)
        return {"data": 1.0, "physics": progress * final_physics_weight}

    return {"data": 1.0, "physics": final_physics_weight}


def get_custom_dynamic_loss_weight(
    epoch: int,
    total_epochs: int,
    stage1_end: int = 1000,
    stage2_end: int = 3000
) -> float:
    """Compute custom dynamic physics loss weight with fixed stage boundaries.
    
    This is an alternative scheduler with fixed epoch boundaries instead of
    fractions, useful when you know exact training duration.
    
    Schedule:
    1. Stage 1 (0 to stage1_end): weight = 0.0
    2. Stage 2 (stage1_end to stage2_end): linear ramp from 0.0 to 1.0
    3. Stage 3 (after stage2_end): weight = 1.0
    
    Args:
        epoch: Current epoch number (0-indexed)
        total_epochs: Total number of training epochs (unused, for API consistency)
        stage1_end: Epoch when stage 1 ends (default: 1000)
        stage2_end: Epoch when stage 2 ends (default: 3000)
        
    Returns:
        Physics loss weight for current epoch
        
    Note:
        Unlike get_dynamic_loss_weight(), this uses absolute epoch numbers
        and ramps to weight=1.0 instead of a configurable final_weight.
        
    Example:
        >>> # In training loop
        >>> for epoch in range(5000):
        ...     w_phys = get_custom_dynamic_loss_weight(epoch, 5000)
        
        Epoch 0-999: w_phys = 0.0
        Epoch 1000-2999: w_phys linear from 0.0 to 1.0
        Epoch 3000+: w_phys = 1.0
    """
    if epoch < stage1_end:
        # Stage 1: No physics loss
        return 0.0
    elif epoch < stage2_end:
        # Stage 2: Linear ramp-up
        start_weight, end_weight = 0.0, 1.0
        progress = (epoch - stage1_end) / (stage2_end - stage1_end)
        return start_weight + progress * (end_weight - start_weight)
    else:
        # Stage 3: Full weight
        return 1.0


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup followed by a base schedule.
    
    Gradually increases learning rate from 0 to initial_lr during warmup,
    then follows a base schedule (e.g., cosine decay, exponential decay).
    
    This helps training stability by avoiding large gradient updates at the
    start when weights are randomly initialized.
    
    Attributes:
        base_schedule: Base learning rate schedule to use after warmup
        warmup_steps: Number of steps for linear warmup
        initial_lr: Target learning rate at end of warmup
        
    Example:
        >>> # Cosine decay with warmup
        >>> cosine = tf.keras.optimizers.schedules.CosineDecay(
        ...     initial_learning_rate=1e-3,
        ...     decay_steps=5000,
        ...     alpha=0.01
        ... )
        >>> schedule = WarmupSchedule(cosine, warmup_steps=500, initial_lr=1e-3)
        >>> optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    """
    
    def __init__(
        self,
        base_schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
        warmup_steps: int,
        initial_lr: float
    ):
        """Initialize warmup schedule.
        
        Args:
            base_schedule: Learning rate schedule to use after warmup
            warmup_steps: Number of steps for linear warmup phase
            initial_lr: Target learning rate at end of warmup
        """
        super().__init__()
        self.base_schedule = base_schedule
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.initial_lr = initial_lr
    
    def __call__(self, step: Union[int, tf.Tensor]) -> tf.Tensor:
        """Compute learning rate for given step.
        
        Args:
            step: Current training step (0-indexed)
            
        Returns:
            Learning rate for this step
        """
        step = tf.cast(step, tf.float32)
        
        # Warmup phase: linear increase from 0 to initial_lr
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        
        # After warmup: use base schedule (adjusted for warmup steps)
        post_warmup_lr = self.base_schedule(step - self.warmup_steps)
        
        # Choose based on whether we're still in warmup
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: post_warmup_lr
        )
    
    def get_config(self):
        """Get configuration for serialization."""
        return {
            'base_schedule': tf.keras.optimizers.schedules.serialize(self.base_schedule),
            'warmup_steps': int(self.warmup_steps.numpy()),
            'initial_lr': self.initial_lr
        }


def create_warmup_cosine_schedule(
    initial_learning_rate: float,
    total_epochs: int,
    warmup_epochs: int = 500,
    min_lr_fraction: float = 0.01
) -> WarmupSchedule:
    """Create a warmup + cosine decay learning rate schedule.
    
    This is a convenience function that combines warmup with cosine decay,
    a common and effective schedule for PINN training.
    
    Args:
        initial_learning_rate: Peak learning rate (after warmup)
        total_epochs: Total number of training epochs
        warmup_epochs: Number of epochs for warmup (default: 500)
        min_lr_fraction: Minimum LR as fraction of initial (default: 0.01)
        
    Returns:
        WarmupSchedule instance ready to use with optimizer
        
    Example:
        >>> schedule = create_warmup_cosine_schedule(
        ...     initial_learning_rate=5e-4,
        ...     total_epochs=3000,
        ...     warmup_epochs=500
        ... )
        >>> optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    """
    # Create cosine decay schedule
    cosine = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=total_epochs - warmup_epochs,
        alpha=min_lr_fraction
    )
    
    # Wrap with warmup
    schedule = WarmupSchedule(
        base_schedule=cosine,
        warmup_steps=warmup_epochs,
        initial_lr=initial_learning_rate
    )
    
    return schedule


def create_warmup_exponential_schedule(
    initial_learning_rate: float,
    total_epochs: int,
    warmup_epochs: int = 500,
    decay_rate: float = 0.96,
    decay_steps: int = 1000
) -> WarmupSchedule:
    """Create a warmup + exponential decay learning rate schedule.
    
    Alternative to cosine decay, exponential decay can be more aggressive
    in reducing learning rate.
    
    Args:
        initial_learning_rate: Peak learning rate (after warmup)
        total_epochs: Total number of training epochs
        warmup_epochs: Number of epochs for warmup (default: 500)
        decay_rate: Decay rate (default: 0.96)
        decay_steps: Steps between decay (default: 1000)
        
    Returns:
        WarmupSchedule instance ready to use with optimizer
        
    Example:
        >>> schedule = create_warmup_exponential_schedule(
        ...     initial_learning_rate=5e-4,
        ...     total_epochs=3000
        ... )
        >>> optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    """
    # Create exponential decay schedule
    exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False
    )
    
    # Wrap with warmup
    schedule = WarmupSchedule(
        base_schedule=exp_decay,
        warmup_steps=warmup_epochs,
        initial_lr=initial_learning_rate
    )
    
    return schedule


__all__ = [
    'get_dynamic_loss_weight',
    'get_custom_dynamic_loss_weight',
    'progressive_loss_weights',
    'WarmupSchedule',
    'create_warmup_cosine_schedule',
    'create_warmup_exponential_schedule'
]
