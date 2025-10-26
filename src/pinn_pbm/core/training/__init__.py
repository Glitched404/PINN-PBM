"""
Training utilities for PINN-PBM.

Includes:
- optimizers: L-BFGS (scipy and TensorFlow Probability)
- schedulers: Learning rate schedules, loss weight schedules
Note: Loss functions are in core.utils.helper_functions
"""

# Optimizers (Step 5 - Complete)
from .optimizers import (
    lbfgs_optimizer_scipy,
    lbfgs_optimizer_tfp,
    check_tfp_availability,
    TFP_AVAILABLE
)

# Schedulers (Step 5 - Complete)
from .schedulers import (
    get_dynamic_loss_weight,
    get_custom_dynamic_loss_weight,
    WarmupSchedule,
    create_warmup_cosine_schedule,
    create_warmup_exponential_schedule
)

__all__ = [
    'lbfgs_optimizer_scipy',
    'lbfgs_optimizer_tfp',
    'check_tfp_availability',
    'TFP_AVAILABLE',
    'get_dynamic_loss_weight',
    'get_custom_dynamic_loss_weight',
    'WarmupSchedule',
    'create_warmup_cosine_schedule',
    'create_warmup_exponential_schedule'
]
