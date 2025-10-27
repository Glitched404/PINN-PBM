"""
Utility functions for PINN-PBM.

Includes:
- result_manager: Timestamped result saving and loading
- config_loader: YAML configuration management
- helper_functions: General-purpose utilities
"""

# Result management (Step 2 - Complete)
from .result_manager import ResultManager
from .training_logger import TrainingLogger

# Config loader (Step 3 - Complete)
from .config_loader import (
    load_config,
    load_yaml,
    validate_config,
    get_config_value,
    merge_configs,
    save_config,
    load_breakage_config,
    get_default_breakage_config,
    validate_breakage_config,
    ConfigurationError
)

# Helper functions (Step 4 - Complete)
from .helper_functions import (
    trapz_tf,
    tail_trapz,
    huber_loss,
    percentile_clip_loss,
    delta_peak,
    set_random_seed,
    configure_gpu_memory_growth,
    check_tensorflow_probability,
    TFP_AVAILABLE
)

__all__ = [
    'ResultManager',
    'TrainingLogger',
    'load_config',
    'load_yaml',
    'validate_config',
    'get_config_value',
    'merge_configs',
    'save_config',
    'load_breakage_config',
    'get_default_breakage_config',
    'validate_breakage_config',
    'ConfigurationError',
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
