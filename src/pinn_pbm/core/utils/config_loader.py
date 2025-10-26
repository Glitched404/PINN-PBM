"""
Configuration Loader for PINN-PBM.

Handles loading and validation of YAML configuration files for experiments.
Provides defaults for optional parameters and type checking.
"""

import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

import yaml


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


def load_config(
    config_path: str,
    validate: bool = True,
    required_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Load and validate a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        validate: Whether to validate required fields (default: True)
        required_fields: List of required field names. If None, uses default set.
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigurationError: If validation fails or YAML is invalid
        
    Example:
        >>> config = load_config('configs/breakage/case1_config.yaml')
        >>> print(config['problem_type'])
        'breakage'
        >>> print(config['domain']['v_min'])
        0.001
    """
    # Check file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file: {e}")
    
    if config is None:
        raise ConfigurationError(f"Empty configuration file: {config_path}")
    
    # Validate if requested
    if validate:
        if required_fields is None:
            required_fields = ['problem_type', 'case_name']
        
        validate_config(config, required_fields)
    
    return config


def validate_config(
    config: Dict[str, Any],
    required_fields: List[str]
) -> None:
    """Validate that required fields are present in configuration.
    
    Args:
        config: Configuration dictionary to validate
        required_fields: List of required field names (supports nested fields with dots)
        
    Raises:
        ConfigurationError: If any required field is missing
        
    Example:
        >>> config = {'problem_type': 'breakage', 'domain': {'v_min': 0.001}}
        >>> validate_config(config, ['problem_type', 'domain.v_min'])
        # Passes validation
        >>> validate_config(config, ['problem_type', 'missing_field'])
        # Raises ConfigurationError
    """
    missing_fields = []
    
    for field in required_fields:
        # Support nested fields with dot notation (e.g., 'domain.v_min')
        if '.' in field:
            parts = field.split('.')
            current = config
            try:
                for part in parts:
                    current = current[part]
            except (KeyError, TypeError):
                missing_fields.append(field)
        else:
            if field not in config:
                missing_fields.append(field)
    
    if missing_fields:
        raise ConfigurationError(
            f"Missing required fields in configuration: {', '.join(missing_fields)}"
        )


def get_config_value(
    config: Dict[str, Any],
    key: str,
    default: Any = None
) -> Any:
    """Get a configuration value with optional default.
    
    Supports nested keys with dot notation (e.g., 'domain.v_min').
    
    Args:
        config: Configuration dictionary
        key: Key to retrieve (supports dot notation for nested fields)
        default: Default value if key not found
        
    Returns:
        Configuration value or default if not found
        
    Example:
        >>> config = {'domain': {'v_min': 0.001, 'v_max': 10.0}}
        >>> get_config_value(config, 'domain.v_min')
        0.001
        >>> get_config_value(config, 'domain.t_min', default=0.0)
        0.0
    """
    if '.' in key:
        parts = key.split('.')
        current = config
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default
    else:
        return config.get(key, default)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence.
    
    Performs deep merge for nested dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to override base values
        
    Returns:
        Merged configuration dictionary
        
    Example:
        >>> base = {'epochs': 1000, 'network': {'layers': 4, 'neurons': 32}}
        >>> override = {'epochs': 2000, 'network': {'neurons': 64}}
        >>> merged = merge_configs(base, override)
        >>> merged['epochs']
        2000
        >>> merged['network']['layers']
        4
        >>> merged['network']['neurons']
        64
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged


def save_config(
    config: Dict[str, Any],
    config_path: str,
    overwrite: bool = False
) -> None:
    """Save configuration dictionary to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the YAML file
        overwrite: Whether to overwrite existing file (default: False)
        
    Raises:
        FileExistsError: If file exists and overwrite=False
        
    Example:
        >>> config = {'problem_type': 'breakage', 'epochs': 3000}
        >>> save_config(config, 'my_config.yaml', overwrite=True)
    """
    if os.path.exists(config_path) and not overwrite:
        raise FileExistsError(
            f"Configuration file already exists: {config_path}. "
            "Set overwrite=True to replace it."
        )
    
    # Create parent directories if needed
    os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_breakage_config(
    case_name: str,
    config_dir: str = "configs/breakage"
) -> Dict[str, Any]:
    """Convenience function to load breakage case configurations.
    
    Args:
        case_name: Name of the case (e.g., 'case1', 'case1_linear')
        config_dir: Directory containing breakage configs
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigurationError: If validation fails
        
    Example:
        >>> config = load_breakage_config('case1_linear')
        >>> # Automatically loads from configs/breakage/case1_linear_config.yaml
    """
    # Normalize case name
    if not case_name.endswith('_config'):
        case_name = f"{case_name}_config"
    if not case_name.endswith('.yaml'):
        case_name = f"{case_name}.yaml"
    
    config_path = os.path.join(config_dir, case_name)
    
    # Define required fields for breakage problems
    required_fields = [
        'problem_type',
        'case_name',
        'domain.v_min',
        'domain.v_max',
        'domain.t_max'
    ]
    
    return load_config(config_path, validate=True, required_fields=required_fields)


def get_default_breakage_config() -> Dict[str, Any]:
    """Get default configuration template for breakage problems.
    
    Returns:
        Dictionary with default configuration values
        
    Example:
        >>> defaults = get_default_breakage_config()
        >>> # Merge with case-specific overrides
        >>> config = merge_configs(defaults, case_config)
    """
    return {
        'problem_type': 'breakage',
        'case_name': 'default',
        
        'domain': {
            'v_min': 1e-3,
            'v_max': 10.0,
            't_min': 0.0,
            't_max': 10.0,
            'n_v': 500,
            'n_t': 100
        },
        
        'network': {
            'n_hidden_layers': 8,
            'n_neurons': 128,
            'activation': 'tanh',
            'use_residual': True,
            'residual_freq': 2
        },
        
        'training': {
            'epochs': 3000,
            'initial_learning_rate': 5e-4,
            'data_batch_size': 1024,
            'phys_batch_size': 2048,
            'phys_loss_weight': 0.01,
            'loss_schedule': 'dynamic',
            'loss_scaling': 'adaptive_huber',
            'use_warmup': True,
            'warmup_steps': 500
        },
        
        'lbfgs': {
            'enabled': True,
            'max_iter': 2500,
            'n_colloc': 2048
        },
        
        'rar': {
            'enabled': True,
            'refinement_interval': 5000
        },
        
        'output': {
            'save_results': True,
            'save_plots': True,
            'plot_format': 'png',
            'plot_dpi': 300
        }
    }


def validate_breakage_config(config: Dict[str, Any]) -> None:
    """Validate breakage-specific configuration requirements.
    
    Checks value ranges and consistency of breakage configurations.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ConfigurationError: If configuration values are invalid
        
    Example:
        >>> config = load_breakage_config('case1_linear')
        >>> validate_breakage_config(config)
    """
    # Check problem type
    if config.get('problem_type') != 'breakage':
        raise ConfigurationError(
            f"Expected problem_type='breakage', got '{config.get('problem_type')}'"
        )
    
    # Check domain values
    domain = config.get('domain', {})
    v_min = domain.get('v_min', 0)
    v_max = domain.get('v_max', 0)
    t_min = domain.get('t_min', 0)
    t_max = domain.get('t_max', 0)
    
    if v_min <= 0:
        raise ConfigurationError(f"v_min must be positive, got {v_min}")
    if v_max <= v_min:
        raise ConfigurationError(f"v_max ({v_max}) must be greater than v_min ({v_min})")
    if t_min < 0:
        raise ConfigurationError(f"t_min must be non-negative, got {t_min}")
    if t_max <= t_min:
        raise ConfigurationError(f"t_max ({t_max}) must be greater than t_min ({t_min})")
    
    # Check network parameters
    network = config.get('network', {})
    n_layers = network.get('n_hidden_layers', 1)
    n_neurons = network.get('n_neurons', 1)
    
    if n_layers < 1:
        raise ConfigurationError(f"n_hidden_layers must be at least 1, got {n_layers}")
    if n_neurons < 1:
        raise ConfigurationError(f"n_neurons must be at least 1, got {n_neurons}")
    
    # Check training parameters
    training = config.get('training', {})
    epochs = training.get('epochs', 1)
    lr = training.get('initial_learning_rate', 0)
    
    if epochs < 1:
        raise ConfigurationError(f"epochs must be at least 1, got {epochs}")
    if lr <= 0:
        raise ConfigurationError(f"initial_learning_rate must be positive, got {lr}")


# Convenience function aliases
load_yaml = load_config  # Alias for backwards compatibility


__all__ = [
    'load_config',
    'load_yaml',
    'validate_config',
    'get_config_value',
    'merge_configs',
    'save_config',
    'load_breakage_config',
    'get_default_breakage_config',
    'validate_breakage_config',
    'ConfigurationError'
]
