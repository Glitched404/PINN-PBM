"""
Base model classes for PINN-PBM.

This module contains the BasePINN class that all problem-specific
PINNs inherit from.
"""
Base PINN models and architectures.

Includes:
- BasePINN: Abstract base class for all PINNs
- Network builders and utilities
"""

# Base PINN (Step 8 - Complete)
from .base_pinn import BasePINN

__all__ = ['BasePINN']
