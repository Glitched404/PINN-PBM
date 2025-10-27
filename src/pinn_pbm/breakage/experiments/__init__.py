"""Experiment runners for breakage test cases.

The :mod:`pinn_pbm.breakage.experiments.runner` module exposes the high-level
``run_case`` helper leveraged by the Colab notebook, while the individual
script modules remain available for CLI execution (e.g.
``python -m pinn_pbm.breakage.experiments.case1_linear``).
"""

from .runner import CaseConfig, run_case

__all__ = ["CaseConfig", "run_case"]
