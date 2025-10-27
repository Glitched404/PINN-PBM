"""Smoke test for the Case 1 breakage runner.

This keeps epochs very small to verify the training pipeline runs end-to-end
without numerical errors. It exercises the progressive scheduler and ensures the
runner returns the expected fields.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from pinn_pbm.breakage.experiments import run_case


def test_case1_smoke_runs_quickly():
    result = run_case(
        case_type="case1",
        adam_epochs=5,
        lbfgs="none",
        seed=123,
        make_plots=False,
        verbose=False,
    )

    assert "config" in result and result["config"].case_type == "case1"
    assert "losses" in result and result["losses"] is not None

    losses = result["losses"]
    for key in ("total", "data", "physics"):
        assert key in losses
        arr = np.asarray(losses[key], dtype=np.float32)
        assert arr.ndim == 1 and arr.size == 5
        assert np.all(np.isfinite(arr))

    assert result["adam_duration_sec"] >= 0.0
    assert result["lbfgs_backend"] == "none"
    assert result["lbfgs"] is None

    v_grid = result["predictions"]["v_grid"]
    f_pred = result["predictions"]["f_pred"]
    assert isinstance(v_grid, np.ndarray)
    assert isinstance(f_pred, np.ndarray)
    assert v_grid.ndim == 1
    assert f_pred.shape[0] == len(result["config"].t_slices)
