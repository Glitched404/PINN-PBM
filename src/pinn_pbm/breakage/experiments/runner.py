"""High-level runner for breakage PINN experiments (Colab integration).

This module lifts the training workflow that previously lived in
``author_references/original_code.ipynb`` into a reusable Python API. The
primary entry point is :func:`run_case`, which orchestrates the Adam training
stage, optional residual-based adaptive refinement (RAR) refreshes, and
follow-up L-BFGS fine-tuning using either SciPy or TensorFlow Probability.

The goal is to expose an ergonomic function that the Colab notebook can call to
reproduce the original training behaviour while keeping the notebook itself
minimal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from pinn_pbm.breakage.models import BreakagePINN
from pinn_pbm.breakage.solutions import get_analytical_solution
from pinn_pbm.core.training.optimizers import lbfgs_optimizer_tfp
from pinn_pbm.core.training.schedulers import progressive_loss_weights
from pinn_pbm.core.utils import TrainingLogger, set_random_seed

try:
    import tensorflow_probability as tfp

    TFP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency on Colab
    tfp = None  # type: ignore
    TFP_AVAILABLE = False

try:  # SciPy is optional but recommended
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency on Colab
    minimize = None  # type: ignore
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseConfig:
    """Configuration describing an analytical breakage benchmark case."""

    case_type: Literal["case1", "case2", "case3", "case4"]
    v_min: float
    v_max: float
    t_min: float
    t_max: float
    t_slices: Tuple[float, ...]
    adam_epochs: int
    lbfgs_iterations: int
    phys_loss_weight: float
    learning_rate: float
    rarity_interval: int
    loss_schedule: Literal["fixed", "dynamic", "custom"]
    adam_log_every: int = 100
    rar_percentile: float = 95.0
    rar_noise_fraction: float = 0.1


def _build_case_config(case_type: str) -> CaseConfig:
    case = case_type.lower()
    if case == "case1":
        return CaseConfig(
            case_type="case1",
            v_min=1e-3,
            v_max=1e1,
            t_min=0.0,
            t_max=10.0,
            t_slices=(0.0, 2.0, 5.0, 10.0),
            adam_epochs=5000,
            lbfgs_iterations=3000,
            phys_loss_weight=100.0,
            learning_rate=1e-3,
            rarity_interval=500,
            loss_schedule="progressive",
            adam_log_every=250,
            rar_percentile=95.0,
            rar_noise_fraction=0.1,
        )
    if case == "case2":
        return CaseConfig(
            case_type="case2",
            v_min=1e-3,
            v_max=1e2,
            t_min=0.0,
            t_max=5.0,
            t_slices=(0.0, 1.0, 2.0, 5.0),
            adam_epochs=4000,
            lbfgs_iterations=3000,
            phys_loss_weight=50.0,
            learning_rate=5e-4,
            rarity_interval=750,
            loss_schedule="progressive",
            adam_log_every=250,
            rar_percentile=95.0,
            rar_noise_fraction=0.1,
        )
    if case == "case3":
        return CaseConfig(
            case_type="case3",
            v_min=1e-4,
            v_max=1.0,
            t_min=0.0,
            t_max=1000.0,
            t_slices=(0.0, 200.0, 500.0, 1000.0),
            adam_epochs=6000,
            lbfgs_iterations=3500,
            phys_loss_weight=75.0,
            learning_rate=5e-4,
            rarity_interval=750,
            loss_schedule="progressive",
            adam_log_every=300,
            rar_percentile=97.0,
            rar_noise_fraction=0.12,
        )
    if case == "case4":
        return CaseConfig(
            case_type="case4",
            v_min=1e-4,
            v_max=1.5,
            t_min=0.0,
            t_max=2000.0,
            t_slices=(0.0, 500.0, 1000.0, 2000.0),
            adam_epochs=6500,
            lbfgs_iterations=3500,
            phys_loss_weight=75.0,
            learning_rate=5e-4,
            rarity_interval=750,
            loss_schedule="fixed",
            adam_log_every=300,
            rar_percentile=97.0,
            rar_noise_fraction=0.12,
        )
    raise ValueError(f"Unsupported case_type={case_type!r}")


# ---------------------------------------------------------------------------
# Utility helpers for adaptive schedules
# ---------------------------------------------------------------------------


def _dynamic_phys_weight(epoch: int, total_epochs: int, final_weight: float) -> float:
    burn_in_fraction = 0.3
    ramp_fraction = 0.6
    burn_epochs = int(total_epochs * burn_in_fraction)
    ramp_epochs = int(total_epochs * ramp_fraction)
    if epoch < burn_epochs:
        return 0.0
    if epoch < burn_epochs + ramp_epochs:
        progress = (epoch - burn_epochs) / max(ramp_epochs, 1)
        return final_weight * np.sqrt(progress)
    return final_weight


def _custom_phys_weight(epoch: int, total_epochs: int) -> float:
    if epoch < 1000:
        return 0.0
    if epoch < 3000:
        return (epoch - 1000) / 2000.0
    return 1.0


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------


def _build_training_grids(config: CaseConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio = 2.0 ** (1.0 / 3.0)
    n_v = int(np.ceil(np.log(config.v_max / config.v_min) / np.log(ratio))) + 1
    v_grid = (config.v_min * ratio ** np.arange(n_v)).astype(np.float32)

    vs: list[float] = []
    ts: list[float] = []
    fs: list[float] = []
    for t in config.t_slices:
        f_vals = get_analytical_solution(v_grid, t, config.case_type)
        vs.extend(v_grid)
        ts.extend([t] * len(v_grid))
        fs.extend(f_vals)

    v_train = np.array(vs, dtype=np.float32)
    t_train = np.array(ts, dtype=np.float32)
    f_train = np.array(fs, dtype=np.float32).reshape(-1, 1)
    return v_train, t_train, f_train


def _prepare_collocation_candidates(config: CaseConfig, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_v = 1000
    n_t = 200
    v_candidates = np.logspace(np.log10(config.v_min), np.log10(config.v_max), n_v).astype(np.float32)
    t_candidates = np.linspace(config.t_min, config.t_max, n_t, dtype=np.float32)
    V, T = np.meshgrid(v_candidates, t_candidates)
    return V.flatten(), T.flatten()


def _select_collocation_batch(
    rng: np.random.Generator,
    v_candidates: np.ndarray,
    t_candidates: np.ndarray,
    batch_size: int,
    v_min: float,
    v_max: float,
    t_min_eff: float,
    t_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    v_phys = np.exp(rng.uniform(np.log(v_min), np.log(v_max), batch_size)).astype(np.float32)
    t_phys = rng.uniform(t_min_eff, t_max, batch_size).astype(np.float32)
    return v_phys, t_phys


def _residual_adaptive_refinement_optimized(
    pinn: BreakagePINN,
    v_candidates: np.ndarray,
    t_candidates: np.ndarray,
    *,
    percentile: float,
    max_new_points: int,
    rng: np.random.Generator,
    sample_size: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample-based RAR that limits residual evaluations for performance."""

    total_points = len(v_candidates)
    if total_points == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    n_sample = min(sample_size, total_points)
    if n_sample == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    sample_idx = rng.choice(total_points, size=n_sample, replace=False)
    v_sample = v_candidates[sample_idx].astype(np.float32)
    t_sample = t_candidates[sample_idx].astype(np.float32)

    residuals, _ = pinn.compute_pointwise_residuals(
        tf.constant(v_sample, dtype=tf.float32),
        tf.constant(t_sample, dtype=tf.float32),
    )

    res_abs = np.abs(residuals.numpy())
    if res_abs.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    threshold = np.percentile(res_abs, percentile)
    high_mask = res_abs >= threshold

    high_v = v_sample[high_mask][:max_new_points].astype(np.float32)
    high_t = t_sample[high_mask][:max_new_points].astype(np.float32)

    return high_v, high_t


def _run_adam_stage(
    config: CaseConfig,
    pinn: BreakagePINN,
    v_train: np.ndarray,
    t_train: np.ndarray,
    f_train: np.ndarray,
    v_candidates: np.ndarray,
    t_candidates: np.ndarray,
    *,
    adam_batch_data: int = 1024,
    adam_batch_phys: int = 2048,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    logger = TrainingLogger()

    refined_v_np = np.empty((0,), dtype=np.float32)
    refined_t_np = np.empty((0,), dtype=np.float32)

    t_min_eff = max(config.t_min + 1e-6, 1e-3 * config.t_max)

    max_collocation_points = 8000
    weight_cache_interval = 100
    rar_interval_multiplier = 2 if config.rarity_interval else 1
    cached_w_data = 1.0
    cached_w_phys = config.phys_loss_weight if config.loss_schedule == "dynamic" else 1.0

    progress = tqdm(range(config.adam_epochs), desc="Adam", disable=not verbose)
    for epoch in progress:
        if config.loss_schedule == "dynamic":
            weights = _dynamic_phys_weight(epoch, config.adam_epochs, config.phys_loss_weight)
            w_data = 1.0
            w_phys = weights
        elif config.loss_schedule == "custom":
            w_data = 1.0
            w_phys = _custom_phys_weight(epoch, config.adam_epochs)
        else:  # progressive
            if epoch == 0 or epoch % weight_cache_interval == 0:
                progressive = progressive_loss_weights(
                    epoch,
                    config.adam_epochs,
                    final_physics_weight=config.phys_loss_weight,
                )
                cached_w_data = progressive["data"]
                cached_w_phys = progressive["physics"]
            w_data = cached_w_data
            w_phys = cached_w_phys

        refined_v_np = np.empty((0,), dtype=np.float32)
        refined_t_np = np.empty((0,), dtype=np.float32)
        if config.rarity_interval and epoch > 0:
            rar_interval = config.rarity_interval * rar_interval_multiplier
            if epoch % rar_interval == 0 and hasattr(pinn, "compute_pointwise_residuals"):
                new_v, new_t = _residual_adaptive_refinement_optimized(
                    pinn,
                    v_candidates,
                    t_candidates,
                    percentile=config.rar_percentile,
                    max_new_points=500,
                    rng=rng,
                )
                if new_v.size:
                    v_candidates = np.concatenate([v_candidates, new_v])
                    t_candidates = np.concatenate([t_candidates, new_t])

                    if len(v_candidates) > max_collocation_points:
                        idx = rng.choice(len(v_candidates), max_collocation_points, replace=False)
                        v_candidates = v_candidates[idx]
                        t_candidates = t_candidates[idx]

                    refined_v_np = new_v
                    refined_t_np = new_t

        data_idx = rng.integers(0, v_train.shape[0], size=adam_batch_data)
        v_data = tf.constant(v_train[data_idx])
        t_data = tf.constant(t_train[data_idx])
        f_data = tf.constant(f_train[data_idx])

        v_phys_rand, t_phys_rand = _select_collocation_batch(
            rng,
            v_candidates,
            t_candidates,
            adam_batch_phys,
            config.v_min,
            config.v_max,
            t_min_eff,
            config.t_max,
        )
        if refined_v_np.size:
            v_phys_rand = np.concatenate([v_phys_rand, refined_v_np])
            t_phys_rand = np.concatenate([t_phys_rand, refined_t_np])

        v_phys = tf.constant(v_phys_rand, dtype=tf.float32)
        t_phys = tf.constant(t_phys_rand, dtype=tf.float32)

        total, data_loss, phys_loss = pinn.train_step(
            v_data=v_data,
            t_data=t_data,
            f_data=f_data,
            v_physics=v_phys,
            t_physics=t_phys,
            w_data=tf.constant(w_data, dtype=tf.float32),
            w_physics=tf.constant(w_phys, dtype=tf.float32),
            optimizer=optimizer,
        )

        total_scalar = float(total.numpy())
        data_scalar = float(data_loss.numpy())
        phys_scalar = float(phys_loss.numpy())

        logger.log_epoch(epoch, total_scalar, phys_scalar, data_scalar)

        if verbose and (epoch + 1) % config.adam_log_every == 0:
            progress.set_postfix(
                total=f"{total_scalar:.2e}",
                data=f"{data_scalar:.2e}",
                phys=f"{phys_scalar:.2e}",
                w_phys=f"{w_phys:.2e}",
            )

    progress.close()

    return logger


def _lbfgs_scipy(
    pinn: BreakagePINN,
    v_train_tf: tf.Tensor,
    t_train_tf: tf.Tensor,
    f_train_tf: tf.Tensor,
    v_colloc_tf: tf.Tensor,
    t_colloc_tf: tf.Tensor,
    max_iter: int,
    verbose: bool,
) -> Optional[Dict[str, float]]:
    if not SCIPY_AVAILABLE:
        return None

    var_list = list(pinn.model.trainable_variables)
    if pinn.amt is not None:
        var_list += list(pinn.amt.trainable_variables)

    shapes = [v.shape.as_list() for v in var_list]
    w0 = np.concatenate([v.numpy().ravel() for v in var_list]).astype(np.float64)

    def pack(values: Iterable[np.ndarray]) -> np.ndarray:
        return np.concatenate([v.reshape(-1) for v in values])

    def unpack(flat: np.ndarray) -> list[np.ndarray]:
        out = []
        idx = 0
        for shape in shapes:
            size = int(np.prod(shape))
            out.append(flat[idx : idx + size].reshape(shape))
            idx += size
        return out

    def set_weights(flat: np.ndarray) -> None:
        tensors = unpack(flat)
        for tensor, var in zip(tensors, var_list):
            var.assign(tf.convert_to_tensor(tensor, dtype=var.dtype))

    def loss_and_grad(flat: np.ndarray) -> Tuple[float, np.ndarray]:
        set_weights(flat)
        with tf.GradientTape() as tape:
            data_loss = pinn.compute_data_loss(v_train_tf, t_train_tf, f_train_tf)
            phys_loss = pinn.compute_physics_loss(v_colloc_tf, t_colloc_tf)
            total_loss = data_loss + phys_loss
        grads = tape.gradient(total_loss, var_list)
        safe_grads = [(g if g is not None else tf.zeros_like(v)) for g, v in zip(grads, var_list)]
        grad_flat = pack([g.numpy().astype(np.float64) for g in safe_grads])
        return float(total_loss.numpy()), grad_flat

    progress = tqdm(total=max_iter, desc="L-BFGS (scipy)", disable=not verbose)

    def callback(xk: np.ndarray) -> None:  # pragma: no cover - callback uses TF runtime
        if not verbose:
            return
        data_loss = pinn.compute_data_loss(v_train_tf, t_train_tf, f_train_tf)
        phys_loss = pinn.compute_physics_loss(v_colloc_tf, t_colloc_tf)
        progress.set_postfix(
            total=float((data_loss + phys_loss).numpy()),
            data=float(data_loss.numpy()),
            phys=float(phys_loss.numpy()),
        )
        progress.update(1)

    result = minimize(
        fun=loss_and_grad,
        x0=w0,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": max_iter, "disp": verbose},
    )
    progress.close()
    set_weights(result.x)
    return {"success": float(result.success), "nit": float(result.nit)}


def _lbfgs_tfp(
    pinn: BreakagePINN,
    v_train_tf: tf.Tensor,
    t_train_tf: tf.Tensor,
    f_train_tf: tf.Tensor,
    v_colloc_tf: tf.Tensor,
    t_colloc_tf: tf.Tensor,
    max_iter: int,
    verbose: bool,
) -> Optional[Dict[str, float]]:
    if not TFP_AVAILABLE:
        return None

    var_list = list(pinn.model.trainable_variables)
    if pinn.amt is not None:
        var_list += list(pinn.amt.trainable_variables)

    shapes = [v.shape for v in var_list]
    sizes = [int(np.prod(s)) for s in shapes]
    idxs = np.cumsum([0] + sizes)

    def pack(vars_: Iterable[tf.Tensor]) -> tf.Tensor:
        return tf.concat([tf.reshape(v, [-1]) for v in vars_], axis=0)

    def unpack(flat: tf.Tensor) -> list[tf.Tensor]:
        return [
            tf.reshape(flat[idxs[i] : idxs[i + 1]], shapes[i])
            for i in range(len(var_list))
        ]

    @tf.function
    def value_and_gradients(w: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        tensors = unpack(w)
        for tensor, var in zip(tensors, var_list):
            var.assign(tf.cast(tensor, var.dtype))
        with tf.GradientTape() as tape:
            data_loss = pinn.compute_data_loss(v_train_tf, t_train_tf, f_train_tf)
            phys_loss = pinn.compute_physics_loss(v_colloc_tf, t_colloc_tf)
            total = data_loss + phys_loss
        grads = tape.gradient(total, var_list)
        safe_grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, var_list)]
        grad_flat = pack([tf.cast(g, w.dtype) for g in safe_grads])
        return tf.cast(total, w.dtype), grad_flat

    w0 = pack(var_list)
    if verbose:
        print(f"Starting TFP L-BFGS with {len(w0)} parameters…")

    results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=value_and_gradients,
        initial_position=w0,
        max_iterations=max_iter,
        tolerance=1e-7,
        f_relative_tolerance=1e-9,
        x_tolerance=1e-12,
    )

    tensors = unpack(results.position)
    for tensor, var in zip(tensors, var_list):
        var.assign(tf.cast(tensor, var.dtype))

    if verbose:
        print(
            "TFP L-BFGS complete: converged=%s, iterations=%d"
            % (bool(results.converged.numpy()), int(results.num_iterations.numpy()))
        )
    return {
        "converged": float(results.converged.numpy()),
        "iterations": float(results.num_iterations.numpy()),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_case(
    case_type: str = "case1",
    *,
    adam_epochs: Optional[int] = None,
    seed: int = 42,
    lbfgs: Literal["tfp", "scipy", "none"] = "tfp",
    make_plots: bool = True,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train and evaluate a breakage PINN for the given benchmark case.

    Parameters
    ----------
    case_type:
        Identifier for the benchmark (``case1`` – ``case4``).
    adam_epochs:
        Override for the number of Adam epochs. Defaults to the case preset.
    seed:
        Random seed used for data sampling and TensorFlow.
    lbfgs:
        Which L-BFGS implementation to run (``"tfp"``, ``"scipy"``, ``"none"``).
        ``"tfp"`` prefers TensorFlow Probability but falls back to ``"scipy"`` if
        TFP is unavailable.
    make_plots:
        Whether to assemble Matplotlib figures for predictions and loss curves.
    output_dir:
        Optional directory for saving the generated figures.
    verbose:
        Control progress logging.

    Returns
    -------
    dict
        Dictionary containing training histories, predictions, optional figure
        handles, and metadata describing the run.
    """

    config = _build_case_config(case_type)
    if adam_epochs is not None:
        config = CaseConfig(**{**config.__dict__, "adam_epochs": adam_epochs})

    set_random_seed(seed)

    v_train, t_train, f_train = _build_training_grids(config)
    v_candidates, t_candidates = _prepare_collocation_candidates(config, seed)

    pinn = BreakagePINN(
        v_min=config.v_min,
        v_max=config.v_max,
        t_min=config.t_min,
        t_max=config.t_max,
        case_type=config.case_type,
        n_hidden_layers=8,
        n_neurons=128,
    )

    adam_start = perf_counter()
    adam_history = _run_adam_stage(
        config,
        pinn,
        v_train,
        t_train,
        f_train,
        v_candidates,
        t_candidates,
        seed=seed,
        verbose=verbose,
    )
    adam_duration = perf_counter() - adam_start

    lbfgs_summary: Optional[Dict[str, float]] = None
    lbfgs_used: Optional[str] = None

    if lbfgs != "none":
        v_colloc_lbfgs = np.exp(
            np.random.default_rng(seed).uniform(
                np.log(config.v_min),
                np.log(config.v_max),
                2048,
            )
        ).astype(np.float32)
        t_colloc_lbfgs = np.random.default_rng(seed).uniform(
            config.t_min,
            config.t_max,
            2048,
        ).astype(np.float32)

        v_train_tf = tf.convert_to_tensor(v_train, tf.float32)
        t_train_tf = tf.convert_to_tensor(t_train, tf.float32)
        f_train_tf = tf.convert_to_tensor(f_train, tf.float32)
        v_colloc_tf = tf.convert_to_tensor(v_colloc_lbfgs, tf.float32)
        t_colloc_tf = tf.convert_to_tensor(t_colloc_lbfgs, tf.float32)

        if lbfgs == "tfp" and TFP_AVAILABLE:
            lbfgs_summary = lbfgs_optimizer_tfp(
                model=pinn.model,
                loss_fn=lambda: pinn.compute_data_loss(v_train_tf, t_train_tf, f_train_tf)
                + pinn.compute_physics_loss(v_colloc_tf, t_colloc_tf),
                initial_weights=pinn.get_trainable_variables(),
                max_iter=config.lbfgs_max_iter,
                tolerance=config.lbfgs_tolerance,
                verbose=verbose,
                fallback_to_scipy=False,
                line_search_iterations=config.lbfgs_line_search_iterations,
            )
            lbfgs_used = "tfp"
        elif lbfgs in {"tfp", "scipy"} and SCIPY_AVAILABLE:
            lbfgs_summary = _lbfgs_scipy(
                pinn,
                v_train_tf,
                t_train_tf,
                f_train_tf,
                v_colloc_tf,
                t_colloc_tf,
                config.lbfgs_iterations,
                verbose,
            )
            lbfgs_used = "scipy"
        else:
            lbfgs_used = "none_available"

    v_plot = np.logspace(np.log10(config.v_min), np.log10(config.v_max), 500, dtype=np.float32)
    t_plot = np.array(config.t_slices, dtype=np.float32)
    f_pred = pinn.predict(v_plot, t_plot)
    f_exact = np.array([get_analytical_solution(v_plot, float(t), config.case_type) for t in t_plot])

    rel_errors = np.abs((f_pred - f_exact) / (f_exact + 1e-30))
    rel_error_by_slice = rel_errors.mean(axis=1)

    figures: Dict[str, plt.Figure] = {}
    if make_plots:
        fig_loss = adam_history.plot_losses()
        figures["loss"] = fig_loss

        fig_pred, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        colors = plt.cm.viridis(np.linspace(0, 1, len(t_plot)))
        for ax, t_val, color, rel_err_row in zip(axes, t_plot, colors, rel_errors):
            ax.semilogx(v_plot, f_exact[int(np.where(t_plot == t_val)[0][0])], color=color, lw=2.5, label="Analytical")
            ax.semilogx(v_plot, f_pred[int(np.where(t_plot == t_val)[0][0])], "r--", lw=1.8, label="PINN")
            ax.set_title(f"t = {t_val:.1f} (mean rel err {rel_err_row.mean():.2e})")
            ax.set_xlabel("Volume v")
            ax.set_ylabel("f(v,t)")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.legend()
        plt.suptitle(f"{config.case_type.title()}: PINN vs analytical solution")
        plt.tight_layout()
        figures["prediction"] = fig_pred

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig_loss.savefig(Path(output_dir) / f"{config.case_type}_loss.png", dpi=200)
            fig_pred.savefig(Path(output_dir) / f"{config.case_type}_pred.png", dpi=200)

    return {
        "config": config,
        "pinn": pinn,
        "adam_history": adam_history,
        "relative_errors": rel_error_by_slice,
        "figures": figures,
        "adam_duration_sec": adam_duration,
        "lbfgs_backend": lbfgs_used,
        "lbfgs": lbfgs_summary,
        "losses": adam_history.to_dict() if hasattr(adam_history, "to_dict") else None,
        "predictions": {
            "v_grid": v_plot,
            "t_grid": t_plot,
            "f_pred": f_pred,
            "f_exact": f_exact,
        },
    }

__all__ = ["run_case", "CaseConfig"]
