# PINN-PBM Technical Reference

**Physics-Informed Neural Networks for Population Balance Modeling**

*A comprehensive technical reference covering mathematical foundations, implementation architecture, and design decisions for the PINN-PBM framework.*

**Author:** Research Team  
**Created:** October 27, 2025  
**Last Updated:** October 27, 2025  
**Version:** 1.0 (Steps 1-9 Complete)

---

## Table of Contents

1. [Mathematics & PBM Theory](#1-mathematics--pbm-theory)
2. [Implementation Architecture](#2-implementation-architecture)
3. [Code Details](#3-code-details) *(To be added)*
4. [Training Strategies](#4-training-strategies) *(To be added)*
5. [Test Cases](#5-test-cases) *(To be added)*
6. [Quick Start Guide](#6-quick-start-guide) *(To be added)*

---

# 1. Mathematics & PBM Theory

## 1.1 Population Balance Equation (PBE)

The general form of the population balance equation describes the evolution of the number density function `f(v,t)` representing the particle size distribution:

```
∂f(v,t)/∂t = Birth(v,t) - Death(v,t)
```

Where:
- `f(v,t)`: Number density function (particles per unit volume at volume `v` and time `t`)
- `v`: Particle volume (internal coordinate)
- `t`: Time
- `Birth(v,t)`: Rate of particles of volume `v` created by breakage of larger particles
- `Death(v,t)`: Rate of particles of volume `v` disappearing due to breakage

## 1.2 Breakage-Only PBE

For pure breakage processes, the PBE takes the form:

```
∂f(v,t)/∂t = ∫[v to ∞] β(v,v') S(v') f(v',t) dv' - S(v) f(v,t)
```

**Terms:**
1. **Birth Term (Integral):** `∫[v to ∞] β(v,v') S(v') f(v',t) dv'`
   - Particles of size `v` created from breakage of larger particles `v' > v`
   - `β(v,v')`: Breakage distribution function (daughter particle distribution)
   - `S(v')`: Selection function (breakage frequency)

2. **Death Term:** `S(v) f(v,t)`
   - Particles of size `v` breaking into smaller particles
   - Linear in `f(v,t)`

## 1.3 Selection Functions

The selection function `S(v)` represents the frequency of breakage events for particles of volume `v`.

### Case 1 & 3: Linear Selection
```
S(v) = v
```
**Physical meaning:** Breakage rate proportional to particle volume. Larger particles break more frequently.

**Applications:** Simple first-order breakage kinetics.

### Case 2 & 4: Quadratic Selection
```
S(v) = v²
```
**Physical meaning:** Breakage rate proportional to surface area (for spherical particles). Much stronger size dependence.

**Applications:** Surface-mediated breakage mechanisms.

## 1.4 Breakage Distribution Function

For all four test cases, we use the **symmetric binary breakage** distribution:

```
β(v,v') = 2/v'
```

**Properties:**
1. **Normalization:** `∫[0 to v'] β(v,v') dv = 1` (probability distribution)
2. **Symmetry:** Equal probability for any daughter size combination
3. **Mass conservation:** `∫[0 to v'] v · β(v,v') dv = v'` (total mass conserved)
4. **Factor of 2:** Accounts for two daughter particles per breakage event

**Physical interpretation:** When a particle of volume `v'` breaks, it can form daughters of any size `v` with `v < v'` with equal probability. The distribution is uniform over the daughter particle sizes.

## 1.5 Analytical Solutions

Closed-form analytical solutions exist for specific combinations of `S(v)`, `β(v,v')`, and initial conditions.

### Case 1: Linear Selection, Exponential IC

**Problem:**
```
∂f/∂t = ∫[v to ∞] (2/v') · v' · f(v',t) dv' - v·f(v,t)
Initial condition: f(v,0) = exp(-v)
```

**Analytical Solution:**
```
f(v,t) = exp(-v(1+t)) · (1+t)²
```

**Derivation highlights:**
- Exponential form suggests exponential solution
- Time-dependent amplitude factor `(1+t)²` emerges from integral balance
- Birth and death terms balance through the `(1+t)` factor

### Case 2: Quadratic Selection, Exponential IC

**Problem:**
```
∂f/∂t = ∫[v to ∞] (2/v') · v'² · f(v',t) dv' - v²·f(v,t)
Initial condition: f(v,0) = exp(-v)
```

**Analytical Solution:**
```
f(v,t) = exp(-tv² - v) · (1 + 2t(1+v))
```

**Key features:**
- Quadratic selection leads to `exp(-tv²)` decay
- Polynomial correction factor `(1 + 2t(1+v))` from birth term
- Faster decay for large particles due to `v²` dependence

### Case 3: Linear Selection, Delta Peak IC

**Problem:**
```
∂f/∂t = ∫[v to ∞] (2/v') · v' · f(v',t) dv' - v·f(v,t)
Initial condition: f(v,0) = δ(v - Rx)
```
Where `Rx = 1.0` is the reference volume (monodisperse initial condition).

**Analytical Solution:**

The solution has **two components**:

1. **Delta peak** (residual unbroken particles):
   ```
   f_delta(v,t) = exp(-t·Rx) · δ(v - Rx)
   ```
   - Area decays exponentially: `A(t) = exp(-t·Rx)`
   - Peak remains at `v = Rx`

2. **Continuous distribution** (broken particles, v < Rx):
   ```
   f_continuous(v,t) = exp(-t·v) · (2t + t²(Rx - v))  for v < Rx
                     = 0                               for v ≥ Rx
   ```

**Physical interpretation:**
- Initially, all particles at `v = Rx`
- As time progresses, particles break → continuous distribution builds up for `v < Rx`
- Unbroken particles remain at `Rx` with decaying amplitude
- Heaviside cutoff: no particles can exist above `Rx`

### Case 4: Quadratic Selection, Delta Peak IC

**Problem:**
```
∂f/∂t = ∫[v to ∞] (2/v') · v'² · f(v',t) dv' - v²·f(v,t)
Initial condition: f(v,0) = δ(v - Rx)
```

**Analytical Solution:**

1. **Delta peak:**
   ```
   f_delta(v,t) = exp(-t·Rx²) · δ(v - Rx)
   ```
   - Faster decay due to quadratic selection: `exp(-t·Rx²)` vs `exp(-t·Rx)`

2. **Continuous distribution:**
   ```
   f_continuous(v,t) = exp(-t·v²) · 2t·Rx  for v < Rx
                     = 0                    for v ≥ Rx
   ```

**Key differences from Case 3:**
- Much faster decay of delta peak (`Rx² >> Rx` for `Rx = 1`)
- Simpler continuous part (no polynomial in `v`)
- Stronger decay for larger `v` in continuous part

## 1.6 Numerical Approximation of Delta Functions

In numerical implementations, the Dirac delta `δ(v - Rx)` is approximated using a narrow Gaussian:

```
δ(v - Rx) ≈ (1/(σ√(2π))) · exp(-0.5·((v - Rx)/σ)²)
```

Where:
- `σ = 0.01 · Rx` (1% of peak location)
- Normalized to ensure `∫ δ(v - Rx) dv ≈ 1`

**Implementation:**
```python
def delta_peak(v, center, area, width_fraction=0.01):
    σ = center * width_fraction
    height = area / (σ * np.sqrt(2*π))
    return height * np.exp(-0.5 * ((v - center)/σ)²)
```

## 1.7 Integral Equation Form

The birth term requires computing tail integrals:

```
Birth(v,t) = ∫[v to ∞] g(v',t) dv'
```

Where `g(v',t) = β(v,v') · S(v') · f(v',t) = (2/v') · S(v') · f(v',t)`

**Numerical implementation:**
1. Discretize `v'` on a grid: `v'_1, v'_2, ..., v'_N`
2. Compute `g(v'_i, t)` for all grid points
3. For each collocation point `v_c`, find tail integral `∫[v_c to ∞] g(v') dv'`
4. Use trapezoidal rule with efficient cumulative sum

**TensorFlow implementation:**
```python
@tf.function
def tail_trapz(g, v_grid):
    # Compute segment areas
    dv = v_grid[1:] - v_grid[:-1]
    g_avg = 0.5 * (g[..., :-1] + g[..., 1:])
    segments = g_avg * dv
    
    # Reverse cumulative sum
    reversed_segments = tf.reverse(segments, axis=[-1])
    reversed_cumsum = tf.cumsum(reversed_segments, axis=-1)
    tail_integrals = tf.reverse(reversed_cumsum, axis=[-1])
    
    # Append zero for last point (no tail)
    return tf.concat([tail_integrals, tf.zeros_like(tail_integrals[..., :1])], axis=-1)
```

---

# 2. Implementation Architecture

## 2.1 Overall Design Philosophy

The PINN-PBM framework follows a **modular, hierarchical architecture** with clear separation of concerns:

```
Core Infrastructure (Shared)
    ↓
Problem-Specific Modules (Breakage, Aggregation, etc.)
    ↓
Experiment Scripts (Training, Validation, Visualization)
```

**Key principles:**
1. **DRY (Don't Repeat Yourself):** Shared utilities in core module
2. **Single Responsibility:** Each class/function has one clear purpose
3. **Open/Closed:** Open for extension, closed for modification
4. **Dependency Inversion:** Depend on abstractions (BasePINN), not concrete classes
5. **Composition over Inheritance:** Use composition where appropriate

## 2.2 Project Structure

```
pinn-pbm/
├── src/pinn_pbm/              # Source code package
│   ├── core/                  # SHARED INFRASTRUCTURE
│   │   ├── models/
│   │   │   └── base_pinn.py           # Abstract base class
│   │   ├── training/
│   │   │   ├── schedulers.py          # LR & loss weight schedules
│   │   │   └── optimizers.py          # L-BFGS wrappers
│   │   ├── utils/
│   │   │   ├── result_manager.py      # Timestamped saves
│   │   │   ├── config_loader.py       # YAML configs
│   │   │   └── helper_functions.py    # Numerical utilities
│   │   └── visualization/             # (Future: Step 11)
│   │
│   ├── breakage/              # BREAKAGE MODULE
│   │   ├── models/
│   │   │   └── breakage_pinn.py       # Breakage-specific PINN
│   │   ├── physics/
│   │   │   └── kernels.py             # S(v), β(v,v')
│   │   ├── solutions/
│   │   │   └── analytical.py          # Exact solutions
│   │   └── experiments/               # (Future: Step 12)
│   │
│   ├── aggregation/           # (Future: Phase 2)
│   └── combined/              # (Future: Phase 3)
│
├── configs/                   # YAML configuration files
│   └── breakage/
│       └── case1_config.yaml
│
├── tests/                     # Unit & integration tests
├── results/                   # Auto-generated experiment outputs
├── docs/                      # Documentation
├── README.md
├── requirements.txt
└── setup.py
```

## 2.3 Core Infrastructure Layer

### 2.3.1 BasePINN (Abstract Base Class)

**File:** `src/pinn_pbm/core/models/base_pinn.py`

**Purpose:** Provide common infrastructure for all PINN implementations.

**Design pattern:** Template Method Pattern
- Base class defines skeleton of algorithm
- Subclasses implement problem-specific steps

**Abstract methods** (must be implemented by subclasses):
```python
@abstractmethod
def compute_physics_loss(self, *args, **kwargs) -> tf.Tensor:
    """Compute PDE residuals."""
    pass

@abstractmethod
def compute_data_loss(self, *args, **kwargs) -> tf.Tensor:
    """Compute data fitting loss."""
    pass

@abstractmethod
def predict(self, *args, **kwargs) -> np.ndarray:
    """Generate predictions."""
    pass
```

**Concrete methods** (shared by all subclasses):
```python
def build_model(self, input_dim, output_dim) -> tf.keras.Model:
    """Build neural network with residual connections."""
    # Standard architecture with skip connections
    
def train_step(self, ...) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Single training step with automatic differentiation."""
    # Compute losses, gradients, apply optimizer
    
def save_weights(self, filepath: str):
    """Save model weights."""
    
def load_weights(self, filepath: str):
    """Load model weights."""
```

**Residual connections:** Every 2 layers, add skip connection
```
Layer 0 → Layer 1 → Layer 2 (+Layer 0) → Layer 3 → Layer 4 (+Layer 2) → ...
```
Helps with gradient flow in deep networks.

### 2.3.2 Result Manager

**File:** `src/pinn_pbm/core/utils/result_manager.py`

**Purpose:** Organize and persist experiment results with timestamps.

**Directory structure:**
```
results/
└── {problem_type}/          # e.g., "breakage"
    └── {case_name}/         # e.g., "case1_linear"
        └── YYYYMMDD_HHMMSS/ # Timestamped run
            ├── config.yaml
            ├── model.weights.h5
            ├── losses.npz
            ├── predictions.npz
            ├── metadata.json
            └── plots/
                ├── comparison.png
                └── loss_history.png
```

**Key methods:**
```python
def save_model(self, model, subdir='')
def save_losses(self, train_loss, data_loss, physics_loss)
def save_predictions(self, v_grid, t_grid, predictions, analytical=None)
def save_metadata(self, metadata_dict)
def save_plot(self, fig, filename)
```

**Benefits:**
- Never overwrite previous results
- Easy to compare runs
- Complete reproducibility (config + weights + data)
- Automatic directory creation

### 2.3.3 Configuration Loader

**File:** `src/pinn_pbm/core/utils/config_loader.py`

**Purpose:** Centralized YAML configuration management.

**Example config:**
```yaml
problem_type: breakage
case_name: case1_linear

domain:
  v_min: 0.001
  v_max: 10.0
  t_max: 10.0

network:
  n_hidden_layers: 8
  n_neurons: 128
  activation: tanh

training:
  epochs: 3000
  learning_rate: 5.0e-4
  batch_size: 512
```

**Key functions:**
```python
def load_config(path, validate=True) -> Dict
def validate_config(config, required_fields) -> bool
def merge_configs(base, override) -> Dict
def save_config(config, path)
```

**Validation:** Ensures required fields exist, types are correct, values are in valid ranges.

### 2.3.4 Helper Functions

**File:** `src/pinn_pbm/core/utils/helper_functions.py`

**Purpose:** Reusable numerical and utility functions.

**Categories:**

1. **Numerical methods:**
   - `trapz_tf()`: TensorFlow trapezoidal integration
   - `tail_trapz()`: Efficient tail integral computation

2. **Loss functions:**
   - `huber_loss()`: Robust loss (quadratic → linear transition)
   - `percentile_clip_loss()`: Outlier rejection

3. **Specialized functions:**
   - `delta_peak()`: Gaussian approximation of Dirac delta

4. **Utilities:**
   - `set_random_seed()`: Reproducibility
   - `configure_gpu_memory_growth()`: Prevent OOM errors

**Design:** All marked with `@tf.function` for performance (graph compilation).

### 2.3.5 Training Utilities

**Files:** 
- `src/pinn_pbm/core/training/schedulers.py`
- `src/pinn_pbm/core/training/optimizers.py`

**Learning rate schedules:**
```python
# Warmup + Cosine Decay
schedule = create_warmup_cosine_schedule(
    initial_learning_rate=5e-4,
    total_epochs=3000,
    warmup_epochs=500
)
```

**Loss weight schedules:**
```python
# Dynamic physics weight (3-phase)
w_phys = get_dynamic_loss_weight(
    epoch=current_epoch,
    total_epochs=3000,
    final_weight=0.01,
    burn_in_fraction=0.3,  # Phase 1: w=0
    ramp_up_fraction=0.6   # Phase 2: w increases
)
# Phase 3: w=final_weight
```

**L-BFGS optimizers:**
- `lbfgs_optimizer_scipy()`: Uses scipy.optimize
- `lbfgs_optimizer_tfp()`: Uses TensorFlow Probability (faster, stays in graph)

## 2.4 Breakage Module Layer

### 2.4.1 BreakagePINN Model

**File:** `src/pinn_pbm/breakage/models/breakage_pinn.py`

**Inheritance hierarchy:**
```
BasePINN (abstract)
    ↓
BreakagePINN (concrete)
```

**Key architectural decisions:**

1. **Log-space representation:**
   ```python
   def net_logf(self, v, t) -> tf.Tensor:
       """Predict log(f) instead of f directly."""
   ```
   **Why:** Better numerical stability, wider dynamic range

2. **Domain normalization:**
   ```python
   xi = 2.0 * (log(v) - log(v_min)) / (log(v_max) - log(v_min)) - 1.0
   tau = 2.0 * (log(t) - log(t_min)) / (log(t_max) - log(t_min)) - 1.0
   ```
   **Why:** Map inputs to [-1, 1] for better network conditioning

3. **Dual-component prediction (Cases 3 & 4):**
   ```python
   f_total = f_continuous + f_delta_peak
   ```
   - Main network: Predicts continuous part
   - Auxiliary network (amt): Predicts delta peak amplitude
   - Prevents network from "fighting" between continuous and discontinuous parts

4. **Adaptive loss weighting:**
   ```python
   # Adaptive epsilon based on batch statistics
   median_f_sq = tfp.stats.percentile(f², 50.0)
   adaptive_eps = 0.01 * median_f_sq + 1e-10
   weights = 1.0 / (f² + adaptive_eps)
   ```
   **Why:** Prevents over-emphasis on low-density regions

### 2.4.2 Physics Kernels

**File:** `src/pinn_pbm/breakage/physics/kernels.py`

**Design:** Pure functions, no state

```python
def selection_linear(v: tf.Tensor) -> tf.Tensor:
    return v

def selection_quadratic(v: tf.Tensor) -> tf.Tensor:
    return tf.square(v)

def breakage_symmetric(v: tf.Tensor, vp: tf.Tensor) -> tf.Tensor:
    return 2.0 / vp

def get_selection_function(case_type: str) -> Callable:
    """Factory pattern for selection functions."""
    ...
```

**Why separate file:** Easy to add new kernels, test independently, reuse across models.

### 2.4.3 Analytical Solutions

**File:** `src/pinn_pbm/breakage/solutions/analytical.py`

**Design:** One function per test case

```python
def analytic_f_case1(v, t) -> np.ndarray
def analytic_f_case2(v, t) -> np.ndarray
def analytic_f_case3(v, t, Rx=1.0) -> np.ndarray
def analytic_f_case4(v, t, Rx=1.0) -> np.ndarray

def get_analytical_solution(v, t, case_type, **kwargs) -> np.ndarray:
    """Dispatcher function."""
```

**Validation functions:**
```python
def validate_analytical_solution(v, t, f_analytical, case_type) -> dict:
    """Check: non-negativity, finite values, conservation."""
```

## 2.5 Data Flow Architecture

### Training Loop Data Flow

```
┌─────────────────┐
│  Load Config    │
│  (YAML file)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate Data   │
│ (v, t, f_exact) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create PINN     │
│ BreakagePINN()  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        ADAM Training Loop           │
│  ┌──────────────────────────────┐  │
│  │ For each epoch:              │  │
│  │  1. Sample batch (data)      │  │
│  │  2. Sample collocation (phys)│  │
│  │  3. Compute losses           │  │
│  │  4. Backpropagate            │  │
│  │  5. Update weights           │  │
│  │  6. [Optional] RAR refinement│  │
│  └──────────────────────────────┘  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ L-BFGS Refine   │
│ (Fine-tuning)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Save Results    │
│ ResultManager   │
└─────────────────┘
```

### Physics Loss Computation

```
Collocation Points (v_c, t_c)
         │
         ▼
┌────────────────────────────┐
│  Compute f(v_c, t_c)       │
│  & ∂f/∂t                   │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Death Term: -S(v)·f(v,t)  │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Birth Term:                       │
│  1. Evaluate f on grid             │
│  2. Compute g = β·S·f              │
│  3. Tail integrals ∫[v to ∞] g dv' │
│  4. Interpolate to v_c             │
└────────┬───────────────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Residual = ∂f/∂t - (B-D)  │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Weighted Loss:            │
│  w·Huber(Residual)         │
└────────────────────────────┘
```

## 2.6 Key Design Patterns Used

1. **Template Method** (BasePINN)
   - Base class defines algorithm structure
   - Subclasses fill in specific steps

2. **Factory** (get_selection_function, get_analytical_solution)
   - Create objects based on string identifier
   - Easy to extend with new cases

3. **Strategy** (Loss scaling strategies)
   - Different algorithms for same task
   - Switchable via parameter

4. **Composition** (BreakagePINN has a model + optional amt model)
   - Prefer composition over deep inheritance

5. **Singleton-like** (ResultManager per experiment)
   - One instance manages one experiment's results

## 2.7 TensorFlow Graph Optimization

**All performance-critical functions use `@tf.function`:**

```python
@tf.function
def compute_pointwise_residuals(self, v_c, t_c):
    """Compiled to static graph, ~10-100x faster."""
```

**Benefits:**
- Graph compilation and optimization
- Fusion of operations
- Reduced Python overhead
- GPU kernel fusion

**Trade-offs:**
- Longer first call (tracing)
- Debugging harder (no intermediate prints)
- Type/shape must be consistent

**Best practices:**
- Use for inner loops and loss functions
- Keep I/O and dynamic operations outside
- Profile before/after to verify speedup

---

## Performance Optimizations (Oct 2025)

### Phase 1: Adaptive Loss Scaling without TFP
- **Problem**: `tfp.stats.percentile` caused 28x slowdown in Adam training
- **Solution**: Replace with `tf.nn.moments` for mean/variance approximation
- **Key changes**:
  ```python
  # Physics loss (adaptive_huber)
  f_sq = tf.square(f_pred)
  f_sq_mean, _ = tf.nn.moments(f_sq, axes=[0])
  adaptive_eps = 0.01 * f_sq_mean + 1e-10
  weights = 1.0 / (f_sq + adaptive_eps)
  ```
  **Location**: `src/pinn_pbm/breakage/models/breakage_pinn.py#compute_physics_loss`

### Phase 2: Batched Residual Computation
- **Problem**: Full vectorization caused OOM and retracing
- **Solution**: Process collocation points in 1k batches with `tf.while_loop`
- **Key innovation**:
  ```python
  batch_size = tf.maximum(tf.minimum(total_points, 1000), 1)
  residuals_ta = tf.TensorArray(tf.float32, size=n_batches)
  # ... batched residual calculation ...
  ```
  **Edge handling**: Zero-point case via `tf.cond`
  **Location**: `src/pinn_pbm/breakage/models/breakage_pinn.py#compute_pointwise_residuals`

### Phase 3: Training Loop Optimizations
1. **RAR Frequency**: Reduced from every 500 → 750 steps
2. **Pool Capping**: Max collocation points = 8000
3. **Weight Caching**:
   ```python
   weight_cache_interval = 100
   if epoch % weight_cache_interval == 0:
       cached_w_phys = ...
   ```
4. **L-BFGS Stability**:
   - Disabled SciPy fallback (`fallback_to_scipy=False`)
   - Defined per-case parameters (`lbfgs_max_iter`, `tolerance`)

### Configuration Updates
- Added missing L-BFGS params to `CaseConfig`:
  ```python
  lbfgs_max_iter: int = 1500
  lbfgs_tolerance: float = 1e-12
  lbfgs_line_search_iterations: int = 50
  ```
  **Location**: `src/pinn_pbm/breakage/experiments/runner.py`

### Verification Metrics
- Adam training time restored to 1-2 minutes (from 28 minutes)
- Eliminated NaN losses while preserving adaptive weighting
- Maintained relative errors < 5% across all cases

---

## Appendix A: Glossary

- **PBE:** Population Balance Equation
- **PINN:** Physics-Informed Neural Network
- **Birth term:** Source term in PBE (particles created)
- **Death term:** Sink term in PBE (particles destroyed)
- **Selection function:** Breakage frequency S(v)
- **Breakage distribution:** Daughter particle distribution β(v,v')
- **Collocation points:** Points where PDE is enforced
- **RAR:** Residual Adaptive Refinement
- **L-BFGS:** Limited-memory Broyden-Fletcher-Goldfarb-Shanno optimizer
- **Huber loss:** Robust loss function (L2 → L1 transition)

---

## Appendix B: References

1. Ramkrishna, D. (2000). *Population Balances: Theory and Applications to Particulate Systems in Engineering*. Academic Press.

2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

3. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. *Journal of Machine Learning Research*, 18(153), 1-43.

---

## Document Changelog

### Version 1.0 (October 27, 2025)
- Initial creation with Mathematics & PBM Theory (Section 1)
- Implementation Architecture (Section 2)
- Covers Steps 1-9 of refactoring project
- Sections 3-6 to be added as needed

---

**End of Current Document**

*This is a living document. Update as new features are added, bugs are discovered, or better approaches are found.*
