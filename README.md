# PINN-PBM: Physics-Informed Neural Networks for Population Balance Modeling

A modular framework for solving Population Balance Equations using Physics-Informed Neural Networks.

## Overview

PINN-PBM provides a clean, extensible architecture for applying PINNs to population balance modeling problems. The framework separates shared infrastructure from problem-specific implementations, making it easy to add new problem types.

## Features

- **Modular Architecture**: Core infrastructure shared across all problem types
- **Breakage Module**: Complete implementation with 4 validated test cases
- **Timestamped Results**: Automatic result management with timestamps
- **YAML Configuration**: Easy experiment configuration
- **Comprehensive Testing**: Unit tests for all components
- **Extensible Design**: Ready for aggregation and combined modules

## Project Structure

```
pinn-pbm/
├── src/pinn_pbm/
│   ├── core/              # Shared infrastructure
│   │   ├── models/        # Base PINN classes
│   │   ├── training/      # Optimizers, schedulers, losses
│   │   ├── utils/         # ResultManager, config loader, helpers
│   │   └── visualization/ # Plotting utilities
│   └── breakage/          # Breakage module
│       ├── models/        # BreakagePINN
│       ├── physics/       # Kernels and equations
│       ├── solutions/     # Analytical solutions
│       └── experiments/   # Case 1-4 runners
├── configs/breakage/      # YAML configurations
├── tests/                 # Unit tests
└── results/               # Timestamped outputs
```

## Installation

### Development Installation

```bash
# Clone or navigate to the repository
cd pinn-pbm

# Install in editable mode with dev dependencies
pip install -e .[dev]
```

### Standard Installation

```bash
pip install -e .
```

### Google Colab Installation

Colab currently ships with Python 3.12. Use the TensorFlow 2.19 stack provided in `requirements-colab.txt`:

```python
%pip install -q -r requirements-colab.txt

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

print("numpy:", np.__version__)
print("tensorflow:", tf.__version__)
print("tensorflow_probability:", tfp.__version__)

%pip check

from pinn_pbm.breakage.experiments import run_case

result = run_case(
    case_type="case1",   # "case1", "case2", "case3", "case4"
    adam_epochs=None,     # override progressive defaults by passing an int
    lbfgs="tfp",          # "tfp", "scipy", or "none"
    seed=42,
    make_plots=True,
)

print("Adam duration (s) =", result["adam_duration_sec"])
print("L-BFGS backend:", result["lbfgs_backend"], result["lbfgs"])

losses = result.get("losses")
if losses:
    print("Final total loss =", losses["total"][-1])

```

Alternatively install the package directly with the Colab extra:

```bash
pip install -e .[colab]
```

## Quick Start

### Run Individual Cases

```bash
# Case 1: Linear selection
python -m pinn_pbm.breakage.experiments.case1_linear

# Case 2: Quadratic selection
python -m pinn_pbm.breakage.experiments.case2_quadratic

# Case 3: Linear selection with delta peak
python -m pinn_pbm.breakage.experiments.case3_linear_delta

# Case 4: Quadratic selection with delta peak
python -m pinn_pbm.breakage.experiments.case4_quadratic_delta
```

### Quick Test Mode

Use `--quick-test` flag for rapid validation (reduced epochs):

```bash
python -m pinn_pbm.breakage.experiments.case1_linear --quick-test
```

### Run All Breakage Cases

```bash
python -m pinn_pbm.breakage.experiments.run_all_breakage --quick-test
```

## Results

Results are automatically saved with timestamps:
```
results/breakage/case1_linear/20241027_012345/
├── model_weights.h5
├── loss_history.npy
├── predictions.npy
└── plots/
```

## Testing

Run the complete test suite:

```bash
# All tests with coverage
pytest tests/ -v --cov=pinn_pbm

# Specific module tests
pytest tests/core/ -v
pytest tests/breakage/ -v
```

## Configuration

All experiments use YAML configuration files in `configs/breakage/`. Example:

```yaml
problem_type: "breakage"
case_name: "case1_linear"

domain:
  v_min: 0.001
  v_max: 10.0
  t_min: 0.0
  t_max: 10.0

network:
  n_hidden_layers: 8
  n_neurons: 128

training:
  epochs: 3000
  learning_rate: 0.0005
  ...
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

### Adding New Problem Types

1. Create new module under `src/pinn_pbm/` (e.g., `aggregation/`)
2. Inherit from `BasePINN` in `core.models`
3. Implement problem-specific physics, solutions, and experiments
4. Add configurations and tests

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

If you use this framework in your research, please cite:

```
[Citation information to be added]
```

## Contact

[Contact information to be added]
