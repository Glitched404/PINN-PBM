# Contributing to PINN-PBM

Thank you for your interest in contributing to PINN-PBM! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository** (or navigate to your local copy)
   ```bash
   cd pinn-pbm
   ```

2. **Install in editable mode with dev dependencies**
   ```bash
   pip install -e .[dev]
   ```

3. **Verify installation**
   ```bash
   pytest tests/ -v
   ```

## Code Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Keep functions focused and modular (single responsibility)
- Maximum line length: 100 characters (120 for complex lines)

### Documentation

- **All functions must have Google-style docstrings**
- Include type hints for all function parameters and returns
- Add inline comments for complex logic
- Update README.md when adding new features

Example docstring:
```python
def my_function(param1: float, param2: str) -> np.ndarray:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> result = my_function(1.0, "test")
        >>> print(result.shape)
        (10,)
    """
    pass
```

### Type Hints

Use type hints for all function signatures:
```python
from typing import List, Dict, Optional, Tuple
import numpy as np
import tensorflow as tf

def process_data(
    data: np.ndarray,
    config: Dict[str, any],
    normalize: bool = True
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Process input data."""
    pass
```

## Testing

### Writing Tests

- Write unit tests for all new functionality
- Place tests in appropriate `tests/` subdirectory
- Use descriptive test names: `test_function_name_expected_behavior`
- Aim for >70% code coverage

Example test structure:
```python
import pytest
import numpy as np
from pinn_pbm.core.utils import helper_functions

def test_trapz_tf_basic():
    """Test trapezoidal integration with simple linear function."""
    x = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
    y = tf.constant([0.0, 1.0, 2.0], dtype=tf.float32)
    result = helper_functions.trapz_tf(y, x)
    expected = 2.0  # Area under y=x from 0 to 2
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
```

### Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=pinn_pbm --cov-report=html

# Specific module
pytest tests/core/test_result_manager.py -v

# Specific test
pytest tests/core/test_result_manager.py::test_save_and_load -v
```

## Git Workflow

### Commit Messages

Use clear, descriptive commit messages following this format:

```
Type: Brief description (50 chars max)

Detailed explanation if needed (wrap at 72 chars)
- Change 1
- Change 2
- Change 3
```

**Commit Types:**
- `Add`: New feature or file
- `Refactor`: Code restructuring without behavior change
- `Fix`: Bug fix
- `Docs`: Documentation only changes
- `Test`: Adding or updating tests
- `Style`: Formatting, missing semicolons, etc.
- `Perf`: Performance improvement

**Examples:**
```
Add: Core infrastructure - ResultManager

Implement timestamped result saving and loading
- Created ResultManager class
- Added save_model, save_results methods
- Included comprehensive docstrings
```

```
Fix: Handle edge case in trapz_tf for single point

Added conditional check to return 0.0 when input
has fewer than 2 points, preventing integration error.
```

### Branching Strategy

- `main`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/xyz`: Individual features
- `fix/xyz`: Bug fixes

## Project Structure Guidelines

### Adding New Problem Modules

When adding a new problem type (e.g., aggregation):

1. **Create module structure**
   ```
   src/pinn_pbm/aggregation/
   ├── __init__.py
   ├── models/
   │   └── aggregation_pinn.py
   ├── physics/
   │   └── kernels.py
   ├── solutions/
   │   └── analytical.py
   └── experiments/
       └── case1.py
   ```

2. **Inherit from BasePINN**
   ```python
   from pinn_pbm.core.models.base_pinn import BasePINN
   
   class AggregationPINN(BasePINN):
       """PINN for aggregation problems."""
       pass
   ```

3. **Add configurations**
   ```
   configs/aggregation/
   └── case1_config.yaml
   ```

4. **Write tests**
   ```
   tests/aggregation/
   ├── test_aggregation_kernels.py
   └── test_aggregation_pinn.py
   ```

## Code Review Process

1. Ensure all tests pass
2. Check code coverage (>70%)
3. Verify documentation is complete
4. Run a quick test to validate functionality
5. Review commit messages for clarity

## Questions?

If you have questions about contributing:
- Open an issue on GitHub
- Contact the maintainers
- Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
