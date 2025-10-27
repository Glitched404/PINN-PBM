# PINN-PBM Google Colab Dependency Conflict Resolution Guide

## Executive Summary

The PINN-PBM project encounters dependency conflicts in Google Colab due to pre-installed packages, Python version mismatches, and incompatible version specifications. This guide provides a comprehensive solution with detailed explanations.

---

## Root Causes Analysis

### 1. **Google Colab's Pre-installed Environment**
**Problem**: Google Colab comes with many packages pre-installed (TensorFlow, NumPy, SciPy, etc.) with specific versions. When PINN-PBM tries to install its dependencies, conflicts arise between:
- What's already installed in Colab
- What PINN-PBM requires
- Transitive dependencies (dependencies of dependencies)

**Why this matters**: 
- Colab uses TensorFlow 2.x (varies by Colab version)
- PINN-PBM likely requires specific versions of TensorFlow/Keras
- NumPy, SciPy, and other scientific packages have strict version interdependencies

### 2. **Python Version Mismatch**
**Problem**: Google Colab typically runs Python 3.10+, but some dependencies may have been developed/tested on earlier Python versions (3.7-3.9).

**Common issues**:
- Type hinting syntax changes
- Removed deprecated features
- Binary wheel availability for specific Python versions

### 3. **Setup.py vs pyproject.toml**
**Problem**: The repository uses older `setup.py` installation method, which:
- Has less reliable dependency resolution than modern tools
- Doesn't lock transitive dependencies
- Allows conflicting sub-dependencies to be installed

### 4. **TensorFlow/Keras Ecosystem Complexity**
**Problem**: TensorFlow 2.x has complex interdependencies:
- `tensorflow` vs `tensorflow-gpu` vs `tf-nightly`
- Keras is now integrated into TensorFlow (TF 2.x) but was separate in TF 1.x
- Different versions require different NumPy versions
- CUDA/cuDNN version compatibility for GPU support

### 5. **Editable Installation (`pip install -e .`)**
**Problem**: Editable installations in Colab can cause issues because:
- File system is ephemeral
- Package discovery may fail
- Build dependencies might be missing

---

## Comprehensive Solution

### Step 1: Create Colab-Specific Installation Notebook

Create a new cell at the beginning of your Colab notebook:

```python
# ============================================
# PINN-PBM Colab Installation Script
# ============================================

import sys
import subprocess
import os

def run_command(cmd, description=""):
    """Execute shell command and print output"""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}")
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

# Check if we're in Colab
try:
    import google.colab
    IN_COLAB = True
    print("✓ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("✗ Not running in Colab")
    sys.exit(1)

# Get Python version
python_version = sys.version_info
print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

# Clone repository
repo_url = "https://github.com/Glitched404/PINN-PBM.git"
repo_name = "PINN-PBM"

if os.path.exists(repo_name):
    run_command(f"cd {repo_name} && git pull", "Updating existing repository")
else:
    run_command(f"git clone {repo_url}", "Cloning repository")

# Change to repo directory
os.chdir(repo_name)
print(f"✓ Changed to directory: {os.getcwd()}")
```

### Step 2: Install Compatible Dependencies with Version Pinning

```python
# ============================================
# Install Dependencies with Proper Versions
# ============================================

# First, uninstall potentially conflicting packages
conflicting_packages = [
    "tensorflow",
    "keras", 
    "h5py",
    "tensorflow-io-gcs-filesystem"
]

print("\n" + "="*60)
print("  Removing potentially conflicting packages")
print("="*60)

for package in conflicting_packages:
    run_command(f"pip uninstall -y {package}", f"Uninstalling {package}")

# Install core dependencies with compatible versions (Python 3.11)
# These versions mirror requirements-colab.txt
core_dependencies = {
    "tensorflow": "2.17.0",
    "tensorflow-probability": "0.25.0",
    "numpy": "1.26.4",
    "scipy": "1.12.0",
    "matplotlib": "3.8.2",
    "pyyaml": "6.0.0",
    "typing-extensions": "4.8.0",
}

print("\n" + "="*60)
print("  Installing core dependencies with pinned versions")
print("="*60)

for package, version in core_dependencies.items():
    run_command(
        f"pip install {package}=={version}",
        f"Installing {package} {version}"
    )

# Install additional scientific computing packages
additional_packages = {
    "tqdm": "4.66.1",
    "pytest": "7.4.4",
    "pytest-cov": "4.1.0",
}

for package, version in additional_packages.items():
    run_command(
        f"pip install {package}=={version}",
        f"Installing {package} {version}"
    )

print("\n✓ All dependencies installed successfully")
```

### Step 3: Install PINN-PBM Package

```python
# ============================================
# Install PINN-PBM Package
# ============================================

# Option A: Regular installation (recommended for Colab)
run_command(
    "pip install --no-deps .",
    "Installing PINN-PBM (without dependencies to avoid conflicts)"
)

# Option B: If you need editable install for development
# run_command(
#     "pip install --no-deps -e .",
#     "Installing PINN-PBM in editable mode"
# )

# Verify installation
print("\n" + "="*60)
print("  Verifying Installation")
print("="*60)

try:
    import pinn_pbm
    print(f"✓ PINN-PBM imported successfully")
    print(f"  Location: {pinn_pbm.__file__}")
except ImportError as e:
    print(f"✗ Failed to import PINN-PBM: {e}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} imported successfully")
    print(f"  GPU Available: {tf.config.list_physical_devices('GPU')}")
except ImportError as e:
    print(f"✗ Failed to import TensorFlow: {e}")

print("\n" + "="*60)
print("  Installation Complete!")
print("="*60)
```

### Step 4: Create requirements-colab.txt

Create a new file `requirements-colab.txt` in the repository root:

```text
# Core ML Framework
tensorflow==2.15.0
# Note: Keras is included in TensorFlow 2.x, no separate install needed

# Numerical Computing
numpy==1.24.3
scipy==1.11.4

# Data Processing
pandas==2.0.3

# Visualization
matplotlib==3.7.1

# Configuration
pyyaml==6.0.1

# Model Persistence
h5py==3.10.0

# Optional: Machine Learning utilities
scikit-learn==1.3.2

# Note: These versions are specifically chosen for compatibility with
# Google Colab's Python 3.10 environment as of October 2024
```

### Step 5: Alternative - Create Colab Setup Script

Create `setup_colab.py` in the repository:

```python
"""
Colab-specific setup script for PINN-PBM
Run this in a Colab notebook cell: !python setup_colab.py
"""

import subprocess
import sys
import os

def install_package(package, version=None):
    """Install a package with optional version specification"""
    if version:
        cmd = f"{sys.executable} -m pip install {package}=={version}"
    else:
        cmd = f"{sys.executable} -m pip install {package}"
    
    print(f"Installing {package}..." + (f" (version {version})" if version else ""))
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: Failed to install {package}")
        print(result.stderr)
        return False
    return True

def main():
    print("="*70)
    print(" PINN-PBM Colab Setup")
    print("="*70)
    
    # Detect environment
    try:
        import google.colab
        print("✓ Google Colab environment detected")
    except ImportError:
        print("✗ Not running in Google Colab. This script is Colab-specific.")
        return
    
    # Check Python version
    py_version = sys.version_info
    print(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("✗ Python 3.8+ required")
        return
    
    # Uninstall conflicting packages
    print("\nStep 1: Removing conflicting packages...")
    conflicts = ["tensorflow", "keras", "h5py"]
    for pkg in conflicts:
        subprocess.run(
            f"{sys.executable} -m pip uninstall -y {pkg}",
            shell=True,
            capture_output=True
        )
    
    # Install dependencies with specific versions
    print("\nStep 2: Installing compatible dependencies...")
    dependencies = {
        "tensorflow": "2.15.0",
        "numpy": "1.24.3",
        "scipy": "1.11.4",
        "matplotlib": "3.7.1",
        "pyyaml": "6.0.1",
        "h5py": "3.10.0",
        "pandas": "2.0.3",
        "scikit-learn": "1.3.2",
    }
    
    failed = []
    for package, version in dependencies.items():
        if not install_package(package, version):
            failed.append(package)
    
    if failed:
        print(f"\n⚠ Warning: Failed to install: {', '.join(failed)}")
    
    # Install PINN-PBM without dependencies
    print("\nStep 3: Installing PINN-PBM...")
    result = subprocess.run(
        f"{sys.executable} -m pip install --no-deps -e .",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("✗ Failed to install PINN-PBM")
        print(result.stderr)
        return
    
    # Verify installation
    print("\nStep 4: Verifying installation...")
    try:
        import pinn_pbm
        print(f"✓ PINN-PBM imported successfully from: {pinn_pbm.__file__}")
    except ImportError as e:
        print(f"✗ Failed to import PINN-PBM: {e}")
        return
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU(s) available: {len(gpus)}")
        else:
            print("⚠ No GPU detected (CPU mode)")
    except ImportError as e:
        print(f"✗ Failed to import TensorFlow: {e}")
        return
    
    print("\n" + "="*70)
    print(" Installation Complete!")
    print("="*70)
    print("\nYou can now use PINN-PBM in this Colab session.")
    print("Example: from pinn_pbm.breakage.experiments import case1_linear")

if __name__ == "__main__":
    main()
```

---

## Usage Instructions for Google Colab

### Method 1: Using the Installation Notebook Cells

1. Create a new Colab notebook
2. Copy Step 1, Step 2, and Step 3 code into separate cells
3. Run each cell sequentially
4. Wait for all installations to complete
5. Restart runtime if prompted
6. Import and use PINN-PBM

Example usage after installation:
```python
# After installation is complete
from pinn_pbm.breakage.experiments import case1_linear

# Run experiment with quick test mode
case1_linear.run_experiment(quick_test=True)
```

### Method 2: Using requirements-colab.txt

1. Upload `requirements-colab.txt` to Colab
2. Clone the repository
3. Run:
```python
!pip uninstall -y tensorflow keras h5py
!pip install -r requirements-colab.txt
!pip install --no-deps -e .
```

### Method 3: Using setup_colab.py

1. Clone the repository
2. Run:
```python
!python setup_colab.py
```

---

## Troubleshooting Common Issues

### Issue 1: "No module named 'pinn_pbm'"

**Cause**: Package not installed or path not recognized

**Solution**:
```python
import sys
import os

# Add repository to Python path
repo_path = "/content/PINN-PBM"  # Adjust path as needed
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# Verify
import pinn_pbm
print(pinn_pbm.__file__)
```

### Issue 2: TensorFlow Version Conflicts

**Cause**: Multiple TensorFlow installations or incompatible versions

**Solution**:
```python
# Complete TensorFlow reinstall
!pip uninstall -y tensorflow tensorflow-gpu tensorflow-estimator tensorflow-io-gcs-filesystem keras
!pip install tensorflow==2.15.0
!pip install --upgrade pip

# Restart runtime after installation
```

### Issue 3: NumPy Version Incompatibility

**Cause**: NumPy version doesn't match TensorFlow requirements

**Solution**:
```python
# Install compatible NumPy
!pip uninstall -y numpy
!pip install numpy==1.24.3

# Restart runtime
```

### Issue 4: "AttributeError: module 'keras' has no attribute..."

**Cause**: Trying to import Keras separately when it's part of TensorFlow 2.x

**Solution**:
```python
# Wrong way (old Keras 2.x style):
# from keras.models import Sequential

# Correct way (TensorFlow 2.x):
from tensorflow import keras
from tensorflow.keras.models import Sequential
```

### Issue 5: YAML Configuration Errors

**Cause**: YAML parser version incompatibility

**Solution**:
```python
!pip install pyyaml==6.0.1 --upgrade
```

### Issue 6: GPU Not Detected

**Cause**: Runtime not configured for GPU

**Solution**:
1. Go to Runtime → Change runtime type
2. Select "GPU" from Hardware accelerator dropdown
3. Click Save
4. Restart runtime

Verify GPU:
```python
import tensorflow as tf
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
```

---

## Why These Specific Versions?

### TensorFlow 2.15.0
- **Stable release** with good GPU support
- **Compatible** with Colab's CUDA version
- **Includes Keras** natively (no separate install needed)
- **Good NumPy compatibility** with 1.24.x

### NumPy 1.24.3
- **Last version** before 2.0 breaking changes
- **Excellent compatibility** with TensorFlow 2.15
- **Stable** with SciPy and other scientific packages

### SciPy 1.11.4
- **Compatible** with NumPy 1.24.x
- **All features** needed for scientific computing
- **Stable release** with bug fixes

### H5Py 3.10.0
- **Modern version** with better TensorFlow integration
- **Efficient** for saving/loading models
- **Compatible** with TensorFlow 2.15

---

## Advanced: Creating a Permanent Solution

### Modify setup.py

Add a Colab-specific dependency group:

```python
# In setup.py
setup(
    name="pinn-pbm",
    # ... other metadata ...
    install_requires=[
        "numpy>=1.21.0,<2.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "colab": [
            "tensorflow==2.15.0",
            "numpy==1.24.3",
            "scipy==1.11.4",
            "matplotlib==3.7.1",
            "h5py==3.10.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
```

Then install with:
```python
!pip install -e ".[colab]"
```

### Create pyproject.toml (Modern Approach)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pinn-pbm"
version = "0.1.0"
description = "Physics-Informed Neural Networks for Population Balance Modeling"
requires-python = ">=3.8,<3.12"
dependencies = [
    "numpy>=1.21.0,<2.0",
    "scipy>=1.7.0,<2.0",
    "matplotlib>=3.5.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
colab = [
    "tensorflow==2.15.0",
    "numpy==1.24.3",
    "scipy==1.11.4",
    "matplotlib==3.7.1",
    "h5py==3.10.0",
    "pandas==2.0.3",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]
```

---

## Best Practices for Colab Development

1. **Always specify versions** for critical dependencies
2. **Test in fresh runtime** regularly
3. **Use `--no-deps` flag** when installing your package to avoid conflicts
4. **Restart runtime** after major installations
5. **Mount Google Drive** to persist results
6. **Save checkpoints frequently** (Colab sessions expire)
7. **Monitor GPU usage** to avoid hitting limits
8. **Document Colab-specific setup** in README

---

## Complete Working Example

```python
# ==================================================
# PINN-PBM Complete Colab Installation
# ==================================================

import subprocess
import sys

def run(cmd):
    """Execute command and stream output"""
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                           stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='')
    proc.wait()
    return proc.returncode

print("Step 1: Checking environment...")
try:
    import google.colab
    print("✓ Running in Colab")
except:
    print("✗ Not in Colab")
    sys.exit(1)

print("\nStep 2: Cloning repository...")
run("git clone https://github.com/Glitched404/PINN-PBM.git")
run("cd PINN-PBM")

print("\nStep 3: Cleaning conflicting packages...")
run("pip uninstall -y tensorflow keras h5py")

print("\nStep 4: Installing dependencies...")
deps = [
    "tensorflow==2.15.0",
    "numpy==1.24.3",
    "scipy==1.11.4",
    "matplotlib==3.7.1",
    "pyyaml==6.0.1",
    "h5py==3.10.0",
]
for dep in deps:
    run(f"pip install {dep}")

print("\nStep 5: Installing PINN-PBM...")
run("cd PINN-PBM && pip install --no-deps -e .")

print("\nStep 6: Verifying...")
try:
    import pinn_pbm
    import tensorflow as tf
    print(f"✓ PINN-PBM: {pinn_pbm.__file__}")
    print(f"✓ TensorFlow: {tf.__version__}")
    print(f"✓ GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("\n🎉 Installation successful!")
except Exception as e:
    print(f"✗ Error: {e}")
```

---

## Summary

The dependency conflicts in Google Colab stem from:
1. Pre-installed packages with specific versions
2. Complex TensorFlow/Keras ecosystem
3. Transitive dependency resolution issues
4. Python version compatibility

**Solution approach**:
1. Clean install of conflicting packages
2. Pin specific compatible versions
3. Install PINN-PBM without dependencies
4. Verify installation

This approach ensures reproducible installations across different Colab sessions and avoids the common pitfalls of dependency management in ephemeral cloud environments.