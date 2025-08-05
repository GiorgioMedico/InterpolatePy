# Installation Guide

InterpolatePy is available on PyPI and can be installed with pip. We support Python 3.10+ on Windows, macOS, and Linux.

## Requirements

- **Python**: ≥3.10
- **NumPy**: ≥2.0.0 
- **SciPy**: ≥1.15.2
- **Matplotlib**: ≥3.10.1

## Quick Installation

### Using pip (Recommended)

```bash
pip install InterpolatePy
```

This installs the core library with all required dependencies.

### Using conda

```bash
conda install -c conda-forge interpolatepy
```

!!! note "Conda Availability"
    Conda packages may take a few days to update after PyPI releases.

## Development Installation

For contributing to InterpolatePy or accessing the latest features:

### Clone and Install

```bash
git clone https://github.com/GiorgioMedico/InterpolatePy.git
cd InterpolatePy
pip install -e '.[all]'  # Includes testing and development tools
```

### With Development Dependencies

```bash
# Install with all optional dependencies
pip install -e '.[all]'

# Or install specific groups
pip install -e '.[test]'     # Testing tools
pip install -e '.[dev]'      # Development tools
```

### Development Tools Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run code quality checks
ruff format interpolatepy/
ruff check interpolatepy/
mypy interpolatepy/

# Run tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=interpolatepy --cov-report=html --cov-report=term
```

## Optional Dependencies

### Testing Dependencies
- `pytest>=7.3.1` - Test framework
- `pytest-cov>=4.1.0` - Coverage reporting
- `pytest-benchmark>=4.0.0` - Performance benchmarking
- `codecov>=2.1.13` - Coverage upload

### Development Dependencies  
- `ruff>=0.1.5` - Linting and formatting
- `mypy>=1.6.1` - Type checking
- `pre-commit>=4.1.0` - Git hooks
- `pyright>=1.1.335` - Additional type checking
- `build>=1.0.3` - Package building
- `twine>=4.0.2` - Package publishing

## Verification

Verify your installation by running:

```python
import interpolatepy
print(f"InterpolatePy version: {interpolatepy.__version__}")

# Quick test
from interpolatepy import CubicSpline
spline = CubicSpline([0, 1, 2], [0, 1, 0])
print(f"Test evaluation: {spline.evaluate(0.5)}")
```

Expected output:
```
InterpolatePy version: 2.0.0
Test evaluation: 0.75
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'interpolatepy'

**Solution**: Make sure you've installed the package correctly:
```bash
pip install --upgrade InterpolatePy
```

#### ModuleNotFoundError: No module named 'numpy'

**Solution**: Install dependencies manually:
```bash
pip install numpy>=2.0.0 scipy>=1.15.2 matplotlib>=3.10.1
pip install InterpolatePy
```

#### Permission denied during installation

**Solution**: Use user installation:
```bash
pip install --user InterpolatePy
```

Or use a virtual environment:
```bash
python -m venv interp_env
source interp_env/bin/activate  # On Windows: interp_env\Scripts\activate
pip install InterpolatePy
```

#### Version conflicts with existing packages

**Solution**: Create a fresh virtual environment:
```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install InterpolatePy
```

### Platform-Specific Notes

#### Windows
- Use `python` instead of `python3`
- Activate virtual environments with `venv\Scripts\activate`
- Consider using Anaconda for easier dependency management

#### macOS
- May require Xcode command line tools: `xcode-select --install`
- Use `python3` and `pip3` if system Python 2 is present

#### Linux
- Install development headers if compiling from source:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install python3-dev
  
  # CentOS/RHEL/Fedora
  sudo yum install python3-devel
  ```

## Docker Installation

For containerized environments:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install InterpolatePy
RUN pip install InterpolatePy

# Your application code
COPY . /app
WORKDIR /app
```

## Performance Considerations

### NumPy Optimization

For best performance, ensure NumPy is compiled with optimized BLAS:

```python
import numpy as np
print(np.show_config())  # Check BLAS/LAPACK configuration
```

Consider installing optimized NumPy builds:
```bash
# Intel MKL (recommended for Intel CPUs)
pip install mkl-service mkl numpy

# OpenBLAS (good general performance)  
pip install numpy[openblas]
```

### Memory Usage

InterpolatePy is memory-efficient, but for large trajectories consider:

- Use `float32` instead of `float64` for reduced precision requirements
- Process trajectories in chunks for very large datasets
- Enable vectorized operations when possible

## Next Steps

Once installed, check out:

1. **[Quick Start Guide](quickstart.md)** - Your first trajectories
2. **[User Guide](user-guide.md)** - Comprehensive tutorials  
3. **[API Reference](api-reference.md)** - Complete documentation
4. **[Examples](examples.md)** - Real-world use cases

## Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting) above
2. Search [GitHub Issues](https://github.com/GiorgioMedico/InterpolatePy/issues)
3. Create a new issue with:
   - Python version (`python --version`)
   - InterpolatePy version (`import interpolatepy; print(interpolatepy.__version__)`)
   - Complete error traceback
   - Minimal code example reproducing the issue