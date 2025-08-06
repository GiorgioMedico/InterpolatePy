# Installation Guide

InterpolatePy is available on PyPI and can be installed with pip. We support Python 3.10+ on Windows, macOS, and Linux.

## Requirements

- **Python**: ≥3.10
- **NumPy**: ≥2.0.0 
- **SciPy**: ≥1.15.2
- **Matplotlib**: ≥3.10.1

## Installation

### Standard Installation

**Step 1: Verify Python Version**
```bash
python --version
# Should show Python 3.10 or higher
```

**Step 2: Install InterpolatePy**
```bash
pip install InterpolatePy
```

**Step 3: Verify Installation**
```python
import interpolatepy
print(f"InterpolatePy version: {interpolatepy.__version__}")
```

This installs the core library with all required dependencies (NumPy, SciPy, Matplotlib).

### Virtual Environment Installation (Recommended)

Using a virtual environment prevents dependency conflicts and is considered best practice:

**Step 1: Create Virtual Environment**
```bash
# Create a new virtual environment
python -m venv interpolate_env

# Activate the environment
# On Windows:
interpolate_env\Scripts\activate
# On macOS/Linux:
source interpolate_env/bin/activate
```

**Step 2: Install InterpolatePy**
```bash
pip install --upgrade pip  # Ensure latest pip
pip install InterpolatePy
```

**Step 3: Verify Installation**
```python
python -c "import interpolatepy; print('Installation successful!')"
```


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

#### Slow installation or timeout errors

**Solution**: Use faster mirrors or increase timeout:
```bash
# Use PyPI mirrors
pip install -i https://pypi.org/simple/ InterpolatePy

# Increase timeout
pip install --timeout 300 InterpolatePy

# Use --no-cache-dir to avoid cache issues
pip install --no-cache-dir InterpolatePy
```

#### SSL certificate errors

**Solution**: Upgrade certificates or use trusted hosts:
```bash
# Upgrade certificates (macOS)
/Applications/Python\ 3.x/Install\ Certificates.command

# Use trusted host (temporary solution)
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org InterpolatePy
```

#### ImportError: No module named 'interpolatepy'

**Solution 1**: Verify installation:
```bash
pip list | grep -i interpolate
# Should show: InterpolatePy x.x.x
```

**Solution 2**: Reinstall the package:
```bash
pip uninstall InterpolatePy
pip install InterpolatePy
```

**Solution 3**: Check Python environment:
```bash
which python
pip show InterpolatePy
```

#### ModuleNotFoundError: No module named 'numpy' or other dependencies

**Solution 1**: Upgrade pip and try again:
```bash
pip install --upgrade pip setuptools wheel
pip install InterpolatePy
```

**Solution 2**: Install dependencies manually:
```bash
pip install numpy>=2.0.0 scipy>=1.15.2 matplotlib>=3.10.1
pip install InterpolatePy
```

**Solution 3**: Use explicit dependency installation:
```bash
pip install InterpolatePy --force-reinstall --no-deps
pip install numpy scipy matplotlib
```

#### Permission denied during installation

**Solution**: Use user installation:
```bash
pip install --user InterpolatePy
```

Or create a clean virtual environment:
```bash
# Create new environment
python -m venv fresh_env

# Activate environment
# On Windows:
fresh_env\Scripts\activate
# On macOS/Linux: 
source fresh_env/bin/activate

# Install InterpolatePy
pip install --upgrade pip
pip install InterpolatePy
```

#### Version conflicts with existing packages

See the clean virtual environment solution above.

### Platform-Specific Notes

#### Windows
- Use `python` instead of `python3` if Python 2 is not installed
- Activate virtual environments with `venv\Scripts\activate`
- Install Microsoft C++ Build Tools if compilation issues occur:
  - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - Or install via: `winget install Microsoft.VisualStudio.2022.BuildTools`
- Use PowerShell or Command Prompt for installation commands

#### macOS
- May require Xcode command line tools for dependency compilation:
  ```bash
  xcode-select --install
  ```
- Use `python3` and `pip3` if system Python 2 is present
- Install Homebrew for better Python management:
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  brew install python@3.11  # or latest version
  ```
- Consider using pyenv for multiple Python versions:
  ```bash
  brew install pyenv
  pyenv install 3.11.0
  pyenv global 3.11.0
  ```

#### Linux
- Install development headers and build tools if needed:
  ```bash
  # Ubuntu/Debian
  sudo apt update
  sudo apt install python3-dev python3-pip python3-venv build-essential
  
  # CentOS/RHEL/Fedora
  sudo yum install python3-devel python3-pip gcc gcc-c++ make
  
  # Arch Linux
  sudo pacman -S python python-pip base-devel
  
  # Alpine Linux
  apk add python3 python3-dev py3-pip gcc musl-dev
  ```
- Use package manager Python when possible:
  ```bash
  # Ubuntu/Debian
  sudo apt install python3-numpy python3-scipy python3-matplotlib
  pip3 install --user InterpolatePy
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