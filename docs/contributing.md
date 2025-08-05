# Contributing to InterpolatePy

Thank you for your interest in contributing to InterpolatePy! This guide will help you get set up and explain our development processes.

## Ways to Contribute

- 🐛 **Report bugs** and issues
- ✨ **Suggest new features** or algorithms
- 📝 **Improve documentation**
- 🧪 **Add tests** and examples
- 🔧 **Fix bugs** and implement features
- 📊 **Performance optimizations**
- 🎨 **Code quality improvements**

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/InterpolatePy.git
   cd InterpolatePy
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e '.[all]'  # Installs package + dev dependencies
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Verify installation**:
   ```bash
   python -c "import interpolatepy; print(interpolatepy.__version__)"
   pytest tests/ -v
   ```

### Project Structure

```
InterpolatePy/
├── interpolatepy/          # Main package source
│   ├── __init__.py        # Package exports
│   ├── cubic_spline.py    # Core algorithms
│   ├── double_s.py        # Motion profiles
│   └── ...                # Other modules
├── tests/                 # Test suite
│   ├── test_splines.py    # Algorithm tests
│   └── ...                # Other test files
├── examples/              # Usage examples
├── docs/                  # Documentation source
├── pyproject.toml         # Build configuration
└── README.md              # Project overview
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

Follow our [coding standards](#coding-standards) and ensure your changes:

- ✅ Include appropriate tests
- ✅ Follow existing code style
- ✅ Add documentation for new features
- ✅ Don't break existing functionality

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_cubic_spline.py -v

# Run with coverage
pytest tests/ --cov=interpolatepy --cov-report=html --cov-report=term

# Run benchmarks
pytest tests/ -k "benchmark" --benchmark-only
```

### 4. Check Code Quality

```bash
# Format code
ruff format interpolatepy/

# Check linting
ruff check interpolatepy/

# Type checking
mypy interpolatepy/

# Run pre-commit checks
pre-commit run --all-files
```

### 5. Commit Changes

We use conventional commits for clear history:

```bash
git add .
git commit -m "feat: add new smoothing spline algorithm"
# or
git commit -m "fix: resolve numerical instability in cubic splines"
# or
git commit -m "docs: improve quickstart examples"
```

**Commit Types**:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Formatting changes
- `chore`: Maintenance tasks

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- References to related issues
- Screenshots/examples if applicable
- Checklist of completed items

## Coding Standards

### Python Style

We follow PEP 8 with some modifications configured in `pyproject.toml`:

```python
# Good: Clear function names
def evaluate_cubic_spline(coefficients: np.ndarray, t: float) -> float:
    """Evaluate cubic spline at time t."""
    return np.polyval(coefficients, t)

# Good: Type hints
def create_spline(
    t_points: list[float], 
    q_points: list[float],
    v0: float = 0.0
) -> CubicSpline:
    """Create cubic spline with boundary conditions."""
    return CubicSpline(t_points, q_points, v0=v0)

# Good: Docstrings (NumPy style)
def smooth_trajectory(data: np.ndarray, smoothing: float = 0.1) -> np.ndarray:
    """
    Apply smoothing to trajectory data.
    
    Parameters
    ----------
    data : np.ndarray
        Input trajectory data
    smoothing : float, optional
        Smoothing parameter, by default 0.1
        
    Returns
    -------
    np.ndarray
        Smoothed trajectory
        
    Raises
    ------
    ValueError
        If smoothing parameter is negative
    """
    if smoothing < 0:
        raise ValueError("Smoothing parameter must be non-negative")
    
    # Implementation here
    return smoothed_data
```

### Algorithm Implementation Guidelines

#### 1. Class Structure

```python
class NewAlgorithm:
    """
    Brief description of the algorithm.
    
    Longer description with mathematical background,
    use cases, and key properties.
    
    Parameters
    ----------
    param1 : type
        Description
    param2 : type, optional
        Description, by default value
        
    Examples
    --------
    >>> algorithm = NewAlgorithm(param1, param2)
    >>> result = algorithm.evaluate(t)
    """
    
    def __init__(self, param1: Type1, param2: Type2 = default_value):
        # Input validation
        self._validate_inputs(param1, param2)
        
        # Store parameters
        self.param1 = param1
        self.param2 = param2
        
        # Compute derived quantities
        self._setup_algorithm()
    
    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate algorithm at time(s) t."""
        # Handle both scalar and array inputs
        return self._evaluate_implementation(t)
    
    def evaluate_velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate first derivative at time(s) t."""
        return self._evaluate_derivative_implementation(t, order=1)
        
    def plot(self, num_points: int = 1000) -> None:
        """Plot algorithm results."""
        # Implementation
        pass
    
    def _validate_inputs(self, param1: Type1, param2: Type2) -> None:
        """Validate input parameters."""
        # Validation logic
        pass
    
    def _setup_algorithm(self) -> None:
        """Setup internal algorithm state."""
        # Setup logic
        pass
```

#### 2. Numerical Robustness

```python
# Good: Handle edge cases
def safe_division(a: float, b: float, epsilon: float = 1e-12) -> float:
    """Safely divide a by b, handling near-zero denominators."""
    if abs(b) < epsilon:
        return 0.0 if abs(a) < epsilon else np.sign(a) * np.inf
    return a / b

# Good: Vectorized operations
def evaluate_polynomial(coeffs: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate polynomial using Horner's method."""
    t = np.atleast_1d(t)
    result = np.full_like(t, coeffs[0])
    for coeff in coeffs[1:]:
        result = result * t + coeff
    return result if t.ndim > 0 else result.item()

# Good: Input validation
def validate_time_sequence(t_points: list[float]) -> None:
    """Validate that time points are strictly increasing."""
    if len(t_points) < 2:
        raise ValueError("Need at least 2 time points")
    
    if not all(t_points[i] < t_points[i+1] for i in range(len(t_points)-1)):
        raise ValueError("Time points must be strictly increasing")
    
    if not all(np.isfinite(t) for t in t_points):
        raise ValueError("All time points must be finite")
```

### Testing Guidelines

#### 1. Test Structure

```python
import pytest
import numpy as np
from interpolatepy import YourAlgorithm

class TestYourAlgorithm:
    """Test suite for YourAlgorithm."""
    
    def test_basic_functionality(self):
        """Test basic algorithm functionality."""
        # Arrange
        t_points = [0, 1, 2]
        q_points = [0, 1, 0]
        
        # Act
        algorithm = YourAlgorithm(t_points, q_points)
        result = algorithm.evaluate(0.5)
        
        # Assert
        assert isinstance(result, float)
        assert 0 <= result <= 1  # Expected range
    
    def test_boundary_conditions(self):
        """Test boundary condition handling."""
        algorithm = YourAlgorithm([0, 1, 2], [0, 1, 0])
        
        # Test endpoints
        assert algorithm.evaluate(0) == 0
        assert algorithm.evaluate(2) == 0
    
    def test_continuity_properties(self):
        """Test continuity properties."""
        algorithm = YourAlgorithm([0, 1, 2], [0, 1, 0])
        
        # Test C1 continuity at waypoint
        t_test = 1.0
        eps = 1e-8
        
        pos_left = algorithm.evaluate(t_test - eps)
        pos_right = algorithm.evaluate(t_test + eps)
        
        assert abs(pos_left - pos_right) < 1e-6
    
    def test_vectorized_evaluation(self):
        """Test vectorized evaluation."""
        algorithm = YourAlgorithm([0, 1, 2], [0, 1, 0])
        
        t_array = np.linspace(0, 2, 10)
        results = algorithm.evaluate(t_array)
        
        assert isinstance(results, np.ndarray)
        assert results.shape == t_array.shape
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        with pytest.raises(ValueError):
            YourAlgorithm([0, 0, 1], [0, 1, 0])  # Non-monotonic times
            
        with pytest.raises(ValueError):
            YourAlgorithm([0, 1], [0, 1, 0])  # Mismatched lengths
    
    @pytest.mark.parametrize("smoothing", [0.0, 0.1, 1.0])
    def test_parameter_variations(self, smoothing):
        """Test algorithm with different parameters."""
        algorithm = YourAlgorithm([0, 1, 2], [0, 1, 0], smoothing=smoothing)
        result = algorithm.evaluate(1.0)
        assert np.isfinite(result)
```

#### 2. Performance Tests

```python
import pytest
import numpy as np
from interpolatepy import YourAlgorithm

class TestYourAlgorithmPerformance:
    """Performance tests for YourAlgorithm."""
    
    def test_large_dataset_performance(self, benchmark):
        """Benchmark with large dataset."""
        n_points = 1000
        t_points = np.linspace(0, 10, n_points)
        q_points = np.sin(t_points)
        
        def setup_and_evaluate():
            algorithm = YourAlgorithm(t_points.tolist(), q_points.tolist())
            t_eval = np.linspace(0, 10, 10000)
            return algorithm.evaluate(t_eval)
        
        result = benchmark(setup_and_evaluate)
        assert len(result) == 10000
    
    def test_memory_usage(self):
        """Test memory usage with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Create large algorithm
        n_points = 10000
        t_points = np.linspace(0, 100, n_points)
        q_points = np.random.randn(n_points)
        algorithm = YourAlgorithm(t_points.tolist(), q_points.tolist())
        
        memory_after = process.memory_info().rss
        memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
        
        # Should use reasonable memory (adjust threshold as needed)
        assert memory_used < 100  # Less than 100 MB
```

### Documentation Guidelines

#### 1. Docstring Format

We use NumPy-style docstrings:

```python
def complex_function(
    param1: np.ndarray,
    param2: float = 1.0,
    param3: str | None = None
) -> tuple[np.ndarray, float]:
    """
    One-line summary of function purpose.
    
    Longer description explaining the algorithm, mathematical
    background, and usage context. Can include equations using
    LaTeX notation: $x = \\frac{a}{b}$.
    
    Parameters
    ----------
    param1 : np.ndarray
        Description of param1, including shape requirements
        and expected value ranges.
    param2 : float, optional
        Description of param2, by default 1.0
    param3 : str or None, optional
        Description of param3, by default None
        
    Returns
    -------
    result : np.ndarray
        Description of first return value
    metric : float
        Description of second return value
        
    Raises
    ------
    ValueError
        If param1 has wrong shape
    RuntimeError
        If algorithm fails to converge
        
    Notes
    -----
    Additional notes about algorithm complexity, numerical
    stability, or usage recommendations.
    
    References
    ----------
    .. [1] Author, A. (2023). "Paper Title." Journal Name.
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4])
    >>> result, metric = complex_function(data, param2=2.0)
    >>> print(f"Result shape: {result.shape}")
    Result shape: (4,)
    """
    # Implementation
    pass
```

#### 2. Code Examples

Include practical examples in docstrings and documentation:

```python
def create_robot_trajectory():
    """
    Example: Multi-axis robot trajectory.
    
    This example shows how to create synchronized trajectories
    for a 3-DOF robot arm.
    """
    from interpolatepy import CubicSpline
    import numpy as np
    
    # Joint waypoints (degrees)
    waypoints = {
        'joint1': [0, 45, 90, 45, 0],
        'joint2': [0, -30, 60, -30, 0], 
        'joint3': [0, 20, -45, 20, 0]
    }
    
    time_points = [0, 2, 4, 6, 8]
    trajectories = {}
    
    for joint, angles in waypoints.items():
        trajectories[joint] = CubicSpline(
            time_points,
            np.radians(angles),  # Convert to radians
            v0=0.0, vn=0.0      # Zero velocity at endpoints
        )
    
    return trajectories
```

## Algorithm Contributions

### Adding New Algorithms

When contributing new interpolation algorithms:

1. **Research Background**: Include references to papers/books
2. **Mathematical Foundation**: Document the theory clearly
3. **Implementation**: Follow our class structure guidelines
4. **Testing**: Comprehensive test coverage
5. **Documentation**: Examples and usage guidelines
6. **Performance**: Benchmarks and complexity analysis

### Algorithm Checklist

- [ ] Clear mathematical documentation
- [ ] Robust input validation
- [ ] Vectorized evaluation support
- [ ] Boundary condition handling
- [ ] Error handling and edge cases
- [ ] Comprehensive tests (>90% coverage)
- [ ] Performance benchmarks
- [ ] Usage examples
- [ ] API consistency with existing algorithms

## Performance Considerations

### Optimization Guidelines

1. **Use NumPy**: Vectorized operations over pure Python loops
2. **Memory Layout**: Prefer C-contiguous arrays
3. **Algorithm Complexity**: Document and optimize time/space complexity
4. **Numerical Stability**: Handle edge cases and ill-conditioned problems
5. **Caching**: Cache expensive computations when appropriate

### Benchmarking

```python
# Add benchmarks for new algorithms
def test_algorithm_performance(benchmark):
    """Benchmark algorithm performance."""
    # Setup
    n_points = 1000
    t_points = np.linspace(0, 10, n_points)
    q_points = np.sin(t_points)
    
    # Benchmark setup + evaluation
    def run_algorithm():
        algorithm = YourAlgorithm(t_points.tolist(), q_points.tolist())
        t_eval = np.linspace(0, 10, 10000)
        return algorithm.evaluate(t_eval)
    
    result = benchmark(run_algorithm)
    
    # Verify correctness
    assert len(result) == 10000
    assert np.all(np.isfinite(result))
```

## Issue Guidelines

### Reporting Bugs

When reporting bugs, include:

1. **Minimal example** that reproduces the issue
2. **Expected vs actual behavior**
3. **System information**: OS, Python version, InterpolatePy version
4. **Full error traceback**
5. **Steps to reproduce**

### Feature Requests

For new features, provide:

1. **Use case description**: Why is this needed?
2. **Mathematical background**: References if applicable  
3. **API design ideas**: How should it work?
4. **Examples**: Show intended usage
5. **Alternative solutions**: What exists currently?

## Review Process

### Pull Request Review

All PRs are reviewed for:

- ✅ **Correctness**: Does it work as intended?
- ✅ **Code Quality**: Follows style guidelines?
- ✅ **Testing**: Adequate test coverage?
- ✅ **Documentation**: Clear docs and examples?
- ✅ **Performance**: No significant regressions?
- ✅ **API Design**: Consistent with existing code?

### Merge Requirements

Before merging, PRs must:

- [ ] Pass all CI checks
- [ ] Have at least one approving review
- [ ] Include tests for new functionality
- [ ] Update documentation as needed
- [ ] Maintain backward compatibility (unless breaking change is justified)

## Release Process

InterpolatePy follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Version Updates

1. Update version in `interpolatepy/version.py`
2. Update `CHANGELOG.md` with new features/fixes
3. Create GitHub release with release notes
4. Automated CI publishes to PyPI

## Community

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Documentation**: Comprehensive guides and examples

### Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/):

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment for all contributors

### Recognition

Contributors are recognized in:

- GitHub contributors list
- Release notes for significant contributions
- Documentation acknowledgments

---

Thank you for contributing to InterpolatePy! Your contributions help make trajectory planning more accessible for the robotics and scientific computing communities. 🚀