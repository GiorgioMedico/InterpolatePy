# Troubleshooting Guide

This guide helps you solve common issues when using InterpolatePy. Each problem includes the error message, likely cause, and solution with code examples.

## Common Issues

### Import Errors

#### Problem: Module Not Found
```python
ModuleNotFoundError: No module named 'interpolatepy'
```

**Cause**: InterpolatePy is not installed or not in the Python path.

**Solution**:
```bash
# Install from PyPI
pip install interpolatepy

# Or install from source for development
pip install -e .
```

#### Problem: Specific Class Import Failed
```python
ImportError: cannot import name 'SomeClass' from 'interpolatepy'
```

**Cause**: Using an outdated version or importing a class that doesn't exist.

**Solution**: Check the [API Reference](api-reference.md) for correct class names and use the public API:
```python
# ✅ Correct - use public API
from interpolatepy import CubicSpline, DoubleSTrajectory

# ❌ Incorrect - don't import from internal modules  
from interpolatepy.cubic_spline import CubicSpline  # Works but not recommended
```

### Data Input Errors

#### Problem: Non-Monotonic Time Points
```python
ValueError: Time points must be strictly increasing
```

**Cause**: Time points are not in ascending order.

**Solution**: Sort your data before creating trajectories:
```python
import numpy as np
from interpolatepy import CubicSpline

# ❌ Unsorted data
t_bad = [0, 2, 1, 3, 4]
q_bad = [0, 1, 2, 3, 4]

# ✅ Sort the data
sorted_indices = np.argsort(t_bad)
t_sorted = [t_bad[i] for i in sorted_indices]
q_sorted = [q_bad[i] for i in sorted_indices]

spline = CubicSpline(t_sorted, q_sorted)
```

#### Problem: Mismatched Array Lengths
```python
ValueError: t_points and q_points must have the same length
```

**Cause**: Time and position arrays have different sizes.

**Solution**: Ensure arrays are the same length:
```python
# Check lengths before creating spline
if len(t_points) != len(q_points):
    print(f"Length mismatch: t_points={len(t_points)}, q_points={len(q_points)}")
    # Fix by truncating to minimum length
    min_len = min(len(t_points), len(q_points))
    t_points = t_points[:min_len]
    q_points = q_points[:min_len]

spline = CubicSpline(t_points, q_points)
```

#### Problem: Duplicate Time Points
```python
ValueError: Duplicate time points found
```

**Cause**: Multiple data points at the same time value.

**Solution**: Remove duplicates or add small offsets:
```python
def remove_duplicates(t_points, q_points, min_spacing=1e-6):
    """Remove duplicate time points."""
    t_clean, q_clean = [], []
    
    for i, (t, q) in enumerate(zip(t_points, q_points)):
        if i == 0 or t > t_clean[-1] + min_spacing:
            t_clean.append(t)
            q_clean.append(q)
        else:
            # Add small offset to avoid duplicate
            t_clean.append(t_clean[-1] + min_spacing)
            q_clean.append(q)
    
    return t_clean, q_clean

# Clean the data
t_clean, q_clean = remove_duplicates(t_points, q_points)
spline = CubicSpline(t_clean, q_clean)
```

### Motion Profile Errors

#### Problem: Invalid Trajectory Bounds
```python
ValueError: Bounds must be positive values
```

**Cause**: Negative or zero values for velocity, acceleration, or jerk bounds.

**Solution**: Use positive values for all bounds:
```python
from interpolatepy import TrajectoryBounds

# ❌ Invalid bounds
bounds = TrajectoryBounds(v_bound=-1.0, a_bound=2.0, j_bound=1.0)

# ✅ Valid bounds  
bounds = TrajectoryBounds(v_bound=1.0, a_bound=2.0, j_bound=1.0)
```

#### Problem: Impossible Motion Profile
```python
ValueError: Cannot achieve target state with given bounds
```

**Cause**: The trajectory constraints are too restrictive for the desired motion.

**Solution**: Relax the constraints or adjust the target state:
```python
from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds

# If this fails, try increasing the bounds
state = StateParams(q_0=0, q_1=100, v_0=0, v_1=0)
bounds = TrajectoryBounds(v_bound=1.0, a_bound=0.5, j_bound=0.1)

try:
    traj = DoubleSTrajectory(state, bounds)
except ValueError as e:
    print(f"Failed with tight bounds: {e}")
    # Increase bounds and try again
    bounds = TrajectoryBounds(v_bound=5.0, a_bound=2.0, j_bound=1.0)
    traj = DoubleSTrajectory(state, bounds)
    print(f"Success with relaxed bounds: {traj.get_duration():.2f}s")
```

### Smoothing Spline Errors

#### Problem: Incorrect Smoothing Function Usage
```python
TypeError: smoothing_spline_with_tolerance() missing required argument 'config'
```

**Cause**: Using the old API signature. The function now requires a `SplineConfig` object.

**Solution**: Use the correct API:
```python
from interpolatepy import smoothing_spline_with_tolerance, SplineConfig
import numpy as np

# ✅ Correct usage with config
t_points = np.array([0, 1, 2, 3])
q_points = np.array([0, 1.1, 1.9, 3.0])
config = SplineConfig(max_iterations=50)

spline, mu, error, iterations = smoothing_spline_with_tolerance(
    t_points, q_points, tolerance=0.1, config=config
)
print(f"Found spline with μ={mu:.6f}")
```

#### Problem: Smoothing Parameter Out of Range
```python
ValueError: Smoothing parameter μ must be in (0, 1]
```

**Cause**: Invalid smoothing parameter value.

**Solution**: Use values between 0 and 1:
```python
from interpolatepy import CubicSmoothingSpline

# ❌ Invalid μ values
# mu = 0.0    # Too small
# mu = 1.5    # Too large

# ✅ Valid μ values
mu = 0.01     # Light smoothing
# mu = 0.1    # Medium smoothing  
# mu = 1.0    # No smoothing (exact interpolation)

spline = CubicSmoothingSpline(t_points, q_points, mu=mu)
```

### Quaternion Errors

#### Problem: Invalid Quaternion Values
```python
ValueError: Quaternion magnitude is zero or invalid
```

**Cause**: Quaternion with zero magnitude or NaN values.

**Solution**: Normalize quaternions and check for valid values:
```python
from interpolatepy import Quaternion
import numpy as np

def safe_quaternion(s, v1, v2, v3):
    """Create a safe normalized quaternion."""
    q = Quaternion(s, v1, v2, v3)
    
    # Check for NaN or infinite values
    if not np.isfinite([q.s_, *q.v_]).all():
        print("Warning: Invalid quaternion values, using identity")
        return Quaternion.identity()
    
    # Check magnitude
    magnitude = q.norm()
    if magnitude < 1e-10:
        print("Warning: Near-zero quaternion magnitude, using identity")
        return Quaternion.identity()
    
    return q.unit()  # Return normalized quaternion

# Use safe creation
q = safe_quaternion(0.0, 0.0, 0.0, 0.0)  # Will return identity
```

### Evaluation Errors

#### Problem: Time Outside Trajectory Range
```python
ValueError: Evaluation time outside trajectory domain
```

**Cause**: Trying to evaluate trajectory at times outside the defined range.

**Solution**: Check time bounds before evaluation:
```python
from interpolatepy import CubicSpline

spline = CubicSpline([0, 1, 2, 3], [0, 1, 4, 2])

def safe_evaluate(spline, t):
    """Safely evaluate spline with bounds checking."""
    if hasattr(spline, 't_points'):
        t_min, t_max = spline.t_points[0], spline.t_points[-1]
    else:
        # For motion profiles
        t_min, t_max = 0.0, spline.get_duration()
    
    if t < t_min:
        print(f"Warning: t={t} < t_min={t_min}, clamping")
        t = t_min
    elif t > t_max:
        print(f"Warning: t={t} > t_max={t_max}, clamping")
        t = t_max
    
    return spline.evaluate(t)

# Safe evaluation
result = safe_evaluate(spline, 5.0)  # Outside range, will be clamped
```

## Performance Issues

### Problem: Slow Evaluation for Large Arrays
```python
# Slow scalar evaluation
result = [spline.evaluate(t) for t in t_array]  # ❌ Slow
```

**Solution**: Use vectorized evaluation:
```python
import numpy as np

# ✅ Fast vectorized evaluation
t_array = np.linspace(0, 10, 1000)
result = spline.evaluate(t_array)  # Much faster
```

### Problem: Memory Issues with Large Datasets
**Cause**: Processing very large datasets without chunking.

**Solution**: Process data in chunks:
```python
def evaluate_large_dataset(spline, t_array, chunk_size=10000):
    """Evaluate spline over large time array in chunks."""
    results = []
    
    for i in range(0, len(t_array), chunk_size):
        chunk = t_array[i:i + chunk_size]
        results.append(spline.evaluate(chunk))
    
    return np.concatenate(results)

# Use chunked evaluation for very large arrays
large_t_array = np.linspace(0, 100, 1000000)
results = evaluate_large_dataset(spline, large_t_array)
```

## Debugging Tips

### Enable Debug Output
Many InterpolatePy classes support debug output:

```python
# Enable debug output in splines
spline = CubicSpline(t_points, q_points, debug=True)

# Enable debug in smoothing search
config = SplineConfig(debug=True)
spline, mu, error, iterations = smoothing_spline_with_tolerance(
    t_points, q_points, tolerance=0.1, config=config
)
```

### Validate Your Data
Create a helper function to validate input data:

```python
def validate_trajectory_data(t_points, q_points):
    """Validate trajectory input data."""
    issues = []
    
    # Check lengths
    if len(t_points) != len(q_points):
        issues.append(f"Length mismatch: t={len(t_points)}, q={len(q_points)}")
    
    # Check monotonicity  
    if not all(t_points[i] < t_points[i+1] for i in range(len(t_points)-1)):
        issues.append("Time points not strictly increasing")
    
    # Check for NaN/inf
    if not np.isfinite(t_points).all():
        issues.append("Invalid values in t_points")
    
    if not np.isfinite(q_points).all():
        issues.append("Invalid values in q_points")
    
    # Check minimum data points
    if len(t_points) < 2:
        issues.append("Need at least 2 data points")
    
    if issues:
        raise ValueError("Data validation failed: " + "; ".join(issues))
    
    return True

# Use validation
try:
    validate_trajectory_data(t_points, q_points)
    spline = CubicSpline(t_points, q_points)
except ValueError as e:
    print(f"Data validation error: {e}")
```

## Getting Help

If you encounter issues not covered here:

1. **Check the [API Reference](api-reference.md)** for correct usage
2. **Review the [Examples](examples.md)** for working code patterns  
3. **Enable debug output** to see detailed algorithm information
4. **Create minimal reproducible examples** when reporting bugs
5. **Check your InterpolatePy version**: `python -c "import interpolatepy; print(interpolatepy.__version__)"`

### Reporting Bugs

When reporting issues, please include:

- InterpolatePy version
- Python version and platform
- Minimal code example that reproduces the issue
- Complete error message and traceback
- Expected vs actual behavior

Example bug report template:
```python
# InterpolatePy version: 2.0.0
# Python version: 3.10.0 
# Platform: Ubuntu 22.04

from interpolatepy import CubicSpline

# Minimal example that fails
t_points = [0, 1, 2]
q_points = [0, 1, 0]

try:
    spline = CubicSpline(t_points, q_points)
    result = spline.evaluate(1.5)
    print(f"Expected: ~0.5, Got: {result}")
except Exception as e:
    print(f"Error: {e}")
```

This systematic approach helps identify and resolve issues quickly while improving your understanding of InterpolatePy's behavior.