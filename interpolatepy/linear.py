import numpy as np


def linear_traj(
    p0: float | list[float] | np.ndarray,
    p1: float | list[float] | np.ndarray,
    t0: float,
    t1: float,
    time_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate points along a linear trajectory using NumPy vectorization.

    This function computes positions, velocities, and accelerations for points
    along a linear trajectory between starting position p0 and ending position p1.
    The trajectory is calculated for each time point in time_array.

    Parameters
    ----------
    p0 : float or list[float] or np.ndarray
        Starting position. Can be a scalar for 1D motion or an array/list for
        multi-dimensional motion.
    p1 : float or list[float] or np.ndarray
        Ending position. Must have the same dimensionality as p0.
    t0 : float
        Start time of the trajectory.
    t1 : float
        End time of the trajectory.
    time_array : np.ndarray
        Array of time points at which to calculate the trajectory.

    Returns
    -------
    positions : np.ndarray
        Array of positions at each time point. For scalar inputs, shape is (len(time_array),).
        For vector inputs, shape is (len(time_array), dim) where dim is the dimension of p0/p1.
    velocities : np.ndarray
        Constant velocity at each time point, with the same shape as positions.
    accelerations : np.ndarray
        Zero acceleration at each time point, with the same shape as positions.

    Examples
    --------
    Scalar positions (1D motion):

    >>> import numpy as np
    >>> times = np.linspace(0, 1, 5)
    >>> pos, vel, acc = linear_traj(0, 1, 0, 1, times)
    >>> print(pos)
    [0.   0.25 0.5  0.75 1.  ]
    >>> print(vel)
    [1. 1. 1. 1. 1.]
    >>> print(acc)
    [0. 0. 0. 0. 0.]

    Vector positions (2D motion):

    >>> import numpy as np
    >>> times = np.linspace(0, 2, 3)
    >>> p0 = [0, 0]  # Start at origin
    >>> p1 = [4, 6]  # End at point (4, 6)
    >>> pos, vel, acc = linear_traj(p0, p1, 0, 2, times)
    >>> print(pos)
    [[0. 0.]
     [2. 3.]
     [4. 6.]]
    >>> print(vel)
    [[2. 3.]
     [2. 3.]
     [2. 3.]]

    Notes
    -----
    - This function implements linear interpolation with constant velocity and
      zero acceleration.
    - For vector inputs (multi-dimensional motion), proper broadcasting is applied
      to ensure correct calculation across all dimensions.
    - The function handles both scalar and vector inputs automatically:
        * For scalar inputs: outputs have shape (len(time_array),)
        * For vector inputs: outputs have shape (len(time_array), dim)
    - Time points outside the range [t0, t1] will still produce valid positions
      by extrapolating the linear trajectory.
    - The velocity is always constant and equal to (p1 - p0) / (t1 - t0).
    - The acceleration is always zero.
    """
    # Convert inputs to numpy arrays if they aren't already
    p0 = np.array(p0)
    p1 = np.array(p1)
    time_array = np.array(time_array)

    # Calculate coefficients
    a0 = p0
    a1 = (p1 - p0) / (t1 - t0)

    # Handle broadcasting differently based on whether positions are scalar or vector
    if np.isscalar(p0) or p0.ndim == 0:
        positions = a0 + a1 * (time_array - t0)
        velocities = np.ones_like(time_array) * a1
        accelerations = np.zeros_like(time_array)
    else:
        # Vector case - reshape for proper broadcasting
        time_offset = (time_array - t0).reshape(-1, 1)
        positions = a0 + a1 * time_offset
        velocities = np.tile(a1, (len(time_array), 1))
        accelerations = np.zeros((len(time_array), len(a0)))

    return positions, velocities, accelerations
