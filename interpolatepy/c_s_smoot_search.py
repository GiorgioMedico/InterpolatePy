from dataclasses import dataclass

import numpy as np

from interpolatepy.c_s_smoothing import CubicSmoothingSpline


# Constants to replace magic values
EPSILON = 1e-6  # Convergence tolerance


@dataclass
class SplineConfig:
    """Configuration parameters for smoothing spline calculation."""

    weights: list[float] | np.ndarray | None = None
    v0: float = 0.0
    vn: float = 0.0
    max_iterations: int = 50
    debug: bool = False


def smoothing_spline_with_tolerance(
    t_points: np.ndarray,
    q_points: np.ndarray,
    tolerance: float,
    config: SplineConfig,
) -> tuple[CubicSmoothingSpline, float, float, int]:
    """Find a cubic smoothing spline with a maximum approximation error smaller than
    a given tolerance using binary search on the μ parameter.

    This implements the algorithm for "Smoothing spline with prescribed tolerance".

    Parameters
    ----------
    t_points : np.ndarray
        Time points [t₀, t₁, t₂, ..., tₙ]
    q_points : np.ndarray
        Position points [q₀, q₁, q₂, ..., qₙ]
    tolerance : float
        Maximum allowed approximation error δ between original and smoothed points
    config : SplineConfig
        Configuration object with optional parameters:
        - weights: Individual point weights [w₀, w₁, ..., wₙ] (None = equal weights)
        - v0: Initial velocity constraint at t₀
        - vn: Final velocity constraint at tₙ
        - max_iterations: Maximum number of iterations for the binary search
        - debug: Whether to print debug information

    Returns
    -------
    spline : CubicSmoothingSpline
        The final CubicSmoothingSpline object
    mu : float
        The found value of μ parameter
    e_max : float
        The maximum approximation error achieved
    iterations : int
        Number of iterations performed

    Examples
    --------
    >>> import numpy as np
    >>> from interpolatepy.c_s_smoothing import CubicSmoothingSpline
    >>> # Create sample data
    >>> t = np.linspace(0, 10, 100)
    >>> q = np.sin(t) + 0.1 * np.random.randn(100)
    >>> config = SplineConfig(max_iterations=20)
    >>> # Find spline with tolerance of 0.05
    >>> spline, mu, error, iterations = smoothing_spline_with_tolerance(t, q, 0.05, config)
    >>> print(f"Found spline with μ={mu:.6f}, error={error:.6f} in {iterations} iterations")

    Notes
    -----
    The algorithm uses binary search to find the optimal μ parameter value that
    produces a smoothing spline with maximum error below the specified tolerance.
    The parameter μ controls the trade-off between smoothness and accuracy, with
    values closer to 0 producing smoother curves and values closer to 1 producing
    more accurate but less smooth curves.
    """

    # Initialize the search range for μ
    # Note: CubicSmoothingSpline requires 0 < μ ≤ 1, so we start with a small positive value
    lower_bound = EPSILON  # lower_bound(0) = small positive number (close to 0)
    upper_bound = 1.0  # upper_bound(0) = 1

    if config.debug:
        print(f"Starting binary search with tolerance δ={tolerance}")
        print(f"Initial lower_bound={lower_bound}, upper_bound={upper_bound}")

    # Create initial (fallback) spline with μ=1.0 (most accurate)
    # This ensures we always have a valid spline to return
    default_spline = CubicSmoothingSpline(
        t_points, q_points, mu=1.0, weights=config.weights, v0=config.v0, vn=config.vn, debug=False
    )
    default_error = np.max(np.abs(default_spline.q - default_spline.s))

    # Keep track of the best solution found so far
    best_spline = default_spline
    best_mu = 1.0
    best_error = default_error

    # Iteration loop
    for i in range(config.max_iterations):
        # Step 1: Assume μ(i) = (lower_bound(i) + upper_bound(i))/2
        mu = (lower_bound + upper_bound) / 2

        if config.debug:
            print(f"\nIteration {i + 1}: μ={mu}")

        # Step 2: Compute spline and maximum error
        try:
            # Create spline with current μ
            spline = CubicSmoothingSpline(
                t_points,
                q_points,
                mu=mu,
                weights=config.weights,
                v0=config.v0,
                vn=config.vn,
                debug=False,
            )

            # Calculate maximum error e_max(i)
            e_max = np.max(np.abs(spline.q - spline.s))

            if config.debug:
                print(f"  Maximum error e_max({i})={e_max}")

            # Update best solution if better
            if e_max < best_error:
                best_spline = spline
                best_mu = mu
                best_error = e_max

                if config.debug:
                    print(f"  New best solution: μ={mu}, error={e_max}")

            # Step 3: Update lower_bound and upper_bound according to e_max
            if e_max > tolerance:
                # Error is too large, need more accuracy, increase μ
                lower_bound_new = mu
                upper_bound_new = upper_bound
                if config.debug:
                    print(f"  Error > tolerance, updating lower_bound({i + 1})={lower_bound_new}")
            else:
                # Error is acceptable, can try more smoothing
                upper_bound_new = mu
                lower_bound_new = lower_bound
                if config.debug:
                    print(f"  Error ≤ tolerance, updating upper_bound({i + 1})={upper_bound_new}")

            # Update lower_bound and upper_bound for next iteration
            lower_bound = lower_bound_new
            upper_bound = upper_bound_new

            # Check for convergence or solution
            if abs(e_max - tolerance) < EPSILON or (
                e_max < tolerance and upper_bound - lower_bound < EPSILON
            ):
                if config.debug:
                    print(f"\nConverged to solution with error {e_max} after {i + 1} iterations")
                return spline, mu, e_max, i + 1

        except ValueError as e:
            # Handle potential errors with invalid μ values
            if config.debug:
                print(f"  Error with μ={mu}: {e}")
            # If μ caused an error, try a value closer to 1 (more accuracy)
            lower_bound = mu

    if config.debug:
        print(f"\nReached maximum iterations ({config.max_iterations})")
        print(f"Best solution found: μ={best_mu}, error={best_error}")

    # Return the best solution found - this is guaranteed to be non-None now
    return best_spline, best_mu, best_error, config.max_iterations
