"""
B-spline curve approximation with least squares fitting.

This module provides B-spline approximation algorithms that fit curves to datasets
with fewer control points than data points. The approximation balances computational
efficiency with curve quality using least squares optimization.
"""

import numpy as np

from ._approximation_system import ApproximationProblem
from ._approximation_system import approximate_control_points
from ._parameterization import parameterize_points
from .core import BSpline


class ApproximationBSpline(BSpline):
    """A class for B-spline curve approximation of a set of points.

    Inherits from BSpline class.

    The endpoint-constrained least-squares approximation has these properties:
    - The end points are exactly interpolated
    - The internal points are approximated in the least squares sense
    - Degree 3 (cubic) is typically used to ensure C2 continuity

    Attributes
    ----------
    original_points : np.ndarray
        The original points being approximated.
    original_parameters : np.ndarray
        The parameter values corresponding to original points.
    """

    def __init__(  # noqa: PLR0913
        self,
        points: list | np.ndarray,
        num_control_points: int,
        *,  # Make remaining parameters keyword-only
        degree: int = 3,
        weights: list | np.ndarray | None = None,
        method: str = "chord_length",
        debug: bool = False,
    ) -> None:
        """Initialize an approximation B-spline.

        Parameters
        ----------
        points : list or np.ndarray
            The points to approximate.
        num_control_points : int
            The number of control points to use.
        degree : int, default=3
            The degree of the B-spline. Defaults to 3 for cubic.
        weights : list or np.ndarray or None, default=None
            Weights for points in approximation. If None, uniform weighting is used.
        method : str, default="chord_length"
            Method for parameter calculation. Options are 'equally_spaced',
            'chord_length', or 'centripetal'.
        debug : bool, default=False
            Whether to print debug information.

        Raises
        ------
        ValueError
            If inputs do not satisfy approximation requirements.
        """
        # Validate inputs
        if degree < 1:
            raise ValueError("Degree must be at least 1")
        if num_control_points <= degree:
            raise ValueError("Number of control points must be greater than the degree")
        if len(points) <= num_control_points:
            raise ValueError("Number of points must be greater than number of control points")

        # Store debug flag
        self.debug = debug

        if self.debug:
            print("\n" + "=" * 50)
            print("INITIALIZING APPROXIMATION B-SPLINE")
            print(f"Degree: {degree}")
            print(f"Number of control points: {num_control_points}")
            print(f"Number of points to approximate: {len(points)}")
            print(f"Parameterization method: {method}")
            print("=" * 50)

        # Convert points to numpy array if needed
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)

        # Set default weights if not provided (uniform weights)
        n = len(points) - 1  # Number of points minus 1
        if weights is None:
            weights = np.ones(n - 1)  # Exclude first and last points
        elif not isinstance(weights, np.ndarray):
            weights = np.array(weights, dtype=np.float64)

        # Calculate parameter values for the points
        u_bar = self._compute_parameters(points, method)

        # Calculate knot vector
        knots = self._compute_knots(degree, num_control_points, len(points), u_bar)

        # Calculate control points using least squares approximation
        control_points = self._approximate_control_points(
            points, degree, knots, u_bar, num_control_points, weights
        )

        # Initialize parent class with calculated values
        super().__init__(degree, knots, control_points)

        # Store the original points and parameter values for reference
        self.original_points = points
        self.original_parameters = u_bar

        if self.debug:
            print("\nFINAL RESULTS:")
            print(f"Degree: {degree}")
            print(f"Number of control points: {len(control_points)}")
            print(f"Knot vector: {knots}")
            print("\nControl points:")
            for i, cp in enumerate(control_points):
                print(f"  P{i}: {cp}")

    def _compute_parameters(self, points: np.ndarray, method: str = "chord_length") -> np.ndarray:
        """Calculate normalized parameter values for the approximation points."""
        parameters = parameterize_points(points, method)
        if self.debug:
            print(f"\nPARAMETER VALUES (using '{method}' method):")
            for index, parameter in enumerate(parameters):
                print(f"  u_bar[{index}] = {parameter:.6f}")
        return parameters

    def _compute_knots(
        self, degree: int, num_control_points: int, num_points: int, u_bar: np.ndarray
    ) -> np.ndarray:
        """Compute knot vector following the algorithm in Section 8.5.1.

        Parameters
        ----------
        degree : int
            The degree of the B-spline.
        num_control_points : int
            The number of control points.
        num_points : int
            The number of points to approximate.
        u_bar : np.ndarray
            Parameter values for the points.

        Returns
        -------
        np.ndarray
            The knot vector.
        """
        # Total number of knots
        num_knots = num_control_points + degree + 1

        if hasattr(self, "debug") and self.debug:
            print("\nKNOT VECTOR CALCULATION:")
            print(f"  Number of knots needed: {num_knots}")

        # Initialize knot vector
        knots = np.zeros(num_knots)

        # Repeat endpoint knots p+1 times to interpolate the endpoints.
        knots[: degree + 1] = u_bar[0]
        knots[-(degree + 1) :] = u_bar[-1]

        # Compute internal knots using the method from Section 8.5.1
        n = num_points - 1  # Number of points minus 1
        m = num_control_points - 1  # Number of control points minus 1

        # Scale data-parameter indices across the available internal knots.
        d = (n + 1) / (m - degree + 1)

        if hasattr(self, "debug") and self.debug:
            print(f"  d = (n+1)/(m-p+1) = ({n + 1})/({m}-{degree}+1) = {d:.6f}")

        # Compute internal knots by interpolating neighboring data parameters.
        # For j=1,...,m-p compute:
        # i = floor(j*d)
        # a = j*d - i
        # u_(j+p) = (1-a)ū_(i-1) + aū_i
        for j in range(1, m - degree + 1):
            i = int(j * d)  # floor(j*d)
            alpha = j * d - i  # j*d - floor(j*d)
            knots[j + degree] = (1 - alpha) * u_bar[i - 1] + alpha * u_bar[i]

            if hasattr(self, "debug") and self.debug:
                print(f"  j={j}, i=floor({j}*{d:.6f})={i}, a={alpha:.6f}")
                print(
                    f"  u_{j + degree} = (1-{alpha:.6f})*{u_bar[i - 1]:.6f} + "
                    f"{alpha:.6f}*{u_bar[i]:.6f} = {knots[j + degree]:.6f}"
                )

        if hasattr(self, "debug") and self.debug:
            print("\nFINAL KNOT VECTOR:")
            knot_str = "  ["
            for k in knots:
                knot_str += f"{k:.6f}, "
            knot_str = knot_str[:-2] + "]"
            print(knot_str)

        return knots

    def _approximate_control_points(  # noqa: PLR0913
        self,
        points: np.ndarray,
        degree: int,
        knots: np.ndarray,
        u_bar: np.ndarray,
        num_control_points: int,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Compute endpoint-constrained weighted least-squares control points."""
        problem = ApproximationProblem(
            points=points,
            degree=degree,
            knots=knots,
            parameters=u_bar,
            control_point_count=num_control_points,
            weights=weights,
        )
        return approximate_control_points(self, problem)

    def calculate_approximation_error(
        self, points: np.ndarray | None = None, u_bar: np.ndarray | None = None
    ) -> float:
        """Calculate the approximation error as the sum of squared distances.

        Computes sum of squared distances between the points and the corresponding
        points on the B-spline.

        Parameters
        ----------
        points : np.ndarray or None, default=None
            The points to compare with the B-spline. If None, the original points
            used for approximation are used.
        u_bar : np.ndarray or None, default=None
            Parameter values for the points. If None, the original parameters are
            used for original points, or computed for new points.

        Returns
        -------
        float
            The sum of squared distances.
        """
        # Use original points and parameters if not provided
        if points is None:
            points = self.original_points
            u_bar = self.original_parameters

        # Calculate the sum of squared distances
        sum_squared_dist = 0.0
        for i, point in enumerate(points):
            # Evaluate the B-spline at the parameter value
            spline_point = self.evaluate(u_bar[i])  # type: ignore

            # Calculate squared distance
            squared_dist = np.sum((point - spline_point) ** 2)
            sum_squared_dist += squared_dist

        return sum_squared_dist

    def refine(
        self, max_error: float = 0.1, max_control_points: int = 100
    ) -> "ApproximationBSpline":
        """Refine the approximation by adding more control points.

        Adds control points until the maximum error is below a threshold or the
        maximum number of control points is reached.

        Parameters
        ----------
        max_error : float, default=0.1
            Maximum acceptable error.
        max_control_points : int, default=100
            Maximum number of control points.

        Returns
        -------
        ApproximationBSpline
            A refined approximation B-spline.
        """
        # Start with the current number of control points
        num_control_points = len(self.control_points)

        # Calculate initial error
        error = self.calculate_approximation_error()

        while error > max_error and num_control_points < max_control_points:
            # Increase the number of control points
            num_control_points += 1

            # Create a new approximation B-spline with more control points
            new_spline = ApproximationBSpline(
                self.original_points, num_control_points, degree=self.degree
            )

            # Calculate the new error
            error = new_spline.calculate_approximation_error()

            # If error is below threshold, return the new spline
            if error <= max_error:
                return new_spline

        # If we've reached max_control_points but error is still above threshold,
        # return the best approximation we have
        return ApproximationBSpline(
            self.original_points,
            min(num_control_points, max_control_points),
            degree=self.degree,
        )
