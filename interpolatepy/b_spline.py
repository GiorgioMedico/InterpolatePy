import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - Imported for type hinting


class BSpline:
    """
    A class for representing and evaluating B-spline curves.

    Parameters
    ----------
    degree : int
        The degree of the B-spline.
    knots : array_like
        The knot vector.
    control_points : array_like
        The control points defining the B-spline.

    Attributes
    ----------
    degree : int
        The degree of the B-spline.
    knots : ndarray
        The knot vector.
    control_points : ndarray
        The control points defining the B-spline.
    u_min : float
        Minimum valid parameter value.
    u_max : float
        Maximum valid parameter value.
    dimension : int
        The dimension of the control points (2D, 3D, etc.).
    """

    # Constants for dimension comparisons
    DIM_2 = 2
    DIM_3 = 3

    def __init__(
        self, degree: int, knots: list | np.ndarray, control_points: list | np.ndarray
    ) -> None:
        """
        Initialize a B-spline curve.

        Parameters
        ----------
        degree : int
            The degree of the B-spline (must be positive).
        knots : array_like
            The knot vector (must be non-decreasing).
        control_points : array_like
            The control points defining the B-spline.
            Can be multidimensional (e.g., 2D or 3D points).

        Raises
        ------
        ValueError
            If the inputs do not satisfy B-spline requirements.
        """
        # Convert inputs to numpy arrays if they're not already
        if not isinstance(knots, np.ndarray):
            knots = np.array(knots, dtype=np.float64)
        else:
            # Ensure knots are float64 for precision
            knots = knots.astype(np.float64)

        if not isinstance(control_points, np.ndarray):
            control_points = np.array(control_points, dtype=np.float64)
        else:
            # Ensure control points are float64 for precision
            control_points = control_points.astype(np.float64)

        # Validate inputs
        if degree < 0:
            raise ValueError("Degree must be non-negative")

        if not np.all(np.diff(knots) >= 0):
            raise ValueError("Knot vector must be non-decreasing")

        n_control_points = len(control_points)
        n_knots = len(knots)

        # Check relationship between degree, control points, and knots
        if n_knots != n_control_points + degree + 1:
            raise ValueError(
                f"Invalid knot vector length for the given degree and number of control points. "
                f"Expected {n_control_points + degree + 1}, got {n_knots}. "
                f"The relationship must satisfy: n_knots = n_control_points + degree + 1"
            )

        self.degree = degree
        self.knots = knots
        self.control_points = control_points

        # Store min and max parameter values for convenience
        self.u_min = knots[degree]
        self.u_max = knots[-(degree + 1)]

        # Store the dimension of the control points
        self.dimension = 1 if control_points.ndim == 1 else control_points.shape[1]

        # Precompute knot spans for common parameter values
        self._cached_spans: dict[float, int] = {}

        # Tolerance for floating-point comparisons
        self.eps = 1e-10

    def find_knot_span(self, u: float) -> int:
        """
        Find the knot span index for a given parameter value u.

        Parameters
        ----------
        u : float
            The parameter value.

        Returns
        -------
        int
            The index of the knot span containing u.

        Raises
        ------
        ValueError
            If u is outside the valid parameter range.
        """
        # Check cache first for frequently used values
        if u in self._cached_spans:
            return self._cached_spans[u]

        # Validate parameter range
        if u < self.u_min - self.eps or u > self.u_max + self.eps:
            raise ValueError(f"Parameter u={u} outside valid range [{self.u_min}, {self.u_max}]")

        # Clamp parameter to valid range to handle numerical issues
        u = np.clip(u, self.u_min, self.u_max)

        # Handle endpoint case more efficiently
        if abs(u - self.u_max) <= self.eps:
            # For maximum parameter value, return the last valid span
            span = len(self.knots) - self.degree - 2
            self._cached_spans[u] = span
            return span

        # More efficient binary search implementation
        n = len(self.control_points) - 1
        low = self.degree
        high = n + 1

        # Binary search with simplified condition
        while low < high - 1:
            mid = (low + high) // 2
            if u >= self.knots[mid]:
                low = mid
            else:
                high = mid

        # Cache the result for future use
        self._cached_spans[u] = low
        return low

    def basis_functions(self, u: float, span_index: int) -> np.ndarray:
        """
        Calculate all non-zero basis functions at parameter value u.

        Parameters
        ----------
        u : float
            The parameter value.
        span_index : int
            The knot span index containing u.

        Returns
        -------
        ndarray
            Array of basis function values (p+1 values).
        """
        # Initialize the basis functions array
        n = np.zeros(self.degree + 1, dtype=np.float64)

        # Initialize left and right arrays
        left = np.zeros(self.degree + 1, dtype=np.float64)
        right = np.zeros(self.degree + 1, dtype=np.float64)

        # First basis function is always 1
        n[0] = 1.0

        # For each degree, update the basis functions
        for d in range(1, self.degree + 1):
            left[d] = u - self.knots[span_index + 1 - d]
            right[d] = self.knots[span_index + d] - u

            saved = 0.0

            for r in range(d):
                # Avoid division by zero more robustly
                denominator = right[r + 1] + left[d - r]

                # Use small epsilon for numerical stability - using ternary operator
                temp = 0.0 if abs(denominator) < self.eps else n[r] / denominator

                # Update the basis function using the recurrence relation
                n[r] = saved + right[r + 1] * temp
                saved = left[d - r] * temp

            n[d] = saved

        return n

    def evaluate(self, u: float) -> np.ndarray:
        """
        Evaluate the B-spline curve at parameter value u.

        Parameters
        ----------
        u : float
            The parameter value.

        Returns
        -------
        ndarray
            The point on the B-spline curve at parameter u.
        """
        # Handle edge cases exactly at endpoints to avoid numerical issues
        if abs(u - self.u_min) <= self.eps:
            return self.control_points[0].copy()
        if abs(u - self.u_max) <= self.eps:
            return self.control_points[-1].copy()

        # Clamp parameter to valid range using numpy's clip function
        u = np.clip(u, self.u_min, self.u_max)

        # Find the knot span
        span = self.find_knot_span(u)

        # Calculate the basis functions
        n = self.basis_functions(u, span)

        # Calculate the point by multiplying basis functions with control points
        # Use advanced indexing for efficiency
        relevant_controls = self.control_points[span - self.degree : span + 1]

        # Handle 1D control points differently
        if self.dimension == 1:
            point = np.dot(n, relevant_controls)
        else:
            # For multi-dimensional control points
            point = np.zeros(self.dimension, dtype=np.float64)
            for i in range(self.degree + 1):
                point += n[i] * relevant_controls[i]

        return point

    def basis_function_derivatives(self, u: float, span_index: int, order: int) -> np.ndarray:
        """
        Calculate derivatives of basis functions up to the specified order.

        Parameters
        ----------
        u : float
            The parameter value.
        span_index : int
            The knot span index containing u.
        order : int
            The maximum order of derivatives to calculate.

        Returns
        -------
        ndarray
            2D array where ders[k][j] is the k-th derivative of the j-th basis function.
        """
        # Ensure order doesn't exceed degree
        order = min(order, self.degree)

        # Initialize the result array
        ders = np.zeros((order + 1, self.degree + 1), dtype=np.float64)

        # Initialize temporary arrays
        left = np.zeros(self.degree + 1, dtype=np.float64)
        right = np.zeros(self.degree + 1, dtype=np.float64)
        a = np.zeros((2, self.degree + 1), dtype=np.float64)

        # Initialize the basis function table (ndu)
        ndu = np.zeros((self.degree + 1, self.degree + 1), dtype=np.float64)
        ndu[0, 0] = 1.0

        for j in range(1, self.degree + 1):
            left[j] = u - self.knots[span_index + 1 - j]
            right[j] = self.knots[span_index + j] - u
            saved = 0.0

            for r in range(j):
                # Lower triangle
                ndu[j, r] = right[r + 1] + left[j - r]

                # Avoid division by zero - using ternary operator
                temp = 0.0 if abs(ndu[j, r]) < self.eps else ndu[r, j - 1] / ndu[j, r]

                # Upper triangle
                ndu[r, j] = saved + right[r + 1] * temp
                saved = left[j - r] * temp

            ndu[j, j] = saved

        # Load the basis functions
        for j in range(self.degree + 1):
            ders[0, j] = ndu[j, self.degree]

        # Calculate the derivatives
        for r in range(self.degree + 1):
            # Index of current column in the table
            s1 = 0
            s2 = 1

            # Initialize a array
            a[0, 0] = 1.0

            # Loop to compute k-th derivative
            for k in range(1, order + 1):
                d = 0.0
                rk = r - k
                pk = self.degree - k

                if r >= k:
                    a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                    d = a[s2, 0] * ndu[rk, pk]

                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if r - 1 <= pk else self.degree - r

                for j in range(j1, j2 + 1):
                    a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                    d += a[s2, j] * ndu[rk + j, pk]

                if r <= pk:
                    a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                    d += a[s2, k] * ndu[r, pk]

                ders[k, r] = d

                # Switch row indices
                j = s1
                s1 = s2
                s2 = j

        # Multiply by the correct factors (p!/(p-k)!)
        r = self.degree
        for k in range(1, order + 1):
            for j in range(self.degree + 1):
                ders[k, j] *= r
            r *= self.degree - k

        return ders

    def evaluate_derivative(self, u: float, order: int = 1) -> np.ndarray:
        """
        Evaluate the derivative of the B-spline curve at parameter value u.

        Parameters
        ----------
        u : float
            The parameter value.
        order : int, optional
            The order of the derivative. Default is 1.

        Returns
        -------
        ndarray
            The derivative of the B-spline curve at parameter u.

        Raises
        ------
        ValueError
            If the order is greater than the degree of the B-spline.
        """
        if order > self.degree:
            raise ValueError(f"Derivative order {order} exceeds B-spline degree {self.degree}")

        if order == 0:
            return self.evaluate(u)

        # Clamp parameter to valid range
        u = np.clip(u, self.u_min + self.eps, self.u_max - self.eps)

        # Find the knot span
        span = self.find_knot_span(u)

        # Calculate derivatives of basis functions
        ders = self.basis_function_derivatives(u, span, order)

        # Calculate the derivative of the curve more efficiently
        relevant_controls = self.control_points[span - self.degree : span + 1]

        # Handle 1D control points differently
        if self.dimension == 1:
            derivative = np.dot(ders[order], relevant_controls)
        else:
            derivative = np.zeros(self.dimension, dtype=np.float64)
            for j in range(self.degree + 1):
                derivative += ders[order, j] * relevant_controls[j]

        return derivative

    def generate_curve_points(self, num_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate points along the B-spline curve for visualization.

        Parameters
        ----------
        num_points : int, optional
            Number of points to generate. Default is 100.

        Returns
        -------
        u_values : ndarray
            Parameter values.
        curve_points : ndarray
            Corresponding points on the curve.
        """
        # Generate parameter values evenly spaced in the valid range
        u_values = np.linspace(self.u_min, self.u_max, num_points)

        # Vectorize for better performance if curve is 1D or 2D
        if self.dimension <= self.DIM_2:
            # Pre-allocate array for curve points
            curve_points = np.zeros((num_points, self.dimension), dtype=np.float64)

            # Evaluate the curve at each parameter value
            for i, u in enumerate(u_values):
                curve_points[i] = self.evaluate(u)
        else:
            # For higher dimensions, use list comprehension
            curve_points = np.array([self.evaluate(u) for u in u_values])

        return u_values, curve_points

    def plot_2d(
        self,
        num_points: int = 100,
        show_control_polygon: bool = True,
        show_knots: bool = False,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot a 2D B-spline curve with customizable styling.

        Parameters
        ----------
        num_points : int, optional
            Number of points to generate for the curve. Default is 100.
        show_control_polygon : bool, optional
            Whether to show the control polygon. Default is True.
        show_knots : bool, optional
            Whether to show the knot points on the curve. Default is False.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis to use for plotting. If None, a new figure is created.
        curve_style : dict, optional
            Style parameters for the curve.
        control_style : dict, optional
            Style parameters for the control polygon.
        knot_style : dict, optional
            Style parameters for the knot points.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axis object.

        Raises
        ------
        ValueError
            If the dimension of control points is not 2.
        """
        if self.dimension != self.DIM_2:
            raise ValueError(
                f"Control points must be 2D for this plot function, got {self.dimension}D"
            )

        # Default styles
        default_curve_style = {"color": "blue", "linewidth": 2, "label": "B-spline curve"}
        default_control_style = {
            "color": "red",
            "linestyle": "--",
            "marker": "o",
            "linewidth": 1,
            "markersize": 8,
            "label": "Control polygon",
        }
        default_knot_style = {
            "color": "green",
            "marker": "x",
            "markersize": 10,
            "label": "Knot points",
        }

        # Create a new figure if needed
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Generate points along the curve
        _, curve_points = self.generate_curve_points(num_points)

        # Plot the curve
        ax.plot(curve_points[:, 0], curve_points[:, 1], **default_curve_style)

        # Plot the control points and polygon if requested
        if show_control_polygon:
            ax.plot(self.control_points[:, 0], self.control_points[:, 1], **default_control_style)

        # Plot the knot points if requested
        if show_knots:
            # Only plot the knots within the valid parameter range
            valid_knots = [k for k in self.knots if self.u_min <= k <= self.u_max]
            unique_knots = np.unique(valid_knots)

            knot_points = np.array([self.evaluate(k) for k in unique_knots])
            ax.plot(knot_points[:, 0], knot_points[:, 1], **default_knot_style)

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"B-spline curve of degree {self.degree}")
        ax.legend()
        ax.grid(True)

        return ax

    def plot_3d(
        self,
        num_points: int = 100,
        show_control_polygon: bool = True,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot a 3D B-spline curve.

        Parameters
        ----------
        num_points : int, optional
            Number of points to generate for the curve. Default is 100.
        show_control_polygon : bool, optional
            Whether to show the control polygon. Default is True.
        ax : matplotlib.axes.Axes, optional
            Matplotlib 3D axis to use for plotting. If None, a new figure is created.
        curve_style : dict, optional
            Style parameters for the curve.
        control_style : dict, optional
            Style parameters for the control polygon.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib 3D axis object.

        Raises
        ------
        ValueError
            If the dimension of control points is not 3.
        """
        if self.dimension != self.DIM_3:
            raise ValueError(
                f"Control points must be 3D for this plot function, got {self.dimension}D"
            )

        # Default styles
        default_curve_style = {"color": "blue", "linewidth": 2, "label": "B-spline curve"}
        default_control_style = {
            "color": "red",
            "linestyle": "--",
            "marker": "o",
            "linewidth": 1,
            "markersize": 8,
            "label": "Control polygon",
        }

        # Create a new figure if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

        # Generate points along the curve
        _, curve_points = self.generate_curve_points(num_points)

        # Plot the curve
        ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2], **default_curve_style)

        # Plot the control points and polygon if requested
        if show_control_polygon:
            ax.plot(
                self.control_points[:, 0],
                self.control_points[:, 1],
                self.control_points[:, 2],
                **default_control_style,
            )

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"3D B-spline curve of degree {self.degree}")
        ax.legend()

        return ax

    @staticmethod
    def create_uniform_knots(
        degree: int, num_control_points: int, domain_min: float = 0.0, domain_max: float = 1.0
    ) -> np.ndarray:
        """
        Create a uniform knot vector for a B-spline with appropriate
        multiplicity at endpoints to ensure interpolation.

        Parameters
        ----------
        degree : int
            The degree of the B-spline.
        num_control_points : int
            The number of control points.
        domain_min : float, optional
            Minimum value of the parameter domain. Default is 0.0.
        domain_max : float, optional
            Maximum value of the parameter domain. Default is 1.0.

        Returns
        -------
        ndarray
            The uniform knot vector with knots in the specified domain.

        Raises
        ------
        ValueError
            If degree is negative or num_control_points is less than or equal to degree.
        """
        # Input validation
        if degree < 0:
            raise ValueError("Degree must be non-negative")
        if num_control_points <= degree:
            raise ValueError("Number of control points must be greater than the degree")

        # Calculate total number of knots required by the formula: n = m + p + 1
        # where n is number of knots, m is number of control points, p is degree
        num_knots = num_control_points + degree + 1

        # Initialize the knot vector with domain_min values (for the first p+1 knots)
        knots = np.zeros(num_knots, dtype=np.float64)

        # Calculate the number of internal knots needed
        n_internal = num_knots - 2 * (degree + 1)

        # Set the first degree+1 knots to domain_min
        knots[: degree + 1] = domain_min

        # Create internal knots uniformly distributed (if any)
        if n_internal >= 0:
            # Generate values between domain_min and domain_max, excluding endpoints
            internal_values = np.linspace(domain_min, domain_max, n_internal + 2)[1:-1]
            knots[degree + 1 : degree + 1 + n_internal] = internal_values

        # Set end knots to domain_max with multiplicity p+1
        knots[-(degree + 1) :] = domain_max

        return knots

    @staticmethod
    def create_periodic_knots(
        degree: int, num_control_points: int, domain_min: float = 0.0, domain_max: float = 1.0
    ) -> np.ndarray:
        """
        Create a periodic (uniform) knot vector for a B-spline.

        Parameters
        ----------
        degree : int
            The degree of the B-spline.
        num_control_points : int
            The number of control points.
        domain_min : float, optional
            Minimum value of the parameter domain. Default is 0.0.
        domain_max : float, optional
            Maximum value of the parameter domain. Default is 1.0.

        Returns
        -------
        ndarray
            The periodic knot vector.

        Raises
        ------
        ValueError
            If degree is negative or num_control_points is less than degree+1.
        """
        # Input validation
        if degree < 0:
            raise ValueError("Degree must be non-negative")
        if num_control_points < degree + 1:
            raise ValueError(
                "For a periodic B-spline, number of control points must be at least degree+1"
            )

        # Calculate total number of knots required
        num_knots = num_control_points + degree + 1

        # Create uniformly spaced knots
        knots = np.linspace(domain_min, domain_max, num_knots - 2 * degree)

        # Extend the knot vector for periodicity
        extended_knots = np.zeros(num_knots, dtype=np.float64)

        # Add degree knots at the beginning (below domain_min)
        step = (domain_max - domain_min) / (num_knots - 2 * degree - 1)
        for i in range(degree):
            extended_knots[i] = domain_min - (degree - i) * step

        # Add the regular knots
        extended_knots[degree : degree + len(knots)] = knots

        # Add degree knots at the end (above domain_max)
        for i in range(degree):
            extended_knots[degree + len(knots) + i] = domain_max + (i + 1) * step

        return extended_knots

    def __repr__(self) -> str:
        """
        Return a string representation of the B-spline.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"BSpline(degree={self.degree}, "
            f"control_points={len(self.control_points)}, "
            f"dimension={self.dimension})"
        )
