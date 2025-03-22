import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.b_spline_smooth import BSplineParams
from interpolatepy.b_spline_smooth import SmoothingCubicBSpline
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def example_8_12() -> None:
    """
    Implementation of Example 8.12 from the document:
    3D cubic B-spline smoothing through given points with different lambda values.
    """

    # Define the points to approximate as given in the example
    points = np.array([[0, 0, 0], [1, 2, 1], [2, 3, 0], [4, 3, 0], [5, 2, 2], [6, 0, 2]])

    # Initial and final tangent vectors as given in the example
    # v0 = np.array([4.43, 8.87, 4.43])
    # vn = np.array([4.85, -9.71, 0])

    # Create three smoothing B-splines with different lambda values
    lambda_values = [1e-4, 1e-5, 1e-6]
    splines = []

    for lambda_val in lambda_values:
        # Convert lambda to mu: lambda = (1-mu)/(6*mu)
        mu = 1 / (6 * lambda_val + 1)

        # Create a BSplineParams object with the desired parameters
        params = BSplineParams(
            mu=mu,
            # v0=v0,
            # vn=vn,
            method="chord_length",
            enforce_endpoints=True,
            auto_derivatives=True,  # Not needed when v0 and vn are provided
        )

        # Create the smoothing B-spline with the parameters object
        spline = SmoothingCubicBSpline(
            points,
            params=params,
        )
        splines.append(spline)

    # Create a figure for visualization
    fig = plt.figure(figsize=(15, 10))

    # Plot the curves
    for i, (spline, lambda_val) in enumerate(zip(splines, lambda_values)):
        # Create a 3D subplot
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")

        # Plot the B-spline curve with control polygon
        spline.plot_3d(num_points=200, show_control_polygon=True, ax=ax)

        # Add the approximation points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="green",
            marker="o",
            s=100,
            label="Approximation points",
        )

        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Lambda = {lambda_val:.6f}\nMu = {spline.mu:.6f}")

        # Add a legend
        ax.legend()

        # Adjust the viewing angle
        ax.view_init(elev=30, azim=45)

        # Print the control points for this spline (as shown in the example)
        print(f"\nControl points for Lambda = {lambda_val}:")
        print(np.array_str(spline.control_points, precision=2))

        # Calculate and print errors
        errors = spline.calculate_approximation_error()
        print(f"Approximation errors: {np.array_str(errors, precision=4)}")
        print(f"Total error: {spline.calculate_total_error():.6f}")
        print(f"Smoothness measure: {spline.calculate_smoothness_measure():.6f}")

    # Plot the second derivatives for comparison
    ax = fig.add_subplot(2, 2, 4)
    u_values = np.linspace(0, 1, 200)

    for _, (spline, lambda_val) in enumerate(zip(splines, lambda_values)):
        # Calculate the magnitude of the second derivative
        d2_magnitudes = []
        for u in u_values:
            d2 = spline.evaluate_derivative(u, order=2)
            d2_magnitudes.append(np.linalg.norm(d2))

        # Plot the second derivative magnitude
        ax.plot(u_values, d2_magnitudes, label=f"Lambda = {lambda_val}")

    ax.set_xlabel("u")
    ax.set_ylabel("||s''(u)||")
    ax.set_title("Magnitude of Second Derivative")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_8_12()
