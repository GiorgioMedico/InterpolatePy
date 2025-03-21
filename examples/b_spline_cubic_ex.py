import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.b_spline_cubic import CubicBSplineInterpolation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def example_8_8() -> None:
    """
    Implementation of Example 8.8 from the document:
    3D cubic B-spline interpolation through the given points.
    """
    # Define the points to interpolate as given in the matrix
    points = np.array(
        [
            [83, -54, 119],
            [-64, 10, 124],
            [42, 79, 226],
            [-98, 23, 222],
            [-13, 125, 102],
            [140, 81, 92],
            [43, 32, 92],
            [-65, -17, 134],
            [-45, -89, 182],
            [71, 90, 192],
        ]
    )

    # Create the cubic B-spline interpolation
    # Use chord length parameterization as recommended for 3D curves
    interpolation = CubicBSplineInterpolation(points, method="chord_length", auto_derivatives=True)

    # Create a figure for 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the B-spline curve with control polygon using inherited method
    interpolation.plot_3d(num_points=200, show_control_polygon=True, ax=ax)

    # Add the interpolation points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="green",
        marker="o",
        s=100,
        label="Interpolation points",
    )

    # Set labels and adjust the view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Cubic B-spline Interpolation (Example 8.8)")

    # Add a legend
    ax.legend()

    # Adjust the viewing angle to better see the 3D shape
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()

    # Print some information about the interpolation
    print("Cubic B-spline Interpolation Information:")
    print(f"Number of interpolation points: {len(points)}")
    print(f"Degree of B-spline: {interpolation.degree}")
    print(f"Control points:\n [{(interpolation.control_points)}]")
    print(f"Knots vector: \n[{interpolation.knots}]")
    print(f"V0: \n[{interpolation.v0}]")
    print(f"Vn: \n[{interpolation.vn}]")


if __name__ == "__main__":
    example_8_8()
