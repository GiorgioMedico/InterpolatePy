import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.b_spline_interpolate import BSplineInterpolator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# Example 1: Cubic B-spline interpolation with velocity constraints (Fig. 4.18)
def example_cubic_bspline() -> None:
    """
    Recreate example 4.16 from the document (page 198).
    Cubic B-spline curve with velocity constraints.
    """

    # Data from Example 4.16
    times = np.array([0, 5, 7, 8, 10, 15, 18])
    points = np.array([3, -2, -5, 0, 6, 12, 8])

    # Create a degree 3 interpolator with velocity constraints
    interpolator = BSplineInterpolator(
        degree=3, points=points, times=times, initial_velocity=2, final_velocity=-3
    )

    # Generate curve points for plotting
    t_values = np.linspace(times[0], times[-1], 500)
    positions = np.array([interpolator.evaluate(t) for t in t_values])
    velocities = np.array([interpolator.evaluate_derivative(t, 1) for t in t_values])
    accelerations = np.array([interpolator.evaluate_derivative(t, 2) for t in t_values])

    # Plot the results
    _, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Position
    axs[0].plot(t_values, positions)
    axs[0].plot(times, points, "ro")
    axs[0].set_ylabel("Position")
    axs[0].grid(True)

    # Velocity
    axs[1].plot(t_values, velocities)
    axs[1].set_ylabel("Velocity")
    axs[1].grid(True)

    # Acceleration
    axs[2].plot(t_values, accelerations)
    axs[2].set_ylabel("Acceleration")
    axs[2].set_xlabel("Time")
    axs[2].grid(True)

    plt.tight_layout()
    plt.suptitle("Cubic B-spline with Velocity Constraints", fontsize=16)
    plt.subplots_adjust(top=0.93)
    plt.show()


# Example 2: B-spline with jerk continuity (Fig. 4.18)
def example_jerk_continuous_bspline() -> None:
    """
    Recreate example from the document (page 198).
    B-spline of degree 4 with jerk continuity.
    """

    # Data from Example 4.16
    times = np.array([0, 5, 7, 8, 10, 15, 18])
    points = np.array([3, -2, -5, 0, 6, 12, 8])

    # Create a degree 4 interpolator with velocity and acceleration constraints
    interpolator = BSplineInterpolator(
        degree=4,
        points=points,
        times=times,
        initial_velocity=2,
        final_velocity=-3,
        initial_acceleration=0,
        final_acceleration=0,
    )

    # Generate curve points for plotting
    t_values = np.linspace(times[0], times[-1], 500)
    positions = np.array([interpolator.evaluate(t) for t in t_values])
    velocities = np.array([interpolator.evaluate_derivative(t, 1) for t in t_values])
    accelerations = np.array([interpolator.evaluate_derivative(t, 2) for t in t_values])
    jerks = np.array([interpolator.evaluate_derivative(t, 3) for t in t_values])

    # Plot the results
    _, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Position
    axs[0].plot(t_values, positions)
    axs[0].plot(times, points, "ro")
    axs[0].set_ylabel("Position")
    axs[0].grid(True)

    # Velocity
    axs[1].plot(t_values, velocities)
    axs[1].set_ylabel("Velocity")
    axs[1].grid(True)

    # Acceleration
    axs[2].plot(t_values, accelerations)
    axs[2].set_ylabel("Acceleration")
    axs[2].grid(True)

    # Jerk
    axs[3].plot(t_values, jerks)
    axs[3].set_ylabel("Jerk")
    axs[3].set_xlabel("Time")
    axs[3].grid(True)

    plt.tight_layout()
    plt.suptitle("B-spline with Continuous Jerk (Degree 4)", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()


# Example 3: Cyclic B-spline (Fig. 4.21)
def example_cyclic_bspline() -> None:
    """
    Recreate example 4.17 from the document (page 202).
    Cyclic B-spline of degree 4.
    """

    # Data from Example 4.17
    times = np.array([0, 5, 7, 8, 10, 15, 18])
    points = np.array([3, -2, -5, 0, 6, 12, 3])  # Note last point = first point

    # Create a degree 4 cyclic interpolator
    interpolator = BSplineInterpolator(degree=4, points=points, times=times, cyclic=True)

    # Generate curve points for plotting
    t_values = np.linspace(times[0], times[-1], 500)
    positions = np.array([interpolator.evaluate(t) for t in t_values])
    velocities = np.array([interpolator.evaluate_derivative(t, 1) for t in t_values])
    accelerations = np.array([interpolator.evaluate_derivative(t, 2) for t in t_values])
    jerks = np.array([interpolator.evaluate_derivative(t, 3) for t in t_values])
    snaps = np.array([interpolator.evaluate_derivative(t, 4) for t in t_values])

    # Plot the results
    _, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    # Position
    axs[0].plot(t_values, positions)
    axs[0].plot(times, points, "ro")
    axs[0].set_ylabel("Position")
    axs[0].grid(True)

    # Velocity
    axs[1].plot(t_values, velocities)
    axs[1].set_ylabel("Velocity")
    axs[1].grid(True)

    # Acceleration
    axs[2].plot(t_values, accelerations)
    axs[2].set_ylabel("Acceleration")
    axs[2].grid(True)

    # Jerk
    axs[3].plot(t_values, jerks)
    axs[3].set_ylabel("Jerk")
    axs[3].grid(True)

    # Snap
    axs[4].plot(t_values, snaps)
    axs[4].set_ylabel("Snap")
    axs[4].set_xlabel("Time")
    axs[4].grid(True)

    plt.tight_layout()
    plt.suptitle("Cyclic B-spline (Degree 4)", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()


# Example 4: 3D B-spline interpolation
def simple_3d_example() -> None:
    """
    Demonstrate a simple 3D B-spline interpolation with a basic curved path.
    """

    # Define simple 3D points to interpolate (a curved path in 3D)
    times = [0, 1, 2, 3, 4]
    points = np.array(
        [
            [0, 0, 0],  # Start point
            [1, 1, 2],  # Point 1
            [2, 0, 3],  # Point 2
            [3, -1, 2],  # Point 3
            [4, 0, 0],  # End point
        ]
    )

    # Demonstrate with different degrees
    try:
        degree = 3  # This should work fine with 5 points
        print(f"\nTrying to create a B-spline with degree {degree} using {len(points)} points...")
        interpolator = BSplineInterpolator(degree=degree, points=points, times=times)

        # Plot the 3D curve with interpolation points
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        interpolator.plot_with_points_3d(ax=ax)

        # Set equal aspect ratio for better visualization
        ax.set_box_aspect([1, 1, 1])

        # Set nice viewing angle
        ax.view_init(elev=30, azim=45)

        # Add title and show
        ax.set_title(f"3D B-Spline Interpolation (Degree {degree})")
        plt.tight_layout()
        plt.show()

        # Print some information about the interpolation
        print("\nInterpolation successful!")
        print(f"Number of points: {len(points)}")
        print(f"Degree of B-spline: {degree}")
        print(
            f"Continuity: C^{degree - 1} (continuous "
            f"{['position', 'velocity', 'acceleration', 'jerk', 'snap'][min(degree - 1, 4)]})"
        )

        # Print the original points and a few interpolated points
        print("\nOriginal points to interpolate:")
        for i, point in enumerate(points):
            print(f"Point {i} (t={times[i]}): ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")

        print("\nInterpolated points:")
        for t in [0.5, 1.5, 2.5, 3.5]:
            point = interpolator.evaluate(t)
            print(f"t = {t}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")

    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Try with degree 4 - this will likely show a warning or fail
        degree = 4
        print(f"\nTrying to create a B-spline with degree {degree} using {len(points)} points...")
        interpolator = BSplineInterpolator(degree=degree, points=points, times=times)
        print(f"Succeeded with degree {degree}!")
    except ValueError as e:
        print(f"Failed with degree {degree}: {e}")

    try:
        # Try with degree 5 - this should fail
        degree = 5
        print(f"\nTrying to create a B-spline with degree {degree} using {len(points)} points...")
        interpolator = BSplineInterpolator(degree=degree, points=points, times=times)
        print(f"Succeeded with degree {degree}!")
    except ValueError as e:
        print(f"Failed with degree {degree}: {e}")

    # Now try with more points for higher degrees
    print("\nAdding more points to support higher degree splines...")

    # Define more points for higher degree interpolation
    times_extended = [0, 1, 2, 3, 4, 5, 6]
    points_extended = np.array(
        [
            [0, 0, 0],  # Start point
            [1, 1, 2],  # Point 1
            [2, 0, 3],  # Point 2
            [3, -1, 2],  # Point 3
            [4, 0, 0],  # Point 4
            [5, 1, -1],  # Point 5
            [6, 0, -2],  # Point 6
        ]
    )

    try:
        # Try with degree 5 and more points
        degree = 5
        print(
            f"\nTrying to create a B-spline with degree {degree} "
            f"using {len(points_extended)} points..."
        )
        interpolator = BSplineInterpolator(
            degree=degree, points=points_extended, times=times_extended
        )

        # Plot the 3D curve with interpolation points
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        interpolator.plot_with_points_3d(ax=ax)

        # Set equal aspect ratio for better visualization
        ax.set_box_aspect([1, 1, 1])

        # Set nice viewing angle
        ax.view_init(elev=30, azim=45)

        # Add title and show
        ax.set_title(f"3D B-Spline Interpolation (Degree {degree})")
        plt.tight_layout()
        plt.show()

        print(
            f"Successfully created a degree {degree} B-spline with {len(points_extended)} points!"
        )
    except ValueError as e:
        print(f"Failed with degree {degree} and {len(points_extended)} points: {e}")


if __name__ == "__main__":
    example_cubic_bspline()
    example_jerk_continuous_bspline()
    example_cyclic_bspline()
    simple_3d_example()
