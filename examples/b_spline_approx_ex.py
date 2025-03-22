from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.b_spline import BSpline
from interpolatepy.b_spline_approx import ApproximationBSpline


def example_approximation(debug: bool = False) -> None:
    """
    Demonstrate B-spline approximation with the example from Section 8.5.

    Args:
        debug (bool, optional): Whether to print debug information. Defaults to False.
    """
    print("\n" + "=" * 80)
    print("RUNNING EXAMPLE FROM SECTION 8.5")
    print("=" * 80)

    # Create some test points (similar to the example in the PDF)
    # Extract x,y coordinates from the control points in the example 8.10
    x = [137, 101, 177, 93, 62, 49, 104, 141, 147, 138]
    y = [229, 201, 121, 44, 203, 272, 402, 277, 258, 231]

    print("\nControl Points from Example 8.10:")
    for i in range(len(x)):
        print(f"  P{i}: ({x[i]}, {y[i]})")

    # Convert the control points to numpy array
    control_points = np.column_stack((x, y))

    # Create a BSpline for generating sample points
    # Using a cubic B-spline (degree=3) with uniform knots
    degree = 3
    knots = BSpline.create_uniform_knots(degree, len(control_points))
    interpolation_spline = BSpline(degree, knots, control_points)

    # Generate sample points using the B-spline
    # Explicit unpacking to help type checker
    result = interpolation_spline.generate_curve_points(84)
    sample_points = result[1]  # Access the second element directly

    print(f"\nGenerated {len(sample_points)} points using BSpline")

    # Try different numbers of control points and degrees
    # Define test cases with explicit types
    test_cases: list[dict[str, Any]] = [
        {"num_cps": 10, "degree": 3, "title": "Cubic (p=3) with 10 control points"},
        {"num_cps": 10, "degree": 4, "title": "Quartic (p=4) with 10 control points"},
        {"num_cps": 20, "degree": 3, "title": "Cubic (p=3) with 20 control points"},
    ]

    plt.figure(figsize=(15, 5))

    for i, case in enumerate(test_cases):
        # Extract directly without casting - let Python handle the types
        num_cps = case["num_cps"]  # Already an integer
        degree = case["degree"]  # Already an integer
        title = case["title"]

        print(f"\n{'-' * 80}")
        print(f"TEST CASE {i + 1}: {title}")
        print(f"{'-' * 80}")

        # Create the approximation B-spline
        approx_spline = ApproximationBSpline(sample_points, num_cps, degree=degree, debug=debug)

        # Calculate error
        error = approx_spline.calculate_approximation_error()
        print(f"Approximation error: {error:.2f}")

        # Print knots and control points
        print(f"\nKnot vector (length={len(approx_spline.knots)}):")
        knot_str = "  ["
        for k in approx_spline.knots:
            knot_str += f"{k:.4f}, "
        knot_str = knot_str[:-2] + "]"
        print(knot_str)

        print(f"\nControl points (count={len(approx_spline.control_points)}):")
        for j, cp in enumerate(approx_spline.control_points):
            print(f"  P{j}: {cp}")

        # Plot in a subplot
        plt.subplot(1, 3, i + 1)

        # Plot original sample points
        plt.plot(
            sample_points[:, 0], sample_points[:, 1], "k.", markersize=4, label="Sample points"
        )

        # Generate and plot the B-spline curve
        curve_result = approx_spline.generate_curve_points(100)
        curve_points = curve_result[1]  # Access second element directly

        plt.plot(
            curve_points[:, 0],
            curve_points[:, 1],
            "b-",
            linewidth=2,
            label="Approximation B-spline",
        )

        # Plot control polygon
        plt.plot(
            approx_spline.control_points[:, 0],
            approx_spline.control_points[:, 1],
            "r--",
            marker="x",
            markersize=8,
            label="Control polygon",
        )

        plt.title(f"{title}\nError: {error:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def create_test_shapes() -> dict[str, np.ndarray]:
    """Create a dictionary of test shapes for B-spline approximation.

    Returns:
        Dict[str, np.ndarray]: Dictionary of named test shapes.
    """
    shapes = {}

    # 1. Circle
    t = np.linspace(0, 2 * np.pi, 100)
    circle_x = 100 * np.cos(t) + 150
    circle_y = 100 * np.sin(t) + 150
    shapes["Circle"] = np.column_stack((circle_x, circle_y))

    # 2. Figure-8 (lemniscate)
    t = np.linspace(0, 2 * np.pi, 100)
    a = 100
    figure8_x = a * np.cos(t) / (1 + np.sin(t) ** 2) + 150
    figure8_y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2) + 150
    shapes["Figure-8"] = np.column_stack((figure8_x, figure8_y))

    # 3. Spiral
    t = np.linspace(0, 6 * np.pi, 100)
    a = 5
    b = 15
    spiral_x = (a + b * t) * np.cos(t) + 150
    spiral_y = (a + b * t) * np.sin(t) + 150
    shapes["Spiral"] = np.column_stack((spiral_x, spiral_y))

    # 4. Heart shape
    t = np.linspace(0, 2 * np.pi, 100)
    heart_x = 16 * np.sin(t) ** 3 * 10 + 150
    heart_y = (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) * 10 + 150
    shapes["Heart"] = np.column_stack((heart_x, heart_y))

    return shapes


def example_different_shapes(debug: bool = False) -> None:
    """
    Demonstrate B-spline approximation on different shapes.

    Args:
        debug (bool, optional): Whether to print debug information. Defaults to False.
    """
    print("\n" + "=" * 80)
    print("RUNNING DIFFERENT SHAPES EXAMPLE")
    print("=" * 80)

    # Create test shapes
    test_shapes = create_test_shapes()

    # For each shape, create approximations with different control points
    for shape_name, sample_points in test_shapes.items():
        print(f"\n{'-' * 80}")
        print(f"SHAPE: {shape_name}")
        print(f"{'-' * 80}")

        # Try different numbers of control points
        control_point_counts = [5, 10, 20]
        degree = 3  # Use cubic splines for all examples

        plt.figure(figsize=(15, 5))

        for i, num_cps in enumerate(control_point_counts):
            print(f"\nUsing {num_cps} control points:")

            # Create the approximation B-spline
            approx_spline = ApproximationBSpline(sample_points, num_cps, degree=degree, debug=debug)

            # Calculate error
            error = approx_spline.calculate_approximation_error()
            print(f"  Approximation error: {error:.2f}")

            # Plot in a subplot
            plt.subplot(1, 3, i + 1)

            # Plot original sample points
            plt.plot(
                sample_points[:, 0], sample_points[:, 1], "k.", markersize=4, label="Sample points"
            )

            # Generate and plot the B-spline curve
            curve_result = approx_spline.generate_curve_points(100)
            curve_points = curve_result[1]

            plt.plot(
                curve_points[:, 0],
                curve_points[:, 1],
                "b-",
                linewidth=2,
                label="Approximation B-spline",
            )

            # Plot control polygon
            plt.plot(
                approx_spline.control_points[:, 0],
                approx_spline.control_points[:, 1],
                "r--",
                marker="x",
                markersize=8,
                label="Control polygon",
            )

            plt.title(f"{shape_name} with {num_cps} CPs\nError: {error:.2f}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()


def example_method_comparison(debug: bool = False) -> None:
    """
    Demonstrate B-spline approximation with different methods.

    Since the ApproximationBSpline class doesn't support a direct parameterization_method
    parameter, this example focuses on comparing different approximation methods
    (e.g., by varying the number of control points).

    Args:
        debug (bool, optional): Whether to print debug information. Defaults to False.
    """
    print("\n" + "=" * 80)
    print("RUNNING METHOD COMPARISON EXAMPLE")
    print("=" * 80)

    # Create a test shape - use the spiral which has varying curvature
    shapes = create_test_shapes()
    sample_points = shapes["Spiral"]

    # Try different numbers of control points for comparison
    control_point_counts = [8, 15, 25]
    degree = 3

    plt.figure(figsize=(15, 5))

    for i, num_cps in enumerate(control_point_counts):
        print(f"\n{'-' * 80}")
        print(f"CONTROL POINTS: {num_cps}")
        print(f"{'-' * 80}")

        # Create the approximation B-spline
        approx_spline = ApproximationBSpline(sample_points, num_cps, degree=degree, debug=debug)

        # Calculate error
        error = approx_spline.calculate_approximation_error()
        print(f"Approximation error: {error:.2f}")

        # Plot in a subplot
        plt.subplot(1, 3, i + 1)

        # Plot original sample points
        plt.plot(
            sample_points[:, 0], sample_points[:, 1], "k.", markersize=4, label="Sample points"
        )

        # Generate and plot the B-spline curve
        curve_result = approx_spline.generate_curve_points(100)
        curve_points = curve_result[1]

        plt.plot(
            curve_points[:, 0],
            curve_points[:, 1],
            "b-",
            linewidth=2,
            label="Approximation B-spline",
        )

        # Plot control polygon
        plt.plot(
            approx_spline.control_points[:, 0],
            approx_spline.control_points[:, 1],
            "r--",
            marker="x",
            markersize=8,
            label="Control polygon",
        )

        plt.title(f"CPs: {num_cps}\nError: {error:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def example_degree_comparison(debug: bool = False) -> None:
    """
    Demonstrate B-spline approximation with different degrees.

    Args:
        debug (bool, optional): Whether to print debug information. Defaults to False.
    """
    print("\n" + "=" * 80)
    print("RUNNING DEGREE COMPARISON EXAMPLE")
    print("=" * 80)

    # Create a test shape - use the heart
    shapes = create_test_shapes()
    sample_points = shapes["Heart"]

    # Try different degrees
    degrees = [2, 3, 4]  # quadratic, cubic, quartic
    num_cps = 12

    plt.figure(figsize=(15, 5))

    for i, degree in enumerate(degrees):
        print(f"\n{'-' * 80}")
        print(f"DEGREE: {degree}")
        print(f"{'-' * 80}")

        # Create the approximation B-spline with the specific degree
        approx_spline = ApproximationBSpline(sample_points, num_cps, degree=degree, debug=debug)

        # Calculate error
        error = approx_spline.calculate_approximation_error()
        print(f"Approximation error: {error:.2f}")

        # Plot in a subplot
        plt.subplot(1, 3, i + 1)

        # Plot original sample points
        plt.plot(
            sample_points[:, 0], sample_points[:, 1], "k.", markersize=4, label="Sample points"
        )

        # Generate and plot the B-spline curve
        curve_result = approx_spline.generate_curve_points(100)
        curve_points = curve_result[1]

        plt.plot(
            curve_points[:, 0],
            curve_points[:, 1],
            "b-",
            linewidth=2,
            label="Approximation B-spline",
        )

        # Plot control polygon
        plt.plot(
            approx_spline.control_points[:, 0],
            approx_spline.control_points[:, 1],
            "r--",
            marker="x",
            markersize=8,
            label="Control polygon",
        )

        plt.title(f"Degree: {degree}\nError: {error:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def example_noise_sensitivity(debug: bool = False) -> None:
    """
    Demonstrate B-spline approximation's sensitivity to noise.

    This example shows how B-spline approximation handles noisy data
    by adding random noise to a circle and comparing approximation quality.

    Args:
        debug (bool, optional): Whether to print debug information. Defaults to False.
    """
    print("\n" + "=" * 80)
    print("RUNNING NOISE SENSITIVITY EXAMPLE")
    print("=" * 80)

    # Create a circle
    t = np.linspace(0, 2 * np.pi, 100)
    circle_x = 100 * np.cos(t) + 150
    circle_y = 100 * np.sin(t) + 150
    clean_points = np.column_stack((circle_x, circle_y))

    # Create versions with different noise levels
    np.random.seed(42)  # For reproducibility
    noise_levels = [0, 5, 15]
    noise_points = []

    for noise in noise_levels:
        if noise == 0:
            noise_points.append(clean_points)
        else:
            # Add Gaussian noise
            noisy_x = circle_x + np.random.normal(0, noise, len(circle_x))
            noisy_y = circle_y + np.random.normal(0, noise, len(circle_y))
            noise_points.append(np.column_stack((noisy_x, noisy_y)))

    plt.figure(figsize=(15, 5))

    for i, (noise, points) in enumerate(zip(noise_levels, noise_points)):
        print(f"\n{'-' * 80}")
        print(f"NOISE LEVEL: {noise}")
        print(f"{'-' * 80}")

        # Create the approximation B-spline
        num_cps = 12
        degree = 3
        approx_spline = ApproximationBSpline(points, num_cps, degree=degree, debug=debug)

        # Calculate error
        error = approx_spline.calculate_approximation_error()
        print(f"Approximation error: {error:.2f}")

        # Plot in a subplot
        plt.subplot(1, 3, i + 1)

        # Plot original sample points
        plt.plot(points[:, 0], points[:, 1], "k.", markersize=4, label="Sample points")

        # Generate and plot the B-spline curve
        curve_result = approx_spline.generate_curve_points(100)
        curve_points = curve_result[1]

        plt.plot(
            curve_points[:, 0],
            curve_points[:, 1],
            "b-",
            linewidth=2,
            label="Approximation B-spline",
        )

        # Plot control polygon
        plt.plot(
            approx_spline.control_points[:, 0],
            approx_spline.control_points[:, 1],
            "r--",
            marker="x",
            markersize=8,
            label="Control polygon",
        )

        plt.title(f"Noise: {noise}\nError: {error:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the example with debug output
    import argparse

    parser = argparse.ArgumentParser(description="Approximation B-Spline Examples")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--example",
        choices=["basic", "shapes", "methods", "degrees", "noise", "all"],
        default="all",
        help="Which example to run",
    )

    args = parser.parse_args()

    if args.example in {"basic", "all"}:
        example_approximation(debug=args.debug)

    if args.example in {"shapes", "all"}:
        example_different_shapes(debug=args.debug)

    if args.example in {"methods", "all"}:
        example_method_comparison(debug=args.debug)

    if args.example in {"degrees", "all"}:
        example_degree_comparison(debug=args.debug)

    if args.example in {"noise", "all"}:
        example_noise_sensitivity(debug=args.debug)
