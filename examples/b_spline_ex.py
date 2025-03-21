import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.b_spline import BSpline


def example_bspline() -> BSpline:
    """
    Create an example B-spline curve.

    Returns:
        BSpline: An example B-spline object.
    """
    # Define the degree
    degree = 3

    # Define the control points (2D for this example)
    control_points = np.array([[1, 2], [2, 3], [3, -3], [4, 4], [5, 5], [6, -5], [7, -6]])

    # Create knot vector similar to the example in the document
    knots = np.array([0, 0, 0, 0, 1, 2, 4, 7, 7, 7, 7])

    # Create and return the B-spline
    return BSpline(degree, knots, control_points)


def demonstration() -> None:
    """
    Demonstrate the B-spline implementation with the example from the document.
    """
    # Create the example B-spline
    bspline = example_bspline()

    # Plot the B-spline curve
    bspline.plot_2d(num_points=100)

    # Add title that matches the example
    plt.title("Cubic B-spline and its control polygon")

    # Evaluate the B-spline at the specific value mentioned in the document
    u_value = 1.5
    point = bspline.evaluate(u_value)

    # Calculate and print the basis functions at this point
    span = bspline.find_knot_span(u_value)
    basis_values = bspline.basis_functions(u_value, span)

    print(f"For u = {u_value}, the non-zero basis functions are:")
    for i, value in enumerate(basis_values):
        print(f"B^3_{span - bspline.degree + i} = {value:.4f}")

    # Mark the evaluated point on the plot
    plt.plot(point[0], point[1], "go", markersize=10, label=f"Point at u={u_value}")
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def example_b6() -> None:
    """
    Implements Example B.6 from the document:
    Calculates the basis functions of degree 3 and their derivatives
    at u = 4.5 for knot vector [0, 0, 0, 0, 1, 2, 4, 7, 7, 7, 7]
    """
    # Define the degree
    degree = 3

    # Define the knot vector as specified in Example B.6
    knots = np.array([0, 0, 0, 0, 1, 2, 4, 7, 7, 7, 7])

    # For this example, we need control points, but they don't affect the basis functions
    # So we'll create dummy 2D control points (7 points as required by the knot vector)
    control_points = np.zeros((7, 2))

    # Create the B-spline
    bspline = BSpline(degree, knots, control_points)

    # Evaluate at u = 4.5
    u_value = 4.5

    # Find the knot span index (should be 6 according to the example)
    span = bspline.find_knot_span(u_value)
    print(f"For u = {u_value}, the knot span index is: {span}")

    # Calculate derivatives up to order 3
    derivatives = bspline.basis_function_derivatives(u_value, span, 3)

    # Display the results in the format shown in the example
    print("\nBasis function values and derivatives at u = 4.5:")
    print("-" * 80)

    for k in range(4):  # Derivatives 0 to 3
        line = (
            f"Ders[{k}][0] = {derivatives[k, 0]:.4f}, "
            f"Ders[{k}][1] = {derivatives[k, 1]:.4f}, "
            f"Ders[{k}][2] = {derivatives[k, 2]:.4f}, "
            f"Ders[{k}][3] = {derivatives[k, 3]:.4f},"
        )
        print(line)

    print("\nWhich correspond to:")
    print("-" * 80)

    # First row: B₃³, B₄³, B₅³, B₆³
    print(
        f"B₃³     = {derivatives[0, 0]:.4f}, "
        f"B₄³     = {derivatives[0, 1]:.4f}, "
        f"B₅³     = {derivatives[0, 2]:.4f}, "
        f"B₆³     = {derivatives[0, 3]:.4f},"
    )

    # Second row: B₃³⁽¹⁾, B₄³⁽¹⁾, B₅³⁽¹⁾, B₆³⁽¹⁾
    print(
        f"B₃³⁽¹⁾   = {derivatives[1, 0]:.4f}, "
        f"B₄³⁽¹⁾   = {derivatives[1, 1]:.4f}, "
        f"B₅³⁽¹⁾   = {derivatives[1, 2]:.4f}, "
        f"B₆³⁽¹⁾   = {derivatives[1, 3]:.4f},"
    )

    # Third row: B₃³⁽²⁾, B₄³⁽²⁾, B₅³⁽²⁾, B₆³⁽²⁾
    print(
        f"B₃³⁽²⁾   = {derivatives[2, 0]:.4f}, "
        f"B₄³⁽²⁾   = {derivatives[2, 1]:.4f}, "
        f"B₅³⁽²⁾   = {derivatives[2, 2]:.4f}, "
        f"B₆³⁽²⁾   = {derivatives[2, 3]:.4f},"
    )

    # Fourth row: B₃³⁽³⁾, B₄³⁽³⁾, B₅³⁽³⁾, B₆³⁽³⁾
    print(
        f"B₃³⁽³⁾   = {derivatives[3, 0]:.4f}, "
        f"B₄³⁽³⁾   = {derivatives[3, 1]:.4f}, "
        f"B₅³⁽³⁾   = {derivatives[3, 2]:.4f}, "
        f"B₆³⁽³⁾   = {derivatives[3, 3]:.4f}."
    )

    print("\nAll the other terms B_j^3(k) are null.")

    # Plot the basis functions
    plot_basis_functions(bspline, u_value)


def plot_basis_functions(bspline: BSpline, u_value: float) -> None:
    """
    Plot the basis functions and mark the evaluation point.

    Args:
        bspline: The B-spline object
        u_value: The parameter value to evaluate
    """
    # Create figure
    _fig, ax = plt.subplots(figsize=(10, 6))

    # Generate parameter values within the valid range
    u_range = np.linspace(bspline.u_min, bspline.u_max, 500)

    # Calculate basis functions for each u in the range
    basis_values = []
    for u in u_range:
        span = bspline.find_knot_span(u)
        values = bspline.basis_functions(u, span)
        start_index = span - bspline.degree
        basis_values.append((start_index, values))

    # Plot each basis function separately
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    for i in range(len(bspline.control_points)):
        y_values = np.zeros_like(u_range)
        for j, (start_index, values) in enumerate(basis_values):
            idx = i - start_index
            if 0 <= idx < len(values):
                y_values[j] = values[idx]

        ax.plot(u_range, y_values, color=colors[i % len(colors)], label=f"B_{i}^{bspline.degree}")

    # Find the non-zero basis functions at u_value
    span = bspline.find_knot_span(u_value)
    values = bspline.basis_functions(u_value, span)

    # Mark the evaluation point on each non-zero basis function
    for i in range(bspline.degree + 1):
        idx = span - bspline.degree + i
        ax.plot(u_value, values[i], "ko", markersize=6)
        ax.text(
            u_value,
            values[i] + 0.02,
            f"B_{idx}^{bspline.degree}({u_value:.1f})={values[i]:.4f}",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    # Add vertical line at the evaluation point
    ax.axvline(x=u_value, color="k", linestyle="--", alpha=0.5)

    # Add knot locations as vertical lines
    for knot in np.unique(bspline.knots):
        if bspline.u_min <= knot <= bspline.u_max:
            ax.axvline(x=knot, color="gray", linestyle="-", alpha=0.3)
            ax.text(knot, -0.05, f"{knot}", horizontalalignment="center")

    # Set labels and title
    ax.set_xlabel("Parameter u")
    ax.set_ylabel("Basis function value")
    ax.set_title(f"B-spline basis functions of degree {bspline.degree}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def create_simple_3d_bspline() -> BSpline:
    """
    Create a simple 3D B-spline curve.

    Returns:
        BSpline: A 3D B-spline object.
    """
    # Define the degree
    degree = 3

    # Define simple 3D control points for a curve
    control_points = np.array(
        [
            [0, 0, 0],  # Start point
            [1, 1, 2],
            [2, -1, 1],
            [3, 0, 3],
            [4, 2, 0],
            [5, 0, 1],  # End point
        ]
    )

    # Create a uniform knot vector
    knots = BSpline.create_uniform_knots(degree, len(control_points))

    # Create and return the B-spline
    return BSpline(degree, knots, control_points)


def demonstrate_3d_bspline() -> None:
    """
    Demonstrate a simple 3D B-spline curve.
    """
    # Create the B-spline
    bspline = create_simple_3d_bspline()

    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the B-spline curve with control polygon
    bspline.plot_3d(num_points=100, show_control_polygon=True, ax=ax)

    # Set the title and adjust view
    ax.set_title("Simple 3D B-spline Curve (Degree 3)")
    ax.view_init(elev=30, azim=45)

    # Set equal aspect ratio for better visualization
    ax.set_box_aspect([1, 1, 1])

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Print some basic information about the curve
    print("\nB-spline Information:")
    print(f"- Degree: {bspline.degree}")
    print(f"- Number of control points: {len(bspline.control_points)}")
    print(f"- Parameter range: [{bspline.u_min}, {bspline.u_max}]")

    # Evaluate a point in the middle of the curve
    mid_point = bspline.evaluate((bspline.u_min + bspline.u_max) / 2)
    print(
        f"\nPoint at middle of curve: ({mid_point[0]:.2f}, {mid_point[1]:.2f}, {mid_point[2]:.2f})"
    )


if __name__ == "__main__":
    demonstration()
    example_b6()
    demonstrate_3d_bspline()
