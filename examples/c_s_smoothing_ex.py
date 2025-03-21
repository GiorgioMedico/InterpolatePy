import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.c_s_smoothing import CubicSmoothingSpline


# Example usage 1: Textbook example
def textbook_example() -> list[CubicSmoothingSpline]:
    """Recreate the example from the textbook (Figure 4.10)."""
    print("Textbook Example: Smoothing splines with different μ values")

    # Define points from the textbook
    t_points = [0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0]
    q_points = [3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0]

    # Create weights matching the textbook (W^(-1) = diag[0, 1, 1, 1, 1, 1, 0])
    weights = np.ones(len(t_points))
    weights[0] = weights[-1] = np.inf  # Fixed endpoints (infinite weight)

    # Create splines with different μ values
    mu_values = [0.3, 0.6, 1.0]
    splines = []

    # Plotting
    _fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Generate evaluation points
    t_eval = np.linspace(t_points[0], t_points[-1], 1000)

    line_styles = ["--", "-.", "-"]
    colors = ["g", "r", "b"]

    for i, mu in enumerate(mu_values):
        spline = CubicSmoothingSpline(t_points, q_points, mu=mu, weights=weights, debug=True)
        splines.append(spline)

        # Position
        ax1.plot(
            t_eval,
            spline.evaluate(t_eval),
            line_styles[i],
            color=colors[i],
            linewidth=2,
            label=f"μ={mu}",
        )

        # Velocity
        ax2.plot(
            t_eval, spline.evaluate_velocity(t_eval), line_styles[i], color=colors[i], linewidth=2
        )

        # Acceleration
        ax3.plot(
            t_eval,
            spline.evaluate_acceleration(t_eval),
            line_styles[i],
            color=colors[i],
            linewidth=2,
        )

    # Add waypoints to the position plot
    ax1.plot(t_points, q_points, "ko", markersize=8, label="Waypoints")

    # Set labels and titles
    ax1.set_ylabel("Position")
    ax1.grid(True)
    ax1.legend()
    ax1.set_title("Smoothing Splines with Different μ Values")

    ax2.set_ylabel("Velocity")
    ax2.grid(True)

    ax3.set_ylabel("Acceleration")
    ax3.set_xlabel("Time")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    return splines


if __name__ == "__main__":
    # Run the examples
    textbook_example()
