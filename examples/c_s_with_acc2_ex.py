from interpolatepy.c_s_with_acc2 import CubicSplineWithAcceleration2
from interpolatepy.c_s_with_acc2 import SplineParameters


# Example usage
if __name__ == "__main__":
    # Define waypoints
    t_points = [0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0]
    q_points = [3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0]

    # Create parameters object
    params = SplineParameters(
        v0=2.0,  # Initial velocity
        vn=-3.0,  # Final velocity
        a0=0.0,  # Initial acceleration
        an=0.0,  # Final acceleration
        debug=False,
    )

    # Create cubic spline with initial and final accelerations
    spline = CubicSplineWithAcceleration2(t_points, q_points, params)

    # Plot the trajectory
    spline.plot(1000)

    # Verify initial and final conditions
    print("\nVerifying boundary conditions:")

    # Initial velocity check
    initial_vel = spline.evaluate_velocity(t_points[0])
    print(f"Initial velocity: {initial_vel:.6f} (expected: {params.v0:.6f})")

    # Final velocity check
    final_vel = spline.evaluate_velocity(t_points[-1])
    print(f"Final velocity: {final_vel:.6f} (expected: {params.vn:.6f})")

    # Initial acceleration check
    initial_acc = spline.evaluate_acceleration(t_points[0])
    expected_a0 = params.a0
    print(f"Initial acceleration: {initial_acc:.6f} (expected: {expected_a0:.6f})")

    # Final acceleration check
    final_acc = spline.evaluate_acceleration(t_points[-1])
    expected_an = params.an
    print(f"Final acceleration: {final_acc:.6f} (expected: {expected_an:.6f})")
