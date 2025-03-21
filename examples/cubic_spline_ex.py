from interpolatepy.cubic_spline import CubicSpline


# Example usage
if __name__ == "__main__":
    # Define waypoints
    t_points = [0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0]
    q_points = [3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0]

    # Create cubic spline with zero initial and final velocities
    spline = CubicSpline(t_points, q_points, v0=2.0, vn=-3.0, debug=False)

    # Plot the trajectory
    spline.plot(1000)
