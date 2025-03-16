# import matplotlib.pyplot as plt
# import numpy as np

# from interpolatepy.polynomials import ORDER_3
# from interpolatepy.polynomials import ORDER_5
# from interpolatepy.polynomials import ORDER_7
# from interpolatepy.polynomials import BoundaryCondition
# from interpolatepy.polynomials import PolynomialTrajectory
# from interpolatepy.polynomials import TimeInterval
# from interpolatepy.polynomials import TrajectoryParams


# def plot_trajectory(trajectory, times, title, points=None, velocities=None):
#     """
#     Plot the position, velocity, acceleration, and jerk profiles of a trajectory.

#     Parameters
#     ----------
#     trajectory : Callable
#         The trajectory function returned by PolynomialTrajectory methods
#     times : list or array
#         Time points for evaluation
#     title : str
#         Title for the plot
#     points : list, optional
#         Position points to mark on the plot
#     velocities : list, optional
#         Velocity points to mark on the plot
#     """
#     # Generate trajectory points
#     t_eval = np.linspace(times[0], times[-1], 500)
#     positions = []
#     velocities_plot = []
#     accelerations = []
#     jerks = []

#     for t in t_eval:
#         p, v, a, j = trajectory(t)
#         positions.append(p)
#         velocities_plot.append(v)
#         accelerations.append(a)
#         jerks.append(j)

#     # Create plot
#     fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
#     fig.suptitle(title, fontsize=16)

#     # Position
#     axs[0].plot(t_eval, positions, "b-", linewidth=2)
#     axs[0].set_ylabel("Position")
#     if points is not None:
#         axs[0].plot(times, points, "ro", markersize=8)

#     # Velocity
#     axs[1].plot(t_eval, velocities_plot, "g-", linewidth=2)
#     axs[1].set_ylabel("Velocity")
#     if velocities is not None:
#         axs[1].plot(times, velocities, "ro", markersize=8)

#     # Acceleration
#     axs[2].plot(t_eval, accelerations, "r-", linewidth=2)
#     axs[2].set_ylabel("Acceleration")

#     # Jerk
#     axs[3].plot(t_eval, jerks, "m-", linewidth=2)
#     axs[3].set_ylabel("Jerk")
#     axs[3].set_xlabel("Time (s)")

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.92)
#     plt.show()


# def example_order3_multipoint():
#     """Example of a 3rd order polynomial trajectory through multiple points."""
#     print("\n=== 3rd Order Multipoint Trajectory Example ===")

#     # Define points and times
#     points = [0.0, 2.5, 1.0, 4.0, 3.0]
#     times = [0.0, 2.0, 4.0, 7.0, 10.0]

#     # Create trajectory parameters
#     params = TrajectoryParams(points=points, times=times, order=ORDER_3)

#     # Generate trajectory
#     trajectory = PolynomialTrajectory.multipoint_trajectory(params)

#     # Calculate velocities for plotting
#     velocities = PolynomialTrajectory.heuristic_velocities(points, times)
#     print(f"Heuristic velocities: {velocities}")

#     # Plot trajectory
#     plot_trajectory(
#         trajectory, times, "3rd Order Multipoint Trajectory", points=points, velocities=velocities
#     )

#     # Demonstrate trajectory evaluation at specific times
#     print("\nTrajectory values at specific times:")
#     for t in [1.0, 3.0, 5.0, 9.0]:
#         pos, vel, acc, jerk = trajectory(t)
#         print(f"t={t:.1f}: pos={pos:.2f}, vel={vel:.2f}, acc={acc:.2f}, jerk={jerk:.2f}")


# def example_order5_multipoint():
#     """Example of a 5th order polynomial trajectory """
#     print("\n=== 5th Order Multipoint Trajectory Example ===")

#     # Define points, times, and additional constraints
#     points = [0.0, 5.0, 3.0, 8.0]
#     times = [0.0, 3.0, 6.0, 10.0]
#     velocities = [0.0, 2.0, -1.0, 0.0]
#     accelerations = [0.0, 0.0, 0.0, 0.0]  # Zero acceleration at each point

#     # Create trajectory parameters
#     params = TrajectoryParams(
#         points=points,
#         times=times,
#         velocities=velocities,
#         accelerations=accelerations,
#         order=ORDER_5,
#     )

#     # Generate trajectory
#     trajectory = PolynomialTrajectory.multipoint_trajectory(params)

#     # Plot trajectory
#     plot_trajectory(
#         trajectory,
#         times,
#         "5th Order Multipoint Trajectory with Specified Velocities",
#         points=points,
#         velocities=velocities,
#     )


# def example_order7_multipoint():
#     """Example of a 7th order polynomial trajectory with full constraints."""
#     print("\n=== 7th Order Multipoint Trajectory Example ===")

#     # Define points, times, and additional constraints
#     points = [0.0, 4.0, 6.0, 2.0]
#     times = [0.0, 2.5, 5.0, 8.0]
#     velocities = [0.0, 1.5, 0.0, -1.0]
#     accelerations = [0.0, 0.0, -1.0, 0.0]
#     jerks = [0.0, 0.5, 0.0, 0.0]

#     # Create trajectory parameters
#     params = TrajectoryParams(
#         points=points,
#         times=times,
#         velocities=velocities,
#         accelerations=accelerations,
#         jerks=jerks,
#         order=ORDER_7,
#     )

#     # Generate trajectory
#     trajectory = PolynomialTrajectory.multipoint_trajectory(params)

#     # Plot trajectory
#     plot_trajectory(
#         trajectory,
#         times,
#         "7th Order Multipoint Trajectory with Full Constraints",
#         points=points,
#         velocities=velocities,
#     )


# def example_robot_joint_trajectory():
#     """Example simulating a robot joint moving through multiple points."""
#     print("\n=== Robot Joint Trajectory Example ===")

#     # Joint angle waypoints (in radians)
#     angles = [0.0, 0.5, 1.2, 0.8, 0.3]
#     times = [0.0, 2.0, 4.0, 6.0, 8.0]

#     # Define velocity constraints (start and end with zero velocity)
#     velocities = PolynomialTrajectory.heuristic_velocities(angles, times)
#     velocities[0] = 0.0  # Zero initial velocity
#     velocities[-1] = 0.0  # Zero final velocity

#     # Define acceleration constraints (all zero for smooth motion)
#     accelerations = [0.0] * len(angles)

#     # Create trajectory parameters for 5th order polynomial
#     params = TrajectoryParams(
#         points=angles,
#         times=times,
#         velocities=velocities,
#         accelerations=accelerations,
#         order=ORDER_5,
#     )

#     # Generate trajectory
#     trajectory = PolynomialTrajectory.multipoint_trajectory(params)

#     # Plot trajectory
#     plot_trajectory(
#         trajectory, times, "Robot Joint Trajectory", points=angles, velocities=velocities
#     )

#     print("\nJoint trajectory values at specific times:")
#     for t in [1.0, 3.0, 5.0, 7.0]:
#         pos, vel, acc, jerk = trajectory(t)
#         print(
#             f"t={t:.1f}: angle={pos:.2f} rad, vel={vel:.2f} rad/s,
#               acc={acc:.2f} rad/s², jerk={jerk:.2f} rad/s³"
#         )


# def example_comparing_polynomial_orders():
#     """Example comparing different polynomial orders for the same waypoints."""
#     print("\n=== Comparing Different Polynomial Orders ===")

#     # Define common points and times
#     points = [0.0, 3.0, 1.0, 5.0]
#     times = [0.0, 2.0, 4.0, 6.0]

#     # Generate heuristic velocities for 3rd order
#     velocities = PolynomialTrajectory.heuristic_velocities(points, times)

#     # Create zero accelerations and jerks for higher orders
#     accelerations = [0.0] * len(points)
#     jerks = [0.0] * len(points)

#     # Generate trajectories of different orders
#     trajectories = {}

#     # 3rd order
#     params3 = TrajectoryParams(points=points, times=times, velocities=velocities, order=ORDER_3)
#     trajectories[ORDER_3] = PolynomialTrajectory.multipoint_trajectory(params3)

#     # 5th order
#     params5 = TrajectoryParams(
#         points=points,
#         times=times,
#         velocities=velocities,
#         accelerations=accelerations,
#         order=ORDER_5,
#     )
#     trajectories[ORDER_5] = PolynomialTrajectory.multipoint_trajectory(params5)

#     # 7th order
#     params7 = TrajectoryParams(
#         points=points,
#         times=times,
#         velocities=velocities,
#         accelerations=accelerations,
#         jerks=jerks,
#         order=ORDER_7,
#     )
#     trajectories[ORDER_7] = PolynomialTrajectory.multipoint_trajectory(params7)

#     # Generate trajectory points for comparison
#     t_eval = np.linspace(times[0], times[-1], 500)

#     # Plot comparison
#     fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
#     fig.suptitle("Comparison of Different Polynomial Orders", fontsize=16)

#     labels = {ORDER_3: "3rd Order", ORDER_5: "5th Order", ORDER_7: "7th Order"}

#     colors = {ORDER_3: "b-", ORDER_5: "g-", ORDER_7: "r-"}

#     # Plot each trajectory
#     for order, traj in trajectories.items():
#         positions = []
#         velocities_plot = []
#         accelerations = []
#         jerks = []

#         for t in t_eval:
#             p, v, a, j = traj(t)
#             positions.append(p)
#             velocities_plot.append(v)
#             accelerations.append(a)
#             jerks.append(j)

#         axs[0].plot(t_eval, positions, colors[order], linewidth=2, label=labels[order])
#         axs[1].plot(t_eval, velocities_plot, colors[order], linewidth=2, label=labels[order])
#         axs[2].plot(t_eval, accelerations, colors[order], linewidth=2, label=labels[order])
#         axs[3].plot(t_eval, jerks, colors[order], linewidth=2, label=labels[order])

#     # Add waypoints
#     axs[0].plot(times, points, "ko", markersize=8, label="Waypoints")

#     # Labels and legend
#     axs[0].set_ylabel("Position")
#     axs[1].set_ylabel("Velocity")
#     axs[2].set_ylabel("Acceleration")
#     axs[3].set_ylabel("Jerk")
#     axs[3].set_xlabel("Time (s)")

#     axs[0].legend()

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.92)
#     plt.show()


# if __name__ == "__main__":
#     # Run all examples
#     example_order3_multipoint()
#     example_order5_multipoint()
#     example_order7_multipoint()
#     example_robot_joint_trajectory()
#     example_comparing_polynomial_orders()
