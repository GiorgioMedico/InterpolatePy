import time
from typing import Tuple

import numpy as np

import pytest
from interpolatepy.tridiagonal_inv import solve_tridiagonal


def generate_test_case(size: int) -> Tuple:
    """Generate a consistent test case for benchmarking."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random diagonals
    main_diag = np.random.uniform(10, 20, size)
    lower_diag = np.random.uniform(1, 5, size)
    upper_diag = np.random.uniform(1, 5, size)

    # First element of lower diagonal and last element of upper diagonal are not used
    lower_diag[0] = 0.0
    upper_diag[-1] = 0.0

    # Create the full matrix for numpy.linalg.solve
    matrix = np.zeros((size, size))
    for i in range(size):
        matrix[i, i] = main_diag[i]
        if i > 0:
            matrix[i, i - 1] = lower_diag[i]
        if i < size - 1:
            matrix[i, i + 1] = upper_diag[i]

    # Generate a random true solution
    true_sol = np.random.uniform(-10, 10, size)

    # Calculate the right-hand side
    rhs = np.dot(matrix, true_sol)

    return lower_diag, main_diag, upper_diag, rhs, true_sol, matrix


# Define test case parameters
test_sizes = [10, 100, 1000]


@pytest.mark.parametrize("size", test_sizes)
def test_solver_correctness(size):
    """Test that the tridiagonal solver produces correct results."""
    # Generate test case
    lower_diag, main_diag, upper_diag, rhs, true_sol, matrix = generate_test_case(size)

    # Solve using custom solver
    custom_solution = solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

    # Solve using numpy
    numpy_solution = np.linalg.solve(matrix, rhs)

    # Check that the custom solution is close to the true solution
    assert np.allclose(custom_solution, true_sol, rtol=1e-10, atol=1e-10)

    # Check that the custom solution matches numpy's solution
    assert np.allclose(custom_solution, numpy_solution, rtol=1e-10, atol=1e-10)


def test_zero_pivot_raises_error():
    """Test that the solver raises a ValueError when encountering a zero pivot."""
    size = 10
    main_diag = np.random.uniform(10, 20, size)
    main_diag[0] = 0.0  # Set the first pivot to zero
    lower_diag = np.random.uniform(1, 5, size)
    upper_diag = np.random.uniform(1, 5, size)
    rhs = np.random.uniform(-10, 10, size)

    # Check that ValueError is raised
    with pytest.raises(ValueError):
        solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)


# Store performance data between tests
performance_data = {}


@pytest.mark.parametrize("size", test_sizes)
def test_custom_solver_benchmark(size, benchmark):
    """Benchmark the custom tridiagonal solver."""
    # Generate test case
    lower_diag, main_diag, upper_diag, rhs, true_sol, _ = generate_test_case(size)

    # Benchmark the custom solver
    result = benchmark(solve_tridiagonal, lower_diag, main_diag, upper_diag, rhs)

    # Store performance data for comparison
    performance_data[f"custom_{size}"] = benchmark.stats.stats.mean

    # Verify result correctness
    assert np.allclose(result, true_sol, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("size", test_sizes)
def test_numpy_solver_benchmark(size, benchmark):
    """Benchmark NumPy's general solver."""
    # Generate test case
    _, _, _, rhs, true_sol, matrix = generate_test_case(size)

    # Benchmark NumPy's solver
    result = benchmark(np.linalg.solve, matrix, rhs)

    # Store performance data and compare if custom solver data is available
    numpy_time = benchmark.stats.stats.mean
    performance_data[f"numpy_{size}"] = numpy_time

    # If we have data for the custom solver, print a comparison
    if f"custom_{size}" in performance_data:
        custom_time = performance_data[f"custom_{size}"]
        speedup = numpy_time / custom_time
        print(f"\nMatrix size: {size}x{size}")
        print(f"  Custom solver time:  {custom_time:.6f} seconds")
        print(f"  NumPy solver time:   {numpy_time:.6f} seconds")
        print(f"  Speedup factor:      {speedup:.2f}x")

    # Verify result correctness
    assert np.allclose(result, true_sol, rtol=1e-10, atol=1e-10)


def test_known_system():
    """Test the solver with a simple known system."""
    # A simple tridiagonal system with a known solution
    lower_diag = np.array([0, 1, 1, 1])
    main_diag = np.array([2, 2, 2, 2])
    upper_diag = np.array([1, 1, 1, 0])
    rhs = np.array([1, 2, 3, 4])

    # Solve with custom solver
    solution = solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

    # Create the full matrix and solve with numpy for comparison
    matrix = np.zeros((4, 4))
    for i in range(4):
        matrix[i, i] = main_diag[i]
        if i > 0:
            matrix[i, i - 1] = lower_diag[i]
        if i < 3:
            matrix[i, i + 1] = upper_diag[i]

    numpy_solution = np.linalg.solve(matrix, rhs)

    # Check that both solutions match
    assert np.allclose(solution, numpy_solution, rtol=1e-10, atol=1e-10)


# Alternative benchmark approach using Python's built-in timing functions
def test_performance_comparison():
    """Compare performance using Python's built-in timing functions."""
    print("\nPerformance comparison using time.time():")

    for size in test_sizes:
        # Generate test case
        lower_diag, main_diag, upper_diag, rhs, _, matrix = generate_test_case(size)

        # Time the custom solver
        repetitions = max(1, int(10000 / size))  # Adjust repetitions based on size

        start_time = time.time()
        for _ in range(repetitions):
            solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)
        end_time = time.time()
        custom_time = (end_time - start_time) / repetitions

        # Time NumPy's solver
        start_time = time.time()
        for _ in range(repetitions):
            np.linalg.solve(matrix, rhs)
        end_time = time.time()
        numpy_time = (end_time - start_time) / repetitions

        # Print comparison
        speedup = numpy_time / custom_time
        print(
            f"  Size {size}x{size}: Custom={custom_time:.6f}s, NumPy={numpy_time:.6f}s, Speedup={speedup:.2f}x"
        )
