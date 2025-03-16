"""
Tests for the tridiagonal matrix solver implementation.

This module contains comprehensive tests for the tridiagonal solver that implements
the Thomas algorithm for efficiently solving Ax = b where A is a tridiagonal matrix.

The tests cover:
1. Correctness across different matrix sizes
2. Numerical stability with ill-conditioned matrices
3. Edge cases (zero/near-zero pivots, etc.)
4. Performance benchmarking against NumPy's general solver
5. Verification with analytically solvable systems
"""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

import pytest
from interpolatepy.tridiagonal_inv import solve_tridiagonal


# Type alias for pytest benchmark fixture
if not hasattr(pytest, "FixtureFunction"):
    setattr(pytest, "FixtureFunction", Any)


class TestTridiagonalSolver:
    """Test suite for the tridiagonal matrix solver."""

    # Test case sizes for parametrization
    SIZES = [10, 100, 1000]

    # Tolerance values for different test types
    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    STABILITY_RTOL = 1e-6
    STABILITY_ATOL = 1e-6

    @staticmethod
    def generate_tridiagonal_system(
        size: int, seed: int = 42, condition: str = "well"
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Generate a tridiagonal system with controlled properties.

        Parameters
        ----------
        size : int
            Size of the system to generate
        seed : int, optional
            Random seed for reproducibility, by default 42
        condition : str, optional
            Condition of the matrix: "well" for well-conditioned,
            "ill" for ill-conditioned, by default "well"

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]
            lower_diag, main_diag, upper_diag, rhs, true_sol, full_matrix
        """
        np.random.seed(seed)

        # Generate diagonals based on desired conditioning
        if condition == "well":
            # Well-conditioned: Diagonally dominant
            main_diag = np.random.uniform(10, 20, size)
            lower_diag = np.random.uniform(1, 5, size)
            upper_diag = np.random.uniform(1, 5, size)
        elif condition == "ill":
            # Ill-conditioned: Close to singular
            main_diag = np.random.uniform(1e-3, 1e-2, size)
            lower_diag = np.random.uniform(1e-3, 1e-2, size)
            upper_diag = np.random.uniform(1e-3, 1e-2, size)
        else:
            raise ValueError(f"Unknown condition type: {condition}")

        # First element of lower diagonal and last element of upper diagonal are not used
        lower_diag[0] = 0.0
        upper_diag[-1] = 0.0

        # Create the full matrix for numpy.linalg.solve
        full_matrix = np.zeros((size, size))
        for i in range(size):
            full_matrix[i, i] = main_diag[i]
            if i > 0:
                full_matrix[i, i - 1] = lower_diag[i]
            if i < size - 1:
                full_matrix[i, i + 1] = upper_diag[i]

        # Generate a random true solution
        true_sol = np.random.uniform(-10, 10, size)

        # Calculate the right-hand side
        rhs = np.dot(full_matrix, true_sol)

        return lower_diag, main_diag, upper_diag, rhs, true_sol, full_matrix

    @staticmethod
    def get_analytical_tridiagonal_system(
        size: int = 4,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Create a tridiagonal system with a known analytical solution.

        This creates a system that has integer solutions for easier verification.

        Parameters
        ----------
        size : int, optional
            Size of the system, by default 4

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray, NDArray]
            lower_diag, main_diag, upper_diag, rhs, expected_solution
        """
        # Create a simple tridiagonal system with constants
        lower_diag = np.ones(size)
        lower_diag[0] = 0  # First element not used

        main_diag = 2 * np.ones(size)

        upper_diag = np.ones(size)
        upper_diag[-1] = 0  # Last element not used

        # Solution is [1, 2, ..., size]
        expected_solution = np.arange(1, size + 1, dtype=float)

        # Calculate right-hand side from the matrix and expected solution
        matrix = np.zeros((size, size))
        for i in range(size):
            matrix[i, i] = main_diag[i]
            if i > 0:
                matrix[i, i - 1] = lower_diag[i]
            if i < size - 1:
                matrix[i, i + 1] = upper_diag[i]

        rhs = np.dot(matrix, expected_solution)

        return lower_diag, main_diag, upper_diag, rhs, expected_solution

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_solver_correctness(self, size: int, seed: int) -> None:
        """
        Test that the tridiagonal solver produces correct results.

        Parameters
        ----------
        size : int
            Size of the system to test
        seed : int
            Random seed for reproducibility
        """
        # Generate well-conditioned test case
        lower_diag, main_diag, upper_diag, rhs, true_sol, matrix = self.generate_tridiagonal_system(
            size, seed=seed
        )

        # Solve using custom solver
        custom_solution = solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

        # Solve using numpy as reference
        numpy_solution = np.linalg.solve(matrix, rhs)

        # Verify against true solution and numpy solution
        assert np.allclose(
            custom_solution, true_sol, rtol=self.REGULAR_RTOL, atol=self.REGULAR_ATOL
        )
        assert np.allclose(
            custom_solution, numpy_solution, rtol=self.REGULAR_RTOL, atol=self.REGULAR_ATOL
        )

    @pytest.mark.parametrize("size", [4, 8, 16])
    def test_analytical_system(self, size: int) -> None:
        """
        Test the solver against a system with known analytical solution.

        Parameters
        ----------
        size : int
            Size of the system to test
        """
        lower_diag, main_diag, upper_diag, rhs, expected = self.get_analytical_tridiagonal_system(
            size
        )

        # Solve with custom solver
        solution = solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

        # Check against expected analytical solution
        assert np.allclose(solution, expected, rtol=self.REGULAR_RTOL, atol=self.REGULAR_ATOL)

    def test_numerical_stability(self) -> None:
        """Test the solver's stability with ill-conditioned matrices."""
        for size in [10, 50]:
            # Generate an ill-conditioned test case
            lower_diag, main_diag, upper_diag, rhs, true_sol, matrix = (
                self.generate_tridiagonal_system(size, condition="ill")
            )

            # Solve using custom solver
            custom_solution = solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

            # Check against true solution with more relaxed tolerances
            assert np.allclose(
                custom_solution, true_sol, rtol=self.STABILITY_RTOL, atol=self.STABILITY_ATOL
            )

    def test_zero_pivot_raises_error(self) -> None:
        """Test that the solver raises a ValueError when encountering a zero pivot."""
        size = 10

        # Generate a system with a zero pivot
        _, main_diag, upper_diag, rhs, _, _ = self.generate_tridiagonal_system(size)
        main_diag[0] = 0.0  # Set the first pivot to zero
        lower_diag = np.zeros(size)

        # Check that ValueError is raised
        with pytest.raises(ValueError, match="Pivot cannot be zero"):
            solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

    def test_near_zero_pivot_stability(self) -> None:
        """Test that the solver remains stable with very small but non-zero pivots."""
        size = 10

        # Generate a system with a near-zero pivot
        _, main_diag, upper_diag, _, _, _ = self.generate_tridiagonal_system(size)
        main_diag[0] = 1e-10  # Set the first pivot to near-zero
        lower_diag = np.zeros(size)

        # Create a known solution
        true_sol = np.ones(size)

        # Create the full matrix
        matrix = np.zeros((size, size))
        for i in range(size):
            matrix[i, i] = main_diag[i]
            if i < size - 1:
                matrix[i, i + 1] = upper_diag[i]

        # Calculate the right-hand side
        rhs = np.dot(matrix, true_sol)

        # Solve with custom solver
        solution = solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

        # Check with relaxed tolerances
        assert np.allclose(solution, true_sol, rtol=1e-5, atol=1e-5)

    def test_input_validation(self) -> None:
        """Test the solver with invalid inputs."""
        # Test with mismatched sizes
        # The current implementation doesn't check for matching sizes explicitly
        # and will raise an IndexError when trying to access elements
        with pytest.raises(IndexError):
            solve_tridiagonal(
                np.array([0, 1, 2]),
                np.array([2, 3, 4, 5]),
                np.array([1, 2, 3, 0]),
                np.array([1, 2, 3, 4]),
            )

    @pytest.mark.parametrize("size", SIZES)
    def test_custom_solver_benchmark(self, size: int, benchmark: pytest.FixtureFunction) -> None:
        """
        Benchmark the custom tridiagonal solver.

        Parameters
        ----------
        size : int
            Size of the system to benchmark
        benchmark : pytest.FixtureFunction
            pytest-benchmark fixture for measuring performance
        """
        # Generate test case
        lower_diag, main_diag, upper_diag, rhs, true_sol, _ = self.generate_tridiagonal_system(size)

        # Benchmark the custom solver
        result = benchmark(solve_tridiagonal, lower_diag, main_diag, upper_diag, rhs)

        # Verify result correctness
        assert np.allclose(result, true_sol, rtol=self.REGULAR_RTOL, atol=self.REGULAR_ATOL)

    @pytest.mark.parametrize("size", SIZES)
    def test_numpy_solver_benchmark(self, size: int, benchmark: pytest.FixtureFunction) -> None:
        """
        Benchmark NumPy's general solver for comparison.

        Parameters
        ----------
        size : int
            Size of the system to benchmark
        benchmark : pytest.FixtureFunction
            pytest-benchmark fixture for measuring performance
        """
        # Generate test case
        _, _, _, rhs, true_sol, matrix = self.generate_tridiagonal_system(size)

        # Benchmark NumPy's solver
        result = benchmark(np.linalg.solve, matrix, rhs)

        # Verify result correctness
        assert np.allclose(result, true_sol, rtol=self.REGULAR_RTOL, atol=self.REGULAR_ATOL)

    def test_performance_scaling(self) -> None:
        """Analyze how solver performance scales with matrix size."""
        sizes = [10, 50, 100, 500, 1000]
        custom_times = []
        numpy_times = []

        for size in sizes:
            # Generate test case
            lower_diag, main_diag, upper_diag, rhs, _, matrix = self.generate_tridiagonal_system(
                size
            )

            # Time the custom solver
            repetitions = max(1, int(10000 / size))  # Adjust repetitions based on size

            # Warm-up
            _ = solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)
            _ = np.linalg.solve(matrix, rhs)

            # Time custom solver
            start_time = time.time()
            for _ in range(repetitions):
                solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)
            end_time = time.time()
            custom_time = (end_time - start_time) / repetitions
            custom_times.append(custom_time)

            # Time NumPy's solver
            start_time = time.time()
            for _ in range(repetitions):
                np.linalg.solve(matrix, rhs)
            end_time = time.time()
            numpy_time = (end_time - start_time) / repetitions
            numpy_times.append(numpy_time)

            # Print comparison
            speedup = numpy_time / custom_time
            print(
                f"Size {size}x{size}: Custom={custom_time:.6f}s, NumPy={numpy_time:.6f}s, Speedup={speedup:.2f}x"
            )

        # Check larger size scaling patterns
        # For a proper O(n) algorithm, we expect roughly linear scaling,
        # but in real-world conditions, various factors like cache effects,
        # memory access patterns, and system load can impact measurements
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = custom_times[i] / custom_times[i - 1]

            # Log the scaling behavior
            print(
                f"Size increase {sizes[i-1]} → {sizes[i]} ({size_ratio:.1f}x): Time increase {time_ratio:.2f}x"
            )

            # For very large matrices, we should see closer to linear scaling
            # Allow more variation for smaller matrices
            if sizes[i] >= 500:
                # For larger matrices, scaling should be closer to linear
                assert time_ratio < size_ratio * 3.5, (
                    f"Performance scaling for large matrices worse than expected: "
                    f"size increased by {size_ratio:.1f}x but time increased by {time_ratio:.2f}x"
                )


if __name__ == "__main__":
    # This allows running the tests standalone with more detailed output
    pytest.main(["-xvs", __file__])
