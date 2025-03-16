import numpy as np


def solve_tridiagonal(
    lower_diagonal: np.ndarray,
    main_diagonal: np.ndarray,
    upper_diagonal: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """
    Solve a tridiagonal system using the Thomas algorithm.

    This function solves the equation Ax = b where A is a tridiagonal matrix.
    The system is solved efficiently using the Thomas algorithm (also known as
    the tridiagonal matrix algorithm).

    Parameters
    ----------
    lower_diagonal : np.ndarray
        Lower diagonal elements (first element is not used).
        Must have the same length as main_diagonal.
    main_diagonal : np.ndarray
        Main diagonal elements.
    upper_diagonal : np.ndarray
        Upper diagonal elements (last element is not used).
        Must have the same length as main_diagonal.
    right_hand_side : np.ndarray
        Right-hand side vector of the equation.

    Returns
    -------
    np.ndarray
        Solution vector x.

    Raises
    ------
    ValueError
        If a pivot is zero during forward elimination.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0, 1, 2, 3])  # Lower diagonal (a[0] is not used)
    >>> b = np.array([2, 3, 4, 5])  # Main diagonal
    >>> c = np.array([1, 2, 3, 0])  # Upper diagonal (c[-1] is not used)
    >>> d = np.array([1, 2, 3, 4])  # Right hand side
    >>> x = solve_tridiagonal(a, b, c, d)
    >>> print(x)

    Notes
    -----
    The Thomas algorithm is a specialized form of Gaussian elimination for
    tridiagonal systems. It is much more efficient than general Gaussian
    elimination, with a time complexity of O(n) instead of O(n³).

    The algorithm consists of two phases:
    1. Forward elimination to transform the matrix into an upper triangular form
    2. Back substitution to find the solution

    For a system where the matrix A is:
        [b₀ c₀  0  0  0]
        [a₁ b₁ c₁  0  0]
        [0  a₂ b₂ c₂  0]
        [0   0 a₃ b₃ c₃]
        [0   0  0 a₄ b₄]

    References
    ----------
    .. [1] Thomas, L.H. (1949). "Elliptic Problems in Linear Differential
           Equations over a Network". Watson Sci. Comput. Lab Report.
    """
    n = len(right_hand_side)

    # Create copies of the input arrays to avoid modifying them
    a_copy = np.array(lower_diagonal, dtype=float)
    b_copy = np.array(main_diagonal, dtype=float)
    c_copy = np.array(upper_diagonal, dtype=float)
    d_copy = np.array(right_hand_side, dtype=float)

    # Check for zero pivot
    if b_copy[0] == 0:
        raise ValueError("Pivot cannot be zero. The system cannot be solved with this method.")

    # Forward elimination
    for k in range(1, n):
        m = a_copy[k] / b_copy[k - 1]
        b_copy[k] -= m * c_copy[k - 1]
        d_copy[k] -= m * d_copy[k - 1]

    # Back substitution
    x = np.zeros(n)
    x[n - 1] = d_copy[n - 1] / b_copy[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = (d_copy[k] - c_copy[k] * x[k + 1]) / b_copy[k]

    return x
