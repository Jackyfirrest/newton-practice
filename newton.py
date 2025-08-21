import warnings
import numpy as np
from scipy.optimize import approx_fprime, approx_derivative

# Newton's method implementation with derivatives

def finite_difference_first(f, x, eps=1e-5):
    """
    Approximate the first derivative of a function at a point using finite differences.

    Parameters
    ----------
    f : callable
        The function whose derivative is to be computed.
    x : float
        The point at which to evaluate the derivative.
    eps : float, optional
        The step size for the finite difference (default is 1e-5).

    Returns
    -------
    float
        The approximate first derivative of f at x.
    """
    return (f(x + eps) - f(x)) / (eps)


def finite_difference_second(f, x, eps=1e-5):
    """
    Approximate the second derivative of a function at a point using finite differences.

    Parameters
    ----------
    f : callable
        The function whose second derivative is to be computed.
    x : float
        The point at which to evaluate the second derivative.
    eps : float, optional
        The step size for the finite difference (default is 1e-5).

    Returns
    -------
    float
        The approximate second derivative of f at x.
    """
    return (finite_difference_first(f, x + eps) - finite_difference_first(f, x)) / (eps)


def newton_method(f, x0, tol=1e-6, iteration=100):
    """
    Find a root of a function using Newton's method with numerical derivatives.

    Parameters
    ----------
    f : callable
        The function whose root is to be found.
    x0 : float
        Initial guess for the root.
    tol : float, optional
        Tolerance for convergence (default is 1e-6).
    iteration : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    float
        The estimated root of the function.

    Notes
    -----
    This implementation uses finite differences to approximate the first and second derivatives.
    """
    if not callable(f):
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")
    if not isinstance(x0, (int, float)):
        raise TypeError(f"Initial guess x0 must be a number, got {type(x0)}")

    x = x0
    for i in range(iteration):
        # Compute first and second derivatives numerically
        first_derivatives = finite_difference_first(f, x)
        second_derivatives = finite_difference_second(f, x)
        # check whether second derivitives is zero
        assert second_derivatives == 0, "Second derivative is zero."
        # Update estimate using Newton's method formula
        x_t = x - first_derivatives / second_derivatives
        if abs(x_t - x) > 1e6:
            warnings.warn(f"Large step detected")
        # Check for convergence
        if abs(x_t - x) < tol:
            print("stop iteration")
            return x_t
        x = x_t
    return x_t

def multivariate_newton(f, x0, tol=1e-6, max_iter=100):
    """
    Multivariate Newton's method for finding a root or minimum of f.

    Parameters
    ----------
    f : callable
        Function to optimize. Should take a 1D numpy array and return a scalar.
    x0 : array-like
        Initial guess (1D array).
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    x : ndarray
        Estimated optimum.
    """
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = approx_fprime(x, f, epsilon=1e-8)
        hess = approx_derivative(f, x, method='2-point', abs_step=1e-5)
        delta = np.dot(np.linalg.inv(hess), grad)
        x_new = x - delta
        if np.linalg.norm(x_new - x) < tol:
            print("Stop iteration.")
            return x_new
        x = x_new
    return x