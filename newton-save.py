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
    x = x0
    for i in range(iteration):
        # Compute first and second derivatives numerically
        first_derivatives = finite_difference_first(f, x)
        second_derivatives = finite_difference_second(f, x)
        # Update estimate using Newton's method formula
        x_t = x - first_derivatives / second_derivatives
        # Check for convergence
        if abs(x_t - x) < tol:
            print("stop iteration")
            return x_t
        x = x_t
    return x_t
            