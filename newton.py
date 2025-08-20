def finite_difference_first(f, x, eps=1e-5):
    return (f(x + eps) - f(x)) / (eps)

def finite_difference_second(f, x, eps=1e-5):
    return (finite_difference_first(f, x + eps) - finite_difference_first(f, x)) / (eps)

def newton_method(f, x0, tol=1e-6, iteration=100):
    x = x0
    for i in range(iteration):
        first_derivatives = finite_difference_first(f, x)
        second_derivatives = finite_difference_second(f, x)
        x_t = x - first_derivatives / second_derivatives
        if abs(x_t - x) < tol:
            print("stop iteration")
            return x_t
        x = x_t
    return x_t
            