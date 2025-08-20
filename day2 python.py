from newton import newton_method

def f(x):
    return x**2 - 2

root = newton_method(f, x0=1.0)
print("root:", root)