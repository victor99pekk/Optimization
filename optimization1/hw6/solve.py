import numpy as np
from scipy.sparse.linalg import cg

def create_system(n):
    A = np.diag(np.arange(1, n + 1))
    b = np.ones(n)                  
    return A, b

def solve_direct(A, b):
    return np.linalg.solve(A, b)

for n in [20, 100]:
    print(f"\nRunning for n = {n}")
    A, b = create_system(n)
    x0 = np.zeros(n)

    x_star = solve_direct(A, b)
    print("Optimal x*:", x_star)
