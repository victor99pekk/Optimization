import math
from typing import Callable, List, Tuple
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


dim = 20
iterations = 15000


twenty =  np.array([1.0, 0.5, 0.33333333, 0.25, 0.2, 0.16666667,
                   0.14285714, 0.125, 0.11111111, 0.1, 0.09090909, 0.08333333,
                   0.07692308, 0.07142857, 0.06666667, 0.0625, 0.05882353, 0.05555556,
                   0.05263158, 0.05])
hundred = np.array([
    1.,         0.5,        0.33333333, 0.25,       0.2,        0.16666667,
    0.14285714, 0.125,      0.11111111, 0.1 ,       0.09090909, 0.08333333,
    0.07692308, 0.07142857, 0.06666667, 0.0625 ,    0.05882353 ,0.05555556,
    0.05263158, 0.05 ,      0.04761905, 0.04545455, 0.04347826, 0.04166667,
    0.04,       0.03846154, 0.03703704, 0.03571429, 0.03448276, 0.03333333,
    0.03225806, 0.03125 ,   0.03030303, 0.02941176, 0.02857143, 0.02777778,
    0.02702703, 0.02631579, 0.02564103, 0.025     , 0.02439024, 0.02380952,
    0.02325581, 0.02272727, 0.02222222, 0.02173913, 0.0212766 , 0.02083333,
    0.02040816, 0.02      , 0.01960784, 0.01923077, 0.01886792, 0.01851852,
    0.01818182, 0.01785714, 0.01754386, 0.01724138, 0.01694915, 0.01666667,
    0.01639344, 0.01612903, 0.01587302, 0.015625  , 0.01538462, 0.01515152,
    0.01492537, 0.01470588, 0.01449275, 0.01428571, 0.01408451, 0.01388889,
    0.01369863, 0.01351351, 0.01333333, 0.01315789, 0.01298701, 0.01282051,
    0.01265823, 0.0125    , 0.01234568, 0.01219512, 0.01204819, 0.01190476,
    0.01176471, 0.01162791, 0.01149425, 0.01136364, 0.01123596, 0.01111111,
    0.01098901, 0.01086957, 0.01075269, 0.0106383 , 0.01052632, 0.01041667,
    0.01030928, 0.01020408, 0.01010101, 0.01])
optimal_solutions = [twenty, hundred]

def function(x:np.ndarray) -> np.ndarray:
    n = x.shape[0]
    diagonal_elements = np.arange(1, n + 1)
    matrix = np.diag(diagonal_elements)
    return 0.5 * np.dot(x.T, np.dot(matrix, x)) - np.dot(np.array([1 for _ in range(n)]).T, x)

def function_of_a(x: np.ndarray, conjugate_vector:np.ndarray):
    def inner_function(a: float) -> float:
        n = x.shape[0]
        diagonal_elements = np.arange(1, n + 1)
        matrix = np.diag(diagonal_elements)
        return 0.5 * np.dot((x + a * conjugate_vector).T, np.dot(matrix, (x + a * conjugate_vector))) - np.dot(np.ones(n), (x + a * conjugate_vector))
    return inner_function

def gradient(x:np.ndarray) -> np.ndarray:
    n = x.shape[0]
    diagonal_elements = np.arange(1, n + 1)
    matrix = np.diag(diagonal_elements)
    return np.dot(matrix, x) - np.ones(n).T


def gradient_descent(optimal_solution,step_size:float=1e-3) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    x = np.array([0 for _ in range(dim)], dtype=float)
    xlist = []
    ylist = []
    for _ in range(iterations):
        x -= step_size * gradient(x)
        xlist.append(np.linalg.norm(x - optimal_solution))
        ylist.append(np.abs(function(x)-function(optimal_solution)))
    print("\nGradient descent: \n\nx: ",x, "\n\nf(x): ", function(x), "\n")
    return xlist, ylist

def conjugate_gradient(optimal_solution) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    x = np.array([0 for _ in range(dim)], dtype=float)
    n = x.shape[0]
    xlist = []
    ylist = []

    diagonal_elements = np.arange(1, n + 1)
    matrix = np.diag(diagonal_elements)
    
    r = -gradient(x)
    p = r.copy()
    rsold = np.dot(r.T, r)
    
    for i in range(iterations):
        Ap = np.dot(matrix, p)
        alfa = rsold / np.dot(p.T, Ap)
        x += alfa * p
        r = r - alfa * Ap
        rsnew = np.dot(r.T, r)
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
        
        xlist.append(np.linalg.norm(x - optimal_solution))
        ylist.append(np.abs(function(x) - function(optimal_solution)))
    
    print("\nConjugate gradient: \n\nx: ", x, "\n\nf(x): ", function(x), "\n")
    return xlist, ylist

        

# def conjugate_direction(optimal_solution) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#     x = np.array([0 for _ in range(dim)], dtype=float)
#     n = x.shape[0]
#     vectors = []
#     xlist = []
#     ylist = []
#     for i in range(n):
#         array = np.zeros(n)
#         array[i] = 1
#         vectors.append(array)

#     for vector in vectors:
#         f = function_of_a(x, vector)
#         optimal_a = minimize_scalar(f).x
#         x += optimal_a * vector
#         xlist.append(max(np.linalg.norm(x - optimal_solution), 1e-16))
#         ylist.append(max(function(x)-function(optimal_solution), 1e-16))
#     print("\nConjugate descent: \n\nx: ",x, "\n\nf(x): ", function(x), "\n")
#     return xlist,ylist

gd_iter = [i for i in range(iterations)]

for optimal_solution in optimal_solutions:
    cg_iter = [i for i in range(len(optimal_solution))]
    x_c , y_c = conjugate_gradient(optimal_solution)
    x_gd , y_gd = gradient_descent(optimal_solution)

    plt.figure()
    plt.plot(gd_iter, y_c, label='Conjugate Direction-method')
    plt.plot(gd_iter, x_gd, label='Gradient Descent')
    plt.xlabel('Iterations')
    plt.ylabel('|| x_i - x_* ||')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'convergence_comparison_{optimal_solution[0]}.png')
    plt.show()
    

    plt.figure()
    plt.plot(gd_iter, y_c, label= 'Conjugate Gradient-method')
    plt.plot(gd_iter, y_gd, label='Gradient Descent')
    plt.xlabel('Iterations')
    plt.ylabel('f(x) - f(x_*)')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'function_value_comparison_{optimal_solution[0]}.png')
    plt.show()
