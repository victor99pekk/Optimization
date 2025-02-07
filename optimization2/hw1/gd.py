import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

start1 = np.array([0.1, 0.1, 0.1, 0.1], dtype=float)
start2 = np.array([15, -20, 10, 10], dtype=float)
start_points = [start1, start2]

def cvx(x: np.ndarray) -> float:
    return x[0]**2 + x[1]**2 + x[2]**2

def strongly_cvx(x: np.ndarray) -> float:
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2

def gradient_cvx(x: np.ndarray) -> np.ndarray:
    return np.array([2*x[0], 2*x[1], 2*x[2], 0], dtype=float)

def gradient_strongly_cvx(x: np.ndarray) -> np.ndarray:
    return np.array([2*x[0], 2*x[1], 2*x[2], 2*x[3]], dtype=float)

def gradient_descent(function: Callable[[np.ndarray], float], gradient: Callable[[np.ndarray], np.ndarray]) -> None:
    lr = 3e-1
    iteration_nbr = 5
    iterationList = []
    yList = []
    for start_point in start_points:
        x = start_point.astype(float)
        yList.append(function(x))
        iterationList.append(1)
        for i in range(2, iteration_nbr):
            iterationList.append(i)
            x -= lr * gradient(x)
            yList.append(function(x))
    length = len(yList) // 2
    iterationList = iterationList[:length]
    list1 = yList[:length]
    list2 = yList[length:]

    # Plotting list1 and list2
    plt.figure(figsize=(10, 6))
    plt.plot(iterationList, list1, label='Start Point = [0.1, 0.1, 0.1, 0.1]', color='blue')
    plt.plot(iterationList, list2, label='Start Point = [15, -20, 10, 10]', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('F(x)')
    plt.title('Gradient Descent for Strongly convex function (a)')
    plt.legend()
    plt.grid(True)
    plt.savefig('strong_cvx2.png')
    plt.show()

# Example usage
#gradient_descent(cvx, gradient_cvx)
gradient_descent(strongly_cvx, gradient_strongly_cvx)


# For cvx function, you need to use 4-dimensional start points
# start1 = np.array([0.1, 0.1, 0.1, 0.1], dtype=float)
# start2 = np.array([15, -20, 10, 10], dtype=float)
# start_points = [start1, start2]
# gradient_descent(cvx, gradient_cvx)