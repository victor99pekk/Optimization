from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('HW7Q3.csv', header=None)


A = df.iloc[:, :50].to_numpy()
b = df.iloc[:, 50:].to_numpy().flatten()
s = 48
iterations = 10**5
x_optimal = np.zeros(50, dtype=float)
x_optimal[0] = 1
x_optimal[14] = 3

def function(x:np.ndarray) -> np.ndarray:
    return 0.5*(np.linalg.norm(np.dot(A, x) - b)**2)

def gradient(x:np.ndarray) -> np.ndarray:
    return np.dot(A.T, (A @ x) - b)

def projection(x:np.ndarray, s:int) -> np.ndarray:
    arr = x.copy()
    for _ in range(s):
        min_index = np.argmin(arr)
        x[min_index] = 0
        arr[min_index] = np.finfo(np.float64).max
    return x

def gradient_descent(x:np.ndarray, lr:float=1e-4) -> float:
    f_optimal = function(x_optimal)
    ylist = [np.abs(function(x) - f_optimal)]
    for i in range(iterations):
        x = projection(x - lr * gradient(x), s)
        ylist.append(np.abs(function(x) - f_optimal))
        # if i % 100 == 0:
    print(function(x))
    print(x)
    return ylist

ylist = gradient_descent(np.array([np.random.uniform(0,10) for _ in range(50)]))
iterationList = [i for i in range(iterations+1)]
plt.plot(iterationList, ylist)
plt.xlabel('Iteration')
plt.ylabel('|| f(x) - f(x*) ||')
plt.legend()
plt.savefig('q3_plot.png')
plt.show()