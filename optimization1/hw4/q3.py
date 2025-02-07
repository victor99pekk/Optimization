import numpy as np
import matplotlib.pyplot as plt

def function(x:tuple[float, float]) -> float:
    x1,x2 = x
    return (x1-2)**2 + (x1-2)**2 - x1 * x2
def gradient(x:tuple[float, float]) -> tuple[float, float]:
    x1,x2 = x
    return (2*(x1-2)-x2), (2*(x2-2)-x1)
def projection(x: tuple[float, float]) -> tuple[float, float]:
    return tuple(max(0, min(1, xi)) for xi in x)
def gradient_descent(x:tuple[float, float], iterations:int, step:float) -> float:
    y_values = []
    x_values = []
    for i in range(iterations):
        x_values.append(i+1)
        x1 , x2= gradient(x)
        a,b = x
        x = projection((a-step*x1 ,b-step * x2))
        y_values.append(function(x))
    print(x, function(x))
    return x_values, y_values


x_values, y_values = gradient_descent(x=(0.5, 0.5), iterations=175, step=0.001)
plt.plot(x_values, y_values)
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Gradient Descent Progress')
plt.show()
