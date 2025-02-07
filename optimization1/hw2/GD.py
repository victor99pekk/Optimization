import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def gradient(A:np.array, b:np.array, x:np.array):
    return np.dot(2 * np.transpose(A), (np.dot(A, x) - b))
    sum1 = 0
    sum2 = 0
    for i in range(999):
        inner_term = (A[i][0] * x[0] + A[i][1] * x[1] - 1)
        sum1 += 2 * inner_term * A[i][0]
        sum2 += 2 * inner_term * A[i][1]
    return np.array([sum1 / 999, sum2 / 999])
    # return 2 * np.dot((np.dot(A, x) - b).T, A)

def learning_rate(A:np.array):
    # 0.012774369968774082
    return abs(0.5 / np.linalg.norm(A))

def get_A(file='HW2_ellipse.csv'):
    df = pd.read_csv(file)
    df.columns = ['x', 'y']
    x = df['x']
    y = df['y']
    x_squared = x ** 2
    y_squared = y ** 2
    # Create a matrix with x_squared as column one and y_squared as column two
    A = np.column_stack((x_squared, y_squared))
    return A

def print_result(A:np.array, b:np.array, x:np.array, dim:int=999):
    print("\nX värde : ", x, "\n")
    print("Function-value : ", function(A, b, x), "\n")
    
def function(A:np.array, b:np.array, x:np.array, dim:int=999):
    sum = 0
    for i in range(dim):
        sum += (A[i][0] * x[0] + A[i][1] * x[1] - 1) ** 2
    return sum

def plot_ellipse(x: np.array, file:str='HW2_ellipse.csv'):
    # Ensure a has two elements
    # Scatter plot from the file
    df = pd.read_csv(file)
    df.columns = ['x', 'y']
    scatter_x = df['x']
    scatter_y = df['y']
    plt.scatter(scatter_x, scatter_y, label='Data Points')

    assert len(x) == 2, "Parameter array must have exactly two elements."

    a1, a2 = x

    # Generate points for the ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = np.cos(theta) / np.sqrt(a1)
    ellipse_y = np.sin(theta) / np.sqrt(a2)

    # Plot the ellipse
    # plt.figure()
    plt.plot(ellipse_x, ellipse_y, color='red', label=f'Ellipse: {a1:.2f}x^2 + {a2:.2f}y^2 = 1')


    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ellipse and Data Points')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def main():
    dim = 999
    x = np.array([random.uniform(0, 1) for _ in range(2)])
    x = np.array([0,0])
    A = get_A()
    b = np.array([1 for _ in range(dim)])
    stopping_criteria = 1e-5

    while np.linalg.norm(gradient(A, b, x)) > stopping_criteria:
        x -= learning_rate(A) * gradient(A, b, x)
    print_result(A, b, x)
    plot_ellipse(x)



if __name__ == "__main__":
    main()