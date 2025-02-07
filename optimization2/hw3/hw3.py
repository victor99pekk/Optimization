import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file = "mnist_train.csv" 
test_csv = "mnist_test.csv" 
data = pd.read_csv(csv_file)

labels = data.iloc[:, 0].to_numpy()
pixels = data.iloc[:, 1:].to_numpy()
ones_column = np.ones((pixels.shape[0], 1))
pixels = np.hstack((pixels, ones_column))
normalize = False

mask = (labels == 1) | (labels == 2)
X = pixels[mask].astype(float)
y = labels[mask]

y = np.where(y == 1, 1, -1)

if normalize:
    X /= 255.0

X_1 = X[y == 1][:2000]
X_2 = X[y == -1][:2000]
y_1 = y[y == 1][:2000]
y_2 = y[y == -1][:2000]

X = np.concatenate((X_1, X_2), axis=0)
y = np.concatenate((y_1, y_2), axis=0)


lr = 1e-5

N = 4000


def gradient(weights:np.ndarray[float], batch_size:int=1) -> np.ndarray[float]:
    indices = np.random.choice(np.arange(0, 4000), size=batch_size, replace=True)
    grad = np.zeros_like(weights)
    for index in indices:
        x_multiplied_y = (-y[index]) * X[index]
        denominator = 1 + np.exp(-np.dot(weights, X[index]) * y[index])
        grad += (1 / batch_size) * x_multiplied_y * (1 - (1 / denominator))
    return grad   

def classify(weights:np.ndarray[float]) -> float:
    test_data = pd.read_csv(test_csv)
    test_labels = test_data.iloc[:, 0].to_numpy()
    test_pixels = test_data.iloc[:, 1:].to_numpy()

    ones_column = np.ones((test_pixels.shape[0], 1))
    test_pixels = np.hstack((test_pixels, ones_column))

    test_mask = (test_labels == 1) | (test_labels == 2)
    test_X = test_pixels[test_mask].astype(float)[:500]
    
    test_y = test_labels[test_mask][:500]

    test_y = np.where(test_y == 1, 1, -1)

    if normalize:
        test_X /= 255.0
    predictions = np.dot(test_X, weights)    
    predicted_labels = np.sign(predictions)

    summation = 0
    for index in range(500):
        if predicted_labels[index] == test_y[index]:
            summation += 1
    return summation / 500

def log_function(weights:np.ndarray[float]) -> float:
    return np.log(function(weights))  

def function(weights:np.ndarray[float]) -> float:
    #return (1 / N) * np.sum(np.log(1 + np.exp(-y[:N] * np.dot(X[:N], weights))))
    return (1 / N) * np.sum(np.log(1 + np.exp(-y[i] * np.dot(weights, X[i]))) for i in range(N))

def learning_rate(kind, t) -> float:
    if kind == 'a':
        return 1e-5
    else:
        return 1e-4 * np.sqrt(1/(1+t))

def sgd(lr:str, save_fig:bool=False, nbr_iterations:int = 2000, batch_size=1) -> None:
    weights = np.zeros(X.shape[1])
    function_list = []
    iteration_list = []
    weigh_list = []
    for i in range(nbr_iterations):
        iteration_list.append(i)
        function_list.append(log_function(weights))
        weights -= learning_rate(kind=lr, t=i) * gradient(weights, batch_size=batch_size)
    
    if save_fig:
        plt.plot(iteration_list, function_list, label='Log-function value')
        plt.xlabel('#iterations')
        plt.ylabel('Log (function(w))')
        plt.title('SGD Optimization')
        plt.legend()
        plt.savefig('sgd_optimization_plot2.png')
        plt.show()
    print(function_list[-1])
    return weights

# 1a
w = sgd(lr='a', save_fig=False, nbr_iterations=2000) 

# 2a
# w = sgd(lr='b', save_fig=False, nbr_iterations=2000) 

# 3a
# w = sgd(lr='b', save_fig=True, nbr_iterations=2000, batch_size=4000) 

print(classify(w))
