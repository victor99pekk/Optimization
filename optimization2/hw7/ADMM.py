import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_loss(w, X, y):
    return np.mean(np.log(1 + np.exp(-y * (X @ w))))

def ADMM(X_train, y_train, lambda_, rho, max_iter=100, tol=1e-4):
    d = X_train.shape[1]
    w = np.zeros(d)
    z = np.zeros(d)
    y = np.zeros(d)
    
    obj_values = []
    w_z_diffs = []
    
    for _ in range(max_iter):
        for _ in range(40):  # find approximation of optimal w with GD
            gradient = -(X_train.T @ (y_train * sigmoid(-y_train * (X_train @ w)))) / len(y_train) + rho * (w - z + y / rho)
            w -= 0.1 * gradient       
        z = (rho * w + y) / (rho + lambda_)   
        y += rho * (w - z)
        
        obj_values.append(logistic_loss(w, X_train, y_train) + (lambda_/2) * np.linalg.norm(z)**2)
        w_z_diffs.append(np.linalg.norm(w - z))

        if np.linalg.norm(w - z) < tol: # we break if it is close enough
            break
    
    return w, obj_values, w_z_diffs

data = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = data.data, data.target.astype(int)
digit1, digit2 = 3, 8
mask = (y == digit1) | (y == digit2)
X, y = X[mask], y[mask]
y = np.where(y == digit1, -1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lambda_ = 0.1
rho = 1.0
w, obj_values, w_z_diffs = ADMM(X_train, y_train, lambda_, rho)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(obj_values)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('F(w, z)')
plt.title('Objective Function')

plt.subplot(1, 2, 2)
plt.plot(w_z_diffs)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('||w - z||')
plt.title('Convergence of w and z')
plt.show()

y_pred_train = np.sign(X_train @ w)
y_pred_test = np.sign(X_test @ w)
train_error = np.mean(y_pred_train != y_train)
test_error = np.mean(y_pred_test != y_test)

print(f"Training error: {train_error:.4f}")
print(f"Test error: {test_error:.4f}")
