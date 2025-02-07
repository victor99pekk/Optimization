from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('HW7Q4.csv', header=None)
blue = df.iloc[:100, :2].to_numpy().astype(float)
red = df.iloc[100:, :2].to_numpy().astype(float)
all_points = df.iloc[:].to_numpy().astype(float)
start = np.array([-1,1],dtype=float)


plt.scatter(blue[:, 0], blue[:, 1], color='blue', label='Blue')
plt.scatter(red[:, 0], red[:, 1], color='red', label='Red')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of Blue and Red Points')
plt.legend()
plt.savefig('scatter_plt.png')
plt.show()

def indicator(weights:np.ndarray, index:int):
    if 1 - np.dot(all_points[index][:2], weights) * all_points[index][2] > 0:
          return 1
    return 0


def loss(w:np.ndarray) -> float:
    term = 0.5 * np.linalg.norm(w)**2
    for i in range(200):
        inner_product = np.dot(all_points[i][:2], w) * all_points[i][2]
        term += max(0, 1 - inner_product)
    return term

def loss_grad(weight:np.ndarray) -> float:
    term = weight.copy()
    for i in range(200):
        term -= all_points[i][2] * all_points[i][:2] * indicator(index=i, weights=weight)
    return term.astype(float)

def classification(w:np.ndarray) -> float:
    num = 0 
    tot = 0
    for index in range(200):
        num += 1 if np.dot(all_points[index][:2], w) * all_points[index][2] > 0 else 0
        tot += 1
    return num / tot
    

def gd(w:np.array= start, lr:float=1e-4,iterations:int=10**3):
    errorList = [classification(w)]
    margin = [2 / np.linalg.norm(w)]
    for i in range(iterations-1):
        w = w - lr * loss_grad(w)
        errorList.append(classification(w))
        margin.append(2 / np.linalg.norm(w))
        # if i % 100 == 0:
            # print(loss(w))
    return w, errorList, margin

def importance_function(w:np.ndarray) -> float:
    # print(1 - all_points[index][2] * np.dot(all_points[index][:2],w))
    list = []
    for index in range(200):
        list.append(np.abs(1 - all_points[index][2] * np.dot(all_points[index][:2],w)))
    maxval = max(list)
    for item in range(200):
        list[item] /= maxval
    return list

# w, errorList, margin = gd()
# iterationList = [i for i in range(10**3)]

# plt.plot(iterationList, errorList)
# plt.xlabel('Iteration')
# plt.ylabel('Classification Accuracy')
# plt.title('Classification Accuracy over Iterations')
# plt.legend()
# plt.savefig('accuracy.png')
# plt.show()

# plt.plot(iterationList, margin)
# plt.xlabel('Iteration')
# plt.ylabel('2 / || w ||')
# plt.title('margin as function of iterations')
# plt.legend()
# plt.savefig('margin.png')
# plt.show()


# #scatter plot
# colors = [(0.8, 0.8, 0.8), (1, 0, 0)]  # Light grey to red
# n_bins = 100  # Discretize the colormap into 100 bins
# cmap_name = 'custom_cmap'
# custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
# plt.scatter(blue[:, 0], blue[:, 1], color='blue')
# plt.scatter(red[:, 0], red[:, 1], color='red')

# colorList = importance_function(w=w)
# print(colorList)

# for i in range(len(all_points)):
#     plt.scatter(all_points[i][0], all_points[i][1], color=custom_cmap(np.abs(colorList[i])))


# # Add labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Color indicates a point\'s importance in SVM')
# plt.legend()
# plt.savefig('scatter_plt.png')
# plt.show()