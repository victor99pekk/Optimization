import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = 'HW2_ellipse.csv'

df = pd.read_csv(csv_file_path)
df.columns = ['x', 'y']
x = df['x']
y = df['y']

plt.scatter(x,y)
plt.xlabel('X-value')
plt.ylabel('Y-value')

plt.show()