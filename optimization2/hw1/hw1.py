import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the uniform distribution
a = 0  # lower bound
b = 1  # upper bound

# Define the cumulative distribution function (CDF)
def uniform_cdf(x, a, b):
    if x < a:
        return 0
    elif x > b:
        return 0
    else:
        return (x - a) / (b - a)

# Generate x values for plotting
x_values = np.linspace(a - 0.5, b + 0.5, 500)
y_values = [uniform_cdf(x, a, b) for x in x_values]

# Plot the CDF
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label=f"CDF of U({a}, {b})", color="blue")
plt.axvline(a, linestyle="--", color="gray", label="a")
plt.axvline(b, linestyle="--", color="gray", label="b")
plt.title("Cumulative Distribution Function of U(a, b)")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.grid(alpha=0.5)
plt.legend()

# Set specific x and y axis values
plt.xticks([a, b], ['a', 'b'])
plt.yticks([0, 1], ['0', '1'])
plt.savefig('cdf_plot.pdf')


plt.show()
