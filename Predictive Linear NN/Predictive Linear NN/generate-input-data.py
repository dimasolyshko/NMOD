import numpy as np
import math
import matplotlib.pyplot as plt

A = 3
B = 6
D = 0.1


def calculate_period():
    return 2*np.pi/abs(B)

# Generate data points based on the function y = 3*sin(6x) + 0.1
def generate_data(num_points=45):
    X = np.linspace(0, calculate_period(), num=num_points)  # Random x values
    y = A * np.sin(B * X) + D  # y = 3*sin(6x) + 0.1

    with open('data.txt', 'w') as data:
        for i in range(num_points):
            data.write(f"{X[i]},{y[i]}\n")

    return X, y

# Generate data
X, y = generate_data()

# Plot the generated data
plt.scatter(X, y, label="Data")
plt.legend()
plt.show()
