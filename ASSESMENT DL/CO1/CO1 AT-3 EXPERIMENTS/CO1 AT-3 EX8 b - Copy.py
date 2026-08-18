import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([1,2,3,4,5], dtype=float)
Y = np.array([2,4,6,8,10], dtype=float)

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
epochs = 100

cost_history = []

for epoch in range(epochs):

    for i in range(len(X)):

        x = X[i]
        y = Y[i]

        # Prediction
        y_pred = m*x + b

        # Error
        error = y - y_pred

        # Update weights immediately
        m = m + learning_rate * 2 * error * x
        b = b + learning_rate * 2 * error

    # Compute Cost after each epoch
    predictions = m*X + b
    cost = np.mean((Y - predictions)**2)
    cost_history.append(cost)

print("Final Slope (m):", m)
print("Final Intercept (b):", b)

# Plot Cost
plt.plot(cost_history)
plt.title("SGD Cost")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()
