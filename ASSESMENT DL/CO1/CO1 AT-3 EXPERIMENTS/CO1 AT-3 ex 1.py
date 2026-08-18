import numpy as np
import matplotlib.pyplot as plt

# Sample Dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 4, 6, 8, 10], dtype=float)

# Initialize parameters
m = 0       # Slope
b = 0       # Intercept

learning_rate = 0.01
iterations = 1000
n = len(X)

loss_history = []

# Gradient Descent
for i in range(iterations):
    # Predictions
    Y_pred = m * X + b

    # Calculate Loss (Mean Squared Error)
    loss = (1/n) * np.sum((Y - Y_pred) ** 2)
    loss_history.append(loss)

    # Compute Gradients
    dm = (-2/n) * np.sum(X * (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    # Update Parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

# Final Parameters
print("Final Slope (m):", round(m, 4))
print("Final Intercept (b):", round(b, 4))
print("Final Loss:", round(loss_history[-1], 6))

# Plot Regression Line
plt.figure(figsize=(6,4))
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, m*X + b, color='red', label='Regression Line')
plt.title("Linear Regression using Gradient Descent")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Plot Learning Curve
plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.title("Learning Curve (Loss vs Iterations)")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (Loss)")
plt.grid(True)
plt.show()
