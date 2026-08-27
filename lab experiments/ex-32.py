import numpy as np
import matplotlib.pyplot as plt

# Generate two circular classes
np.random.seed(1)

n = 100

# Class 0 - inner circle
angle1 = np.random.uniform(0, 2 * np.pi, n)
radius1 = np.random.uniform(1, 2, n)

x1 = radius1 * np.cos(angle1)
y1 = radius1 * np.sin(angle1)

# Class 1 - outer circle
angle2 = np.random.uniform(0, 2 * np.pi, n)
radius2 = np.random.uniform(4, 5, n)

x2 = radius2 * np.cos(angle2)
y2 = radius2 * np.sin(angle2)

# Combine the data
X = np.vstack((
    np.column_stack((x1, y1)),
    np.column_stack((x2, y2))
))

Y = np.hstack((
    np.zeros(n),
    np.ones(n)
)).reshape(-1, 1)

# Normalize input
X = X / 5

# Parameters
learning_rate = 0.1
hidden_layers = 3
hidden_neurons = 3

# Tanh activation
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Initialize weights
np.random.seed(1)

W1 = np.random.randn(2, 3) * 0.5
W2 = np.random.randn(3, 3) * 0.5
W3 = np.random.randn(3, 3) * 0.5
W4 = np.random.randn(3, 1) * 0.5

b1 = np.zeros((1, 3))
b2 = np.zeros((1, 3))
b3 = np.zeros((1, 3))
b4 = np.zeros((1, 1))

# Training
epochs = 5000

for epoch in range(epochs):

    # Forward propagation
    Z1 = X @ W1 + b1
    A1 = tanh(Z1)

    Z2 = A1 @ W2 + b2
    A2 = tanh(Z2)

    Z3 = A2 @ W3 + b3
    A3 = tanh(Z3)

    Z4 = A3 @ W4 + b4
    A4 = tanh(Z4)

    # Error
    error = A4 - Y

    # Backpropagation
    dZ4 = error * tanh_derivative(Z4)

    dW4 = (A3.T @ dZ4) / len(X)
    db4 = np.sum(dZ4, axis=0, keepdims=True) / len(X)

    dA3 = dZ4 @ W4.T
    dZ3 = dA3 * tanh_derivative(Z3)

    dW3 = (A2.T @ dZ3) / len(X)
    db3 = np.sum(dZ3, axis=0, keepdims=True) / len(X)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * tanh_derivative(Z2)

    dW2 = (A1.T @ dZ2) / len(X)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / len(X)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * tanh_derivative(Z1)

    dW1 = (X.T @ dZ1) / len(X)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / len(X)

    # Update weights
    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# Prediction
Z1 = X @ W1 + b1
A1 = tanh(Z1)

Z2 = A1 @ W2 + b2
A2 = tanh(Z2)

Z3 = A2 @ W3 + b3
A3 = tanh(Z3)

Z4 = A3 @ W4 + b4
A4 = tanh(Z4)

# Convert output to class
predicted = (A4 >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(predicted == Y) * 100

print("Actual Classes:")
print(Y.flatten())

print("\nPredicted Classes:")
print(predicted.flatten())

print("\nAccuracy:", accuracy, "%")


# Plot the circular classes
plt.figure(figsize=(8, 6))

plt.scatter(
    X[Y.flatten() == 0, 0],
    X[Y.flatten() == 0, 1],
    label="Class 0"
)

plt.scatter(
    X[Y.flatten() == 1, 0],
    X[Y.flatten() == 1, 1],
    label="Class 1"
)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Neural Network Analysis - Two Circular Classes")
plt.legend()
plt.grid(True)

plt.show()
