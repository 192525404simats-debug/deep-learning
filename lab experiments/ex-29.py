import numpy as np
import matplotlib.pyplot as plt

# Multi-class dataset
X = np.array([
    [1, 1], [1, 1.5], [1.5, 1], [1.5, 1.5],
    [5, 1], [5, 1.5], [5.5, 1], [5.5, 1.5],
    [3, 5], [3, 5.5], [3.5, 5], [3.5, 5.5]
])

# Class labels: 0, 1, 2
Y = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2
])

# One-hot encoding
T = np.zeros((len(Y), 3))

for i in range(len(Y)):
    T[i, Y[i]] = 1

# Neural network parameters
learning_rate = 0.1
hidden_layers = 3
hidden_neurons = 3
classes = 3

# Tanh activation function
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Initialize weights
np.random.seed(1)

W1 = np.random.randn(2, 3) * 0.5
W2 = np.random.randn(3, 3) * 0.5
W3 = np.random.randn(3, 3) * 0.5
W4 = np.random.randn(3, 3) * 0.5

b1 = np.zeros((1, 3))
b2 = np.zeros((1, 3))
b3 = np.zeros((1, 3))
b4 = np.zeros((1, 3))

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

    # Softmax output
    exp_Z4 = np.exp(Z4 - np.max(Z4, axis=1, keepdims=True))
    A4 = exp_Z4 / np.sum(exp_Z4, axis=1, keepdims=True)

    # Error
    error = A4 - T

    # Backpropagation

    dW4 = (A3.T @ error) / len(X)
    db4 = np.sum(error, axis=0, keepdims=True) / len(X)

    dA3 = error @ W4.T
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

exp_Z4 = np.exp(Z4 - np.max(Z4, axis=1, keepdims=True))
A4 = exp_Z4 / np.sum(exp_Z4, axis=1, keepdims=True)

# Predicted classes
predicted = np.argmax(A4, axis=1)

# Accuracy
accuracy = np.mean(predicted == Y) * 100

print("Actual Classes:   ", Y)
print("Predicted Classes:", predicted)
print("Accuracy:", accuracy, "%")


# Plot
plt.figure(figsize=(8, 6))

plt.scatter(X[Y == 0, 0], X[Y == 0, 1], label="Class 0")
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], label="Class 1")
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], label="Class 2")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Multi-Class Neural Network Analysis - Tanh")
plt.legend()
plt.grid(True)

plt.show()
