import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# Multi-Class Data
# ---------------------------------------

X = np.array([
    [1, 1], [1.5, 1], [1, 1.5], [1.5, 1.5],
    [5, 1], [5.5, 1], [5, 1.5], [5.5, 1.5],
    [3, 5], [3.5, 5], [3, 5.5], [3.5, 5.5]
])

# Three classes
Y = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2
])

# ---------------------------------------
# One-Hot Encoding
# ---------------------------------------

T = np.zeros((len(Y), 3))

for i in range(len(Y)):
    T[i, Y[i]] = 1

# ---------------------------------------
# Parameters
# ---------------------------------------

learning_rate = 0.1
hidden_layers = 2
hidden_neurons = 2
classes = 3

# ---------------------------------------
# Tanh Activation Function
# ---------------------------------------

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# ---------------------------------------
# Initialize Weights
# ---------------------------------------

np.random.seed(1)

W1 = np.random.randn(2, 2) * 0.5
W2 = np.random.randn(2, 2) * 0.5
W3 = np.random.randn(2, 3) * 0.5

b1 = np.zeros((1, 2))
b2 = np.zeros((1, 2))
b3 = np.zeros((1, 3))

# ---------------------------------------
# Training
# ---------------------------------------

epochs = 5000

for epoch in range(epochs):

    # Forward propagation

    Z1 = np.dot(X, W1) + b1
    A1 = tanh(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = tanh(Z2)

    Z3 = np.dot(A2, W3) + b3

    # Softmax output
    exp_values = np.exp(Z3 - np.max(Z3, axis=1, keepdims=True))
    A3 = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # -----------------------------------
    # Cross Entropy Loss
    # -----------------------------------

    loss = -np.mean(np.sum(T * np.log(A3 + 1e-10), axis=1))

    # -----------------------------------
    # Backpropagation
    # -----------------------------------

    dZ3 = A3 - T
    dW3 = np.dot(A2.T, dZ3) / len(X)
    db3 = np.sum(dZ3, axis=0, keepdims=True) / len(X)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * tanh_derivative(Z2)

    dW2 = np.dot(A1.T, dZ2) / len(X)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / len(X)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * tanh_derivative(Z1)

    dW1 = np.dot(X.T, dZ1) / len(X)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / len(X)

    # -----------------------------------
    # Update Weights
    # -----------------------------------

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    # Display loss
    if epoch % 500 == 0:
        print("Epoch:", epoch, "Loss:", loss)

# ---------------------------------------
# Prediction
# ---------------------------------------

Z1 = np.dot(X, W1) + b1
A1 = tanh(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = tanh(Z2)

Z3 = np.dot(A2, W3) + b3

exp_values = np.exp(Z3 - np.max(Z3, axis=1, keepdims=True))
A3 = exp_values / np.sum(exp_values, axis=1, keepdims=True)

predictions = np.argmax(A3, axis=1)

# ---------------------------------------
# Accuracy
# ---------------------------------------

accuracy = np.mean(predictions == Y) * 100

print("\nActual Classes:   ", Y)
print("Predicted Classes:", predictions)
print("Accuracy: {:.2f}%".format(accuracy))

# ---------------------------------------
# Plot Multi-Class Data
# ---------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X[Y == 0, 0],
    X[Y == 0, 1],
    label="Class 0"
)

plt.scatter(
    X[Y == 1, 0],
    X[Y == 1, 1],
    label="Class 1"
)

plt.scatter(
    X[Y == 2, 0],
    X[Y == 2, 1],
    label="Class 2"
)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Multi-Class Neural Network Analysis - Tanh")
plt.legend()
plt.grid(True)

plt.show()
