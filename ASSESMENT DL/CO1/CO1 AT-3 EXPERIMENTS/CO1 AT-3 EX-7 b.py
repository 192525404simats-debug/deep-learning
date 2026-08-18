import numpy as np

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Input
X = np.array([[0.5, 0.8]])

# Target Output
Y = np.array([[1]])

# Initial Weights
W1 = np.array([[0.1, 0.4],
               [0.3, 0.2]])

W2 = np.array([[0.7],
               [0.5]])

# Biases
b1 = np.array([[0.2, 0.1]])
b2 = np.array([[0.3]])

learning_rate = 0.5

# ---------------- Forward Propagation ----------------

hidden_input = np.dot(X, W1) + b1
hidden_output = sigmoid(hidden_input)

output_input = np.dot(hidden_output, W2) + b2
predicted_output = sigmoid(output_input)

print("Predicted Output Before Training:")
print(predicted_output)

# ---------------- Backpropagation ----------------

# Output Error
error = Y - predicted_output

# Output Delta
d_output = error * sigmoid_derivative(predicted_output)

# Hidden Layer Error
hidden_error = d_output.dot(W2.T)

# Hidden Delta
d_hidden = hidden_error * sigmoid_derivative(hidden_output)

# ---------------- Weight Updates ----------------

W2 += hidden_output.T.dot(d_output) * learning_rate
b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

W1 += X.T.dot(d_hidden) * learning_rate
b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("\nUpdated Weights W2:")
print(W2)

print("\nUpdated Bias b2:")
print(b2)

print("\nUpdated Weights W1:")
print(W1)

print("\nUpdated Bias b1:")
print(b1)
